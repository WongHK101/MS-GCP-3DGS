from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np


METRIC_PACKET_SCHEMA = "ms_gcp_metric_depth_packet_v2"
METRIC_PACKET_MANIFEST_SCHEMA = "ms_gcp_metric_depth_packet_manifest_v2"
METRIC_PACKET_TENSOR_NAMES = [
    "accumulated_alpha",
    "weighted_camera_z_sum",
    "weighted_camera_z_second_moment",
    "weighted_inverse_camera_z_sum",
    "alpha_normalized_expected_camera_z",
    "alpha_normalized_expected_inverse_camera_z",
    "harmonic_camera_z",
    "camera_z_variance",
    "metric_depth_valid_mask",
]
RAW_ACCUMULATOR_TENSORS = [
    "accumulated_alpha",
    "weighted_camera_z_sum",
    "weighted_camera_z_second_moment",
    "weighted_inverse_camera_z_sum",
]
PRIMARY_DEPTH_TENSOR = "alpha_normalized_expected_camera_z"
PRIMARY_DEPTH_SEMANTICS = "camera_z"
HISTORICAL_INVALID_TENSOR = "historical_invalid_unnormalized_inverse_depth"
DEFAULT_NUMERICAL_SUPPORT_FLOOR = 1e-6
DEFAULT_NORMALIZATION_EPSILON = 1e-12
DEFAULT_VARIANCE_CLAMP_TOLERANCE = 1e-6
DEFAULT_ALPHA_CUTOFF = 1.0 / 255.0
DEFAULT_EARLY_TERMINATION_THRESHOLD = 1e-4


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def directory_tree_hash(path: Path, ignore_dirs: Iterable[str] = ("outputs", "runs", "logs", "__pycache__")) -> Dict[str, Any]:
    path = path.resolve()
    if path.is_file():
        return {
            "path": str(path),
            "kind": "file",
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
    files = []
    digest = hashlib.sha256()
    if path.exists():
        ignored = set(ignore_dirs)
        for root, dirs, names in os.walk(path):
            dirs[:] = [d for d in dirs if d not in ignored]
            for name in sorted(names):
                file_path = Path(root) / name
                rel = file_path.relative_to(path).as_posix()
                sha = file_sha256(file_path)
                size = file_path.stat().st_size
                digest.update(rel.encode("utf-8"))
                digest.update(sha.encode("ascii"))
                files.append({"path": rel, "sha256": sha, "bytes": size})
    return {
        "path": str(path),
        "kind": "directory",
        "sha256": digest.hexdigest(),
        "file_count": len(files),
        "files": files,
    }


def derive_metric_depth_packet(
    accumulated_alpha: np.ndarray,
    weighted_camera_z_sum: np.ndarray,
    weighted_camera_z_second_moment: np.ndarray,
    weighted_inverse_camera_z_sum: np.ndarray,
    numerical_support_floor: float = DEFAULT_NUMERICAL_SUPPORT_FLOOR,
    variance_clamp_tolerance: float = DEFAULT_VARIANCE_CLAMP_TOLERANCE,
) -> Dict[str, np.ndarray]:
    a = np.asarray(accumulated_alpha, dtype=np.float64)
    m1 = np.asarray(weighted_camera_z_sum, dtype=np.float64)
    m2 = np.asarray(weighted_camera_z_second_moment, dtype=np.float64)
    h = np.asarray(weighted_inverse_camera_z_sum, dtype=np.float64)
    if not (a.shape == m1.shape == m2.shape == h.shape):
        raise ValueError("All metric depth accumulators must have the same shape")
    valid = a > float(numerical_support_floor)
    expected_z = np.full(a.shape, np.nan, dtype=np.float64)
    expected_inv_z = np.full(a.shape, np.nan, dtype=np.float64)
    harmonic_z = np.full(a.shape, np.nan, dtype=np.float64)
    variance = np.full(a.shape, np.nan, dtype=np.float64)
    expected_z[valid] = m1[valid] / a[valid]
    expected_inv_z[valid] = h[valid] / a[valid]
    harmonic_valid = valid & (h > 0)
    harmonic_z[harmonic_valid] = a[harmonic_valid] / h[harmonic_valid]
    variance[valid] = m2[valid] / a[valid] - expected_z[valid] * expected_z[valid]
    tiny_negative = valid & (variance < 0) & (variance >= -float(variance_clamp_tolerance))
    variance[tiny_negative] = 0.0
    clearly_negative = valid & (variance < -float(variance_clamp_tolerance))
    if np.any(clearly_negative):
        worst = float(np.nanmin(variance[clearly_negative]))
        raise ValueError(f"Metric depth variance has clearly negative values; worst={worst}")
    return {
        "accumulated_alpha": a.astype(np.float32),
        "weighted_camera_z_sum": m1.astype(np.float32),
        "weighted_camera_z_second_moment": m2.astype(np.float32),
        "weighted_inverse_camera_z_sum": h.astype(np.float32),
        "alpha_normalized_expected_camera_z": expected_z.astype(np.float32),
        "alpha_normalized_expected_inverse_camera_z": expected_inv_z.astype(np.float32),
        "harmonic_camera_z": harmonic_z.astype(np.float32),
        "camera_z_variance": variance.astype(np.float32),
        "metric_depth_valid_mask": valid.astype(np.bool_),
    }


def cpu_reference_from_layers(
    camera_z: np.ndarray,
    alpha: np.ndarray,
    numerical_support_floor: float = DEFAULT_NUMERICAL_SUPPORT_FLOOR,
    early_termination_threshold: float = DEFAULT_EARLY_TERMINATION_THRESHOLD,
    alpha_cutoff: float = DEFAULT_ALPHA_CUTOFF,
    variance_clamp_tolerance: float = DEFAULT_VARIANCE_CLAMP_TOLERANCE,
) -> Dict[str, np.ndarray]:
    z = np.asarray(camera_z, dtype=np.float64)
    a = np.asarray(alpha, dtype=np.float64)
    if z.shape != a.shape or z.ndim < 1:
        raise ValueError("camera_z and alpha must have identical layer-first shapes")
    spatial_shape = z.shape[1:]
    accum = np.zeros(spatial_shape, dtype=np.float64)
    m1 = np.zeros(spatial_shape, dtype=np.float64)
    m2 = np.zeros(spatial_shape, dtype=np.float64)
    h = np.zeros(spatial_shape, dtype=np.float64)
    t = np.ones(spatial_shape, dtype=np.float64)
    done = np.zeros(spatial_shape, dtype=bool)
    for layer in range(z.shape[0]):
        alpha_layer = np.minimum(0.99, a[layer])
        active = (~done) & (alpha_layer >= alpha_cutoff)
        weight = np.where(active, alpha_layer * t, 0.0)
        accum += weight
        m1 += weight * z[layer]
        m2 += weight * z[layer] * z[layer]
        h += weight / z[layer]
        test_t = t * (1.0 - alpha_layer)
        done |= active & (test_t < early_termination_threshold)
        t = np.where(active, test_t, t)
    return derive_metric_depth_packet(
        accum,
        m1,
        m2,
        h,
        numerical_support_floor=numerical_support_floor,
        variance_clamp_tolerance=variance_clamp_tolerance,
    )


def tensor_stats(array: np.ndarray) -> Dict[str, Any]:
    arr = np.asarray(array)
    finite = np.isfinite(arr)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "finite_count": int(np.count_nonzero(finite)),
        "nan_count": int(np.count_nonzero(np.isnan(arr))) if np.issubdtype(arr.dtype, np.floating) else 0,
        "min": float(np.nanmin(arr)) if finite.any() else None,
        "max": float(np.nanmax(arr)) if finite.any() else None,
        "mean": float(np.nanmean(arr)) if finite.any() else None,
    }


def recompute_and_compare_packet(packet: Dict[str, np.ndarray], atol: float = 1e-5, rtol: float = 1e-5) -> Dict[str, Any]:
    derived = derive_metric_depth_packet(
        packet["accumulated_alpha"],
        packet["weighted_camera_z_sum"],
        packet["weighted_camera_z_second_moment"],
        packet["weighted_inverse_camera_z_sum"],
    )
    rows = []
    ok = True
    for name in METRIC_PACKET_TENSOR_NAMES:
        actual = np.asarray(packet[name])
        expected = np.asarray(derived[name])
        if actual.dtype == np.bool_ or expected.dtype == np.bool_:
            equal = np.array_equal(actual.astype(bool), expected.astype(bool))
            abs_err = 0.0 if equal else 1.0
            rel_err = abs_err
        else:
            diff = np.nan_to_num(actual - expected, nan=0.0)
            abs_err = float(np.max(np.abs(diff))) if diff.size else 0.0
            denom = np.maximum(np.abs(np.nan_to_num(expected, nan=0.0)), 1e-12)
            rel_err = float(np.max(np.abs(diff) / denom)) if diff.size else 0.0
            equal = bool(np.allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=True))
        rows.append(
            {
                "tensor": name,
                "passed": equal,
                "max_abs_error": abs_err,
                "max_rel_error": rel_err,
                "atol": atol,
                "rtol": rtol,
            }
        )
        ok = ok and equal
    return {"passed": ok, "rows": rows}


def packet_manifest_tensor_formulas() -> Dict[str, str]:
    return {
        "accumulated_alpha": "A=sum_i alpha_i*T_i",
        "weighted_camera_z_sum": "M1=sum_i alpha_i*T_i*z_i",
        "weighted_camera_z_second_moment": "M2=sum_i alpha_i*T_i*z_i^2",
        "weighted_inverse_camera_z_sum": "H=sum_i alpha_i*T_i/z_i",
        "alpha_normalized_expected_camera_z": "M1/A for A>floor else NaN",
        "alpha_normalized_expected_inverse_camera_z": "H/A for A>floor else NaN",
        "harmonic_camera_z": "A/H for A>floor and H>0 else NaN",
        "camera_z_variance": "M2/A-(M1/A)^2 for A>floor else NaN",
        "metric_depth_valid_mask": "A>numerical_support_floor",
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
