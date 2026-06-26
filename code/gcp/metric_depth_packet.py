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
VARIANCE_VALIDATION_POLICY = "float_forward_error_bound_v1"
DEFAULT_VARIANCE_VALIDATION_ABS_FLOOR = 1e-5
DEFAULT_VARIANCE_VALIDATION_ULP_FACTOR = 8.0
DEFAULT_VARIANCE_VALIDATION_DTYPE = "float32"
DEFAULT_VARIANCE_VALIDATION_RTOL = 0.0


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


def recompute_and_compare_packet(
    packet: Dict[str, np.ndarray],
    atol: float = 1e-5,
    rtol: float = 1e-5,
    numerical_support_floor: float = DEFAULT_NUMERICAL_SUPPORT_FLOOR,
    variance_clamp_tolerance: float = DEFAULT_VARIANCE_CLAMP_TOLERANCE,
    variance_validation_policy: str = VARIANCE_VALIDATION_POLICY,
    variance_validation_abs_floor: float = DEFAULT_VARIANCE_VALIDATION_ABS_FLOOR,
    variance_validation_ulp_factor: float = DEFAULT_VARIANCE_VALIDATION_ULP_FACTOR,
    variance_validation_dtype: str = DEFAULT_VARIANCE_VALIDATION_DTYPE,
    variance_validation_rtol: float = DEFAULT_VARIANCE_VALIDATION_RTOL,
) -> Dict[str, Any]:
    policy = validate_variance_validation_policy(
        variance_validation_policy=variance_validation_policy,
        variance_validation_abs_floor=variance_validation_abs_floor,
        variance_validation_ulp_factor=variance_validation_ulp_factor,
        variance_validation_dtype=variance_validation_dtype,
        variance_validation_rtol=variance_validation_rtol,
    )
    derived = derive_metric_depth_packet(
        packet["accumulated_alpha"],
        packet["weighted_camera_z_sum"],
        packet["weighted_camera_z_second_moment"],
        packet["weighted_inverse_camera_z_sum"],
        numerical_support_floor=numerical_support_floor,
        variance_clamp_tolerance=variance_clamp_tolerance,
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
        elif name == "camera_z_variance":
            equal, metrics = compare_variance_forward_error_bound(
                packet=packet,
                expected_variance=expected,
                policy=policy,
            )
            abs_err = float(metrics["max_abs_error"])
            rel_err = float(metrics["max_error_to_bound_ratio"])
        else:
            diff = np.nan_to_num(actual - expected, nan=0.0)
            abs_err = float(np.max(np.abs(diff))) if diff.size else 0.0
            denom = np.maximum(np.abs(np.nan_to_num(expected, nan=0.0)), 1e-12)
            rel_err = float(np.max(np.abs(diff) / denom)) if diff.size else 0.0
            equal = bool(np.allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=True))
        row = {
            "tensor": name,
            "passed": equal,
            "max_abs_error": abs_err,
            "max_rel_error": rel_err,
            "atol": atol,
            "rtol": rtol,
        }
        if name == "camera_z_variance":
            row.update(metrics)
        rows.append(row)
        ok = ok and equal
    return {"passed": ok, "rows": rows}


def validate_variance_validation_policy(
    variance_validation_policy: str,
    variance_validation_abs_floor: float,
    variance_validation_ulp_factor: float,
    variance_validation_dtype: str,
    variance_validation_rtol: float,
) -> Dict[str, Any]:
    if variance_validation_policy != VARIANCE_VALIDATION_POLICY:
        raise ValueError(f"Unsupported variance validation policy: {variance_validation_policy!r}")
    try:
        abs_floor = float(variance_validation_abs_floor)
        ulp_factor = float(variance_validation_ulp_factor)
        rtol = float(variance_validation_rtol)
    except Exception as exc:
        raise ValueError("Variance validation fields must be numeric") from exc
    if not math.isfinite(abs_floor) or abs_floor < 0:
        raise ValueError(f"variance_validation_abs_floor must be finite and non-negative: {abs_floor}")
    if not math.isfinite(ulp_factor) or ulp_factor < 0:
        raise ValueError(f"variance_validation_ulp_factor must be finite and non-negative: {ulp_factor}")
    if not math.isfinite(rtol) or rtol != 0.0:
        raise ValueError(f"variance_validation_rtol is locked to 0 for {VARIANCE_VALIDATION_POLICY}: {rtol}")
    dtype = np.dtype(str(variance_validation_dtype))
    if not np.issubdtype(dtype, np.floating):
        raise ValueError(f"variance_validation_dtype must be a floating dtype: {variance_validation_dtype!r}")
    return {
        "variance_validation_policy": variance_validation_policy,
        "variance_validation_abs_floor": abs_floor,
        "variance_validation_ulp_factor": ulp_factor,
        "variance_validation_dtype": dtype.name,
        "variance_validation_rtol": rtol,
        "variance_validation_eps": float(np.finfo(dtype).eps),
    }


def variance_validation_manifest_fields() -> Dict[str, Any]:
    policy = validate_variance_validation_policy(
        variance_validation_policy=VARIANCE_VALIDATION_POLICY,
        variance_validation_abs_floor=DEFAULT_VARIANCE_VALIDATION_ABS_FLOOR,
        variance_validation_ulp_factor=DEFAULT_VARIANCE_VALIDATION_ULP_FACTOR,
        variance_validation_dtype=DEFAULT_VARIANCE_VALIDATION_DTYPE,
        variance_validation_rtol=DEFAULT_VARIANCE_VALIDATION_RTOL,
    )
    return {
        "variance_validation_policy": policy["variance_validation_policy"],
        "variance_validation_abs_floor": policy["variance_validation_abs_floor"],
        "variance_validation_ulp_factor": policy["variance_validation_ulp_factor"],
        "variance_validation_dtype": policy["variance_validation_dtype"],
        "variance_validation_rtol": policy["variance_validation_rtol"],
    }


def compare_variance_forward_error_bound(
    packet: Dict[str, np.ndarray],
    expected_variance: np.ndarray,
    policy: Dict[str, Any],
) -> tuple[bool, Dict[str, Any]]:
    actual = np.asarray(packet["camera_z_variance"])
    if str(actual.dtype) != policy["variance_validation_dtype"]:
        raise ValueError(
            "camera_z_variance dtype does not match manifest policy: "
            f"{actual.dtype} != {policy['variance_validation_dtype']}"
        )
    a = np.asarray(packet["accumulated_alpha"], dtype=np.float64)
    m1 = np.asarray(packet["weighted_camera_z_sum"], dtype=np.float64)
    m2 = np.asarray(packet["weighted_camera_z_second_moment"], dtype=np.float64)
    valid = np.asarray(packet["metric_depth_valid_mask"]).astype(bool)
    expected = np.asarray(expected_variance, dtype=np.float64)
    actual64 = actual.astype(np.float64)

    invalid = ~valid
    invalid_nan_ok = bool(np.all(np.isnan(actual64[invalid])) and np.all(np.isnan(expected[invalid])))
    valid_actual_finite = bool(np.all(np.isfinite(actual64[valid])))
    valid_expected_finite = bool(np.all(np.isfinite(expected[valid])))

    mu = np.full(a.shape, np.nan, dtype=np.float64)
    second = np.full(a.shape, np.nan, dtype=np.float64)
    variance_ref = np.full(a.shape, np.nan, dtype=np.float64)
    mu[valid] = m1[valid] / a[valid]
    second[valid] = m2[valid] / a[valid]
    variance_ref[valid] = second[valid] - mu[valid] * mu[valid]
    scale = np.maximum.reduce([np.abs(second), np.abs(mu * mu), np.ones_like(a, dtype=np.float64)])
    allowed = policy["variance_validation_abs_floor"] + policy["variance_validation_ulp_factor"] * policy["variance_validation_eps"] * scale
    diff = np.abs(actual64 - variance_ref)
    valid_diff = diff[valid]
    valid_allowed = allowed[valid]
    if valid_diff.size:
        ratios = valid_diff / valid_allowed
        worst_flat = int(np.nanargmax(ratios))
        valid_coords = np.argwhere(valid)
        worst_coord = valid_coords[worst_flat].tolist()
        max_abs_error = float(valid_diff[worst_flat])
        max_allowed_error = float(valid_allowed[worst_flat])
        max_ratio = float(ratios[worst_flat])
        failing = int(np.count_nonzero(valid_diff > valid_allowed))
        wy, wx = worst_coord[-2], worst_coord[-1]
        worst_payload = {
            "coordinate": worst_coord,
            "A": float(a[tuple(worst_coord)]),
            "M1": float(m1[tuple(worst_coord)]),
            "M2": float(m2[tuple(worst_coord)]),
            "mu": float(mu[tuple(worst_coord)]),
            "second": float(second[tuple(worst_coord)]),
            "variance_packet": float(actual64[tuple(worst_coord)]),
            "variance_ref": float(variance_ref[tuple(worst_coord)]),
            "y": int(wy),
            "x": int(wx),
        }
    else:
        max_abs_error = 0.0
        max_allowed_error = float(policy["variance_validation_abs_floor"])
        max_ratio = 0.0
        failing = 0
        worst_payload = {}
    equal = invalid_nan_ok and valid_actual_finite and valid_expected_finite and failing == 0 and max_ratio <= 1.0
    return equal, {
        "variance_validation_policy": policy["variance_validation_policy"],
        "variance_validation_abs_floor": policy["variance_validation_abs_floor"],
        "variance_validation_ulp_factor": policy["variance_validation_ulp_factor"],
        "variance_validation_dtype": policy["variance_validation_dtype"],
        "variance_validation_rtol": policy["variance_validation_rtol"],
        "max_abs_error": max_abs_error,
        "max_allowed_error": max_allowed_error,
        "max_error_to_bound_ratio": max_ratio,
        "failing_pixel_count": failing,
        "valid_pixel_count": int(np.count_nonzero(valid)),
        "invalid_nan_ok": invalid_nan_ok,
        "valid_actual_finite": valid_actual_finite,
        "valid_expected_finite": valid_expected_finite,
        "worst_pixel": worst_payload,
    }


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
