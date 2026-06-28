from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
COLMAP_UTILS = REPO_ROOT / "code" / "colmap" / "utils"
sys.path.insert(0, str(COLMAP_UTILS))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from read_write_model import qvec2rotmat, read_model  # noqa: E402
from triangulate_gcp_points import pixel_to_normalized  # noqa: E402
from metric_depth_packet import (  # noqa: E402
    DEFAULT_VARIANCE_VALIDATION_DTYPE,
    DIAGNOSTIC_VARIANCE_TENSOR,
    DIAGNOSTIC_VARIANCE_VALID_MASK_TENSOR,
    METRIC_PACKET_MANIFEST_SCHEMA,
    METRIC_PACKET_TENSOR_NAMES,
    PRIMARY_DEPTH_SEMANTICS,
    PRIMARY_DEPTH_TENSOR,
    VARIANCE_NEGATIVE_HANDLING,
    VARIANCE_NONNEGATIVITY_POLICY,
    VARIANCE_RAW_PACKET_MODIFIED,
    VARIANCE_VALIDATION_POLICY,
    recompute_and_compare_packet,
    validate_variance_validation_policy,
)
from fit_gcp_sim3 import (  # noqa: E402
    DEFAULT_TARGET_FIELDS,
    apply_similarity,
    fit_similarity_umeyama,
    parse_name_set,
    residual_stats,
)
from gcp_pixel_domain_v1_2 import (  # noqa: E402
    RELEASE_V12_SCHEMA,
    validate_release_v12_rows_for_evaluator,
    verify_payload_integrity,
)


DEPTH_SUFFIXES = (".npy", ".npz", ".tif", ".tiff", ".png")
SUPPORTED_EVALUATOR_DEPTH_SEMANTICS = {
    "camera_z",
    "ray_distance",
    "inverse_camera_z",
    "inverse_ray_distance",
}
UNSUPPORTED_FORMAL_DEPTH_SEMANTICS = {
    "alpha_weighted_unnormalized_inverse_camera_z",
    "unnormalized_inverse_camera_z",
    "sum_alpha_transmittance_inverse_camera_z",
}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_release_config(path: Path) -> Dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    supported = {
        "ms_gcp_3dgs_benchmark_release_config_v1_1",
        RELEASE_V12_SCHEMA,
    }
    if config.get("schema") not in supported:
        raise ValueError(f"Unsupported release config schema: {config.get('schema')}")
    return config


def verify_release_files(config_path: Path, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    base = config_path.parent
    if config.get("schema") == RELEASE_V12_SCHEMA:
        manifest_path = base / "v1_2_release_file_manifest.json"
        root_record_path = base / "v1_2_release_root_digest.json"
        integrity = verify_payload_integrity(base, manifest_path, root_record_path)
        if not integrity["passed"]:
            raise ValueError(f"v1.2 release integrity failed: {integrity}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        verified = []
        for item in manifest.get("files", []):
            rel = item.get("path", "")
            path = base / rel
            verified.append(
                {
                    "path": str(path),
                    "release_relative_path": rel,
                    "sha256": item.get("sha256", ""),
                    "bytes": int(item.get("bytes", 0)),
                }
            )
        verified.extend(
            [
                {
                    "path": str(manifest_path),
                    "release_relative_path": "v1_2_release_file_manifest.json",
                    "sha256": file_sha256(manifest_path),
                    "bytes": manifest_path.stat().st_size,
                },
                {
                    "path": str(root_record_path),
                    "release_relative_path": "v1_2_release_root_digest.json",
                    "sha256": file_sha256(root_record_path),
                    "bytes": root_record_path.stat().st_size,
                },
            ]
        )
        return verified
    verified = []
    for item in config.get("files", []):
        rel = item.get("path", "")
        expected = item.get("sha256", "")
        if not rel or not expected:
            raise ValueError(f"Malformed release file entry: {item}")
        path = base / rel
        if not path.exists():
            raise FileNotFoundError(f"Release file missing: {path}")
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(
                f"Release hash mismatch for {rel}: expected {expected}, got {actual}"
            )
        verified.append(
            {
                "path": str(path),
                "release_relative_path": rel,
                "sha256": actual,
                "bytes": path.stat().st_size,
            }
        )
    return verified


def release_file_registry(verified_files: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    registry: Dict[str, Dict[str, Any]] = {}
    for item in verified_files:
        path = Path(str(item["path"])).resolve()
        registry[str(path)] = dict(item)
        registry[path.name] = dict(item)
        registry[str(item.get("release_relative_path", ""))] = dict(item)
    return registry


def load_scene_metadata(path: Path) -> Dict[str, Dict[str, str]]:
    scenes: Dict[str, Dict[str, str]] = {}
    for row in read_csv(path):
        scene = row.get("scene", "").strip()
        if scene:
            scenes[scene] = row
    if not scenes:
        raise ValueError(f"No scene metadata rows found: {path}")
    return scenes


def validate_annotation_rows_scene(rows: Sequence[Dict[str, str]], scene: str) -> None:
    mismatched_scenes = sorted(
        {
            row.get("scene", "").strip()
            for row in rows
            if row.get("scene", "").strip() and row.get("scene", "").strip() != scene
        }
    )
    if mismatched_scenes:
        raise ValueError(f"Annotation rows do not match requested scene {scene}: {mismatched_scenes}")


def git_commit(path: Path = REPO_ROOT) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def load_depth_manifest(path: Path) -> Dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    schema = manifest.get("schema", "")
    if schema == METRIC_PACKET_MANIFEST_SCHEMA:
        required = [
            "primary_depth_tensor",
            "primary_depth_semantics",
            "tensor_names",
            "depth_index",
            "model_content_hash",
            "renderer_commit",
            "rasterizer_commit",
            "rasterizer_tree_hash",
            "exporter_commit",
            "image_domain",
            "pixel_coordinate_convention",
            "numerical_support_floor",
            "variance_clamp_tolerance",
            "variance_validation_policy",
            "variance_validation_abs_floor",
            "variance_validation_ulp_factor",
            "variance_validation_dtype",
            "variance_validation_rtol",
            "variance_nonnegativity_policy",
            "variance_negative_handling",
            "variance_raw_packet_modified",
        ]
        missing = []
        for name in required:
            if name not in manifest:
                missing.append(name)
                continue
            value = manifest.get(name)
            if value is None or value == "" or value == []:
                missing.append(name)
        if missing:
            raise ValueError(f"Metric depth packet manifest missing required fields {missing}: {path}")
        if manifest["primary_depth_tensor"] != PRIMARY_DEPTH_TENSOR:
            raise ValueError(f"Unsupported primary depth tensor: {manifest['primary_depth_tensor']}")
        if manifest["primary_depth_semantics"] != PRIMARY_DEPTH_SEMANTICS:
            raise ValueError(f"Unsupported primary depth semantics: {manifest['primary_depth_semantics']}")
        for name in ["numerical_support_floor", "variance_clamp_tolerance"]:
            try:
                value = float(manifest[name])
            except Exception as exc:
                raise ValueError(f"Metric depth packet manifest field {name} must be numeric: {manifest.get(name)!r}") from exc
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Metric depth packet manifest field {name} must be finite and non-negative: {value}")
        validate_variance_validation_policy(
            variance_validation_policy=str(manifest["variance_validation_policy"]),
            variance_validation_abs_floor=float(manifest["variance_validation_abs_floor"]),
            variance_validation_ulp_factor=float(manifest["variance_validation_ulp_factor"]),
            variance_validation_dtype=str(manifest["variance_validation_dtype"]),
            variance_validation_rtol=float(manifest["variance_validation_rtol"]),
        )
        if manifest["variance_nonnegativity_policy"] != VARIANCE_NONNEGATIVITY_POLICY:
            raise ValueError(f"Unsupported variance nonnegativity policy: {manifest['variance_nonnegativity_policy']}")
        if manifest["variance_negative_handling"] != VARIANCE_NEGATIVE_HANDLING:
            raise ValueError(f"Unsupported variance negative handling: {manifest['variance_negative_handling']}")
        if bool(manifest["variance_raw_packet_modified"]) != VARIANCE_RAW_PACKET_MODIFIED:
            raise ValueError(f"Unsupported variance_raw_packet_modified flag: {manifest['variance_raw_packet_modified']}")
        tensors = set(manifest.get("tensor_names", []))
        missing_tensors = [name for name in METRIC_PACKET_TENSOR_NAMES if name not in tensors]
        if missing_tensors:
            raise ValueError(f"Metric depth packet manifest missing tensors: {missing_tensors}")
        return manifest
    if "depth_semantics" not in manifest:
        raise ValueError(f"Depth manifest missing depth_semantics: {path}")
    if "mapping_csv" not in manifest and "depth_index" not in manifest:
        raise ValueError(f"Depth manifest missing mapping_csv/depth_index: {path}")
    return manifest


def load_depth_index(manifest_path: Path, manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    base = manifest_path.parent
    rows: List[Dict[str, str]] = []
    if manifest.get("depth_index"):
        for row in manifest["depth_index"]:
            rows.append({k: str(v) for k, v in dict(row).items()})
    else:
        mapping_path = Path(str(manifest["mapping_csv"]))
        if not mapping_path.is_absolute():
            mapping_path = base / mapping_path
        rows = read_csv(mapping_path)
    index: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        image_name = row.get("image_name", "").strip()
        if not image_name:
            continue
        path_text = row.get("packet_path", row.get("depth_path", "")).strip()
        if not path_text:
            raise ValueError(f"Depth index row has no packet_path/depth_path for {image_name}")
        depth_path = Path(path_text)
        if not depth_path.is_absolute():
            depth_path = base / depth_path
        height = int(float(row.get("height", "0") or 0))
        width = int(float(row.get("width", "0") or 0))
        if height <= 0 or width <= 0:
            raise ValueError(f"Depth index row has invalid shape for {image_name}: {width}x{height}")
        payload = dict(row)
        payload["depth_path"] = str(depth_path)
        payload["packet_path"] = str(depth_path)
        payload["height"] = height
        payload["width"] = width
        index[image_name] = payload
        index[Path(image_name).name] = payload
    if not index:
        raise ValueError(f"Depth manifest index is empty: {manifest_path}")
    return index


def reject_unsupported_depth_semantics(semantics: str) -> None:
    normalized = semantics.strip().lower()
    if normalized in UNSUPPORTED_FORMAL_DEPTH_SEMANTICS:
        raise ValueError(
            f"Depth semantics '{semantics}' is not a metric camera-z depth input. "
            "It must be normalized with accumulated alpha/weight or re-exported as "
            "camera_z/ray_distance before formal GCP evaluation."
        )
    if normalized not in SUPPORTED_EVALUATOR_DEPTH_SEMANTICS:
        raise ValueError(f"Unsupported depth semantics: {semantics}")


def cli_flag_supplied(argv: Sequence[str], flag_name: str) -> bool:
    flag = "--" + flag_name
    return any(arg == flag or arg.startswith(flag + "=") for arg in argv)


def load_split_roles(path: Path, scene: str) -> tuple[set[str], set[str]]:
    controls: set[str] = set()
    checkpoints: set[str] = set()
    for row in read_csv(path):
        if row.get("scene", "").strip() != scene:
            continue
        point = row.get("point_name", "").strip()
        role = row.get("role", "").strip().lower()
        if not point:
            continue
        if role == "control":
            controls.add(point)
        elif role == "checkpoint":
            checkpoints.add(point)
    overlap = controls & checkpoints
    if overlap:
        raise ValueError(f"Control/checkpoint overlap in {path}: {sorted(overlap)}")
    if not controls or not checkpoints:
        raise ValueError(f"No complete control/checkpoint split for scene {scene} in {path}")
    return controls, checkpoints


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_target_points(path: Path, target_fields: Sequence[str]) -> Dict[str, np.ndarray]:
    points: Dict[str, np.ndarray] = {}
    for row in read_csv(path):
        name = row.get("point_name", "").strip()
        if not name:
            continue
        missing = [field for field in target_fields if row.get(field, "") == ""]
        if missing:
            continue
        points[name] = np.asarray([float(row[field]) for field in target_fields], dtype=np.float64)
    return points


def observation_is_usable(row: Dict[str, str], min_confidence: float) -> tuple[bool, str]:
    u = row.get("u_px", row.get("manual_x", ""))
    v = row.get("v_px", row.get("manual_y", ""))
    if u == "" or v == "":
        return False, "missing_pixel"
    visible = str(row.get("visible", "1")).strip()
    if visible not in {"", "1", "true", "True", "yes", "Y"}:
        return False, "annotation_not_visible"
    quality = str(row.get("quality", "")).strip()
    if quality in {"not_visible", "reject", "rejected"}:
        return False, "annotation_rejected"
    if quality and quality != "good":
        return False, "annotation_not_good"
    confidence_text = str(row.get("confidence", "")).strip()
    if confidence_text:
        try:
            if float(confidence_text) < min_confidence:
                return False, "low_annotation_confidence"
        except ValueError:
            pass
    return True, "ok"


def depth_candidates(depth_dir: Path, image_name: str) -> List[Path]:
    image = Path(image_name)
    names: List[str] = []
    for stem in (image.name, image.stem):
        for suffix in DEPTH_SUFFIXES:
            names.append(stem + suffix)
    return [depth_dir / name for name in names]


def find_depth_path(depth_dir: Path, image_name: str) -> Path | None:
    for candidate in depth_candidates(depth_dir, image_name):
        if candidate.exists():
            return candidate
    return None


def load_depth_map(path: Path, scale: float = 1.0, offset: float = 0.0, npz_key: str = "depth") -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
    elif suffix == ".npz":
        payload = np.load(path)
        if npz_key:
            if npz_key not in payload:
                raise KeyError(f"NPZ key {npz_key!r} missing from {path}; available={sorted(payload.files)}")
            key = npz_key
        else:
            key = sorted(payload.files)[0]
        arr = payload[key]
    else:
        arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return np.asarray(arr, dtype=np.float64) * float(scale) + float(offset)


def validate_metric_packet_npz(
    path: Path,
    entry: Dict[str, Any],
    numerical_support_floor: float,
    variance_clamp_tolerance: float,
    variance_validation_policy: str,
    variance_validation_abs_floor: float,
    variance_validation_ulp_factor: float,
    variance_validation_dtype: str,
    variance_validation_rtol: float,
    variance_nonnegativity_policy: str = VARIANCE_NONNEGATIVITY_POLICY,
    variance_negative_handling: str = VARIANCE_NEGATIVE_HANDLING,
    variance_raw_packet_modified: bool = VARIANCE_RAW_PACKET_MODIFIED,
) -> Dict[str, np.ndarray]:
    if variance_nonnegativity_policy != VARIANCE_NONNEGATIVITY_POLICY:
        raise ValueError(f"Unsupported variance nonnegativity policy: {variance_nonnegativity_policy}")
    if variance_negative_handling != VARIANCE_NEGATIVE_HANDLING:
        raise ValueError(f"Unsupported variance negative handling: {variance_negative_handling}")
    if bool(variance_raw_packet_modified) != VARIANCE_RAW_PACKET_MODIFIED:
        raise ValueError(f"Unsupported variance_raw_packet_modified flag: {variance_raw_packet_modified}")
    with np.load(path) as payload:
        available = set(payload.files)
        missing = [name for name in METRIC_PACKET_TENSOR_NAMES if name not in available]
        if missing:
            raise ValueError(f"Metric packet {path} missing required tensors: {missing}")
        packet = {name: np.asarray(payload[name]) for name in METRIC_PACKET_TENSOR_NAMES}
    shapes = {name: packet[name].shape for name in METRIC_PACKET_TENSOR_NAMES}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"Metric packet tensors have inconsistent shapes: {shapes}")
    expected_shape = (int(entry["height"]), int(entry["width"]))
    if tuple(next(iter(shapes.values()))) != expected_shape:
        raise ValueError(f"Metric packet shape mismatch for {path}: expected {expected_shape}, got {shapes}")
    dtypes = {name: str(packet[name].dtype) for name in METRIC_PACKET_TENSOR_NAMES}
    for name, dtype in dtypes.items():
        if name == "metric_depth_valid_mask":
            if dtype not in {"bool", "bool_"}:
                raise ValueError(f"Metric packet valid mask must be bool, got {dtype}")
        elif dtype != "float32":
            raise ValueError(f"Metric packet tensor {name} must be float32, got {dtype}")
    recompute = recompute_and_compare_packet(
        packet,
        numerical_support_floor=float(numerical_support_floor),
        variance_clamp_tolerance=float(variance_clamp_tolerance),
        variance_validation_policy=variance_validation_policy,
        variance_validation_abs_floor=float(variance_validation_abs_floor),
        variance_validation_ulp_factor=float(variance_validation_ulp_factor),
        variance_validation_dtype=variance_validation_dtype,
        variance_validation_rtol=float(variance_validation_rtol),
    )
    if not recompute["passed"]:
        raise ValueError(f"Metric packet derived tensor recomputation failed for {path}: {recompute}")
    diagnostic = recompute.get("diagnostic_tensors", {})
    for name, value in diagnostic.items():
        packet[name] = np.asarray(value)
    return packet


def camera_z_from_depth_value(depth_value: float, x_norm: float, y_norm: float, semantics: str) -> float:
    semantics = semantics.strip().lower()
    if semantics in UNSUPPORTED_FORMAL_DEPTH_SEMANTICS:
        raise ValueError(
            f"Depth semantics '{semantics}' is an unnormalized alpha/transmittance weighted "
            "quantity and cannot be converted with camera_z=1/depth."
        )
    if semantics == "camera_z":
        return float(depth_value)
    if semantics == "ray_distance":
        return float(depth_value) / math.sqrt(x_norm * x_norm + y_norm * y_norm + 1.0)
    if semantics == "inverse_camera_z":
        if depth_value == 0:
            return math.nan
        return 1.0 / float(depth_value)
    if semantics == "inverse_ray_distance":
        if depth_value == 0:
            return math.nan
        ray_distance = 1.0 / float(depth_value)
        return ray_distance / math.sqrt(x_norm * x_norm + y_norm * y_norm + 1.0)
    raise ValueError(f"Unsupported depth semantics: {semantics}")


def backproject_world(camera: Any, image: Any, u: float, v: float, camera_z: float) -> np.ndarray:
    x_norm, y_norm = pixel_to_normalized(camera, u, v)
    xyz_cam = np.asarray([x_norm * camera_z, y_norm * camera_z, camera_z], dtype=np.float64)
    rotation = qvec2rotmat(image.qvec)
    return rotation.T @ (xyz_cam - image.tvec)


def robust_depth_patch(
    depth: np.ndarray,
    camera: Any,
    u: float,
    v: float,
    depth_u: float,
    depth_v: float,
    depth_pixel_scale_x: float,
    depth_pixel_scale_y: float,
    patch_size: int,
    min_valid_ratio: float,
    min_depth: float,
    depth_semantics: str,
) -> tuple[bool, Dict[str, Any]]:
    if patch_size % 2 != 1 or patch_size < 1:
        raise ValueError("patch_size must be a positive odd integer")
    height, width = depth.shape[:2]
    if depth_u < 0 or depth_v < 0 or depth_u >= width or depth_v >= height:
        return False, {"failure_reason": "pixel_out_of_bounds", "image_width": width, "image_height": height}
    cx = int(round(float(depth_u)))
    cy = int(round(float(depth_v)))
    half = patch_size // 2
    x0 = max(0, cx - half)
    x1 = min(width, cx + half + 1)
    y0 = max(0, cy - half)
    y1 = min(height, cy + half + 1)
    patch = np.asarray(depth[y0:y1, x0:x1], dtype=np.float64)
    finite = np.isfinite(patch)
    if depth_semantics.startswith("inverse"):
        finite &= patch > 0
    else:
        finite &= patch > float(min_depth)
    valid_count = int(np.count_nonzero(finite))
    total_count = int(patch.size)
    valid_ratio = valid_count / max(1, total_count)
    if valid_ratio < float(min_valid_ratio):
        return False, {
            "failure_reason": "insufficient_finite_depth",
            "patch_valid_pixels": valid_count,
            "patch_total_pixels": total_count,
            "patch_valid_ratio": valid_ratio,
        }
    values = patch[finite]
    median_raw = float(np.median(values))
    mad_raw = float(np.median(np.abs(values - median_raw)))
    p10 = float(np.percentile(values, 10))
    p90 = float(np.percentile(values, 90))
    scale_x = float(depth_pixel_scale_x)
    scale_y = float(depth_pixel_scale_y)
    if scale_x <= 0 or scale_y <= 0:
        raise ValueError(f"Invalid depth pixel scale: {scale_x}, {scale_y}")
    camera_z_values: List[float] = []
    valid_ys, valid_xs = np.nonzero(finite)
    for yy, xx, raw_value in zip(valid_ys, valid_xs, values):
        depth_x = x0 + int(xx)
        depth_y = y0 + int(yy)
        geom_u = depth_x / scale_x
        geom_v = depth_y / scale_y
        x_norm, y_norm = pixel_to_normalized(camera, float(geom_u), float(geom_v))
        z_value = camera_z_from_depth_value(float(raw_value), x_norm, y_norm, depth_semantics)
        if np.isfinite(z_value):
            camera_z_values.append(float(z_value))
    if not camera_z_values:
        return False, {
            "failure_reason": "nonpositive_depth",
            "patch_valid_pixels": valid_count,
            "patch_total_pixels": total_count,
            "patch_valid_ratio": valid_ratio,
            "depth_raw_median": median_raw,
            "camera_z": math.nan,
        }
    camera_z = float(np.median(np.asarray(camera_z_values, dtype=np.float64)))
    camera_z_mad = float(np.median(np.abs(np.asarray(camera_z_values) - camera_z)))
    if not np.isfinite(camera_z) or camera_z <= float(min_depth):
        return False, {
            "failure_reason": "nonpositive_depth",
            "patch_valid_pixels": valid_count,
            "patch_total_pixels": total_count,
            "patch_valid_ratio": valid_ratio,
            "depth_raw_median": median_raw,
            "camera_z": camera_z,
        }
    return True, {
        "failure_reason": "",
        "patch_x0": x0,
        "patch_x1": x1,
        "patch_y0": y0,
        "patch_y1": y1,
        "patch_valid_pixels": valid_count,
        "patch_total_pixels": total_count,
        "patch_valid_ratio": valid_ratio,
        "depth_raw_median": median_raw,
        "depth_raw_mad": mad_raw,
        "depth_raw_p10": p10,
        "depth_raw_p90": p90,
        "depth_raw_p90_minus_p10": p90 - p10,
        "camera_z": camera_z,
        "camera_z_mad": camera_z_mad,
    }


def metric_packet_patch_diagnostics(
    packet: Dict[str, np.ndarray],
    depth_u: float,
    depth_v: float,
    patch_size: int,
) -> Dict[str, Any]:
    required = [
        "accumulated_alpha",
        "camera_z_variance",
        "metric_depth_valid_mask",
        DIAGNOSTIC_VARIANCE_TENSOR,
        DIAGNOSTIC_VARIANCE_VALID_MASK_TENSOR,
    ]
    if any(name not in packet for name in required):
        return {}
    alpha = np.asarray(packet["accumulated_alpha"], dtype=np.float64)
    raw_variance = np.asarray(packet["camera_z_variance"], dtype=np.float64)
    metric_valid_mask = np.asarray(packet["metric_depth_valid_mask"]).astype(bool)
    diagnostic_variance = np.asarray(packet[DIAGNOSTIC_VARIANCE_TENSOR], dtype=np.float64)
    diagnostic_mask = np.asarray(packet[DIAGNOSTIC_VARIANCE_VALID_MASK_TENSOR]).astype(bool)
    height, width = alpha.shape[:2]
    if depth_u < 0 or depth_v < 0 or depth_u >= width or depth_v >= height:
        return {}
    cx = int(round(float(depth_u)))
    cy = int(round(float(depth_v)))
    half = int(patch_size) // 2
    x0 = max(0, cx - half)
    x1 = min(width, cx + half + 1)
    y0 = max(0, cy - half)
    y1 = min(height, cy + half + 1)
    alpha_patch = alpha[y0:y1, x0:x1]
    raw_variance_patch = raw_variance[y0:y1, x0:x1]
    metric_valid_patch = metric_valid_mask[y0:y1, x0:x1]
    diagnostic_variance_patch = diagnostic_variance[y0:y1, x0:x1]
    diagnostic_mask_patch = diagnostic_mask[y0:y1, x0:x1]
    alpha_finite = alpha_patch[np.isfinite(alpha_patch)]
    diag_values = diagnostic_variance_patch[diagnostic_mask_patch & np.isfinite(diagnostic_variance_patch)]
    total = int(np.count_nonzero(metric_valid_patch))
    diagnostic_valid = int(np.count_nonzero(diagnostic_mask_patch))
    raw_negative = int(np.count_nonzero(np.isfinite(raw_variance_patch) & (raw_variance_patch < 0)))
    unresolved = int(np.count_nonzero(metric_valid_patch & (~diagnostic_mask_patch)))
    out: Dict[str, Any] = {
        "accumulated_alpha_patch_min": float(np.min(alpha_finite)) if alpha_finite.size else math.nan,
        "accumulated_alpha_patch_p10": float(np.percentile(alpha_finite, 10)) if alpha_finite.size else math.nan,
        "accumulated_alpha_patch_median": float(np.median(alpha_finite)) if alpha_finite.size else math.nan,
        "variance_diagnostic_valid_ratio": diagnostic_valid / max(1, total),
        "variance_raw_negative_count": raw_negative,
        "variance_nonnegativity_unresolved_count": unresolved,
        "observation_view_count": 1,
    }
    if diag_values.size:
        out["variance_diagnostic_median"] = float(np.median(diag_values))
        out["variance_diagnostic_p90"] = float(np.percentile(diag_values, 90))
    else:
        out["variance_diagnostic_median"] = math.nan
        out["variance_diagnostic_p90"] = math.nan
    return out


def aggregate_points(points: List[np.ndarray]) -> tuple[np.ndarray, Dict[str, float]]:
    stack = np.vstack(points)
    aggregate = np.median(stack, axis=0)
    distances = np.linalg.norm(stack - aggregate.reshape(1, 3), axis=1)
    return aggregate, {
        "scatter_median_m": float(np.median(distances)),
        "scatter_p90_m": float(np.percentile(distances, 90)),
        "scatter_max_m": float(np.max(distances)),
        "scatter_mean_m": float(np.mean(distances)),
    }


def aggregation_mode(valid_count: int) -> str:
    if valid_count <= 0:
        return ""
    if valid_count == 1:
        return "single_view"
    if valid_count == 2:
        return "two_view_median"
    return "robust_multiview_median"


def residual_row(point_name: str, role: str, model_xyz: np.ndarray, target_xyz: np.ndarray, pred_xyz: np.ndarray) -> Dict[str, Any]:
    residual = pred_xyz - target_xyz
    return {
        "point_name": point_name,
        "role": role,
        "model_x": model_xyz[0],
        "model_y": model_xyz[1],
        "model_z": model_xyz[2],
        "target_x": target_xyz[0],
        "target_y": target_xyz[1],
        "target_z": target_xyz[2],
        "predicted_x": pred_xyz[0],
        "predicted_y": pred_xyz[1],
        "predicted_z": pred_xyz[2],
        "residual_x_m": residual[0],
        "residual_y_m": residual[1],
        "residual_z_m": residual[2],
        "error_h_m": float(np.linalg.norm(residual[:2])),
        "error_z_m": float(abs(residual[2])),
        "error_3d_m": float(np.linalg.norm(residual)),
    }


def make_report(summary: Dict[str, Any], checkpoint_rows: List[Dict[str, Any]], failure_rows: List[Dict[str, Any]]) -> str:
    lines = [
        "# Gaussian GCP Geometry Evaluation",
        "",
        f"- Scene: `{summary['scene']}`",
        f"- Method: `{summary['method_id']}`",
        f"- Evaluator mode: `{summary['evaluator_mode']}`",
        f"- Status: `{summary['status']}`",
        "",
        "## Counts",
        "",
        f"- Raw observations: `{summary['raw_observation_rows']}`",
        f"- Valid observations: `{summary['valid_observation_rows']}`",
        f"- Aggregated GCPs: `{summary['aggregated_gcp_count']}`",
        f"- Control GCPs used: `{summary['control_count']}`",
        f"- Checkpoint GCPs used: `{summary['checkpoint_count']}`",
        "",
        "## Checkpoint Residuals",
        "",
    ]
    stats = summary["residual_stats"].get("checkpoint", {})
    if stats.get("count", 0):
        lines.extend(
            [
                f"- RMSE-H: `{stats['rmse_h_m']:.4f} m`",
                f"- RMSE-Z: `{stats['rmse_z_m']:.4f} m`",
                f"- RMSE-3D: `{stats['rmse_3d_m']:.4f} m`",
                f"- P95-3D: `{stats['p95_3d_m']:.4f} m`",
                f"- Max-3D: `{stats['max_3d_m']:.4f} m`",
                "",
                "| Point | eH (m) | eZ (m) | e3D (m) |",
                "|---|---:|---:|---:|",
            ]
        )
        for row in sorted(checkpoint_rows, key=lambda x: float(x["error_3d_m"]), reverse=True):
            lines.append(
                f"| {row['point_name']} | {float(row['error_h_m']):.4f} | "
                f"{float(row['error_z_m']):.4f} | {float(row['error_3d_m']):.4f} |"
            )
    else:
        lines.append("No valid held-out checkpoint residuals were produced.")
    lines.extend(["", "## Failure Summary", ""])
    if failure_rows:
        lines.extend(["| Reason | Count |", "|---|---:|"])
        for row in failure_rows:
            lines.append(f"| {row['failure_reason']} | {row['count']} |")
    else:
        lines.append("No failures recorded.")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This report evaluates method-derived depth/support positions at manually annotated GCP image locations. "
            "It does not identify individual Gaussian primitives and does not report COLMAP-camera triangulation as a method result.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_roundtrip_unit_test() -> Dict[str, Any]:
    cases = []
    for x_norm, y_norm, camera_z in [(0.0, 0.0, 10.0), (0.2, -0.1, 30.0), (-0.35, 0.18, 80.0)]:
        ray_norm = math.sqrt(x_norm * x_norm + y_norm * y_norm + 1.0)
        values = {
            "camera_z": camera_z,
            "ray_distance": camera_z * ray_norm,
            "inverse_camera_z": 1.0 / camera_z,
            "inverse_ray_distance": 1.0 / (camera_z * ray_norm),
        }
        for semantics, value in values.items():
            recovered = camera_z_from_depth_value(value, x_norm, y_norm, semantics)
            cases.append(
                {
                    "x_norm": x_norm,
                    "y_norm": y_norm,
                    "semantics": semantics,
                    "stored_value": value,
                    "expected_camera_z": camera_z,
                    "recovered_camera_z": recovered,
                    "abs_error": abs(recovered - camera_z),
                }
            )
    max_error = max(row["abs_error"] for row in cases)
    return {
        "schema": "ms_gcp_depth_semantics_roundtrip_unit_test_v1",
        "case_count": len(cases),
        "max_abs_error": max_error,
        "passed": bool(max_error < 1e-9),
        "cases": cases,
    }


def main() -> None:
    argv = sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="Depth-only Gaussian GCP geometry evaluator core."
    )
    parser.add_argument("--run_roundtrip_unit_test", action="store_true")
    parser.add_argument(
        "--release_config",
        default="",
        help=(
            "Frozen benchmark release config. When supplied, evaluator inputs are "
            "resolved from this config, SHA-256 hashes are verified, and "
            "control/checkpoint roles are loaded from the frozen split."
        ),
    )
    parser.add_argument("--scene", default="")
    parser.add_argument("--method_id", default="unknown_method")
    parser.add_argument("--colmap_model")
    parser.add_argument("--depth_dir")
    parser.add_argument("--depth_manifest")
    parser.add_argument("--annotations_csv")
    parser.add_argument("--gcp_csv")
    parser.add_argument("--split_csv")
    parser.add_argument("--scene_metadata_csv")
    parser.add_argument("--out_dir")
    parser.add_argument("--control_points", default="")
    parser.add_argument("--checkpoint_points", default="")
    parser.add_argument(
        "--control_policy",
        default="diagnostic_available_subset",
        choices=["require_all", "diagnostic_available_subset"],
    )
    parser.add_argument("--target_fields", default=",".join(DEFAULT_TARGET_FIELDS))
    parser.add_argument("--depth_semantics", default="", choices=[
        "",
        "camera_z",
        "ray_distance",
        "inverse_camera_z",
        "inverse_ray_distance",
        "alpha_weighted_unnormalized_inverse_camera_z",
    ])
    parser.add_argument("--image_domain", default="same_as_colmap_camera")
    parser.add_argument("--pixel_coordinate_convention", default="zero_indexed_pixel_centers")
    parser.add_argument("--depth_scale", type=float, default=1.0)
    parser.add_argument("--depth_offset", type=float, default=0.0)
    parser.add_argument("--depth_pixel_scale_x", type=float, default=1.0)
    parser.add_argument("--depth_pixel_scale_y", type=float, default=1.0)
    parser.add_argument("--npz_key", default="depth")
    parser.add_argument("--patch_size", type=int, default=7)
    parser.add_argument("--min_patch_valid_ratio", type=float, default=0.60)
    parser.add_argument("--min_depth", type=float, default=1e-6)
    parser.add_argument("--min_confidence", type=float, default=1.0)
    parser.add_argument("--min_valid_observations", type=int, default=3)
    parser.add_argument("--max_multiview_scatter_m", type=float, default=0.0)
    args = parser.parse_args(argv)

    if args.run_roundtrip_unit_test:
        result = run_roundtrip_unit_test()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not result["passed"]:
            raise SystemExit(1)
        return

    release_config_path: Path | None = None
    release_config: Dict[str, Any] | None = None
    release_verified_files: List[Dict[str, Any]] = []
    canonical_release_config_sha256 = ""
    relocated_release_config_sha256 = ""
    split_csv: Path | None = None
    scene_metadata_csv: Path | None = None
    release_config_sha256 = ""
    depth_manifest_path: Path | None = None
    depth_manifest: Dict[str, Any] | None = None
    depth_manifest_sha256 = ""
    depth_index: Dict[str, Dict[str, Any]] = {}
    metric_packet_manifest = False
    metric_packet_numerical_support_floor = 0.0
    metric_packet_variance_clamp_tolerance = 0.0
    metric_packet_variance_validation_policy = VARIANCE_VALIDATION_POLICY
    metric_packet_variance_validation_abs_floor = 0.0
    metric_packet_variance_validation_ulp_factor = 0.0
    metric_packet_variance_validation_dtype = DEFAULT_VARIANCE_VALIDATION_DTYPE
    metric_packet_variance_validation_rtol = 0.0
    metric_packet_variance_nonnegativity_policy = VARIANCE_NONNEGATIVITY_POLICY
    metric_packet_variance_negative_handling = VARIANCE_NEGATIVE_HANDLING
    metric_packet_variance_raw_packet_modified = VARIANCE_RAW_PACKET_MODIFIED
    scene_metadata_rows: Dict[str, Dict[str, str]] = {}

    if args.release_config:
        forbidden_overrides = [
            "annotations_csv",
            "gcp_csv",
            "split_csv",
            "scene_metadata_csv",
            "depth_dir",
            "depth_semantics",
            "image_domain",
            "pixel_coordinate_convention",
            "depth_scale",
            "depth_offset",
            "npz_key",
        ]
        supplied = [name for name in forbidden_overrides if cli_flag_supplied(argv, name)]
        if supplied:
            raise SystemExit(
                "--release_config mode resolves these inputs from frozen manifests; "
                f"do not pass manual overrides: {', '.join('--' + name for name in supplied)}"
            )
        if not args.depth_manifest:
            raise SystemExit("--depth_manifest is required when --release_config is used")
        release_config_path = Path(args.release_config)
        release_config_sha256 = file_sha256(release_config_path)
        release_config = load_release_config(release_config_path)
        release_schema = str(release_config.get("schema", ""))
        release_verified_files = verify_release_files(release_config_path, release_config)
        release_registry = release_file_registry(release_verified_files)
        release_base = release_config_path.parent
        if not args.scene:
            raise SystemExit("--scene is required when --release_config is used")
        if args.control_points or args.checkpoint_points:
            raise SystemExit(
                "--release_config uses the frozen split; do not also pass "
                "--control_points or --checkpoint_points"
            )
        if cli_flag_supplied(argv, "min_valid_observations") and int(args.min_valid_observations) != 1:
            raise SystemExit("--release_config mode fixes --min_valid_observations to 1")
        args.control_policy = "require_all"
        args.min_valid_observations = 1
        canonical_release_config_sha256 = str(
            release_config.get("canonical_release_config_sha256")
            or release_config.get("canonical_config_sha256")
            or release_config_sha256
        )
        relocated_release_config_sha256 = release_config_sha256
        args.gcp_csv = str(release_base / release_config["gcp_csv"])
        args.split_csv = str(release_base / release_config["split_csv"])
        args.scene_metadata_csv = str(release_base / release_config["scene_metadata_csv"])
        scene_metadata_rows = load_scene_metadata(Path(args.scene_metadata_csv))
        if args.scene not in scene_metadata_rows:
            raise SystemExit(f"Unknown scene for frozen release config: {args.scene}")
        if release_schema == RELEASE_V12_SCHEMA:
            annotation_name = f"{args.scene}_gcp_annotations_pixel_domain_v1_2.csv"
        else:
            annotation_name = f"{args.scene}_gcp_annotations_final_good_nadir_v1.csv"
        annotation_path = release_base / annotation_name
        if annotation_name not in release_registry and str(annotation_path.resolve()) not in release_registry:
            raise SystemExit(
                f"Frozen annotation file for scene {args.scene} is not in release registry: {annotation_name}"
            )
        args.annotations_csv = str(annotation_path)

    if args.depth_manifest:
        depth_manifest_path = Path(args.depth_manifest)
        depth_manifest_sha256 = file_sha256(depth_manifest_path)
        depth_manifest = load_depth_manifest(depth_manifest_path)
        metric_packet_manifest = depth_manifest.get("schema", "") == METRIC_PACKET_MANIFEST_SCHEMA
        if metric_packet_manifest:
            manifest_semantics = str(depth_manifest["primary_depth_semantics"]).strip()
            manifest_npz_key = str(depth_manifest["primary_depth_tensor"]).strip()
            if manifest_npz_key != PRIMARY_DEPTH_TENSOR:
                raise SystemExit(f"Unsupported metric packet primary tensor: {manifest_npz_key}")
            args.npz_key = manifest_npz_key
            args.depth_scale = 1.0
            args.depth_offset = 0.0
            metric_packet_numerical_support_floor = float(depth_manifest["numerical_support_floor"])
            metric_packet_variance_clamp_tolerance = float(depth_manifest["variance_clamp_tolerance"])
            metric_packet_variance_validation_policy = str(depth_manifest["variance_validation_policy"])
            metric_packet_variance_validation_abs_floor = float(depth_manifest["variance_validation_abs_floor"])
            metric_packet_variance_validation_ulp_factor = float(depth_manifest["variance_validation_ulp_factor"])
            metric_packet_variance_validation_dtype = str(depth_manifest["variance_validation_dtype"])
            metric_packet_variance_validation_rtol = float(depth_manifest["variance_validation_rtol"])
            metric_packet_variance_nonnegativity_policy = str(depth_manifest["variance_nonnegativity_policy"])
            metric_packet_variance_negative_handling = str(depth_manifest["variance_negative_handling"])
            metric_packet_variance_raw_packet_modified = bool(depth_manifest["variance_raw_packet_modified"])
        else:
            manifest_semantics = str(depth_manifest["depth_semantics"]).strip()
        if args.depth_semantics and args.depth_semantics != manifest_semantics:
            raise SystemExit(
                f"CLI --depth_semantics ({args.depth_semantics}) does not match depth manifest "
                f"({manifest_semantics})"
            )
        args.depth_semantics = manifest_semantics
        reject_unsupported_depth_semantics(args.depth_semantics)
        manifest_domain = str(depth_manifest.get("image_domain", "")).strip()
        if manifest_domain:
            args.image_domain = manifest_domain
        manifest_pixel_convention = str(depth_manifest.get("pixel_coordinate_convention", "")).strip()
        if manifest_pixel_convention:
            args.pixel_coordinate_convention = manifest_pixel_convention
        depth_index = load_depth_index(depth_manifest_path, depth_manifest)
        if depth_manifest.get("depth_output_dir"):
            manifest_depth_dir = Path(str(depth_manifest["depth_output_dir"]))
            if not manifest_depth_dir.is_absolute():
                manifest_depth_dir = depth_manifest_path.parent / manifest_depth_dir
            args.depth_dir = str(manifest_depth_dir)
        elif depth_manifest.get("mapping_csv"):
            manifest_mapping = Path(str(depth_manifest["mapping_csv"]))
            if not manifest_mapping.is_absolute():
                manifest_mapping = depth_manifest_path.parent / manifest_mapping
            args.depth_dir = str(manifest_mapping.parent)
        if not metric_packet_manifest:
            args.depth_scale = float(depth_manifest.get("depth_scale_for_evaluator", args.depth_scale))
            args.depth_offset = float(depth_manifest.get("depth_offset_for_evaluator", args.depth_offset))
    elif args.depth_semantics:
        reject_unsupported_depth_semantics(args.depth_semantics)
    else:
        raise SystemExit("--depth_semantics is required unless --depth_manifest is supplied")

    required = ["colmap_model", "depth_dir", "annotations_csv", "gcp_csv", "out_dir"]
    missing = [name for name in required if not getattr(args, name)]
    if missing:
        raise SystemExit(f"Missing required arguments: {', '.join(missing)}")

    colmap_model = Path(args.colmap_model)
    depth_dir = Path(args.depth_dir)
    annotations_csv = Path(args.annotations_csv)
    gcp_csv = Path(args.gcp_csv)
    split_csv = Path(args.split_csv) if args.split_csv else None
    scene_metadata_csv = Path(args.scene_metadata_csv) if args.scene_metadata_csv else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if scene_metadata_csv and not scene_metadata_rows:
        scene_metadata_rows = load_scene_metadata(scene_metadata_csv)
        if args.scene and args.scene not in scene_metadata_rows:
            raise SystemExit(f"Unknown scene in scene metadata: {args.scene}")

    cameras, images, _points3d = read_model(colmap_model)
    images_by_name = {image.name: image for image in images.values()}
    target_fields = [field.strip() for field in args.target_fields.split(",") if field.strip()]
    target_points = load_target_points(gcp_csv, target_fields)
    if split_csv:
        if not args.scene:
            raise SystemExit("--scene is required when --split_csv is used")
        if args.control_points or args.checkpoint_points:
            raise SystemExit(
                "Use either --split_csv or explicit --control_points/--checkpoint_points, not both"
            )
        control_points, checkpoint_points = load_split_roles(split_csv, args.scene)
    else:
        control_points = parse_name_set(args.control_points)
        checkpoint_points = parse_name_set(args.checkpoint_points)

    raw_rows = read_csv(annotations_csv)
    if args.scene:
        try:
            validate_annotation_rows_scene(raw_rows, args.scene)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    if release_config and release_config.get("schema") == RELEASE_V12_SCHEMA:
        if depth_manifest is None:
            raise SystemExit("v1.2 release mode requires --depth_manifest")
        try:
            raw_rows = validate_release_v12_rows_for_evaluator(
                release_base=release_config_path.parent if release_config_path else annotations_csv.parent,
                scene=args.scene,
                rows=raw_rows,
                colmap_cameras=cameras,
                colmap_images=images,
                depth_manifest=depth_manifest,
            )
        except ValueError as exc:
            raise SystemExit(f"v1.2 release pixel-domain validation failed: {exc}") from exc
    observation_rows: List[Dict[str, Any]] = []
    valid_points_by_gcp: Dict[str, List[np.ndarray]] = defaultdict(list)
    failure_counter: Counter[str] = Counter()
    depth_cache: Dict[str, tuple[Path, np.ndarray, Dict[str, Any]]] = {}

    for row in raw_rows:
        point_name = row.get("point_name", "").strip()
        image_name = row.get("image_name", "").strip()
        base_out: Dict[str, Any] = {
            "scene": args.scene or row.get("scene", ""),
            "method_id": args.method_id,
            "point_name": point_name,
            "image_name": image_name,
            "u_px": row.get("u_px", row.get("manual_x", "")),
            "v_px": row.get("v_px", row.get("manual_y", "")),
            "valid": 0,
            "failure_reason": "",
        }
        usable, reason = observation_is_usable(row, min_confidence=float(args.min_confidence))
        if not usable:
            base_out["failure_reason"] = reason
            failure_counter[reason] += 1
            observation_rows.append(base_out)
            continue
        if image_name not in images_by_name:
            base_out["failure_reason"] = "missing_image_in_colmap"
            failure_counter["missing_image_in_colmap"] += 1
            observation_rows.append(base_out)
            continue
        if point_name not in target_points:
            base_out["failure_reason"] = "missing_survey_coordinate"
            failure_counter["missing_survey_coordinate"] += 1
            observation_rows.append(base_out)
            continue
        image = images_by_name[image_name]
        camera = cameras[image.camera_id]
        camera_width = int(camera.width)
        camera_height = int(camera.height)
        depth_entry: Dict[str, Any] = {}
        if depth_index:
            depth_entry = depth_index.get(image_name) or depth_index.get(Path(image_name).name) or {}
            if not depth_entry:
                base_out["failure_reason"] = "missing_depth_manifest_entry"
                failure_counter["missing_depth_manifest_entry"] += 1
                observation_rows.append(base_out)
                continue
            depth_path = Path(str(depth_entry["depth_path"]))
            if not depth_path.exists():
                base_out["failure_reason"] = "missing_depth_map"
                failure_counter["missing_depth_map"] += 1
                observation_rows.append(base_out)
                continue
        else:
            depth_path = find_depth_path(depth_dir, image_name)
            if depth_path is None:
                base_out["failure_reason"] = "missing_depth_map"
                failure_counter["missing_depth_map"] += 1
                observation_rows.append(base_out)
                continue
        cache_key = str(depth_path)
        if cache_key not in depth_cache:
            packet_validation: Dict[str, Any] = {}
            if metric_packet_manifest:
                expected_hash = str(
                    depth_entry.get("packet_sha256")
                    or depth_entry.get("sha256")
                    or depth_entry.get("file_sha256")
                    or ""
                ).strip()
                if not expected_hash:
                    raise SystemExit(f"Metric packet index row is missing packet SHA-256: {image_name}")
                actual_hash = file_sha256(depth_path)
                if actual_hash != expected_hash:
                    raise SystemExit(
                        f"Metric packet hash mismatch for {image_name}: expected {expected_hash}, got {actual_hash}"
                    )
                packet_payload = validate_metric_packet_npz(
                    depth_path,
                    depth_entry,
                    numerical_support_floor=metric_packet_numerical_support_floor,
                    variance_clamp_tolerance=metric_packet_variance_clamp_tolerance,
                    variance_validation_policy=metric_packet_variance_validation_policy,
                    variance_validation_abs_floor=metric_packet_variance_validation_abs_floor,
                    variance_validation_ulp_factor=metric_packet_variance_validation_ulp_factor,
                    variance_validation_dtype=metric_packet_variance_validation_dtype,
                    variance_validation_rtol=metric_packet_variance_validation_rtol,
                    variance_nonnegativity_policy=metric_packet_variance_nonnegativity_policy,
                    variance_negative_handling=metric_packet_variance_negative_handling,
                    variance_raw_packet_modified=metric_packet_variance_raw_packet_modified,
                )
                packet_validation = {
                    "packet_sha256": actual_hash,
                    "packet_tensor_count": len(packet_payload),
                    "primary_depth_tensor": PRIMARY_DEPTH_TENSOR,
                    "_metric_packet_payload": packet_payload,
                }
            depth = load_depth_map(
                depth_path,
                scale=float(args.depth_scale),
                offset=float(args.depth_offset),
                npz_key=str(args.npz_key),
            )
            if depth.ndim != 2:
                raise SystemExit(f"Depth map must be 2D after loading: {depth_path} shape={depth.shape}")
            depth_height, depth_width = int(depth.shape[0]), int(depth.shape[1])
            if depth_entry:
                expected_h = int(depth_entry["height"])
                expected_w = int(depth_entry["width"])
                if depth_height != expected_h or depth_width != expected_w:
                    raise SystemExit(
                        f"Depth shape mismatch for {image_name}: manifest {expected_w}x{expected_h}, "
                        f"loaded {depth_width}x{depth_height}"
                    )
            depth_cache[cache_key] = (
                depth_path,
                depth,
                {
                    "depth_width": depth_width,
                    "depth_height": depth_height,
                    **packet_validation,
                },
            )
        _depth_path, depth, depth_meta = depth_cache[cache_key]
        depth_width = int(depth_meta["depth_width"])
        depth_height = int(depth_meta["depth_height"])
        derived_scale_x = depth_width / max(1, camera_width)
        derived_scale_y = depth_height / max(1, camera_height)
        if depth_index:
            depth_pixel_scale_x = derived_scale_x
            depth_pixel_scale_y = derived_scale_y
        else:
            depth_pixel_scale_x = float(args.depth_pixel_scale_x)
            depth_pixel_scale_y = float(args.depth_pixel_scale_y)
            if (
                abs(depth_pixel_scale_x - derived_scale_x) > 1e-6
                or abs(depth_pixel_scale_y - derived_scale_y) > 1e-6
            ):
                raise SystemExit(
                    f"Depth pixel scale mismatch for {image_name}: supplied "
                    f"{depth_pixel_scale_x},{depth_pixel_scale_y}; derived "
                    f"{derived_scale_x},{derived_scale_y} from depth/camera dimensions"
                )
        u = float(base_out["u_px"])
        v = float(base_out["v_px"])
        depth_u = u * depth_pixel_scale_x
        depth_v = v * depth_pixel_scale_y
        base_out["geometry_u_px"] = u
        base_out["geometry_v_px"] = v
        base_out["depth_u_px"] = depth_u
        base_out["depth_v_px"] = depth_v
        base_out["camera_width"] = camera_width
        base_out["camera_height"] = camera_height
        base_out["depth_width"] = depth_width
        base_out["depth_height"] = depth_height
        if metric_packet_manifest:
            base_out["depth_packet_schema"] = METRIC_PACKET_MANIFEST_SCHEMA
            base_out["primary_depth_tensor"] = PRIMARY_DEPTH_TENSOR
            base_out["packet_sha256"] = depth_meta.get("packet_sha256", "")
        base_out["depth_pixel_scale_x"] = depth_pixel_scale_x
        base_out["depth_pixel_scale_y"] = depth_pixel_scale_y
        valid, patch_stats = robust_depth_patch(
            depth=depth,
            camera=camera,
            u=u,
            v=v,
            depth_u=depth_u,
            depth_v=depth_v,
            depth_pixel_scale_x=depth_pixel_scale_x,
            depth_pixel_scale_y=depth_pixel_scale_y,
            patch_size=int(args.patch_size),
            min_valid_ratio=float(args.min_patch_valid_ratio),
            min_depth=float(args.min_depth),
            depth_semantics=str(args.depth_semantics),
        )
        base_out.update(patch_stats)
        if metric_packet_manifest and "_metric_packet_payload" in depth_meta:
            base_out.update(
                metric_packet_patch_diagnostics(
                    packet=depth_meta["_metric_packet_payload"],
                    depth_u=depth_u,
                    depth_v=depth_v,
                    patch_size=int(args.patch_size),
                )
            )
        base_out["depth_path"] = str(depth_path)
        if not valid:
            reason = str(patch_stats.get("failure_reason", "invalid_depth"))
            failure_counter[reason] += 1
            observation_rows.append(base_out)
            continue
        xyz = backproject_world(camera, image, u, v, float(patch_stats["camera_z"]))
        base_out.update(
            {
                "valid": 1,
                "model_x": xyz[0],
                "model_y": xyz[1],
                "model_z": xyz[2],
            }
        )
        valid_points_by_gcp[point_name].append(xyz)
        observation_rows.append(base_out)

    aggregated_rows: List[Dict[str, Any]] = []
    scatter_rows: List[Dict[str, Any]] = []
    gcp_failure_counter: Counter[str] = Counter()
    aggregated_points: Dict[str, np.ndarray] = {}
    for point_name in sorted(set(row.get("point_name", "") for row in raw_rows if row.get("point_name", ""))):
        points = valid_points_by_gcp.get(point_name, [])
        raw_count = sum(1 for row in raw_rows if row.get("point_name", "") == point_name)
        valid_count = len(points)
        row_base = {
            "scene": args.scene,
            "method_id": args.method_id,
            "point_name": point_name,
            "raw_observation_count": raw_count,
            "valid_observation_count": valid_count,
            "valid_observation_ratio": valid_count / max(1, raw_count),
            "aggregation_mode": aggregation_mode(valid_count),
            "multiview_robust_eligible": int(valid_count >= 3),
            "valid": 0,
            "failure_reason": "",
        }
        if valid_count < int(args.min_valid_observations):
            row_base["failure_reason"] = "insufficient_valid_observations"
            gcp_failure_counter["insufficient_valid_observations"] += 1
            aggregated_rows.append(row_base)
            continue
        aggregate, scatter = aggregate_points(points)
        row_base.update(scatter)
        if float(args.max_multiview_scatter_m) > 0 and scatter["scatter_max_m"] > float(args.max_multiview_scatter_m):
            row_base["failure_reason"] = "high_multiview_scatter"
            gcp_failure_counter["high_multiview_scatter"] += 1
            aggregated_rows.append(row_base)
            continue
        row_base.update(
            {
                "valid": 1,
                "model_x": aggregate[0],
                "model_y": aggregate[1],
                "model_z": aggregate[2],
            }
        )
        aggregated_points[point_name] = aggregate
        aggregated_rows.append(row_base)
        for point in points:
            distance = float(np.linalg.norm(point - aggregate))
            scatter_rows.append(
                {
                    "scene": args.scene,
                    "method_id": args.method_id,
                    "point_name": point_name,
                    "distance_to_aggregate_m": distance,
                }
            )

    multiview_robust_subset = [
        {
            "scene": row["scene"],
            "method_id": row["method_id"],
            "point_name": row["point_name"],
            "valid_observation_count": row["valid_observation_count"],
            "aggregation_mode": row.get("aggregation_mode", ""),
            "scatter_median_m": row.get("scatter_median_m", ""),
            "scatter_p90_m": row.get("scatter_p90_m", ""),
            "scatter_max_m": row.get("scatter_max_m", ""),
            "scatter_mean_m": row.get("scatter_mean_m", ""),
        }
        for row in aggregated_rows
        if int(row.get("valid", 0)) and int(row.get("valid_observation_count", 0)) >= 3
    ]

    requested_controls = sorted(control_points & set(target_points))
    requested_checkpoints = sorted(checkpoint_points & set(target_points))
    common_controls = sorted(set(requested_controls) & set(aggregated_points))
    common_checkpoints = sorted(set(requested_checkpoints) & set(aggregated_points))
    missing_controls = sorted(set(requested_controls) - set(common_controls))
    missing_checkpoints = sorted(set(requested_checkpoints) - set(common_checkpoints))
    control_coverage = len(common_controls) / max(1, len(requested_controls))
    checkpoint_coverage = len(common_checkpoints) / max(1, len(requested_checkpoints))
    control_rows: List[Dict[str, Any]] = []
    checkpoint_rows: List[Dict[str, Any]] = []
    residual_summary = {
        "control": residual_stats(np.empty((0, 3), dtype=np.float64)),
        "checkpoint": residual_stats(np.empty((0, 3), dtype=np.float64)),
        "all": residual_stats(np.empty((0, 3), dtype=np.float64)),
    }
    transform_payload: Dict[str, Any] | None = None
    status = "failed"
    allow_fit = True
    if args.control_policy == "require_all" and missing_controls:
        status = "incomplete_fixed_control_coverage"
        allow_fit = False
    if allow_fit and len(common_controls) >= 3 and common_checkpoints:
        source_control = np.vstack([aggregated_points[name] for name in common_controls])
        target_control = np.vstack([target_points[name] for name in common_controls])
        scale, rotation, translation = fit_similarity_umeyama(source_control, target_control, estimate_scale=True)
        transform_payload = {
            "scale": scale,
            "rotation": rotation.tolist(),
            "translation": translation.tolist(),
            "definition": "target_xyz = scale * rotation @ model_xyz + translation",
        }
        all_residuals: List[np.ndarray] = []
        control_residuals: List[np.ndarray] = []
        checkpoint_residuals: List[np.ndarray] = []
        for point_name in common_controls:
            model_xyz = aggregated_points[point_name]
            target_xyz = target_points[point_name]
            pred_xyz = apply_similarity(model_xyz.reshape(1, 3), scale, rotation, translation)[0]
            row = residual_row(point_name, "control", model_xyz, target_xyz, pred_xyz)
            control_rows.append(row)
            residual = pred_xyz - target_xyz
            control_residuals.append(residual)
            all_residuals.append(residual)
        for point_name in common_checkpoints:
            model_xyz = aggregated_points[point_name]
            target_xyz = target_points[point_name]
            pred_xyz = apply_similarity(model_xyz.reshape(1, 3), scale, rotation, translation)[0]
            row = residual_row(point_name, "checkpoint", model_xyz, target_xyz, pred_xyz)
            checkpoint_rows.append(row)
            residual = pred_xyz - target_xyz
            checkpoint_residuals.append(residual)
            all_residuals.append(residual)
        residual_summary = {
            "control": residual_stats(np.vstack(control_residuals)),
            "checkpoint": residual_stats(np.vstack(checkpoint_residuals)),
            "all": residual_stats(np.vstack(all_residuals)),
        }
        status = "ok" if args.control_policy == "require_all" else "diagnostic_available_subset"
    elif allow_fit and len(common_controls) >= 3:
        status = "smoke_only"

    failure_rows = [
        {"level": "observation", "failure_reason": reason, "count": count}
        for reason, count in sorted(failure_counter.items())
    ] + [
        {"level": "gcp", "failure_reason": reason, "count": count}
        for reason, count in sorted(gcp_failure_counter.items())
    ]

    summary = {
        "schema": "ms_gcp_depth_only_gaussian_geometry_eval_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scene": args.scene,
        "method_id": args.method_id,
        "evaluator_mode": "depth_only_p1",
        "status": status,
        "evaluator_source_commit": git_commit(REPO_ROOT),
        "colmap_model": str(colmap_model),
        "depth_dir": str(depth_dir),
        "annotations_csv": str(annotations_csv),
        "gcp_csv": str(gcp_csv),
        "target_fields": target_fields,
        "image_domain": args.image_domain,
        "pixel_coordinate_convention": args.pixel_coordinate_convention,
        "depth_semantics": args.depth_semantics,
        "depth_pixel_scale_x": float(args.depth_pixel_scale_x),
        "depth_pixel_scale_y": float(args.depth_pixel_scale_y),
        "release_config": str(release_config_path) if release_config_path else "",
        "release_config_sha256": release_config_sha256,
        "release_config_schema": release_config.get("schema", "") if release_config else "",
        "canonical_release_config_sha256": canonical_release_config_sha256,
        "relocated_release_config_sha256": relocated_release_config_sha256,
        "release_id": release_config.get("release_id", "") if release_config else "",
        "release_verified_files": release_verified_files,
        "depth_manifest": str(depth_manifest_path) if depth_manifest_path else "",
        "depth_manifest_sha256": depth_manifest_sha256,
        "depth_manifest_summary": {
            "schema": depth_manifest.get("schema", "") if depth_manifest else "",
            "packet_schema": depth_manifest.get("packet_schema", "") if depth_manifest else "",
            "primary_depth_tensor": depth_manifest.get("primary_depth_tensor", "") if depth_manifest else "",
            "primary_depth_semantics": depth_manifest.get("primary_depth_semantics", "") if depth_manifest else "",
            "depth_semantics": depth_manifest.get("depth_semantics", depth_manifest.get("primary_depth_semantics", "")) if depth_manifest else "",
            "depth_units": depth_manifest.get("depth_units", "") if depth_manifest else "",
            "image_domain": depth_manifest.get("image_domain", "") if depth_manifest else "",
            "renderer_repository": depth_manifest.get("renderer_repository", depth_manifest.get("train_repo", "")) if depth_manifest else "",
            "renderer_commit": depth_manifest.get("renderer_commit", "") if depth_manifest else "",
            "rasterizer_commit": depth_manifest.get("rasterizer_commit", "") if depth_manifest else "",
            "rasterizer_tree_hash": depth_manifest.get("rasterizer_tree_hash", "") if depth_manifest else "",
            "exporter_commit": depth_manifest.get("exporter_commit", "") if depth_manifest else "",
            "model_content_hash": depth_manifest.get("model_content_hash", "") if depth_manifest else "",
            "numerical_support_floor": depth_manifest.get("numerical_support_floor", "") if depth_manifest else "",
            "variance_clamp_tolerance": depth_manifest.get("variance_clamp_tolerance", "") if depth_manifest else "",
            "variance_validation_policy": depth_manifest.get("variance_validation_policy", "") if depth_manifest else "",
            "variance_validation_abs_floor": depth_manifest.get("variance_validation_abs_floor", "") if depth_manifest else "",
            "variance_validation_ulp_factor": depth_manifest.get("variance_validation_ulp_factor", "") if depth_manifest else "",
            "variance_validation_dtype": depth_manifest.get("variance_validation_dtype", "") if depth_manifest else "",
            "variance_validation_rtol": depth_manifest.get("variance_validation_rtol", "") if depth_manifest else "",
            "variance_nonnegativity_policy": depth_manifest.get("variance_nonnegativity_policy", "") if depth_manifest else "",
            "variance_negative_handling": depth_manifest.get("variance_negative_handling", "") if depth_manifest else "",
            "variance_raw_packet_modified": depth_manifest.get("variance_raw_packet_modified", "") if depth_manifest else "",
            "normalization_epsilon": depth_manifest.get("normalization_epsilon", "") if depth_manifest else "",
            "tensor_names": depth_manifest.get("tensor_names", []) if depth_manifest else [],
            "alpha_map_available": bool(depth_manifest.get("alpha_map_available", depth_manifest.get("uses_alpha_map", False))) if depth_manifest else False,
            "depth_second_moment_available": bool(depth_manifest.get("depth_second_moment_available", depth_manifest.get("uses_depth_second_moment", False))) if depth_manifest else False,
            "depth_index_entry_count": len(depth_index),
            "rasterizer_source_trace": depth_manifest.get("rasterizer_source_trace", []) if depth_manifest else [],
        },
        "input_files": {
            "colmap_model": str(colmap_model),
            "depth_dir": str(depth_dir),
            "annotations_csv": str(annotations_csv),
            "gcp_csv": str(gcp_csv),
            "split_csv": str(split_csv) if split_csv else "",
            "scene_metadata_csv": str(scene_metadata_csv) if scene_metadata_csv else "",
            "depth_manifest": str(depth_manifest_path) if depth_manifest_path else "",
        },
        "control_policy": args.control_policy,
        "alpha_map_used": bool(metric_packet_manifest),
        "depth_second_moment_used": bool(metric_packet_manifest),
        "patch_size": int(args.patch_size),
        "min_patch_valid_ratio": float(args.min_patch_valid_ratio),
        "min_valid_observations": int(args.min_valid_observations),
        "multiview_robust_subset_count": len(multiview_robust_subset),
        "multiview_robust_subset_points": [row["point_name"] for row in multiview_robust_subset],
        "raw_observation_rows": len(raw_rows),
        "valid_observation_rows": int(sum(int(row.get("valid", 0)) for row in observation_rows)),
        "aggregated_gcp_count": len(aggregated_points),
        "control_points_requested": sorted(control_points),
        "checkpoint_points_requested": sorted(checkpoint_points),
        "frozen_control_points_requested": requested_controls,
        "frozen_checkpoint_points_requested": requested_checkpoints,
        "control_points_used": common_controls,
        "checkpoint_points_used": common_checkpoints,
        "missing_control_points": missing_controls,
        "missing_checkpoint_points": missing_checkpoints,
        "control_count": len(common_controls),
        "checkpoint_count": len(common_checkpoints),
        "frozen_control_count": len(requested_controls),
        "frozen_checkpoint_count": len(requested_checkpoints),
        "control_coverage": control_coverage,
        "checkpoint_coverage": checkpoint_coverage,
        "transform": transform_payload,
        "residual_stats": residual_summary,
        "failure_counts": failure_rows,
    }

    write_csv(
        out_dir / "method_gcp_observation_points.csv",
        observation_rows,
        [
            "scene",
            "method_id",
            "point_name",
            "image_name",
            "u_px",
            "v_px",
            "geometry_u_px",
            "geometry_v_px",
            "depth_u_px",
            "depth_v_px",
            "camera_width",
            "camera_height",
            "depth_width",
            "depth_height",
            "depth_pixel_scale_x",
            "depth_pixel_scale_y",
            "valid",
            "failure_reason",
            "depth_path",
            "depth_packet_schema",
            "primary_depth_tensor",
            "packet_sha256",
            "patch_x0",
            "patch_x1",
            "patch_y0",
            "patch_y1",
            "patch_valid_pixels",
            "patch_total_pixels",
            "patch_valid_ratio",
            "depth_raw_median",
            "depth_raw_mad",
            "depth_raw_p10",
            "depth_raw_p90",
            "depth_raw_p90_minus_p10",
            "camera_z",
            "camera_z_mad",
            "accumulated_alpha_patch_min",
            "accumulated_alpha_patch_p10",
            "accumulated_alpha_patch_median",
            "variance_diagnostic_median",
            "variance_diagnostic_p90",
            "variance_diagnostic_valid_ratio",
            "variance_raw_negative_count",
            "variance_nonnegativity_unresolved_count",
            "observation_view_count",
            "model_x",
            "model_y",
            "model_z",
        ],
    )
    write_csv(
        out_dir / "method_gcp_aggregated_points.csv",
        aggregated_rows,
        [
            "scene",
            "method_id",
            "point_name",
            "raw_observation_count",
            "valid_observation_count",
            "valid_observation_ratio",
            "aggregation_mode",
            "multiview_robust_eligible",
            "valid",
            "failure_reason",
            "model_x",
            "model_y",
            "model_z",
            "scatter_median_m",
            "scatter_p90_m",
            "scatter_max_m",
            "scatter_mean_m",
        ],
    )
    write_csv(
        out_dir / "method_gcp_multiview_scatter.csv",
        scatter_rows,
        ["scene", "method_id", "point_name", "distance_to_aggregate_m"],
    )
    write_csv(
        out_dir / "method_gcp_multiview_robust_subset.csv",
        multiview_robust_subset,
        [
            "scene",
            "method_id",
            "point_name",
            "valid_observation_count",
            "aggregation_mode",
            "scatter_median_m",
            "scatter_p90_m",
            "scatter_max_m",
            "scatter_mean_m",
        ],
    )
    write_csv(
        out_dir / "method_gcp_sim3_control_residuals.csv",
        control_rows,
        [
            "point_name",
            "role",
            "model_x",
            "model_y",
            "model_z",
            "target_x",
            "target_y",
            "target_z",
            "predicted_x",
            "predicted_y",
            "predicted_z",
            "residual_x_m",
            "residual_y_m",
            "residual_z_m",
            "error_h_m",
            "error_z_m",
            "error_3d_m",
        ],
    )
    write_csv(
        out_dir / "method_gcp_checkpoint_residuals.csv",
        checkpoint_rows,
        [
            "point_name",
            "role",
            "model_x",
            "model_y",
            "model_z",
            "target_x",
            "target_y",
            "target_z",
            "predicted_x",
            "predicted_y",
            "predicted_z",
            "residual_x_m",
            "residual_y_m",
            "residual_z_m",
            "error_h_m",
            "error_z_m",
            "error_3d_m",
        ],
    )
    write_csv(out_dir / "method_gcp_failure_summary.csv", failure_rows, ["level", "failure_reason", "count"])
    (out_dir / "method_gcp_eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "evaluator_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "method_gcp_eval_report.md").write_text(
        make_report(summary, checkpoint_rows, failure_rows), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
