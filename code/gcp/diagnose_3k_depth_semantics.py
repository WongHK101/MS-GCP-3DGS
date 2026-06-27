from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "colmap" / "utils"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_gaussian_gcp_geometry import (  # noqa: E402
    DIAGNOSTIC_VARIANCE_TENSOR,
    DIAGNOSTIC_VARIANCE_VALID_MASK_TENSOR,
    backproject_world,
    metric_packet_patch_diagnostics,
    pixel_to_normalized,
    validate_metric_packet_npz,
)
from fit_gcp_sim3 import (  # noqa: E402
    DEFAULT_TARGET_FIELDS,
    apply_similarity,
    fit_similarity_umeyama,
    residual_stats,
)
from read_write_model import qvec2rotmat, read_model  # noqa: E402


SCENE = "gcp_3000_20260602"
ZIP_PREFIX = "GPT_GCP_3SCENE_METRIC_DEPTH_REGRESSION_RELEASEMODE_REVIEW_20260627"
FORMAL_EVAL_DIR = f"{ZIP_PREFIX}/evaluations/{SCENE}_formal_expected_camera_z_release"
PACKET_MANIFEST_DIR = f"{ZIP_PREFIX}/packet_manifests/{SCENE}_full_reused_release"
RELEASE_DIR = f"{ZIP_PREFIX}/release"


def read_csv_path(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_csv_zip(zf: zipfile.ZipFile, name: str) -> List[Dict[str, str]]:
    text = zf.read(name).decode("utf-8-sig").splitlines()
    return list(csv.DictReader(text))


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def finite_or_blank(value: Any) -> Any:
    if value is None:
        return ""
    try:
        f = float(value)
        if not np.isfinite(f):
            return ""
        return f
    except Exception:
        return value


def annotation_key(row: Dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("scene", "")).strip(),
        str(row.get("point_name", "")).strip(),
        str(row.get("image_name", "")).strip(),
        f"{safe_float(row.get('manual_x', row.get('u_px')), 0.0):.3f}",
        f"{safe_float(row.get('manual_y', row.get('v_px')), 0.0):.3f}",
    )


def formal_key(row: Dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("scene", "")).strip(),
        str(row.get("point_name", "")).strip(),
        str(row.get("image_name", "")).strip(),
        f"{safe_float(row.get('u_px'), 0.0):.3f}",
        f"{safe_float(row.get('v_px'), 0.0):.3f}",
    )


def unique_index(rows: List[Dict[str, str]], key_fn) -> tuple[Dict[tuple[str, str, str, str, str], Dict[str, str]], int]:
    counts: Counter = Counter(key_fn(row) for row in rows)
    dup = sum(1 for _key, count in counts.items() if count > 1)
    return {key_fn(row): row for row in rows}, dup


def load_release_file(zf: zipfile.ZipFile, release_root: Path, name: str, expected_sha: str) -> tuple[Path, str]:
    path = release_root / name
    if not path.exists():
        raise FileNotFoundError(path)
    actual = file_sha256(path)
    if actual.lower() != expected_sha.lower():
        raise RuntimeError(f"Release SHA mismatch for {path}: {actual} != {expected_sha}")
    return path, actual


def load_packet_for_mapping(packet_dir: Path, row: Dict[str, str], manifest: Dict[str, Any]) -> Dict[str, np.ndarray]:
    packet_name = Path(row["packet_path"]).name
    packet_path = packet_dir / packet_name
    if not packet_path.exists():
        raise FileNotFoundError(packet_path)
    actual_sha = file_sha256(packet_path)
    if actual_sha.lower() != row["packet_sha256"].lower():
        raise RuntimeError(f"Packet SHA mismatch for {packet_path}: {actual_sha} != {row['packet_sha256']}")
    entry = {
        "height": int(row["height"]),
        "width": int(row["width"]),
    }
    return validate_metric_packet_npz(
        packet_path,
        entry,
        numerical_support_floor=float(manifest["numerical_support_floor"]),
        variance_clamp_tolerance=float(manifest["variance_clamp_tolerance"]),
        variance_validation_policy=str(manifest["variance_validation_policy"]),
        variance_validation_abs_floor=float(manifest["variance_validation_abs_floor"]),
        variance_validation_ulp_factor=float(manifest["variance_validation_ulp_factor"]),
        variance_validation_dtype=str(manifest["variance_validation_dtype"]),
        variance_validation_rtol=float(manifest["variance_validation_rtol"]),
        variance_nonnegativity_policy=str(manifest["variance_nonnegativity_policy"]),
        variance_negative_handling=str(manifest["variance_negative_handling"]),
        variance_raw_packet_modified=bool(manifest["variance_raw_packet_modified"]),
    )


def sample_patch_values(packet: Dict[str, np.ndarray], depth_u: float, depth_v: float, patch_size: int) -> Dict[str, Any]:
    alpha = np.asarray(packet["accumulated_alpha"], dtype=np.float64)
    h, w = alpha.shape
    cx = int(round(depth_u))
    cy = int(round(depth_v))
    half = patch_size // 2
    x0 = max(0, cx - half)
    x1 = min(w, cx + half + 1)
    y0 = max(0, cy - half)
    y1 = min(h, cy + half + 1)
    if cx < 0 or cy < 0 or cx >= w or cy >= h:
        return {"valid": False, "failure_reason": "pixel_outside_packet"}

    metric_valid = np.asarray(packet["metric_depth_valid_mask"]).astype(bool)[y0:y1, x0:x1]
    expected = np.asarray(packet["alpha_normalized_expected_camera_z"], dtype=np.float64)[y0:y1, x0:x1]
    harmonic = np.asarray(packet["harmonic_camera_z"], dtype=np.float64)[y0:y1, x0:x1]
    hsum = np.asarray(packet["weighted_inverse_camera_z_sum"], dtype=np.float64)[y0:y1, x0:x1]
    expected_inverse = np.asarray(packet["alpha_normalized_expected_inverse_camera_z"], dtype=np.float64)[y0:y1, x0:x1]

    def patch_median(arr: np.ndarray, mask: np.ndarray) -> float:
        values = arr[mask & np.isfinite(arr)]
        return float(np.median(values)) if values.size else math.nan

    old = np.full_like(hsum, np.nan, dtype=np.float64)
    ok_old = np.isfinite(hsum) & (hsum > 0)
    old[ok_old] = 1.0 / hsum[ok_old]

    return {
        "valid": True,
        "patch_x0": x0,
        "patch_x1": x1,
        "patch_y0": y0,
        "patch_y1": y1,
        "patch_valid_pixels": int(np.count_nonzero(metric_valid)),
        "patch_total_pixels": int(metric_valid.size),
        "expected_camera_z": patch_median(expected, metric_valid),
        "harmonic_camera_z": patch_median(harmonic, metric_valid),
        "expected_inverse_camera_z": patch_median(expected_inverse, metric_valid),
        "historical_invalid_pseudo_depth": patch_median(old, metric_valid),
        "A_center": float(packet["accumulated_alpha"][cy, cx]) if 0 <= cy < h and 0 <= cx < w else math.nan,
        "M1_center": float(packet["weighted_camera_z_sum"][cy, cx]) if 0 <= cy < h and 0 <= cx < w else math.nan,
        "M2_center": float(packet["weighted_camera_z_second_moment"][cy, cx]) if 0 <= cy < h and 0 <= cx < w else math.nan,
        "H_center": float(packet["weighted_inverse_camera_z_sum"][cy, cx]) if 0 <= cy < h and 0 <= cx < w else math.nan,
    }


def camera_center(image: Any) -> np.ndarray:
    rotation = qvec2rotmat(image.qvec)
    return -rotation.T @ image.tvec


def camera_ray_model(camera: Any, image: Any, u: float, v: float) -> tuple[np.ndarray, float, float]:
    x_norm, y_norm = pixel_to_normalized(camera, u, v)
    ray_cam = np.asarray([x_norm, y_norm, 1.0], dtype=np.float64)
    ray_cam_unit = ray_cam / np.linalg.norm(ray_cam)
    rotation = qvec2rotmat(image.qvec)
    ray_model = rotation.T @ ray_cam_unit
    ray_model = ray_model / np.linalg.norm(ray_model)
    off_axis = math.degrees(math.atan(math.sqrt(x_norm * x_norm + y_norm * y_norm)))
    return ray_model, off_axis, float(np.linalg.norm(ray_cam))


def aggregate_points(points: List[np.ndarray]) -> tuple[np.ndarray, Dict[str, float]]:
    stack = np.vstack(points)
    agg = np.median(stack, axis=0)
    dist = np.linalg.norm(stack - agg.reshape(1, 3), axis=1)
    return agg, {
        "scatter_median_m": float(np.median(dist)),
        "scatter_mad_m": float(np.median(np.abs(dist - np.median(dist)))),
        "scatter_p90_m": float(np.percentile(dist, 90)),
        "scatter_max_m": float(np.max(dist)),
        "scatter_mean_m": float(np.mean(dist)),
    }


def fit_semantic_sim3(
    semantic: str,
    aggregates: Dict[str, np.ndarray],
    target_points: Dict[str, np.ndarray],
    controls: Sequence[str],
    checkpoints: Sequence[str],
) -> Dict[str, Any]:
    control_names = sorted(name for name in controls if name in aggregates and name in target_points)
    checkpoint_names = sorted(name for name in checkpoints if name in aggregates and name in target_points)
    source = np.vstack([aggregates[name] for name in control_names])
    target = np.vstack([target_points[name] for name in control_names])
    scale, rotation, translation = fit_similarity_umeyama(source, target, estimate_scale=True)
    centered = source - source.mean(axis=0)
    cov = (centered.T @ centered) / max(1, centered.shape[0])
    singular = np.linalg.svd(cov, compute_uv=False)
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else math.inf
    extent = np.ptp(source, axis=0)
    height_range = float(np.ptp(source[:, 2])) if source.shape[0] else math.nan

    residual_rows: List[Dict[str, Any]] = []
    by_role: Dict[str, List[np.ndarray]] = {"control": [], "checkpoint": [], "all": []}
    for role, names in [("control", control_names), ("checkpoint", checkpoint_names)]:
        for name in names:
            pred = apply_similarity(aggregates[name].reshape(1, 3), scale, rotation, translation)[0]
            residual = pred - target_points[name]
            by_role[role].append(residual)
            by_role["all"].append(residual)
            residual_rows.append(
                {
                    "semantic": semantic,
                    "point_name": name,
                    "role": role,
                    "model_x": aggregates[name][0],
                    "model_y": aggregates[name][1],
                    "model_z": aggregates[name][2],
                    "predicted_x": pred[0],
                    "predicted_y": pred[1],
                    "predicted_z": pred[2],
                    "target_x": target_points[name][0],
                    "target_y": target_points[name][1],
                    "target_z": target_points[name][2],
                    "residual_x_m": residual[0],
                    "residual_y_m": residual[1],
                    "residual_z_m": residual[2],
                    "error_h_m": float(np.linalg.norm(residual[:2])),
                    "error_z_m": float(abs(residual[2])),
                    "error_3d_m": float(np.linalg.norm(residual)),
                }
            )
    residual_arrays = {
        role: np.vstack(values) if values else np.empty((0, 3), dtype=np.float64)
        for role, values in by_role.items()
    }
    control_rmse = residual_stats(residual_arrays["control"])
    return {
        "semantic": semantic,
        "control_points": control_names,
        "checkpoint_points": checkpoint_names,
        "transform": {
            "scale": scale,
            "rotation": rotation.tolist(),
            "translation": translation.tolist(),
            "definition": "target_xyz = scale * rotation @ model_xyz + translation",
        },
        "conditioning": {
            "centered_control_source_covariance_singular_values": singular.tolist(),
            "smallest_singular_value": float(singular[-1]),
            "largest_singular_value": float(singular[0]),
            "condition_number": condition,
            "control_spatial_extent_x_m": float(extent[0]),
            "control_spatial_extent_y_m": float(extent[1]),
            "control_spatial_extent_z_m": float(extent[2]),
            "control_height_range_m": height_range,
            "fitted_scale": scale,
            "rotation_determinant": float(np.linalg.det(rotation)),
            "control_rmse_3d_m": control_rmse["rmse_3d_m"],
            "control_near_coplanar": bool(condition > 1e6 or singular[-1] < 1e-9),
        },
        "residual_stats": {role: residual_stats(arr) for role, arr in residual_arrays.items()},
        "residual_rows": residual_rows,
    }


def rank(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=np.float64)
    i = 0
    while i < len(arr):
        j = i + 1
        while j < len(arr) and arr[order[j]] == arr[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> tuple[float, int]:
    pairs = [(float(a), float(b)) for a, b in zip(x, y) if np.isfinite(a) and np.isfinite(b)]
    if len(pairs) < 3:
        return math.nan, len(pairs)
    rx = rank([p[0] for p in pairs])
    ry = rank([p[1] for p in pairs])
    if np.std(rx) == 0 or np.std(ry) == 0:
        return math.nan, len(pairs)
    return float(np.corrcoef(rx, ry)[0, 1]), len(pairs)


def summarize_stats(values: Sequence[float]) -> Dict[str, Any]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


def markdown_table(rows: List[Dict[str, Any]], columns: Sequence[str]) -> str:
    lines = ["|" + "|".join(columns) + "|", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        lines.append("|" + "|".join(str(row.get(col, "")) for col in columns) + "|")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="No-GPU 3K depth semantic diagnostic.")
    parser.add_argument("--review_zip", required=True)
    parser.add_argument("--packet_dir", required=True)
    parser.add_argument("--release_dir", default=r"E:\datasets\M3M-GCP\scenes\gcp_manual_annotations")
    parser.add_argument(
        "--colmap_model",
        default=r"E:\M3M-GCP-3DGS\outputs\sibr_models_3scenes_20260624\sources\gcp_3000_20260602\sparse\0",
    )
    parser.add_argument(
        "--old_summary_csv",
        default=r"E:\M3M-GCP-3DGS\outputs\gcp_diagnostics_three_fixed_20260624\three_scene_gaussian_vs_colmap_summary.csv",
    )
    parser.add_argument(
        "--old_report_md",
        default=r"E:\M3M-GCP-3DGS\outputs\gcp_diagnostics_three_fixed_20260624\THREE_FIXED_SCENE_GCP_DIAGNOSTIC_REPORT.md",
    )
    parser.add_argument("--out_root", default="")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_root) if args.out_root else REPO_ROOT / "outputs" / f"gcp_3k_depth_semantics_diagnostic_20260628_{timestamp}"
    if out_root.exists():
        raise SystemExit(f"Output root already exists: {out_root}")
    out_root.mkdir(parents=True)
    input_dir = out_root / "inputs"
    input_dir.mkdir()

    review_zip = Path(args.review_zip)
    packet_dir = Path(args.packet_dir)
    release_root = Path(args.release_dir)
    colmap_model = Path(args.colmap_model)
    if not review_zip.exists():
        raise FileNotFoundError(review_zip)
    if not packet_dir.exists():
        raise FileNotFoundError(packet_dir)
    if not colmap_model.exists():
        raise FileNotFoundError(colmap_model)

    with zipfile.ZipFile(review_zip) as zf:
        formal_obs = read_csv_zip(zf, f"{FORMAL_EVAL_DIR}/method_gcp_observation_points.csv")
        formal_agg = read_csv_zip(zf, f"{FORMAL_EVAL_DIR}/method_gcp_aggregated_points.csv")
        formal_ck = read_csv_zip(zf, f"{FORMAL_EVAL_DIR}/method_gcp_checkpoint_residuals.csv")
        formal_ctl = read_csv_zip(zf, f"{FORMAL_EVAL_DIR}/method_gcp_sim3_control_residuals.csv")
        formal_scatter = read_csv_zip(zf, f"{FORMAL_EVAL_DIR}/method_gcp_multiview_scatter.csv")
        formal_eval_summary = json.loads(zf.read(f"{FORMAL_EVAL_DIR}/method_gcp_eval_summary.json"))
        formal_eval_manifest = json.loads(zf.read(f"{FORMAL_EVAL_DIR}/evaluator_manifest.json"))
        packet_mapping = read_csv_zip(zf, f"{PACKET_MANIFEST_DIR}/metric_depth_mapping.csv")
        packet_manifest = json.loads(zf.read(f"{PACKET_MANIFEST_DIR}/metric_depth_manifest.json"))
        release_config = json.loads(zf.read(f"{RELEASE_DIR}/relocated_release_config.json"))
        review_hashes = {
            "review_zip": str(review_zip),
            "review_zip_sha256": file_sha256(review_zip),
            "formal_observation_csv_sha256": sha256_bytes(zf.read(f"{FORMAL_EVAL_DIR}/method_gcp_observation_points.csv")),
            "packet_mapping_csv_sha256": sha256_bytes(zf.read(f"{PACKET_MANIFEST_DIR}/metric_depth_mapping.csv")),
        }

    release_files = {entry["path"]: entry["sha256"] for entry in release_config["files"]}
    annotation_path, annotation_sha = load_release_file(
        None, release_root, f"{SCENE}_gcp_annotations_final_good_nadir_v1.csv", release_files[f"{SCENE}_gcp_annotations_final_good_nadir_v1.csv"]
    )
    gcp_path, gcp_sha = load_release_file(None, release_root, release_config["gcp_csv"], release_files[release_config["gcp_csv"]])
    split_path, split_sha = load_release_file(None, release_root, release_config["split_csv"], release_files[release_config["split_csv"]])
    metadata_path, metadata_sha = load_release_file(None, release_root, release_config["scene_metadata_csv"], release_files[release_config["scene_metadata_csv"]])

    annotations = read_csv_path(annotation_path)
    gcp_rows = read_csv_path(gcp_path)
    split_rows = [r for r in read_csv_path(split_path) if r.get("scene") == SCENE]
    metadata_rows = read_csv_path(metadata_path)
    roles = {r["point_name"]: r["role"].strip().lower() for r in split_rows}
    controls = sorted([name for name, role in roles.items() if role == "control"])
    checkpoints = sorted([name for name, role in roles.items() if role == "checkpoint"])
    target_points = {
        r["point_name"]: np.asarray([float(r[field]) for field in DEFAULT_TARGET_FIELDS], dtype=np.float64)
        for r in gcp_rows
        if r.get("point_name")
    }

    annotation_index, annotation_dup = unique_index(annotations, annotation_key)
    formal_index, formal_dup = unique_index(formal_obs, formal_key)
    if annotation_dup:
        raise SystemExit(f"Duplicate frozen annotation keys: {annotation_dup}")
    packet_by_image = {r["image_name"]: r for r in packet_mapping}

    cameras, images, _points3d = read_model(colmap_model)
    images_by_name = {img.name: img for img in images.values()}

    packet_cache: Dict[str, Dict[str, np.ndarray]] = {}
    packet_sha_failures: List[str] = []
    for row in packet_mapping:
        try:
            packet_cache[row["image_name"]] = load_packet_for_mapping(packet_dir, row, packet_manifest)
        except Exception as exc:
            packet_sha_failures.append(f"{row.get('image_name')}: {exc}")
    if packet_sha_failures:
        raise SystemExit("Packet verification failed:\n" + "\n".join(packet_sha_failures[:20]))

    patch_size = int(formal_eval_manifest["patch_size"])
    depth_scale_x = float(formal_eval_manifest["depth_pixel_scale_x"])
    depth_scale_y = float(formal_eval_manifest["depth_pixel_scale_y"])
    min_patch_valid_ratio = float(formal_eval_manifest["min_patch_valid_ratio"])
    pixel_convention = str(formal_eval_manifest["pixel_coordinate_convention"])

    sem_points_by_gcp: Dict[str, Dict[str, List[np.ndarray]]] = {
        "expected": defaultdict(list),
        "harmonic": defaultdict(list),
        "old": defaultdict(list),
    }
    sem_obs_points: Dict[str, Dict[tuple[str, str, str, str, str], np.ndarray]] = {s: {} for s in sem_points_by_gcp}
    obs_rows: List[Dict[str, Any]] = []
    unmatched_annotations: List[Dict[str, Any]] = []
    packet_missing = 0
    semantic_invalid = Counter()

    for ann in annotations:
        key = annotation_key(ann)
        point = ann["point_name"]
        image_name = ann["image_name"]
        role = roles.get(point, "")
        u = safe_float(ann["manual_x"])
        v = safe_float(ann["manual_y"])
        row: Dict[str, Any] = {
            "scene": ann["scene"],
            "point_name": point,
            "role": role,
            "image_name": image_name,
            "manual_x": u,
            "manual_y": v,
            "annotation_key": "|".join(key),
        }
        formal = formal_index.get(key)
        row["formal_output_matched"] = int(formal is not None)
        if formal is None:
            unmatched_annotations.append(row.copy())
        else:
            row["formal_valid"] = formal.get("valid", "")
            row["formal_failure_reason"] = formal.get("failure_reason", "")
            row["formal_expected_model_x"] = formal.get("model_x", "")
            row["formal_expected_model_y"] = formal.get("model_y", "")
            row["formal_expected_model_z"] = formal.get("model_z", "")

        image = images_by_name.get(image_name)
        if image is None:
            row["colmap_present"] = 0
            row["diagnostic_flags"] = "missing_colmap_image"
            obs_rows.append(row)
            semantic_invalid["missing_colmap_image"] += 1
            continue
        camera = cameras[image.camera_id]
        row["colmap_present"] = 1
        row["image_id"] = image.id
        row["camera_id"] = image.camera_id
        row["camera_width"] = camera.width
        row["camera_height"] = camera.height
        row["pixel_in_bounds"] = int(0 <= u < camera.width and 0 <= v < camera.height)
        row["camera_center_x"], row["camera_center_y"], row["camera_center_z"] = camera_center(image).tolist()

        mapping = packet_by_image.get(image_name)
        if mapping is None:
            packet_missing += 1
            row["packet_present"] = 0
            row["diagnostic_flags"] = "missing_packet_mapping"
            obs_rows.append(row)
            continue
        row["packet_present"] = 1
        row["packet_sha256"] = mapping["packet_sha256"]
        packet = packet_cache[image_name]
        depth_u = u * float(mapping["width"]) / float(camera.width)
        depth_v = v * float(mapping["height"]) / float(camera.height)
        row["depth_u_px"] = depth_u
        row["depth_v_px"] = depth_v
        row["depth_width"] = mapping["width"]
        row["depth_height"] = mapping["height"]
        row["depth_pixel_scale_x"] = float(mapping["width"]) / float(camera.width)
        row["depth_pixel_scale_y"] = float(mapping["height"]) / float(camera.height)

        patch = sample_patch_values(packet, depth_u, depth_v, patch_size)
        if not patch["valid"]:
            row["diagnostic_flags"] = patch.get("failure_reason", "invalid_patch")
            obs_rows.append(row)
            continue
        row.update({k: v for k, v in patch.items() if k != "valid"})
        patch_diag = metric_packet_patch_diagnostics(packet, depth_u, depth_v, patch_size)
        row.update(patch_diag)
        row["same_patch_size"] = patch_size
        row["formal_min_patch_valid_ratio"] = min_patch_valid_ratio
        row["pixel_coordinate_convention"] = pixel_convention

        ray_model, off_axis_deg, ray_norm = camera_ray_model(camera, image, u, v)
        row["off_axis_angle_deg"] = off_axis_deg
        row["ray_norm"] = ray_norm
        row["ray_model_x"], row["ray_model_y"], row["ray_model_z"] = ray_model.tolist()

        semantic_depths = {
            "expected": row.get("expected_camera_z", math.nan),
            "harmonic": row.get("harmonic_camera_z", math.nan),
            "old": row.get("historical_invalid_pseudo_depth", math.nan),
        }
        for sem, depth in semantic_depths.items():
            if not np.isfinite(depth) or depth <= 0:
                row[f"{sem}_valid"] = 0
                semantic_invalid[f"{sem}_invalid"] += 1
                continue
            xyz = backproject_world(camera, image, u, v, float(depth))
            row[f"{sem}_valid"] = 1
            row[f"{sem}_model_x"] = xyz[0]
            row[f"{sem}_model_y"] = xyz[1]
            row[f"{sem}_model_z"] = xyz[2]
            sem_points_by_gcp[sem][point].append(xyz)
            sem_obs_points[sem][key] = xyz
        if row.get("expected_valid") and row.get("harmonic_valid"):
            row["expected_minus_harmonic_depth_m"] = float(row["expected_camera_z"] - row["harmonic_camera_z"])
        if row.get("expected_valid") and row.get("old_valid"):
            row["expected_minus_old_depth_m"] = float(row["expected_camera_z"] - row["historical_invalid_pseudo_depth"])
        obs_rows.append(row)

    # Aggregate and fit semantic-specific Sim(3).
    semantic_aggregates: Dict[str, Dict[str, np.ndarray]] = {}
    per_gcp_rows: List[Dict[str, Any]] = []
    scatter_rows: List[Dict[str, Any]] = []
    for sem, by_gcp in sem_points_by_gcp.items():
        semantic_aggregates[sem] = {}
        for point, points in by_gcp.items():
            if not points:
                continue
            aggregate, scatter = aggregate_points(points)
            semantic_aggregates[sem][point] = aggregate
            stack = np.vstack(points)
            depth_values = []
            alpha_values = []
            off_axis_values = []
            for row in obs_rows:
                if row.get("point_name") == point and int(row.get(f"{sem}_valid", 0)):
                    depth_values.append(safe_float(row.get(f"{sem if sem != 'old' else 'historical_invalid_pseudo'}_depth", row.get("historical_invalid_pseudo_depth"))))
                    alpha_values.append(safe_float(row.get("accumulated_alpha_patch_median")))
                    off_axis_values.append(safe_float(row.get("off_axis_angle_deg")))
            loo = []
            for idx in range(len(points)):
                if len(points) <= 1:
                    loo.append(0.0)
                else:
                    other = [p for j, p in enumerate(points) if j != idx]
                    other_agg, _ = aggregate_points(other)
                    loo.append(float(np.linalg.norm(other_agg - aggregate)))
            dmat = []
            for i in range(len(points)):
                for j in range(i + 1, len(points)):
                    dmat.append(float(np.linalg.norm(points[i] - points[j])))
            out = {
                "scene": SCENE,
                "semantic": sem,
                "point_name": point,
                "role": roles.get(point, ""),
                "valid_observation_count": len(points),
                "aggregate_x": aggregate[0],
                "aggregate_y": aggregate[1],
                "aggregate_z": aggregate[2],
                **scatter,
                "leave_one_view_out_shift_max_m": max(loo) if loo else math.nan,
                "pairwise_distance_median_m": float(np.median(dmat)) if dmat else 0.0,
                "pairwise_distance_p90_m": float(np.percentile(dmat, 90)) if dmat else 0.0,
                "pairwise_distance_max_m": max(dmat) if dmat else 0.0,
                "depth_range_m": float(np.nanmax(depth_values) - np.nanmin(depth_values)) if depth_values else math.nan,
                "alpha_range": float(np.nanmax(alpha_values) - np.nanmin(alpha_values)) if alpha_values else math.nan,
                "off_axis_range_deg": float(np.nanmax(off_axis_values) - np.nanmin(off_axis_values)) if off_axis_values else math.nan,
            }
            per_gcp_rows.append(out)
            for p in points:
                scatter_rows.append(
                    {
                        "scene": SCENE,
                        "semantic": sem,
                        "point_name": point,
                        "distance_to_aggregate_m": float(np.linalg.norm(p - aggregate)),
                    }
                )

    sim3 = {
        sem: fit_semantic_sim3(sem, semantic_aggregates[sem], target_points, controls, checkpoints)
        for sem in ["expected", "harmonic", "old"]
    }
    expected_transform = sim3["expected"]["transform"]
    expected_scale = float(expected_transform["scale"])
    expected_rot = np.asarray(expected_transform["rotation"], dtype=np.float64)
    expected_trans = np.asarray(expected_transform["translation"], dtype=np.float64)

    # Attach residuals, aggregate distances, and fixed-transform displacements.
    residual_by_sem_point = {
        sem: {r["point_name"]: r for r in payload["residual_rows"]}
        for sem, payload in sim3.items()
    }
    fixed_rows: List[Dict[str, Any]] = []
    for row in obs_rows:
        key = annotation_key(row)
        point = row["point_name"]
        target = target_points.get(point)
        if target is None:
            continue
        ray_model = np.asarray([safe_float(row.get("ray_model_x")), safe_float(row.get("ray_model_y")), safe_float(row.get("ray_model_z"))])
        if np.linalg.norm(ray_model) > 0:
            ray_survey_expected = expected_rot @ ray_model
            ray_survey_expected = ray_survey_expected / np.linalg.norm(ray_survey_expected)
        else:
            ray_survey_expected = np.asarray([math.nan, math.nan, math.nan])
        for sem in ["expected", "harmonic", "old"]:
            point_xyz = sem_obs_points[sem].get(key)
            if point_xyz is None:
                continue
            agg = semantic_aggregates[sem].get(point)
            if agg is not None:
                row[f"{sem}_distance_to_aggregate_m"] = float(np.linalg.norm(point_xyz - agg))
            sem_payload = sim3[sem]["transform"]
            sem_scale = float(sem_payload["scale"])
            sem_rot = np.asarray(sem_payload["rotation"], dtype=np.float64)
            sem_trans = np.asarray(sem_payload["translation"], dtype=np.float64)
            pred = apply_similarity(point_xyz.reshape(1, 3), sem_scale, sem_rot, sem_trans)[0]
            residual = pred - target
            row[f"{sem}_residual_h_m"] = float(np.linalg.norm(residual[:2]))
            row[f"{sem}_residual_z_m"] = float(abs(residual[2]))
            row[f"{sem}_residual_3d_m"] = float(np.linalg.norm(residual))
            ray_survey_sem = sem_rot @ ray_model
            ray_survey_sem = ray_survey_sem / np.linalg.norm(ray_survey_sem)
            along = float(np.dot(residual, ray_survey_sem))
            cross = residual - along * ray_survey_sem
            row[f"{sem}_survey_signed_along_ray_residual_m"] = along
            row[f"{sem}_survey_abs_along_ray_residual_m"] = abs(along)
            row[f"{sem}_survey_cross_ray_residual_m"] = float(np.linalg.norm(cross))
        exp_xyz = sem_obs_points["expected"].get(key)
        if exp_xyz is None:
            continue
        exp_pred_fixed = apply_similarity(exp_xyz.reshape(1, 3), expected_scale, expected_rot, expected_trans)[0]
        for sem in ["harmonic", "old"]:
            xyz = sem_obs_points[sem].get(key)
            if xyz is None:
                continue
            semantic_delta_model = xyz - exp_xyz
            semantic_along_model = float(np.dot(semantic_delta_model, ray_model))
            semantic_cross_model = semantic_delta_model - semantic_along_model * ray_model
            pred_fixed = apply_similarity(xyz.reshape(1, 3), expected_scale, expected_rot, expected_trans)[0]
            delta_survey = pred_fixed - exp_pred_fixed
            fixed_along = float(np.dot(delta_survey, ray_survey_expected))
            fixed_cross = delta_survey - fixed_along * ray_survey_expected
            fixed_rows.append(
                {
                    "scene": SCENE,
                    "point_name": point,
                    "role": roles.get(point, ""),
                    "image_name": row["image_name"],
                    "semantic": sem,
                    "depth_difference_vs_expected_m": safe_float(row.get(f"{sem}_camera_z" if sem == "harmonic" else "historical_invalid_pseudo_depth")) - safe_float(row.get("expected_camera_z")),
                    "semantic_displacement_camera_ray_along_m": semantic_along_model,
                    "semantic_displacement_camera_ray_cross_m": float(np.linalg.norm(semantic_cross_model)),
                    "fixed_transform_survey_horizontal_m": float(np.linalg.norm(delta_survey[:2])),
                    "fixed_transform_survey_vertical_m": float(abs(delta_survey[2])),
                    "fixed_transform_survey_3d_m": float(np.linalg.norm(delta_survey)),
                    "fixed_transform_survey_along_ray_m": fixed_along,
                    "fixed_transform_survey_cross_ray_m": float(np.linalg.norm(fixed_cross)),
                }
            )

    # Representative points.
    checkpoint_residuals = sorted(
        [(safe_float(r["error_3d_m"]), r["point_name"]) for r in formal_ck],
        key=lambda x: (x[0], x[1]),
    )
    low_error_checkpoint = checkpoint_residuals[0][1] if checkpoint_residuals else ""
    formal_agg_by_point = {r["point_name"]: r for r in formal_agg}
    control_scatter = sorted(
        [
            (safe_float(formal_agg_by_point.get(name, {}).get("scatter_max_m")), name)
            for name in controls
            if name in formal_agg_by_point
        ],
        key=lambda x: (x[0], x[1]),
    )
    stable_control = control_scatter[0][1] if control_scatter else ""
    high_scatter_control = sorted(control_scatter, key=lambda x: (-x[0], x[1]))[0][1] if control_scatter else ""
    focus_points = []
    for name in ["G11", "G16", low_error_checkpoint, stable_control, high_scatter_control]:
        if name and name not in focus_points:
            focus_points.append(name)

    # Protocol provenance reconciliation.
    old_summary_rows = read_csv_path(Path(args.old_summary_csv)) if Path(args.old_summary_csv).exists() else []
    old_3k_gaussian = [
        r for r in old_summary_rows
        if r.get("scene") == SCENE and r.get("evaluator") == "gaussian_depth_only_p1" and r.get("role") == "checkpoint"
    ]
    current_expected_stats = sim3["expected"]["residual_stats"]["checkpoint"]
    current_old_stats = sim3["old"]["residual_stats"]["checkpoint"]
    provenance_rows = [
        {
            "comparison": "archived_old_result",
            "status": "available" if old_3k_gaussian else "unresolved",
            "checkpoint_rmse_3d_m": old_3k_gaussian[0].get("rmse_3d_m", "") if old_3k_gaussian else "",
            "depth_semantics": "archived_depth_only_p1_semantics_not_fully_locked_in_current_release",
            "pointset_split": "old_same-split reported, exact release-lock provenance partially unresolved",
            "patch_aggregation": "unresolved from archived summary",
            "notes": str(Path(args.old_summary_csv)),
        },
        {
            "comparison": "old_semantic_under_current_protocol",
            "status": "computed",
            "checkpoint_rmse_3d_m": current_old_stats["rmse_3d_m"],
            "depth_semantics": "historical_invalid_pseudo_depth_1_over_H",
            "pointset_split": "current frozen v1.1",
            "patch_aggregation": f"current patch_size={patch_size}",
            "notes": "diagnostic only; not a formal metric",
        },
        {
            "comparison": "expected_z_under_current_protocol",
            "status": "computed",
            "checkpoint_rmse_3d_m": current_expected_stats["rmse_3d_m"],
            "depth_semantics": "alpha_normalized_expected_camera_z_M1_over_A",
            "pointset_split": "current frozen v1.1",
            "patch_aggregation": f"current patch_size={patch_size}",
            "notes": "formal P1 release-mode result",
        },
        {
            "comparison": "expected_z_under_old_protocol",
            "status": "unresolved",
            "checkpoint_rmse_3d_m": "",
            "depth_semantics": "expected_z",
            "pointset_split": "old protocol inputs not fully reconstructable without old packet/render provenance",
            "patch_aggregation": "unresolved",
            "notes": "not computed to avoid attributing old-vs-new delta solely to depth semantics",
        },
    ]

    # Correlations.
    expected_residual_by_key = [
        (row, safe_float(row.get("expected_residual_3d_m")))
        for row in obs_rows
        if np.isfinite(safe_float(row.get("expected_residual_3d_m")))
    ]
    correlation_specs = {
        "residual_vs_alpha_median": "accumulated_alpha_patch_median",
        "residual_vs_variance_median": "variance_diagnostic_median",
        "residual_vs_off_axis": "off_axis_angle_deg",
        "residual_vs_expected_harmonic_depth_delta": "expected_minus_harmonic_depth_m",
        "residual_vs_expected_old_depth_delta": "expected_minus_old_depth_m",
    }
    correlation_rows: List[Dict[str, Any]] = []
    for label, col in correlation_specs.items():
        rho, n = spearman([safe_float(row.get(col)) for row, _res in expected_residual_by_key], [res for _row, res in expected_residual_by_key])
        correlation_rows.append({"level": "observation", "relationship": label, "spearman_rho": rho, "n": n, "interpretation": "diagnostic_only_no_significance_claim"})

    # Causal attribution.
    delta_depth_old_current = safe_float(current_expected_stats["rmse_3d_m"]) - safe_float(current_old_stats["rmse_3d_m"])
    g11_error = safe_float(residual_by_sem_point["expected"].get("G11", {}).get("error_3d_m"))
    g16_error = safe_float(residual_by_sem_point["expected"].get("G16", {}).get("error_3d_m"))
    causal_rows = [
        {
            "candidate_cause": "depth semantics change",
            "status": "confirmed" if abs(delta_depth_old_current) > 0.1 else "not_supported",
            "supporting_metrics": f"current expected checkpoint RMSE-3D={current_expected_stats['rmse_3d_m']}; current old-pseudo checkpoint RMSE-3D={current_old_stats['rmse_3d_m']}",
            "contradicting_evidence": "archived old protocol provenance not fully identical",
            "affected_points_views": "all current frozen rows",
            "confidence_rationale": "Same rows/split/aggregation isolate semantic effect, but old 0.252 comparison also has protocol uncertainty.",
        },
        {
            "candidate_cause": "split/pointset change",
            "status": "unresolved",
            "supporting_metrics": "archived old report used same-split wording, but exact release-lock file hashes are not available in the old summary",
            "contradicting_evidence": "current 3K point set still has 5 controls and 4 checkpoints",
            "affected_points_views": "protocol-level",
            "confidence_rationale": "Cannot claim zero contribution without old frozen release artifacts.",
        },
        {
            "candidate_cause": "patch/aggregation protocol change",
            "status": "unresolved",
            "supporting_metrics": f"current manifest patch_size={patch_size}; old summary lacks full patch provenance",
            "contradicting_evidence": "current within-protocol old/expected comparison uses identical patch pixels",
            "affected_points_views": "protocol-level",
            "confidence_rationale": "Current diagnostic isolates it, but archived old settings remain incomplete.",
        },
        {
            "candidate_cause": "opacity-dependent old pseudo-depth compensation",
            "status": "likely" if safe_float(current_old_stats["rmse_3d_m"]) < safe_float(current_expected_stats["rmse_3d_m"]) else "not_supported",
            "supporting_metrics": f"old-pseudo current-protocol checkpoint RMSE-3D={current_old_stats['rmse_3d_m']} vs expected={current_expected_stats['rmse_3d_m']}",
            "contradicting_evidence": "z_old is semantically invalid and cannot be interpreted as metric depth",
            "affected_points_views": "checkpoint/control rows where expected-old depth delta is large",
            "confidence_rationale": "Diagnostic comparison can show numerical compensation but not physical correctness.",
        },
        {
            "candidate_cause": "multiview surface inconsistency",
            "status": "likely" if max(safe_float(r.get("scatter_max_m")) for r in formal_agg) > 1.0 else "not_supported",
            "supporting_metrics": "formal aggregate scatter and per-GCP pairwise distances reported in per_gcp table",
            "contradicting_evidence": "requires visual/rendered surface evidence for full confirmation",
            "affected_points_views": "G11/G16 and high-scatter control if applicable",
            "confidence_rationale": "Large scatter supports multi-view inconsistency; exact surface identity remains unresolved without renderer internals.",
        },
        {
            "candidate_cause": "off-axis horizontal amplification",
            "status": "likely" if any(abs(safe_float(r.get("expected_survey_signed_along_ray_residual_m"))) > 0.5 for r in obs_rows) else "not_supported",
            "supporting_metrics": "survey residual along-ray/cross-ray decomposition and off-axis correlations",
            "contradicting_evidence": "not all high-error rows need be high off-axis",
            "affected_points_views": "reported in per-observation table",
            "confidence_rationale": "Along-ray residual projected through oblique views can explain horizontal error growth when present.",
        },
        {
            "candidate_cause": "pixel/camera mapping error",
            "status": "not_supported",
            "supporting_metrics": "COLMAP image/camera rows matched; output row count preserved; depth/image scale locked from manifest",
            "contradicting_evidence": "no current hard-stop mismatch detected",
            "affected_points_views": "G11/G16 sanity report",
            "confidence_rationale": "Only sanity checks are run; no mapping mismatch has been found.",
        },
        {
            "candidate_cause": "Sim(3) control conditioning",
            "status": "likely" if sim3["expected"]["conditioning"]["condition_number"] > 1000 else "not_supported",
            "supporting_metrics": json.dumps(sim3["expected"]["conditioning"], ensure_ascii=False),
            "contradicting_evidence": "same frozen controls fit all semantic comparisons",
            "affected_points_views": "global transform",
            "confidence_rationale": "Conditioning is recorded; no split is changed.",
        },
        {
            "candidate_cause": "annotation error",
            "status": "not_supported",
            "supporting_metrics": "frozen annotations preserve all rows; old COLMAP triangulation sanity was about 0.028 m",
            "contradicting_evidence": "large expected-z errors are not by themselves annotation proof",
            "affected_points_views": "all",
            "confidence_rationale": "No reannotation evidence is introduced in this no-GPU diagnostic.",
        },
        {
            "candidate_cause": "unresolved renderer information",
            "status": "unresolved",
            "supporting_metrics": "metric packets expose A/M1/M2/H but not full per-layer Gaussian identities or surface provenance",
            "contradicting_evidence": "packet-level diagnostics still explain many numerical effects",
            "affected_points_views": "multi-layer/floater interpretation",
            "confidence_rationale": "Cannot confirm specific Gaussian layers or floaters without additional renderer outputs.",
        },
    ]

    # Write outputs.
    all_obs_fields = sorted({k for row in obs_rows for k in row.keys()})
    preferred_obs = [
        "scene", "point_name", "role", "image_name", "manual_x", "manual_y", "formal_output_matched",
        "formal_valid", "formal_failure_reason", "packet_present", "colmap_present", "image_id", "camera_id",
        "expected_camera_z", "harmonic_camera_z", "expected_inverse_camera_z", "historical_invalid_pseudo_depth",
        "expected_minus_harmonic_depth_m", "expected_minus_old_depth_m", "A_center", "M1_center", "M2_center", "H_center",
        "accumulated_alpha_patch_min", "accumulated_alpha_patch_p10", "accumulated_alpha_patch_median",
        "variance_diagnostic_median", "variance_diagnostic_p90", "variance_diagnostic_valid_ratio",
        "variance_raw_negative_count", "variance_nonnegativity_unresolved_count", "off_axis_angle_deg", "ray_norm",
        "expected_residual_h_m", "expected_residual_z_m", "expected_residual_3d_m",
        "harmonic_residual_h_m", "harmonic_residual_z_m", "harmonic_residual_3d_m",
        "old_residual_h_m", "old_residual_z_m", "old_residual_3d_m",
    ]
    obs_fields = preferred_obs + [f for f in all_obs_fields if f not in set(preferred_obs)]
    write_csv(out_root / "per_observation_semantic_comparison.csv", obs_rows, obs_fields)
    write_csv(out_root / "per_gcp_semantic_comparison.csv", per_gcp_rows, sorted({k for r in per_gcp_rows for k in r.keys()}))
    sim3_rows = []
    residual_rows = []
    for sem, payload in sim3.items():
        for role, stats in payload["residual_stats"].items():
            row = {"semantic": sem, "role": role, **stats, **payload["conditioning"]}
            sim3_rows.append(row)
        residual_rows.extend(payload["residual_rows"])
    write_csv(out_root / "semantic_specific_sim3_summary.csv", sim3_rows, sorted({k for r in sim3_rows for k in r.keys()}))
    write_csv(out_root / "semantic_specific_sim3_residuals.csv", residual_rows, sorted({k for r in residual_rows for k in r.keys()}))
    write_json(out_root / "semantic_specific_sim3_summary.json", {k: {kk: vv for kk, vv in v.items() if kk != "residual_rows"} for k, v in sim3.items()})
    write_csv(out_root / "fixed_transform_displacement.csv", fixed_rows, sorted({k for r in fixed_rows for k in r.keys()}))
    write_csv(out_root / "protocol_provenance_reconciliation.csv", provenance_rows, list(provenance_rows[0].keys()))
    write_json(out_root / "protocol_provenance_reconciliation.json", provenance_rows)
    write_csv(out_root / "correlation_error_decomposition.csv", correlation_rows, list(correlation_rows[0].keys()))
    write_csv(out_root / "causal_attribution_table.csv", causal_rows, list(causal_rows[0].keys()))

    # Markdown reports.
    (out_root / "correlation_error_decomposition.md").write_text(
        "# Correlation/Error Decomposition\n\n"
        "Spearman correlations are diagnostic only; no significance or universal claim is made.\n\n"
        + markdown_table(correlation_rows, ["level", "relationship", "spearman_rho", "n", "interpretation"]),
        encoding="utf-8",
    )
    (out_root / "causal_attribution_table.md").write_text(
        "# Causal Attribution Table\n\n"
        + markdown_table(
            causal_rows,
            ["candidate_cause", "status", "supporting_metrics", "contradicting_evidence", "affected_points_views", "confidence_rationale"],
        ),
        encoding="utf-8",
    )

    focus_dir = out_root / "focus_points"
    focus_dir.mkdir()
    for point in focus_points:
        point_rows = [r for r in obs_rows if r.get("point_name") == point]
        report = [
            f"# {point} Depth-Semantics Diagnostic",
            "",
            f"- Role: `{roles.get(point, '')}`",
            f"- Observation rows: `{len(point_rows)}`",
            "",
            "## Semantic residuals",
        ]
        for sem in ["expected", "harmonic", "old"]:
            rr = residual_by_sem_point.get(sem, {}).get(point)
            if rr:
                report.append(f"- {sem}: eH={float(rr['error_h_m']):.4f} m, eZ={float(rr['error_z_m']):.4f} m, e3D={float(rr['error_3d_m']):.4f} m")
        report.extend(["", "## Observations", ""])
        cols = ["image_name", "expected_camera_z", "harmonic_camera_z", "historical_invalid_pseudo_depth", "accumulated_alpha_patch_median", "variance_diagnostic_median", "off_axis_angle_deg", "expected_distance_to_aggregate_m", "expected_residual_3d_m"]
        report.append(markdown_table([{c: finite_or_blank(r.get(c)) for c in cols} for r in point_rows], cols))
        (focus_dir / f"{point}_report.md").write_text("\n".join(report), encoding="utf-8")

    sanity = {
        "scene": SCENE,
        "frozen_annotation_rows": len(annotations),
        "formal_output_rows": len(formal_obs),
        "formal_output_matched_count": sum(int(r.get("formal_output_matched", 0)) for r in obs_rows),
        "unmatched_annotation_rows": len(unmatched_annotations),
        "duplicate_annotation_key_count": annotation_dup,
        "duplicate_formal_key_count": formal_dup,
        "packet_missing_count": packet_missing,
        "semantic_invalid_counts": dict(semantic_invalid),
        "packet_count": len(packet_mapping),
        "colmap_image_count": len(images_by_name),
        "patch_size": patch_size,
        "min_patch_valid_ratio": min_patch_valid_ratio,
        "pixel_coordinate_convention": pixel_convention,
        "image_domain": formal_eval_manifest["image_domain"],
        "depth_pixel_scale_x": depth_scale_x,
        "depth_pixel_scale_y": depth_scale_y,
        "focus_points": focus_points,
        "release_hashes": {
            "annotation_csv": annotation_sha,
            "gcp_csv": gcp_sha,
            "split_csv": split_sha,
            "scene_metadata_csv": metadata_sha,
            **review_hashes,
        },
        "inputs": {
            "review_zip": str(review_zip),
            "packet_dir": str(packet_dir),
            "release_dir": str(release_root),
            "colmap_model": str(colmap_model),
            "old_summary_csv": args.old_summary_csv,
        },
    }
    write_json(out_root / "pixel_camera_sanity_report.json", sanity)
    (out_root / "pixel_camera_sanity_report.md").write_text(
        "# Pixel/Camera Sanity Report\n\n"
        f"- Frozen annotation rows: `{sanity['frozen_annotation_rows']}`\n"
        f"- Formal output matched count: `{sanity['formal_output_matched_count']}`\n"
        f"- Unmatched annotation rows: `{sanity['unmatched_annotation_rows']}`\n"
        f"- Duplicate annotation keys: `{sanity['duplicate_annotation_key_count']}`\n"
        f"- Packet missing count: `{sanity['packet_missing_count']}`\n"
        f"- COLMAP image count: `{sanity['colmap_image_count']}`\n"
        f"- Patch size: `{patch_size}`\n"
        f"- Pixel convention: `{pixel_convention}`\n"
        f"- Image domain: `{formal_eval_manifest['image_domain']}`\n",
        encoding="utf-8",
    )
    write_json(out_root / "run_input_hashes.json", sanity["release_hashes"])

    review_brief = [
        "# 3K Depth Semantics Diagnostic Review Brief",
        "",
        "This package is a no-GPU diagnostic. It does not rerender packets, train, mutate checkpoints, edit pointsets/splits, or select a new metric.",
        "",
        "## Key Results",
        "",
        f"- Frozen annotation rows: `{len(annotations)}`",
        f"- Formal output matched rows: `{sanity['formal_output_matched_count']}`",
        f"- Expected-z current checkpoint RMSE-3D: `{current_expected_stats['rmse_3d_m']}`",
        f"- Harmonic current checkpoint RMSE-3D: `{sim3['harmonic']['residual_stats']['checkpoint']['rmse_3d_m']}`",
        f"- Historical invalid old-pseudo current checkpoint RMSE-3D: `{current_old_stats['rmse_3d_m']}`",
        f"- Archived old checkpoint RMSE-3D: `{old_3k_gaussian[0].get('rmse_3d_m', 'unresolved') if old_3k_gaussian else 'unresolved'}`",
        "",
        "## Interpretation Boundary",
        "",
        "The old pseudo-depth remains an invalid historical diagnostic. Any improvement relative to expected-z is treated as numerical compensation, not as a valid depth metric.",
    ]
    (out_root / "REVIEW_BRIEF.md").write_text("\n".join(review_brief) + "\n", encoding="utf-8")

    commands = {
        "diagnostic_command": " ".join(sys.argv),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "git_branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO_ROOT, text=True).strip(),
        "git_status_porcelain": subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=REPO_ROOT, text=True),
    }
    write_json(out_root / "commands_and_status.json", commands)

    # Include source/provenance before packaging.
    code_dir = out_root / "code"
    code_dir.mkdir()
    script_path = Path(__file__).resolve()
    shutil.copy2(script_path, code_dir / script_path.name)
    (code_dir / "git_status_porcelain.txt").write_text(commands["git_status_porcelain"], encoding="utf-8")
    diff_text = subprocess.check_output(
        ["git", "diff", "--", "code/gcp/diagnose_3k_depth_semantics.py"],
        cwd=REPO_ROOT,
        text=True,
        errors="replace",
    )
    (code_dir / "diagnostic_script_diff.patch").write_text(diff_text, encoding="utf-8")

    package_manifest: List[Dict[str, Any]] = []
    for path in sorted(out_root.rglob("*")):
        if path.is_file():
            rel = path.relative_to(out_root)
            package_manifest.append({"path": rel.as_posix(), "sha256": file_sha256(path), "bytes": path.stat().st_size})
    write_json(out_root / "package_manifest.json", package_manifest)
    sha_lines = [f"{row['sha256']}  {row['path']}" for row in package_manifest]
    (out_root / "PACKAGE_CONTENT_SHA256SUMS.txt").write_text("\n".join(sha_lines) + "\n", encoding="utf-8")

    # Package selected outputs.
    package_dir = REPO_ROOT / "outputs" / "gpt_review_packages"
    package_dir.mkdir(parents=True, exist_ok=True)
    zip_base = package_dir / "GPT_GCP_3K_DEPTH_SEMANTICS_DIAGNOSTIC_REVIEW_20260628.zip"
    package_path = zip_base
    counter = 1
    while package_path.exists():
        package_path = package_dir / f"GPT_GCP_3K_DEPTH_SEMANTICS_DIAGNOSTIC_REVIEW_20260628_{counter:02d}.zip"
        counter += 1
    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(out_root.rglob("*")):
            if path.is_file():
                arc = path.relative_to(out_root.parent)
                zf.write(path, arc.as_posix())
    package_sha = file_sha256(package_path)
    (package_path.with_suffix(package_path.suffix + ".sha256")).write_text(f"{package_sha}  {package_path.name}\n", encoding="utf-8")
    print(json.dumps({"out_root": str(out_root), "package": str(package_path), "package_sha256": package_sha, **sanity}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
