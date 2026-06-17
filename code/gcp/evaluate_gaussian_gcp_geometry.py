from __future__ import annotations

import argparse
import csv
import json
import math
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
from fit_gcp_sim3 import (  # noqa: E402
    DEFAULT_TARGET_FIELDS,
    apply_similarity,
    fit_similarity_umeyama,
    parse_name_set,
    residual_stats,
)


DEPTH_SUFFIXES = (".npy", ".npz", ".tif", ".tiff", ".png")


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


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
        key = npz_key if npz_key in payload else sorted(payload.files)[0]
        arr = payload[key]
    else:
        arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return np.asarray(arr, dtype=np.float64) * float(scale) + float(offset)


def camera_z_from_depth_value(depth_value: float, x_norm: float, y_norm: float, semantics: str) -> float:
    semantics = semantics.strip().lower()
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
    patch_size: int,
    min_valid_ratio: float,
    min_depth: float,
    depth_semantics: str,
) -> tuple[bool, Dict[str, Any]]:
    if patch_size % 2 != 1 or patch_size < 1:
        raise ValueError("patch_size must be a positive odd integer")
    height, width = depth.shape[:2]
    if u < 0 or v < 0 or u >= width or v >= height:
        return False, {"failure_reason": "pixel_out_of_bounds", "image_width": width, "image_height": height}
    cx = int(round(float(u)))
    cy = int(round(float(v)))
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
    x_norm, y_norm = pixel_to_normalized(camera, float(u), float(v))
    camera_z = camera_z_from_depth_value(median_raw, x_norm, y_norm, depth_semantics)
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
    }


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
    parser = argparse.ArgumentParser(
        description="Depth-only Gaussian GCP geometry evaluator core."
    )
    parser.add_argument("--run_roundtrip_unit_test", action="store_true")
    parser.add_argument("--scene", default="")
    parser.add_argument("--method_id", default="unknown_method")
    parser.add_argument("--colmap_model")
    parser.add_argument("--depth_dir")
    parser.add_argument("--annotations_csv")
    parser.add_argument("--gcp_csv")
    parser.add_argument("--out_dir")
    parser.add_argument("--control_points", default="")
    parser.add_argument("--checkpoint_points", default="")
    parser.add_argument("--target_fields", default=",".join(DEFAULT_TARGET_FIELDS))
    parser.add_argument("--depth_semantics", default="camera_z", choices=[
        "camera_z",
        "ray_distance",
        "inverse_camera_z",
        "inverse_ray_distance",
    ])
    parser.add_argument("--image_domain", default="same_as_colmap_camera")
    parser.add_argument("--pixel_coordinate_convention", default="zero_indexed_pixel_centers")
    parser.add_argument("--depth_scale", type=float, default=1.0)
    parser.add_argument("--depth_offset", type=float, default=0.0)
    parser.add_argument("--npz_key", default="depth")
    parser.add_argument("--patch_size", type=int, default=7)
    parser.add_argument("--min_patch_valid_ratio", type=float, default=0.60)
    parser.add_argument("--min_depth", type=float, default=1e-6)
    parser.add_argument("--min_confidence", type=float, default=0.0)
    parser.add_argument("--min_valid_observations", type=int, default=3)
    parser.add_argument("--max_multiview_scatter_m", type=float, default=0.0)
    args = parser.parse_args()

    if args.run_roundtrip_unit_test:
        result = run_roundtrip_unit_test()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not result["passed"]:
            raise SystemExit(1)
        return

    required = ["colmap_model", "depth_dir", "annotations_csv", "gcp_csv", "out_dir"]
    missing = [name for name in required if not getattr(args, name)]
    if missing:
        raise SystemExit(f"Missing required arguments: {', '.join(missing)}")

    colmap_model = Path(args.colmap_model)
    depth_dir = Path(args.depth_dir)
    annotations_csv = Path(args.annotations_csv)
    gcp_csv = Path(args.gcp_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cameras, images, _points3d = read_model(colmap_model)
    images_by_name = {image.name: image for image in images.values()}
    target_fields = [field.strip() for field in args.target_fields.split(",") if field.strip()]
    target_points = load_target_points(gcp_csv, target_fields)
    control_points = parse_name_set(args.control_points)
    checkpoint_points = parse_name_set(args.checkpoint_points)

    raw_rows = read_csv(annotations_csv)
    observation_rows: List[Dict[str, Any]] = []
    valid_points_by_gcp: Dict[str, List[np.ndarray]] = defaultdict(list)
    failure_counter: Counter[str] = Counter()
    depth_cache: Dict[str, tuple[Path, np.ndarray]] = {}

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
        depth_path = find_depth_path(depth_dir, image_name)
        if depth_path is None:
            base_out["failure_reason"] = "missing_depth_map"
            failure_counter["missing_depth_map"] += 1
            observation_rows.append(base_out)
            continue
        if image_name not in depth_cache:
            depth_cache[image_name] = (
                depth_path,
                load_depth_map(
                    depth_path,
                    scale=float(args.depth_scale),
                    offset=float(args.depth_offset),
                    npz_key=str(args.npz_key),
                ),
            )
        _depth_path, depth = depth_cache[image_name]
        image = images_by_name[image_name]
        camera = cameras[image.camera_id]
        u = float(base_out["u_px"])
        v = float(base_out["v_px"])
        valid, patch_stats = robust_depth_patch(
            depth=depth,
            camera=camera,
            u=u,
            v=v,
            patch_size=int(args.patch_size),
            min_valid_ratio=float(args.min_patch_valid_ratio),
            min_depth=float(args.min_depth),
            depth_semantics=str(args.depth_semantics),
        )
        base_out.update(patch_stats)
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

    common_controls = sorted(control_points & set(aggregated_points) & set(target_points))
    common_checkpoints = sorted(checkpoint_points & set(aggregated_points) & set(target_points))
    control_rows: List[Dict[str, Any]] = []
    checkpoint_rows: List[Dict[str, Any]] = []
    residual_summary = {
        "control": residual_stats(np.empty((0, 3), dtype=np.float64)),
        "checkpoint": residual_stats(np.empty((0, 3), dtype=np.float64)),
        "all": residual_stats(np.empty((0, 3), dtype=np.float64)),
    }
    transform_payload: Dict[str, Any] | None = None
    status = "failed"
    if len(common_controls) >= 3 and common_checkpoints:
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
        status = "ok"
    elif len(common_controls) >= 3:
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
        "colmap_model": str(colmap_model),
        "depth_dir": str(depth_dir),
        "annotations_csv": str(annotations_csv),
        "gcp_csv": str(gcp_csv),
        "target_fields": target_fields,
        "image_domain": args.image_domain,
        "pixel_coordinate_convention": args.pixel_coordinate_convention,
        "depth_semantics": args.depth_semantics,
        "alpha_map_used": False,
        "depth_second_moment_used": False,
        "patch_size": int(args.patch_size),
        "min_patch_valid_ratio": float(args.min_patch_valid_ratio),
        "min_valid_observations": int(args.min_valid_observations),
        "raw_observation_rows": len(raw_rows),
        "valid_observation_rows": int(sum(int(row.get("valid", 0)) for row in observation_rows)),
        "aggregated_gcp_count": len(aggregated_points),
        "control_points_requested": sorted(control_points),
        "checkpoint_points_requested": sorted(checkpoint_points),
        "control_points_used": common_controls,
        "checkpoint_points_used": common_checkpoints,
        "control_count": len(common_controls),
        "checkpoint_count": len(common_checkpoints),
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
            "valid",
            "failure_reason",
            "depth_path",
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

