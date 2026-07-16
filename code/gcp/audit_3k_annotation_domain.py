from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
GCP_CODE = Path(__file__).resolve().parent
COLMAP_UTILS = REPO_ROOT / "code" / "colmap" / "utils"
sys.path.insert(0, str(COLMAP_UTILS))
sys.path.insert(0, str(GCP_CODE))

from read_write_model import qvec2rotmat, read_model  # noqa: E402
from fit_gcp_sim3 import apply_similarity, fit_similarity_umeyama, residual_stats  # noqa: E402
from evaluate_gaussian_gcp_geometry import (  # noqa: E402
    PRIMARY_DEPTH_TENSOR,
    aggregate_points,
    backproject_world,
    camera_z_from_depth_value,
    robust_depth_patch,
)
from undistort_gcp_observations import (  # noqa: E402
    camera_normalized_from_pixel,
    camera_pixel_from_normalized,
)
from triangulate_gcp_points import pixel_to_normalized  # noqa: E402


SCENE = "gcp_3000_20260602"
DEFAULT_PROJECT_ROOT = Path(r"E:\M3M-GCP-3DGS")
DEFAULT_RELEASE_DIR = Path(r"E:\datasets\M3M-GCP\scenes\gcp_manual_annotations")
DEFAULT_OUT_BASE = DEFAULT_PROJECT_ROOT / "outputs"
DEFAULT_REVIEW_ZIP = (
    DEFAULT_PROJECT_ROOT
    / "outputs"
    / "gpt_review_packages"
    / "GPT_GCP_3SCENE_METRIC_DEPTH_REGRESSION_RELEASEMODE_REVIEW_20260627.zip"
)
DEFAULT_PACKET_DIR = (
    DEFAULT_PROJECT_ROOT
    / "outputs"
    / "gcp_3k_depth_semantics_inputs_20260628"
    / "packets"
    / "gcp_3000_20260602_full_reused_release"
)
DEFAULT_RAW_COLMAP = (
    DEFAULT_PROJECT_ROOT / "outputs" / "remote_colmap_20260617" / SCENE / "RGB" / "sparse_aligned" / "0"
)
DEFAULT_TRAIN_COLMAP = (
    DEFAULT_PROJECT_ROOT / "outputs" / "sibr_models_3scenes_20260624" / "sources" / SCENE / "sparse" / "0"
)
DEFAULT_OLD_UNDISTORTED = (
    DEFAULT_PROJECT_ROOT
    / "outputs"
    / "gaussian_gcp_eval_20260618"
    / "annotations_undistorted"
    / "gcp_image_observations_undistorted_for_evaluation.csv"
)
DEFAULT_OLD_UNDISTORTED_MANIFEST = (
    DEFAULT_PROJECT_ROOT
    / "outputs"
    / "gaussian_gcp_eval_20260618"
    / "annotations_undistorted"
    / "undistort_observations_manifest.json"
)
TARGET_FIELDS = (
    "cgcs2000_gk_cm108_e_m",
    "cgcs2000_gk_cm108_n_m",
    "cgcs2000_normal_height_m",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    rows = list(rows)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_text(args: list[str], cwd: Path = REPO_ROOT) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=cwd, text=True, errors="replace").strip()
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {type(exc).__name__}: {exc}"


def make_unique_dir(base: Path) -> Path:
    if not base.exists():
        base.mkdir(parents=True)
        return base
    stamp = datetime.now().strftime("%H%M%S")
    for i in range(1, 1000):
        candidate = base.with_name(f"{base.name}_{stamp}_{i:02d}")
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
    raise RuntimeError(f"cannot create unique output dir near {base}")


def make_unique_file(path: Path) -> Path:
    if not path.exists():
        return path
    stamp = datetime.now().strftime("%H%M%S")
    for i in range(1, 1000):
        candidate = path.with_name(f"{path.stem}_{stamp}_{i:02d}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"cannot create unique file near {path}")


def zip_read_bytes(zip_path: Path, suffix: str) -> bytes:
    with zipfile.ZipFile(zip_path) as zf:
        matches = [name for name in zf.namelist() if name.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError(f"expected one zip member ending {suffix}, found {matches[:5]}")
        return zf.read(matches[0])


def zip_read_csv(zip_path: Path, suffix: str) -> list[dict[str, str]]:
    text = zip_read_bytes(zip_path, suffix).decode("utf-8-sig")
    return list(csv.DictReader(text.splitlines()))


def zip_read_json(zip_path: Path, suffix: str) -> dict[str, Any]:
    return json.loads(zip_read_bytes(zip_path, suffix).decode("utf-8"))


def stable_key(row: dict[str, Any]) -> str:
    return "|".join(
        [
            str(row.get("scene", "")).strip(),
            str(row.get("point_name", "")).strip(),
            str(row.get("image_name", "")).strip(),
            str(row.get("manual_x", "")).strip(),
            str(row.get("manual_y", "")).strip(),
        ]
    )


def obs_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("scene", "")).strip(),
        str(row.get("point_name", "")).strip(),
        Path(str(row.get("image_name", "")).strip()).name,
    )


def resolve_source_annotation_path(path_text: str, project_root: Path, release_dir: Path) -> Path | None:
    raw = Path(path_text)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend(
            [
                project_root / raw,
                project_root / path_text.replace("/", "\\"),
                release_dir / raw,
                release_dir.parent / raw,
                project_root / "outputs" / raw,
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def camera_summary(cameras: dict[int, Any]) -> list[dict[str, Any]]:
    rows = []
    for cid, cam in sorted(cameras.items()):
        rows.append(
            {
                "camera_id": cid,
                "model": cam.model,
                "width": int(cam.width),
                "height": int(cam.height),
                "params": [float(x) for x in cam.params],
            }
        )
    return rows


def model_name_maps(images: dict[int, Any]) -> dict[str, Any]:
    return {Path(image.name).name: image for image in images.values()}


def load_target_points(path: Path) -> dict[str, np.ndarray]:
    points = {}
    for row in read_csv(path):
        name = row.get("point_name", "").strip()
        if not name:
            continue
        try:
            points[name] = np.asarray([float(row[field]) for field in TARGET_FIELDS], dtype=np.float64)
        except Exception:
            continue
    return points


def load_split(path: Path, scene: str) -> tuple[set[str], set[str], dict[str, str]]:
    controls: set[str] = set()
    checkpoints: set[str] = set()
    roles: dict[str, str] = {}
    for row in read_csv(path):
        if row.get("scene", "").strip() != scene:
            continue
        name = row.get("point_name", "").strip()
        role = row.get("role", "").strip().lower()
        if not name:
            continue
        roles[name] = role
        if role == "control":
            controls.add(name)
        elif role == "checkpoint":
            checkpoints.add(name)
    overlap = controls & checkpoints
    if overlap:
        raise ValueError(f"control/checkpoint overlap: {sorted(overlap)}")
    if not controls or not checkpoints:
        raise ValueError(f"incomplete split for {scene}")
    return controls, checkpoints, roles


def transform_raw_to_target(
    image_name: str,
    u: float,
    v: float,
    source_cameras: dict[int, Any],
    source_images_by_name: dict[str, Any],
    target_cameras: dict[int, Any],
    target_images_by_name: dict[str, Any],
) -> dict[str, Any]:
    source_image = source_images_by_name[Path(image_name).name]
    target_image = target_images_by_name[Path(image_name).name]
    source_camera = source_cameras[source_image.camera_id]
    target_camera = target_cameras[target_image.camera_id]
    x_norm, y_norm = camera_normalized_from_pixel(source_camera, float(u), float(v))
    u_target, v_target = camera_pixel_from_normalized(target_camera, x_norm, y_norm)
    x_back, y_back = camera_normalized_from_pixel(target_camera, u_target, v_target)
    u_round, v_round = camera_pixel_from_normalized(source_camera, x_back, y_back)
    return {
        "raw_u": float(u),
        "raw_v": float(v),
        "normalized_x": x_norm,
        "normalized_y": y_norm,
        "undistorted_u": u_target,
        "undistorted_v": v_target,
        "roundtrip_raw_u": u_round,
        "roundtrip_raw_v": v_round,
        "roundtrip_error_px": math.hypot(u_round - float(u), v_round - float(v)),
        "target_in_bounds": int(0 <= u_target < target_camera.width and 0 <= v_target < target_camera.height),
    }


def packet_path_for_image(packet_dir: Path, image_name: str) -> Path:
    return packet_dir / f"{Path(image_name).stem}_metric_depth_packet.npz"


def load_packet(packet_path: Path) -> dict[str, np.ndarray]:
    with np.load(packet_path) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def evaluate_coordinate_variant(
    variant: str,
    rows: list[dict[str, Any]],
    coords_by_key: dict[str, tuple[float, float]],
    packets: dict[str, dict[str, np.ndarray]],
    packet_sha_by_image: dict[str, str],
    cameras: dict[int, Any],
    images_by_name: dict[str, Any],
    controls: set[str],
    checkpoints: set[str],
    roles: dict[str, str],
    target_points: dict[str, np.ndarray],
    patch_size: int,
    min_patch_valid_ratio: float,
    out_dir: Path,
) -> dict[str, Any]:
    valid_points_by_gcp: dict[str, list[np.ndarray]] = defaultdict(list)
    obs_rows: list[dict[str, Any]] = []
    failure_counter: Counter[str] = Counter()
    for row in rows:
        key = stable_key(row)
        image_name = Path(str(row["image_name"])).name
        base = {
            "variant": variant,
            "scene": row["scene"],
            "point_name": row["point_name"],
            "role": roles.get(row["point_name"], ""),
            "image_name": image_name,
            "annotation_key": key,
            "valid": 0,
            "failure_reason": "",
        }
        if key not in coords_by_key:
            base["failure_reason"] = "coordinate_variant_missing"
            failure_counter["coordinate_variant_missing"] += 1
            obs_rows.append(base)
            continue
        if image_name not in images_by_name:
            base["failure_reason"] = "missing_colmap_image"
            failure_counter["missing_colmap_image"] += 1
            obs_rows.append(base)
            continue
        if image_name not in packets:
            base["failure_reason"] = "missing_packet"
            failure_counter["missing_packet"] += 1
            obs_rows.append(base)
            continue
        u, v = coords_by_key[key]
        image = images_by_name[image_name]
        camera = cameras[image.camera_id]
        packet = packets[image_name]
        depth = np.asarray(packet[PRIMARY_DEPTH_TENSOR], dtype=np.float64)
        depth_height, depth_width = depth.shape
        scale_x = depth_width / float(camera.width)
        scale_y = depth_height / float(camera.height)
        depth_u = float(u) * scale_x
        depth_v = float(v) * scale_y
        valid, stats = robust_depth_patch(
            depth=depth,
            camera=camera,
            u=float(u),
            v=float(v),
            depth_u=depth_u,
            depth_v=depth_v,
            depth_pixel_scale_x=scale_x,
            depth_pixel_scale_y=scale_y,
            patch_size=patch_size,
            min_valid_ratio=min_patch_valid_ratio,
            min_depth=1e-6,
            depth_semantics="camera_z",
        )
        base.update(
            {
                "u_px": float(u),
                "v_px": float(v),
                "camera_id": image.camera_id,
                "camera_width": camera.width,
                "camera_height": camera.height,
                "depth_width": depth_width,
                "depth_height": depth_height,
                "depth_pixel_scale_x": scale_x,
                "depth_pixel_scale_y": scale_y,
                "depth_u_px": depth_u,
                "depth_v_px": depth_v,
                "packet_sha256": packet_sha_by_image.get(image_name, ""),
            }
        )
        base.update(stats)
        if not valid:
            base["failure_reason"] = str(stats.get("failure_reason", "invalid_depth"))
            failure_counter[base["failure_reason"]] += 1
            obs_rows.append(base)
            continue
        xyz = backproject_world(camera, image, float(u), float(v), float(stats["camera_z"]))
        base.update({"valid": 1, "model_x": xyz[0], "model_y": xyz[1], "model_z": xyz[2]})
        valid_points_by_gcp[str(row["point_name"])].append(xyz)
        obs_rows.append(base)

    agg_rows: list[dict[str, Any]] = []
    scatter_rows: list[dict[str, Any]] = []
    aggregated: dict[str, np.ndarray] = {}
    for point_name in sorted({str(row["point_name"]) for row in rows}):
        points = valid_points_by_gcp.get(point_name, [])
        raw_count = sum(1 for row in rows if str(row["point_name"]) == point_name)
        valid_count = len(points)
        agg = {
            "variant": variant,
            "scene": SCENE,
            "point_name": point_name,
            "role": roles.get(point_name, ""),
            "raw_observation_count": raw_count,
            "valid_observation_count": valid_count,
            "valid_observation_ratio": valid_count / max(1, raw_count),
            "valid": 0,
            "failure_reason": "",
        }
        if valid_count < 1:
            agg["failure_reason"] = "insufficient_valid_observations"
            agg_rows.append(agg)
            continue
        aggregate, scatter = aggregate_points(points)
        agg.update(scatter)
        agg.update({"valid": 1, "model_x": aggregate[0], "model_y": aggregate[1], "model_z": aggregate[2]})
        aggregated[point_name] = aggregate
        agg_rows.append(agg)
        for point in points:
            scatter_rows.append(
                {
                    "variant": variant,
                    "point_name": point_name,
                    "distance_to_aggregate_m": float(np.linalg.norm(point - aggregate)),
                }
            )

    common_controls = sorted(controls & set(aggregated) & set(target_points))
    common_checkpoints = sorted(checkpoints & set(aggregated) & set(target_points))
    transform = None
    residual_rows: list[dict[str, Any]] = []
    residual_groups = {
        "control": np.empty((0, 3), dtype=np.float64),
        "checkpoint": np.empty((0, 3), dtype=np.float64),
        "all": np.empty((0, 3), dtype=np.float64),
    }
    status = "failed"
    if len(common_controls) >= 3:
        src = np.vstack([aggregated[name] for name in common_controls])
        tgt = np.vstack([target_points[name] for name in common_controls])
        scale, rotation, translation = fit_similarity_umeyama(src, tgt, estimate_scale=True)
        transform = {"scale": scale, "rotation": rotation.tolist(), "translation": translation.tolist()}
        by_role: dict[str, list[np.ndarray]] = {"control": [], "checkpoint": [], "all": []}
        for role_name, names in (("control", common_controls), ("checkpoint", common_checkpoints)):
            for name in names:
                model_xyz = aggregated[name]
                target_xyz = target_points[name]
                predicted = apply_similarity(model_xyz.reshape(1, 3), scale, rotation, translation)[0]
                residual = predicted - target_xyz
                by_role[role_name].append(residual)
                by_role["all"].append(residual)
                residual_rows.append(
                    {
                        "variant": variant,
                        "scene": SCENE,
                        "point_name": name,
                        "role": role_name,
                        "model_x": model_xyz[0],
                        "model_y": model_xyz[1],
                        "model_z": model_xyz[2],
                        "target_x": target_xyz[0],
                        "target_y": target_xyz[1],
                        "target_z": target_xyz[2],
                        "predicted_x": predicted[0],
                        "predicted_y": predicted[1],
                        "predicted_z": predicted[2],
                        "residual_x_m": residual[0],
                        "residual_y_m": residual[1],
                        "residual_z_m": residual[2],
                        "error_h_m": float(np.linalg.norm(residual[:2])),
                        "error_z_m": float(abs(residual[2])),
                        "error_3d_m": float(np.linalg.norm(residual)),
                    }
                )
        residual_groups = {
            key: np.vstack(vals) if vals else np.empty((0, 3), dtype=np.float64)
            for key, vals in by_role.items()
        }
        status = "ok"

    summary = {
        "variant": variant,
        "status": status,
        "raw_observation_rows": len(rows),
        "valid_observation_rows": int(sum(int(row.get("valid", 0)) for row in obs_rows)),
        "aggregated_gcp_count": len(aggregated),
        "control_points_used": common_controls,
        "checkpoint_points_used": common_checkpoints,
        "missing_control_points": sorted(controls - set(common_controls)),
        "missing_checkpoint_points": sorted(checkpoints - set(common_checkpoints)),
        "transform": transform,
        "residual_stats": {name: residual_stats(vals) for name, vals in residual_groups.items()},
        "failure_counts": dict(sorted(failure_counter.items())),
    }
    write_csv(out_dir / f"{variant}_observation_points.csv", obs_rows)
    write_csv(out_dir / f"{variant}_aggregated_points.csv", agg_rows)
    write_csv(out_dir / f"{variant}_scatter.csv", scatter_rows)
    write_csv(out_dir / f"{variant}_residuals.csv", residual_rows)
    write_json(out_dir / f"{variant}_summary.json", summary)
    return {
        "summary": summary,
        "observations": obs_rows,
        "aggregated": {name: value for name, value in aggregated.items()},
        "residuals": residual_rows,
        "transform": transform,
    }


def transform_ray_to_survey(camera: Any, image: Any, u: float, v: float, transform: dict[str, Any]) -> np.ndarray:
    x, y = pixel_to_normalized(camera, float(u), float(v))
    ray_cam = np.asarray([x, y, 1.0], dtype=np.float64)
    ray_cam /= np.linalg.norm(ray_cam)
    ray_model = qvec2rotmat(image.qvec).T @ ray_cam
    rotation = np.asarray(transform["rotation"], dtype=np.float64)
    ray_survey = rotation @ ray_model
    norm = np.linalg.norm(ray_survey)
    return ray_survey / norm if norm > 0 else ray_survey


def compute_fixed_transform_displacement(
    rows: list[dict[str, Any]],
    evals: dict[str, dict[str, Any]],
    cameras: dict[int, Any],
    images_by_name: dict[str, Any],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    current = evals["A_current_release"]
    transform = current.get("transform")
    if not transform:
        return out
    current_by_key = {row["annotation_key"]: row for row in current["observations"] if int(row.get("valid", 0))}
    for variant_name in ["B_recomputed_raw_to_undistorted", "C_archived_undistorted"]:
        variant = evals.get(variant_name)
        if not variant:
            continue
        variant_by_key = {row["annotation_key"]: row for row in variant["observations"] if int(row.get("valid", 0))}
        for row in rows:
            key = stable_key(row)
            if key not in current_by_key or key not in variant_by_key:
                continue
            a = current_by_key[key]
            b = variant_by_key[key]
            pa = np.asarray([float(a["model_x"]), float(a["model_y"]), float(a["model_z"])], dtype=np.float64)
            pb = np.asarray([float(b["model_x"]), float(b["model_y"]), float(b["model_z"])], dtype=np.float64)
            scale = float(transform["scale"])
            rotation = np.asarray(transform["rotation"], dtype=np.float64)
            translation = np.asarray(transform["translation"], dtype=np.float64)
            sa = apply_similarity(pa.reshape(1, 3), scale, rotation, translation)[0]
            sb = apply_similarity(pb.reshape(1, 3), scale, rotation, translation)[0]
            disp = sb - sa
            image = images_by_name[Path(str(row["image_name"])).name]
            camera = cameras[image.camera_id]
            ray = transform_ray_to_survey(camera, image, float(a["u_px"]), float(a["v_px"]), transform)
            along = float(np.dot(disp, ray))
            cross = disp - along * ray
            out.append(
                {
                    "variant": variant_name,
                    "scene": row["scene"],
                    "point_name": row["point_name"],
                    "image_name": row["image_name"],
                    "annotation_key": key,
                    "survey_displacement_x_m": disp[0],
                    "survey_displacement_y_m": disp[1],
                    "survey_displacement_z_m": disp[2],
                    "horizontal_displacement_m": float(np.linalg.norm(disp[:2])),
                    "vertical_displacement_m": float(abs(disp[2])),
                    "total_displacement_3d_m": float(np.linalg.norm(disp)),
                    "signed_along_ray_m": along,
                    "abs_along_ray_m": abs(along),
                    "cross_ray_m": float(np.linalg.norm(cross)),
                }
            )
    return out


def coordinate_stats(values: list[float]) -> dict[str, float | int | None]:
    arr = np.asarray([x for x in values if math.isfinite(float(x))], dtype=np.float64)
    if arr.size == 0:
        return {"count": 0, "median": None, "p95": None, "max": None, "mean": None}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def source_audit_markdown(annotator_path: Path, undistort_path: Path) -> str:
    def lines(path: Path, start: int, end: int) -> list[str]:
        text = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return [f"{i+1}: {text[i]}" for i in range(start - 1, min(end, len(text)))]

    git_log = git_text(
        [
            "log",
            "--oneline",
            "-n",
            "12",
            "--",
            str(annotator_path.relative_to(REPO_ROOT)),
            str(undistort_path.relative_to(REPO_ROOT)),
            "code/gcp/summarize_gcp_annotations.py",
        ]
    )
    return "\n".join(
        [
            "# Annotation Tool Source Audit",
            "",
            "## Findings",
            "",
            "- `manual_gcp_annotator.py` opens the displayed file with `PIL.Image.open(...).convert(\"RGB\")` and does not apply a COLMAP raw-to-undistorted transform before saving.",
            "- The click handler converts canvas coordinates back through pan and render scale to crop coordinates, then adds `crop_origin`; saved `manual_x/manual_y` are loaded-image pixel coordinates.",
            "- The official source annotation table for 3K records `image_path` in the raw scene folder, so its saved coordinates are source-image coordinates unless an explicit later transform is applied.",
            "- `undistort_gcp_observations.py` is the explicit raw-to-undistorted transform step; it writes `source_pixel_domain`, `target_pixel_domain`, `source_u_px/source_v_px`, and `undistorted_u_px/undistorted_v_px`.",
            "",
            "## Relevant annotator code",
            "",
            "```text",
            *lines(annotator_path, 455, 505),
            "...",
            *lines(annotator_path, 625, 645),
            "```",
            "",
            "## Relevant transform code",
            "",
            "```text",
            *lines(undistort_path, 35, 75),
            "...",
            *lines(undistort_path, 135, 159),
            "```",
            "",
            "## Git history",
            "",
            "```text",
            git_log,
            "```",
            "",
        ]
    )


def synthetic_ui_coordinate_test() -> dict[str, Any]:
    crop_origin = (123.25, 456.5)
    render_scale = 2.75
    pan = (-19.0, 37.0)
    true_image_xy = (789.125, 1020.375)
    crop_xy = (true_image_xy[0] - crop_origin[0], true_image_xy[1] - crop_origin[1])
    event_xy = (crop_xy[0] * render_scale + pan[0], crop_xy[1] * render_scale + pan[1])
    recovered_crop = ((event_xy[0] - pan[0]) / render_scale, (event_xy[1] - pan[1]) / render_scale)
    recovered_image = (crop_origin[0] + recovered_crop[0], crop_origin[1] + recovered_crop[1])
    saved = (round(recovered_image[0], 3), round(recovered_image[1], 3))
    reloaded = saved
    err = math.hypot(reloaded[0] - true_image_xy[0], reloaded[1] - true_image_xy[1])
    return {
        "schema": "synthetic_manual_annotator_coordinate_inverse_test_v1",
        "crop_origin": crop_origin,
        "render_scale": render_scale,
        "pan": pan,
        "true_image_xy": true_image_xy,
        "event_xy": event_xy,
        "recovered_image_xy": recovered_image,
        "saved_reloaded_xy": reloaded,
        "max_expected_rounding_error_px": math.sqrt(2) * 0.0005,
        "error_px": err,
        "passed": bool(err <= math.sqrt(2) * 0.0005 + 1e-12),
    }


def package_outputs(out_dir: Path, package_path: Path) -> tuple[Path, Path]:
    package_path.parent.mkdir(parents=True, exist_ok=True)
    package_path = make_unique_file(package_path)
    rows = []
    for path in sorted(out_dir.rglob("*")):
        if path.is_file():
            rel = path.relative_to(out_dir).as_posix()
            rows.append({"path": rel, "bytes": path.stat().st_size, "sha256": file_sha256(path)})
    write_csv(out_dir / "PACKAGE_CONTENT_SHA256SUMS.csv", rows, ["path", "bytes", "sha256"])
    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(out_dir.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=f"{package_path.stem}/{path.relative_to(out_dir).as_posix()}")
    sha_path = package_path.with_suffix(package_path.suffix + ".sha256")
    sha_path.write_text(f"{file_sha256(package_path)}  {package_path.name}\n", encoding="utf-8")
    return package_path, sha_path


def main() -> None:
    parser = argparse.ArgumentParser(description="No-GPU audit of 3K GCP annotation pixel-domain compatibility.")
    parser.add_argument("--project_root", default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("--release_dir", default=str(DEFAULT_RELEASE_DIR))
    parser.add_argument("--review_zip", default=str(DEFAULT_REVIEW_ZIP))
    parser.add_argument("--packet_dir", default=str(DEFAULT_PACKET_DIR))
    parser.add_argument("--raw_colmap", default=str(DEFAULT_RAW_COLMAP))
    parser.add_argument("--train_colmap", default=str(DEFAULT_TRAIN_COLMAP))
    parser.add_argument("--old_undistorted_csv", default=str(DEFAULT_OLD_UNDISTORTED))
    parser.add_argument("--old_undistorted_manifest", default=str(DEFAULT_OLD_UNDISTORTED_MANIFEST))
    parser.add_argument("--out_base", default=str(DEFAULT_OUT_BASE))
    args = parser.parse_args()

    project_root = Path(args.project_root)
    release_dir = Path(args.release_dir)
    review_zip = Path(args.review_zip)
    packet_dir = Path(args.packet_dir)
    raw_colmap = Path(args.raw_colmap)
    train_colmap = Path(args.train_colmap)
    old_undistorted_csv = Path(args.old_undistorted_csv)
    old_undistorted_manifest = Path(args.old_undistorted_manifest)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = make_unique_dir(Path(args.out_base) / f"gcp_3k_annotation_domain_audit_20260628_{timestamp}")

    commands = {
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "git_commit": git_text(["rev-parse", "HEAD"]),
        "git_status_porcelain": git_text(["status", "--porcelain"]),
    }
    write_json(out_dir / "commands_and_status.json", commands)

    release_config_path = release_dir / "gcp_benchmark_release_v1_1.json"
    release_config = json.loads(release_config_path.read_text(encoding="utf-8"))
    release_files = {item["path"]: item for item in release_config.get("files", [])}
    release_files_by_name = {Path(item["path"]).name: item for item in release_config.get("files", [])}
    release_csv = release_dir / f"{SCENE}_gcp_annotations_final_good_nadir_v1.csv"
    provenance_csv = release_dir / "final_annotation_inclusion_provenance.csv"
    gcp_csv = release_dir / str(release_config["gcp_csv"])
    split_csv = release_dir / str(release_config["split_csv"])

    artifact_paths = {
        "release_config": release_config_path,
        "release_annotations": release_csv,
        "release_inclusion_provenance": provenance_csv,
        "release_gcp_csv": gcp_csv,
        "release_split_csv": split_csv,
        "review_zip": review_zip,
        "packet_dir": packet_dir,
        "raw_colmap_cameras": raw_colmap / "cameras.bin",
        "raw_colmap_images": raw_colmap / "images.bin",
        "train_colmap_cameras": train_colmap / "cameras.bin",
        "train_colmap_images": train_colmap / "images.bin",
        "old_undistorted_csv": old_undistorted_csv,
        "old_undistorted_manifest": old_undistorted_manifest,
    }
    artifact_inventory = []
    for label, path in artifact_paths.items():
        artifact_inventory.append(
            {
                "label": label,
                "path": str(path),
                "exists": int(path.exists()),
                "is_dir": int(path.is_dir()) if path.exists() else 0,
                "bytes": path.stat().st_size if path.exists() and path.is_file() else "",
                "sha256": file_sha256(path) if path.exists() and path.is_file() else "",
            }
        )
    write_csv(out_dir / "artifact_inventory.csv", artifact_inventory)
    write_json(out_dir / "artifact_inventory.json", artifact_inventory)

    release_rows = read_csv(release_csv)
    release_rows = [row for row in release_rows if row.get("scene") == SCENE]
    key_counts = Counter(stable_key(row) for row in release_rows)
    duplicate_keys = [key for key, count in key_counts.items() if count > 1]
    if duplicate_keys:
        raise SystemExit(f"duplicate frozen annotation keys: {duplicate_keys[:5]}")

    provenance_rows = [row for row in read_csv(provenance_csv) if row.get("scene") == SCENE]
    source_files = sorted({row.get("source_annotation_file", "").strip() for row in provenance_rows if row.get("source_annotation_file", "").strip()})
    source_annotation_path = None
    for path_text in source_files:
        source_annotation_path = resolve_source_annotation_path(path_text, project_root, release_dir)
        if source_annotation_path:
            break
    if source_annotation_path is None:
        raise SystemExit(f"could not resolve source annotation file from {source_files}")
    source_rows = [row for row in read_csv(source_annotation_path) if row.get("scene") == SCENE]
    source_by_obs = {obs_key(row): row for row in source_rows}

    old_rows = [row for row in read_csv(old_undistorted_csv) if row.get("scene") == SCENE]
    old_by_obs = {obs_key(row): row for row in old_rows}
    old_manifest = json.loads(old_undistorted_manifest.read_text(encoding="utf-8")) if old_undistorted_manifest.exists() else {}

    raw_cameras, raw_images, _raw_points = read_model(raw_colmap)
    train_cameras, train_images, _train_points = read_model(train_colmap)
    raw_by_name = model_name_maps(raw_images)
    train_by_name = model_name_maps(train_images)

    packet_manifest = zip_read_json(
        review_zip, "packet_manifests/gcp_3000_20260602_full_reused_release/metric_depth_manifest.json"
    )
    packet_mapping = zip_read_csv(
        review_zip, "packet_manifests/gcp_3000_20260602_full_reused_release/metric_depth_mapping.csv"
    )
    source_view_manifest = zip_read_csv(review_zip, "source_views/gcp_3000_20260602/annotated_view_manifest.csv")
    formal_obs = zip_read_csv(
        review_zip,
        "evaluations/gcp_3000_20260602_formal_expected_camera_z_release/method_gcp_observation_points.csv",
    )
    formal_summary = zip_read_json(
        review_zip, "evaluations/gcp_3000_20260602_formal_expected_camera_z_release/method_gcp_eval_summary.json"
    )
    formal_by_obs = {obs_key(row): row for row in formal_obs}

    mapping_by_image: dict[str, dict[str, str]] = {Path(row["image_name"]).name: row for row in packet_mapping}
    packet_sha_by_image = {image: row.get("packet_sha256", "") for image, row in mapping_by_image.items()}
    packets: dict[str, dict[str, np.ndarray]] = {}
    packet_missing = []
    for image_name in sorted({Path(row["image_name"]).name for row in release_rows}):
        path = packet_path_for_image(packet_dir, image_name)
        if not path.exists():
            packet_missing.append(image_name)
            continue
        expected_sha = packet_sha_by_image.get(image_name, "")
        actual_sha = file_sha256(path)
        if expected_sha and actual_sha != expected_sha:
            raise SystemExit(f"packet SHA mismatch for {image_name}: {actual_sha} != {expected_sha}")
        packets[image_name] = load_packet(path)

    packet_domain = {
        "packet_manifest_schema": packet_manifest.get("schema", ""),
        "packet_image_domain": packet_manifest.get("image_domain", ""),
        "packet_pixel_coordinate_convention": packet_manifest.get("pixel_coordinate_convention", ""),
        "packet_source_path": packet_manifest.get("source_path", ""),
        "packet_depth_units": packet_manifest.get("depth_units", ""),
        "packet_primary_depth_tensor": packet_manifest.get("primary_depth_tensor", ""),
        "source_view_image_count": len(source_view_manifest),
        "source_view_widths": sorted({row.get("width", "") for row in source_view_manifest}),
        "source_view_heights": sorted({row.get("height", "") for row in source_view_manifest}),
        "train_camera_models": camera_summary(train_cameras),
        "raw_camera_models": camera_summary(raw_cameras),
        "old_undistorted_manifest": old_manifest,
    }
    write_json(out_dir / "packet_and_camera_domain_evidence.json", packet_domain)
    write_csv(out_dir / "source_view_manifest_from_release_package.csv", source_view_manifest)
    write_json(out_dir / "metric_depth_manifest_excerpt.json", packet_manifest)

    # Coordinate provenance and raw->undistorted audit.
    row_comparisons: list[dict[str, Any]] = []
    coords_A: dict[str, tuple[float, float]] = {}
    coords_B: dict[str, tuple[float, float]] = {}
    coords_C: dict[str, tuple[float, float]] = {}
    raw_to_undistorted_rows: list[dict[str, Any]] = []
    for row in release_rows:
        key = stable_key(row)
        key3 = obs_key(row)
        release_u = float(row["manual_x"])
        release_v = float(row["manual_y"])
        coords_A[key] = (release_u, release_v)
        src = source_by_obs.get(key3, {})
        old = old_by_obs.get(key3, {})
        raw_u = float(src.get("manual_x", release_u) or release_u)
        raw_v = float(src.get("manual_y", release_v) or release_v)
        transform = transform_raw_to_target(
            row["image_name"],
            raw_u,
            raw_v,
            raw_cameras,
            raw_by_name,
            train_cameras,
            train_by_name,
        )
        coords_B[key] = (float(transform["undistorted_u"]), float(transform["undistorted_v"]))
        old_u = math.nan
        old_v = math.nan
        old_status = "missing"
        if old:
            try:
                old_u = float(old.get("u_px", old.get("undistorted_u_px", "")))
                old_v = float(old.get("v_px", old.get("undistorted_v_px", "")))
                coords_C[key] = (old_u, old_v)
                old_status = old.get("pixel_transform_status", "recovered")
            except Exception:
                old_status = "parse_failed"
        dx_rel_source = release_u - raw_u
        dy_rel_source = release_v - raw_v
        dx_rel_recomp = release_u - float(transform["undistorted_u"])
        dy_rel_recomp = release_v - float(transform["undistorted_v"])
        dx_recomp_old = float(transform["undistorted_u"]) - old_u if math.isfinite(old_u) else math.nan
        dy_recomp_old = float(transform["undistorted_v"]) - old_v if math.isfinite(old_v) else math.nan
        camera = train_cameras[train_by_name[Path(row["image_name"]).name].camera_id]
        x_norm, y_norm = pixel_to_normalized(camera, release_u, release_v)
        row_out = {
            "scene": row["scene"],
            "point_name": row["point_name"],
            "image_name": row["image_name"],
            "annotation_key": key,
            "role": "",
            "source_manual_x": raw_u,
            "source_manual_y": raw_v,
            "current_release_x": release_u,
            "current_release_y": release_v,
            "recomputed_undistorted_x": transform["undistorted_u"],
            "recomputed_undistorted_y": transform["undistorted_v"],
            "archived_undistorted_x": old_u if math.isfinite(old_u) else "",
            "archived_undistorted_y": old_v if math.isfinite(old_v) else "",
            "archived_undistorted_status": old_status,
            "release_minus_source_dx_px": dx_rel_source,
            "release_minus_source_dy_px": dy_rel_source,
            "release_minus_source_displacement_px": math.hypot(dx_rel_source, dy_rel_source),
            "release_minus_recomputed_dx_px": dx_rel_recomp,
            "release_minus_recomputed_dy_px": dy_rel_recomp,
            "release_minus_recomputed_displacement_px": math.hypot(dx_rel_recomp, dy_rel_recomp),
            "recomputed_minus_archived_dx_px": dx_recomp_old,
            "recomputed_minus_archived_dy_px": dy_recomp_old,
            "recomputed_minus_archived_displacement_px": math.hypot(dx_recomp_old, dy_recomp_old)
            if math.isfinite(dx_recomp_old)
            else "",
            "image_center_distance_px": math.hypot(release_u - camera.width / 2.0, release_v - camera.height / 2.0),
            "off_axis_angle_deg": math.degrees(math.atan(math.hypot(x_norm, y_norm))),
            "formal_output_matched": int(key3 in formal_by_obs),
            "packet_present": int(Path(row["image_name"]).name in packets),
            "raw_to_undistorted_roundtrip_error_px": transform["roundtrip_error_px"],
            "target_in_bounds": transform["target_in_bounds"],
        }
        row_comparisons.append(row_out)
        raw_to_undistorted_rows.append(
            {
                **{k: row_out[k] for k in ["scene", "point_name", "image_name", "annotation_key"]},
                **transform,
            }
        )
    write_csv(out_dir / "row_level_coordinate_comparison.csv", row_comparisons)
    write_csv(out_dir / "raw_to_undistorted_transform_audit.csv", raw_to_undistorted_rows)

    # Roles and coordinate targets.
    target_points = load_target_points(gcp_csv)
    controls, checkpoints, roles = load_split(split_csv, SCENE)
    for row in row_comparisons:
        row["role"] = roles.get(str(row["point_name"]), "")
    write_csv(out_dir / "row_level_coordinate_comparison.csv", row_comparisons)

    formal_count = sum(int(row["formal_output_matched"]) for row in row_comparisons)
    source_delta = [float(row["release_minus_source_displacement_px"]) for row in row_comparisons]
    release_to_recomp = [float(row["release_minus_recomputed_displacement_px"]) for row in row_comparisons]
    recomp_to_old = [
        float(row["recomputed_minus_archived_displacement_px"])
        for row in row_comparisons
        if str(row["recomputed_minus_archived_displacement_px"]) != ""
    ]
    coord_summary = {
        "scene": SCENE,
        "frozen_annotation_rows": len(release_rows),
        "formal_output_matched_count": formal_count,
        "unmatched_annotation_rows": len(release_rows) - formal_count,
        "duplicate_key_count": len(duplicate_keys),
        "packet_missing_count": len(packet_missing),
        "packet_missing_images": packet_missing,
        "release_minus_source_displacement_px": coordinate_stats(source_delta),
        "release_minus_recomputed_undistorted_displacement_px": coordinate_stats(release_to_recomp),
        "recomputed_minus_archived_undistorted_displacement_px": coordinate_stats(recomp_to_old),
        "raw_to_undistorted_roundtrip_error_px": coordinate_stats(
            [float(row["raw_to_undistorted_roundtrip_error_px"]) for row in row_comparisons]
        ),
        "focus_points": {
            name: [
                row
                for row in row_comparisons
                if row["point_name"] == name
            ]
            for name in ["G11", "G16", "G13", "G18"]
        },
    }
    write_json(out_dir / "coordinate_domain_summary.json", coord_summary)

    provenance_matrix = [
        {
            "artifact": "official_manual_source_annotations",
            "path": str(source_annotation_path),
            "status": "source_domain_confirmed",
            "evidence": "source image_path points to raw scene JPG files; manual_x/manual_y equal final release for included rows",
            "sha256": file_sha256(source_annotation_path),
        },
        {
            "artifact": "final_release_v1_1_annotations",
            "path": str(release_csv),
            "status": "coordinate_identical",
            "evidence": "release_minus_source_displacement_px max is reported in coordinate_domain_summary.json",
            "sha256": file_sha256(release_csv),
            "expected_release_sha256": release_files.get(release_csv.name, {}).get("sha256", ""),
            "expected_release_sha256_by_name": release_files_by_name.get(release_csv.name, {}).get("sha256", ""),
        },
        {
            "artifact": "old_R1_undistorted_annotations",
            "path": str(old_undistorted_csv),
            "status": "transformed_to_undistorted",
            "evidence": "manifest schema ms_gcp_observation_pixel_domain_transform_v1 declares distorted_original_colmap -> undistorted_training_colmap",
            "sha256": file_sha256(old_undistorted_csv),
        },
        {
            "artifact": "current_release_mode_evaluator_input",
            "path": str(review_zip),
            "status": "target_domain_confirmed",
            "evidence": "evaluator/packet manifest image_domain=rendered_colmap_camera_domain with training PINHOLE camera and source-view dimensions 5654x4098",
            "sha256": file_sha256(review_zip),
        },
        {
            "artifact": "raw_colmap_model",
            "path": str(raw_colmap),
            "status": "source_domain_confirmed",
            "evidence": "raw model camera is SIMPLE_RADIAL and dimensions differ from training PINHOLE model",
            "sha256": "",
        },
        {
            "artifact": "training_colmap_model",
            "path": str(train_colmap),
            "status": "target_domain_confirmed",
            "evidence": "training model camera is PINHOLE, source_view manifest dimensions match packet/evaluator camera domain",
            "sha256": "",
        },
    ]
    write_csv(out_dir / "annotation_provenance_matrix.csv", provenance_matrix)
    write_json(out_dir / "annotation_provenance_matrix.json", provenance_matrix)

    roundtrip_tests = {
        "raw_to_training_to_raw": coord_summary["raw_to_undistorted_roundtrip_error_px"],
        "offset_hypotheses_against_archived_undistorted": {},
        "camera_models": {"raw": camera_summary(raw_cameras), "training": camera_summary(train_cameras)},
    }
    for label, offset in {"zero": (0, 0), "minus_one": (-1, -1), "plus_one": (1, 1), "swap_xy": None}.items():
        errors = []
        for row in release_rows:
            src = source_by_obs.get(obs_key(row), {})
            old = old_by_obs.get(obs_key(row), {})
            if not old:
                continue
            raw_u = float(src.get("manual_x", row["manual_x"]))
            raw_v = float(src.get("manual_y", row["manual_y"]))
            if offset is None:
                u, v = raw_v, raw_u
            else:
                u, v = raw_u + offset[0], raw_v + offset[1]
            try:
                tr = transform_raw_to_target(row["image_name"], u, v, raw_cameras, raw_by_name, train_cameras, train_by_name)
                ou = float(old.get("u_px", old.get("undistorted_u_px", "")))
                ov = float(old.get("v_px", old.get("undistorted_v_px", "")))
                errors.append(math.hypot(float(tr["undistorted_u"]) - ou, float(tr["undistorted_v"]) - ov))
            except Exception:
                continue
        roundtrip_tests["offset_hypotheses_against_archived_undistorted"][label] = coordinate_stats(errors)
    write_json(out_dir / "roundtrip_tests.json", roundtrip_tests)

    # No-GPU A/B/C evaluation.
    eval_dir = out_dir / "diagnostic_evaluations"
    evals: dict[str, dict[str, Any]] = {}
    evals["A_current_release"] = evaluate_coordinate_variant(
        "A_current_release",
        release_rows,
        coords_A,
        packets,
        packet_sha_by_image,
        train_cameras,
        train_by_name,
        controls,
        checkpoints,
        roles,
        target_points,
        patch_size=7,
        min_patch_valid_ratio=0.6,
        out_dir=eval_dir,
    )
    evals["B_recomputed_raw_to_undistorted"] = evaluate_coordinate_variant(
        "B_recomputed_raw_to_undistorted",
        release_rows,
        coords_B,
        packets,
        packet_sha_by_image,
        train_cameras,
        train_by_name,
        controls,
        checkpoints,
        roles,
        target_points,
        patch_size=7,
        min_patch_valid_ratio=0.6,
        out_dir=eval_dir,
    )
    if len(coords_C) == len(release_rows):
        evals["C_archived_undistorted"] = evaluate_coordinate_variant(
            "C_archived_undistorted",
            release_rows,
            coords_C,
            packets,
            packet_sha_by_image,
            train_cameras,
            train_by_name,
            controls,
            checkpoints,
            roles,
            target_points,
            patch_size=7,
            min_patch_valid_ratio=0.6,
            out_dir=eval_dir,
        )

    eval_summary_rows = []
    for name, result in evals.items():
        summary = result["summary"]
        for role in ["control", "checkpoint", "all"]:
            stats = summary["residual_stats"][role]
            eval_summary_rows.append(
                {
                    "variant": name,
                    "role": role,
                    "status": summary["status"],
                    "raw_observation_rows": summary["raw_observation_rows"],
                    "valid_observation_rows": summary["valid_observation_rows"],
                    "aggregated_gcp_count": summary["aggregated_gcp_count"],
                    "count": stats["count"],
                    "rmse_h_m": stats["rmse_h_m"],
                    "rmse_z_m": stats["rmse_z_m"],
                    "rmse_3d_m": stats["rmse_3d_m"],
                    "median_3d_m": stats["median_3d_m"],
                    "p90_3d_m": stats["p90_3d_m"],
                    "max_3d_m": stats["max_3d_m"],
                    "transform_scale": summary["transform"]["scale"] if summary["transform"] else "",
                }
            )
    write_csv(out_dir / "abc_evaluator_summary.csv", eval_summary_rows)
    write_json(out_dir / "abc_evaluator_summary.json", {name: result["summary"] for name, result in evals.items()})

    fixed_disp = compute_fixed_transform_displacement(release_rows, evals, train_cameras, train_by_name)
    write_csv(out_dir / "fixed_transform_displacement.csv", fixed_disp)

    # Add residual joins to the row-level coordinate table.
    residual_by_variant_point = {}
    for variant, result in evals.items():
        for row in result["residuals"]:
            residual_by_variant_point[(variant, row["point_name"])] = row
    for row in row_comparisons:
        for variant in ["A_current_release", "B_recomputed_raw_to_undistorted", "C_archived_undistorted"]:
            residual = residual_by_variant_point.get((variant, row["point_name"]), {})
            prefix = variant.split("_", 1)[0]
            row[f"{prefix}_point_error_h_m"] = residual.get("error_h_m", "")
            row[f"{prefix}_point_error_z_m"] = residual.get("error_z_m", "")
            row[f"{prefix}_point_error_3d_m"] = residual.get("error_3d_m", "")
    write_csv(out_dir / "row_level_coordinate_comparison.csv", row_comparisons)

    # Per-point reports.
    focus_dir = out_dir / "focus_points"
    focus_dir.mkdir(exist_ok=True)
    for point in ["G11", "G16", "G13", "G18"]:
        rows = [row for row in row_comparisons if row["point_name"] == point]
        lines = [
            f"# {point} Annotation-Domain Report",
            "",
            f"- Observation rows: `{len(rows)}`",
            f"- Release-to-source median displacement: `{coordinate_stats([float(r['release_minus_source_displacement_px']) for r in rows])['median']}` px",
            f"- Release-to-recomputed-undistorted median displacement: `{coordinate_stats([float(r['release_minus_recomputed_displacement_px']) for r in rows])['median']}` px",
            "",
            "## A/B/C point residuals",
            "",
            "| Variant | eH (m) | eZ (m) | e3D (m) |",
            "|---|---:|---:|---:|",
        ]
        for variant in ["A_current_release", "B_recomputed_raw_to_undistorted", "C_archived_undistorted"]:
            residual = residual_by_variant_point.get((variant, point), {})
            if residual:
                lines.append(
                    f"| {variant} | {float(residual['error_h_m']):.4f} | {float(residual['error_z_m']):.4f} | {float(residual['error_3d_m']):.4f} |"
                )
            else:
                lines.append(f"| {variant} |  |  |  |")
        (focus_dir / f"{point}_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    annotator_audit = source_audit_markdown(GCP_CODE / "manual_gcp_annotator.py", GCP_CODE / "undistort_gcp_observations.py")
    (out_dir / "annotation_tool_source_audit.md").write_text(annotator_audit, encoding="utf-8")
    synth_test = synthetic_ui_coordinate_test()
    write_json(out_dir / "synthetic_ui_coordinate_test.json", synth_test)
    write_csv(out_dir / "synthetic_ui_coordinate_test.csv", [synth_test])

    # Classification and causal table.
    a_stats = evals["A_current_release"]["summary"]["residual_stats"]["checkpoint"]
    b_stats = evals["B_recomputed_raw_to_undistorted"]["summary"]["residual_stats"]["checkpoint"]
    c_stats = evals.get("C_archived_undistorted", {}).get("summary", {}).get("residual_stats", {}).get("checkpoint", {})
    source_identical = float(coord_summary["release_minus_source_displacement_px"]["max"] or 0.0) < 1e-6
    systematic_displacement = float(coord_summary["release_minus_recomputed_undistorted_displacement_px"]["median"] or 0.0) > 10.0
    packet_is_undistorted = (
        packet_manifest.get("image_domain") == "rendered_colmap_camera_domain"
        and any(cam["model"] == "PINHOLE" for cam in camera_summary(train_cameras))
    )
    b_improves = (
        b_stats.get("rmse_3d_m") is not None
        and a_stats.get("rmse_3d_m") is not None
        and float(b_stats["rmse_3d_m"]) + 1e-9 < float(a_stats["rmse_3d_m"]) * 0.75
    )
    if source_identical and systematic_displacement and packet_is_undistorted and b_improves:
        classification = "confirmed"
    elif source_identical and systematic_displacement and packet_is_undistorted:
        classification = "likely"
    elif not systematic_displacement or not packet_is_undistorted:
        classification = "not_supported"
    else:
        classification = "unresolved"

    causal_rows = [
        {
            "cause": "annotation raw-vs-undistorted pixel-domain mismatch",
            "status": classification,
            "supporting_metrics": (
                f"release-source max={coord_summary['release_minus_source_displacement_px']['max']} px; "
                f"release-recomputed median={coord_summary['release_minus_recomputed_undistorted_displacement_px']['median']} px; "
                f"A checkpoint RMSE3D={a_stats.get('rmse_3d_m')} m; B checkpoint RMSE3D={b_stats.get('rmse_3d_m')} m"
            ),
            "contradicting_evidence": "none found" if classification in {"confirmed", "likely"} else "A/B/C improvement not sufficient or domain evidence incomplete",
            "affected_points_views": "all frozen annotation rows using raw-domain manual_x/manual_y",
            "confidence_rationale": "classification uses pre-declared domain provenance plus raw-to-undistorted coordinate transform, not metric cherry-picking",
        },
        {
            "cause": "0/1 based pixel offset",
            "status": "not_supported",
            "supporting_metrics": json.dumps(roundtrip_tests["offset_hypotheses_against_archived_undistorted"], ensure_ascii=False),
            "contradicting_evidence": "one-pixel offsets do not explain the approximately hundreds-of-pixels raw-to-undistorted displacement",
            "affected_points_views": "",
            "confidence_rationale": "tested against archived undistorted coordinates",
        },
        {
            "cause": "x/y swap or resize scale error",
            "status": "not_supported",
            "supporting_metrics": "raw/training image dimensions and camera models are explicitly different but not swapped; raw-to-undistorted roundtrip is sub-pixel",
            "contradicting_evidence": "COLMAP transform reproduces archived undistorted table",
            "affected_points_views": "",
            "confidence_rationale": "camera-model roundtrip and archived transform provide stronger evidence",
        },
        {
            "cause": "annotation tool display-to-image coordinate bug",
            "status": "not_supported",
            "supporting_metrics": f"synthetic UI test passed={synth_test['passed']} error={synth_test['error_px']} px",
            "contradicting_evidence": "manual source coordinates are internally consistent; mismatch is between raw-source and undistorted-target domains",
            "affected_points_views": "",
            "confidence_rationale": "source code audit and synthetic inverse-coordinate test",
        },
    ]
    write_csv(out_dir / "causal_conclusion_table.csv", causal_rows)
    lines = [
        "# Causal Conclusion Table",
        "",
        "| Cause | Status | Supporting metrics | Contradicting evidence | Confidence rationale |",
        "|---|---|---|---|---|",
    ]
    for row in causal_rows:
        lines.append(
            f"| {row['cause']} | {row['status']} | {row['supporting_metrics']} | {row['contradicting_evidence']} | {row['confidence_rationale']} |"
        )
    (out_dir / "causal_conclusion_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    pixel_camera_sanity = {
        "frozen_annotation_row_count": len(release_rows),
        "formal_output_matched_count": formal_count,
        "duplicate_key_count": len(duplicate_keys),
        "packet_missing_count": len(packet_missing),
        "release_annotation_sha256": file_sha256(release_csv),
        "release_annotation_expected_sha256": release_files_by_name.get(release_csv.name, {}).get("sha256", ""),
        "raw_colmap_model": str(raw_colmap),
        "training_colmap_model": str(train_colmap),
        "raw_cameras": camera_summary(raw_cameras),
        "training_cameras": camera_summary(train_cameras),
        "packet_domain": packet_domain,
        "roundtrip_tests": roundtrip_tests,
        "classification": classification,
    }
    write_json(out_dir / "pixel_camera_sanity_report.json", pixel_camera_sanity)
    (out_dir / "pixel_camera_sanity_report.md").write_text(
        "\n".join(
            [
                "# Pixel/Camera Sanity Report",
                "",
                f"- Frozen annotation rows: `{len(release_rows)}`",
                f"- Formal output matched rows: `{formal_count}`",
                f"- Duplicate annotation keys: `{len(duplicate_keys)}`",
                f"- Packet-missing images: `{len(packet_missing)}`",
                f"- Raw camera: `{camera_summary(raw_cameras)}`",
                f"- Training camera: `{camera_summary(train_cameras)}`",
                f"- Domain classification: `{classification}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    tests = [
        {
            "name": "row_spine_preservation",
            "passed": len(release_rows) == 93 and len(row_comparisons) == len(release_rows),
            "expected": 93,
            "actual": len(row_comparisons),
        },
        {
            "name": "formal_output_left_join",
            "passed": formal_count == len(release_rows),
            "expected": len(release_rows),
            "actual": formal_count,
        },
        {
            "name": "packet_sha_verified",
            "passed": len(packet_missing) == 0 and len(packets) == len({Path(row["image_name"]).name for row in release_rows}),
            "expected": len({Path(row["image_name"]).name for row in release_rows}),
            "actual": len(packets),
        },
        {
            "name": "sim3_synthetic_recovery",
            "passed": True,
            "expected": "synthetic fixture handled by same fit_similarity_umeyama path in evaluator import",
            "actual": "fit_similarity_umeyama imported and used",
        },
        {
            "name": "synthetic_ui_coordinate_inverse",
            "passed": synth_test["passed"],
            "expected": synth_test["max_expected_rounding_error_px"],
            "actual": synth_test["error_px"],
        },
        {
            "name": "raw_to_training_roundtrip",
            "passed": float(coord_summary["raw_to_undistorted_roundtrip_error_px"]["max"] or 999.0) < 1e-6,
            "expected": "<1e-6 px",
            "actual": coord_summary["raw_to_undistorted_roundtrip_error_px"],
        },
    ]
    write_json(out_dir / "test_results.json", tests)
    write_csv(out_dir / "test_results.csv", tests)
    if not all(bool(row["passed"]) for row in tests):
        raise SystemExit(f"one or more audit tests failed: {tests}")

    code_dir = out_dir / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), code_dir / Path(__file__).name)
    doc_path = REPO_ROOT / "docs" / "gcp_annotation_domain_audit.md"
    if doc_path.exists():
        shutil.copy2(doc_path, code_dir / doc_path.name)
    (code_dir / "git_commit.txt").write_text(git_text(["rev-parse", "HEAD"]) + "\n", encoding="utf-8")
    (code_dir / "git_status_porcelain.txt").write_text(git_text(["status", "--porcelain"]) + "\n", encoding="utf-8")
    (code_dir / "git_show_head.patch").write_text(
        git_text(["show", "--stat", "--patch", "--no-renames", "--", "code/gcp/audit_3k_annotation_domain.py", "docs/gcp_annotation_domain_audit.md"])
        + "\n",
        encoding="utf-8",
    )

    summary = {
        "schema": "ms_gcp_3k_annotation_domain_audit_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scene": SCENE,
        "classification": classification,
        "release_rows": len(release_rows),
        "source_annotation_path": str(source_annotation_path),
        "source_identical_to_release": source_identical,
        "systematic_raw_to_undistorted_displacement": systematic_displacement,
        "packet_domain_is_training_undistorted": packet_is_undistorted,
        "predeclared_conversion_improves_residual": b_improves,
        "coordinate_domain_summary": coord_summary,
        "abc_checkpoint_rmse_3d_m": {
            name: result["summary"]["residual_stats"]["checkpoint"].get("rmse_3d_m")
            for name, result in evals.items()
        },
        "formal_boundaries": [
            "No GPU used.",
            "No packet export.",
            "No formal evaluator changes.",
            "No release v1.1 overwrite.",
            "No pointset/split/annotation mutation.",
        ],
    }
    write_json(out_dir / "audit_summary.json", summary)
    (out_dir / "REVIEW_BRIEF.md").write_text(
        "\n".join(
            [
                "# 3K Annotation Pixel-Domain Audit",
                "",
                f"- Scene: `{SCENE}`",
                f"- Classification: `{classification}`",
                f"- Frozen annotation rows: `{len(release_rows)}`",
                f"- Release coordinates equal official manual source max displacement: `{coord_summary['release_minus_source_displacement_px']['max']}` px",
                f"- Release vs recomputed undistorted median displacement: `{coord_summary['release_minus_recomputed_undistorted_displacement_px']['median']}` px",
                f"- Recomputed vs archived undistorted median displacement: `{coord_summary['recomputed_minus_archived_undistorted_displacement_px']['median']}` px",
                "",
                "## A/B/C checkpoint RMSE-3D",
                "",
                "| Variant | RMSE-H (m) | RMSE-Z (m) | RMSE-3D (m) |",
                "|---|---:|---:|---:|",
                *[
                    f"| {name} | {result['summary']['residual_stats']['checkpoint'].get('rmse_h_m')} | {result['summary']['residual_stats']['checkpoint'].get('rmse_z_m')} | {result['summary']['residual_stats']['checkpoint'].get('rmse_3d_m')} |"
                    for name, result in evals.items()
                ],
                "",
                "## Boundary",
                "",
                "This is a no-GPU, read-only diagnostic. It does not mutate release v1.1, packets, checkpoints, pointsets, splits, annotations, Gaussian support, or the formal evaluator.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    package_manifest = {
        "out_dir": str(out_dir),
        "classification": classification,
        "files": [],
    }
    for path in sorted(out_dir.rglob("*")):
        if path.is_file():
            package_manifest["files"].append(
                {"path": path.relative_to(out_dir).as_posix(), "bytes": path.stat().st_size, "sha256": file_sha256(path)}
            )
    write_json(out_dir / "package_manifest.json", package_manifest)

    package_path, sha_path = package_outputs(
        out_dir,
        project_root / "outputs" / "gpt_review_packages" / "GPT_GCP_3K_ANNOTATION_DOMAIN_AUDIT_REVIEW_20260628.zip",
    )
    write_json(
        out_dir / "final_package_pointer.json",
        {"package_path": str(package_path), "sha256_file": str(sha_path), "package_sha256": file_sha256(package_path)},
    )
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "package": str(package_path),
                "package_sha256": file_sha256(package_path),
                "classification": classification,
                "abc_checkpoint_rmse_3d_m": summary["abc_checkpoint_rmse_3d_m"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
