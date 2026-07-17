#!/usr/bin/env python3
"""Rebuild v1.3 annotation candidates after the SIMPLE_RADIAL fold fix.

The audit is deliberately residual-blind.  It uses only manual Good pixels,
raw COLMAP cameras, image metadata, and the already-reviewed geometry-only
split candidate.  It never edits an annotation or frozen release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from prepare_direct_multiview_annotation_tasks import (
    CANDIDATE_FIELDS,
    NADIR_MAX_OFF_NADIR_DEG,
    write_candidate_csv,
)
from prepare_followup_annotation_tasks_50k100k import (
    annotation_ray,
    qvec_to_rotation,
    triangulate_annotation_rays,
    visible_good,
)
from triangulate_gcp_points import simple_radial_principal_branch_is_valid


SCENES = [
    "gcp_3000_20260602",
    "gcp_5000_20260602",
    "gcp_10000_20260610",
    "gcp_20000_20260602",
    "gcp_50000_20260610",
    "gcp_100000_20260610",
]

ANNOTATION_RELATIVE_PATHS = {
    "gcp_3000_20260602": "outputs/gcp_multiview_direct_annotation_tasks_20260713_annotate_direct_v2/working_annotations/gcp_3000_20260602_manual_annotations_v1_3_draft_working.csv",
    "gcp_5000_20260602": "outputs/gcp_map_defined_core_annotation_tasks_20260714_map_core_G04_G07_G09_v1/working_annotations/gcp_5000_20260602_map_core_v1_3_draft_working.csv",
    "gcp_10000_20260610": "outputs/gcp_multiview_direct_annotation_tasks_20260713_annotate_direct_v2/working_annotations/gcp_10000_20260610_manual_annotations_v1_3_draft_working.csv",
    "gcp_20000_20260602": "outputs/gcp_multiview_direct_annotation_tasks_20260713_annotate_direct_v2/working_annotations/gcp_20000_20260602_manual_annotations_v1_3_draft_working.csv",
    "gcp_50000_20260610": "outputs/gcp_followup_annotation_tasks_50k100k_20260715_followup_after_map_core_v1/working_annotations/gcp_50000_20260610_followup_v1_3_draft_working.csv",
    "gcp_100000_20260610": "outputs/gcp_followup_annotation_tasks_50k100k_20260715_followup_after_map_core_v1/working_annotations/gcp_100000_20260610_followup_v1_3_draft_working.csv",
}

FOCUS_POINTS = {
    ("gcp_20000_20260602", "G36"),
    ("gcp_50000_20260610", "dyl2"),
    ("gcp_100000_20260610", "G33"),
    ("gcp_100000_20260610", "dyl2"),
}

ANNOTATION_DOMAIN = "raw_dji_decoded_pixel_matrix_ignore_exif_orientation"
COORDINATE_DOMAIN = "raw_image_zero_based_pixel_centers"
EDGE_PREFERENCE_PX = 96.0
SUPPLEMENT_CANDIDATES_PER_POINT = 12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def git_output(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def camera_indices(scene_record: dict[str, Any]) -> tuple[dict[int, dict[str, Any]], dict[str, dict[str, Any]]]:
    model = scene_record["raw_model"]
    cameras = {int(row["camera_id"]): row for row in model["cameras"]}
    images = {str(row["image_name"]): row for row in model["images"]}
    return cameras, images


def camera_center(image: dict[str, Any]) -> np.ndarray:
    rotation = qvec_to_rotation(image["qvec"])
    translation = np.asarray(image["tvec"], dtype=np.float64)
    return -rotation.T @ translation


def project_point(
    xyz: np.ndarray,
    image: dict[str, Any],
    camera: dict[str, Any],
) -> tuple[float, float, float] | None:
    if str(camera["model"]).upper() != "SIMPLE_RADIAL" or len(camera["params"]) != 4:
        raise RuntimeError(f"Unsupported raw camera: {camera}")
    focal, cx, cy, radial = (float(value) for value in camera["params"])
    rotation = qvec_to_rotation(image["qvec"])
    translation = np.asarray(image["tvec"], dtype=np.float64)
    camera_xyz = rotation @ xyz + translation
    z = float(camera_xyz[2])
    if not math.isfinite(z) or z <= 0:
        return None
    x = float(camera_xyz[0] / z)
    y = float(camera_xyz[1] / z)
    if not simple_radial_principal_branch_is_valid(x, y, radial):
        return None
    scale = 1.0 + radial * (x * x + y * y)
    u = focal * x * scale + cx
    v = focal * y * scale + cy
    if not all(math.isfinite(value) for value in (u, v)):
        return None
    return u, v, z


def image_metadata(candidate_root: Path, scene: str) -> dict[str, dict[str, Any]]:
    frame = pd.read_csv(candidate_root / scene / "image_metadata.csv", dtype=str, keep_default_na=False)
    if frame["image_name"].duplicated().any():
        raise RuntimeError(f"{scene}: duplicate image metadata")
    return {str(row["image_name"]): row for row in frame.to_dict("records")}


def is_nadir(metadata: dict[str, Any] | None) -> bool:
    if not metadata:
        return False
    return abs(float(metadata["pitch_deg"]) + 90.0) <= NADIR_MAX_OFF_NADIR_DEG


def azimuth_deg(point_xyz: np.ndarray, image: dict[str, Any]) -> float:
    delta = camera_center(image)[:2] - point_xyz[:2]
    return (math.degrees(math.atan2(float(delta[0]), float(delta[1]))) + 360.0) % 360.0


def azimuth_bin(value: float) -> int:
    return int(((float(value) + 22.5) % 360.0) // 45.0)


def good_rows(frame: pd.DataFrame, point_name: str) -> list[dict[str, Any]]:
    rows = frame[frame["point_name"].eq(point_name)].to_dict("records")
    return [row for row in rows if visible_good(row)]


def reprojection_errors(
    xyz: np.ndarray,
    rows: list[dict[str, Any]],
    cameras: dict[int, dict[str, Any]],
    images: dict[str, dict[str, Any]],
) -> list[float]:
    errors: list[float] = []
    for row in rows:
        image = images[str(row["image_name"])]
        camera = cameras[int(image["camera_id"])]
        projected = project_point(xyz, image, camera)
        if projected is None:
            errors.append(float("inf"))
            continue
        errors.append(
            float(
                np.linalg.norm(
                    np.asarray(projected[:2], dtype=np.float64)
                    - np.asarray([float(row["manual_x"]), float(row["manual_y"])])
                )
            )
        )
    return errors


def leave_one_out_rows(
    scene: str,
    point_name: str,
    rows: list[dict[str, Any]],
    cameras: dict[int, dict[str, Any]],
    images: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, hidden in enumerate(rows):
        rest = rows[:index] + rows[index + 1 :]
        if len(rest) < 2:
            output.append(
                {
                    "scene": scene,
                    "point_name": point_name,
                    "hidden_image_name": hidden["image_name"],
                    "status": "not_testable_fewer_than_two_remaining_good_views",
                    "pixel_error": np.nan,
                    "recalled_in_image": False,
                }
            )
            continue
        try:
            xyz, condition = triangulate_annotation_rays(rest, cameras, images)
            image = images[str(hidden["image_name"])]
            camera = cameras[int(image["camera_id"])]
            projected = project_point(xyz, image, camera)
            if projected is None:
                raise RuntimeError("hidden view projection invalid")
            u, v, _ = projected
            error = float(
                np.linalg.norm(
                    np.asarray([u, v])
                    - np.asarray([float(hidden["manual_x"]), float(hidden["manual_y"])])
                )
            )
            inside = 0 <= u < int(camera["width"]) and 0 <= v < int(camera["height"])
            output.append(
                {
                    "scene": scene,
                    "point_name": point_name,
                    "hidden_image_name": hidden["image_name"],
                    "status": "recalled" if inside else "projected_outside_image",
                    "pixel_error": error,
                    "recalled_in_image": bool(inside),
                    "triangulation_condition_number": condition,
                    "predicted_x": u,
                    "predicted_y": v,
                    "manual_x": float(hidden["manual_x"]),
                    "manual_y": float(hidden["manual_y"]),
                }
            )
        except Exception as exc:
            output.append(
                {
                    "scene": scene,
                    "point_name": point_name,
                    "hidden_image_name": hidden["image_name"],
                    "status": f"failed:{exc}",
                    "pixel_error": np.nan,
                    "recalled_in_image": False,
                }
            )
    return output


def candidate_pool_for_point(
    scene: str,
    point_name: str,
    frame: pd.DataFrame,
    cameras: dict[int, dict[str, Any]],
    images: dict[str, dict[str, Any]],
    metadata_by_name: dict[str, dict[str, Any]],
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows = good_rows(frame, point_name)
    mapped = [row for row in rows if str(row["image_name"]) in images]
    if len(mapped) < 2:
        raise RuntimeError(f"{scene}/{point_name}: fewer than two mapped Good observations")
    xyz, condition = triangulate_annotation_rays(mapped, cameras, images)
    errors = reprojection_errors(xyz, mapped, cameras, images)
    loo = leave_one_out_rows(scene, point_name, mapped, cameras, images)
    attempted = set(frame.loc[frame["point_name"].eq(point_name), "image_name"].astype(str))
    good_names = {str(row["image_name"]) for row in mapped}
    current_bins = {
        azimuth_bin(azimuth_deg(xyz, images[name]))
        for name in good_names
    }
    current_types = {
        "nadir" if is_nadir(metadata_by_name.get(name)) else "oblique"
        for name in good_names
    }
    candidates: list[dict[str, Any]] = []
    for image_name, image in images.items():
        camera = cameras[int(image["camera_id"])]
        projected = project_point(xyz, image, camera)
        if projected is None:
            continue
        u, v, z = projected
        width = int(camera["width"])
        height = int(camera["height"])
        if not (0 <= u < width and 0 <= v < height):
            continue
        metadata = metadata_by_name.get(image_name)
        view_type = "nadir" if is_nadir(metadata) else "oblique"
        azimuth = azimuth_deg(xyz, image)
        edge = min(u, v, width - 1 - u, height - 1 - v)
        center_score = max(
            0.0,
            1.0
            - max(
                abs(u - width / 2.0) / (width / 2.0),
                abs(v - height / 2.0) / (height / 2.0),
            ),
        )
        candidates.append(
            {
                "scene": scene,
                "point_name": point_name,
                "image_name": image_name,
                "image_path": str((metadata or {}).get("image_path", "")),
                "pixel_x": u,
                "pixel_y": v,
                "center_score": center_score,
                "inside_image": True,
                "edge_margin_px": edge,
                "projection_uncertainty_px": max(60.0, float(np.percentile(errors, 95)) * 3.0),
                "candidate_source": "fixed_principal_branch_triangulated_annotation_rays_v2",
                "view_type": view_type,
                "camera_azimuth_deg": azimuth,
                "azimuth_bin_45deg": azimuth_bin(azimuth),
                "off_nadir_deg": abs(float((metadata or {}).get("pitch_deg", -45.0)) + 90.0),
                "image_width": width,
                "image_height": height,
                "task_action": "annotate_visibility_during_labeling",
                "annotation_image_domain": ANNOTATION_DOMAIN,
                "annotation_coordinate_domain": COORDINATE_DOMAIN,
                "already_attempted": image_name in attempted,
                "already_good": image_name in good_names,
                "camera_id": int(image["camera_id"]),
                "image_id": int(image["image_id"]),
                "image_pose_record_sha256": image["record_sha256"],
                "camera_record_sha256": camera["record_sha256"],
                "camera_z_model_units": z,
                "adds_new_azimuth_bin": azimuth_bin(azimuth) not in current_bins,
                "adds_missing_view_type": view_type not in current_types,
            }
        )
    summary = {
        "scene": scene,
        "point_name": point_name,
        "good_view_count": len(mapped),
        "good_nadir_count": sum(is_nadir(metadata_by_name.get(str(row["image_name"]))) for row in mapped),
        "good_oblique_count": sum(not is_nadir(metadata_by_name.get(str(row["image_name"]))) for row in mapped),
        "good_azimuth_bin_count": len(current_bins),
        "triangulation_condition_number": condition,
        "all_view_reprojection_median_px": float(np.median(errors)),
        "all_view_reprojection_p95_px": float(np.percentile(errors, 95)),
        "all_view_reprojection_max_px": float(np.max(errors)),
        "corrected_in_bounds_image_count": len(candidates),
        "unattempted_corrected_candidate_count": sum(not row["already_attempted"] for row in candidates),
        "unattempted_nadir_count": sum(
            not row["already_attempted"] and row["view_type"] == "nadir" for row in candidates
        ),
        "unattempted_oblique_count": sum(
            not row["already_attempted"] and row["view_type"] == "oblique" for row in candidates
        ),
    }
    return xyz, candidates, loo, summary


def select_supplemental(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    available = [row for row in rows if not row["already_attempted"]]
    available.sort(
        key=lambda row: (
            not bool(row["adds_missing_view_type"]),
            not bool(row["adds_new_azimuth_bin"]),
            float(row["edge_margin_px"]) < EDGE_PREFERENCE_PX,
            -float(row["center_score"]),
            -float(row["edge_margin_px"]),
            str(row["image_name"]),
        )
    )
    selected: list[dict[str, Any]] = []
    used_bins: defaultdict[int, int] = defaultdict(int)
    while available and len(selected) < count:
        best_index = min(
            range(len(available)),
            key=lambda index: (
                used_bins[int(available[index]["azimuth_bin_45deg"])],
                not bool(available[index]["adds_missing_view_type"]),
                not bool(available[index]["adds_new_azimuth_bin"]),
                -float(available[index]["center_score"]),
                str(available[index]["image_name"]),
            ),
        )
        row = available.pop(best_index)
        selected.append(row)
        used_bins[int(row["azimuth_bin_45deg"])] += 1
    output = []
    for rank, row in enumerate(selected, start=1):
        item = dict(row)
        item["rank_for_gcp"] = rank
        output.append(item)
    return output


def old_bug_summary(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame["principal_branch_valid"] = frame["principal_branch_valid"].str.lower().eq("true")
    return (
        frame.groupby(["scene", "point_name"], as_index=False)
        .agg(
            old_selected_candidate_count=("image_name", "size"),
            old_valid_candidate_count=("principal_branch_valid", "sum"),
        )
        .assign(
            old_false_fold_candidate_count=lambda df: df["old_selected_candidate_count"]
            - df["old_valid_candidate_count"],
            old_false_fold_fraction=lambda df: df["old_false_fold_candidate_count"]
            / df["old_selected_candidate_count"],
        )
    )


def geodetic_to_ecef(lat_deg: float, lon_deg: float, height: float) -> np.ndarray:
    a = 6378137.0
    e2 = 6.69437999014e-3
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    radius = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
    return np.asarray(
        [
            (radius + height) * cos_lat * math.cos(lon),
            (radius + height) * cos_lat * math.sin(lon),
            (radius * (1.0 - e2) + height) * sin_lat,
        ],
        dtype=np.float64,
    )


def geodetic_to_enu(
    lat: float,
    lon: float,
    height: float,
    origin: Iterable[float],
) -> np.ndarray:
    lat0, lon0, h0 = (float(value) for value in origin)
    delta = geodetic_to_ecef(lat, lon, height) - geodetic_to_ecef(lat0, lon0, h0)
    phi = math.radians(lat0)
    lam = math.radians(lon0)
    transform = np.asarray(
        [
            [-math.sin(lam), math.cos(lam), 0.0],
            [-math.sin(phi) * math.cos(lam), -math.sin(phi) * math.sin(lam), math.cos(phi)],
            [math.cos(phi) * math.cos(lam), math.cos(phi) * math.sin(lam), math.sin(phi)],
        ],
        dtype=np.float64,
    )
    return transform @ delta


def g39_gps_audit(
    annotation: pd.DataFrame,
    cameras: dict[int, dict[str, Any]],
    images: dict[str, dict[str, Any]],
    metadata_by_name: dict[str, dict[str, Any]],
    alignment_summary: dict[str, Any],
    loo: pd.DataFrame,
) -> pd.DataFrame:
    origin = alignment_summary["enu_origin_lat_lon_alt"]
    rows = []
    for row in good_rows(annotation, "G39"):
        name = str(row["image_name"])
        image = images[name]
        metadata = metadata_by_name[name]
        gps_enu = geodetic_to_enu(
            float(metadata["lat"]),
            float(metadata["lon"]),
            float(metadata["ellipsoid_alt_m"]),
            origin,
        )
        center = camera_center(image)
        delta = center - gps_enu
        match = loo[loo["hidden_image_name"].eq(name)]
        rows.append(
            {
                "scene": "gcp_50000_20260610",
                "point_name": "G39",
                "image_name": name,
                "manual_x": float(row["manual_x"]),
                "manual_y": float(row["manual_y"]),
                "gps_lat": float(metadata["lat"]),
                "gps_lon": float(metadata["lon"]),
                "gps_ellipsoid_alt_m": float(metadata["ellipsoid_alt_m"]),
                "gps_to_aligned_colmap_e_m": float(delta[0]),
                "gps_to_aligned_colmap_n_m": float(delta[1]),
                "gps_to_aligned_colmap_u_m": float(delta[2]),
                "gps_to_aligned_colmap_3d_m": float(np.linalg.norm(delta)),
                "gcp_leave_one_out_pixel_error": float(match.iloc[0]["pixel_error"]) if len(match) else np.nan,
                "image_pose_record_sha256": image["record_sha256"],
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["gps_residual_rank_desc"] = result["gps_to_aligned_colmap_3d_m"].rank(
            method="min", ascending=False
        ).astype(int)
    return result


def worklist_decision(summary: dict[str, Any]) -> tuple[str, str]:
    key = (str(summary["scene"]), str(summary["point_name"]))
    if key == ("gcp_20000_20260602", "G36"):
        return "supplement_required", "old_candidate_fold_bug_and_only_5_good_views"
    if key == ("gcp_50000_20260610", "dyl2"):
        if int(summary["unattempted_nadir_count"]) > 0:
            return "supplement_required", "3_good_oblique_views_precise_nadir_candidates_now_available"
        return "diagnostic_only", "3_good_views_and_no_precise_nadir_candidate"
    if key == ("gcp_100000_20260610", "G33"):
        if int(summary["good_nadir_count"]) == 0 and int(summary["unattempted_nadir_count"]) == 0:
            return "exclude_from_future_formal_primary", "no_nadir_coverage_after_fixed_candidate_recall"
        return "supplement_review", "3_good_views_candidate_bug_may_have_omitted_valid_views"
    if key == ("gcp_100000_20260610", "dyl2"):
        return "supplement_required", "formal_checkpoint_has_only_4_good_views"
    return "no_targeted_action", "coverage_sufficient_or_not_a_focus_point"


def write_launcher(path: Path, repo: Path, candidate_path: Path, out_csv: Path, image_root: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "$ErrorActionPreference = 'Stop'",
                f"Set-Location '{repo}'",
                "python code\\gcp\\manual_gcp_annotator.py `",
                f"  --candidates_csv '{candidate_path}' `",
                f"  --out_csv '{out_csv}' `",
                f"  --image_root '{image_root}' `",
                "  --crop_size 720 `",
                "  --display_size 900 `",
                "  --annotator user",
                "",
            ]
        ),
        encoding="utf-8-sig",
    )


def candidate_csv_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{field: row.get(field, "") for field in CANDIDATE_FIELDS} for row in rows]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--remote_manifest",
        type=Path,
        default=Path(
            r"E:\M3M-GCP-3DGS\outputs\gcp_6scene_annotation_domain_inputs_20260628"
            r"\gcp_6scene_annotation_domain_jsonlight_20260628\remote_light_manifest.json"
        ),
    )
    parser.add_argument(
        "--candidate_root",
        type=Path,
        default=Path(r"E:\M3M-GCP-3DGS\outputs\gcp_annotation_candidates_20260617_all"),
    )
    parser.add_argument(
        "--split_candidate",
        type=Path,
        default=Path(
            r"E:\M3M-GCP-3DGS\outputs\gcp_v13_geometry_only_split_candidate_20260716_190827"
            r"\gcp_control_checkpoint_split_v1_3_candidate.csv"
        ),
    )
    parser.add_argument(
        "--old_bug_audit",
        type=Path,
        default=Path(
            r"E:\M3M-GCP-3DGS\outputs\gcp_candidate_projection_radial_fold_audit_20260716"
            r"\all_selected_triangulated_candidate_branch_audit.csv"
        ),
    )
    parser.add_argument("--stamp", default=time.strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    generator_repo = Path(__file__).resolve().parents[2]
    if git_output(generator_repo, "status", "--porcelain"):
        raise RuntimeError("Candidate recall audit requires a clean generator worktree")
    output_root = args.repo / "outputs" / f"gcp_v13_fixed_candidate_recall_audit_{args.stamp}"
    output_root.mkdir(parents=True, exist_ok=False)
    candidate_dir = output_root / "candidate_lists"
    launcher_dir = output_root / "launchers"
    candidate_dir.mkdir()
    launcher_dir.mkdir()

    manifest = json.loads(args.remote_manifest.read_text(encoding="utf-8"))
    split = pd.read_csv(args.split_candidate, dtype=str, keep_default_na=False)
    formal_keys = set(zip(split["scene"], split["point_name"]))
    old_bug = old_bug_summary(args.old_bug_audit)
    old_bug_map = {
        (row.scene, row.point_name): row._asdict() for row in old_bug.itertuples(index=False)
    }

    annotations: dict[str, pd.DataFrame] = {}
    camera_sets: dict[str, tuple[dict[int, dict[str, Any]], dict[str, dict[str, Any]]]] = {}
    metadata_sets: dict[str, dict[str, dict[str, Any]]] = {}
    input_records = []
    for scene in SCENES:
        path = args.repo / ANNOTATION_RELATIVE_PATHS[scene]
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        if frame[["scene", "point_name", "image_name"]].duplicated().any():
            raise RuntimeError(f"{scene}: duplicate latest annotation keys")
        annotations[scene] = frame
        camera_sets[scene] = camera_indices(manifest["scenes"][scene])
        metadata_sets[scene] = image_metadata(args.candidate_root, scene)
        input_records.append({"scene": scene, "path": str(path), "sha256": sha256_file(path)})

    all_pool: list[dict[str, Any]] = []
    all_loo: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    xyz_by_key: dict[tuple[str, str], np.ndarray] = {}
    points_to_audit = sorted(
        {
            (str(row.scene), str(row.point_name))
            for row in split.itertuples(index=False)
        }
        | FOCUS_POINTS
    )
    for scene, point_name in points_to_audit:
        frame = annotations[scene]
        cameras, images = camera_sets[scene]
        try:
            xyz, pool, loo, summary = candidate_pool_for_point(
                scene,
                point_name,
                frame,
                cameras,
                images,
                metadata_sets[scene],
            )
            xyz_by_key[(scene, point_name)] = xyz
            all_pool.extend(pool)
            all_loo.extend(loo)
            old = old_bug_map.get((scene, point_name), {})
            summary.update(
                {
                    "is_formal_split_candidate": (scene, point_name) in formal_keys,
                    "old_selected_candidate_count": int(old.get("old_selected_candidate_count", 0)),
                    "old_false_fold_candidate_count": int(old.get("old_false_fold_candidate_count", 0)),
                    "old_false_fold_fraction": float(old.get("old_false_fold_fraction", 0.0)),
                }
            )
            decision, reason = worklist_decision(summary)
            summary["candidate_recall_decision"] = decision
            summary["decision_reason"] = reason
            summaries.append(summary)
        except Exception as exc:
            summaries.append(
                {
                    "scene": scene,
                    "point_name": point_name,
                    "is_formal_split_candidate": (scene, point_name) in formal_keys,
                    "candidate_recall_decision": "blocked",
                    "decision_reason": str(exc),
                }
            )

    pool_df = pd.DataFrame(all_pool)
    loo_df = pd.DataFrame(all_loo)
    summary_df = pd.DataFrame(summaries)
    if any(summary_df["candidate_recall_decision"].eq("blocked")):
        blocked = summary_df[summary_df["candidate_recall_decision"].eq("blocked")]
        raise RuntimeError(f"Candidate recall blocked:\n{blocked.to_string(index=False)}")

    selected_all: list[dict[str, Any]] = []
    worklist_rows = []
    for row in summary_df.itertuples(index=False):
        key = (str(row.scene), str(row.point_name))
        decision = str(row.candidate_recall_decision)
        group = [candidate for candidate in all_pool if (candidate["scene"], candidate["point_name"]) == key]
        selected: list[dict[str, Any]] = []
        if decision in {"supplement_required", "supplement_review"}:
            selected = select_supplemental(group, SUPPLEMENT_CANDIDATES_PER_POINT)
            selected_all.extend(selected)
        worklist_rows.append(
            {
                "scene": key[0],
                "point_name": key[1],
                "decision": decision,
                "decision_reason": str(row.decision_reason),
                "current_good_views": int(row.good_view_count),
                "current_nadir_views": int(row.good_nadir_count),
                "current_oblique_views": int(row.good_oblique_count),
                "unattempted_corrected_candidates": int(row.unattempted_corrected_candidate_count),
                "unattempted_nadir_candidates": int(row.unattempted_nadir_count),
                "unattempted_oblique_candidates": int(row.unattempted_oblique_count),
                "selected_supplemental_rows": len(selected),
                "formal_split_candidate": bool(row.is_formal_split_candidate),
            }
        )

    selected_df = pd.DataFrame(selected_all)
    worklist_df = pd.DataFrame(worklist_rows)

    per_scene_launchers = []
    for scene, group in selected_df.groupby("scene", sort=True) if not selected_df.empty else []:
        rows = group.sort_values(["point_name", "rank_for_gcp", "image_name"]).to_dict("records")
        candidate_path = candidate_dir / f"{scene}_fixed_candidate_recall_supplement.csv"
        write_candidate_csv(candidate_path, candidate_csv_rows(rows))
        annotation_path = args.repo / ANNOTATION_RELATIVE_PATHS[scene]
        launcher_path = launcher_dir / f"launch_{scene}_fixed_candidate_supplement.ps1"
        write_launcher(
            launcher_path,
            generator_repo,
            candidate_path,
            annotation_path,
            Path(r"E:\datasets\M3M-GCP\scenes") / scene,
        )
        per_scene_launchers.append(
            {
                "scene": scene,
                "candidate_csv": str(candidate_path),
                "launcher": str(launcher_path),
                "annotation_output": str(annotation_path),
                "rows": len(rows),
            }
        )

    # G33 is retained as a complete diagnostic candidate pool, but is not opened
    # automatically when fixed recall still shows no nadir coverage.
    g33 = pool_df[
        pool_df["scene"].eq("gcp_100000_20260610")
        & pool_df["point_name"].eq("G33")
        & ~pool_df["already_attempted"]
    ]
    if not g33.empty:
        g33.to_csv(candidate_dir / "gcp_100000_20260610_G33_fixed_full_diagnostic_pool.csv", index=False, encoding="utf-8-sig")

    g39_loo = loo_df[
        loo_df["scene"].eq("gcp_50000_20260610") & loo_df["point_name"].eq("G39")
    ]
    alignment_path = (
        args.remote_manifest.parent
        / "models"
        / "gcp_50000_20260610"
        / "raw_model"
        / "georegistration_alignment_summary.json"
    )
    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    g39_gps = g39_gps_audit(
        annotations["gcp_50000_20260610"],
        *camera_sets["gcp_50000_20260610"],
        metadata_sets["gcp_50000_20260610"],
        alignment,
        g39_loo,
    )

    summary_df.to_csv(output_root / "six_scene_candidate_recall_summary.csv", index=False, encoding="utf-8-sig")
    pool_df.to_csv(output_root / "fixed_corrected_candidate_pool.csv", index=False, encoding="utf-8-sig")
    loo_df.to_csv(output_root / "leave_one_good_view_out_recall.csv", index=False, encoding="utf-8-sig")
    old_bug.to_csv(output_root / "old_radial_fold_bug_impact.csv", index=False, encoding="utf-8-sig")
    worklist_df.to_csv(output_root / "supplemental_annotation_decisions.csv", index=False, encoding="utf-8-sig")
    if not selected_df.empty:
        selected_df.to_csv(output_root / "selected_supplemental_candidates.csv", index=False, encoding="utf-8-sig")
    g39_gps.to_csv(output_root / "G39_gps_colmap_annotation_audit.csv", index=False, encoding="utf-8-sig")

    g39_0002 = g39_gps[g39_gps["image_name"].eq("DJI_20260610161948_0002_D.JPG")]
    g39_norm = float(g39_0002.iloc[0]["gps_to_aligned_colmap_3d_m"]) if len(g39_0002) else float("nan")
    g39_pixel = float(g39_0002.iloc[0]["gcp_leave_one_out_pixel_error"]) if len(g39_0002) else float("nan")
    readme = f"""# v1.3 修正候选召回与 G39 GPS 审计

## 边界

- 使用最新工作标注和已修正的 SIMPLE_RADIAL principal-branch 投影。
- 不修改 v1.2.2、现有标注、split、packet 或 evaluator。
- 不读取 3DGS residual；候选选择不使用模型误差。

## 候选 bug 结论

- 旧投影 bug 会让非可逆畸变分支上的射线折回画面并占用候选名额，但审计确认这些假候选没有形成 Good 标注。
- `six_scene_candidate_recall_summary.csv` 给出所有未来 formal 候选点的修正召回与旧 bug 影响。
- `supplemental_annotation_decisions.csv` 是是否补标的正式工作结论；只有 `supplement_required` / `supplement_review` 生成启动器。
- 100K G33 若修正召回后仍无正射覆盖，保留 diagnostic，不进入 future formal primary。

## G39

- `DJI_20260610161948_0002_D.JPG` 的 EXIF GPS 与对齐后 COLMAP camera center 相差 {g39_norm:.6f} m；G39 leave-one-view-out 像素差为 {g39_pixel:.6f} px。
- 这证明该图的机载 GPS 是异常值，但不能单凭相关性证明 24 px 全由 GPS 导致。COLMAP 位姿经过视觉匹配与 BA 优化，并非直接复制 GPS。
- 人工标注保存的是 raw image pixel；GPS 漂移不会改写用户点击。GPS 只会影响粗候选准心，或通过 SfM 初始化间接影响最终相机位姿。

## 用户操作

仅运行 `launchers` 中存在的脚本。候选图中真实可见就标 Good，不可见标 Not visible，不要为了达到数量猜测位置。
"""
    (output_root / "README_zh.md").write_text(readme, encoding="utf-8")

    run_manifest = {
        "schema": "ms_gcp_v13_fixed_candidate_recall_audit_v1",
        "status": "candidate_worklist_not_release",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "generator": {
            "repo": str(generator_repo),
            "commit": git_output(generator_repo, "rev-parse", "HEAD"),
            "branch": git_output(generator_repo, "branch", "--show-current"),
            "clean": True,
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "inputs": {
            "annotations": input_records,
            "remote_light_manifest": {"path": str(args.remote_manifest), "sha256": sha256_file(args.remote_manifest)},
            "split_candidate": {"path": str(args.split_candidate), "sha256": sha256_file(args.split_candidate)},
            "old_bug_audit": {"path": str(args.old_bug_audit), "sha256": sha256_file(args.old_bug_audit)},
            "alignment_summary": {"path": str(alignment_path), "sha256": sha256_file(alignment_path)},
        },
        "user_qc_decisions": {
            "gcp_5000_20260602/G07": "reviewed_usable_20260716",
            "gcp_5000_20260602/G09": "reviewed_usable_20260716",
        },
        "launchers": per_scene_launchers,
        "selection_policy": {
            "forbidden": ["residual", "RMSE", "depth", "alpha", "variance", "3DGS scatter"],
            "principal_branch_gate": "scale>0 and 1+3*k*r^2>0",
            "supplement_candidates_per_point": SUPPLEMENT_CANDIDATES_PER_POINT,
            "manual_visibility_is_authoritative": True,
        },
    }
    write_json(output_root / "audit_manifest.json", run_manifest)

    files = sorted(path for path in output_root.rglob("*") if path.is_file())
    write_json(
        output_root / "OUTPUT_SHA256_MANIFEST.json",
        {
            "schema": "sha256_file_manifest_v1",
            "files": [
                {
                    "relative_path": path.relative_to(output_root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in files
            ],
        },
    )
    print(output_root)
    print(worklist_df.to_string(index=False))
    print(g39_gps.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
