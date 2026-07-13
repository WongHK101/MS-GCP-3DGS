#!/usr/bin/env python
"""Prepare direct multi-view annotation tasks from spatial candidate evidence.

This workflow intentionally does not perform a separate visual-feasibility pass.
Every emitted row is a spatially plausible, in-image candidate. The annotator
decides whether the GCP is good, ambiguous, or not visible while labeling.

The script is read-only with respect to dataset and release inputs. It writes a
new working task directory under ``outputs`` and byte-copies the historical
manual annotation tables there as editable working files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from build_gcp_projection_candidates import CameraMeta, load_gcps, project_gcp


REPO_ROOT = Path(__file__).resolve().parents[2]
SCENES = [
    "gcp_3000_20260602",
    "gcp_5000_20260602",
    "gcp_10000_20260610",
    "gcp_20000_20260602",
    "gcp_50000_20260610",
    "gcp_100000_20260610",
]
LOW_VIEW_POINTS = {
    "gcp_5000_20260602": {"G11", "G12", "G13", "G18", "NC94"},
    "gcp_20000_20260602": {"wy3_1"},
}
PRIMARY_NEW_VIEWS_PER_POINT = 16
MAX_OFF_NADIR_DEG = 60.0
NADIR_MAX_OFF_NADIR_DEG = 10.0

CANDIDATE_FIELDS = [
    "scene",
    "point_name",
    "image_name",
    "image_path",
    "rank_for_gcp",
    "pixel_x",
    "pixel_y",
    "center_score",
    "inside_image",
    "edge_margin_px",
    "projection_uncertainty_px",
    "candidate_source",
    "view_type",
    "camera_azimuth_deg",
    "azimuth_bin_45deg",
    "off_nadir_deg",
    "image_width",
    "image_height",
    "task_action",
    "annotation_image_domain",
    "annotation_coordinate_domain",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def finite_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def angle_deg(dx_e: float, dy_n: float) -> float:
    return (math.degrees(math.atan2(dx_e, dy_n)) + 360.0) % 360.0


def normalized_candidate_row(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row.get(field, "") for field in CANDIDATE_FIELDS}


def write_candidate_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CANDIDATE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(normalized_candidate_row(row))


def resolve_annotation_sources(release_dir: Path, repo: Path) -> dict[str, Path]:
    provenance = pd.read_csv(
        release_dir / "final_annotation_inclusion_provenance.csv",
        dtype=str,
        keep_default_na=False,
    )
    result: dict[str, Path] = {}
    for scene, group in provenance.groupby("scene", sort=True):
        raw_paths = sorted({p for p in group["source_annotation_file"] if p})
        if len(raw_paths) != 1:
            raise RuntimeError(f"{scene}: expected one source annotation file, found {raw_paths}")
        source = Path(raw_paths[0])
        if not source.is_absolute():
            source = repo / source
        if not source.is_file():
            raise FileNotFoundError(f"{scene}: source annotation file missing: {source}")
        result[str(scene)] = source.resolve()
    missing = sorted(set(SCENES) - set(result))
    if missing:
        raise RuntimeError(f"Missing source annotations for scenes: {missing}")
    return result


def read_existing_annotation_keys(path: Path) -> set[tuple[str, str, str]]:
    rows = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"scene", "point_name", "image_name"}
    if not required.issubset(rows.columns):
        raise RuntimeError(f"Missing annotation key fields in {path}")
    keys = list(zip(rows["scene"], rows["point_name"], rows["image_name"]))
    if len(keys) != len(set(keys)):
        raise RuntimeError(f"Duplicate annotation key in {path}")
    return set(keys)


def load_camera_metadata(candidate_root: Path, scene: str) -> tuple[list[CameraMeta], list[str]]:
    path = candidate_root / scene / "image_metadata.csv"
    rows = pd.read_csv(path, dtype=str, keep_default_na=False)
    cameras: list[CameraMeta] = []
    missing_images: list[str] = []
    seen_names: set[str] = set()
    for row in rows.to_dict("records"):
        image_path = Path(row["image_path"]).resolve()
        image_name = str(row["image_name"])
        if image_name in seen_names:
            raise RuntimeError(f"{scene}: duplicate image metadata name {image_name}")
        seen_names.add(image_name)
        if not image_path.is_file():
            missing_images.append(str(image_path))
            continue
        cameras.append(
            CameraMeta(
                image_path=image_path,
                image_name=image_name,
                width=int(row["width"]),
                height=int(row["height"]),
                lat=float(row["lat"]),
                lon=float(row["lon"]),
                projected_e=float(row["projected_e"]),
                projected_n=float(row["projected_n"]),
                ellipsoid_alt_m=float(row["ellipsoid_alt_m"]),
                rel_alt_m=float(row["rel_alt_m"]) if row["rel_alt_m"] else None,
                yaw_deg=float(row["yaw_deg"]),
                pitch_deg=float(row["pitch_deg"]),
                roll_deg=float(row["roll_deg"]),
                focal_px=float(row["focal_px"]),
            )
        )
    return cameras, missing_images


def build_broad_spatial_candidates(
    scene: str,
    cameras: list[CameraMeta],
    gcps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for gcp in gcps:
        for camera in cameras:
            projected = project_gcp(camera, gcp)
            if not projected or not bool(projected["inside_image"]):
                continue
            off_nadir = float(projected["off_nadir_deg"])
            if off_nadir > MAX_OFF_NADIR_DEG:
                continue
            point_to_camera_e = camera.projected_e - float(gcp["projected_e"])
            point_to_camera_n = camera.projected_n - float(gcp["projected_n"])
            azimuth = angle_deg(point_to_camera_e, point_to_camera_n)
            view_type = "nadir" if off_nadir <= NADIR_MAX_OFF_NADIR_DEG else "oblique"
            rows.append(
                {
                    "scene": scene,
                    "point_name": str(gcp["point_name"]),
                    "image_name": camera.image_name,
                    "image_path": str(camera.image_path),
                    "pixel_x": float(projected["pixel_x"]),
                    "pixel_y": float(projected["pixel_y"]),
                    "center_score": float(projected["center_score"]),
                    "inside_image": True,
                    "edge_margin_px": float(projected["edge_margin_px"]),
                    "projection_uncertainty_px": 800.0 if view_type == "nadir" else 1200.0,
                    "candidate_source": "coarse_exif_gimbal_all_orientations",
                    "view_type": view_type,
                    "camera_azimuth_deg": azimuth,
                    "azimuth_bin_45deg": int(((azimuth + 22.5) % 360.0) // 45.0),
                    "off_nadir_deg": off_nadir,
                    "image_width": int(camera.width),
                    "image_height": int(camera.height),
                    "task_action": "annotate_visibility_during_labeling",
                    "annotation_image_domain": "raw_dji_decoded_pixel_matrix_ignore_exif_orientation",
                    "annotation_coordinate_domain": "raw_image_zero_based_pixel_centers",
                }
            )
    return rows


def load_triangulated_candidates(
    audit_root: Path,
    image_lookup: dict[tuple[str, str], CameraMeta],
) -> list[dict[str, Any]]:
    matrix_path = audit_root / "point_image_candidate_coverage_matrix.csv"
    matrix = pd.read_csv(matrix_path, dtype=str, keep_default_na=False)
    matrix = matrix[
        matrix["candidate_source"].eq("triangulated_annotation_rays")
        & matrix["inside_image"].map(parse_bool)
    ]
    rows: list[dict[str, Any]] = []
    for raw in matrix.to_dict("records"):
        key = (raw["scene"], raw["image_name"])
        camera = image_lookup.get(key)
        if camera is None:
            raise RuntimeError(f"Triangulated candidate has no image metadata: {key}")
        point_e = finite_float(raw.get("cgcs2000_gk_cm108_e_m"))
        point_n = finite_float(raw.get("cgcs2000_gk_cm108_n_m"))
        azimuth = finite_float(raw.get("camera_azimuth_deg"))
        if not math.isfinite(azimuth) and math.isfinite(point_e) and math.isfinite(point_n):
            azimuth = angle_deg(camera.projected_e - point_e, camera.projected_n - point_n)
        off_nadir = abs(float(camera.pitch_deg) + 90.0)
        view_type = "nadir" if off_nadir <= NADIR_MAX_OFF_NADIR_DEG else "oblique"
        x = finite_float(raw.get("candidate_x"))
        y = finite_float(raw.get("candidate_y"))
        if not (0.0 <= x < camera.width and 0.0 <= y < camera.height):
            continue
        rows.append(
            {
                "scene": raw["scene"],
                "point_name": raw["point_name"],
                "image_name": raw["image_name"],
                "image_path": str(camera.image_path),
                "pixel_x": x,
                "pixel_y": y,
                "center_score": max(
                    0.0,
                    1.0
                    - max(
                        abs(x - camera.width / 2) / (camera.width / 2),
                        abs(y - camera.height / 2) / (camera.height / 2),
                    ),
                ),
                "inside_image": True,
                "edge_margin_px": min(x, y, camera.width - 1 - x, camera.height - 1 - y),
                "projection_uncertainty_px": finite_float(raw.get("projection_uncertainty_px"), 300.0),
                "candidate_source": "triangulated_annotation_rays",
                "view_type": view_type,
                "camera_azimuth_deg": azimuth,
                "azimuth_bin_45deg": int(((azimuth + 22.5) % 360.0) // 45.0) if math.isfinite(azimuth) else -1,
                "off_nadir_deg": off_nadir,
                "image_width": int(camera.width),
                "image_height": int(camera.height),
                "task_action": "annotate_visibility_during_labeling",
                "annotation_image_domain": "raw_dji_decoded_pixel_matrix_ignore_exif_orientation",
                "annotation_coordinate_domain": "raw_image_zero_based_pixel_centers",
            }
        )
    return rows


def deduplicate_candidates(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    source_priority = {
        "triangulated_annotation_rays": 0,
        "coarse_exif_gimbal_all_orientations": 1,
    }
    best: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["scene"]), str(row["point_name"]), str(row["image_name"]))
        current = best.get(key)
        candidate_order = (
            source_priority.get(str(row["candidate_source"]), 99),
            -finite_float(row.get("center_score"), 0.0),
            -finite_float(row.get("edge_margin_px"), -1.0),
        )
        if current is None:
            best[key] = row
            continue
        current_order = (
            source_priority.get(str(current["candidate_source"]), 99),
            -finite_float(current.get("center_score"), 0.0),
            -finite_float(current.get("edge_margin_px"), -1.0),
        )
        if candidate_order < current_order:
            best[key] = row
    return list(best.values())


def selection_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    source_priority = 0 if row["candidate_source"] == "triangulated_annotation_rays" else 1
    return (
        source_priority,
        -finite_float(row.get("center_score"), 0.0),
        -finite_float(row.get("edge_margin_px"), -1.0),
        str(row["image_name"]),
    )


def select_diverse(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0 or not rows:
        return []
    bins: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        bin_id = int(row.get("azimuth_bin_45deg", -1))
        bins.setdefault(bin_id, []).append(row)
    for values in bins.values():
        values.sort(key=selection_sort_key)
    selected: list[dict[str, Any]] = []
    ordered_bins = sorted(bins)
    while len(selected) < count:
        added = False
        for bin_id in ordered_bins:
            if bins[bin_id]:
                selected.append(bins[bin_id].pop(0))
                added = True
                if len(selected) >= count:
                    break
        if not added:
            break
    return selected


def select_primary_candidates(
    rows: list[dict[str, Any]],
    per_point: int = PRIMARY_NEW_VIEWS_PER_POINT,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["scene"]), str(row["point_name"])), []).append(row)
    output: list[dict[str, Any]] = []
    for key in sorted(grouped):
        group = grouped[key]
        nadir = [r for r in group if r["view_type"] == "nadir"]
        oblique = [r for r in group if r["view_type"] == "oblique"]
        if nadir and oblique:
            nadir_quota = (per_point + 1) // 2
            oblique_quota = per_point // 2
        elif nadir:
            nadir_quota, oblique_quota = per_point, 0
        else:
            nadir_quota, oblique_quota = 0, per_point
        chosen = select_diverse(nadir, nadir_quota) + select_diverse(oblique, oblique_quota)
        chosen_keys = {(r["scene"], r["point_name"], r["image_name"]) for r in chosen}
        if len(chosen) < per_point:
            remaining = [
                r
                for r in sorted(group, key=selection_sort_key)
                if (r["scene"], r["point_name"], r["image_name"]) not in chosen_keys
            ]
            chosen.extend(remaining[: per_point - len(chosen)])
        chosen.sort(key=selection_sort_key)
        for rank, row in enumerate(chosen, start=1):
            out = dict(row)
            out["rank_for_gcp"] = rank
            output.append(out)
    return output


def rank_all_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["scene"]), str(row["point_name"])), []).append(row)
    output: list[dict[str, Any]] = []
    for key in sorted(grouped):
        group = sorted(grouped[key], key=selection_sort_key)
        for rank, row in enumerate(group, start=1):
            out = dict(row)
            out["rank_for_gcp"] = rank
            output.append(out)
    return output


def validate_task_rows(
    rows: list[dict[str, Any]],
    existing_keys: set[tuple[str, str, str]],
    label: str,
) -> dict[str, Any]:
    keys = [(r["scene"], r["point_name"], r["image_name"]) for r in rows]
    duplicates = len(keys) - len(set(keys))
    overlap = len(set(keys) & existing_keys)
    missing_images = sum(not Path(str(r["image_path"])).is_file() for r in rows)
    out_of_bounds = sum(
        not (
            0.0 <= float(r["pixel_x"]) < int(r["image_width"])
            and 0.0 <= float(r["pixel_y"]) < int(r["image_height"])
        )
        for r in rows
    )
    if duplicates or overlap or missing_images or out_of_bounds:
        raise RuntimeError(
            f"{label}: duplicates={duplicates}, existing_overlap={overlap}, "
            f"missing_images={missing_images}, out_of_bounds={out_of_bounds}"
        )
    return {
        "row_count": len(rows),
        "unique_key_count": len(set(keys)),
        "duplicate_key_count": duplicates,
        "existing_annotation_overlap_count": overlap,
        "missing_image_count": missing_images,
        "out_of_bounds_count": out_of_bounds,
        "point_count": len({(r["scene"], r["point_name"]) for r in rows}),
        "nadir_count": sum(r["view_type"] == "nadir" for r in rows),
        "oblique_count": sum(r["view_type"] == "oblique" for r in rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare direct spatial candidate tasks for the GCP annotator.")
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--release_dir",
        type=Path,
        default=Path(r"E:\datasets\M3M-GCP\scenes\gcp_manual_annotations_v1_2_2"),
    )
    parser.add_argument(
        "--candidate_root",
        type=Path,
        default=REPO_ROOT / "outputs" / "gcp_annotation_candidates_20260617_all",
    )
    parser.add_argument(
        "--audit_root",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "gcp_multiview_annotation_expansion_control_heavy_audit_20260702_160926",
    )
    parser.add_argument("--primary_new_views_per_point", type=int, default=PRIMARY_NEW_VIEWS_PER_POINT)
    parser.add_argument("--stamp", default=time.strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    output_root = args.repo / "outputs" / f"gcp_multiview_direct_annotation_tasks_{args.stamp}"
    output_root.mkdir(parents=True, exist_ok=False)
    candidate_dir = output_root / "candidate_lists"
    working_dir = output_root / "working_annotations"
    launch_dir = output_root / "launchers"
    candidate_dir.mkdir()
    working_dir.mkdir()
    launch_dir.mkdir()

    annotation_sources = resolve_annotation_sources(args.release_dir, args.repo)
    gcp_path = args.release_dir / "gcp_points_primary_usable_cgcs2000_cm108_v1.csv"
    gcps = load_gcps(gcp_path)

    cameras_by_scene: dict[str, list[CameraMeta]] = {}
    missing_metadata_images: dict[str, list[str]] = {}
    image_lookup: dict[tuple[str, str], CameraMeta] = {}
    for scene in SCENES:
        cameras, missing_images = load_camera_metadata(args.candidate_root, scene)
        cameras_by_scene[scene] = cameras
        missing_metadata_images[scene] = missing_images
        for camera in cameras:
            image_lookup[(scene, camera.image_name)] = camera

    broad: list[dict[str, Any]] = []
    for scene in SCENES:
        broad.extend(build_broad_spatial_candidates(scene, cameras_by_scene[scene], gcps))
    triangulated = load_triangulated_candidates(args.audit_root, image_lookup)
    candidates = deduplicate_candidates([*broad, *triangulated])

    all_existing_keys: set[tuple[str, str, str]] = set()
    source_records: dict[str, Any] = {}
    for scene in SCENES:
        source = annotation_sources[scene]
        existing_keys = read_existing_annotation_keys(source)
        all_existing_keys.update(existing_keys)
        working = working_dir / f"{scene}_manual_annotations_v1_3_draft_working.csv"
        shutil.copy2(source, working)
        source_records[scene] = {
            "source_path": str(source),
            "source_sha256": sha256_file(source),
            "working_path": str(working),
            "working_initial_sha256": sha256_file(working),
            "existing_annotation_row_count": len(existing_keys),
        }

    candidates = [
        r
        for r in candidates
        if (r["scene"], r["point_name"], r["image_name"]) not in all_existing_keys
    ]
    all_ranked = rank_all_candidates(candidates)
    primary = select_primary_candidates(candidates, int(args.primary_new_views_per_point))

    validation: dict[str, Any] = {
        "all_spatial": validate_task_rows(all_ranked, all_existing_keys, "all_spatial"),
        "primary": validate_task_rows(primary, all_existing_keys, "primary"),
    }
    primary_keys = {(r["scene"], r["point_name"], r["image_name"]) for r in primary}
    all_keys = {(r["scene"], r["point_name"], r["image_name"]) for r in all_ranked}
    if not primary_keys.issubset(all_keys):
        raise RuntimeError("Primary candidate set is not a subset of all_spatial")

    scene_summary: list[dict[str, Any]] = []
    for scene in SCENES:
        scene_all = [r for r in all_ranked if r["scene"] == scene]
        scene_primary = [r for r in primary if r["scene"] == scene]
        low_points = LOW_VIEW_POINTS.get(scene, set())
        low_primary = [r for r in scene_primary if r["point_name"] in low_points]
        write_candidate_csv(candidate_dir / f"{scene}_all_spatial_candidates.csv", scene_all)
        write_candidate_csv(candidate_dir / f"{scene}_primary_candidates.csv", scene_primary)
        if low_points:
            write_candidate_csv(candidate_dir / f"{scene}_low_view_priority_candidates.csv", low_primary)

        working = Path(source_records[scene]["working_path"])
        image_root = Path(r"E:\datasets\M3M-GCP\scenes") / scene
        for mode, candidate_file in [
            ("primary", candidate_dir / f"{scene}_primary_candidates.csv"),
            ("all_spatial", candidate_dir / f"{scene}_all_spatial_candidates.csv"),
        ]:
            launcher = launch_dir / f"launch_{scene}_{mode}.ps1"
            launcher.write_text(
                "\n".join(
                    [
                        "$ErrorActionPreference = 'Stop'",
                        f"Set-Location '{args.repo}'",
                        "python code\\gcp\\manual_gcp_annotator.py `",
                        f"  --candidates_csv '{candidate_file}' `",
                        f"  --out_csv '{working}' `",
                        f"  --image_root '{image_root}' `",
                        "  --crop_size 720 `",
                        "  --display_size 900 `",
                        "  --annotator user",
                        "",
                    ]
                ),
                encoding="utf-8-sig",
            )
        if low_points:
            candidate_file = candidate_dir / f"{scene}_low_view_priority_candidates.csv"
            launcher = launch_dir / f"launch_{scene}_low_view_priority.ps1"
            launcher.write_text(
                "\n".join(
                    [
                        "$ErrorActionPreference = 'Stop'",
                        f"Set-Location '{args.repo}'",
                        "python code\\gcp\\manual_gcp_annotator.py `",
                        f"  --candidates_csv '{candidate_file}' `",
                        f"  --out_csv '{working}' `",
                        f"  --image_root '{image_root}' `",
                        "  --crop_size 720 `",
                        "  --display_size 900 `",
                        "  --annotator user",
                        "",
                    ]
                ),
                encoding="utf-8-sig",
            )

        scene_summary.append(
            {
                "scene": scene,
                "existing_annotation_rows_seeded": source_records[scene]["existing_annotation_row_count"],
                "primary_new_candidate_rows": len(scene_primary),
                "all_spatial_new_candidate_rows": len(scene_all),
                "primary_point_count": len({r["point_name"] for r in scene_primary}),
                "primary_nadir_rows": sum(r["view_type"] == "nadir" for r in scene_primary),
                "primary_oblique_rows": sum(r["view_type"] == "oblique" for r in scene_primary),
                "low_view_priority_rows": len(low_primary),
                "working_annotation_csv": str(working),
            }
        )

    summary_df = pd.DataFrame(scene_summary)
    summary_df.to_csv(output_root / "scene_task_summary.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "schema": "ms_gcp_direct_multiview_annotation_tasks_v1",
        "stamp": args.stamp,
        "scope": "working_annotation_tasks_only_no_release_mutation",
        "candidate_policy": {
            "visual_prefilter": False,
            "visibility_decision_stage": "inside_manual_annotation_tool",
            "candidate_sources": [
                "triangulated_annotation_rays",
                "coarse_exif_gimbal_all_orientations",
            ],
            "max_off_nadir_deg": MAX_OFF_NADIR_DEG,
            "primary_new_views_per_point": int(args.primary_new_views_per_point),
            "primary_view_mix": "balanced_nadir_oblique_when_both_are_available_then_fill_by_spatial_rank",
            "view_diversity": "round_robin_45_degree_camera_azimuth_bins",
        },
        "pixel_domain": {
            "image_domain": "raw_dji_decoded_pixel_matrix_ignore_exif_orientation",
            "coordinate_domain": "raw_image_zero_based_pixel_centers",
            "predicted_crosshair_is_ground_truth": False,
            "prediction_use": "search_center_only",
        },
        "inputs": {
            "release_dir": str(args.release_dir),
            "gcp_table": str(gcp_path),
            "gcp_table_sha256": sha256_file(gcp_path),
            "candidate_root": str(args.candidate_root),
            "audit_root": str(args.audit_root),
            "annotation_sources": source_records,
            "missing_raw_images_referenced_only_by_stale_metadata": missing_metadata_images,
        },
        "validation": validation,
        "scene_summary": scene_summary,
        "hard_boundaries": [
            "v1.2.2 release not modified",
            "source annotation CSVs not modified",
            "no split or survey-coordinate changes",
            "no evaluator or metric changes",
            "no GPU",
        ],
    }
    write_json(output_root / "task_manifest.json", manifest)

    readme = """# 多视角直接补标任务

本目录不再要求先做 HTML 可见性筛查。候选图由空间投影和已有标注射线生成，进入标注工具后直接判断：

- `1 Good`：点清晰可见，并已点击真实点位；
- `2 Ambiguous`：点可能可见但模糊、遮挡或无法唯一定位；
- `3 Not visible`：画面中不可见；
- 未判断时保持空白，不会自动写成 Good。

黄色准心只是空间预测的搜索中心，不是真值。黄色圆表示预测不确定性范围；紫色准心是根据已保存标注历史估计的校正提示；青色准心才是人工点击位置。

每个场景有两套清单：

- `primary_candidates.csv`：按正射/倾斜和相机方位分层选出的优先任务；
- `all_spatial_candidates.csv`：全部图内空间候选。若某点完成 primary 后 Good 少于目标数量，再切到 all_spatial 继续。

5K 和 20K 另有 `low_view_priority_candidates.csv`，可先处理低视图点。所有输出均写入 `working_annotations`，不会修改 v1.2.2 或历史源标注。
"""
    (output_root / "README_zh.md").write_text(readme, encoding="utf-8")

    manifest_files = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "OUTPUT_SHA256_MANIFEST.json":
            manifest_files.append(
                {
                    "path": path.relative_to(output_root).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    write_json(output_root / "OUTPUT_SHA256_MANIFEST.json", {"files": manifest_files})
    print(json.dumps({"output_root": str(output_root), "scene_summary": scene_summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
