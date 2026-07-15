#!/usr/bin/env python
"""Audit the latest 5K annotations and prepare focused 50K/100K follow-up tasks.

This script only creates new working annotation artifacts. It does not modify
the v1.2.2 release, existing working CSVs, splits, packets, or evaluator output.
Candidate images are restricted to the current benchmark source-camera track.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from prepare_direct_multiview_annotation_tasks import (
    build_broad_spatial_candidates,
    load_camera_metadata,
    select_diverse,
    select_primary_candidates,
    sha256_file,
    validate_task_rows,
    write_candidate_csv,
    write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCENE_50K = "gcp_50000_20260610"
SCENE_100K = "gcp_100000_20260610"
FIVE_K_QC_POINTS = ["G04", "G07", "G09"]
FIFTY_K_GAP_POINTS = ["G39", "G43", "G45", "G46"]
FIFTY_K_TASK_POINTS = ["G39"]
HUNDRED_K_NEW_POINTS = ["G20", "dyl2", "G08", "G33"]
HUNDRED_K_SUPPLEMENT_POINTS = ["G35", "G38", "k01", "NC94"]
MIN_FORMAL_CANDIDATE_VIEWS = 4
FIVE_K_REPROJECTION_QC_MAX_PX = 5.0


def visible_good(row: dict[str, Any]) -> bool:
    return str(row.get("quality", "")).strip().lower() == "good" and str(
        row.get("visible", "1")
    ).strip().lower() not in {"0", "false", "no"}


def qvec_to_rotation(qvec: list[Any]) -> np.ndarray:
    w, x, y, z = (float(value) for value in qvec)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def source_camera_track(
    camera_provenance: dict[str, Any], scene: str
) -> tuple[set[str], dict[int, dict[str, Any]], dict[str, dict[str, Any]]]:
    model = camera_provenance["scenes"][scene]["source_model"]
    cameras = {int(row["camera_id"]): row for row in model["cameras"]}
    images = {str(row["image_name"]): row for row in model["images"]}
    return set(images), cameras, images


def annotation_ray(
    row: dict[str, Any],
    cameras: dict[int, dict[str, Any]],
    images: dict[str, dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    image = images[str(row["image_name"])]
    camera = cameras[int(image["camera_id"])]
    if camera["model"] != "SIMPLE_RADIAL" or len(camera["params"]) != 4:
        raise RuntimeError(f"Unsupported source camera for annotation QC: {camera}")
    focal, cx, cy, radial = (float(value) for value in camera["params"])
    xd = (float(row["manual_x"]) - cx) / focal
    yd = (float(row["manual_y"]) - cy) / focal
    x, y = xd, yd
    for _ in range(20):
        denominator = 1.0 + radial * (x * x + y * y)
        if not math.isfinite(denominator) or denominator == 0.0:
            raise RuntimeError("Non-finite SIMPLE_RADIAL inversion")
        x, y = xd / denominator, yd / denominator
    rotation = qvec_to_rotation(image["qvec"])
    translation = np.asarray([float(value) for value in image["tvec"]], dtype=np.float64)
    center = -rotation.T @ translation
    direction = rotation.T @ np.asarray([x, y, 1.0], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    return center, direction, rotation, translation, (focal, cx, cy, radial)


def triangulate_annotation_rays(
    rows: list[dict[str, Any]],
    cameras: dict[int, dict[str, Any]],
    images: dict[str, dict[str, Any]],
) -> tuple[np.ndarray, float]:
    matrix = np.zeros((3, 3), dtype=np.float64)
    rhs = np.zeros(3, dtype=np.float64)
    for row in rows:
        center, direction, *_ = annotation_ray(row, cameras, images)
        projector = np.eye(3, dtype=np.float64) - np.outer(direction, direction)
        matrix += projector
        rhs += projector @ center
    return np.linalg.solve(matrix, rhs), float(np.linalg.cond(matrix))


def reproject_source_pixel(
    xyz: np.ndarray,
    row: dict[str, Any],
    cameras: dict[int, dict[str, Any]],
    images: dict[str, dict[str, Any]],
) -> np.ndarray:
    _, _, rotation, translation, (focal, cx, cy, radial) = annotation_ray(
        row, cameras, images
    )
    camera_xyz = rotation @ xyz + translation
    normalized_x = camera_xyz[0] / camera_xyz[2]
    normalized_y = camera_xyz[1] / camera_xyz[2]
    scale = 1.0 + radial * (normalized_x * normalized_x + normalized_y * normalized_y)
    return np.asarray(
        [focal * normalized_x * scale + cx, focal * normalized_y * scale + cy],
        dtype=np.float64,
    )


def audit_five_k_annotations(
    annotation_path: Path,
    camera_provenance: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = pd.read_csv(annotation_path, dtype=str, keep_default_na=False).to_dict("records")
    keys = [(row["scene"], row["point_name"], row["image_name"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Latest 5K working annotations contain duplicate keys")
    _, cameras, images = source_camera_track(camera_provenance, "gcp_5000_20260602")
    report: list[dict[str, Any]] = []
    for point_name in FIVE_K_QC_POINTS:
        all_good = [
            row for row in rows if row["point_name"] == point_name and visible_good(row)
        ]
        mapped_good = [row for row in all_good if row["image_name"] in images]
        if len(mapped_good) < MIN_FORMAL_CANDIDATE_VIEWS:
            raise RuntimeError(f"{point_name}: fewer than four model-mapped Good observations")
        xyz, condition = triangulate_annotation_rays(mapped_good, cameras, images)
        errors = []
        for row in mapped_good:
            observed = np.asarray([float(row["manual_x"]), float(row["manual_y"])])
            errors.append(float(np.linalg.norm(reproject_source_pixel(xyz, row, cameras, images) - observed)))
        max_error = max(errors)
        if max_error > FIVE_K_REPROJECTION_QC_MAX_PX:
            raise RuntimeError(
                f"{point_name}: multi-view reprojection max {max_error:.6f}px exceeds QC limit"
            )
        report.append(
            {
                "scene": "gcp_5000_20260602",
                "point_name": point_name,
                "all_good_observations": len(all_good),
                "model_mapped_good_observations": len(mapped_good),
                "unmapped_good_observations": len(all_good) - len(mapped_good),
                "triangulation_condition_number": condition,
                "reprojection_median_px": float(np.median(errors)),
                "reprojection_p95_px": float(np.percentile(errors, 95)),
                "reprojection_max_px": max_error,
                "qc_limit_px": FIVE_K_REPROJECTION_QC_MAX_PX,
                "qc_status": "pass",
            }
        )
    return report


def load_review_only_points(workbook_path: Path, point_names: list[str]) -> list[dict[str, Any]]:
    frame = pd.read_excel(workbook_path, dtype=str)
    result: list[dict[str, Any]] = []
    for point_name in point_names:
        matches = frame[frame.iloc[:, 0].eq(point_name)]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one review-only coordinate row for {point_name}")
        row = matches.iloc[0]
        result.append(
            {
                "point_name": point_name,
                "projected_e": float(row.iloc[1]),
                "projected_n": float(row.iloc[2]),
                "normal_height_m": float(row.iloc[3]),
                "ellipsoid_height_m": float(row.iloc[6]),
                "point_category": "rtk_report_review_only",
                "quality_evaluation": "coordinate_quality_review_pending",
            }
        )
    return result


def rank_per_point(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["point_name"])].append(row)
    ranked: list[dict[str, Any]] = []
    order = [*FIFTY_K_TASK_POINTS, *HUNDRED_K_NEW_POINTS, *HUNDRED_K_SUPPLEMENT_POINTS]
    for point_name in order:
        for rank, row in enumerate(grouped.get(point_name, []), start=1):
            item = dict(row)
            item["rank_for_gcp"] = rank
            ranked.append(item)
    return ranked


def choose_fifty_k_tasks(
    broad: list[dict[str, Any]], model_images: set[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    counts: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    for point_name in FIFTY_K_GAP_POINTS:
        mapped = [
            row
            for row in broad
            if row["point_name"] == point_name and row["image_name"] in model_images
        ]
        counts.append(
            {
                "scene": SCENE_50K,
                "point_name": point_name,
                "current_model_candidate_count": len(mapped),
                "nadir_candidate_count": sum(row["view_type"] == "nadir" for row in mapped),
                "oblique_candidate_count": sum(row["view_type"] == "oblique" for row in mapped),
                "decision": (
                    "open_working_annotation_coordinate_review_pending"
                    if point_name in FIFTY_K_TASK_POINTS and len(mapped) >= MIN_FORMAL_CANDIDATE_VIEWS
                    else "do_not_open_fewer_than_4_current_model_views"
                ),
            }
        )
        if point_name not in FIFTY_K_TASK_POINTS:
            continue
        chosen = select_diverse(mapped, min(8, len(mapped)))
        if len(chosen) < MIN_FORMAL_CANDIDATE_VIEWS:
            raise RuntimeError(f"{point_name}: insufficient current-model candidates")
        for row in chosen:
            item = dict(row)
            item["task_action"] = "new_raw_annotation_coordinate_quality_review_pending"
            selected.append(item)
    return rank_per_point(selected), counts


def choose_hundred_k_tasks(
    all_spatial_path: Path,
    model_images: set[str],
    existing_keys: set[tuple[str, str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = pd.read_csv(all_spatial_path, dtype=str, keep_default_na=False).to_dict("records")
    available = [
        row
        for row in rows
        if row["image_name"] in model_images
        and (row["scene"], row["point_name"], row["image_name"]) not in existing_keys
    ]
    output: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for point_name in HUNDRED_K_NEW_POINTS:
        group = [row for row in available if row["point_name"] == point_name]
        nadir = sum(row["view_type"] == "nadir" for row in group)
        oblique = len(group) - nadir
        if nadir < 3 or oblique < 2 or len(group) < 8:
            raise RuntimeError(
                f"{point_name}: new-point coverage gate failed: total={len(group)}, nadir={nadir}, oblique={oblique}"
            )
        chosen = select_primary_candidates(group, 8)
        for row in chosen:
            item = dict(row)
            item["task_action"] = "new_map_inside_mixed_view_annotation"
            output.append(item)
        summary.append(
            {
                "scene": SCENE_100K,
                "point_name": point_name,
                "task_kind": "new_spatial_gap_point",
                "available_current_model_views": len(group),
                "available_nadir": nadir,
                "available_oblique": oblique,
                "selected_rows": len(chosen),
            }
        )
    for point_name in HUNDRED_K_SUPPLEMENT_POINTS:
        group = [
            row
            for row in available
            if row["point_name"] == point_name and row["view_type"] == "oblique"
        ]
        chosen = select_diverse(group, 4)
        if len(chosen) < 2:
            raise RuntimeError(f"{point_name}: fewer than two supplemental oblique candidates")
        for row in chosen:
            item = dict(row)
            item["task_action"] = "supplement_existing_point_oblique_view_diversity"
            output.append(item)
        summary.append(
            {
                "scene": SCENE_100K,
                "point_name": point_name,
                "task_kind": "existing_low_mixed_view_point",
                "available_current_model_views": len(group),
                "available_nadir": 0,
                "available_oblique": len(group),
                "selected_rows": len(chosen),
            }
        )
    return rank_per_point(output), summary


def write_launcher(
    path: Path,
    repo: Path,
    candidate_path: Path,
    working_path: Path,
    image_root: Path,
) -> None:
    path.write_text(
        "\n".join(
            [
                "$ErrorActionPreference = 'Stop'",
                f"Set-Location '{repo}'",
                "python code\\gcp\\manual_gcp_annotator.py `",
                f"  --candidates_csv '{candidate_path}' `",
                f"  --out_csv '{working_path}' `",
                f"  --image_root '{image_root}' `",
                "  --crop_size 720 `",
                "  --display_size 900 `",
                "  --annotator user",
                "",
            ]
        ),
        encoding="utf-8-sig",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--release_dir",
        type=Path,
        default=Path(r"E:\datasets\M3M-GCP\scenes\gcp_manual_annotations_v1_2_2"),
    )
    parser.add_argument(
        "--source_task_root",
        type=Path,
        default=REPO_ROOT / "outputs" / "gcp_multiview_direct_annotation_tasks_20260713_annotate_direct_v2",
    )
    parser.add_argument(
        "--latest_5k_working",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "gcp_map_defined_core_annotation_tasks_20260714_map_core_G04_G07_G09_v1"
        / "working_annotations"
        / "gcp_5000_20260602_map_core_v1_3_draft_working.csv",
    )
    parser.add_argument(
        "--candidate_root",
        type=Path,
        default=REPO_ROOT / "outputs" / "gcp_annotation_candidates_20260617_all",
    )
    parser.add_argument(
        "--converted_gcp_workbook",
        type=Path,
        default=Path(r"E:\datasets\M3M-GCP\_gcp_points_converted_20260614.xlsx"),
    )
    parser.add_argument("--stamp", default=time.strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    output_root = args.repo / "outputs" / f"gcp_followup_annotation_tasks_50k100k_{args.stamp}"
    output_root.mkdir(parents=True, exist_ok=False)
    candidate_dir = output_root / "candidate_lists"
    working_dir = output_root / "working_annotations"
    launcher_dir = output_root / "launchers"
    candidate_dir.mkdir()
    working_dir.mkdir()
    launcher_dir.mkdir()

    camera_provenance_path = args.release_dir / "camera_provenance_manifest_v1_2_2.json"
    camera_provenance = json.loads(camera_provenance_path.read_text(encoding="utf-8"))
    five_k_qc = audit_five_k_annotations(args.latest_5k_working, camera_provenance)
    pd.DataFrame(five_k_qc).to_csv(
        output_root / "five_k_completed_annotation_qc.csv",
        index=False,
        encoding="utf-8-sig",
        lineterminator="\r\n",
    )

    source_working_dir = args.source_task_root / "working_annotations"
    source_50k = source_working_dir / f"{SCENE_50K}_manual_annotations_v1_3_draft_working.csv"
    source_100k = source_working_dir / f"{SCENE_100K}_manual_annotations_v1_3_draft_working.csv"
    target_50k = working_dir / f"{SCENE_50K}_followup_v1_3_draft_working.csv"
    target_100k = working_dir / f"{SCENE_100K}_followup_v1_3_draft_working.csv"
    shutil.copy2(source_50k, target_50k)
    shutil.copy2(source_100k, target_100k)

    model_50k, _, _ = source_camera_track(camera_provenance, SCENE_50K)
    review_points = load_review_only_points(args.converted_gcp_workbook, FIFTY_K_GAP_POINTS)
    cameras_50k, missing_50k = load_camera_metadata(args.candidate_root, SCENE_50K)
    stale_deleted = [path for path in missing_50k if Path(path).name == "DJI_20260610161944_0001_D.JPG"]
    unexpected_missing = sorted(set(missing_50k) - set(stale_deleted))
    if unexpected_missing:
        raise RuntimeError(f"Unexpected missing 50K raw images: {unexpected_missing}")
    broad_50k = build_broad_spatial_candidates(SCENE_50K, cameras_50k, review_points)
    tasks_50k, decisions_50k = choose_fifty_k_tasks(broad_50k, model_50k)

    model_100k, _, _ = source_camera_track(camera_provenance, SCENE_100K)
    rows_100k = pd.read_csv(source_100k, dtype=str, keep_default_na=False).to_dict("records")
    existing_100k = {
        (row["scene"], row["point_name"], row["image_name"]) for row in rows_100k
    }
    tasks_100k, summary_100k = choose_hundred_k_tasks(
        args.source_task_root
        / "candidate_lists"
        / f"{SCENE_100K}_all_spatial_candidates.csv",
        model_100k,
        existing_100k,
    )

    existing_50k = {
        (row["scene"], row["point_name"], row["image_name"])
        for row in pd.read_csv(source_50k, dtype=str, keep_default_na=False).to_dict("records")
    }
    validation_50k = validate_task_rows(tasks_50k, existing_50k, "50K_followup")
    validation_100k = validate_task_rows(tasks_100k, existing_100k, "100K_followup")
    if any(row["image_name"] not in model_50k for row in tasks_50k):
        raise RuntimeError("50K task contains a non-model image")
    if any(row["image_name"] not in model_100k for row in tasks_100k):
        raise RuntimeError("100K task contains a non-model image")

    candidate_50k = candidate_dir / f"{SCENE_50K}_G39_followup_candidates.csv"
    candidate_100k = candidate_dir / f"{SCENE_100K}_spatial_and_view_followup_candidates.csv"
    write_candidate_csv(candidate_50k, tasks_50k)
    write_candidate_csv(candidate_100k, tasks_100k)
    pd.DataFrame(decisions_50k).to_csv(
        output_root / "fifty_k_gap_candidate_decisions.csv",
        index=False,
        encoding="utf-8-sig",
        lineterminator="\r\n",
    )
    pd.DataFrame(summary_100k).to_csv(
        output_root / "hundred_k_followup_task_summary.csv",
        index=False,
        encoding="utf-8-sig",
        lineterminator="\r\n",
    )

    launcher_50k = launcher_dir / "launch_50K_G39_followup.ps1"
    launcher_100k = launcher_dir / "launch_100K_spatial_and_view_followup.ps1"
    write_launcher(
        launcher_50k,
        args.repo,
        candidate_50k,
        target_50k,
        Path(r"E:\datasets\M3M-GCP\scenes\gcp_50000_20260610"),
    )
    write_launcher(
        launcher_100k,
        args.repo,
        candidate_100k,
        target_100k,
        Path(r"E:\datasets\M3M-GCP\scenes\gcp_100000_20260610"),
    )

    scene_decisions = [
        {"scene": "gcp_3000_20260602", "open_tool": False, "reason": "map_core_9_of_9_have_at_least_4_good_views"},
        {"scene": "gcp_5000_20260602", "open_tool": False, "reason": "latest_G04_G07_G09_annotations_pass_multiview_qc"},
        {"scene": "gcp_10000_20260610", "open_tool": False, "reason": "map_core_10_of_10_have_at_least_4_good_views"},
        {"scene": "gcp_20000_20260602", "open_tool": False, "reason": "map_core_10_of_10_have_at_least_4_good_views"},
        {"scene": SCENE_50K, "open_tool": True, "reason": "G39_has_6_current_model_views_and_fills_spatial_gap_coordinate_review_pending"},
        {"scene": SCENE_100K, "open_tool": True, "reason": "four_new_map_inside_mixed_view_points_plus_four_low_view_diversity_points"},
    ]
    pd.DataFrame(scene_decisions).to_csv(
        output_root / "six_scene_followup_decisions.csv",
        index=False,
        encoding="utf-8-sig",
        lineterminator="\r\n",
    )

    readme = f"""# 50K / 100K 定向补标任务

本目录是工作标注，不修改 v1.2.2 release。

## 已完成检查

- 5K G04/G07/G09 均无空状态，模型可映射的 Good 视图各 8 张。
- 多视图重投影最大误差均小于 {FIVE_K_REPROJECTION_QC_MAX_PX:.1f} px，未发现错点或像素域偏移。
- 3K、5K、10K、20K 当前不需要立即继续打开。

## 50K

- 仅打开 G39，共 {len(tasks_50k)} 张当前模型可用正射视图。
- G39 坐标仍为 review-only；完成标注不代表可直接进入正式 release。
- G43/G45/G46 在当前 50K 模型中不足 4 张可用相机视图，本轮不打开。

## 100K

- 新增测区内且同时具有正射/倾斜模型视图的点：G20、dyl2、G08、G33。
- 补充方向覆盖较弱的已有点：G35、G38、k01、NC94。
- 无正射覆盖或不足 4 个模型视图的候选未加入。

标注仍在 raw DJI decoded-image pixel domain 完成；Good / ambiguous / not_visible 均应按真实可见性填写。
"""
    (output_root / "README_zh.md").write_text(readme, encoding="utf-8")

    manifest = {
        "schema": "ms_gcp_followup_annotation_tasks_50k100k_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "frozen_release_modified": False,
        "source_annotations_modified": False,
        "inputs": {
            "latest_5k_working": {"path": str(args.latest_5k_working), "sha256": sha256_file(args.latest_5k_working)},
            "source_50k_working": {"path": str(source_50k), "sha256": sha256_file(source_50k)},
            "source_100k_working": {"path": str(source_100k), "sha256": sha256_file(source_100k)},
            "camera_provenance": {"path": str(camera_provenance_path), "sha256": sha256_file(camera_provenance_path)},
            "converted_gcp_workbook": {"path": str(args.converted_gcp_workbook), "sha256": sha256_file(args.converted_gcp_workbook)},
        },
        "validation": {"50k": validation_50k, "100k": validation_100k},
        "task_counts": {"50k": len(tasks_50k), "100k": len(tasks_100k)},
        "launchers": {"50k": str(launcher_50k), "100k": str(launcher_100k)},
    }
    write_json(output_root / "task_manifest.json", manifest)

    output_files = sorted(
        path
        for path in output_root.rglob("*")
        if path.is_file() and path.name != "OUTPUT_SHA256_MANIFEST.json"
    )
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
                for path in output_files
            ],
        },
    )
    print(output_root)
    print(json.dumps(manifest["validation"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
