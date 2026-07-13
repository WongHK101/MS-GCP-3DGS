#!/usr/bin/env python
"""Audit map-defined scene points and prepare missing 5K core annotation tasks.

The point sets are read from the user-confirmed survey-area point-distribution
map. Outputs are working annotation artifacts only. The script
does not modify a frozen release, split, packet, evaluator, or survey record.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import pandas as pd

from build_gcp_projection_candidates import load_gcps
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
MAP_DEFINED_POINTS = {
    "gcp_5000_20260602": [f"G{i:02d}" for i in range(1, 11)],
    "gcp_3000_20260602": ["G11", "G12", "G13", "G14", "G15", "G16", "G17", "G18", "NC94"],
    "gcp_10000_20260610": ["G19", "G20", "G21", "G22", "G23", "G24", "G25", "G26", "G27", "G49"],
    "gcp_20000_20260602": ["G28", "G29", "G30", "G31", "G33", "G35", "G36", "G37", "G38", "dyl2"],
}
FIVE_K_TASK_POINTS = ["G04", "G07", "G09"]
REVIEW_ONLY_FORMAL_STATUS = (
    "annotation_allowed_but_formal_primary_blocked_pending_coordinate_quality_review"
)


def visible_good(row: dict[str, Any]) -> bool:
    quality = str(row.get("quality", "")).strip().lower()
    visible = str(row.get("visible", "1")).strip().lower()
    return quality == "good" and visible not in {"0", "false", "no"}


def classify_coverage(good: int, ambiguous: int, not_visible: int) -> str:
    if good >= 4:
        return "complete_usable_annotations"
    if good > 0:
        return "partial_good_annotations"
    if ambiguous > 0:
        return "review_required_no_good"
    if not_visible > 0:
        return "no_usable_annotation_all_not_visible"
    return "unannotated"


def load_supplemental_gcps(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    rows = pd.read_csv(path, dtype=str, keep_default_na=False).to_dict("records")
    gcps: list[dict[str, Any]] = []
    provenance: dict[str, dict[str, str]] = {}
    for row in rows:
        point = str(row["point_name"])
        gcps.append(
            {
                "point_name": point,
                "projected_e": float(row["cgcs2000_gk_cm108_e_m"]),
                "projected_n": float(row["cgcs2000_gk_cm108_n_m"]),
                "normal_height_m": float(row["cgcs2000_normal_height_m"]),
                "ellipsoid_height_m": float(row["wgs84_ellipsoid_height_m"]),
                "point_category": "rtk_report_review_only",
                "quality_evaluation": "coordinate_quality_review_pending",
            }
        )
        provenance[point] = row
    if set(provenance) != {"G07", "G09"}:
        raise RuntimeError(f"Expected supplemental G07/G09 rows, found {sorted(provenance)}")
    return gcps, provenance


def build_coverage_audit(
    working_dir: Path,
    release_dir: Path,
    primary_names: set[str],
    supplemental: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for scene, points in MAP_DEFINED_POINTS.items():
        working_path = working_dir / f"{scene}_manual_annotations_v1_3_draft_working.csv"
        rows = pd.read_csv(working_path, dtype=str, keep_default_na=False)
        release_path = release_dir / f"{scene}_gcp_annotations_pixel_domain_v1_2_2.csv"
        release = pd.read_csv(release_path, dtype=str, keep_default_na=False)
        for point in points:
            point_rows = rows[rows["point_name"].eq(point)]
            quality = point_rows["quality"].str.lower() if len(point_rows) else pd.Series([], dtype=str)
            good = int(sum(visible_good(row) for row in point_rows.to_dict("records")))
            ambiguous = int(quality.eq("ambiguous").sum())
            not_visible = int(quality.eq("not_visible").sum())
            status = classify_coverage(good, ambiguous, not_visible)
            if point in primary_names:
                coordinate_status = "primary_usable_coordinate_table"
                formal_status = "coordinate_gate_available"
            elif point in supplemental:
                coordinate_status = supplemental[point]["coordinate_status"]
                formal_status = supplemental[point]["formal_primary_eligibility"]
            else:
                coordinate_status = "missing_coordinate_record"
                formal_status = "formal_primary_blocked_missing_coordinate_record"
            result.append(
                {
                    "scene": scene,
                    "point_name": point,
                    "map_defined_scene_member": True,
                    "annotation_row_count": len(point_rows),
                    "good_count": good,
                    "ambiguous_count": ambiguous,
                    "not_visible_count": not_visible,
                    "v1_2_2_release_observation_count": int(release["point_name"].eq(point).sum()),
                    "annotation_coverage_status": status,
                    "needs_new_annotation_or_review": status != "complete_usable_annotations",
                    "coordinate_status": coordinate_status,
                    "formal_primary_coordinate_status": formal_status,
                }
            )
    return result


def choose_five_k_tasks(
    broad: list[dict[str, Any]],
    working_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    broad_by_key = {
        (str(row["point_name"]), str(row["image_name"])): row
        for row in broad
        if str(row["point_name"]) in FIVE_K_TASK_POINTS
    }
    output: list[dict[str, Any]] = []

    g04_ambiguous = sorted(
        (
            row
            for row in working_rows
            if row.get("point_name") == "G04" and str(row.get("quality", "")).lower() == "ambiguous"
        ),
        key=lambda row: str(row["image_name"]),
    )
    if len(g04_ambiguous) != 8:
        raise RuntimeError(f"Expected eight G04 ambiguous rows, found {len(g04_ambiguous)}")
    existing_g04_images = {str(row["image_name"]) for row in g04_ambiguous}
    for annotation in g04_ambiguous:
        key = ("G04", str(annotation["image_name"]))
        candidate = broad_by_key.get(key)
        if candidate is None:
            raise RuntimeError(f"G04 ambiguous annotation has no physical projection candidate: {key}")
        item = dict(candidate)
        item["task_action"] = "independent_re_review_existing_ambiguous_raw_annotation"
        output.append(item)

    g04_oblique = [
        row
        for row in broad
        if row["point_name"] == "G04"
        and row["view_type"] == "oblique"
        and row["image_name"] not in existing_g04_images
    ]
    for candidate in select_diverse(g04_oblique, 8):
        item = dict(candidate)
        item["task_action"] = "new_raw_oblique_annotation_for_view_diversity"
        output.append(item)

    for point in ["G07", "G09"]:
        point_rows = [row for row in broad if row["point_name"] == point]
        selected = select_primary_candidates(point_rows, 16)
        if len(selected) != 16:
            raise RuntimeError(f"{point}: expected 16 candidates, found {len(selected)}")
        for candidate in selected:
            item = dict(candidate)
            item["task_action"] = "new_raw_annotation_coordinate_quality_review_pending"
            output.append(item)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in output:
        grouped.setdefault(str(row["point_name"]), []).append(row)
    ranked: list[dict[str, Any]] = []
    for point in FIVE_K_TASK_POINTS:
        rows = grouped[point]
        for rank, row in enumerate(rows, start=1):
            item = dict(row)
            item["rank_for_gcp"] = rank
            ranked.append(item)
    return ranked


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
        default=REPO_ROOT
        / "outputs"
        / "gcp_multiview_direct_annotation_tasks_20260713_annotate_direct_v2",
    )
    parser.add_argument(
        "--candidate_root",
        type=Path,
        default=REPO_ROOT / "outputs" / "gcp_annotation_candidates_20260617_all",
    )
    parser.add_argument(
        "--supplemental_points",
        type=Path,
        default=REPO_ROOT
        / "evidence"
        / "gcp_coordinates"
        / "gcp_points_review_only_map_core_5k_20260714.csv",
    )
    parser.add_argument(
        "--point_distribution_map",
        type=Path,
        default=Path(r"E:\datasets\M3M-GCP\scenes\点位分布图.png"),
    )
    parser.add_argument("--stamp", default=time.strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    output_root = args.repo / "outputs" / f"gcp_map_defined_core_annotation_tasks_{args.stamp}"
    output_root.mkdir(parents=True, exist_ok=False)
    candidate_dir = output_root / "candidate_lists"
    working_dir = output_root / "working_annotations"
    launcher_dir = output_root / "launchers"
    candidate_dir.mkdir()
    working_dir.mkdir()
    launcher_dir.mkdir()

    primary_path = args.release_dir / "gcp_points_primary_usable_cgcs2000_cm108_v1.csv"
    primary_gcps = load_gcps(primary_path)
    primary_by_name = {str(row["point_name"]): row for row in primary_gcps}
    supplemental_gcps, supplemental_provenance = load_supplemental_gcps(args.supplemental_points)

    source_working_dir = args.source_task_root / "working_annotations"
    coverage = build_coverage_audit(
        source_working_dir,
        args.release_dir,
        set(primary_by_name),
        supplemental_provenance,
    )
    coverage_df = pd.DataFrame(coverage)
    scene_counts = (
        coverage_df.groupby(["scene", "annotation_coverage_status"], sort=True)
        .size()
        .unstack(fill_value=0)
    )
    expected_complete = {
        "gcp_5000_20260602": 7,
        "gcp_3000_20260602": 9,
        "gcp_10000_20260610": 10,
        "gcp_20000_20260602": 10,
    }
    for scene, count in expected_complete.items():
        actual = int(scene_counts.loc[scene].get("complete_usable_annotations", 0))
        if actual != count:
            raise RuntimeError(f"{scene}: expected {count} complete points, found {actual}")

    source_working = source_working_dir / "gcp_5000_20260602_manual_annotations_v1_3_draft_working.csv"
    target_working = working_dir / "gcp_5000_20260602_map_core_v1_3_draft_working.csv"
    shutil.copy2(source_working, target_working)
    working_rows = pd.read_csv(source_working, dtype=str, keep_default_na=False).to_dict("records")

    cameras, missing_images = load_camera_metadata(args.candidate_root, "gcp_5000_20260602")
    if missing_images:
        raise RuntimeError(f"5K metadata references missing raw images: {missing_images}")
    task_gcps = [primary_by_name["G04"], *supplemental_gcps]
    broad = build_broad_spatial_candidates("gcp_5000_20260602", cameras, task_gcps)
    tasks = choose_five_k_tasks(broad, working_rows)
    existing_keys = {
        (str(row["scene"]), str(row["point_name"]), str(row["image_name"]))
        for row in working_rows
    }
    # Existing G04 rows are intentionally re-reviewed; all other task keys must be new.
    g04_existing_keys = {
        key for key in existing_keys if key[1] == "G04" and any(
            row["point_name"] == "G04"
            and row["image_name"] == key[2]
            and row["quality"].lower() == "ambiguous"
            for row in working_rows
        )
    }
    new_task_rows = [
        row
        for row in tasks
        if (row["scene"], row["point_name"], row["image_name"]) not in g04_existing_keys
    ]
    validation = validate_task_rows(new_task_rows, existing_keys, "new_map_core_tasks")
    keys = [(row["scene"], row["point_name"], row["image_name"]) for row in tasks]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Map-core candidate list contains duplicate keys")
    if len(tasks) != 48:
        raise RuntimeError(f"Expected 48 total tasks, found {len(tasks)}")

    candidate_path = candidate_dir / "gcp_5000_20260602_G04_G07_G09_map_core_candidates.csv"
    write_candidate_csv(candidate_path, tasks)
    coverage_path = output_root / "map_defined_point_annotation_coverage_audit.csv"
    coverage_df.to_csv(coverage_path, index=False, encoding="utf-8-sig", lineterminator="\r\n")
    shutil.copy2(args.supplemental_points, output_root / args.supplemental_points.name)

    launcher = launcher_dir / "launch_gcp_5000_map_core_G04_G07_G09.ps1"
    launcher.write_text(
        "\n".join(
            [
                "$ErrorActionPreference = 'Stop'",
                f"Set-Location '{args.repo}'",
                "python code\\gcp\\manual_gcp_annotator.py `",
                f"  --candidates_csv '{candidate_path}' `",
                f"  --out_csv '{target_working}' `",
                "  --image_root 'E:\\datasets\\M3M-GCP\\scenes\\gcp_5000_20260602' `",
                "  --crop_size 720 `",
                "  --display_size 900 `",
                "  --annotator user",
                "",
            ]
        ),
        encoding="utf-8-sig",
    )

    point_task_summary = []
    for point in FIVE_K_TASK_POINTS:
        rows = [row for row in tasks if row["point_name"] == point]
        point_task_summary.append(
            {
                "point_name": point,
                "task_rows": len(rows),
                "nadir_rows": sum(row["view_type"] == "nadir" for row in rows),
                "oblique_rows": sum(row["view_type"] == "oblique" for row in rows),
                "existing_ambiguous_re_review_rows": sum(
                    row["task_action"] == "independent_re_review_existing_ambiguous_raw_annotation"
                    for row in rows
                ),
                "formal_primary_coordinate_status": (
                    "coordinate_gate_available" if point == "G04" else REVIEW_ONLY_FORMAL_STATUS
                ),
            }
        )
    pd.DataFrame(point_task_summary).to_csv(
        output_root / "five_k_task_summary.csv",
        index=False,
        encoding="utf-8-sig",
        lineterminator="\r\n",
    )

    manifest = {
        "schema": "ms_gcp_map_defined_core_annotation_tasks_v1",
        "scope": "working_raw_image_annotation_only_no_release_mutation",
        "map_defined_point_sets": MAP_DEFINED_POINTS,
        "point_distribution_map": {
            "path": str(args.point_distribution_map),
            "sha256": sha256_file(args.point_distribution_map),
        },
        "coverage_conclusion": {
            "gcp_3000_20260602": "9_of_9_have_complete_usable_annotations",
            "gcp_5000_20260602": "7_of_10_complete_G04_review_only_G07_G09_unannotated",
            "gcp_10000_20260610": "10_of_10_have_complete_usable_annotations",
            "gcp_20000_20260602": "10_of_10_have_complete_usable_annotations_G34_G39_excluded_by_map_definition",
        },
        "inputs": {
            "source_working_5k": {
                "path": str(source_working),
                "sha256": sha256_file(source_working),
            },
            "primary_gcp_table": {"path": str(primary_path), "sha256": sha256_file(primary_path)},
            "supplemental_review_only_points": {
                "path": str(args.supplemental_points),
                "sha256": sha256_file(args.supplemental_points),
            },
            "candidate_metadata": str(args.candidate_root / "gcp_5000_20260602" / "image_metadata.csv"),
        },
        "tasks": point_task_summary,
        "validation": {
            **validation,
            "total_candidate_rows": len(tasks),
            "intentional_existing_G04_re_review_rows": len(g04_existing_keys),
            "candidate_duplicate_count": len(keys) - len(set(keys)),
            "raw_image_domain": "raw_dji_decoded_pixel_matrix_ignore_exif_orientation",
            "coordinate_domain": "raw_image_zero_based_pixel_centers",
        },
        "hard_boundaries": [
            "v1.2.2 release not modified",
            "source working annotations not modified",
            "G07 and G09 remain formal-primary blocked pending coordinate-quality review",
            "no split, packet, evaluator, survey record, or frozen metric changes",
        ],
    }
    write_json(output_root / "task_manifest.json", manifest)

    readme = """# 5K 测区内缺失点补标任务

根据 `点位分布图.png`，5K 测区应使用中心绿色框内的 G01-G10。核对结果：

- G01/G02/G03/G05/G06/G08/G10 已有可用 Good 标注；
- G04 有 8 张正射记录，但全部是 ambiguous，需要独立复核；
- G07/G09 尚无任何标注，本任务各提供 8 张正射 + 8 张倾斜候选。

标注仍在原始 raw DJI 图像上进行，输出为 zero-based pixel-center raw 坐标。候选准心只是搜索中心；请按实际画面标记 Good、Ambiguous 或 Not visible。

注意：G07/G09 的坐标来自原始坐标工作簿，但在 RTK primary-usable 清洗中被标记为 review-only。可以先完成影像标注，但在坐标质量复核通过前不得进入 formal primary benchmark。G04 没有这一坐标限制。

本目录是 v1.3 工作草案，不修改 v1.2.2。
"""
    (output_root / "README_zh.md").write_text(readme, encoding="utf-8")

    files = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "OUTPUT_SHA256_MANIFEST.json":
            files.append(
                {
                    "path": path.relative_to(output_root).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    write_json(output_root / "OUTPUT_SHA256_MANIFEST.json", {"files": files})
    print(json.dumps({"output_root": str(output_root), "tasks": point_task_summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
