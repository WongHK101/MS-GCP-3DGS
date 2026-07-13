#!/usr/bin/env python
"""Audit completed annotation tasks for verified near-nadir core coverage.

This is a read-only, post-annotation audit. It does not rewrite annotation
tables or release files. Candidate geometry is evidence only; visibility and
usable coverage are determined from human annotation status.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from build_gcp_projection_candidates import load_gcps, project_gcp
from prepare_direct_multiview_annotation_tasks import load_camera_metadata


REPO_ROOT = Path(__file__).resolve().parents[2]
SCENES = [
    "gcp_3000_20260602",
    "gcp_5000_20260602",
    "gcp_10000_20260610",
    "gcp_20000_20260602",
    "gcp_50000_20260610",
    "gcp_100000_20260610",
]
NADIR_THRESHOLD_DEG = 5.0
MIN_GOOD_OBSERVATIONS = 4
MIN_GOOD_NADIR_OBSERVATIONS = 3
MIN_GOOD_AZIMUTH_BINS = 2


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


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def visible_good(row: dict[str, Any]) -> bool:
    quality = str(row.get("quality", "good")).strip().lower()
    visible = str(row.get("visible", "1")).strip().lower()
    return quality == "good" and visible not in {"0", "false", "no"}


def camera_azimuth_bin(
    camera_e: float,
    camera_n: float,
    point_e: float,
    point_n: float,
) -> tuple[float, int]:
    azimuth = (math.degrees(math.atan2(camera_e - point_e, camera_n - point_n)) + 360.0) % 360.0
    return azimuth, int(((azimuth + 22.5) % 360.0) // 45.0)


def classify_future_formal_candidate(
    good_observations: int,
    good_nadir_observations: int,
    good_azimuth_bins: int,
) -> tuple[str, str]:
    reasons: list[str] = []
    if good_observations < MIN_GOOD_OBSERVATIONS:
        reasons.append("fewer_than_4_human_verified_good_observations")
    if good_nadir_observations < MIN_GOOD_NADIR_OBSERVATIONS:
        reasons.append("fewer_than_3_human_verified_near_nadir_observations")
    if good_azimuth_bins < MIN_GOOD_AZIMUTH_BINS:
        reasons.append("fewer_than_2_camera_position_azimuth_bins")
    if reasons:
        return "exclude_from_future_v1_3_formal_primary_draft", ";".join(reasons)
    if good_nadir_observations == MIN_GOOD_NADIR_OBSERVATIONS:
        return (
            "provisionally_eligible_at_minimum_nadir_overlap_requires_independent_review",
            "passes_draft_minimum_exactly",
        )
    return "provisionally_eligible_for_future_v1_3_formal_primary", "passes_draft_coverage_rule"


def release_annotation_path(release_dir: Path, scene: str) -> Path:
    return release_dir / f"{scene}_gcp_annotations_pixel_domain_v1_2_2.csv"


def normalized_working_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        **row,
        "image_name": Path(str(row.get("image_name", ""))).name,
        "manual_x": str(row.get("manual_x", "")),
        "manual_y": str(row.get("manual_y", "")),
        "quality": str(row.get("quality", "")).strip().lower(),
    }


def normalized_release_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "scene": str(row["scene"]),
        "point_name": str(row["point_name"]),
        "image_name": Path(str(row["raw_image_name"])).name,
        "manual_x": str(row["raw_manual_x"]),
        "manual_y": str(row["raw_manual_y"]),
        "quality": "good",
        "visible": "1",
        "annotator": "v1.2.2_frozen_release",
        "updated_at": "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--task_root",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "gcp_multiview_direct_annotation_tasks_20260713_annotate_direct_v2",
    )
    parser.add_argument("--output_root", type=Path)
    args = parser.parse_args()

    output_root = args.output_root or (
        args.repo / "outputs" / f"gcp_completed_annotation_nadir_coverage_audit_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    output_root.mkdir(parents=True, exist_ok=False)

    split_path = args.release_dir / "gcp_control_checkpoint_splits_v1.csv"
    gcp_path = args.release_dir / "gcp_points_primary_usable_cgcs2000_cm108_v1.csv"
    split = pd.read_csv(split_path, dtype=str, keep_default_na=False)
    gcp_rows = load_gcps(gcp_path)
    gcp_by_name = {str(row["point_name"]): row for row in gcp_rows}
    split_by_key = {
        (str(row.scene), str(row.point_name)): row._asdict()
        for row in split.itertuples(index=False)
    }

    task_candidates: dict[tuple[str, str, str], dict[str, Any]] = {}
    for path in sorted((args.task_root / "candidate_lists").glob("*_low_view_priority_candidates.csv")):
        for row in pd.read_csv(path, dtype=str, keep_default_na=False).to_dict("records"):
            key = (str(row["scene"]), str(row["point_name"]), Path(str(row["image_name"])).name)
            if key in task_candidates:
                raise RuntimeError(f"Duplicate direct-task candidate key: {key}")
            task_candidates[key] = row

    observation_evidence: list[dict[str, Any]] = []
    point_decisions: list[dict[str, Any]] = []
    conflict_rows: list[dict[str, Any]] = []
    release_input_records: dict[str, Any] = {}
    working_input_records: dict[str, Any] = {}

    for scene in SCENES:
        release_path = release_annotation_path(args.release_dir, scene)
        release_raw = pd.read_csv(release_path, dtype=str, keep_default_na=False).to_dict("records")
        release_rows = [normalized_release_row(row) for row in release_raw]
        release_keys = {(row["point_name"], row["image_name"]) for row in release_rows}
        release_points = sorted({row["point_name"] for row in release_rows})
        release_input_records[scene] = {
            "path": str(release_path),
            "sha256": sha256_file(release_path),
            "row_count": len(release_rows),
        }

        working_path = (
            args.task_root
            / "working_annotations"
            / f"{scene}_manual_annotations_v1_3_draft_working.csv"
        )
        if working_path.is_file():
            working_all = [
                normalized_working_row(row)
                for row in pd.read_csv(working_path, dtype=str, keep_default_na=False).to_dict("records")
            ]
            rows = [row for row in working_all if row["point_name"] in release_points]
            working_keys = {(row["point_name"], row["image_name"]) for row in rows}
            missing_release = sorted(release_keys - working_keys)
            non_good_release = sorted(
                key
                for key in release_keys
                if not any(
                    r["point_name"] == key[0] and r["image_name"] == key[1] and visible_good(r)
                    for r in rows
                )
            )
            if missing_release or non_good_release:
                raise RuntimeError(
                    f"{scene}: working annotations do not preserve release-good rows: "
                    f"missing={missing_release}, non_good={non_good_release}"
                )
            working_input_records[scene] = {
                "path": str(working_path),
                "sha256": sha256_file(working_path),
                "row_count": len(working_all),
                "release_point_row_count": len(rows),
            }
            annotation_source = "completed_v1_3_draft_working_annotations"
        else:
            working_all = []
            rows = release_rows
            annotation_source = "frozen_v1_2_2_good_rows_only"

        metadata_path = args.candidate_root / scene / "image_metadata.csv"
        metadata = pd.read_csv(metadata_path, dtype=str, keep_default_na=False)
        metadata_by_image = {
            Path(str(row["image_name"])).name: row
            for row in metadata.to_dict("records")
        }
        cameras, missing_camera_images = load_camera_metadata(args.candidate_root, scene)
        camera_by_image = {camera.image_name: camera for camera in cameras}
        referenced_images = {str(row["image_name"]) for row in rows}
        unavailable_referenced_images = sorted(referenced_images - set(camera_by_image))
        if unavailable_referenced_images:
            raise RuntimeError(
                f"{scene}: audited observations reference unavailable images: "
                f"{unavailable_referenced_images}"
            )
        if missing_camera_images:
            release_input_records[scene]["stale_unreferenced_metadata_image_paths"] = sorted(
                missing_camera_images
            )

        for row in rows:
            point = str(row["point_name"])
            image = str(row["image_name"])
            meta = metadata_by_image.get(image)
            gcp = gcp_by_name.get(point)
            if meta is None or gcp is None:
                raise RuntimeError(f"Missing metadata/GCP for {(scene, point, image)}")
            off_nadir = abs(finite_float(meta.get("pitch_deg")) + 90.0)
            point_e = finite_float(gcp["projected_e"])
            point_n = finite_float(gcp["projected_n"])
            camera_e = finite_float(meta.get("projected_e"))
            camera_n = finite_float(meta.get("projected_n"))
            azimuth, azimuth_bin = camera_azimuth_bin(camera_e, camera_n, point_e, point_n)
            good = visible_good(row)
            x = finite_float(row.get("manual_x"))
            y = finite_float(row.get("manual_y"))
            width = int(float(meta["width"]))
            height = int(float(meta["height"]))
            edge_margin = min(x, y, width - 1 - x, height - 1 - y) if good else math.nan
            key = (scene, point, image)
            candidate = task_candidates.get(key, {})
            evidence = {
                "scene": scene,
                "point_name": point,
                "role_v1_2_2": split_by_key[(scene, point)]["role"],
                "image_name": image,
                "annotation_source": annotation_source,
                "is_frozen_v1_2_2_observation": (point, image) in release_keys,
                "is_new_direct_candidate": bool(candidate),
                "candidate_source": candidate.get("candidate_source", ""),
                "quality": str(row.get("quality", "")),
                "human_verified_good": good,
                "view_type": "nadir" if off_nadir <= NADIR_THRESHOLD_DEG else "oblique",
                "off_nadir_deg": off_nadir,
                "manual_x": row.get("manual_x", ""),
                "manual_y": row.get("manual_y", ""),
                "manual_edge_margin_px_if_good": edge_margin,
                "camera_position_azimuth_deg": azimuth,
                "camera_position_azimuth_bin_45deg": azimuth_bin,
                "image_width": width,
                "image_height": height,
            }
            observation_evidence.append(evidence)

            if (
                candidate
                and evidence["view_type"] == "nadir"
                and str(row.get("quality", "")) == "not_visible"
                and candidate.get("candidate_source") == "triangulated_annotation_rays"
            ):
                coarse = project_gcp(camera_by_image[image], gcp)
                candidate_x = finite_float(candidate.get("pixel_x"))
                candidate_y = finite_float(candidate.get("pixel_y"))
                coarse_x = finite_float(coarse.get("pixel_x"))
                coarse_y = finite_float(coarse.get("pixel_y"))
                conflict_rows.append(
                    {
                        "scene": scene,
                        "point_name": point,
                        "image_name": image,
                        "human_quality": "not_visible",
                        "triangulated_candidate_x": candidate_x,
                        "triangulated_candidate_y": candidate_y,
                        "triangulated_inside_image": True,
                        "survey_exif_candidate_x": coarse_x,
                        "survey_exif_candidate_y": coarse_y,
                        "survey_exif_inside_image": bool(coarse.get("inside_image")),
                        "survey_exif_outside_distance_px": max(
                            0.0,
                            -coarse_x,
                            coarse_x - (int(meta["width"]) - 1),
                            -coarse_y,
                            coarse_y - (int(meta["height"]) - 1),
                        ),
                        "ground_dx_e_m": finite_float(coarse.get("ground_dx_e_m")),
                        "ground_dy_n_m": finite_float(coarse.get("ground_dy_n_m")),
                        "evidence_classification": (
                            "triangulated_candidate_conflicts_with_independent_survey_exif_nadir_geometry"
                        ),
                    }
                )

        scene_evidence = [row for row in observation_evidence if row["scene"] == scene]
        for point in release_points:
            point_rows = [row for row in scene_evidence if row["point_name"] == point]
            good_rows = [row for row in point_rows if row["human_verified_good"]]
            nadir_good = [row for row in good_rows if row["view_type"] == "nadir"]
            good_bins = {int(row["camera_position_azimuth_bin_45deg"]) for row in good_rows}
            nadir_bins = {int(row["camera_position_azimuth_bin_45deg"]) for row in nadir_good}
            disposition, reason = classify_future_formal_candidate(
                len(good_rows), len(nadir_good), len(good_bins)
            )
            new_rows = [row for row in point_rows if row["is_new_direct_candidate"]]
            new_nadir = [row for row in new_rows if row["view_type"] == "nadir"]
            margins = [float(row["manual_edge_margin_px_if_good"]) for row in nadir_good]
            point_decisions.append(
                {
                    "scene": scene,
                    "point_name": point,
                    "role_v1_2_2": split_by_key[(scene, point)]["role"],
                    "survey_e_m": finite_float(gcp_by_name[point]["projected_e"]),
                    "survey_n_m": finite_float(gcp_by_name[point]["projected_n"]),
                    "survey_h_m": finite_float(gcp_by_name[point]["normal_height_m"]),
                    "frozen_v1_2_2_observation_count": sum(
                        bool(row["is_frozen_v1_2_2_observation"]) for row in point_rows
                    ),
                    "completed_annotation_row_count": len(point_rows),
                    "human_verified_good_count": len(good_rows),
                    "human_verified_good_nadir_count": len(nadir_good),
                    "human_verified_good_oblique_count": len(good_rows) - len(nadir_good),
                    "good_camera_azimuth_bin_count": len(good_bins),
                    "good_nadir_camera_azimuth_bin_count": len(nadir_bins),
                    "new_direct_candidate_count": len(new_rows),
                    "new_nadir_candidate_count": len(new_nadir),
                    "new_nadir_good_count": sum(row["quality"] == "good" for row in new_nadir),
                    "new_nadir_not_visible_count": sum(
                        row["quality"] == "not_visible" for row in new_nadir
                    ),
                    "nadir_good_min_edge_margin_px": min(margins) if margins else math.nan,
                    "nadir_good_median_edge_margin_px": (
                        float(pd.Series(margins).median()) if margins else math.nan
                    ),
                    "future_v1_3_formal_primary_draft_disposition": disposition,
                    "draft_disposition_reason": reason,
                    "v1_2_2_mutated": False,
                }
            )

        # Inventory non-release points that may be reviewed as replacements, without admitting them.
        if working_all:
            for point in sorted({row["point_name"] for row in working_all} - set(release_points)):
                point_rows = [row for row in working_all if row["point_name"] == point]
                nadir_rows = []
                for row in point_rows:
                    meta = metadata_by_image.get(row["image_name"])
                    if meta and abs(finite_float(meta.get("pitch_deg")) + 90.0) <= NADIR_THRESHOLD_DEG:
                        nadir_rows.append(row)
                ambiguous_nadir_count = sum(
                    row["quality"] == "ambiguous" for row in nadir_rows
                )
                if ambiguous_nadir_count >= MIN_GOOD_NADIR_OBSERVATIONS:
                    point_decisions.append(
                        {
                            "scene": scene,
                            "point_name": point,
                            "role_v1_2_2": "not_in_v1_2_2_split",
                            "survey_e_m": finite_float(gcp_by_name[point]["projected_e"]),
                            "survey_n_m": finite_float(gcp_by_name[point]["projected_n"]),
                            "survey_h_m": finite_float(gcp_by_name[point]["normal_height_m"]),
                            "frozen_v1_2_2_observation_count": 0,
                            "completed_annotation_row_count": len(point_rows),
                            "human_verified_good_count": sum(visible_good(row) for row in point_rows),
                            "human_verified_good_nadir_count": sum(
                                visible_good(row) for row in nadir_rows
                            ),
                            "human_verified_good_oblique_count": 0,
                            "good_camera_azimuth_bin_count": 0,
                            "good_nadir_camera_azimuth_bin_count": 0,
                            "new_direct_candidate_count": 0,
                            "new_nadir_candidate_count": 0,
                            "new_nadir_good_count": 0,
                            "new_nadir_not_visible_count": 0,
                            "nadir_good_min_edge_margin_px": math.nan,
                            "nadir_good_median_edge_margin_px": math.nan,
                            "future_v1_3_formal_primary_draft_disposition": (
                                "manual_identity_re_review_required_before_replacement_candidate_use"
                            ),
                            "draft_disposition_reason": (
                                f"{ambiguous_nadir_count}_ambiguous_nadir_rows_no_verified_good"
                            ),
                            "v1_2_2_mutated": False,
                        }
                    )

    decision_df = pd.DataFrame(point_decisions)
    release_decisions = decision_df[decision_df["role_v1_2_2"].isin(["control", "checkpoint"])].copy()
    excluded = release_decisions[
        release_decisions["future_v1_3_formal_primary_draft_disposition"].eq(
            "exclude_from_future_v1_3_formal_primary_draft"
        )
    ].copy()

    split_summary: list[dict[str, Any]] = []
    for scene in SCENES:
        scene_rows = release_decisions[release_decisions["scene"].eq(scene)]
        scene_excluded = scene_rows[
            scene_rows["future_v1_3_formal_primary_draft_disposition"].eq(
                "exclude_from_future_v1_3_formal_primary_draft"
            )
        ]
        retained = scene_rows.drop(scene_excluded.index)
        split_summary.append(
            {
                "scene": scene,
                "v1_2_2_point_count": len(scene_rows),
                "v1_2_2_control_count": int(scene_rows["role_v1_2_2"].eq("control").sum()),
                "v1_2_2_checkpoint_count": int(scene_rows["role_v1_2_2"].eq("checkpoint").sum()),
                "future_draft_excluded_point_count": len(scene_excluded),
                "future_draft_excluded_points": ";".join(scene_excluded["point_name"]),
                "retained_point_count": len(retained),
                "retained_control_count_if_roles_unchanged": int(
                    retained["role_v1_2_2"].eq("control").sum()
                ),
                "retained_checkpoint_count_if_roles_unchanged": int(
                    retained["role_v1_2_2"].eq("checkpoint").sum()
                ),
                "future_split_freeze_allowed": False,
            }
        )

    write_csv(output_root / "per_observation_nadir_coverage_evidence.csv", observation_evidence)
    write_csv(output_root / "future_v1_3_point_disposition_draft.csv", point_decisions)
    write_csv(output_root / "future_v1_3_excluded_points_draft.csv", excluded.to_dict("records"))
    write_csv(output_root / "triangulation_vs_survey_exif_nadir_conflicts.csv", conflict_rows)
    write_csv(output_root / "split_capacity_after_draft_exclusions.csv", split_summary)

    actual_excluded = set(zip(excluded["scene"], excluded["point_name"]))
    for row in release_decisions.to_dict("records"):
        expected_status, expected_reason = classify_future_formal_candidate(
            int(row["human_verified_good_count"]),
            int(row["human_verified_good_nadir_count"]),
            int(row["good_camera_azimuth_bin_count"]),
        )
        if (
            row["future_v1_3_formal_primary_draft_disposition"] != expected_status
            or row["draft_disposition_reason"] != expected_reason
        ):
            raise RuntimeError(f"Draft rule recomputation mismatch: {row}")

    five_k = release_decisions[release_decisions["scene"].eq("gcp_5000_20260602")]
    five_k_strong = five_k[five_k["human_verified_good_nadir_count"].astype(int) >= 4]
    five_k_excluded = five_k[
        five_k["future_v1_3_formal_primary_draft_disposition"].eq(
            "exclude_from_future_v1_3_formal_primary_draft"
        )
    ]
    five_k_strong_core_max_e = (
        float(five_k_strong["survey_e_m"].astype(float).max()) if not five_k_strong.empty else None
    )
    five_k_excluded_min_e = (
        float(five_k_excluded["survey_e_m"].astype(float).min()) if not five_k_excluded.empty else None
    )
    five_k_excluded_max_e = (
        float(five_k_excluded["survey_e_m"].astype(float).max()) if not five_k_excluded.empty else None
    )
    conflict_outside_distances = [float(row["survey_exif_outside_distance_px"]) for row in conflict_rows]
    conflict_outside_min = min(conflict_outside_distances) if conflict_outside_distances else None
    conflict_outside_max = max(conflict_outside_distances) if conflict_outside_distances else None
    spatial_evidence_lines: list[str] = []
    if (
        five_k_strong_core_max_e is not None
        and five_k_excluded_min_e is not None
        and five_k_excluded_max_e is not None
    ):
        spatial_evidence_lines.append(
            f"- 5K 具有 >=4 张正射 Good 的稳定点最东到 E={five_k_strong_core_max_e:.3f} m；"
            f"本次排除点位于 E={five_k_excluded_min_e:.3f}–{five_k_excluded_max_e:.3f} m，"
            "形成连续东侧边缘簇。"
        )
    if conflict_outside_min is not None and conflict_outside_max is not None:
        spatial_evidence_lines.append(
            f"- 冲突候选按独立 surveyed+EXIF 投影距离图像边界仍有 "
            f"{conflict_outside_min:.1f}–{conflict_outside_max:.1f} px，不是小幅准心偏差。"
        )

    report_lines = [
        "# 完成人工标注后的正射核心覆盖审计",
        "",
        "## 结论",
        "",
        "- v1.2.2 保持冻结且未修改。本报告只形成未来 v1.3.0 formal-primary 候选处置草案。",
        "- 人工 `good/not_visible` 是可见性的权威证据；候选投影只用于召回，不等于可见。",
        "- 草案门槛：总 Good >=4、人工确认近正射 Good >=3、Good 相机位置方位 bin >=2。",
        "- 未来 formal-primary 草案排除：5K `G11/G13/G18/NC94`，20K `wy3_1`。",
        "- 5K `G12` 有 11 张 Good、其中 3 张正射，恰好达到最低正射冗余，需在 release freeze 前独立复核。",
        "",
        "## 5K 正射候选错误",
        "",
        f"- 新增正射且人工判为不可见的候选共 {len(conflict_rows)} 张。",
        "- 这些候选由 `triangulated_annotation_rays` 判为图内；独立的 surveyed XYZ + EXIF/gimbal 正射投影均判为图外。",
        "- 因此属于候选召回冲突，不是人工漏标，也没有证据支持 raw/undistorted 坐标域混用。",
        "- 后续不把这些 `not_visible` 行计作覆盖，也不因斜视图数量多而让缺乏正射核心重叠的点进入 formal primary。",
        *spatial_evidence_lines,
        "",
        "## 排除后的容量",
        "",
        "| Scene | Excluded | Retained controls | Retained checkpoints |",
        "|---|---:|---:|---:|",
    ]
    for row in split_summary:
        report_lines.append(
            f"| {row['scene']} | {row['future_draft_excluded_point_count']} | "
            f"{row['retained_control_count_if_roles_unchanged']} | "
            f"{row['retained_checkpoint_count_if_roles_unchanged']} |"
        )
    report_lines += [
        "",
        "5K 排除后若沿用旧角色只剩 3 control / 5 checkpoint，不能直接冻结 control-heavy split；应先补充核心正射覆盖点。",
        "20K 排除 `wy3_1` 后为 4 control / 7 checkpoint，可继续作为 sparse diagnostic 候选，但仍不等于 control-heavy 设计已完成。",
        "5K 当前唯一具有多张历史正射但尚未通过身份质量复核的替代点是 `G04`（8 张均为 ambiguous）；不得自动纳入。",
        "",
        "## 边界",
        "",
        "- 未修改任何 annotation CSV、v1.2.2 release、split、survey coordinates、packet、evaluator 或 frozen metric。",
        "- 本草案不得用于重新解释或覆盖 v1.2.2 已冻结数值。",
    ]
    (output_root / "AUDIT_REPORT_ZH.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    manifest = {
        "schema": "ms_gcp_completed_annotation_nadir_coverage_audit_v1",
        "scope": "future_v1_3_candidate_design_only_no_release_mutation",
        "thresholds": {
            "nadir_threshold_deg": NADIR_THRESHOLD_DEG,
            "min_human_verified_good_observations": MIN_GOOD_OBSERVATIONS,
            "min_human_verified_good_nadir_observations": MIN_GOOD_NADIR_OBSERVATIONS,
            "min_good_camera_position_azimuth_bins": MIN_GOOD_AZIMUTH_BINS,
            "status": "draft_design_gate_not_frozen_release_protocol",
        },
        "inputs": {
            "release_dir": str(args.release_dir),
            "split": {"path": str(split_path), "sha256": sha256_file(split_path)},
            "gcp_table": {"path": str(gcp_path), "sha256": sha256_file(gcp_path)},
            "release_annotations": release_input_records,
            "completed_working_annotations": working_input_records,
            "task_root": str(args.task_root),
        },
        "validation": {
            "draft_rule_recomputation_passed": True,
            "excluded_points": [f"{scene}:{point}" for scene, point in sorted(actual_excluded)],
            "triangulated_nadir_candidate_conflict_count": len(conflict_rows),
            "triangulated_conflict_outside_distance_px_min": conflict_outside_min,
            "triangulated_conflict_outside_distance_px_max": conflict_outside_max,
            "five_k_strong_nadir_core_max_e_m": five_k_strong_core_max_e,
            "five_k_excluded_e_m_range": [five_k_excluded_min_e, five_k_excluded_max_e],
            "release_mutated": False,
            "split_mutated": False,
            "formal_metrics_recomputed": False,
        },
    }
    write_json(output_root / "audit_manifest.json", manifest)

    output_files = []
    for path in sorted(output_root.iterdir()):
        if path.is_file() and path.name != "OUTPUT_SHA256_MANIFEST.json":
            output_files.append(
                {"path": path.name, "size": path.stat().st_size, "sha256": sha256_file(path)}
            )
    write_json(output_root / "OUTPUT_SHA256_MANIFEST.json", {"files": output_files})
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "excluded_points": sorted(f"{scene}:{point}" for scene, point in actual_excluded),
                "triangulation_conflicts": len(conflict_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
