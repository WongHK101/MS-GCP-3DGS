#!/usr/bin/env python3
"""Prepare a uniform, residual-blind v1.3 supplemental annotation pass.

Existing annotation rows are immutable inputs.  The task only adds corrected,
previously unattempted image candidates needed to meet the predeclared
multi-view coverage targets.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from audit_v13_candidate_recall_and_gps import (
    ANNOTATION_RELATIVE_PATHS,
    COORDINATE_DOMAIN,
    SCENES,
    annotation_ray,
    azimuth_bin,
    azimuth_deg,
    camera_indices,
    candidate_pool_for_point,
    git_output,
    image_metadata,
    is_nadir,
    leave_one_out_rows,
    sha256_file,
    write_json,
    write_launcher,
)
from prepare_direct_multiview_annotation_tasks import CANDIDATE_FIELDS, write_candidate_csv
from prepare_followup_annotation_tasks_50k100k import visible_good


TARGET_TOTAL_GOOD = 8
TARGET_NADIR_GOOD = 4
TARGET_OBLIQUE_GOOD = 4
TARGET_AZIMUTH_BINS = 4
VISIBILITY_RESERVE_PER_DEFICIENT_CLASS = 2
MAX_NEW_CANDIDATES_PER_POINT = 8
ROBUST_DISCOVERY_ABS_PX = 10.0
ROBUST_DISCOVERY_MEDIAN_FACTOR = 3.0

RECOVERED_EXTRA_POINTS = {("gcp_100000_20260610", "G33")}
EXCLUDED_DIAGNOSTIC_POINTS = {("gcp_50000_20260610", "dyl2")}


def current_good_summary(
    frame: pd.DataFrame,
    point_name: str,
    images: dict[str, dict[str, Any]],
    metadata: dict[str, dict[str, Any]],
    xyz: np.ndarray,
) -> dict[str, Any]:
    rows = [
        row
        for row in frame[frame["point_name"].eq(point_name)].to_dict("records")
        if visible_good(row) and str(row["image_name"]) in images
    ]
    types = ["nadir" if is_nadir(metadata.get(str(row["image_name"]))) else "oblique" for row in rows]
    bins = {azimuth_bin(azimuth_deg(xyz, images[str(row["image_name"])])) for row in rows}
    return {
        "good_view_count": len(rows),
        "good_nadir_count": types.count("nadir"),
        "good_oblique_count": types.count("oblique"),
        "good_azimuth_bin_count": len(bins),
        "good_azimuth_bins": sorted(bins),
    }


def robust_discovery_frame(
    scene: str,
    point_name: str,
    frame: pd.DataFrame,
    cameras: dict[int, dict[str, Any]],
    images: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = [
        row
        for row in frame[frame["point_name"].eq(point_name)].to_dict("records")
        if visible_good(row) and str(row["image_name"]) in images
    ]
    audit = {
        "candidate_discovery_excluded_view": "",
        "candidate_discovery_outlier_rule_triggered": False,
        "candidate_discovery_worst_loo_px": np.nan,
        "candidate_discovery_median_loo_px": np.nan,
    }
    if len(rows) < 4:
        return frame, audit
    loo = leave_one_out_rows(scene, point_name, rows, cameras, images)
    finite = [row for row in loo if math.isfinite(float(row.get("pixel_error", np.nan)))]
    if len(finite) < 4:
        return frame, audit
    values = np.asarray([float(row["pixel_error"]) for row in finite], dtype=np.float64)
    worst_index = int(np.argmax(values))
    worst = finite[worst_index]
    worst_value = float(values[worst_index])
    median_value = float(np.median(values))
    audit["candidate_discovery_worst_loo_px"] = worst_value
    audit["candidate_discovery_median_loo_px"] = median_value
    if worst_value <= ROBUST_DISCOVERY_ABS_PX or worst_value <= ROBUST_DISCOVERY_MEDIAN_FACTOR * median_value:
        return frame, audit
    output = frame.copy()
    mask = output["point_name"].eq(point_name) & output["image_name"].eq(str(worst["hidden_image_name"]))
    if int(mask.sum()) != 1:
        raise RuntimeError(f"{scene}/{point_name}: robust candidate row is not unique")
    output.loc[mask, "quality"] = "candidate_discovery_geometry_outlier_only"
    audit["candidate_discovery_excluded_view"] = str(worst["hidden_image_name"])
    audit["candidate_discovery_outlier_rule_triggered"] = True
    return output, audit


def round_robin_bins(rows: list[dict[str, Any]], count: int, existing_bins: set[int]) -> list[dict[str, Any]]:
    available = sorted(
        rows,
        key=lambda row: (
            int(row["azimuth_bin_45deg"]) in existing_bins,
            float(row["edge_margin_px"]) < 96.0,
            -float(row["center_score"]),
            -float(row["edge_margin_px"]),
            str(row["image_name"]),
        ),
    )
    selected: list[dict[str, Any]] = []
    per_bin: dict[int, int] = {}
    while available and len(selected) < count:
        index = min(
            range(len(available)),
            key=lambda i: (
                per_bin.get(int(available[i]["azimuth_bin_45deg"]), 0),
                int(available[i]["azimuth_bin_45deg"]) in existing_bins,
                -float(available[i]["center_score"]),
                str(available[i]["image_name"]),
            ),
        )
        row = available.pop(index)
        selected.append(row)
        bin_id = int(row["azimuth_bin_45deg"])
        per_bin[bin_id] = per_bin.get(bin_id, 0) + 1
    return selected


def select_uniform_supplement(
    pool: list[dict[str, Any]],
    summary: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    available = [row for row in pool if not bool(row["already_attempted"])]
    existing_bins = set(int(value) for value in summary["good_azimuth_bins"])
    selected: list[dict[str, Any]] = []

    deficits = {
        "nadir": max(0, TARGET_NADIR_GOOD - int(summary["good_nadir_count"])),
        "oblique": max(0, TARGET_OBLIQUE_GOOD - int(summary["good_oblique_count"])),
    }
    for view_type in ("nadir", "oblique"):
        deficit = deficits[view_type]
        if deficit <= 0:
            continue
        candidates = [row for row in available if row["view_type"] == view_type]
        quota = min(
            len(candidates),
            deficit + VISIBILITY_RESERVE_PER_DEFICIENT_CLASS,
            MAX_NEW_CANDIDATES_PER_POINT - len(selected),
        )
        chosen = round_robin_bins(candidates, quota, existing_bins)
        selected.extend(chosen)
        chosen_names = {str(row["image_name"]) for row in chosen}
        available = [row for row in available if str(row["image_name"]) not in chosen_names]
        existing_bins.update(int(row["azimuth_bin_45deg"]) for row in chosen)

    total_deficit_after_planned = max(
        0,
        TARGET_TOTAL_GOOD - int(summary["good_view_count"]) - len(selected),
    )
    bin_deficit_after_planned = max(0, TARGET_AZIMUTH_BINS - len(existing_bins))
    fill = max(total_deficit_after_planned, bin_deficit_after_planned)
    if fill > 0 and len(selected) < MAX_NEW_CANDIDATES_PER_POINT:
        chosen = round_robin_bins(
            available,
            min(fill + VISIBILITY_RESERVE_PER_DEFICIENT_CLASS, MAX_NEW_CANDIDATES_PER_POINT - len(selected)),
            existing_bins,
        )
        selected.extend(chosen)
        existing_bins.update(int(row["azimuth_bin_45deg"]) for row in chosen)

    output: list[dict[str, Any]] = []
    for rank, row in enumerate(selected, start=1):
        item = dict(row)
        item["rank_for_gcp"] = rank
        item["task_action"] = "uniform_fixed_candidate_multiview_supplement"
        output.append(item)
    plan = {
        "nadir_good_deficit": deficits["nadir"],
        "oblique_good_deficit": deficits["oblique"],
        "total_good_deficit": max(0, TARGET_TOTAL_GOOD - int(summary["good_view_count"])),
        "azimuth_bin_deficit": max(0, TARGET_AZIMUTH_BINS - int(summary["good_azimuth_bin_count"])),
        "selected_candidate_count": len(output),
        "selected_nadir_count": sum(row["view_type"] == "nadir" for row in output),
        "selected_oblique_count": sum(row["view_type"] == "oblique" for row in output),
        "available_nadir_count": sum(row["view_type"] == "nadir" for row in available),
        "available_oblique_count": sum(row["view_type"] == "oblique" for row in available),
    }
    return output, plan


def normalized_candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{field: row.get(field, "") for field in CANDIDATE_FIELDS} for row in rows]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(r"E:\M3M-GCP-3DGS"))
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
    parser.add_argument("--stamp", default=time.strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    generator_repo = Path(__file__).resolve().parents[2]
    if git_output(generator_repo, "status", "--porcelain"):
        raise RuntimeError("Uniform supplement generation requires a clean worktree")
    output_root = args.repo / "outputs" / f"gcp_v13_uniform_fixed_candidate_supplement_{args.stamp}"
    output_root.mkdir(parents=True, exist_ok=False)
    candidate_dir = output_root / "candidate_lists"
    launcher_dir = output_root / "launchers"
    candidate_dir.mkdir()
    launcher_dir.mkdir()

    remote = json.loads(args.remote_manifest.read_text(encoding="utf-8"))
    split = pd.read_csv(args.split_candidate, dtype=str, keep_default_na=False)
    point_keys = set(zip(split["scene"], split["point_name"])) | RECOVERED_EXTRA_POINTS
    point_keys -= EXCLUDED_DIAGNOSTIC_POINTS

    all_selected: list[dict[str, Any]] = []
    summaries = []
    input_hashes = []
    annotations: dict[str, Path] = {}
    for scene in SCENES:
        annotation_path = args.repo / ANNOTATION_RELATIVE_PATHS[scene]
        annotations[scene] = annotation_path
        frame = pd.read_csv(annotation_path, dtype=str, keep_default_na=False)
        if frame[["scene", "point_name", "image_name"]].duplicated().any():
            raise RuntimeError(f"{scene}: duplicate annotation key")
        cameras, images = camera_indices(remote["scenes"][scene])
        metadata = image_metadata(args.candidate_root, scene)
        input_hashes.append({"scene": scene, "path": str(annotation_path), "sha256": sha256_file(annotation_path)})

        for _, point_name in sorted(key for key in point_keys if key[0] == scene):
            discovery_frame, robust_audit = robust_discovery_frame(
                scene, point_name, frame, cameras, images
            )
            xyz, pool, _, candidate_summary = candidate_pool_for_point(
                scene,
                point_name,
                discovery_frame,
                cameras,
                images,
                metadata,
            )
            current = current_good_summary(frame, point_name, images, metadata, xyz)
            selected, plan = select_uniform_supplement(pool, current)
            all_selected.extend(selected)
            row = {
                "scene": scene,
                "point_name": point_name,
                **current,
                **plan,
                **robust_audit,
                "corrected_in_bounds_image_count": candidate_summary["corrected_in_bounds_image_count"],
                "unattempted_corrected_candidate_count": candidate_summary["unattempted_corrected_candidate_count"],
                "target_total_good": TARGET_TOTAL_GOOD,
                "target_nadir_good": TARGET_NADIR_GOOD,
                "target_oblique_good": TARGET_OBLIQUE_GOOD,
                "target_azimuth_bins": TARGET_AZIMUTH_BINS,
                "status": "supplement_generated" if selected else "coverage_target_already_met",
            }
            if plan["nadir_good_deficit"] and plan["selected_nadir_count"] == 0:
                row["status"] = "unresolved_no_nadir_candidate"
            if plan["oblique_good_deficit"] and plan["selected_oblique_count"] == 0:
                row["status"] = "unresolved_no_oblique_candidate"
            summaries.append(row)

    selected_df = pd.DataFrame(all_selected)
    summary_df = pd.DataFrame(summaries)
    selected_df.to_csv(output_root / "all_selected_uniform_supplement_candidates.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(output_root / "uniform_supplement_point_summary.csv", index=False, encoding="utf-8-sig")

    launcher_records = []
    for scene, group in selected_df.groupby("scene", sort=True):
        rows = group.sort_values(["point_name", "rank_for_gcp", "image_name"]).to_dict("records")
        path = candidate_dir / f"{scene}_uniform_fixed_supplement.csv"
        write_candidate_csv(path, normalized_candidate_rows(rows))
        launcher = launcher_dir / f"launch_{scene}_uniform_fixed_supplement.ps1"
        write_launcher(
            launcher,
            generator_repo,
            path,
            annotations[scene],
            Path(r"E:\datasets\M3M-GCP\scenes") / scene,
        )
        launcher_records.append(
            {
                "scene": scene,
                "candidate_rows": len(rows),
                "candidate_csv": str(path),
                "launcher": str(launcher),
                "annotation_output": str(annotations[scene]),
            }
        )

    readme = f"""# v1.3 全点统一修正候选补标

## 固定规则

- 沿用所有已有标注，不返工、不覆盖。
- 目标：每点至少 {TARGET_TOTAL_GOOD} 个 Good；正射至少 {TARGET_NADIR_GOOD}；倾斜至少 {TARGET_OBLIQUE_GOOD}；至少 {TARGET_AZIMUTH_BINS} 个 45 度方位桶。
- 对每个不足类别选择 `缺口 + {VISIBILITY_RESERVE_PER_DEFICIENT_CLASS}` 张候选，每点最多 {MAX_NEW_CANDIDATES_PER_POINT} 张。
- 候选只使用修正后的可逆 SIMPLE_RADIAL 主分支、最新 Good annotation rays 和完整 raw COLMAP 相机轨迹。
- 不读取 residual、RMSE、depth、alpha、variance 或 3DGS scatter。
- 50K dyl2 无正射覆盖，保持 diagnostic，不生成强行补标任务。
- 100K G33 已由首轮补标恢复为 15 个 Good，本轮作为 recovered candidate point 一并检查统一覆盖。

## G39 候选发现保护

固定 leave-one-view-out 规则仅用于候选三角化：当唯一最差视图同时超过 {ROBUST_DISCOVERY_ABS_PX:.1f}px 和中位数的 {ROBUST_DISCOVERY_MEDIAN_FACTOR:.1f} 倍时，不用该视图估计后续候选中心，但不会删除或修改原 annotation。触发明细记录在 summary 中。

## 标注

按场景依次运行 `launchers`。看见且身份唯一才标 Good；不可见、遮挡或不确定则标 Not visible/Ambiguous。所有坐标继续保存为 raw image zero-based pixel-center。
"""
    (output_root / "README_zh.md").write_text(readme, encoding="utf-8")
    write_json(
        output_root / "task_manifest.json",
        {
            "schema": "ms_gcp_v13_uniform_fixed_candidate_supplement_v1",
            "status": "working_annotation_tasks_not_release",
            "generator": {
                "repo": str(generator_repo),
                "commit": git_output(generator_repo, "rev-parse", "HEAD"),
                "branch": git_output(generator_repo, "branch", "--show-current"),
                "clean": True,
                "script": str(Path(__file__).resolve()),
                "script_sha256": sha256_file(Path(__file__).resolve()),
            },
            "inputs": {
                "annotations": input_hashes,
                "remote_manifest": {"path": str(args.remote_manifest), "sha256": sha256_file(args.remote_manifest)},
                "split_candidate": {"path": str(args.split_candidate), "sha256": sha256_file(args.split_candidate)},
            },
            "policy": {
                "target_total_good": TARGET_TOTAL_GOOD,
                "target_nadir_good": TARGET_NADIR_GOOD,
                "target_oblique_good": TARGET_OBLIQUE_GOOD,
                "target_azimuth_bins": TARGET_AZIMUTH_BINS,
                "visibility_reserve": VISIBILITY_RESERVE_PER_DEFICIENT_CLASS,
                "max_new_candidates_per_point": MAX_NEW_CANDIDATES_PER_POINT,
                "annotation_domain": COORDINATE_DOMAIN,
                "existing_annotations_preserved": True,
                "model_residuals_used": False,
            },
            "excluded_diagnostic_points": [list(key) for key in sorted(EXCLUDED_DIAGNOSTIC_POINTS)],
            "recovered_extra_points": [list(key) for key in sorted(RECOVERED_EXTRA_POINTS)],
            "launchers": launcher_records,
        },
    )
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
    print(pd.DataFrame(launcher_records).to_string(index=False))
    print(summary_df["status"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
