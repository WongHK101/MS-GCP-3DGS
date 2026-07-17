#!/usr/bin/env python3
"""Audit whether GCP errors are associated with image GPS/pose outliers.

This is a diagnostic-only audit. It never uses the result for candidate
selection, point inclusion, or control/checkpoint assignment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from audit_v13_candidate_recall_and_gps import (
    ANNOTATION_RELATIVE_PATHS,
    SCENES,
    camera_center,
    camera_indices,
    geodetic_to_enu,
    image_metadata,
    leave_one_out_rows,
    sha256_file,
)
from prepare_followup_annotation_tasks_50k100k import visible_good


RELEASE_FILE_PATTERN = "{scene}_gcp_annotations_pixel_domain_v1_2_2.csv"
G39_IMAGE = "DJI_20260610161948_0002_D.JPG"
SCENE_50K = "gcp_50000_20260610"


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def working_good_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows = [row for row in frame.to_dict("records") if visible_good(row)]
    return pd.DataFrame(rows)


def release_rows(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    return frame.rename(
        columns={
            "raw_image_name": "image_name",
            "raw_manual_x": "manual_x",
            "raw_manual_y": "manual_y",
        }
    )


def scene_image_gps_qc(
    scene: str,
    scene_record: dict[str, Any],
    metadata_by_name: dict[str, dict[str, Any]],
    alignment_summary: dict[str, Any],
) -> pd.DataFrame:
    _, images = camera_indices(scene_record)
    origin = alignment_summary["enu_origin_lat_lon_alt"]
    rows: list[dict[str, Any]] = []
    for image_name, image in images.items():
        metadata = metadata_by_name.get(image_name)
        if metadata is None:
            continue
        try:
            gps_enu = geodetic_to_enu(
                float(metadata["lat"]),
                float(metadata["lon"]),
                float(metadata["ellipsoid_alt_m"]),
                origin,
            )
        except (KeyError, TypeError, ValueError):
            continue
        center = camera_center(image)
        delta = center - gps_enu
        rows.append(
            {
                "scene": scene,
                "image_name": image_name,
                "image_id": int(image["image_id"]),
                "camera_id": int(image["camera_id"]),
                "gps_to_colmap_e_m": float(delta[0]),
                "gps_to_colmap_n_m": float(delta[1]),
                "gps_to_colmap_u_m": float(delta[2]),
                "gps_to_colmap_3d_m": float(np.linalg.norm(delta)),
                "image_pose_record_sha256": str(image["record_sha256"]),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError(f"{scene}: no image GPS/COLMAP associations")
    result["gps_residual_rank_desc"] = result["gps_to_colmap_3d_m"].rank(
        method="min", ascending=False
    ).astype(int)
    result["gps_residual_percentile"] = result["gps_to_colmap_3d_m"].rank(
        method="average", pct=True
    ) * 100.0
    return result.sort_values(["scene", "image_name"]).reset_index(drop=True)


def loo_for_frame(
    scene: str,
    frame: pd.DataFrame,
    cameras: dict[int, dict[str, Any]],
    images: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    for point_name, group in frame.groupby("point_name", sort=True):
        rows = group.to_dict("records")
        if len(rows) < 3:
            continue
        output.extend(leave_one_out_rows(scene, str(point_name), rows, cameras, images))
    return pd.DataFrame(output)


def add_observation_diagnostics(
    annotation_set: str,
    scene: str,
    frame: pd.DataFrame,
    image_qc: pd.DataFrame,
    loo: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["scene", "point_name", "image_name"]
    if frame[keys].duplicated().any():
        raise RuntimeError(f"{annotation_set}/{scene}: duplicate observation key")
    result = frame[[*keys, "manual_x", "manual_y"]].copy()
    result.insert(0, "annotation_set", annotation_set)
    result = result.merge(image_qc, on=["scene", "image_name"], how="left", validate="many_to_one")
    if result["gps_to_colmap_3d_m"].isna().any():
        missing = sorted(result.loc[result["gps_to_colmap_3d_m"].isna(), "image_name"].unique())
        raise RuntimeError(f"{annotation_set}/{scene}: missing image GPS QC: {missing[:5]}")
    if not loo.empty:
        selected = loo[["scene", "point_name", "hidden_image_name", "pixel_error"]].rename(
            columns={
                "hidden_image_name": "image_name",
                "pixel_error": "annotation_loo_pixel_error",
            }
        )
        result = result.merge(selected, on=keys, how="left", validate="one_to_one")
    else:
        result["annotation_loo_pixel_error"] = np.nan
    return result


def aggregate_points(observations: pd.DataFrame, residuals: pd.DataFrame) -> pd.DataFrame:
    formal = observations[observations["annotation_set"].eq("v1.2.2_frozen")].copy()

    def finite_percentile(values: pd.Series, percentile: float) -> float:
        finite = pd.to_numeric(values, errors="coerce").dropna()
        return float(np.percentile(finite, percentile)) if len(finite) else float("nan")

    aggregations = formal.groupby(["scene", "point_name"], as_index=False).agg(
        observation_count=("image_name", "size"),
        gps_residual_median_m=("gps_to_colmap_3d_m", "median"),
        gps_residual_p95_m=("gps_to_colmap_3d_m", lambda values: finite_percentile(values, 95)),
        gps_residual_max_m=("gps_to_colmap_3d_m", "max"),
        max_gps_residual_percentile=("gps_residual_percentile", "max"),
        gps_top_0_5pct_view_count=("gps_residual_percentile", lambda values: int((values >= 99.5).sum())),
        loo_pixel_error_median=("annotation_loo_pixel_error", "median"),
        loo_pixel_error_p95=("annotation_loo_pixel_error", lambda values: finite_percentile(values, 95)),
        loo_pixel_error_max=("annotation_loo_pixel_error", "max"),
    )
    keep = [
        "scene",
        "point_name",
        "role",
        "error_h_m",
        "error_z_m",
        "error_3d_m",
        "scatter_median_m",
        "scatter_p90_m",
        "scatter_max_m",
    ]
    return aggregations.merge(residuals[keep], on=["scene", "point_name"], how="left", validate="one_to_one")


def rank_correlation(frame: pd.DataFrame, left: str, right: str) -> float | None:
    pair = frame[[left, right]].dropna()
    if len(pair) < 3 or pair[left].nunique() < 2 or pair[right].nunique() < 2:
        return None
    return float(pair[left].rank(method="average").corr(pair[right].rank(method="average")))


def classify_high_error_points(
    points: pd.DataFrame,
    high_error: pd.DataFrame,
    observations: pd.DataFrame,
    feature_qc: pd.DataFrame,
) -> pd.DataFrame:
    high_keys = set(zip(high_error["scene"], high_error["point_name"]))
    rows: list[dict[str, Any]] = []
    for point in points.itertuples(index=False):
        key = (str(point.scene), str(point.point_name))
        if key not in high_keys:
            continue
        point_obs = observations[
            observations["annotation_set"].eq("v1.2.2_frozen")
            & observations["scene"].eq(key[0])
            & observations["point_name"].eq(key[1])
        ]
        outlier_views = point_obs[point_obs["gps_residual_percentile"].ge(99.5)]
        shared_image_level_evidence = False
        if key[0] == SCENE_50K and G39_IMAGE in set(point_obs["image_name"]):
            feature = feature_qc[feature_qc["image_name"].eq(G39_IMAGE)]
            shared_image_level_evidence = bool(
                len(feature)
                and float(feature.iloc[0]["reprojection_p95_px_percentile"]) >= 99.5
                and int(feature.iloc[0]["registered_track_count"]) < 500
            )
        if shared_image_level_evidence:
            diagnosis = "image_level_pose_quality_contributor_confirmed"
            rationale = "uses_0002_shared_outlier_with_extreme_gps_offset_low_track_support_and_worst_reprojection_tail"
        elif len(outlier_views):
            diagnosis = "gps_pose_contribution_possible_not_proven"
            rationale = "at_least_one_frozen_observation_uses_scene_top_0_5pct_gps_colmap_residual_image"
        else:
            diagnosis = "gps_not_supported_as_primary_explanation"
            rationale = "no_frozen_observation_uses_scene_top_0_5pct_gps_colmap_residual_image"
        rows.append(
            {
                "scene": key[0],
                "point_name": key[1],
                "role": point.role,
                "old_v1_2_2_error_3d_m": float(point.error_3d_m),
                "frozen_observation_count": len(point_obs),
                "max_gps_to_colmap_3d_m": float(point.gps_residual_max_m),
                "max_gps_residual_percentile": float(point.max_gps_residual_percentile),
                "top_0_5pct_gps_view_count": len(outlier_views),
                "top_0_5pct_gps_images": ";".join(sorted(outlier_views["image_name"].astype(str))),
                "diagnosis": diagnosis,
                "rationale": rationale,
            }
        )
    return pd.DataFrame(rows).sort_values(["scene", "old_v1_2_2_error_3d_m"], ascending=[True, False])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--release_root",
        type=Path,
        default=Path(r"E:\datasets\M3M-GCP\scenes\gcp_manual_annotations_v1_2_2"),
    )
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
        "--residual_root",
        type=Path,
        default=Path(r"E:\M3M-GCP-3DGS\outputs\six_scene_residual_outlier_diagnostics_20260702_101500"),
    )
    parser.add_argument(
        "--feature_qc",
        type=Path,
        default=Path(r"E:\M3M-GCP-3DGS\outputs\g39_pose_qc_20260716\all_image_feature_reprojection_qc.csv"),
    )
    parser.add_argument("--stamp", default=time.strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    output_root = args.repo / "outputs" / f"gcp_annotation_gps_pose_association_{args.stamp}"
    output_root.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.remote_manifest.read_text(encoding="utf-8"))
    residuals = pd.read_csv(args.residual_root / "per_point_residual_diagnostics.csv")
    high_error = pd.read_csv(args.residual_root / "high_error_point_diagnostics.csv")
    feature_qc = pd.read_csv(args.feature_qc) if args.feature_qc.exists() else pd.DataFrame()

    image_frames: list[pd.DataFrame] = []
    observation_frames: list[pd.DataFrame] = []
    input_records: list[dict[str, Any]] = []
    for scene in SCENES:
        scene_record = manifest["scenes"][scene]
        cameras, images = camera_indices(scene_record)
        metadata = image_metadata(args.candidate_root, scene)
        alignment_path = (
            args.remote_manifest.parent
            / "models"
            / scene
            / "raw_model"
            / "georegistration_alignment_summary.json"
        )
        alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
        image_qc = scene_image_gps_qc(scene, scene_record, metadata, alignment)
        image_frames.append(image_qc)

        release_path = args.release_root / RELEASE_FILE_PATTERN.format(scene=scene)
        frozen = release_rows(release_path)
        frozen_loo = loo_for_frame(scene, frozen, cameras, images)
        observation_frames.append(
            add_observation_diagnostics("v1.2.2_frozen", scene, frozen, image_qc, frozen_loo)
        )

        working_path = args.repo / ANNOTATION_RELATIVE_PATHS[scene]
        working = working_good_rows(pd.read_csv(working_path, dtype=str, keep_default_na=False))
        working_loo = loo_for_frame(scene, working, cameras, images)
        observation_frames.append(
            add_observation_diagnostics("v1.3_working_snapshot", scene, working, image_qc, working_loo)
        )
        for kind, path in {
            "release_annotation": release_path,
            "working_annotation_snapshot": working_path,
            "alignment_summary": alignment_path,
            "image_metadata": args.candidate_root / scene / "image_metadata.csv",
        }.items():
            input_records.append(
                {"scene": scene, "kind": kind, "path": str(path), "sha256": sha256_file(path)}
            )

    image_qc_all = pd.concat(image_frames, ignore_index=True)
    observations = pd.concat(observation_frames, ignore_index=True)
    if not feature_qc.empty:
        observations = observations.merge(
            feature_qc,
            on="image_name",
            how="left",
            suffixes=("", "_feature"),
            validate="many_to_one",
        )
    points = aggregate_points(observations, residuals)
    high_error_diagnosis = classify_high_error_points(points, high_error, observations, feature_qc)

    correlations = []
    for scene, group in [("all_scenes", points), *list(points.groupby("scene", sort=True))]:
        for metric in [
            "gps_residual_median_m",
            "gps_residual_p95_m",
            "gps_residual_max_m",
            "max_gps_residual_percentile",
        ]:
            correlations.append(
                {
                    "scene": scene,
                    "left": "error_3d_m",
                    "right": metric,
                    "point_count": int(group[["error_3d_m", metric]].dropna().shape[0]),
                    "spearman_rho": rank_correlation(group, "error_3d_m", metric),
                }
            )
    correlation_df = pd.DataFrame(correlations)

    affected_working = observations[
        observations["annotation_set"].eq("v1.3_working_snapshot")
        & observations["scene"].eq(SCENE_50K)
        & observations["image_name"].eq(G39_IMAGE)
    ]
    feature = feature_qc[feature_qc["image_name"].eq(G39_IMAGE)]
    image_row = image_qc_all[
        image_qc_all["scene"].eq(SCENE_50K) & image_qc_all["image_name"].eq(G39_IMAGE)
    ].iloc[0]
    exclusion = pd.DataFrame(
        [
            {
                "scene": SCENE_50K,
                "image_name": G39_IMAGE,
                "affected_working_points": ";".join(sorted(affected_working["point_name"].unique())),
                "raw_working_rows_preserved": True,
                "v1_2_2_release_unchanged": True,
                "future_v1_3_formal_candidate_action": "exclude_all_observations_from_this_image",
                "decision_basis": "independent_image_level_pose_quality_qc_not_gcp_residual",
                "gps_to_colmap_3d_m": float(image_row["gps_to_colmap_3d_m"]),
                "gps_residual_rank_desc": int(image_row["gps_residual_rank_desc"]),
                "gps_residual_percentile": float(image_row["gps_residual_percentile"]),
                "registered_track_count": int(feature.iloc[0]["registered_track_count"]),
                "feature_reprojection_p90_px": float(feature.iloc[0]["reprojection_p90_px"]),
                "feature_reprojection_p90_percentile": float(feature.iloc[0]["reprojection_p90_px_percentile"]),
                "feature_reprojection_p95_px": float(feature.iloc[0]["reprojection_p95_px"]),
                "feature_reprojection_p95_percentile": float(feature.iloc[0]["reprojection_p95_px_percentile"]),
            }
        ]
    )

    image_qc_all.to_csv(output_root / "per_image_gps_colmap_residual.csv", index=False, encoding="utf-8-sig")
    observations.to_csv(output_root / "per_observation_gps_pose_qc.csv", index=False, encoding="utf-8-sig")
    points.to_csv(output_root / "per_point_gps_pose_association.csv", index=False, encoding="utf-8-sig")
    high_error_diagnosis.to_csv(output_root / "high_error_point_gps_pose_diagnosis.csv", index=False, encoding="utf-8-sig")
    correlation_df.to_csv(output_root / "gps_error_rank_correlations.csv", index=False, encoding="utf-8-sig")
    exclusion.to_csv(output_root / "image_level_qc_exclusion_candidate.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(input_records).to_csv(output_root / "input_file_hashes.csv", index=False, encoding="utf-8-sig")

    overall_rho = correlation_df[
        correlation_df["scene"].eq("all_scenes") & correlation_df["right"].eq("gps_residual_max_m")
    ].iloc[0]["spearman_rho"]
    normalized_rho = correlation_df[
        correlation_df["scene"].eq("all_scenes")
        & correlation_df["right"].eq("max_gps_residual_percentile")
    ].iloc[0]["spearman_rho"]
    summary = {
        "schema": "ms_gcp_annotation_gps_pose_association_audit_v1",
        "diagnostic_only": True,
        "used_for_candidate_or_split_selection": False,
        "frozen_release_modified": False,
        "image_count": int(len(image_qc_all)),
        "frozen_observation_count": int(observations["annotation_set"].eq("v1.2.2_frozen").sum()),
        "working_good_observation_snapshot_count": int(
            observations["annotation_set"].eq("v1.3_working_snapshot").sum()
        ),
        "point_count_with_old_v1_2_2_residual": int(points["error_3d_m"].notna().sum()),
        "high_error_point_count": int(len(high_error_diagnosis)),
        "all_scene_error_vs_max_gps_spearman_rho": None if pd.isna(overall_rho) else float(overall_rho),
        "all_scene_error_vs_scene_normalized_max_gps_percentile_spearman_rho": (
            None if pd.isna(normalized_rho) else float(normalized_rho)
        ),
        "g39_0002": exclusion.iloc[0].to_dict(),
    }
    write_json(output_root / "summary.json", summary)
    readme = f"""# 标注、GPS 与相机位姿关联审计

- 本审计仅用于原因诊断，不参与候选选图、点位删除或 control/checkpoint 划分。
- 冻结 v1.2.2 release 未修改。
- 旧三维误差来自已有 v1.2.2/旧模型结果；新补标尚未重新评测。
- `G39/0002` 的建议是保留原始工作标注行，但在未来 v1.3 formal candidate 中按整张影像排除。
- 该排除由独立影像级证据触发，而不是由 G39 或 G33 的 GCP 残差触发。
- 跨场景直接使用米制偏差会混入场景差异，其 Spearman rho: {summary['all_scene_error_vs_max_gps_spearman_rho']}
- 使用每场景内部偏差百分位后，Spearman rho: {summary['all_scene_error_vs_scene_normalized_max_gps_percentile_spearman_rho']}
"""
    (output_root / "README_zh.md").write_text(readme, encoding="utf-8")

    files = sorted(path for path in output_root.iterdir() if path.is_file())
    with (output_root / "SHA256SUMS.txt").open("w", encoding="ascii", newline="\n") as handle:
        for path in files:
            handle.write(f"{sha256_file(path)}  {path.name}\n")
    print(output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
