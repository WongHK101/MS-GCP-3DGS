#!/usr/bin/env python3
"""Validate the completed uniform v1.3 supplement and audit UI hint bias."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from audit_v13_candidate_recall_and_gps import (
    ANNOTATION_RELATIVE_PATHS,
    SCENES,
    azimuth_bin,
    azimuth_deg,
    camera_indices,
    image_metadata,
    is_nadir,
    leave_one_out_rows,
    sha256_file,
    write_json,
    write_launcher,
)
from prepare_direct_multiview_annotation_tasks import CANDIDATE_FIELDS, write_candidate_csv
from prepare_followup_annotation_tasks_50k100k import triangulate_annotation_rays, visible_good
from prepare_v13_uniform_fixed_candidate_supplement import (
    EXCLUDED_DIAGNOSTIC_POINTS,
    TARGET_AZIMUTH_BINS,
    TARGET_NADIR_GOOD,
    TARGET_OBLIQUE_GOOD,
    TARGET_TOTAL_GOOD,
)


IMAGE_LEVEL_FORMAL_EXCLUSIONS = {
    ("gcp_50000_20260610", "DJI_20260610161948_0002_D.JPG"):
        "independent_image_pose_quality_outlier",
}


def image_sequence(image_name: str) -> int | None:
    match = re.search(r"_(\d{4})_D\.JPG$", image_name, re.IGNORECASE)
    return int(match.group(1)) if match else None


def residual_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in rows:
        try:
            mx = float(row.get("manual_x") or "nan")
            my = float(row.get("manual_y") or "nan")
            px = float(row.get("projected_x") or "nan")
            py = float(row.get("projected_y") or "nan")
        except (TypeError, ValueError):
            continue
        if str(row.get("visible", "")) != "1" or not all(math.isfinite(v) for v in [mx, my, px, py]):
            continue
        dx, dy = mx - px, my - py
        records.append(
            {
                "dx": dx,
                "dy": dy,
                "projected_x": px,
                "projected_y": py,
                "image_name": str(row.get("image_name", "")),
                "point_name": str(row.get("point_name", "")),
                "seq": image_sequence(str(row.get("image_name", ""))),
                "norm": math.hypot(dx, dy),
            }
        )
    return records


def robust_filter_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(records) < 6:
        return records
    norms = [float(row["norm"]) for row in records]
    median = statistics.median(norms)
    mad = statistics.median(abs(value - median) for value in norms)
    if mad < 1e-6:
        return records
    threshold = median + 3.5 * 1.4826 * mad
    kept = [row for row in records if float(row["norm"]) <= threshold]
    return kept if len(kept) >= 4 else records


def legacy_history_correction(
    candidate: dict[str, Any], history_rows: list[dict[str, Any]]
) -> tuple[float, float, str] | None:
    records = residual_records(history_rows)
    same_image = [row for row in records if row["image_name"] == candidate["image_name"]]
    if same_image:
        return (
            float(statistics.median(row["dx"] for row in same_image)),
            float(statistics.median(row["dy"] for row in same_image)),
            "same_image",
        )
    filtered = robust_filter_records(records)
    if len(filtered) >= 4:
        px, py = float(candidate["pixel_x"]), float(candidate["pixel_y"])
        sequence = image_sequence(str(candidate["image_name"]))
        scored: list[tuple[float, dict[str, Any]]] = []
        for row in filtered:
            spatial = math.hypot(float(row["projected_x"]) - px, float(row["projected_y"]) - py)
            row_sequence = row.get("seq")
            sequence_distance = (
                abs(int(row_sequence) - sequence)
                if sequence is not None and row_sequence is not None
                else 0
            )
            scored.append((spatial / 900.0 + sequence_distance / 30.0, row))
        nearest = sorted(scored, key=lambda item: item[0])[: min(30, len(scored))]
        weights = [1.0 / ((1.0 + score) ** 2) for score, _ in nearest]
        total = sum(weights)
        return (
            float(sum(weight * row["dx"] for weight, (_, row) in zip(weights, nearest)) / total),
            float(sum(weight * row["dy"] for weight, (_, row) in zip(weights, nearest)) / total),
            "weighted_scene_history",
        )
    same_point = [row for row in records if row["point_name"] == candidate["point_name"]]
    if same_point:
        return (
            float(statistics.median(row["dx"] for row in same_point)),
            float(statistics.median(row["dy"] for row in same_point)),
            "same_point",
        )
    if records:
        return (
            float(statistics.median(row["dx"] for row in records)),
            float(statistics.median(row["dy"] for row in records)),
            "same_scene",
        )
    return None


def finite_percentile(values: pd.Series, percentile: float) -> float:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    return float(np.percentile(finite, percentile)) if len(finite) else float("nan")


def task_validation_rows(
    tasks: pd.DataFrame,
    annotations: pd.DataFrame,
    initial_history: pd.DataFrame,
) -> pd.DataFrame:
    merged = tasks.merge(
        annotations,
        on=["scene", "point_name", "image_name"],
        how="left",
        suffixes=("_task", "_ann"),
        validate="one_to_one",
        indicator=True,
    )
    history_rows = initial_history.to_dict("records")
    output: list[dict[str, Any]] = []
    for row in merged.to_dict("records"):
        found = row["_merge"] == "both"
        quality = str(row.get("quality", ""))
        visible = str(row.get("visible", ""))
        manual_x = pd.to_numeric(pd.Series([row.get("manual_x", "")]), errors="coerce").iloc[0]
        manual_y = pd.to_numeric(pd.Series([row.get("manual_y", "")]), errors="coerce").iloc[0]
        width, height = int(float(row["image_width"])), int(float(row["image_height"]))
        manual_finite = bool(pd.notna(manual_x) and pd.notna(manual_y))
        manual_in_bounds = bool(manual_finite and 0 <= manual_x < width and 0 <= manual_y < height)
        candidate = {
            "scene": row["scene"],
            "point_name": row["point_name"],
            "image_name": row["image_name"],
            "pixel_x": row["pixel_x"],
            "pixel_y": row["pixel_y"],
        }
        correction = legacy_history_correction(candidate, history_rows)
        yellow_x, yellow_y = float(row["pixel_x"]), float(row["pixel_y"])
        yellow_error = (
            math.hypot(float(manual_x) - yellow_x, float(manual_y) - yellow_y)
            if manual_finite
            else float("nan")
        )
        if correction is None:
            hint_dx = hint_dy = float("nan")
            hint_source = "none"
            legacy_x, legacy_y = yellow_x, yellow_y
        else:
            hint_dx, hint_dy, hint_source = correction
            legacy_x, legacy_y = yellow_x + hint_dx, yellow_y + hint_dy
        legacy_error = (
            math.hypot(float(manual_x) - legacy_x, float(manual_y) - legacy_y)
            if manual_finite
            else float("nan")
        )
        output.append(
            {
                "scene": row["scene"],
                "point_name": row["point_name"],
                "image_name": row["image_name"],
                "task_row_found": found,
                "visible": visible,
                "quality": quality,
                "manual_x": manual_x,
                "manual_y": manual_y,
                "manual_coordinate_finite": manual_finite,
                "manual_coordinate_in_bounds": manual_in_bounds,
                "yellow_candidate_x": yellow_x,
                "yellow_candidate_y": yellow_y,
                "manual_minus_yellow_dx": float(manual_x) - yellow_x if manual_finite else np.nan,
                "manual_minus_yellow_dy": float(manual_y) - yellow_y if manual_finite else np.nan,
                "yellow_to_manual_error_px": yellow_error,
                "legacy_hint_correction_dx": hint_dx,
                "legacy_hint_correction_dy": hint_dy,
                "legacy_hint_source": hint_source,
                "legacy_hint_x": legacy_x,
                "legacy_hint_y": legacy_y,
                "legacy_hint_to_manual_error_px": legacy_error,
                "legacy_hint_worse_than_yellow": bool(
                    manual_finite and legacy_error > yellow_error + 1e-12
                ),
                "legacy_hint_upper_right_of_manual": bool(
                    manual_finite and legacy_x > float(manual_x) and legacy_y < float(manual_y)
                ),
            }
        )
    return pd.DataFrame(output)


def point_coverage(
    scene: str,
    annotations: pd.DataFrame,
    cameras: dict[int, dict[str, Any]],
    images: dict[str, dict[str, Any]],
    metadata: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for point_name, group in annotations.groupby("point_name", sort=True):
        all_good = [row for row in group.to_dict("records") if visible_good(row) and row["image_name"] in images]
        formal_good = [
            row
            for row in all_good
            if (scene, str(row["image_name"])) not in IMAGE_LEVEL_FORMAL_EXCLUSIONS
        ]
        types = ["nadir" if is_nadir(metadata.get(str(row["image_name"]))) else "oblique" for row in formal_good]
        bins: set[int] = set()
        triangulation_status = "not_testable"
        if len(formal_good) >= 2:
            try:
                xyz, condition = triangulate_annotation_rays(formal_good, cameras, images)
                bins = {
                    azimuth_bin(azimuth_deg(xyz, images[str(row["image_name"])]))
                    for row in formal_good
                }
                triangulation_status = "pass" if math.isfinite(condition) else "nonfinite_condition"
            except Exception as exc:
                triangulation_status = f"fail:{type(exc).__name__}"
        good_count = len(formal_good)
        nadir_count = types.count("nadir")
        oblique_count = types.count("oblique")
        output.append(
            {
                "scene": scene,
                "point_name": point_name,
                "all_working_good_count": len(all_good),
                "image_qc_excluded_good_count": len(all_good) - len(formal_good),
                "future_formal_good_count": good_count,
                "future_formal_nadir_good_count": nadir_count,
                "future_formal_oblique_good_count": oblique_count,
                "future_formal_azimuth_bin_count": len(bins),
                "future_formal_azimuth_bins": ";".join(str(value) for value in sorted(bins)),
                "triangulation_status": triangulation_status,
                "meets_total_target": good_count >= TARGET_TOTAL_GOOD,
                "meets_nadir_target": nadir_count >= TARGET_NADIR_GOOD,
                "meets_oblique_target": oblique_count >= TARGET_OBLIQUE_GOOD,
                "meets_azimuth_target": len(bins) >= TARGET_AZIMUTH_BINS,
                "meets_all_targets": bool(
                    good_count >= TARGET_TOTAL_GOOD
                    and nadir_count >= TARGET_NADIR_GOOD
                    and oblique_count >= TARGET_OBLIQUE_GOOD
                    and len(bins) >= TARGET_AZIMUTH_BINS
                ),
                "diagnostic_only_predeclared": (scene, str(point_name)) in EXCLUDED_DIAGNOSTIC_POINTS,
            }
        )
    return output


def geometry_qc(
    scene: str,
    annotations: pd.DataFrame,
    cameras: dict[int, dict[str, Any]],
    images: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    for point_name, group in annotations.groupby("point_name", sort=True):
        rows = [
            row
            for row in group.to_dict("records")
            if visible_good(row)
            and row["image_name"] in images
            and (scene, str(row["image_name"])) not in IMAGE_LEVEL_FORMAL_EXCLUSIONS
        ]
        if len(rows) < 4:
            continue
        leave_one_out = leave_one_out_rows(scene, str(point_name), rows, cameras, images)
        finite_values = np.asarray(
            [float(row.get("pixel_error", np.nan)) for row in leave_one_out], dtype=np.float64
        )
        median = float(np.nanmedian(finite_values))
        for row, value in zip(leave_one_out, finite_values):
            robust_outlier = bool(
                math.isfinite(value)
                and value > 10.0
                and value > 3.0 * max(median, 1e-12)
            )
            output.append(
                {
                    **row,
                    "point_loo_median_px": median,
                    "error_to_point_median_ratio": value / median if median > 0 else np.nan,
                    "robust_geometry_recheck_required": robust_outlier,
                }
            )
    return pd.DataFrame(output)


def annotation_row_as_candidate(
    annotation: dict[str, Any],
    image: dict[str, Any],
    camera: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    return {
        "scene": annotation["scene"],
        "point_name": annotation["point_name"],
        "image_name": annotation["image_name"],
        "image_path": annotation["image_path"],
        "pixel_x": annotation["projected_x"],
        "pixel_y": annotation["projected_y"],
        "center_score": "",
        "inside_image": "True",
        "edge_margin_px": "",
        "projection_uncertainty_px": "240",
        "candidate_source": reason,
        "view_type": "",
        "camera_azimuth_deg": "",
        "azimuth_bin_45deg": "",
        "off_nadir_deg": "",
        "image_width": str(camera["width"]),
        "image_height": str(camera["height"]),
        "task_action": "post_supplement_geometry_recheck",
        "annotation_image_domain": "raw_dji_decoded_pixel_matrix_ignore_exif_orientation",
        "annotation_coordinate_domain": "raw_image_zero_based_pixel_centers",
        "already_attempted": "True",
        "already_good": str(visible_good(annotation)),
        "camera_id": str(image["camera_id"]),
        "image_id": str(image["image_id"]),
        "image_pose_record_sha256": image["record_sha256"],
        "camera_record_sha256": camera["record_sha256"],
        "camera_z_model_units": "",
        "adds_new_azimuth_bin": "False",
        "adds_missing_view_type": "False",
        "rank_for_gcp": "1",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(r"E:\M3M-GCP-3DGS"))
    parser.add_argument(
        "--task_root",
        type=Path,
        default=Path(
            r"E:\M3M-GCP-3DGS\outputs\gcp_v13_uniform_fixed_candidate_supplement_20260716_224500"
        ),
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
    parser.add_argument("--stamp", default=time.strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    output_root = args.repo / "outputs" / f"gcp_v13_uniform_supplement_validation_{args.stamp}"
    output_root.mkdir(parents=True, exist_ok=False)
    tasks = pd.read_csv(args.task_root / "all_selected_uniform_supplement_candidates.csv", dtype=str)
    manifest = json.loads((args.task_root / "task_manifest.json").read_text(encoding="utf-8"))
    camera_manifest = json.loads(args.remote_manifest.read_text(encoding="utf-8"))
    input_hashes: list[dict[str, Any]] = []
    task_qc_frames: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, Any]] = []
    geometry_frames: list[pd.DataFrame] = []
    annotations_by_scene: dict[str, pd.DataFrame] = {}
    camera_sets: dict[str, tuple[dict[int, dict[str, Any]], dict[str, dict[str, Any]]]] = {}

    for launcher in manifest["launchers"]:
        scene = str(launcher["scene"])
        annotation_path = Path(launcher["annotation_output"])
        annotation = pd.read_csv(annotation_path, dtype=str, keep_default_na=False)
        annotations_by_scene[scene] = annotation
        if annotation[["scene", "point_name", "image_name"]].duplicated().any():
            raise RuntimeError(f"{scene}: duplicate annotation keys")
        scene_tasks = tasks[tasks["scene"].eq(scene)].copy()
        task_keys = set(zip(scene_tasks["scene"], scene_tasks["point_name"], scene_tasks["image_name"]))
        initial_history = annotation[
            [
                (str(row.scene), str(row.point_name), str(row.image_name)) not in task_keys
                for row in annotation.itertuples(index=False)
            ]
        ]
        task_qc_frames.append(task_validation_rows(scene_tasks, annotation, initial_history))
        cameras, images = camera_indices(camera_manifest["scenes"][scene])
        camera_sets[scene] = (cameras, images)
        metadata = image_metadata(args.candidate_root, scene)
        coverage_rows.extend(point_coverage(scene, annotation, cameras, images, metadata))
        geometry_frames.append(geometry_qc(scene, annotation, cameras, images))
        input_hashes.append(
            {"scene": scene, "kind": "completed_working_annotation", "path": str(annotation_path), "sha256": sha256_file(annotation_path)}
        )

    task_qc = pd.concat(task_qc_frames, ignore_index=True)
    coverage = pd.DataFrame(coverage_rows)
    geometry = pd.concat(geometry_frames, ignore_index=True)
    if len(task_qc) != 450 or task_qc[["scene", "point_name", "image_name"]].duplicated().any():
        raise RuntimeError("Uniform task spine is not exactly 450 unique rows")

    status_summary = (
        task_qc.groupby(["scene", "quality"], dropna=False).size().unstack(fill_value=0).reset_index()
    )
    hint_rows = task_qc[task_qc["quality"].isin(["good", "ambiguous"]) & task_qc["manual_coordinate_finite"]]
    hint_summary: list[dict[str, Any]] = []
    for scene, group in hint_rows.groupby("scene", sort=True):
        hint_summary.append(
            {
                "scene": scene,
                "manual_coordinate_count": len(group),
                "yellow_error_median_px": float(group["yellow_to_manual_error_px"].median()),
                "yellow_error_p95_px": finite_percentile(group["yellow_to_manual_error_px"], 95),
                "yellow_error_max_px": float(group["yellow_to_manual_error_px"].max()),
                "legacy_hint_error_median_px": float(group["legacy_hint_to_manual_error_px"].median()),
                "legacy_hint_error_p95_px": finite_percentile(group["legacy_hint_to_manual_error_px"], 95),
                "legacy_hint_error_max_px": float(group["legacy_hint_to_manual_error_px"].max()),
                "legacy_hint_correction_dx_median_px": float(group["legacy_hint_correction_dx"].median()),
                "legacy_hint_correction_dy_median_px": float(group["legacy_hint_correction_dy"].median()),
                "legacy_hint_worse_fraction": float(group["legacy_hint_worse_than_yellow"].mean()),
                "legacy_hint_upper_right_of_manual_fraction": float(
                    group["legacy_hint_upper_right_of_manual"].mean()
                ),
            }
        )
    hint_summary_df = pd.DataFrame(hint_summary)

    unselected = task_qc[task_qc["quality"].eq("")]
    geometry_rechecks = geometry[geometry["robust_geometry_recheck_required"]]
    recheck_manifest = []
    recheck_dir = output_root / "recheck"
    recheck_dir.mkdir()
    recheck_keys = pd.concat(
        [
            unselected[["scene", "point_name", "image_name"]],
            geometry_rechecks[["scene", "point_name", "hidden_image_name"]].rename(
                columns={"hidden_image_name": "image_name"}
            ),
        ],
        ignore_index=True,
    ).drop_duplicates()
    for scene, group in recheck_keys.groupby("scene", sort=True):
        selected_rows: list[dict[str, Any]] = []
        scene_tasks = tasks[tasks["scene"].eq(scene)]
        annotation = annotations_by_scene[scene]
        cameras, images = camera_sets[scene]
        for key in group.to_dict("records"):
            matched_task = scene_tasks[
                scene_tasks["point_name"].eq(key["point_name"])
                & scene_tasks["image_name"].eq(key["image_name"])
            ]
            if len(matched_task) == 1:
                selected_rows.append(matched_task.iloc[0].to_dict())
                continue
            matched_annotation = annotation[
                annotation["point_name"].eq(key["point_name"])
                & annotation["image_name"].eq(key["image_name"])
            ]
            if len(matched_annotation) != 1:
                raise RuntimeError(f"Cannot resolve recheck row: {key}")
            annotation_row = matched_annotation.iloc[0].to_dict()
            image = images[str(key["image_name"])]
            camera = cameras[int(image["camera_id"])]
            selected_rows.append(
                annotation_row_as_candidate(
                    annotation_row,
                    image,
                    camera,
                    "robust_leave_one_view_out_geometry_recheck",
                )
            )
        selected = pd.DataFrame(selected_rows)
        candidate_path = recheck_dir / f"{scene}_unselected_recheck.csv"
        write_candidate_csv(
            candidate_path,
            [{field: row.get(field, "") for field in CANDIDATE_FIELDS} for row in selected.to_dict("records")],
        )
        annotation_path = Path(
            next(row["annotation_output"] for row in manifest["launchers"] if row["scene"] == scene)
        )
        launcher_path = recheck_dir / f"launch_{scene}_unselected_recheck.ps1"
        write_launcher(
            launcher_path,
            Path(__file__).resolve().parents[2],
            candidate_path,
            annotation_path,
            Path(r"E:\datasets\M3M-GCP\scenes") / scene,
        )
        recheck_manifest.append(
            {
                "scene": scene,
                "candidate_csv": str(candidate_path),
                "launcher": str(launcher_path),
                "row_count": len(selected),
            }
        )

    task_qc.to_csv(output_root / "task_completion_and_hint_qc.csv", index=False, encoding="utf-8-sig")
    status_summary.to_csv(output_root / "task_status_summary.csv", index=False, encoding="utf-8-sig")
    hint_summary_df.to_csv(output_root / "hint_accuracy_summary.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(output_root / "post_supplement_point_coverage.csv", index=False, encoding="utf-8-sig")
    geometry.to_csv(output_root / "post_supplement_leave_one_out_qc.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(input_hashes).to_csv(output_root / "input_annotation_hashes.csv", index=False, encoding="utf-8-sig")
    write_json(output_root / "recheck_manifest.json", recheck_manifest)
    summary = {
        "schema": "ms_gcp_v13_uniform_supplement_completion_validation_v1",
        "task_count": len(task_qc),
        "task_key_missing_count": int((~task_qc["task_row_found"]).sum()),
        "unselected_status_count": int(task_qc["quality"].eq("").sum()),
        "robust_geometry_recheck_count": int(len(geometry_rechecks)),
        "good_count": int(task_qc["quality"].eq("good").sum()),
        "not_visible_count": int(task_qc["quality"].eq("not_visible").sum()),
        "ambiguous_count": int(task_qc["quality"].eq("ambiguous").sum()),
        "good_manual_nonfinite_or_oob_count": int(
            (
                task_qc["quality"].eq("good")
                & (~task_qc["manual_coordinate_finite"] | ~task_qc["manual_coordinate_in_bounds"])
            ).sum()
        ),
        "point_count": len(coverage),
        "point_count_meeting_all_targets": int(coverage["meets_all_targets"].sum()),
        "point_count_not_meeting_all_targets": int((~coverage["meets_all_targets"]).sum()),
        "formal_image_exclusions": [
            {"scene": scene, "image_name": image, "reason": reason}
            for (scene, image), reason in IMAGE_LEVEL_FORMAL_EXCLUSIONS.items()
        ],
        "legacy_history_hint_non_authoritative": True,
        "acceptance_status": (
            "pass"
            if not len(unselected) and not len(geometry_rechecks)
            else "blocked_pending_status_or_geometry_recheck"
        ),
    }
    write_json(output_root / "summary.json", summary)
    print(output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
