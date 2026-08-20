#!/usr/bin/env python3
"""Fail-closed static validator for the M3M-GCP LiDAR formal-v1 contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_SCENES = {
    "gcp_3000_20260602": (94, 82, 12, 71),
    "gcp_5000_20260602": (101, 88, 13, 71),
    "gcp_20000_20260602": (298, 260, 38, 71),
    "gcp_10000_20260610": (976, 854, 122, 63),
    "gcp_50000_20260610": (2208, 1932, 276, 63),
    "gcp_100000_20260610": (2510, 2196, 314, 63),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: dict[str, Any]) -> str:
    clean = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    raw = json.dumps(clean, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def validate_contract(contract: dict[str, Any], split: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(contract.get("schema") == "m3m_gcp_lidar_rendered_surface_contract_v1", "schema mismatch")
    require(contract.get("protocol_id") == "m3m_gcp_lidar_rendered_surface_v1", "protocol id mismatch")
    require(contract.get("source_geometry_protocol_id") == "m3m_gcp_native_quarter_geometry_v2", "source protocol mismatch")
    status = contract.get("status")
    require(status in {"REVIEW_CANDIDATE_NOT_EXECUTION_AUTHORIZED", "ACTIVE_FROZEN"}, "invalid status")
    require(bool(contract.get("execution_authorized")) == (status == "ACTIVE_FROZEN"), "status/authorization mismatch")

    source = contract.get("source_data_release", {})
    require(source.get("release_root_digest_sha256") == "513f8999fe4b110f15bcbecad7932895781cee755ee9ccd7a14ff10298546d75", "release digest mismatch")
    require(split.get("manifest_sha256") == source.get("split_manifest_canonical_sha256"), "split canonical identity mismatch")
    require(split.get("release_root_digest") == source.get("release_root_digest_sha256"), "split release mismatch")

    rows = {row.get("scene"): row for row in contract.get("scenes", [])}
    require(set(rows) == set(EXPECTED_SCENES), "scene set mismatch")
    split_rows = {row.get("scene"): row for row in split.get("scenes", [])}
    for scene, expected in EXPECTED_SCENES.items():
        row = rows.get(scene, {})
        full, train, test, days = expected
        require((row.get("full_views"), row.get("train_views"), row.get("test_views"), row.get("temporal_gap_days")) == expected, f"{scene}: frozen counts/date gap mismatch")
        split_row = split_rows.get(scene, {})
        require((split_row.get("full_view_count"), split_row.get("train_view_count"), split_row.get("test_view_count")) == (full, train, test), f"{scene}: split counts mismatch")

    lidar = contract.get("lidar_source", {})
    require(lidar.get("payload_file_count") == 52 and lidar.get("payload_bytes") == 4008188380, "LiDAR payload inventory mismatch")
    require(lidar.get("training_access") == "FORBIDDEN_EVALUATION_ONLY", "LiDAR training boundary weakened")
    require(lidar.get("method_specific_reference_selection") is False, "method-specific reference selection enabled")

    registration = contract.get("registration", {})
    for key in ("method_specific_sim3_refit", "method_specific_icp", "result_dependent_alignment"):
        require(registration.get(key) == "FORBIDDEN", f"registration gate weakened: {key}")

    reference = contract.get("reference_surface", {})
    require(reference.get("roi_buffer_m") == 8.0, "ROI buffer changed")
    require(reference.get("reference_voxel_m") == 0.05, "reference voxel changed")
    require(reference.get("voxel_representative") == "deterministic_voxel_center", "reference representative changed")
    require(reference.get("coordinate_dtype") == "float64", "reference dtype changed")
    require(reference.get("absolute_utm_float32") == "FORBIDDEN", "absolute UTM float32 permitted")

    surface = contract.get("reconstruction_surface", {})
    require(surface.get("primary_depth") == "alpha_normalized_expected_camera_z", "surface depth semantic changed")
    require(surface.get("view_role") == "train" and surface.get("all_train_views_required") is True, "view allowlist changed")
    require(surface.get("heldout_rgb_read") is False, "heldout RGB access enabled")
    require(surface.get("gcp_annotation_used_for_view_selection") is False, "GCP-dependent view selection enabled")
    require(surface.get("alpha_min_inclusive") == 0.5, "alpha threshold changed")
    require(surface.get("pixel_stride") == 4, "pixel stride changed")
    require(surface.get("reconstruction_voxel_m") == 0.05 and surface.get("voxel_grid_shared_with_reference") is True, "surface voxel contract changed")
    require(surface.get("coordinate_dtype") == "float64", "surface dtype changed")
    require(surface.get("threshold_comparison_epsilon_m") == 1e-9, "threshold epsilon changed")

    metrics = contract.get("metrics", {})
    require(metrics.get("thresholds_m") == [0.05, 0.1, 0.2], "metric thresholds changed")
    require(metrics.get("main_table") == ["fscore_10cm", "precision_10cm", "recall_10cm", "chamfer_l1_mean_m"], "main metric table changed")
    require(len(metrics.get("machine_readable_diagnostics", [])) == 17, "diagnostic metric inventory changed")

    ranking = contract.get("ranking", {})
    require(ranking.get("dataset_aggregation") == "unweighted_arithmetic_macro_average_of_per_scene_metrics", "macro aggregation changed")
    require(ranking.get("micro_pooling_across_scenes") == "FORBIDDEN", "micro pooling permitted")
    require(ranking.get("overall_rank_eligibility") == "COMPLETE_ALL_6_SCENES_ONLY", "overall eligibility changed")
    require(ranking.get("failed_or_oom_scene_metric_imputation") == "FORBIDDEN", "failure metric imputation permitted")
    require(ranking.get("input_class_official_ranking") == "WITHIN_EACH_FROZEN_METHOD_REGISTRY_INPUT_CLASS", "input-class ranking changed")

    evidence = contract.get("evidence", {})
    for key in (
        "required_common_view_allowlist_hash",
        "required_packet_manifest_and_file_hashes",
        "required_evaluator_and_verifier_hashes",
        "required_float64_bidirectional_distance_array_hashes",
        "required_independent_metric_recomputation",
        "distance_arrays_must_be_retained",
    ):
        require(evidence.get(key) is True, f"evidence requirement disabled: {key}")
    require(contract.get("pilot_transition", {}).get("pilot_results_reusable_as_formal") is False, "pilot result promoted to formal")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    split = json.loads(args.split.read_text(encoding="utf-8"))
    errors = validate_contract(contract, split)
    if sha256_file(args.split) != contract.get("source_data_release", {}).get("split_manifest_file_sha256"):
        errors.append("split file SHA mismatch")
    if canonical_sha256(split) != split.get("manifest_sha256"):
        errors.append("split self-canonical SHA mismatch")
    report = {
        "schema": "m3m_gcp_lidar_formal_protocol_validation_v1",
        "status": "PASS" if not errors else "FAIL",
        "protocol_id": contract.get("protocol_id"),
        "contract_file_sha256": sha256_file(args.contract),
        "split_file_sha256": sha256_file(args.split),
        "execution_authorized": contract.get("execution_authorized"),
        "errors": errors,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
