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
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_SIM3 = {
    "gcp_3000_20260602": "2f4894f5753a3c7e2cf4cc6cfd48ce85c67dac1a5329bbc7972088642501c9c5",
    "gcp_5000_20260602": "4ace92000e745a38dfd85ca5dfddbd9f5e63fb2158e78b138f27ee1422b222d5",
    "gcp_20000_20260602": "7985d5e5855b93e1ba2d4fed7bfb1e1c7b8600a12af44fa4781e57427e6ad425",
    "gcp_10000_20260610": "2c2f69990642c43b8253c57eb4f769c0a959473737b23e10e22d1d0eb1879f01",
    "gcp_50000_20260610": "cf76e44faddc922b62235a952e139bde9d9c7cbfb21d440ce35a2093b7ac40bd",
    "gcp_100000_20260610": "f2bdfe649891f666371db64d9b504aee49bb1312fde33801408d72bea6def000",
}
EXPECTED_METHOD_CLASSES = {
    "3dgs_original": "rgb_colmap_only",
    "2dgs": "rgb_colmap_only",
    "pgsr": "rgb_colmap_only",
    "rade_gs": "rgb_colmap_only",
    "qgs": "rgb_colmap_only",
    "gsprior": "rgb_colmap_only",
    "sof": "rgb_colmap_only",
    "citygaussian_v2": "rgb_colmap_external_geometry_prior",
    "citygs_x": "rgb_colmap_external_geometry_prior",
    "metrogs": "rgb_colmap_external_geometry_prior",
}
EXPECTED_FORMAL_INPUTS = {
    "gcp_3000_20260602": ("ae29817198f54f04e4133a7b5fd03df679dd6f259b2d1ef4125e825cbb8e422e", "4ae07aad9278e2eb5af2f04268f3301df56c6f6ada9ee51c6f125fdbb29e7ec8"),
    "gcp_5000_20260602": ("7a57cf1009a360695f87fd8a2f4f881e2892dd8b9562aa0cc5119257bb5c1e38", "4a7a7576b990e3af3ba7918792366106e63935bc3b433e75d5477e51ce568b99"),
    "gcp_20000_20260602": ("8dc444f767f5a65bc8612eacd1527077a9dcd457f7fe04ebd8bf2f683be71fa7", "1d305e78974c299c7e63cc7774ea6e63ac3f428511d5b85b4dd252790cd2e64c"),
    "gcp_10000_20260610": ("3b8e25a46916c26a793d51d39e8c2a0cdcac9b36c9a7a2fa554792e92b886249", "e9b8a74348e8935ad022e35a48c5734d668862e020505dc751f7821b0fc29124"),
    "gcp_50000_20260610": ("35734cdcb94766fcad7bfcdf461e715b4266e98f16b426effd50126ffa9c27e1", "0679bc78d3c6c1e426692c5d19d5b10f2355f42d34d21221970c8e0f96b85c27"),
    "gcp_100000_20260610": ("c2cf9e951d95fee12a28d942e95c5c420df55bc364738b3f8737fed1c78bef3d", "5b4fe34743310bd2225feb2dd236200606be933002fec19d2c9ecb9f3ba6769d"),
}
EXPECTED_LAZ = {
    "lidars/terra_laz_1_4/cloud0.laz": (851011162, "a3828d6ca693219974f7c035983a023a5b0233a852b144119357e048b76aa516"),
    "lidars/terra_laz_1_4/cloud1.laz": (679843190, "717886fd32a6c6eeabd0fe86633f71bde25f3309811eac99e91c5cdee657179f"),
    "lidars/terra_laz_1_4/cloud2.laz": (337330794, "13e5d16b22e1172f94ab345f6786daa2b8649e1fa883fbc0d536b55e17c52a9e"),
    "lidars/terra_laz_1_4/cloud3.laz": (544251683, "31f6296ba830bfc4598c5c0ac95479b98a229ce919df1dc6640bd712b3541061"),
    "lidars/terra_laz_1_4/cloud4.laz": (189672787, "66a20a9ec17493cf22f48ec7ccffc6952f577e48aaeb6897b1678718f50f30dd"),
    "lidars/terra_laz_1_4/cloud5.laz": (291678006, "35a1775abd16d059d6b4c13e0976c4e4079250f60d3d7b02785a8f4e290b5966"),
    "lidars/terra_laz_1_4/cloud6.laz": (487722156, "9f6c9372c8202c6cb8b4b48acb6c7e4e9918f1cc47424df3110445d4b9e9f99f"),
    "lidars/terra_laz_1_4/cloud7.laz": (259018961, "73b3394c4ae62eea20e16c73beb07f2327ff4b6057e6ec64cf5a2c61b747e137"),
    "lidars/terra_laz_1_4/cloud8.laz": (224786263, "320f8610869a82618227bc47b50452e3c0c563b4412dec3caecadc2886abdbd6"),
}
EXPECTED_PACKET_KEYS = [
    "accumulated_alpha", "weighted_camera_z_sum", "weighted_camera_z_second_moment",
    "weighted_inverse_camera_z_sum", "alpha_normalized_expected_camera_z",
    "alpha_normalized_expected_inverse_camera_z", "harmonic_camera_z", "camera_z_variance",
    "metric_depth_valid_mask", "historical_invalid_unnormalized_inverse_depth",
]


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


def validate_contract(
    contract: dict[str, Any], split: dict[str, Any], repo_root: Path = REPO_ROOT
) -> list[str]:
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
    formal_inputs = contract.get("formal_input_binding", {})
    require(formal_inputs.get("source_model_files_exact") == ["cameras.bin", "images.bin", "points3D.bin", "points3D.ply"], "formal input source-model inventory mismatch")
    require(formal_inputs.get("source_model_file_locations") == {
        "cameras.bin": "colmap_model_root/cameras.bin",
        "images.bin": "colmap_model_root/images.bin",
        "points3D.bin": "colmap_model_root/points3D.bin",
        "points3D.ply": "formal_input_root/train/sparse/0/points3D.ply",
    }, "formal input source-model location mapping mismatch")
    actual_input_bindings = {
        scene: (row.get("file_sha256"), row.get("canonical_sha256"))
        for scene, row in formal_inputs.get("scene_manifests", {}).items()
    }
    require(actual_input_bindings == EXPECTED_FORMAL_INPUTS, "formal input manifest byte/canonical bindings mismatch")

    geometry = contract.get("source_geometry_binding", {})
    require(geometry.get("release_pin_path") == "configs/m3m_gcp_native_quarter_protocol_release_v2.json", "geometry release-pin path mismatch")
    require(geometry.get("release_pin_sha256") == "7bf9db0c62bb0bae9ecca06b97b08fcd5f26913e52f9dd2a29745b09e6ffe8e6", "geometry release-pin SHA mismatch")
    require(geometry.get("release_manifest_sha256") == "21fbac75d66433169535ea7440c31393f7a5ecdb4ed94fcefd31d1780c28bea4", "geometry release-manifest SHA mismatch")
    require(geometry.get("gcp_points_sha256") == "45b61e76b3548cb378436609c9c507fb8adacd2bbcc3d18034df179bc03305cc", "GCP coordinate SHA mismatch")
    require(geometry.get("gcp_roles_sha256") == "53400d901738677dfa99b7a001218c5871a8bd51d17fe2c327902a9a0bb3ae19", "GCP role SHA mismatch")
    require(geometry.get("scene_common_sim3_sha256") == EXPECTED_SIM3, "scene common-Sim3 identity mismatch")
    release_pin = repo_root / str(geometry.get("release_pin_path", "__missing__"))
    require(release_pin.is_file(), "geometry release-pin file missing")
    if release_pin.is_file():
        require(sha256_file(release_pin) == geometry.get("release_pin_sha256"), "geometry release-pin file SHA mismatch")

    method_binding = contract.get("method_registry_binding", {})
    require(method_binding.get("path") == "configs/m3m_gcp_native_quarter_method_registry_v3.json", "method-registry path mismatch")
    require(method_binding.get("file_sha256") == "b409b16435642eb02f865b41532b3856ba05a100e8fbef523d9d3401f89a5043", "method-registry SHA mismatch")
    require(method_binding.get("active_method_ids_in_order") == list(EXPECTED_METHOD_CLASSES), "active method order mismatch")
    require(method_binding.get("active_method_input_classes") == EXPECTED_METHOD_CLASSES, "method input-class mapping mismatch")
    registry_path = repo_root / str(method_binding.get("path", "__missing__"))
    require(registry_path.is_file(), "method-registry file missing")
    if registry_path.is_file():
        require(sha256_file(registry_path) == method_binding.get("file_sha256"), "method-registry file SHA mismatch")

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
    require(lidar.get("normal_minus_ellipsoid_m") == 23.980600991639484, "vertical datum bridge changed")
    actual_laz = {
        relative: (row.get("bytes"), row.get("sha256"))
        for relative, row in lidar.get("laz_files_exact", {}).items()
    }
    require(actual_laz == EXPECTED_LAZ, "exact nine-LAZ byte bindings mismatch")

    registration = contract.get("registration", {})
    for key in ("method_specific_sim3_refit", "method_specific_icp", "result_dependent_alignment"):
        require(registration.get(key) == "FORBIDDEN", f"registration gate weakened: {key}")

    reference = contract.get("reference_surface", {})
    require(reference.get("roi") == "convex_hull_of_frozen_active_control_and_checkpoint_points_buffered_in_EPSG32649", "ROI definition changed")
    require(reference.get("roi_buffer_m") == 8.0, "ROI buffer changed")
    require(reference.get("roi_is_identical_across_methods") is True, "ROI became method-dependent")
    require(reference.get("las_class_filter") == "NONE_SOURCE_CLASSIFICATION_IS_NOT_SEMANTICALLY_RELIABLE", "LiDAR class filter changed")
    require(reference.get("return_number_filter") == "NONE_ALL_LIDAR_RETURNS_DEFINE_THE_REFERENCE_SAMPLE_SET", "LiDAR return filter changed")
    require(reference.get("reference_voxel_m") == 0.05, "reference voxel changed")
    require(reference.get("voxel_representative") == "deterministic_voxel_center", "reference representative changed")
    require(reference.get("coordinate_dtype") == "float64", "reference dtype changed")
    require(reference.get("absolute_utm_float32") == "FORBIDDEN", "absolute UTM float32 permitted")

    surface = contract.get("reconstruction_surface", {})
    require(surface.get("primary_depth") == "alpha_normalized_expected_camera_z", "surface depth semantic changed")
    require(surface.get("view_role") == "train" and surface.get("all_train_views_required") is True, "view allowlist changed")
    require(surface.get("heldout_rgb_read") is False, "heldout RGB access enabled")
    require(surface.get("gcp_annotation_used_for_view_selection") is False, "GCP-dependent view selection enabled")
    require(surface.get("common_view_set_required_across_methods") is True, "common method view set disabled")
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
    require(ranking.get("scene_primary") == "fscore_10cm_descending", "scene primary ranking changed")
    require(ranking.get("scene_tiebreakers") == ["chamfer_l1_mean_m_ascending", "precision_10cm_descending"], "scene tiebreakers changed")
    require(ranking.get("overall_primary") == "macro_fscore_10cm_descending", "overall primary ranking changed")
    require(ranking.get("overall_tiebreakers") == ["macro_chamfer_l1_mean_m_ascending", "macro_precision_10cm_descending"], "overall tiebreakers changed")
    require(ranking.get("tie_numeric_tolerance") == 1e-9, "ranking tie tolerance changed")
    require(ranking.get("tie_tolerance_application") == "for_each_key_in_order_if_absolute_difference_is_at_most_tolerance_continue_to_next_key", "ranking tolerance application changed")
    require(ranking.get("all_keys_tied") == "same_competition_rank_then_method_id_lexicographic_display_next_rank_skips_tie_count", "all-keys-tied policy changed")
    require(ranking.get("dataset_aggregation") == "unweighted_arithmetic_macro_average_of_per_scene_metrics", "macro aggregation changed")
    require(ranking.get("micro_pooling_across_scenes") == "FORBIDDEN", "micro pooling permitted")
    require(ranking.get("overall_rank_eligibility") == "COMPLETE_ALL_6_SCENES_ONLY", "overall eligibility changed")
    require(ranking.get("failed_or_oom_scene_metric_imputation") == "FORBIDDEN", "failure metric imputation permitted")
    require(ranking.get("input_class_official_ranking") == "WITHIN_EACH_FROZEN_METHOD_REGISTRY_INPUT_CLASS", "input-class ranking changed")

    failure = contract.get("failure_policy", {})
    require(failure.get("quality_threshold_early_stop") is False, "quality-threshold early stop enabled")
    require(failure.get("record_and_continue") is True, "failure record-and-continue disabled")

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

    implementation = contract.get("implementation", {})
    expected_impl_paths = {
        "evaluator": "code/gcp/evaluate_m3m_gcp_lidar_formal_v1.py",
        "verifier": "code/gcp/verify_m3m_gcp_lidar_formal_v1.py",
        "artifact_schema": "configs/m3m_gcp_lidar_formal_artifact_schema_v1.json",
        "ranker": "code/gcp/rank_m3m_gcp_lidar_formal_v1.py",
        "launch_gate": "code/gcp/check_m3m_gcp_lidar_formal_launch.py",
    }
    for key, expected_path in expected_impl_paths.items():
        require(implementation.get(f"{key}_path") == expected_path, f"{key} path mismatch")
        path = repo_root / expected_path
        require(path.is_file(), f"{key} file missing")
        if path.is_file():
            require(implementation.get(f"{key}_sha256") == sha256_file(path), f"{key} SHA mismatch")
    require(implementation.get("formal_methods_manifest_schema") == "m3m_gcp_lidar_formal_methods_v1", "formal methods schema changed")
    require(implementation.get("implementation_version_commit") == "BOUND_BY_REVIEWED_ACTIVATION_MANIFEST_BEFORE_EXECUTION", "implementation commit binding weakened")

    launch = contract.get("launch_policy", {})
    require(launch.get("activation_manifest_schema") == "m3m_gcp_lidar_formal_activation_v1", "activation schema changed")
    require(launch.get("required_review_verdict") == "PASS_LIDAR_V1_AND_SIX_SCENE_PREPARATION", "activation review verdict changed")
    require(launch.get("active_frozen_required") is True, "ACTIVE_FROZEN launch gate disabled")
    require(launch.get("exact_clean_benchmark_commit_and_tree_required") is True, "exact clean commit gate disabled")
    require(launch.get("implementation_file_hashes_rechecked_before_output_creation") is True, "implementation pre-output hash gate disabled")
    require(launch.get("formal_output_root_must_not_exist") is True, "formal no-overwrite gate disabled")
    require(launch.get("resume_or_overwrite_formal_result") == "FORBIDDEN", "formal overwrite/resume enabled")
    require(launch.get("quality_threshold_early_stop") is False, "launch quality-threshold early stop enabled")
    require(launch.get("scene_execution_authorization_schema") == "m3m_gcp_lidar_scene_execution_authorization_v1", "scene execution authorization schema changed")
    for key in (
        "scene_plan_review_required", "full_ten_method_manifest_required_before_result",
        "scene_authorization_must_bind_methods_manifest_file_and_canonical_sha256",
        "selected_method_packet_bytes_must_be_verified_before_output",
        "lidar_laz_bytes_must_be_verified_before_output",
    ):
        require(launch.get(key) is True, f"launch evidence gate disabled: {key}")

    schema_path = repo_root / str(implementation.get("artifact_schema_path", "__missing__"))
    if schema_path.is_file():
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        require(schema.get("schema") == "m3m_gcp_lidar_formal_artifact_schema_v1", "artifact schema id mismatch")
        require(schema.get("protocol_id") == "m3m_gcp_lidar_rendered_surface_v1", "artifact schema protocol mismatch")
        require(schema.get("distance_npz", {}).get("keys_exact") == ["reconstruction_to_lidar_m", "lidar_to_reconstruction_m"], "distance NPZ keys changed")
        require(schema.get("distance_npz", {}).get("reconstruction_to_lidar_m", {}).get("dtype") == "float64", "distance dtype changed")
        require(schema.get("formal_methods_manifest", {}).get("schema") == "m3m_gcp_lidar_formal_methods_v1", "formal methods artifact schema changed")
        require(schema.get("formal_methods_manifest", {}).get("method_ids") == list(EXPECTED_METHOD_CLASSES), "formal methods exact pool changed")
        require(schema.get("depth_packet_manifest", {}).get("packet_npz", {}).get("keys_exact") == EXPECTED_PACKET_KEYS, "packet NPZ exact keys changed")
        require(schema.get("scene_execution_authorization", {}).get("schema") == "m3m_gcp_lidar_scene_execution_authorization_v1", "scene authorization artifact schema changed")
        require(schema.get("method_verification_report_json", {}).get("required_status") == "PASS_VERIFIED_FORMAL_V1", "independent verification PASS requirement changed")
        require(schema.get("six_scene_results_manifest", {}).get("schema") == "m3m_gcp_lidar_six_scene_results_manifest_v1", "six-scene results manifest schema changed")
        require(schema.get("six_scene_results_manifest", {}).get("scene_entry_fields_exact") == ["scene", "status", "method_result_path", "method_result_sha256", "verification_report_path", "verification_report_sha256"], "ranking verifier-evidence binding changed")
        require(schema.get("method_result_json", {}).get("metric_fields_exact") == metrics.get("machine_readable_diagnostics"), "method-result metric schema changed")
        comparator = schema.get("ranking_comparator", {})
        require(comparator.get("numeric_tolerance") == 1e-9, "artifact ranking tolerance changed")
        require(comparator.get("all_keys_tied_rule") == "assign the same competition rank; order equal-rank display rows lexicographically by method_id; the next rank skips the number of tied rows", "artifact all-tied policy changed")
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
