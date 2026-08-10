#!/usr/bin/env python3
"""Validate the native-quarter nine-method candidate registry."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


SHA1 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_METHODS = {
    "3dgs_original",
    "2dgs",
    "pgsr",
    "rade_gs",
    "gof",
    "qgs",
    "citygaussian_v2",
    "citygs_x",
    "metrogs",
}
EXTERNAL_PRIOR_METHODS = {"citygs_x", "metrogs"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_registry(value: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(value.get("schema") == "m3m_gcp_native_quarter_method_registry_v2", "unknown schema")
    require(value.get("protocol_id") == "m3m_gcp_native_quarter_geometry_v2", "protocol mismatch")
    require(
        value.get("status")
        == "candidate_pool_frozen_3dgs_and_2dgs_3k_complete_ranked_gof_static_pass_gpu_pending_other_methods_locked",
        "registry status mismatch",
    )
    require(value.get("method_count") == 9, "method_count must be 9")
    require(
        bool(SHA256.fullmatch(str(value.get("source_data_release", {}).get("release_root_digest_sha256", "")))),
        "source data release digest is invalid",
    )
    methods = value.get("methods", [])
    ids = [str(method.get("method_id", "")) for method in methods]
    require(len(ids) == len(set(ids)), "duplicate method ids")
    require(set(ids) == EXPECTED_METHODS, f"method set mismatch: {sorted(set(ids) ^ EXPECTED_METHODS)}")
    require(value.get("method_ids") == ids, "method_ids order differs from methods")
    coverage = value.get("coverage_and_ranking_contract", {})
    require(coverage.get("minimum_unique_oblique_azimuth_bins_45deg") == 2, "oblique azimuth-bin gate mismatch")
    require(coverage.get("minimum_oblique_azimuth_bin_circular_separation") == 2, "oblique azimuth separation gate mismatch")
    require(coverage.get("azimuth_bin_count") == 8, "azimuth bin count mismatch")
    require("does not guarantee actual angle >= 90" in str(coverage.get("continuous_angle_claim", "")), "continuous-angle disclaimer missing")
    require("COMPLETE_RANKED" in str(coverage.get("ranked_scene_status", "")), "complete ranking status missing")
    require("INCOMPLETE_UNRANKED" in str(coverage.get("incomplete_scene_status", "")), "incomplete ranking status missing")

    for method in methods:
        method_id = str(method.get("method_id", "<missing>"))
        prefix = method_id + ": "
        source = method.get("source", {})
        require(str(source.get("official_repository", "")).startswith("https://github.com/"), prefix + "official repository missing")
        require(bool(SHA1.fullmatch(str(source.get("commit", "")))), prefix + "commit is not full SHA-1")
        require(bool(SHA1.fullmatch(str(source.get("tree", "")))), prefix + "tree is not full SHA-1")
        require(str(method.get("publication", {}).get("record", "")).startswith("https://"), prefix + "publication record missing")
        input_class = method.get("input_class")
        priors = method.get("external_priors", [])
        if method_id in EXTERNAL_PRIOR_METHODS:
            require(input_class == "rgb_colmap_external_geometry_prior", prefix + "external-prior input class missing")
            require(isinstance(priors, list) and bool(priors), prefix + "external prior inventory missing")
            require(
                all(str(prior.get("freeze_status", "")).startswith("PENDING_") for prior in priors),
                prefix + "current external priors must remain explicitly pending",
            )
        else:
            require(input_class == "rgb_colmap_only", prefix + "unexpected input class")
            require(priors == [], prefix + "RGB+COLMAP method must not carry undeclared priors")
        if method_id == "3dgs_original":
            require(method.get("three_k_training_allowed") is False, prefix + "completed 3K run must not remain launchable")
            require(
                method.get("three_k_qualification_status") == "FORMAL_3K_COMPLETE_RANKED",
                prefix + "qualification status mismatch",
            )
        elif method_id == "2dgs":
            require(method.get("three_k_training_allowed") is False, prefix + "completed 3K run must not remain launchable")
            require(
                method.get("three_k_qualification_status") == "FORMAL_3K_COMPLETE_RANKED",
                prefix + "qualification status mismatch",
            )
        elif method_id == "gof":
            require(method.get("three_k_training_allowed") is False, prefix + "static-only qualification must remain locked")
            require(
                method.get("three_k_qualification_status") == "STATIC_PREFLIGHT_PASS_GPU_PENDING",
                prefix + "qualification status mismatch",
            )
        else:
            require(method.get("three_k_training_allowed") is False, prefix + "training must remain locked")
            require(method.get("three_k_qualification_status") == "NOT_RUN", prefix + "qualification status mismatch")
        require(method.get("full_scene_matrix_eligible") is False, prefix + "full matrix must remain locked")

    qgs = next((method for method in methods if method.get("method_id") == "qgs"), {})
    require(qgs.get("source", {}).get("official_repository") == "https://github.com/will-zzy/QGS", "QGS official repository not recorded")
    require(qgs.get("source", {}).get("license_status") == "present_at_frozen_commit", "QGS license evidence missing")
    require(qgs.get("source", {}).get("license_git_blob") == "c869e695fa63bfde6f887d63a24a2a71f03480ac", "QGS license blob mismatch")

    three_dgs = next((method for method in methods if method.get("method_id") == "3dgs_original"), {})
    adapter = three_dgs.get("common_adapter", {})
    formal = three_dgs.get("formal_3k_result", {})
    require(
        three_dgs.get("recipe_status")
        == "FROZEN_3K_TRAINING_AUTHORIZED",
        "3DGS recipe status mismatch",
    )
    require(
        adapter.get("status")
        == "GPU_BUILD_AND_REAL_3K_PACKET_EVALUATOR_PREFLIGHT_PASS",
        "3DGS adapter status mismatch",
    )

    evidence_specs = [
        (three_dgs, "recipe", "recipe_sha256", "3DGS recipe"),
        (three_dgs, "recipe_validation", "recipe_validation_sha256", "3DGS recipe validation"),
        (adapter, "config", "config_sha256", "3DGS adapter config"),
        (adapter, "static_report", "static_report_sha256", "3DGS static adapter report"),
        (adapter, "cpu_report", "cpu_report_sha256", "3DGS CPU preflight"),
        (adapter, "end_to_end_cpu_smoke", "end_to_end_cpu_smoke_sha256", "3DGS evaluator smoke"),
        (adapter, "gpu_preflight", "gpu_preflight_sha256", "3DGS GPU preflight"),
        (
            adapter,
            "real_3k_packet_camera_preflight",
            "real_3k_packet_camera_preflight_sha256",
            "3DGS real 3K packet-camera preflight",
        ),
        (formal, "report", "report_sha256", "3DGS formal 3K result"),
        (
            formal,
            "adapter_linux_identity_proof",
            "adapter_linux_identity_proof_sha256",
            "3DGS Linux adapter identity proof",
        ),
    ]
    evidence: dict[str, dict[str, Any]] = {}
    resolved_repo = repo_root.resolve()
    for container, path_key, sha_key, label in evidence_specs:
        relative = container.get(path_key)
        expected_sha = str(container.get(sha_key, ""))
        require(isinstance(relative, str) and bool(relative), f"{label} path missing")
        require(bool(SHA256.fullmatch(expected_sha)), f"{label} SHA invalid")
        if isinstance(relative, str) and relative:
            path = (resolved_repo / relative).resolve()
            require(path.is_relative_to(resolved_repo), f"{label} escapes repo")
            require(path.is_file(), f"{label} missing")
            if path.is_file():
                require(file_sha256(path) == expected_sha, f"{label} SHA mismatch")
                evidence[path_key] = json.loads(path.read_text(encoding="utf-8"))

    recipe = evidence.get("recipe", {})
    require(recipe.get("protocol_id") == "m3m_gcp_native_quarter_geometry_v2", "3DGS recipe protocol mismatch")
    require(recipe.get("execution", {}).get("training_authorized") is True, "3DGS recipe training authorization missing")
    require(recipe.get("qualification", {}).get("three_k_training_allowed") is True, "3DGS recipe qualification unlock missing")
    recipe_validation = evidence.get("recipe_validation", {})
    require(recipe_validation.get("passed") is True, "3DGS recipe validation did not pass")
    require(recipe_validation.get("training_allowed") is True, "3DGS recipe validation did not authorize training")
    config = evidence.get("config", {})
    require(
        config.get("status") == "STATIC_PATCH_PREFLIGHT_PASS_GPU_RENDER_PREFLIGHT_PENDING",
        "3DGS adapter config status mismatch",
    )
    static_report = evidence.get("static_report", {})
    require(static_report.get("passed") is True, "3DGS static adapter report did not pass")
    require(static_report.get("gpu_render_preflight_passed") is False, "3DGS static report claims GPU render pass")
    cpu_report = evidence.get("cpu_report", {})
    require(cpu_report.get("status") == "PASS", "3DGS CPU preflight did not pass")
    smoke = evidence.get("end_to_end_cpu_smoke", {})
    require(smoke.get("status") == "PASS", "3DGS evaluator smoke did not pass")
    assertions = smoke.get("assertions", {})
    require(assertions.get("method_specific_sim3_fitted") is False, "3DGS evaluator smoke fitted a method-specific Sim3")
    require(assertions.get("scene_status") == "INCOMPLETE_UNRANKED", "3DGS evaluator smoke did not enforce incomplete status")
    require(assertions.get("ranking_eligible") is False, "3DGS evaluator smoke incorrectly allowed ranking")
    require(assertions.get("maximum_oblique_azimuth_circular_bin_separation", 0) >= 2, "3DGS evaluator smoke did not exercise azimuth separation")
    gpu = evidence.get("gpu_preflight", {})
    require(gpu.get("status") == "PASS" and gpu.get("passed") is True, "3DGS GPU preflight did not pass")
    require(gpu.get("training_started") is False, "3DGS GPU preflight unexpectedly trained a model")
    real = evidence.get("real_3k_packet_camera_preflight", {})
    require(real.get("status") == "PASS", "3DGS real packet-camera preflight did not pass")
    require(
        real.get("packet_export", {}).get("all_packet_recomputations_passed") is True,
        "3DGS real packet recomputation did not pass",
    )
    require(
        real.get("evaluator", {}).get("method_specific_sim3_fitted") is False,
        "3DGS real packet-camera preflight fitted a method-specific Sim(3)",
    )
    require(
        real.get("training_unlock", {})
        == {
            "full_scene_matrix_allowed": False,
            "global_unlock": False,
            "method_id": "3dgs_original",
            "three_k_training_allowed": True,
        },
        "3DGS real packet-camera preflight unlock scope mismatch",
    )
    formal_report = evidence.get("report", {})
    require(formal.get("status") == "COMPLETE_RANKED", "3DGS formal result status mismatch")
    require(formal.get("rerun_allowed") is False, "3DGS completed seed-0 run must not be repeated")
    require(formal_report.get("schema") == "m3m_native_quarter_3dgs_formal_3k_run_v1", "3DGS formal report schema mismatch")
    require(formal_report.get("status") == "PASS" and formal_report.get("passed") is True, "3DGS formal report did not pass")
    require(formal_report.get("result_kind") == "formal_benchmark_result_not_preflight", "3DGS result is not formal")
    require(formal_report.get("protocol_id") == "m3m_gcp_native_quarter_geometry_v2", "3DGS formal protocol mismatch")
    require(formal_report.get("method_id") == "3dgs_original", "3DGS formal method mismatch")
    require(formal_report.get("scene") == "gcp_3000_20260602", "3DGS formal scene mismatch")
    require(formal_report.get("seed") == 0 and formal_report.get("iterations") == 30000, "3DGS formal recipe mismatch")
    require(formal_report.get("input", {}).get("train_view_count") == 82, "3DGS formal train-view count mismatch")
    require(formal_report.get("input", {}).get("heldout_test_view_count") == 12, "3DGS formal holdout count mismatch")
    require(formal_report.get("input", {}).get("training_resolution_argument") == 1, "3DGS formal resolution mismatch")
    require(formal_report.get("training", {}).get("source_commit") == three_dgs.get("source", {}).get("commit"), "3DGS formal source commit mismatch")
    require(formal_report.get("training", {}).get("source_tree") == three_dgs.get("source", {}).get("tree"), "3DGS formal source tree mismatch")
    require(formal_report.get("training", {}).get("source_status_porcelain_after") == "", "3DGS training source is dirty")
    require(formal_report.get("training", {}).get("resource_status") == "PASS", "3DGS training resource probe failed")
    require(formal_report.get("training", {}).get("final_ply_sha256") == formal.get("final_checkpoint_sha256"), "3DGS final checkpoint SHA mismatch")
    packet_export = formal_report.get("packet_export", {})
    require(packet_export.get("manifest_sha256") == formal.get("packet_manifest_sha256"), "3DGS packet manifest SHA mismatch")
    require(packet_export.get("packet_count") == 66, "3DGS formal packet count mismatch")
    require(packet_export.get("packet_disk_mismatch_count") == 0, "3DGS formal packet identity mismatch")
    require(packet_export.get("packet_recomputation_all_passed") is True, "3DGS formal packet recomputation failed")
    require(packet_export.get("variance_validation_fail_pixel_total") == 0, "3DGS formal packet variance validation failed")
    formal_evaluation = formal_report.get("evaluation", {})
    require(formal_evaluation.get("evaluator_commit") == formal.get("evaluator_commit"), "3DGS evaluator commit mismatch")
    require(formal_evaluation.get("status") == "COMPLETE_RANKED", "3DGS formal evaluation is not complete-ranked")
    require(formal_evaluation.get("ranking_eligible") is True, "3DGS formal result is not ranking eligible")
    require(formal_evaluation.get("method_specific_sim3_fitted") is False, "3DGS formal evaluation fitted method-specific Sim(3)")
    require(
        formal_evaluation.get("point_counts")
        == {"checkpoint_passed": 4, "checkpoint_total": 4, "control_passed": 5, "control_total": 5},
        "3DGS formal point coverage mismatch",
    )
    checkpoint = formal_evaluation.get("residual_statistics", {}).get("checkpoint", {})
    require(checkpoint.get("rmse_3d_m") == formal.get("checkpoint_rmse_3d_m"), "3DGS checkpoint RMSE-3D mismatch")
    require(checkpoint.get("rmse_h_m") == formal.get("checkpoint_rmse_h_m"), "3DGS checkpoint RMSE-H mismatch")
    require(checkpoint.get("rmse_z_m") == formal.get("checkpoint_rmse_z_m"), "3DGS checkpoint RMSE-Z mismatch")
    linux_identity = evidence.get("adapter_linux_identity_proof", {})
    require(linux_identity.get("status") == "PASS" and linux_identity.get("passed") is True, "3DGS Linux adapter identity proof failed")
    require(
        linux_identity.get("renderer_patch_sha256")
        == "e88b1ca418751228862c4e4d99c2cf4b5f714838fdbdf1bfa81f75fb85e81363",
        "3DGS Linux renderer patch identity mismatch",
    )
    require(
        linux_identity.get("rasterizer_patch_sha256")
        == "f787df935f45af61dd836e5d430c216a67b93986d9c9fc57e615e1463e6d2068",
        "3DGS Linux rasterizer patch identity mismatch",
    )

    two_dgs = next((method for method in methods if method.get("method_id") == "2dgs"), {})
    two_adapter = two_dgs.get("common_adapter", {})
    two_formal = two_dgs.get("formal_3k_result", {})
    require(two_dgs.get("recipe_status") == "FROZEN_3K_FORMAL_COMPLETE_RELOCKED", "2DGS recipe status mismatch")
    require(
        two_adapter.get("status") == "GPU_BUILD_SYNTHETIC_AND_REAL_3K_PACKET_EVALUATOR_PREFLIGHT_PASS",
        "2DGS adapter status mismatch",
    )
    two_specs = [
        (two_dgs, "recipe", "recipe_sha256", "2DGS recipe"),
        (two_adapter, "config", "config_sha256", "2DGS adapter config"),
        (two_adapter, "static_report", "static_report_sha256", "2DGS static report"),
        (
            two_adapter,
            "gpu_real_3k_qualification_report",
            "gpu_real_3k_qualification_report_sha256",
            "2DGS GPU/real-3K qualification report",
        ),
        (two_formal, "report", "report_sha256", "2DGS formal 3K result"),
    ]
    two_evidence: dict[str, dict[str, Any]] = {}
    for container, path_key, sha_key, label in two_specs:
        relative = container.get(path_key)
        expected_sha = str(container.get(sha_key, ""))
        require(isinstance(relative, str) and bool(relative), f"{label} path missing")
        require(bool(SHA256.fullmatch(expected_sha)), f"{label} SHA invalid")
        if isinstance(relative, str) and relative:
            path = (resolved_repo / relative).resolve()
            require(path.is_relative_to(resolved_repo), f"{label} escapes repo")
            require(path.is_file(), f"{label} missing")
            if path.is_file():
                require(file_sha256(path) == expected_sha, f"{label} SHA mismatch")
                two_evidence[path_key] = json.loads(path.read_text(encoding="utf-8"))
    two_recipe = two_evidence.get("recipe", {})
    require(two_recipe.get("protocol_id") == value.get("protocol_id"), "2DGS recipe protocol mismatch")
    require(two_recipe.get("method", {}).get("method_id") == "2dgs", "2DGS recipe method mismatch")
    require(two_recipe.get("execution", {}).get("training_authorized") is False, "2DGS completed recipe remains training-authorized")
    two_recipe_qualification = two_recipe.get("qualification", {})
    require(two_recipe_qualification.get("three_k_training_allowed") is False, "2DGS completed recipe remains launchable")
    require(two_recipe_qualification.get("formal_3k_completed") is True, "2DGS formal completion state missing")
    require(
        two_recipe_qualification.get("formal_3k_result", {}).get("rerun_allowed") is False,
        "2DGS recipe formal rerun lock missing",
    )
    require(two_recipe_qualification.get("full_scene_matrix_allowed") is False, "2DGS recipe unlocked the full matrix")
    require(two_recipe_qualification.get("global_training_allowed") is False, "2DGS recipe unlocked global training")
    require(two_recipe.get("source_provenance", {}).get("repository_commit") == two_dgs.get("source", {}).get("commit"), "2DGS recipe commit mismatch")
    require(two_recipe.get("source_provenance", {}).get("repository_tree") == two_dgs.get("source", {}).get("tree"), "2DGS recipe tree mismatch")
    two_config = two_evidence.get("config", {})
    require(
        two_config.get("status") == "GPU_BUILD_SYNTHETIC_AND_REAL_3K_PACKET_EVALUATOR_PREFLIGHT_PASS",
        "2DGS adapter config state mismatch",
    )
    require(two_config.get("raw_output", {}).get("primary_track_uses_native_planes_only") is True, "2DGS primary A/M1 planes are not pinned as native")
    require(two_config.get("training_identity", {}).get("training_patch_allowed") is False, "2DGS training patch was allowed")
    require(
        two_config.get("formal_3k_result", {}).get("report_sha256") == two_formal.get("report_sha256"),
        "2DGS adapter formal report SHA mismatch",
    )
    require(
        two_config.get("formal_3k_result", {}).get("checkpoint_sha256") == two_formal.get("final_checkpoint_sha256"),
        "2DGS adapter final checkpoint SHA mismatch",
    )
    require(
        two_config.get("formal_3k_result", {}).get("packet_manifest_sha256") == two_formal.get("packet_manifest_sha256"),
        "2DGS adapter packet manifest SHA mismatch",
    )
    two_static = two_evidence.get("static_report", {})
    require(two_static.get("status") == "PASS" and two_static.get("passed") is True, "2DGS static validation failed")
    require(two_static.get("formal_training_authorized") is False, "2DGS static report prematurely authorizes training")
    require(two_static.get("primary_common_planes_are_native_2dgs_outputs") is True, "2DGS native common-plane proof missing")
    two_qualification = two_evidence.get("gpu_real_3k_qualification_report", {})
    require(
        two_qualification.get("schema") == "m3m_gcp_native_quarter_2dgs_gpu_real_3k_qualification_v1",
        "2DGS qualification report schema mismatch",
    )
    require(two_qualification.get("status") == "PASS" and two_qualification.get("passed") is True, "2DGS qualification failed")
    require(two_qualification.get("protocol_id") == value.get("protocol_id"), "2DGS qualification protocol mismatch")
    require(two_qualification.get("method_id") == "2dgs", "2DGS qualification method mismatch")
    require(two_qualification.get("boundary", {}).get("formal_training_started") is False, "2DGS formal training already started")
    require(two_qualification.get("boundary", {}).get("benchmark_score_claim") is False, "2DGS qualification is mislabeled as a result")
    require(two_qualification.get("source", {}).get("official_training_source_clean_after_qualification") is True, "2DGS training source is dirty")
    require(two_qualification.get("raw_moment_conformance", {}).get("status") == "PASS", "2DGS raw-moment conformance failed")
    require(
        two_qualification.get("raw_moment_conformance", {}).get("primary_common_planes_are_native") == ["A", "M1"],
        "2DGS common A/M1 native-plane proof mismatch",
    )
    two_packet = two_qualification.get("packet_preflight", {})
    require(two_packet.get("formal_packet_camera_count") == 66, "2DGS packet-camera count mismatch")
    require(two_packet.get("all_packet_recomputations_passed") is True, "2DGS packet recomputation failed")
    require(two_packet.get("variance_validation_failing_pixel_total") == 0, "2DGS packet variance validation failed")
    two_eval = two_qualification.get("evaluator", {})
    require(two_eval.get("status") == "COMPLETE_RANKED" and two_eval.get("ranking_eligible") is True, "2DGS qualification evaluator failed")
    require(two_eval.get("method_specific_sim3_fitted") is False, "2DGS qualification fitted a method-specific Sim(3)")
    require(
        two_eval.get("point_counts")
        == {"checkpoint_passed": 4, "checkpoint_total": 4, "control_passed": 5, "control_total": 5},
        "2DGS qualification point coverage mismatch",
    )
    require(two_qualification.get("technical_resume", {}).get("result_driven_retry") is False, "2DGS qualification used a result-driven retry")
    require(
        two_qualification.get("training_unlock")
        == {
            "method_id": "2dgs",
            "scene": "gcp_3000_20260602",
            "seed": 0,
            "iterations": 30000,
            "single_fresh_run_allowed": True,
            "resume_allowed": False,
            "rerun_after_completed_result_allowed": False,
            "full_scene_matrix_allowed": False,
            "global_unlock": False,
        },
        "2DGS qualification unlock scope mismatch",
    )

    two_formal_report = two_evidence.get("report", {})
    require(two_formal.get("status") == "COMPLETE_RANKED", "2DGS formal result status mismatch")
    require(two_formal.get("rerun_allowed") is False, "2DGS completed seed-0 run must not be repeated")
    require(two_formal.get("single_seed_only") is True, "2DGS single-seed boundary missing")
    require(two_formal.get("statistical_significance_claim") is False, "2DGS registry makes a significance claim")
    require(
        two_formal_report.get("schema") == "m3m_native_quarter_2dgs_formal_3k_run_v1",
        "2DGS formal report schema mismatch",
    )
    require(
        two_formal_report.get("status") == "PASS" and two_formal_report.get("passed") is True,
        "2DGS formal report did not pass",
    )
    require(
        two_formal_report.get("result_kind") == "formal_benchmark_result_not_preflight",
        "2DGS result is not formal",
    )
    require(two_formal_report.get("protocol_id") == value.get("protocol_id"), "2DGS formal protocol mismatch")
    require(two_formal_report.get("method_id") == "2dgs", "2DGS formal method mismatch")
    require(two_formal_report.get("scene") == "gcp_3000_20260602", "2DGS formal scene mismatch")
    require(
        two_formal_report.get("seed") == 0 and two_formal_report.get("iterations") == 30000,
        "2DGS formal recipe mismatch",
    )
    require(two_formal_report.get("single_seed_only") is True, "2DGS formal report single-seed boundary missing")
    require(
        two_formal_report.get("statistical_significance_claim") is False,
        "2DGS formal report makes a significance claim",
    )
    two_formal_input = two_formal_report.get("input", {})
    require(two_formal_input.get("train_view_count") == 82, "2DGS formal train-view count mismatch")
    require(two_formal_input.get("heldout_test_view_count") == 12, "2DGS formal holdout count mismatch")
    require(two_formal_input.get("training_resolution_argument") == 1, "2DGS formal resolution mismatch")
    require(two_formal_input.get("depth_ratio") == 0.0, "2DGS formal depth-ratio mismatch")
    require(
        two_formal_input.get("all_94_image_hashes_verified_before_launch") is True,
        "2DGS formal input hashes were not verified",
    )
    two_formal_source = two_formal_report.get("source", {})
    require(
        two_formal_source.get("official_repository_commit") == two_dgs.get("source", {}).get("commit"),
        "2DGS formal source commit mismatch",
    )
    require(
        two_formal_source.get("official_repository_tree") == two_dgs.get("source", {}).get("tree"),
        "2DGS formal source tree mismatch",
    )
    require(two_formal_source.get("official_training_source_modified") is False, "2DGS formal training source was modified")
    require(
        two_formal_source.get("official_training_source_status_porcelain_after") == "",
        "2DGS formal training source is dirty",
    )
    two_formal_training = two_formal_report.get("training", {})
    require(two_formal_training.get("resource_status") == "PASS", "2DGS formal training resource probe failed")
    require(
        two_formal_training.get("memory_events_delta", {}).get("oom") == 0
        and two_formal_training.get("memory_events_delta", {}).get("oom_kill") == 0,
        "2DGS formal training recorded an OOM",
    )
    require(
        two_formal_training.get("final_ply_sha256") == two_formal.get("final_checkpoint_sha256"),
        "2DGS final checkpoint SHA mismatch",
    )
    two_formal_packet = two_formal_report.get("packet_export", {})
    require(two_formal_packet.get("status") == "PASS", "2DGS formal packet export failed")
    require(two_formal_packet.get("manifest_sha256") == two_formal.get("packet_manifest_sha256"), "2DGS packet manifest SHA mismatch")
    require(two_formal_packet.get("packet_count") == 66, "2DGS formal packet count mismatch")
    require(two_formal_packet.get("packet_recomputation_all_passed") is True, "2DGS formal packet recomputation failed")
    require(two_formal_packet.get("variance_validation_fail_pixel_total") == 0, "2DGS formal packet variance validation failed")
    two_formal_evaluation = two_formal_report.get("evaluation", {})
    require(two_formal_evaluation.get("evaluator_commit") == two_formal.get("evaluator_commit"), "2DGS evaluator commit mismatch")
    require(two_formal_evaluation.get("status") == "COMPLETE_RANKED", "2DGS formal evaluation is not complete-ranked")
    require(two_formal_evaluation.get("ranking_eligible") is True, "2DGS formal result is not ranking eligible")
    require(two_formal_evaluation.get("method_specific_sim3_fitted") is False, "2DGS formal evaluation fitted method-specific Sim(3)")
    require(
        two_formal_evaluation.get("point_counts")
        == {"checkpoint_passed": 4, "checkpoint_total": 4, "control_passed": 5, "control_total": 5},
        "2DGS formal point coverage mismatch",
    )
    two_checkpoint = two_formal_evaluation.get("residual_statistics", {}).get("checkpoint", {})
    require(two_checkpoint.get("rmse_3d_m") == two_formal.get("checkpoint_rmse_3d_m"), "2DGS checkpoint RMSE-3D mismatch")
    require(two_checkpoint.get("rmse_h_m") == two_formal.get("checkpoint_rmse_h_m"), "2DGS checkpoint RMSE-H mismatch")
    require(two_checkpoint.get("rmse_z_m") == two_formal.get("checkpoint_rmse_z_m"), "2DGS checkpoint RMSE-Z mismatch")
    require(two_formal_report.get("final_gpu_compute_process_count") == 0, "2DGS formal run left a GPU compute process")

    gof = next((method for method in methods if method.get("method_id") == "gof"), {})
    gof_adapter = gof.get("common_adapter", {})
    require(gof.get("recipe_status") == "FROZEN_STATIC_PREFLIGHT_GPU_PENDING", "GOF recipe status mismatch")
    require(gof_adapter.get("status") == "STATIC_PATCH_PREFLIGHT_PASS_GPU_PENDING", "GOF adapter status mismatch")
    gof_specs = [
        (gof, "recipe", "recipe_sha256", "GOF recipe"),
        (gof_adapter, "config", "config_sha256", "GOF adapter config"),
        (gof_adapter, "static_report", "static_report_sha256", "GOF static report"),
    ]
    gof_evidence: dict[str, dict[str, Any]] = {}
    for container, path_key, sha_key, label in gof_specs:
        relative = container.get(path_key)
        expected_sha = str(container.get(sha_key, ""))
        require(isinstance(relative, str) and bool(relative), f"{label} path missing")
        require(bool(SHA256.fullmatch(expected_sha)), f"{label} SHA invalid")
        if isinstance(relative, str) and relative:
            path = (resolved_repo / relative).resolve()
            require(path.is_relative_to(resolved_repo), f"{label} escapes repo")
            require(path.is_file(), f"{label} missing")
            if path.is_file():
                require(file_sha256(path) == expected_sha, f"{label} SHA mismatch")
                gof_evidence[path_key] = json.loads(path.read_text(encoding="utf-8"))
    gof_recipe = gof_evidence.get("recipe", {})
    require(gof_recipe.get("protocol_id") == value.get("protocol_id"), "GOF recipe protocol mismatch")
    require(gof_recipe.get("method", {}).get("method_id") == "gof", "GOF recipe method mismatch")
    require(gof_recipe.get("source_provenance", {}).get("repository_commit") == gof.get("source", {}).get("commit"), "GOF recipe commit mismatch")
    require(gof_recipe.get("source_provenance", {}).get("repository_tree") == gof.get("source", {}).get("tree"), "GOF recipe tree mismatch")
    require(gof_recipe.get("build_compatibility", {}).get("training_source_modified") is False, "GOF training source was modified")
    require(gof_recipe.get("execution", {}).get("training_authorized") is False, "GOF recipe prematurely authorizes training")
    gof_qualification = gof_recipe.get("qualification", {})
    require(gof_qualification.get("recipe_static_freeze_passed") is True, "GOF static recipe freeze missing")
    require(gof_qualification.get("local_patch_replay_passed") is True, "GOF patch replay missing")
    require(gof_qualification.get("three_k_training_allowed") is False, "GOF recipe prematurely unlocks 3K")
    require(gof_qualification.get("full_scene_matrix_allowed") is False, "GOF recipe unlocks full matrix")
    require(gof_qualification.get("global_training_allowed") is False, "GOF recipe unlocks global training")
    for key in (
        "real_input_loader_preflight_passed",
        "gpu_official_training_extension_build_passed",
        "gpu_evaluation_adapter_build_passed",
        "synthetic_raw_moment_conformance_passed",
        "frozen_3k_real_packet_camera_preflight_passed",
        "one_iteration_technical_smoke_completed",
        "formal_3k_completed",
    ):
        require(gof_qualification.get(key) is False, f"GOF pending gate unexpectedly passed: {key}")
    gof_config = gof_evidence.get("config", {})
    require(gof_config.get("status") == "STATIC_PATCH_PREFLIGHT_PASS_GPU_PENDING", "GOF adapter config state mismatch")
    require(gof_config.get("raw_output", {}).get("rendered_image_plane_indices") == [7, 9, 10, 11], "GOF raw plane map mismatch")
    require(gof_config.get("raw_output", {}).get("physical_surface_claim") is False, "GOF adapter makes physical-surface claim")
    require(gof_config.get("training_identity", {}).get("training_patch_allowed") is False, "GOF training patch was allowed")
    require(gof_config.get("training_identity", {}).get("checkpoint_mutation_allowed") is False, "GOF checkpoint mutation was allowed")
    gof_static = gof_evidence.get("static_report", {})
    require(gof_static.get("schema") == "m3m_gcp_native_quarter_gof_static_validation_v1", "GOF static schema mismatch")
    require(gof_static.get("status") == "PASS" and gof_static.get("passed") is True, "GOF static validation failed")
    require(gof_static.get("formal_training_authorized") is False, "GOF static report prematurely authorizes training")
    require(gof_static.get("native_depth_channel_used_as_common_primary") is False, "GOF native median/max depth leaked into common primary")
    require(gof_static.get("native_opacity_level_set_mesh_role") == "diagnostic_only", "GOF native surface role mismatch")
    require(gof_static.get("physical_surface_claim") is False, "GOF static report makes physical-surface claim")

    city = next((method for method in methods if method.get("method_id") == "citygs_x"), {})
    require("redistribution_blocked" in str(city.get("source", {}).get("license_status", "")), "CityGS-X redistribution risk missing")
    metro = next((method for method in methods if method.get("method_id") == "metrogs"), {})
    prior_names = {str(prior.get("name")) for prior in metro.get("external_priors", [])}
    require(prior_names == {"pointmap dense initialization", "MoGe-2"}, "MetroGS prior inventory incomplete")
    require(value.get("global_training_allowed") is False, "global training lock missing")
    require(
        value.get("per_method_training_allowed_methods") == [],
        "per-method training allowlist mismatch",
    )
    return {
        "schema": "m3m_gcp_native_quarter_method_registry_validation_v2",
        "passed": not errors,
        "method_count": len(methods),
        "method_ids": ids,
        "external_prior_methods": sorted(EXTERNAL_PRIOR_METHODS),
        "training_allowed_methods": sorted(
            method["method_id"] for method in methods if method.get("three_k_training_allowed") is True
        ),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--repo_root", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    repo_root = (args.repo_root or args.registry.resolve().parents[1]).resolve()
    result = validate_registry(json.loads(args.registry.read_text(encoding="utf-8")), repo_root)
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
