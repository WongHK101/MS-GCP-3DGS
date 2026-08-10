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
        value.get("status") == "candidate_pool_frozen_3dgs_3k_complete_ranked_other_methods_locked",
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
