#!/usr/bin/env python3
"""Validate the active v3 method registry under the immutable protocol-v2 contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_ACTIVE = [
    "3dgs_original",
    "2dgs",
    "pgsr",
    "rade_gs",
    "qgs",
    "gsprior",
    "sof",
    "citygaussian_v2",
    "citygs_x",
    "metrogs",
]
EXPECTED_CANDIDATES = [
    "pgsr",
    "rade_gs",
    "qgs",
    "gsprior",
    "sof",
    "citygaussian_v2",
    "citygs_x",
    "metrogs",
]
EXPECTED_SOURCE_IDENTITIES = {
    "pgsr": (
        "https://github.com/zju3dv/PGSR",
        "de24f1a38b350387e8d8fe381b2cd70c1ae946e7",
        "8504a351b4a7938ef0b15647c1e5efb01e7ea013",
    ),
    "rade_gs": (
        "https://github.com/HKUST-SAIL/RaDe-GS",
        "d72f20792005ae1d6555a82aa2d15345f247604e",
        "e37a9f1bfec5b593371402d19fb5259cbcb6efa1",
    ),
    "qgs": (
        "https://github.com/will-zzy/QGS",
        "74d05c945e99fcaef7afe5a8831903be71ad9b55",
        "c20af6da770b9ecc9c4e1730b40671ea63ec1419",
    ),
    "gsprior": (
        "https://github.com/takeshie/GSPrior",
        "dcb7c89fb6b60f068b440de45d064ecc7fbcba55",
        "779073585e88b85217d522d0ab345365346cd17f",
    ),
    "sof": (
        "https://github.com/r4dl/SOF",
        "b9eb4170c843014f5f96d54924976161bd675469",
        "d5ece75b8255c5dd6abf97482ddbf34d20dca707",
    ),
    "citygaussian_v2": (
        "https://github.com/Linketic/CityGaussian",
        "e84c7c8774dd11d3f4189be3488e1220afa20a86",
        "be088977358cb36bac000caec396eff3758c74b2",
    ),
    "citygs_x": (
        "https://github.com/gyy456/CityGS-X",
        "27617f2486505e3b6fe75345edf7c2b11161bc2a",
        "f8b1b5148c1f47420ab698fd069bdb78acf901ab",
    ),
    "metrogs": (
        "https://github.com/M3phist0/MetroGS",
        "8cf9ac13c0c34b65c1a935d181c4634909e60f3f",
        "7e92b13095cf4a031d7eb8593e10616db154abbf",
    ),
}
EXPECTED_FORMAL_REPORTS = {
    "3dgs_original": (
        "docs/protocol_evidence/3dgs_native_quarter_formal_3k_seed0_30k_v1.json",
        "b8311cc687d9ab5f3c49d58f4473d26a70281c7b577177e3742c0abe00163b40",
    ),
    "2dgs": (
        "docs/protocol_evidence/2dgs_native_quarter_formal_3k_seed0_30k_v1.json",
        "d1967db9991e018eb3c8d3d01d95b42eb9dca162ed64d926a70a067567d7a79e",
    ),
    "pgsr": (
        "docs/protocol_evidence/pgsr_native_quarter_formal_3k_seed0_30k_v1.json",
        "48ab33cfd22f170f24a2bd170f69e4f6e8f84281c42475aad62e9a6e43c28d5f",
    ),
    "rade_gs": (
        "docs/protocol_evidence/rade_gs_native_quarter_formal_3k_seed0_30k_v1.json",
        "6586957f3b0da3d2a1a83ac09e65f8228cc395d081f1d1690e2b09047ee674dc",
    ),
    "gof": (
        "docs/protocol_evidence/gof_native_quarter_formal_3k_seed0_30k_v1.json",
        "04f72d495fba0c74cc2124246feea931ffcceaaba12005529b8ec2ad088eda8a",
    ),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def validate_registry(repo_root: Path, registry_path: Path) -> dict[str, Any]:
    value = json.loads(registry_path.read_text(encoding="utf-8"))
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(value.get("schema") == "m3m_gcp_native_quarter_method_registry_v3", "unknown schema")
    require(value.get("protocol_id") == "m3m_gcp_native_quarter_geometry_v2", "protocol semantics changed")
    require(
        value.get("status")
        in {
            "EIGHT_METHOD_3K_BATCH_PGSR_AND_RADE_GS_COMPLETE_QGS_ONE_USE_GATE_OPEN",
            "EIGHT_METHOD_3K_BATCH_ONE_USE_GATE_OPEN",
            "EIGHT_METHOD_3K_BATCH_ACTIVE",
        },
        "unexpected registry status",
    )
    require(value.get("batch_id") == "m3m-gcp-3k-eight-method-seed0-20260818", "batch identity mismatch")

    plan = value.get("execution_plan", {})
    plan_path = repo_root / str(plan.get("path", ""))
    require(plan_path.is_file(), "execution plan missing")
    if plan_path.is_file():
        require(file_sha256(plan_path) == plan.get("sha256"), "execution plan SHA mismatch")
        text = plan_path.read_text(encoding="utf-8")
        for token in (
            "m3m_gcp_native_quarter_geometry_v2",
            "TECHNICALLY_QUALIFIED",
            "COMPLETE_RANKED",
            "INCOMPLETE_UNRANKED",
            "historical_complete_retired",
            "PGSR -> RaDe-GS -> QGS -> GSPrior -> SOF -> CityGaussianV2 -> CityGS-X -> MetroGS",
        ):
            require(token in text, f"execution plan is missing contract token: {token}")

    source_release = value.get("source_data_release", {})
    require(
        source_release.get("release_root_digest_sha256")
        == "513f8999fe4b110f15bcbecad7932895781cee755ee9ccd7a14ff10298546d75",
        "source release digest mismatch",
    )
    require(source_release.get("formal_scene") == "gcp_3000_20260602", "formal 3K scene mismatch")
    require(source_release.get("train_view_count") == 82, "train-view count mismatch")
    require(source_release.get("heldout_test_view_count") == 12, "held-out view count mismatch")
    require(source_release.get("training_resolution_argument") == 1, "native-quarter resolution mismatch")

    common = value.get("formal_common_track", {})
    require(common.get("name") == "render_support_expected_camera_z", "common depth track changed")
    require(common.get("cross_method_primary") is True, "common track is not primary")
    require(common.get("physical_surface_claim") is False, "common track makes a physical-surface claim")

    freeze = value.get("source_freeze", {})
    require(freeze.get("status") == "PASS", "source freeze did not pass")
    require(freeze.get("method_count") == 8, "source freeze method count mismatch")
    require(freeze.get("all_commits_trees_archives_and_clean_linux_checkouts_verified") is True, "source identity proof missing")
    require(freeze.get("windows_crlf_transport_worktrees_rejected") is True, "CRLF rejection evidence missing")
    require(is_sha256(freeze.get("remote_inventory_sha256")), "source inventory SHA malformed")
    freeze_evidence_path = repo_root / str(freeze.get("repository_evidence", ""))
    require(freeze_evidence_path.is_file(), "source freeze repository evidence missing")
    if freeze_evidence_path.is_file():
        require(
            file_sha256(freeze_evidence_path) == freeze.get("repository_evidence_sha256"),
            "source freeze repository evidence SHA mismatch",
        )
        freeze_evidence = json.loads(freeze_evidence_path.read_text(encoding="utf-8"))
        require(freeze_evidence.get("passed") is True, "source freeze repository evidence failed")
        require(len(freeze_evidence.get("methods", [])) == 8, "source freeze repository evidence method count mismatch")
    correction_path = repo_root / str(freeze.get("license_newline_identity_correction", ""))
    require(correction_path.is_file(), "license newline identity correction evidence missing")
    if correction_path.is_file():
        require(
            file_sha256(correction_path) == freeze.get("license_newline_identity_correction_sha256"),
            "license newline identity correction SHA mismatch",
        )
        correction = json.loads(correction_path.read_text(encoding="utf-8"))
        require(correction.get("passed") is True, "license newline identity correction failed")
        require(correction.get("licensed_method_count_audited") == 6, "licensed-method correction count mismatch")

    scope = value.get("batch_execution_scope", {})
    require(scope.get("active_candidate_method_ids") == EXPECTED_CANDIDATES, "candidate order or membership mismatch")
    require(scope.get("formal_execution_order") == EXPECTED_CANDIDATES, "formal execution order mismatch")
    require(scope.get("reused_complete_ranked_method_ids") == ["3dgs_original", "2dgs"], "reused result scope mismatch")
    require(scope.get("historical_complete_retired_method_ids") == ["gof"], "retired GOF scope mismatch")
    require(scope.get("single_seed") == 0 and scope.get("multi_seed_authorized") is False, "seed contract mismatch")
    require(scope.get("per_method_external_audit_pause") is False, "per-method audit pause was reintroduced")
    require(scope.get("single_consolidated_audit_after_batch") is True, "consolidated audit contract missing")
    require(scope.get("technical_qualification_is_separate_from_result_completeness") is True, "qualification/result separation missing")
    require(scope.get("six_scene_matrix_status") == "LOCKED_PENDING_3K_BATCH_CLOSURE", "six-scene matrix unlocked")

    boundary = value.get("training_data_boundary", {})
    require(
        str(boundary.get("allowed_source_scene_root", "")).endswith("/gcp_3000_20260602"),
        "source scene allow-root mismatch",
    )
    require(
        str(boundary.get("formal_training_input_root", "")).endswith(
            "/formal_inputs/gcp_3000_20260602/train"
        ),
        "formal training input root mismatch",
    )
    require(len(boundary.get("denied_truth_roots", [])) >= 2, "truth deny roots missing")
    require(boundary.get("gcp_truth_training_access") is False, "GCP truth training access enabled")
    require(boundary.get("lidar_training_access") is False, "LiDAR training access enabled")
    require(boundary.get("orthophoto_truth_training_access") is False, "orthophoto truth training access enabled")
    require(boundary.get("result_driven_recipe_selection") is False, "result-driven recipe selection enabled")

    methods = value.get("methods", [])
    require(isinstance(methods, list), "methods must be a list")
    by_id = {method.get("method_id"): method for method in methods if isinstance(method, dict)}
    require(len(methods) == 11 and value.get("method_count") == 11, "method count must be 11")
    require(len(by_id) == 11, "method IDs are missing or duplicated")
    require(value.get("active_benchmark_method_count") == 10, "active method count must be 10")
    require(value.get("active_benchmark_method_ids") == EXPECTED_ACTIVE, "active method order or membership mismatch")
    require("gof" not in value.get("active_benchmark_method_ids", []), "retired GOF leaked into active pool")

    for method_id in EXPECTED_CANDIDATES:
        method = by_id.get(method_id, {})
        require(method.get("six_scene_run_allowed") is False, f"{method_id}: six-scene run unlocked")
        source = method.get("source", {})
        expected_url, expected_commit, expected_tree = EXPECTED_SOURCE_IDENTITIES[method_id]
        require(source.get("official_repository") == expected_url, f"{method_id}: official repository mismatch")
        require(source.get("commit") == expected_commit, f"{method_id}: source commit mismatch")
        require(source.get("tree") == expected_tree, f"{method_id}: source tree mismatch")
        require(is_sha256(source.get("transfer_archive_sha256")), f"{method_id}: transfer archive SHA malformed")
        if source.get("license_file"):
            require(is_sha256(source.get("frozen_linux_license_sha256")), f"{method_id}: frozen Linux license SHA malformed")
            require(
                is_sha256(source.get("legacy_rejected_windows_crlf_license_sha256")),
                f"{method_id}: legacy CRLF license SHA malformed",
            )
        if method_id == "pgsr":
            require(method.get("lifecycle_role") == "ACTIVE_3K_COMPLETE_RANKED", "pgsr: completed lifecycle role missing")
            require(
                method.get("technical_qualification_status") == "TECHNICALLY_QUALIFIED",
                "pgsr: technical qualification missing",
            )
            require(method.get("three_k_training_allowed") is False, "pgsr: completed method remains launchable")
            require(
                method.get("technical_full_matrix_eligibility") is True,
                "pgsr: technical matrix eligibility missing",
            )
            expected_refs = {
                "recipe": (
                    "configs/m3m_gcp_native_quarter_pgsr_3k_recipe_v1.json",
                    "ffa153ce396444dbbafedcb9fa9f11fef69ffb6f502315b4f02c14f25b1dabdd",
                ),
                "adapter_config": (
                    "configs/m3m_gcp_native_quarter_pgsr_renderer_adapter_v1.json",
                    "1cbebcf6239a74c9d54aa6c28067f6a837461badb548f22f957d50e24e23acc7",
                ),
                "qualification_report": (
                    "docs/protocol_evidence/pgsr_native_quarter_gpu_real_3k_qualification_v1.json",
                    "9f06e19d865602c1c3e0e73c1c7ab2be7dc00177a81207045becfaf691a80fdd",
                ),
                "truth_deny_report": (
                    "docs/protocol_evidence/pgsr_native_quarter_truth_deny_v1.json",
                    "1108bc61712a6d4066f94c612b03bc8ca6053a38495fbb79a432cc0c6904fc46",
                ),
            }
            for key, (relative, expected_sha) in expected_refs.items():
                require(method.get(key) == relative, f"pgsr: {key} path mismatch")
                require(method.get(f"{key}_sha256") == expected_sha, f"pgsr: {key} SHA mismatch")
                referenced = repo_root / relative
                require(referenced.is_file(), f"pgsr: {key} file missing")
                if referenced.is_file():
                    require(file_sha256(referenced) == expected_sha, f"pgsr: {key} file SHA mismatch")
            formal = method.get("formal_3k_result", {})
            require(formal.get("status") == "COMPLETE_RANKED", "pgsr: formal result not complete-ranked")
            require(formal.get("rerun_allowed") is False, "pgsr: formal rerun was enabled")
        elif method_id == "rade_gs":
            require(method.get("lifecycle_role") == "ACTIVE_3K_COMPLETE_RANKED", "rade_gs: completed lifecycle role missing")
            require(method.get("technical_qualification_status") == "TECHNICALLY_QUALIFIED", "rade_gs: qualification missing")
            require(method.get("three_k_training_allowed") is False, "rade_gs: completed method remains launchable")
            require(method.get("technical_full_matrix_eligibility") is True, "rade_gs: technical matrix eligibility missing")
            expected_refs = {
                "recipe": (
                    "configs/m3m_gcp_native_quarter_rade_gs_3k_recipe_v1.json",
                    "82524b5f0698d44639a2dc1f0dee7ea48699782113d04cc21784400e990393d2",
                ),
                "adapter_config": (
                    "configs/m3m_gcp_native_quarter_rade_gs_renderer_adapter_v1.json",
                    "7eb4afd438d360c880fd626acd01ef34886daddb110865454260bb955dc6cf3a",
                ),
                "qualification_report": (
                    "docs/protocol_evidence/rade_gs_native_quarter_gpu_real_3k_qualification_v1.json",
                    "de52f116f924830aacdebf6453ce2213f7fbf5962534bc321e0da592388477e4",
                ),
                "truth_deny_report": (
                    "docs/protocol_evidence/rade_gs_native_quarter_truth_deny_v1.json",
                    "4ebda6791bf03ae713372708db0b138ced24f71d6e78d51af5b0f30f4f037740",
                ),
            }
            for key, (relative, expected_sha) in expected_refs.items():
                require(method.get(key) == relative, f"rade_gs: {key} path mismatch")
                require(method.get(f"{key}_sha256") == expected_sha, f"rade_gs: {key} SHA mismatch")
                referenced = repo_root / relative
                require(referenced.is_file(), f"rade_gs: {key} file missing")
                if referenced.is_file():
                    require(file_sha256(referenced) == expected_sha, f"rade_gs: {key} file SHA mismatch")
            formal = method.get("formal_3k_result", {})
            require(formal.get("status") == "COMPLETE_RANKED", "rade_gs: formal result not complete-ranked")
            require(formal.get("rerun_allowed") is False, "rade_gs: formal rerun was enabled")
        elif method_id == "qgs":
            expected_refs = {
                "recipe": (
                    "configs/m3m_gcp_native_quarter_qgs_3k_recipe_v1.json",
                    "ef7f12d60a9ed739dcac8ba0407eec2882e5a29c14f87d479ce830531f55b720",
                ),
                "adapter_config": (
                    "configs/m3m_gcp_native_quarter_qgs_renderer_adapter_v1.json",
                    "0d785d7c3442c86d0121dfc2782b577c199443073cb35122888f26221e3d4061",
                ),
                "qualification_report": (
                    "docs/protocol_evidence/qgs_native_quarter_gpu_real_3k_qualification_v1.json",
                    "193f62c1d8cca2eab1a6529377e5095331bd3cf08608de251c61564ad34cd135",
                ),
                "truth_deny_report": (
                    "docs/protocol_evidence/qgs_native_quarter_truth_deny_v1.json",
                    "f0ca27ed879a35526549fa6fe15f9041ece2963561b285d40bbb6fa8327a5764",
                ),
            }
            for key, (relative, expected_sha) in expected_refs.items():
                require(method.get(key) == relative, f"qgs: {key} path mismatch")
                require(method.get(f"{key}_sha256") == expected_sha, f"qgs: {key} SHA mismatch")
                referenced = repo_root / relative
                require(referenced.is_file(), f"qgs: {key} file missing")
                if referenced.is_file():
                    require(file_sha256(referenced) == expected_sha, f"qgs: {key} file SHA mismatch")
            formal = method.get("formal_3k_result", {})
            require(method.get("technical_qualification_status") == "TECHNICALLY_QUALIFIED", "qgs: qualification missing")
            require(method.get("technical_full_matrix_eligibility") is True, "qgs: technical matrix eligibility missing")
            if formal.get("status") == "NOT_ATTEMPTED":
                require(method.get("lifecycle_role") == "ACTIVE_3K_CANDIDATE", "qgs: lifecycle role mismatch")
            elif formal.get("status") == "COMPLETE_RANKED":
                require(
                    method.get("lifecycle_role") == "ACTIVE_3K_COMPLETE_RANKED",
                    "qgs: completed lifecycle role missing",
                )
                require(method.get("three_k_training_allowed") is False, "qgs: completed method remains launchable")
                require(formal.get("rerun_allowed") is False, "qgs: formal rerun was enabled")
            elif formal.get("status") == "INCOMPLETE_UNRANKED":
                require(
                    method.get("lifecycle_role") == "ACTIVE_3K_INCOMPLETE_UNRANKED",
                    "qgs: incomplete lifecycle role missing",
                )
                require(method.get("three_k_training_allowed") is False, "qgs: terminal method remains launchable")
                require(formal.get("rerun_allowed") is False, "qgs: formal rerun was enabled")
            else:
                require(False, "qgs: unknown formal result status")
        else:
            formal = method.get("formal_3k_result", {})
            if formal.get("status") == "NOT_ATTEMPTED":
                require(method.get("lifecycle_role") == "ACTIVE_3K_CANDIDATE", f"{method_id}: lifecycle role mismatch")
                qualified = method.get("technical_qualification_status") == "TECHNICALLY_QUALIFIED"
                require(
                    method.get("technical_full_matrix_eligibility") is qualified,
                    f"{method_id}: technical matrix eligibility disagrees with qualification",
                )
            elif formal.get("status") == "COMPLETE_RANKED":
                require(
                    method.get("lifecycle_role") == "ACTIVE_3K_COMPLETE_RANKED",
                    f"{method_id}: completed lifecycle role missing",
                )
                require(method.get("technical_qualification_status") == "TECHNICALLY_QUALIFIED", f"{method_id}: qualification missing")
                require(method.get("technical_full_matrix_eligibility") is True, f"{method_id}: technical matrix eligibility missing")
                require(method.get("three_k_training_allowed") is False, f"{method_id}: completed method remains launchable")
                require(formal.get("rerun_allowed") is False, f"{method_id}: formal rerun was enabled")
            elif formal.get("status") == "INCOMPLETE_UNRANKED":
                require(
                    method.get("lifecycle_role") == "ACTIVE_3K_INCOMPLETE_UNRANKED",
                    f"{method_id}: incomplete lifecycle role missing",
                )
                require(method.get("three_k_training_allowed") is False, f"{method_id}: terminal method remains launchable")
                require(formal.get("rerun_allowed") is False, f"{method_id}: formal rerun was enabled")
            else:
                require(False, f"{method_id}: unknown formal result status")

        if method.get("technical_qualification_status") == "TECHNICALLY_QUALIFIED":
            qualification_relative = method.get("qualification_report")
            qualification_sha = method.get("qualification_report_sha256")
            require(
                isinstance(qualification_relative, str) and bool(qualification_relative),
                f"{method_id}: qualification report path missing",
            )
            require(is_sha256(qualification_sha), f"{method_id}: qualification report SHA malformed")
            if isinstance(qualification_relative, str) and qualification_relative:
                qualification_path = repo_root / qualification_relative
                require(qualification_path.is_file(), f"{method_id}: qualification report missing")
                if qualification_path.is_file() and is_sha256(qualification_sha):
                    require(
                        file_sha256(qualification_path) == qualification_sha,
                        f"{method_id}: qualification report file SHA mismatch",
                    )

    for method_id in ("gsprior", "citygs_x"):
        license_status = str(by_id.get(method_id, {}).get("source", {}).get("license_status", ""))
        require("internal_test_only" in license_status and "redistribution_blocked" in license_status, f"{method_id}: missing-license boundary absent")
    require(by_id.get("citygs_x", {}).get("internal_numeric_reporting_allowed") is True, "CityGS-X internal numeric-reporting authorization missing")
    require(by_id.get("citygs_x", {}).get("redistribution_allowed") is False, "CityGS-X redistribution was enabled")

    pending_rgb_method_refs = {
        "gsprior": (
            "configs/m3m_gcp_native_quarter_gsprior_3k_recipe_v1.json",
            "e02cfb0bb4db86b915c859c61ad16f98f9c15407f68661c56cb1c2b1642645f0",
            "configs/m3m_gcp_native_quarter_gsprior_renderer_adapter_v1.json",
            "f6ceac12c6c04caa0ab56d2edbb62626ea8ab8a2777526a7c7cd14e9404046ee",
            "docs/protocol_evidence/gsprior_native_quarter_truth_deny_v1.json",
            "63f005775b37e5438bd6aad46b0aa10ffd3ca188e1ea534f13b08091a7802146",
        ),
        "sof": (
            "configs/m3m_gcp_native_quarter_sof_3k_recipe_v1.json",
            "0b9042e5c456f3e6885f27c8f0353de6c1d3e0d752cc77ed6f87f4bee7a8390e",
            "configs/m3m_gcp_native_quarter_sof_renderer_adapter_v1.json",
            "014f19cec9a8c2d44b619565a9abac70a5e97950469eec47b87b26aa96978371",
            "docs/protocol_evidence/sof_native_quarter_truth_deny_v1.json",
            "517571e84ca1fa95c4b1d51d3da4dc0b1db3fcb8e91b8a82b247d5cc63a213cf",
        ),
    }
    for method_id, refs in pending_rgb_method_refs.items():
        method = by_id.get(method_id, {})
        require(method.get("input_class") == "rgb_colmap_only", f"{method_id}: input stratum mismatch")
        for key, relative, expected_sha in (
            ("recipe", refs[0], refs[1]),
            ("adapter_config", refs[2], refs[3]),
            ("truth_deny_report", refs[4], refs[5]),
        ):
            require(method.get(key) == relative, f"{method_id}: {key} path mismatch")
            require(method.get(f"{key}_sha256") == expected_sha, f"{method_id}: {key} SHA mismatch")
            path = repo_root / relative
            require(path.is_file(), f"{method_id}: {key} file missing")
            if path.is_file():
                require(file_sha256(path) == expected_sha, f"{method_id}: {key} file SHA mismatch")

    sof_recipe_path = repo_root / pending_rgb_method_refs["sof"][0]
    sof_conformance_path = repo_root / "code/gcp/test_sof_raw_moments_cuda.py"
    if sof_recipe_path.is_file() and sof_conformance_path.is_file():
        sof_recipe = json.loads(sof_recipe_path.read_text(encoding="utf-8"))
        qualification = sof_recipe.get("qualification_inputs", {})
        require(
            qualification.get("cuda_conformance_script")
            == "code/gcp/test_sof_raw_moments_cuda.py",
            "sof: CUDA conformance script path mismatch",
        )
        require(
            qualification.get("cuda_conformance_script_sha256")
            == file_sha256(sof_conformance_path),
            "sof: CUDA conformance script SHA mismatch",
        )

    require(
        by_id.get("citygaussian_v2", {}).get("input_class") == "rgb_colmap_external_geometry_prior",
        "CityGaussianV2 input stratum mismatch",
    )
    require(by_id.get("citygs_x", {}).get("input_class") == "rgb_colmap_external_geometry_prior", "CityGS-X input stratum mismatch")
    require(by_id.get("metrogs", {}).get("input_class") == "rgb_colmap_external_geometry_prior", "MetroGS input stratum mismatch")
    city_recipe_refs = {
        "citygaussian_v2": (
            "configs/m3m_gcp_native_quarter_citygaussian_v2_3k_recipe_v1.json",
            "2ee37a3e85de03a3fe436b1b842c8caef439e8bedd2835be1167fc46bce6c188",
            "configs/m3m_gcp_native_quarter_citygaussian_v2_renderer_adapter_v1.json",
            "0d7946ee84f4c7f990d0f97e563973002f84a75cd06c8cb27e7b8de3d8bca4ab",
        ),
        "citygs_x": (
            "configs/m3m_gcp_native_quarter_citygs_x_3k_recipe_v1.json",
            "fb6e311a6beef9316b289a0eb8c807b617b1e8384826f71e16f57a522031bdc4",
            "configs/m3m_gcp_native_quarter_citygs_x_renderer_adapter_v1.json",
            "65c3a18f6f88379b2a2add0775ac1e4d00b4c67e56d2347c4a3a47c088b04d43",
        ),
    }
    for method_id, (recipe_relative, recipe_sha, adapter_relative, adapter_sha) in city_recipe_refs.items():
        method = by_id.get(method_id, {})
        require(method.get("recipe") == recipe_relative, f"{method_id}: recipe path mismatch")
        require(method.get("recipe_sha256") == recipe_sha, f"{method_id}: recipe SHA mismatch")
        require(method.get("adapter_config") == adapter_relative, f"{method_id}: adapter path mismatch")
        require(method.get("adapter_config_sha256") == adapter_sha, f"{method_id}: adapter SHA mismatch")
        recipe_path = repo_root / recipe_relative
        adapter_path = repo_root / adapter_relative
        require(recipe_path.is_file(), f"{method_id}: recipe file missing")
        require(adapter_path.is_file(), f"{method_id}: adapter file missing")
        if recipe_path.is_file():
            require(file_sha256(recipe_path) == recipe_sha, f"{method_id}: recipe file SHA mismatch")
            recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
            require(recipe.get("protocol_id") == "m3m_gcp_native_quarter_geometry_v2", f"{method_id}: recipe protocol mismatch")
            require(recipe.get("status") == "FROZEN_PRE_RESULT_QUALIFICATION_PENDING", f"{method_id}: recipe pre-result status mismatch")
            require(recipe.get("method", {}).get("input_class") == "rgb_colmap_external_geometry_prior", f"{method_id}: recipe input stratum mismatch")
            prior = recipe.get("external_geometry_prior", {})
            require(
                prior.get("repository_commit") == "a561b849ebae10a6f5ef49e26c83cbbcd36c71bf",
                f"{method_id}: Depth Anything V2 commit mismatch",
            )
            require(
                prior.get("weight_sha256") == "a7ea19fa0ed99244e67b624c72b8580b7e9553043245905be58796a608eb9345",
                f"{method_id}: Depth Anything V2 weight mismatch",
            )
            expected_preparation_entrypoint = {
                "citygaussian_v2": "code/gcp/prepare_citygaussian_v2_depth_prior.py",
                "citygs_x": "code/gcp/prepare_citygs_x_depth_prior.py",
            }[method_id]
            preparation_entrypoint = repo_root / expected_preparation_entrypoint
            require(
                prior.get("preparation_entrypoint") == expected_preparation_entrypoint,
                f"{method_id}: preparation entrypoint path mismatch",
            )
            require(
                preparation_entrypoint.is_file(),
                f"{method_id}: preparation entrypoint missing",
            )
            if preparation_entrypoint.is_file():
                require(
                    prior.get("preparation_entrypoint_sha256")
                    == file_sha256(preparation_entrypoint),
                    f"{method_id}: preparation entrypoint SHA mismatch",
                )
            if method_id == "citygs_x":
                expected_neighbor_route = {
                    "multi_view_num": 8,
                    "multi_view_max_angle_deg": 15.0,
                    "multi_view_min_dis": 0.01,
                    "multi_view_max_dis": 25.0,
                }
                external_geometry_prior = recipe.get("external_geometry_prior", {})
                training_route = recipe.get("training", {})
                for key, expected in expected_neighbor_route.items():
                    require(
                        external_geometry_prior.get(key) == expected,
                        f"citygs_x: external-prior neighbor route mismatch for {key}",
                    )
                    require(
                        training_route.get(key) == expected,
                        f"citygs_x: training neighbor route mismatch for {key}",
                    )
                require(
                    "upstream MatrixCity aerial" in str(
                        external_geometry_prior.get("neighbor_selection_basis", "")
                    ),
                    "citygs_x: neighbor selection basis is not source-bound",
                )
                compatibility = recipe.get("compatibility", {})
                compatibility_paths = {
                    "camera_utils_patch": "camera_utils_patch_sha256",
                    "dataset_readers_patch": "dataset_readers_patch_sha256",
                }
                for path_key, sha_key in compatibility_paths.items():
                    relative = compatibility.get(path_key)
                    require(
                        isinstance(relative, str) and bool(relative),
                        f"citygs_x: {path_key} path missing",
                    )
                    if isinstance(relative, str) and relative:
                        patch_path = repo_root / relative
                        require(patch_path.is_file(), f"citygs_x: {path_key} missing")
                        if patch_path.is_file():
                            require(
                                compatibility.get(sha_key) == file_sha256(patch_path),
                                f"citygs_x: {path_key} SHA mismatch",
                            )
                require(
                    compatibility.get("patched_camera_utils_sha256")
                    == "9326e6571685177543e34c903823b207b75258e96489d9398b08672637f5c9e3",
                    "citygs_x: patched camera_utils identity mismatch",
                )
                require(
                    compatibility.get("patched_dataset_readers_sha256")
                    == "3d75bbeb16d47f7c078ba2d09a8612dbe4eb6139a865b1d15e1302aa8167a82c",
                    "citygs_x: patched dataset_readers identity mismatch",
                )
            expected_execution_entrypoint = {
                "citygaussian_v2": "code/gcp/run_citygaussian_v2_pipeline.py",
                "citygs_x": "code/gcp/run_citygs_x_training.py",
            }[method_id]
            execution_entrypoint = repo_root / expected_execution_entrypoint
            execution = recipe.get("execution", {})
            require(
                execution.get("entrypoint") == expected_execution_entrypoint,
                f"{method_id}: execution entrypoint path mismatch",
            )
            require(
                execution_entrypoint.is_file(),
                f"{method_id}: execution entrypoint missing",
            )
            if execution_entrypoint.is_file():
                require(
                    execution.get("entrypoint_sha256")
                    == file_sha256(execution_entrypoint),
                    f"{method_id}: execution entrypoint SHA mismatch",
                )
            if method_id == "citygaussian_v2":
                torch_load_compatibility = execution.get("torch_load_compatibility", {})
                require(
                    torch_load_compatibility.get("environment_variable")
                    == "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
                    and torch_load_compatibility.get("value") == "1",
                    "citygaussian_v2: trusted-checkpoint torch.load compatibility is missing",
                )
                scheduler_compatibility = execution.get("single_block_scheduler_compatibility", {})
                require(
                    scheduler_compatibility.get("block_id") == 0
                    and scheduler_compatibility.get("block_dim") == [1, 1]
                    and scheduler_compatibility.get("algorithm_or_hyperparameter_change") is False,
                    "citygaussian_v2: one-block observable scheduler compatibility is missing",
                )
                evaluation_adapter = recipe.get("evaluation_adapter", {})
                exporter_relative = "code/gcp/export_citygaussian_v2_depth_maps.py"
                exporter_path = repo_root / exporter_relative
                require(
                    evaluation_adapter.get("exporter") == exporter_relative,
                    "citygaussian_v2: exporter path mismatch",
                )
                require(exporter_path.is_file(), "citygaussian_v2: exporter missing")
                if exporter_path.is_file():
                    require(
                        evaluation_adapter.get("exporter_sha256") == file_sha256(exporter_path),
                        "citygaussian_v2: exporter SHA mismatch",
                    )
        if adapter_path.is_file():
            require(file_sha256(adapter_path) == adapter_sha, f"{method_id}: adapter file SHA mismatch")
        priors = method.get("external_priors", [])
        require(len(priors) == 1, f"{method_id}: external-prior inventory mismatch")
        if len(priors) == 1:
            prior = priors[0]
            require(prior.get("repository_commit") == "a561b849ebae10a6f5ef49e26c83cbbcd36c71bf", f"{method_id}: registered prior commit mismatch")
            require(prior.get("weight_sha256") == "a7ea19fa0ed99244e67b624c72b8580b7e9553043245905be58796a608eb9345", f"{method_id}: registered prior weight mismatch")
            require(prior.get("freeze_status") == "FROZEN_EXACT_WEIGHTS_AND_COMMAND_PRE_RESULT", f"{method_id}: prior route not frozen")
    metro_route = by_id.get("metrogs", {}).get("selected_external_prior_route", {})
    require(str(metro_route.get("route", "")).startswith("Pi3-Align with Xname=Pi3"), "MetroGS Pi3 route not frozen")
    require(metro_route.get("best_of_routes_allowed") is False, "MetroGS best-of-routes enabled")
    require(
        metro_route.get("freeze_status") == "FROZEN_EXACT_WEIGHTS_AND_COMMAND_PRE_RESULT",
        "MetroGS exact prior route is not frozen",
    )
    require(
        metro_route.get("moge_weight_sha256")
        == "280741fd09bc3f403ccff9967784c2a391b52d2c0742ae3efdb21d9f90cc1a01",
        "MetroGS MoGe weight identity mismatch",
    )
    require(
        metro_route.get("pi3_weight_sha256")
        == "33580e4702ac671558aedeab1148fd08118f7ce45bdbeb99f3e3cf340062875d",
        "MetroGS Pi3 weight identity mismatch",
    )
    metro_method = by_id.get("metrogs", {})
    metro_recipe_relative = "configs/m3m_gcp_native_quarter_metrogs_3k_recipe_v1.json"
    metro_adapter_relative = "configs/m3m_gcp_native_quarter_metrogs_renderer_adapter_v1.json"
    require(metro_method.get("recipe") == metro_recipe_relative, "MetroGS recipe path mismatch")
    require(
        metro_method.get("recipe_sha256")
        == "78787ef3983473766c8ad831a22478edb95f2d59701cad7bdb58b7833c2eb112",
        "MetroGS recipe recorded SHA mismatch",
    )
    require(metro_method.get("renderer_adapter") == metro_adapter_relative, "MetroGS adapter path mismatch")
    require(
        metro_method.get("renderer_adapter_sha256")
        == "20aa83dc12ab0f3087b57c46597845cf2750beacc3b36ba119dd49a6bed78b82",
        "MetroGS adapter recorded SHA mismatch",
    )
    metro_recipe_path = repo_root / metro_recipe_relative
    metro_adapter_path = repo_root / metro_adapter_relative
    require(metro_recipe_path.is_file(), "MetroGS recipe file missing")
    require(metro_adapter_path.is_file(), "MetroGS adapter file missing")
    if metro_recipe_path.is_file() and metro_adapter_path.is_file():
        require(
            file_sha256(metro_recipe_path) == metro_method.get("recipe_sha256"),
            "MetroGS recipe file SHA mismatch",
        )
        require(
            file_sha256(metro_adapter_path) == metro_method.get("renderer_adapter_sha256"),
            "MetroGS adapter file SHA mismatch",
        )
        metro_recipe = json.loads(metro_recipe_path.read_text(encoding="utf-8"))
        metro_adapter = json.loads(metro_adapter_path.read_text(encoding="utf-8"))
        require(
            metro_recipe.get("status") == "FROZEN_PRE_RESULT_QUALIFICATION_PENDING",
            "MetroGS recipe pre-result status mismatch",
        )
        require(
            metro_adapter.get("status") == "FROZEN_PRE_RESULT_CUDA_CONFORMANCE_PENDING",
            "MetroGS adapter pre-result status mismatch",
        )
        require(
            metro_recipe.get("evaluation_adapter", {}).get("config_sha256")
            == file_sha256(metro_adapter_path),
            "MetroGS recipe records the wrong adapter SHA",
        )
        moge_route = metro_recipe.get("external_prior_route", {}).get("moge", {})
        require(
            "all 82 RGB views remain in training"
            in str(moge_route.get("official_scale_filter_semantics", "")),
            "MetroGS official depth-prior filter semantics are not frozen",
        )
        require(
            moge_route.get("qualification_observed_attachment")
            == "72 depth priors attached and 10 skipped by the frozen upstream filter; no RGB training view removed",
            "MetroGS qualification depth-prior attachment accounting mismatch",
        )
        metro_preparation_path = repo_root / "code/gcp/prepare_metrogs_training_priors.py"
        metro_wrapper_path = repo_root / "code/gcp/run_metrogs_training.py"
        require(metro_preparation_path.is_file(), "MetroGS preparation script missing")
        require(metro_wrapper_path.is_file(), "MetroGS training wrapper missing")
        if metro_preparation_path.is_file():
            require(
                metro_recipe.get("external_prior_route", {}).get("preparation_script")
                == "code/gcp/prepare_metrogs_training_priors.py",
                "MetroGS preparation script path mismatch",
            )
            require(
                metro_recipe.get("external_prior_route", {}).get("preparation_script_sha256")
                == file_sha256(metro_preparation_path),
                "MetroGS preparation script SHA mismatch",
            )
        if metro_wrapper_path.is_file():
            require(
                metro_recipe.get("execution", {}).get("wrapper")
                == "code/gcp/run_metrogs_training.py",
                "MetroGS training wrapper path mismatch",
            )
            require(
                metro_recipe.get("execution", {}).get("wrapper_sha256")
                == file_sha256(metro_wrapper_path),
                "MetroGS training wrapper SHA mismatch",
            )

    for method_id in ("3dgs_original", "2dgs"):
        method = by_id.get(method_id, {})
        require(method.get("lifecycle_role") == "REUSED_COMPLETE_RANKED", f"{method_id}: reused role mismatch")
        require(method.get("technical_qualification_status") == "TECHNICALLY_QUALIFIED", f"{method_id}: technical qualification missing")
        require(method.get("formal_3k_result", {}).get("status") == "COMPLETE_RANKED", f"{method_id}: formal result not complete-ranked")
        require(method.get("technical_full_matrix_eligibility") is True, f"{method_id}: technical matrix eligibility missing")
        require(method.get("six_scene_run_allowed") is False, f"{method_id}: six-scene run prematurely authorized")
        require(method.get("three_k_training_allowed") is False, f"{method_id}: completed 3K result remains launchable")

    gof = by_id.get("gof", {})
    require(gof.get("lifecycle_role") == "HISTORICAL_COMPLETE_RETIRED", "GOF is not historical retired")
    require(gof.get("formal_3k_result", {}).get("status") == "HISTORICAL_COMPLETE_RETIRED", "GOF result status mismatch")
    require(gof.get("formal_3k_result", {}).get("original_status") == "COMPLETE_RANKED", "GOF original result provenance missing")
    require(gof.get("three_k_training_allowed") is False and gof.get("six_scene_run_allowed") is False, "retired GOF is launchable")

    for method_id, (report_name, expected_sha) in EXPECTED_FORMAL_REPORTS.items():
        method_report = by_id.get(method_id, {}).get("formal_3k_result", {})
        require(method_report.get("report") == report_name, f"{method_id}: formal report path mismatch")
        require(method_report.get("report_sha256") == expected_sha, f"{method_id}: formal report recorded SHA mismatch")
        report_path = repo_root / report_name
        require(report_path.is_file(), f"{method_id}: formal report missing")
        if report_path.is_file():
            require(file_sha256(report_path) == expected_sha, f"{method_id}: formal report file SHA mismatch")

    for method_id in EXPECTED_CANDIDATES:
        if method_id in EXPECTED_FORMAL_REPORTS:
            continue
        method_report = by_id.get(method_id, {}).get("formal_3k_result", {})
        if method_report.get("status") == "NOT_ATTEMPTED":
            continue
        report_name = method_report.get("report")
        report_sha = method_report.get("report_sha256")
        require(isinstance(report_name, str) and bool(report_name), f"{method_id}: formal report path missing")
        require(is_sha256(report_sha), f"{method_id}: formal report SHA malformed")
        if isinstance(report_name, str) and report_name:
            report_path = repo_root / report_name
            require(report_path.is_file(), f"{method_id}: formal report missing")
            if report_path.is_file() and is_sha256(report_sha):
                require(file_sha256(report_path) == report_sha, f"{method_id}: formal report file SHA mismatch")

    citygaussian = by_id.get("citygaussian_v2", {})
    citygaussian_formal = citygaussian.get("formal_3k_result", {})
    if citygaussian_formal.get("status") == "COMPLETE_RANKED":
        correction_relative = "docs/protocol_evidence/citygaussian_v2_formal_metadata_correction_v1.json"
        correction_sha = "191e4290a9f7b9a69829fa08dba6b1ecdc1507293d7d1f26ac559710807505a7"
        require(
            citygaussian_formal.get("primary_accuracy_scope") == "checkpoint",
            "citygaussian_v2: primary accuracy scope is not checkpoint-only",
        )
        require(
            citygaussian_formal.get("metadata_correction") == correction_relative,
            "citygaussian_v2: metadata correction path mismatch",
        )
        require(
            citygaussian_formal.get("metadata_correction_sha256") == correction_sha,
            "citygaussian_v2: metadata correction recorded SHA mismatch",
        )
        citygaussian_priors = citygaussian.get("external_priors", [])
        if len(citygaussian_priors) == 1:
            registered_prior = citygaussian_priors[0]
            require(
                "official CityGaussianV2 utils/run_depth_anything_v2.py wrapper"
                in str(registered_prior.get("actual_execution_mode", "")),
                "citygaussian_v2: actual DAv2 wrapper execution is not registered",
            )
            require(
                "official_command_mode" not in registered_prior,
                "citygaussian_v2: superseded DAv2 CLI description remains active",
            )
        correction_path = repo_root / correction_relative
        require(correction_path.is_file(), "citygaussian_v2: metadata correction file missing")
        if correction_path.is_file():
            require(
                file_sha256(correction_path) == correction_sha,
                "citygaussian_v2: metadata correction file SHA mismatch",
            )
            correction = json.loads(correction_path.read_text(encoding="utf-8"))
            require(
                correction.get("status") == "PASS_METADATA_CORRECTION_NO_RESULT_CHANGE",
                "citygaussian_v2: metadata correction status mismatch",
            )
            require(
                correction.get("original_formal_evidence", {}).get("sha256")
                == citygaussian_formal.get("report_sha256"),
                "citygaussian_v2: metadata correction is not bound to the formal evidence",
            )
            require(
                correction.get("ranking_scope_correction", {}).get("primary_accuracy_scope")
                == "checkpoint",
                "citygaussian_v2: correction does not freeze checkpoint-primary ranking",
            )
            require(
                correction.get("checkpoint_step_semantics", {})
                .get("merged_checkpoint", {})
                .get("internal_global_step")
                == 60000,
                "citygaussian_v2: merged checkpoint global-step correction mismatch",
            )

    require(value.get("global_training_allowed") is False, "global training lock missing")
    allowlist = value.get("per_method_training_allowed_methods")
    require(isinstance(allowlist, list) and len(allowlist) <= 1, "method allowlist must contain at most one method")
    allowlist = allowlist if isinstance(allowlist, list) else []
    training_flags = sorted(
        method_id for method_id, method in by_id.items() if method.get("three_k_training_allowed") is True
    )
    require(training_flags == sorted(allowlist), "method training flags disagree with the allowlist")
    gate_ref = value.get("current_one_use_launch_gate")
    if not allowlist:
        require(gate_ref is None, "one-use gate must be absent while the allowlist is empty")
        require(value.get("status") == "EIGHT_METHOD_3K_BATCH_ACTIVE", "no-gate registry must use the active status")
    else:
        method_id = str(allowlist[0])
        require(method_id in EXPECTED_CANDIDATES, "one-use gate targets a non-candidate method")
        gated = by_id.get(method_id, {})
        require(gated.get("lifecycle_role") == "ACTIVE_3K_CANDIDATE", "gated method is not an active 3K candidate")
        require(gated.get("technical_qualification_status") == "TECHNICALLY_QUALIFIED", "gated method is not technically qualified")
        require(gated.get("formal_3k_result", {}).get("status") == "NOT_ATTEMPTED", "gated method already has a formal result")
        require(isinstance(gate_ref, dict), "current one-use launch gate is absent")
        if isinstance(gate_ref, dict):
            require(gate_ref.get("method_id") == method_id, "one-use gate method mismatch")
            require(is_sha256(gate_ref.get("sha256")), "one-use gate recorded SHA malformed")
            gate_path = repo_root / str(gate_ref.get("path", ""))
            require(gate_path.is_file(), "one-use gate file missing")
            if gate_path.is_file():
                require(file_sha256(gate_path) == gate_ref.get("sha256"), "one-use gate file SHA mismatch")
    controller = value.get("batch_controller", {})
    require(controller.get("formal_training_requires_method_gate") is True, "method launch-gate requirement missing")
    require(controller.get("maximum_concurrent_formal_trainings") == 1, "formal concurrency is not one")
    require(controller.get("gate_consumed_after_first_launch_attempt") is True, "one-use gate consumption missing")
    require(controller.get("resume_allowed") is False, "resume was enabled")
    require(controller.get("result_driven_retry_allowed") is False, "result-driven retry was enabled")

    return {
        "schema": "m3m_gcp_native_quarter_method_registry_validation_v3",
        "status": "PASS" if not errors else "FAIL",
        "passed": not errors,
        "protocol_id": value.get("protocol_id"),
        "batch_id": value.get("batch_id"),
        "method_count": len(methods) if isinstance(methods, list) else 0,
        "active_method_count": len(value.get("active_benchmark_method_ids", [])),
        "candidate_method_count": len(scope.get("active_candidate_method_ids", [])),
        "training_allowed_methods": sorted(
            method_id for method_id, method in by_id.items() if method.get("three_k_training_allowed") is True
        ),
        "errors": errors,
    }


def main() -> int:
    default_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=default_root)
    parser.add_argument(
        "--registry",
        type=Path,
        default=default_root / "configs" / "m3m_gcp_native_quarter_method_registry_v3.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate_registry(args.repo_root.resolve(), args.registry.resolve())
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
