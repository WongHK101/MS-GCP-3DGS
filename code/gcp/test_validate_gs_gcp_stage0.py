#!/usr/bin/env python3

from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from validate_gs_gcp_stage0 import sha256_file, validate_stage0


ROOT = Path(__file__).resolve().parents[2]


class Stage0ReadinessTests(unittest.TestCase):
    def test_external_review_pass_is_recorded_but_runtime_evidence_is_still_required(self) -> None:
        result = validate_stage0(ROOT, None, "3dgs_original")
        self.assertTrue(result["contracts_valid"])
        self.assertFalse(result["training_ready"])
        self.assertEqual(result["components"]["release_review"]["external_review_status"], "PASS")
        self.assertTrue(result["components"]["release_review"]["training_authorized"])
        self.assertNotIn("v1.3.0_external_release_review_pass_not_recorded", result["blockers"])
        self.assertIn("v1.3.0_release_integrity_not_verified_at_runtime", result["blockers"])
        self.assertIn(
            "autodl_901_orchestrator_deployment_evidence_missing_or_mismatch",
            result["blockers"],
        )

    def test_external_review_evidence_hash_is_bound(self) -> None:
        review = json.loads(
            (ROOT / "configs/gs_gcp_v13_release_review_status_v1.json").read_text(
                encoding="utf-8"
            )
        )
        evidence = ROOT / review["external_review_evidence_path"]
        self.assertTrue(evidence.is_file())
        self.assertEqual(sha256_file(evidence), review["external_review_evidence_sha256"])

    def test_external_review_evidence_hash_mismatch_blocks_contract(self) -> None:
        actual_sha256_file = sha256_file

        def tampered_evidence_hash(path: Path) -> str:
            if path.name == "GS_GCP_V13_STAGE0_EXTERNAL_REVIEW_PASS_20260718.md":
                return "0" * 64
            return actual_sha256_file(path)

        with patch(
            "validate_gs_gcp_stage0.sha256_file",
            side_effect=tampered_evidence_hash,
        ):
            result = validate_stage0(ROOT, None, "3dgs_original")
        self.assertFalse(result["components"]["release_review"]["passed"])
        self.assertFalse(result["contracts_valid"])

    def test_unregistered_method_is_blocked(self) -> None:
        result = validate_stage0(ROOT, None, "unknown_method")
        self.assertFalse(result["training_ready"])
        self.assertIn("method_not_pre_registered_for_3k_qualification:unknown_method", result["blockers"])

    def test_pending_recipe_method_is_blocked(self) -> None:
        result = validate_stage0(ROOT, None, "2dgs")
        self.assertFalse(result["training_ready"])
        self.assertIn("method_not_pre_registered_for_3k_qualification:2dgs", result["blockers"])

    def test_original_3dgs_is_approved_for_full_scene_matrix(self) -> None:
        result = validate_stage0(
            ROOT,
            None,
            "3dgs_original",
            require_full_scene_matrix_eligible=True,
        )
        admission = result["components"]["method_qualification"]
        self.assertTrue(admission["passed"])
        self.assertTrue(admission["full_scene_matrix_eligible"])
        self.assertEqual(admission["three_k_qualification_status"], "PASS")
        self.assertEqual(admission["external_review_status"], "PASS")

    def test_other_method_is_not_approved_for_full_scene_matrix(self) -> None:
        result = validate_stage0(
            ROOT,
            None,
            "2dgs",
            require_full_scene_matrix_eligible=True,
        )
        self.assertFalse(result["components"]["method_qualification"]["passed"])
        self.assertIn("method_not_approved_for_full_scene_matrix:2dgs", result["blockers"])

    def test_execution_and_archive_server_roles_are_distinct(self) -> None:
        result = validate_stage0(ROOT, None, "3dgs_original")
        self.assertEqual(result["components"]["execution_runtime"]["server"], "AutoDL-901")
        self.assertEqual(
            result["components"]["execution_runtime"]["server_role"],
            "experiment_execution",
        )
        self.assertEqual(result["components"]["archive_mirror"]["source_server"], "AutoDL-901")
        self.assertEqual(result["components"]["archive_mirror"]["target_server"], "AutoDL-740")

    def test_repository_rename_and_code_boundary_are_frozen(self) -> None:
        result = validate_stage0(ROOT, None, "3dgs_original")
        promotion = json.loads(
            (ROOT / "configs/gs_gcp_repository_promotion_status_v1.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertTrue(result["components"]["repository_promotion"]["passed"])
        self.assertTrue(result["publication_repository_ready"])
        self.assertEqual(
            promotion["current_github_remote"],
            "https://github.com/WongHK101/GS-GCP-Benchmark.git",
        )
        self.assertFalse(promotion["umgs_training_code_included"])
        self.assertFalse(promotion["gaussian_method_training_code_included"])
        self.assertFalse(promotion["publication_blocking"])

    def test_runtime_config_does_not_authorize_training_without_evidence(self) -> None:
        runtime = json.loads(
            (ROOT / "configs/gs_gcp_autodl901_runtime_status_v1.json").read_text(encoding="utf-8")
        )
        self.assertFalse(runtime["training_ready"])
        self.assertEqual(runtime["orchestrator_deployment_status"], "runtime_evidence_required")


if __name__ == "__main__":
    unittest.main()
