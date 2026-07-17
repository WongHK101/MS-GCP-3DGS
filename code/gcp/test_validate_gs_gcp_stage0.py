#!/usr/bin/env python3

from __future__ import annotations

import json
import unittest
from pathlib import Path

from validate_gs_gcp_stage0 import validate_stage0


ROOT = Path(__file__).resolve().parents[2]


class Stage0ReadinessTests(unittest.TestCase):
    def test_contracts_valid_but_training_blocked_until_external_review(self) -> None:
        result = validate_stage0(ROOT, None, "3dgs_original")
        self.assertTrue(result["contracts_valid"])
        self.assertFalse(result["training_ready"])
        self.assertIn("v1.3.0_external_release_review_pass_not_recorded", result["blockers"])

    def test_unregistered_method_is_blocked(self) -> None:
        result = validate_stage0(ROOT, None, "unknown_method")
        self.assertFalse(result["training_ready"])
        self.assertIn("method_not_pre_registered_for_3k_qualification:unknown_method", result["blockers"])

    def test_pending_recipe_method_is_blocked(self) -> None:
        result = validate_stage0(ROOT, None, "2dgs")
        self.assertFalse(result["training_ready"])
        self.assertIn("method_not_pre_registered_for_3k_qualification:2dgs", result["blockers"])

    def test_execution_and_archive_server_roles_are_distinct(self) -> None:
        result = validate_stage0(ROOT, None, "3dgs_original")
        self.assertEqual(result["components"]["execution_runtime"]["server"], "AutoDL-901")
        self.assertEqual(
            result["components"]["execution_runtime"]["server_role"],
            "experiment_execution",
        )
        self.assertEqual(result["components"]["archive_mirror"]["source_server"], "AutoDL-901")
        self.assertEqual(result["components"]["archive_mirror"]["target_server"], "AutoDL-740")

    def test_runtime_config_does_not_authorize_training_without_evidence(self) -> None:
        runtime = json.loads(
            (ROOT / "configs/gs_gcp_autodl901_runtime_status_v1.json").read_text(encoding="utf-8")
        )
        self.assertFalse(runtime["training_ready"])
        self.assertEqual(runtime["orchestrator_deployment_status"], "runtime_evidence_required")


if __name__ == "__main__":
    unittest.main()
