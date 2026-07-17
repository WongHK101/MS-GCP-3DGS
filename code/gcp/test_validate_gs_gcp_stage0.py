#!/usr/bin/env python3

from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
