#!/usr/bin/env python3
"""Unit tests for the LiDAR formal-v1 fail-closed contract validator."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from validate_m3m_gcp_lidar_formal_protocol import validate_contract  # noqa: E402


class LidarFormalContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = json.loads((ROOT / "configs" / "m3m_gcp_lidar_formal_v1.json").read_text(encoding="utf-8"))
        cls.split = json.loads((ROOT / "configs" / "gs_gcp_rgb_holdout_split_manifest_v1.json").read_text(encoding="utf-8"))

    def test_candidate_contract_passes_static_validation(self) -> None:
        self.assertEqual(validate_contract(self.contract, self.split), [])

    def test_rejects_method_specific_icp(self) -> None:
        mutated = copy.deepcopy(self.contract)
        mutated["registration"]["method_specific_icp"] = "ALLOWED"
        self.assertTrue(any("method_specific_icp" in item for item in validate_contract(mutated, self.split)))

    def test_rejects_heldout_rgb_or_partial_view_sampling(self) -> None:
        mutated = copy.deepcopy(self.contract)
        mutated["reconstruction_surface"]["heldout_rgb_read"] = True
        mutated["reconstruction_surface"]["all_train_views_required"] = False
        errors = validate_contract(mutated, self.split)
        self.assertTrue(any("heldout RGB" in item for item in errors))
        self.assertTrue(any("view allowlist" in item for item in errors))

    def test_rejects_failure_score_imputation(self) -> None:
        mutated = copy.deepcopy(self.contract)
        mutated["ranking"]["failed_or_oom_scene_metric_imputation"] = "ZERO"
        self.assertTrue(any("imputation" in item for item in validate_contract(mutated, self.split)))


if __name__ == "__main__":
    unittest.main()
