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

    def test_rejects_vertical_bridge_or_lidar_filter_changes(self) -> None:
        mutated = copy.deepcopy(self.contract)
        mutated["lidar_source"]["normal_minus_ellipsoid_m"] = 0.0
        mutated["reference_surface"]["las_class_filter"] = "CLASS_2_ONLY"
        errors = validate_contract(mutated, self.split)
        self.assertTrue(any("vertical datum bridge" in item for item in errors))
        self.assertTrue(any("class filter" in item for item in errors))

    def test_rejects_method_dependent_roi_or_view_set(self) -> None:
        mutated = copy.deepcopy(self.contract)
        mutated["reference_surface"]["roi_is_identical_across_methods"] = False
        mutated["reconstruction_surface"]["common_view_set_required_across_methods"] = False
        errors = validate_contract(mutated, self.split)
        self.assertTrue(any("method-dependent" in item for item in errors))
        self.assertTrue(any("common method view" in item for item in errors))

    def test_rejects_ranking_direction_tolerance_or_all_tied_changes(self) -> None:
        mutated = copy.deepcopy(self.contract)
        mutated["ranking"]["scene_primary"] = "fscore_10cm_ascending"
        mutated["ranking"]["tie_numeric_tolerance"] = 1.0
        mutated["ranking"]["all_keys_tied"] = "method_id_breaks_tie"
        errors = validate_contract(mutated, self.split)
        self.assertTrue(any("primary ranking" in item for item in errors))
        self.assertTrue(any("tie tolerance" in item for item in errors))
        self.assertTrue(any("all-keys-tied" in item for item in errors))

    def test_rejects_quality_early_stop(self) -> None:
        mutated = copy.deepcopy(self.contract)
        mutated["failure_policy"]["quality_threshold_early_stop"] = True
        mutated["launch_policy"]["quality_threshold_early_stop"] = True
        errors = validate_contract(mutated, self.split)
        self.assertGreaterEqual(sum("quality-threshold" in item for item in errors), 2)

    def test_rejects_geometry_registry_or_input_class_identity_changes(self) -> None:
        mutated = copy.deepcopy(self.contract)
        mutated["source_geometry_binding"]["release_manifest_sha256"] = "0" * 64
        mutated["source_geometry_binding"]["scene_common_sim3_sha256"]["gcp_3000_20260602"] = "0" * 64
        mutated["method_registry_binding"]["file_sha256"] = "0" * 64
        mutated["method_registry_binding"]["active_method_input_classes"]["metrogs"] = "rgb_colmap_only"
        errors = validate_contract(mutated, self.split)
        self.assertTrue(any("release-manifest" in item for item in errors))
        self.assertTrue(any("Sim3" in item for item in errors))
        self.assertTrue(any("method-registry SHA" in item for item in errors))
        self.assertTrue(any("input-class mapping" in item for item in errors))

    def test_rejects_implementation_or_launch_gate_changes(self) -> None:
        mutated = copy.deepcopy(self.contract)
        mutated["implementation"]["evaluator_sha256"] = "0" * 64
        mutated["launch_policy"]["formal_output_root_must_not_exist"] = False
        mutated["launch_policy"]["exact_clean_benchmark_commit_and_tree_required"] = False
        errors = validate_contract(mutated, self.split)
        self.assertTrue(any("evaluator SHA" in item for item in errors))
        self.assertTrue(any("no-overwrite" in item for item in errors))
        self.assertTrue(any("exact clean commit" in item for item in errors))


if __name__ == "__main__":
    unittest.main()
