#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from validate_gs_gcp_original_3dgs_full_matrix import validate_plan


ROOT = Path(__file__).resolve().parents[2]
PLAN = json.loads((ROOT / "configs/gs_gcp_v13_original_3dgs_full_matrix_v2.json").read_text(encoding="utf-8"))
REGISTRY = json.loads((ROOT / "configs/gs_gcp_method_registry_v1.json").read_text(encoding="utf-8"))


class Original3DGSFullMatrixPlanTests(unittest.TestCase):
    def test_locked_plan_passes_validation_but_cannot_launch(self) -> None:
        result = validate_plan(copy.deepcopy(PLAN), copy.deepcopy(REGISTRY), ROOT)
        self.assertTrue(result["passed"], result["errors"])
        self.assertFalse(result["launch_authorized"])
        self.assertEqual(result["scene_count"], 6)
        self.assertEqual(len(result["execution_order_after_qualification"]), 5)

    def test_runtime_scene_is_rejected_while_locked(self) -> None:
        result = validate_plan(
            copy.deepcopy(PLAN),
            copy.deepcopy(REGISTRY),
            ROOT,
            scene="gcp_100000_20260610",
        )
        self.assertFalse(result["passed"])
        self.assertTrue(any("runtime launch is locked" in error for error in result["errors"]))

    def test_fake_qualification_pass_is_rejected(self) -> None:
        plan = copy.deepcopy(PLAN)
        plan["qualification"]["status"] = "PASS"
        self.assertFalse(validate_plan(plan, copy.deepcopy(REGISTRY), ROOT)["passed"])

    def test_legacy_reuse_is_rejected(self) -> None:
        plan = copy.deepcopy(PLAN)
        plan["legacy_route"]["formal_reuse_allowed"] = True
        self.assertFalse(validate_plan(plan, copy.deepcopy(REGISTRY), ROOT)["passed"])

    def test_formula_change_is_rejected(self) -> None:
        plan = copy.deepcopy(PLAN)
        plan["frozen_method_identity"]["formal_formula"] = "A/H"
        self.assertFalse(validate_plan(plan, copy.deepcopy(REGISTRY), ROOT)["passed"])

    def test_serializer_patch_is_rejected(self) -> None:
        plan = copy.deepcopy(PLAN)
        plan["frozen_method_identity"]["runtime_training_patch"] = "db8deeb"
        self.assertFalse(validate_plan(plan, copy.deepcopy(REGISTRY), ROOT)["passed"])

    def test_scene_set_change_is_rejected(self) -> None:
        plan = copy.deepcopy(PLAN)
        plan["scenes"].pop()
        self.assertFalse(validate_plan(plan, copy.deepcopy(REGISTRY), ROOT)["passed"])

    def test_loaded_dimension_change_is_rejected(self) -> None:
        plan = copy.deepcopy(PLAN)
        plan["scenes"][1]["loaded_height"] += 1
        self.assertFalse(validate_plan(plan, copy.deepcopy(REGISTRY), ROOT)["passed"])

    def test_registry_cannot_claim_legacy_full_matrix_pass(self) -> None:
        registry = copy.deepcopy(REGISTRY)
        registry["methods"][0]["full_scene_matrix_eligible"] = True
        self.assertFalse(validate_plan(copy.deepcopy(PLAN), registry, ROOT)["passed"])

    def test_legacy_launcher_is_hard_disabled(self) -> None:
        launcher = (ROOT / "scripts/gcp_v13/run_original_3dgs_full_scene_30k.sh").read_text(encoding="utf-8")
        self.assertIn("LEGACY_FULL_MATRIX_LAUNCHER_DISABLED", launcher)
        self.assertIn("exit 64", launcher)


if __name__ == "__main__":
    unittest.main()
