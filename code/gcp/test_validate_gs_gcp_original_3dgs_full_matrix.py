#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
import re
import unittest
from pathlib import Path

from validate_gs_gcp_original_3dgs_full_matrix import validate_plan


ROOT = Path(__file__).resolve().parents[2]
PLAN = json.loads((ROOT / "configs/gs_gcp_v13_original_3dgs_full_matrix_v1.json").read_text(encoding="utf-8"))
REGISTRY = json.loads((ROOT / "configs/gs_gcp_method_registry_v1.json").read_text(encoding="utf-8"))


class Original3DGSFullMatrixPlanTests(unittest.TestCase):
    def test_frozen_plan_passes(self) -> None:
        result = validate_plan(copy.deepcopy(PLAN), copy.deepcopy(REGISTRY), ROOT)
        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["scene_count"], 6)
        self.assertEqual(len(result["execution_order"]), 4)

    def test_other_method_eligibility_is_rejected(self) -> None:
        registry = copy.deepcopy(REGISTRY)
        registry["methods"][1]["full_scene_matrix_eligible"] = True
        result = validate_plan(copy.deepcopy(PLAN), registry, ROOT)
        self.assertFalse(result["passed"])

    def test_formula_change_is_rejected(self) -> None:
        plan = copy.deepcopy(PLAN)
        plan["frozen_method_identity"]["formal_formula"] = "A/H"
        self.assertFalse(validate_plan(plan, copy.deepcopy(REGISTRY), ROOT)["passed"])

    def test_scene_set_change_is_rejected(self) -> None:
        plan = copy.deepcopy(PLAN)
        plan["scenes"].pop()
        self.assertFalse(validate_plan(plan, copy.deepcopy(REGISTRY), ROOT)["passed"])

    def test_loaded_dimension_change_is_rejected(self) -> None:
        plan = copy.deepcopy(PLAN)
        plan["scenes"][1]["loaded_height"] += 1
        self.assertFalse(validate_plan(plan, copy.deepcopy(REGISTRY), ROOT)["passed"])

    def test_review_evidence_hash_change_is_rejected(self) -> None:
        plan = copy.deepcopy(PLAN)
        plan["qualification_review"]["evidence_sha256"] = "0" * 64
        self.assertFalse(validate_plan(plan, copy.deepcopy(REGISTRY), ROOT)["passed"])

    def test_three_k_cannot_reenter_remaining_scene_execution(self) -> None:
        result = validate_plan(
            copy.deepcopy(PLAN),
            copy.deepcopy(REGISTRY),
            ROOT,
            scene="gcp_3000_20260602",
        )
        self.assertFalse(result["passed"])

    def test_frozen_five_k_cannot_reenter_remaining_scene_execution(self) -> None:
        result = validate_plan(
            copy.deepcopy(PLAN),
            copy.deepcopy(REGISTRY),
            ROOT,
            scene="gcp_5000_20260602",
        )
        self.assertFalse(result["passed"])

    def test_serializer_patch_identity_change_is_rejected(self) -> None:
        plan = copy.deepcopy(PLAN)
        plan["serializer_compatibility"]["runtime_patch_commit"] = "0" * 40
        self.assertFalse(validate_plan(plan, copy.deepcopy(REGISTRY), ROOT)["passed"])

    def test_launcher_uses_isolation_status_schema(self) -> None:
        launcher = (ROOT / "scripts/gcp_v13/run_original_3dgs_full_scene_30k.sh").read_text(
            encoding="utf-8"
        )
        self.assertRegex(launcher, re.escape('["status"] != "pass"'))
        self.assertNotIn('["passed"]', launcher)

    def test_launcher_binds_upstream_and_serializer_patch_separately(self) -> None:
        launcher = (ROOT / "scripts/gcp_v13/run_original_3dgs_full_scene_30k.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("UPSTREAM_METHOD_COMMIT=2eee0e26d2d5fd00ec462df47752223952f6bf4e", launcher)
        self.assertIn("SERIALIZER_COMMIT=db8deebca67e8d5e1507e67c98de603eca0dfd85", launcher)
        self.assertIn("validate_gs_gcp_original_3dgs_serializer_compatibility.py", launcher)
        self.assertIn('--official_source "$OFFICIAL_SOURCE_ROOT"', launcher)
        self.assertIn('--patched_source "$CODE_ROOT"', launcher)
        self.assertIn("tests/test_save_ply_memory_safe.py", launcher)


if __name__ == "__main__":
    unittest.main()
