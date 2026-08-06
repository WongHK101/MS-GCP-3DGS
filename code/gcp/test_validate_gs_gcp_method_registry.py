#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from validate_gs_gcp_method_registry import validate_registry


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = json.loads((ROOT / "configs" / "gs_gcp_method_registry_v1.json").read_text(encoding="utf-8"))


class MethodRegistryTests(unittest.TestCase):
    def test_frozen_registry_passes(self) -> None:
        result = validate_registry(copy.deepcopy(REGISTRY), ROOT)
        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["method_count"], 10)
        self.assertEqual(result["qualification_allowed"], ["3dgs_original"])
        self.assertEqual(result["full_scene_matrix_eligible"], [])

    def test_duplicate_method_rejected(self) -> None:
        data = copy.deepcopy(REGISTRY)
        data["methods"][1]["method_id"] = data["methods"][0]["method_id"]
        self.assertFalse(validate_registry(data, ROOT)["passed"])

    def test_unfrozen_commit_rejected(self) -> None:
        data = copy.deepcopy(REGISTRY)
        data["methods"][1]["source"]["commit"] = "main"
        self.assertFalse(validate_registry(data, ROOT)["passed"])

    def test_qgs_fake_repository_rejected(self) -> None:
        data = copy.deepcopy(REGISTRY)
        qgs = next(method for method in data["methods"] if method["method_id"] == "qgs")
        qgs["source"]["official_repository"] = "https://example.invalid/qgs"
        self.assertFalse(validate_registry(data, ROOT)["passed"])

    def test_full_matrix_before_3k_rejected(self) -> None:
        data = copy.deepcopy(REGISTRY)
        data["methods"][1]["full_scene_matrix_eligible"] = True
        self.assertFalse(validate_registry(data, ROOT)["passed"])

    def test_legacy_qualification_cannot_be_reused(self) -> None:
        data = copy.deepcopy(REGISTRY)
        data["methods"][0]["legacy_qualification_evidence"]["formal_reuse_allowed"] = True
        result = validate_registry(data, ROOT)
        self.assertFalse(result["passed"])
        self.assertTrue(any("legacy qualification" in error for error in result["errors"]))

    def test_clean_r4_review_evidence_hash_is_bound(self) -> None:
        data = copy.deepcopy(REGISTRY)
        data["methods"][0]["clean_r4_contract_review_evidence"]["verdict_sha256"] = "0" * 64
        result = validate_registry(data, ROOT)
        self.assertFalse(result["passed"])
        self.assertTrue(any("verdict_sha256 mismatch" in error for error in result["errors"]))

    def test_qualification_status_without_full_matrix_rejected(self) -> None:
        data = copy.deepcopy(REGISTRY)
        data["methods"][0]["three_k_qualification_status"] = "PASS"
        self.assertFalse(validate_registry(data, ROOT)["passed"])


if __name__ == "__main__":
    unittest.main()
