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
        result = validate_registry(copy.deepcopy(REGISTRY))
        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["method_count"], 10)
        self.assertEqual(result["qualification_allowed"], ["3dgs_original"])

    def test_duplicate_method_rejected(self) -> None:
        data = copy.deepcopy(REGISTRY)
        data["methods"][1]["method_id"] = data["methods"][0]["method_id"]
        self.assertFalse(validate_registry(data)["passed"])

    def test_unfrozen_commit_rejected(self) -> None:
        data = copy.deepcopy(REGISTRY)
        data["methods"][1]["source"]["commit"] = "main"
        self.assertFalse(validate_registry(data)["passed"])

    def test_qgs_fake_repository_rejected(self) -> None:
        data = copy.deepcopy(REGISTRY)
        qgs = next(method for method in data["methods"] if method["method_id"] == "qgs")
        qgs["source"]["official_repository"] = "https://example.invalid/qgs"
        self.assertFalse(validate_registry(data)["passed"])

    def test_full_matrix_before_3k_rejected(self) -> None:
        data = copy.deepcopy(REGISTRY)
        data["methods"][0]["full_scene_matrix_eligible"] = True
        self.assertFalse(validate_registry(data)["passed"])


if __name__ == "__main__":
    unittest.main()
