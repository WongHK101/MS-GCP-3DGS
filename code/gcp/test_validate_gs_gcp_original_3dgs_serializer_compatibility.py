#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from validate_gs_gcp_original_3dgs_serializer_compatibility import validate_serializer_compatibility


ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads((ROOT / "configs/gs_gcp_original_3dgs_serializer_compatibility_v1.json").read_text(encoding="utf-8"))


class SerializerCompatibilityTests(unittest.TestCase):
    def test_frozen_metadata_passes(self) -> None:
        result = validate_serializer_compatibility(copy.deepcopy(CONFIG), repo_root=ROOT)
        self.assertTrue(result["passed"], result["errors"])

    def test_source_commit_change_is_rejected(self) -> None:
        config = copy.deepcopy(CONFIG)
        config["patch"]["commit"] = "0" * 40
        self.assertFalse(validate_serializer_compatibility(config, repo_root=ROOT)["passed"])

    def test_training_math_change_is_rejected(self) -> None:
        config = copy.deepcopy(CONFIG)
        config["patch"]["training_math_changed"] = True
        self.assertFalse(validate_serializer_compatibility(config, repo_root=ROOT)["passed"])

    def test_formal_formula_change_is_rejected(self) -> None:
        config = copy.deepcopy(CONFIG)
        config["formal_invariants"]["formal_formula"] = "A/H"
        self.assertFalse(validate_serializer_compatibility(config, repo_root=ROOT)["passed"])

    def test_failed_real_parity_is_rejected(self) -> None:
        config = copy.deepcopy(CONFIG)
        config["real_5k_parity"]["patched_output_byte_identical"] = False
        self.assertFalse(validate_serializer_compatibility(config, repo_root=ROOT)["passed"])

    def test_runtime_parity_summary_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / CONFIG["real_5k_parity"]["summary_file"]).write_text("{}\n", encoding="utf-8")
            result = validate_serializer_compatibility(
                copy.deepcopy(CONFIG),
                repo_root=ROOT,
                parity_evidence_root=root,
            )
        self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
