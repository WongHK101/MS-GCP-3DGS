#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

import build_m3m_gcp_100k_gcp_authorization as gcp_authorization
from m3m_gcp_lidar_artifacts import canonical_sha256, sha256_file


REPO = Path(__file__).resolve().parents[2]


class ThreeTrackAddendumStaticTest(unittest.TestCase):
    def test_addendum_canonical_and_bound_files(self) -> None:
        path = REPO / "configs" / "m3m_gcp_native_quarter_100k_three_track_evaluation_addendum_v1.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["canonical_sha256"], canonical_sha256(payload))
        for relative, expected_sha in payload["bound_addendum_files"].items():
            self.assertEqual(sha256_file(REPO / relative), expected_sha, relative)

    def test_100k_rgb_contract_changes_only_binding_derivation_and_gate(self) -> None:
        base = json.loads(
            (REPO / "configs" / "m3m_gcp_native_quarter_rgb_quality_v1.json").read_text(encoding="utf-8")
        )
        extended = json.loads(
            (REPO / "configs" / "m3m_gcp_native_quarter_rgb_quality_100k_v1.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            extended["derivation"]["source_contract_sha256"],
            sha256_file(REPO / "configs" / "m3m_gcp_native_quarter_rgb_quality_v1.json"),
        )
        base_clean = copy.deepcopy(base)
        extended_clean = copy.deepcopy(extended)
        extended_clean.pop("derivation")
        base_clean["input_binding"]["scene_bindings"] = {}
        extended_clean["input_binding"]["scene_bindings"] = {}
        extended_clean["formal_gate"].pop("three_track_addendum_activation_required")
        extended_clean["formal_gate"]["activation_preflight"] = base_clean["formal_gate"]["activation_preflight"]
        extended_clean["formal_gate"]["benchmark_checkout_identity"] = base_clean["formal_gate"][
            "benchmark_checkout_identity"
        ]
        self.assertEqual(extended_clean, base_clean)
        binding = extended["input_binding"]["scene_bindings"]["gcp_100000_20260610"]
        self.assertEqual(
            [
                binding["full_view_count"],
                binding["train_view_count"],
                binding["test_view_count"],
                binding["width"],
                binding["height"],
            ],
            [2510, 2196, 314, 1414, 1024],
        )

    def test_packet_content_hashes_accept_directory_inventory(self) -> None:
        observed = gcp_authorization.packet_content_hashes(
            {
                "sha256": "a" * 64,
                "files": [
                    {"path": "point_cloud.ply", "sha256": "b" * 64},
                    {"path": "cfg_args", "sha256": "c" * 64},
                ],
            }
        )
        self.assertEqual(observed, {"a" * 64, "b" * 64, "c" * 64})
        self.assertEqual(gcp_authorization.packet_content_hashes("d" * 64), {"d" * 64})
        self.assertEqual(gcp_authorization.packet_content_hashes(None), set())


if __name__ == "__main__":
    unittest.main()
