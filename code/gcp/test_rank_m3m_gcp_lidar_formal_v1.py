#!/usr/bin/env python3
"""Six-scene macro-ranking, tie and failure-policy tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rank_m3m_gcp_lidar_formal_v1 import SCENES, build_ranking
from verify_m3m_gcp_lidar_formal_v1 import METRIC_FIELDS, canonical_sha256, sha256_file


class SixSceneRankerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.contract_sha = "c" * 64
        self.activation_sha = "a" * 64

    def tearDown(self) -> None:
        self.temp.cleanup()

    def result_entry(self, method_id: str, scene: str, value: float) -> dict:
        metrics = {field: value for field in METRIC_FIELDS}
        result = {
            "schema": "m3m_gcp_lidar_method_result_v1",
            "protocol_id": "m3m_gcp_lidar_rendered_surface_v1",
            "contract_file_sha256": self.contract_sha,
            "activation_manifest_sha256": self.activation_sha,
            "scene": scene,
            "method_id": method_id,
            "input_class": "rgb_colmap_only",
            "metrics": metrics,
        }
        result["canonical_sha256"] = canonical_sha256(result)
        path = self.root / f"{method_id}-{scene}.json"
        path.write_text(json.dumps(result), encoding="utf-8")
        return {
            "scene": scene,
            "status": "COMPLETE_RANKED",
            "method_result_path": str(path),
            "method_result_sha256": sha256_file(path),
        }

    def manifest(self) -> dict:
        complete_a = [self.result_entry("a", scene, 0.8) for scene in SCENES]
        complete_z = [self.result_entry("z", scene, 0.8 + 5e-10) for scene in SCENES]
        incomplete = [self.result_entry("b", scene, 0.7) for scene in SCENES[:-1]]
        incomplete.append({
            "scene": SCENES[-1], "status": "OOM_UNRANKED",
            "method_result_path": None, "method_result_sha256": None,
        })
        payload = {
            "schema": "m3m_gcp_lidar_six_scene_results_manifest_v1",
            "protocol_id": "m3m_gcp_lidar_rendered_surface_v1",
            "methods": [
                {"method_id": "z", "method_name": "Z", "input_class": "rgb_colmap_only", "scenes": complete_z},
                {"method_id": "a", "method_name": "A", "input_class": "rgb_colmap_only", "scenes": complete_a},
                {"method_id": "b", "method_name": "B", "input_class": "rgb_colmap_only", "scenes": incomplete},
            ],
        }
        payload["canonical_sha256"] = canonical_sha256(payload)
        return payload

    def test_complete_only_competition_ranking_and_partial_macro(self) -> None:
        result = build_ranking(
            self.manifest(),
            contract_sha256=self.contract_sha,
            activation_sha256=self.activation_sha,
        )
        rows = {row["method_id"]: row for row in result["methods"]}
        self.assertEqual(rows["a"]["official_input_class_rank"], 1)
        self.assertEqual(rows["z"]["official_input_class_rank"], 1)
        self.assertFalse(rows["b"]["ranking_eligible"])
        self.assertNotIn("official_input_class_rank", rows["b"])
        self.assertEqual(rows["b"]["completed_scene_count"], 5)
        self.assertIn("partial_macro_diagnostic", rows["b"])

    def test_failed_scene_cannot_carry_fabricated_metric_result(self) -> None:
        manifest = self.manifest()
        bad = manifest["methods"][2]["scenes"][-1]
        bad["method_result_path"] = "fabricated.json"
        manifest["canonical_sha256"] = canonical_sha256(manifest)
        with self.assertRaisesRegex(ValueError, "fabricated"):
            build_ranking(
                manifest,
                contract_sha256=self.contract_sha,
                activation_sha256=self.activation_sha,
            )


if __name__ == "__main__":
    unittest.main()
