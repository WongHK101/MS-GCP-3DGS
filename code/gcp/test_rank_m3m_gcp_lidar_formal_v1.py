#!/usr/bin/env python3
"""Exact-pool, verifier-bound macro-ranking and failure-policy tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rank_m3m_gcp_lidar_formal_v1 import SCENES, build_ranking
from verify_m3m_gcp_lidar_formal_v1 import METRIC_FIELDS, canonical_sha256, sha256_file


METHOD_CLASSES = {
    "3dgs_original": "rgb_colmap_only", "2dgs": "rgb_colmap_only",
    "pgsr": "rgb_colmap_only", "rade_gs": "rgb_colmap_only",
    "qgs": "rgb_colmap_only", "gsprior": "rgb_colmap_only", "sof": "rgb_colmap_only",
    "citygaussian_v2": "rgb_colmap_external_geometry_prior",
    "citygs_x": "rgb_colmap_external_geometry_prior",
    "metrogs": "rgb_colmap_external_geometry_prior",
}


class SixSceneRankerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        repo_root = Path(__file__).resolve().parents[2]
        self.schema = json.loads((repo_root / "configs" / "m3m_gcp_lidar_formal_artifact_schema_v1.json").read_text(encoding="utf-8"))
        registry_path = repo_root / "configs" / "m3m_gcp_native_quarter_method_registry_v3.json"
        self.registry = json.loads(registry_path.read_text(encoding="utf-8"))
        self.registry_names = {
            row["method_id"]: row["display_name"] for row in self.registry["methods"]
            if row["method_id"] in METHOD_CLASSES
        }
        self.contract_sha, self.activation_sha = "c" * 64, "a" * 64
        self.schema_sha, self.registry_sha = "s" * 64, sha256_file(registry_path)
        self.evaluator_sha, self.verifier_sha = "e" * 64, "v" * 64
        self.contract = {
            "implementation": {"evaluator_sha256": self.evaluator_sha, "verifier_sha256": self.verifier_sha, "artifact_schema_sha256": self.schema_sha},
            "method_registry_binding": {"file_sha256": self.registry_sha, "active_method_ids_in_order": list(METHOD_CLASSES), "active_method_input_classes": METHOD_CLASSES},
        }
        self.activation = {"contract_file_sha256": self.contract_sha}
        self.activation["canonical_sha256"] = canonical_sha256(self.activation)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def result_entry(self, method_id: str, scene: str, value: float) -> dict:
        metrics = {field: value for field in METRIC_FIELDS}
        result = {
            "schema": "m3m_gcp_lidar_method_result_v1", "protocol_id": "m3m_gcp_lidar_rendered_surface_v1",
            "contract_file_sha256": self.contract_sha, "activation_manifest_sha256": self.activation_sha,
            "scene_execution_authorization_sha256": "x" * 64, "formal_methods_manifest_sha256": "m" * 64,
            "protocol_manifest_canonical_sha256": "p" * 64, "scene": scene, "method_id": method_id,
            "method": self.registry_names[method_id], "input_class": METHOD_CLASSES[method_id],
            "model_checkpoint_sha256": "1" * 64, "recipe_sha256": "2" * 64,
            "renderer_adapter_sha256": "3" * 64, "packet_manifest_sha256": "4" * 64,
            "surface_npz_sha256": "5" * 64, "distance_npz_sha256": "6" * 64,
            "reference_npz_sha256": "7" * 64, "evaluator_sha256": self.evaluator_sha,
            "verifier_sha256": self.verifier_sha, "artifact_schema_sha256": self.schema_sha,
            "train_view_count": 1, "reference_point_count": 1, "reconstruction_point_count": 1,
            "reconstruction_to_lidar_distance_count": 1, "lidar_to_reconstruction_distance_count": 1,
            "surface_audit": {}, "metrics": metrics, "summary_row": {},
        }
        result["canonical_sha256"] = canonical_sha256(result)
        result_path = self.root / f"{method_id}-{scene}-result.json"
        result_path.write_text(json.dumps(result), encoding="utf-8")
        result_sha = sha256_file(result_path)
        report = {
            "schema": "m3m_gcp_lidar_formal_verification_v1", "protocol_id": "m3m_gcp_lidar_rendered_surface_v1",
            "status": "PASS_VERIFIED_FORMAL_V1", "method_id": method_id, "scene": scene,
            "method_result_sha256": result_sha, "contract_file_sha256": self.contract_sha,
            "activation_manifest_sha256": self.activation_sha,
            "scene_execution_authorization_sha256": result["scene_execution_authorization_sha256"],
            "formal_methods_manifest_sha256": result["formal_methods_manifest_sha256"],
            "artifact_schema_sha256": self.schema_sha, "evaluator_sha256": self.evaluator_sha,
            "verifier_sha256": self.verifier_sha, "surface_npz_sha256": result["surface_npz_sha256"],
            "distance_npz_sha256": result["distance_npz_sha256"], "reference_npz_sha256": result["reference_npz_sha256"],
            "reconstruction_to_lidar_distance_count": 1, "lidar_to_reconstruction_distance_count": 1,
            "errors": [], "recomputed_metrics": metrics,
        }
        report["canonical_sha256"] = canonical_sha256(report)
        report_path = self.root / f"{method_id}-{scene}-verification.json"
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return {
            "scene": scene, "status": "COMPLETE_RANKED",
            "method_result_path": str(result_path), "method_result_sha256": result_sha,
            "verification_report_path": str(report_path), "verification_report_sha256": sha256_file(report_path),
        }

    @staticmethod
    def incomplete_entry(scene: str, status: str = "INCOMPLETE_UNRANKED") -> dict:
        return {"scene": scene, "status": status, "method_result_path": None, "method_result_sha256": None, "verification_report_path": None, "verification_report_sha256": None}

    def manifest(self) -> dict:
        methods = []
        for method_id in METHOD_CLASSES:
            if method_id == "3dgs_original":
                scenes = [self.result_entry(method_id, scene, 0.8) for scene in SCENES]
            elif method_id == "2dgs":
                scenes = [self.result_entry(method_id, scene, 0.8 + 5e-10) for scene in SCENES]
            elif method_id == "pgsr":
                scenes = [self.result_entry(method_id, scene, 0.7) for scene in SCENES[:-1]] + [self.incomplete_entry(SCENES[-1], "OOM_UNRANKED")]
            else:
                scenes = [self.incomplete_entry(scene) for scene in SCENES]
            methods.append({"method_id": method_id, "method_name": self.registry_names[method_id], "input_class": METHOD_CLASSES[method_id], "scenes": scenes})
        payload = {"schema": "m3m_gcp_lidar_six_scene_results_manifest_v1", "protocol_id": "m3m_gcp_lidar_rendered_surface_v1", "methods": methods}
        payload["canonical_sha256"] = canonical_sha256(payload)
        return payload

    def build(self, manifest: dict) -> dict:
        return build_ranking(
            manifest, contract=self.contract, contract_sha256=self.contract_sha,
            activation=self.activation, activation_sha256=self.activation_sha,
            schema=self.schema, schema_sha256=self.schema_sha,
            registry=self.registry, registry_sha256=self.registry_sha,
        )

    def test_exact_pool_verified_competition_ranking_and_partial_macro(self) -> None:
        rows = {row["method_id"]: row for row in self.build(self.manifest())["methods"]}
        self.assertEqual(rows["3dgs_original"]["official_input_class_rank"], 1)
        self.assertEqual(rows["2dgs"]["official_input_class_rank"], 1)
        self.assertFalse(rows["pgsr"]["ranking_eligible"])
        self.assertEqual(rows["pgsr"]["completed_scene_count"], 5)

    def test_arbitrary_method_pool_is_rejected(self) -> None:
        manifest = self.manifest()
        manifest["methods"][0]["method_id"] = "arbitrary"
        manifest["canonical_sha256"] = canonical_sha256(manifest)
        with self.assertRaisesRegex(ValueError, "exact ordered ten-method pool"):
            self.build(manifest)

    def test_failed_scene_cannot_carry_fabricated_result_or_report(self) -> None:
        manifest = self.manifest()
        bad = manifest["methods"][2]["scenes"][-1]
        bad["verification_report_path"] = "fabricated.json"
        manifest["canonical_sha256"] = canonical_sha256(manifest)
        with self.assertRaisesRegex(ValueError, "fabricated"):
            self.build(manifest)

    def test_complete_scene_requires_pass_verification_report(self) -> None:
        manifest = self.manifest()
        entry = manifest["methods"][0]["scenes"][0]
        report_path = Path(entry["verification_report_path"])
        report = json.loads(report_path.read_text())
        report["status"] = "FAIL"
        report["canonical_sha256"] = canonical_sha256(report)
        report_path.write_text(json.dumps(report))
        entry["verification_report_sha256"] = sha256_file(report_path)
        manifest["canonical_sha256"] = canonical_sha256(manifest)
        with self.assertRaisesRegex(ValueError, "not PASS"):
            self.build(manifest)


if __name__ == "__main__":
    unittest.main()
