#!/usr/bin/env python3
"""Exact-pool, freeze-bound ranking and failure-transition tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from m3m_gcp_lidar_artifacts import command_sha256
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


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


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
            "implementation": {
                "evaluator_sha256": self.evaluator_sha,
                "verifier_sha256": self.verifier_sha,
                "artifact_schema_sha256": self.schema_sha,
            },
            "method_registry_binding": {
                "file_sha256": self.registry_sha,
                "active_method_ids_in_order": list(METHOD_CLASSES),
                "active_method_input_classes": METHOD_CLASSES,
            },
        }
        self.activation = {
            "schema": "m3m_gcp_lidar_formal_activation_v1",
            "protocol_id": "m3m_gcp_lidar_rendered_surface_v1",
            "protocol_review_task_id": "phase1",
            "protocol_review_verdict": "PASS_LIDAR_V1_AND_SIX_SCENE_PREPARATION_V2",
            "protocol_reviewed_commit": "1" * 40,
            "protocol_reviewed_tree": "2" * 40,
            "execution_plan_review_task_id": "phase2",
            "execution_plan_review_verdict": "PASS_100K_TIME_SPACE_EXECUTION_PLAN_V1",
            "execution_plan_reviewed_commit": "3" * 40,
            "execution_plan_reviewed_tree": "4" * 40,
            "execution_authorized": True,
            "contract_file_sha256": self.contract_sha,
            "artifact_schema_sha256": self.schema_sha,
            "common_preparation_local_path": "local.json",
            "common_preparation_local_sha256": "5" * 64,
            "common_preparation_remote_path": "remote.json",
            "common_preparation_remote_sha256": "5" * 64,
            "execution_plan_path": "plan.json",
            "execution_plan_sha256": "6" * 64,
            "recipe_manifest_path": "recipes.json",
            "recipe_manifest_sha256": "7" * 64,
            "benchmark_commit": "3" * 40,
            "benchmark_tree": "4" * 40,
        }
        self.activation["canonical_sha256"] = canonical_sha256(self.activation)
        self.frozen_rows: dict[tuple[str, str], dict] = {}
        self.methods_shas: dict[str, str] = {}
        self.freeze_shas: dict[str, str] = {}

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _make_assets(self, scene: str, method_id: str) -> dict:
        root = (self.root / "assets" / scene / method_id).resolve()
        root.mkdir(parents=True)
        model, recipe, adapter = root / "model.ply", root / "recipe.json", root / "adapter.py"
        model.write_text(f"{scene}-{method_id}-model", encoding="utf-8")
        recipe.write_text(f"{scene}-{method_id}-recipe", encoding="utf-8")
        adapter.write_text(f"{scene}-{method_id}-adapter", encoding="utf-8")
        return {
            "root": root, "model": model, "recipe": recipe, "adapter": adapter,
            "model_sha": sha256_file(model), "recipe_sha": sha256_file(recipe),
            "adapter_sha": sha256_file(adapter),
        }

    def _make_failure(self, method_id: str, scene: str, status: str, row: dict,
                      *, failure_stage: str, freeze_sha: str | None = None) -> dict:
        root = Path(row["run_root"])
        stdout, stderr, environment = root / "stdout.log", root / "stderr.log", root / "environment.json"
        stdout.write_text("last progress 10\n", encoding="utf-8")
        is_oom = status == "OOM_UNRANKED"
        stderr.write_text("CUDA out of memory\n" if is_oom else "stage failed\n", encoding="utf-8")
        environment.write_text("{}\n", encoding="utf-8")
        argv = ["python", "train.py"] if failure_stage in {"prior", "training"} else ["python", "verify.py"]
        payload = {
            "schema": "m3m_gcp_lidar_failure_evidence_v1",
            "protocol_id": "m3m_gcp_lidar_rendered_surface_v1",
            "scene": scene, "method_id": method_id, "input_class": METHOD_CLASSES[method_id],
            "seed": 0, "status": status, "failure_stage": failure_stage,
            "run_root": row["run_root"],
            "model_checkpoint_sha256": row["model_checkpoint_sha256"] if status == "INCOMPLETE_UNRANKED" else None,
            "scene_attempt_freeze_sha256": freeze_sha if status == "INCOMPLETE_UNRANKED" else None,
            "command_argv": argv, "command_sha256": command_sha256(argv),
            "environment_manifest_path": str(environment), "environment_manifest_sha256": sha256_file(environment),
            "recipe_sha256": row["recipe_sha256"], "renderer_adapter_sha256": row["renderer_adapter_sha256"],
            "started_at_utc": "2026-08-21T00:00:00Z", "ended_at_utc": "2026-08-21T00:01:00Z",
            "exit_code": 1, "last_valid_progress": {"unit": "iterations", "value": 10},
            "peak_gpu_memory_mib": 1000, "process_maximum_rss_kib": 2000,
            "cgroup_memory_events_delta": {"oom": 0, "oom_kill": 0, "max": 0},
            "oom_signal": "CUDA_OUT_OF_MEMORY" if is_oom else None,
            "stdout_path": str(stdout), "stdout_sha256": sha256_file(stdout),
            "stderr_path": str(stderr), "stderr_sha256": sha256_file(stderr), "errors": ["exit 1"],
        }
        payload["canonical_sha256"] = canonical_sha256(payload)
        path = root / f"{status.lower()}-failure.json"
        write_json(path, payload)
        return {
            "scene": scene, "status": status,
            "method_result_path": None, "method_result_sha256": None,
            "verification_report_path": None, "verification_report_sha256": None,
            "failure_evidence_path": str(path), "failure_evidence_sha256": sha256_file(path),
        }

    def scene_freeze(self, scene: str, failure_statuses: dict[str, str]) -> tuple[dict, dict[str, dict]]:
        rows = []
        failures: dict[str, dict] = {}
        for method_id in METHOD_CLASSES:
            assets = self._make_assets(scene, method_id)
            status = failure_statuses.get(method_id, "READY_FOR_EVALUATION")
            row = {
                "method_id": method_id, "method_name": self.registry_names[method_id],
                "input_class": METHOD_CLASSES[method_id], "attempt_status": status,
                "run_root": str(assets["root"]),
                "model_checkpoint_path": str(assets["model"]) if status == "READY_FOR_EVALUATION" else None,
                "model_checkpoint_sha256": assets["model_sha"] if status == "READY_FOR_EVALUATION" else None,
                "recipe_path": str(assets["recipe"]), "recipe_sha256": assets["recipe_sha"],
                "renderer_adapter_path": str(assets["adapter"]), "renderer_adapter_sha256": assets["adapter_sha"],
                "failure_evidence_path": None, "failure_evidence_sha256": None,
            }
            if status != "READY_FOR_EVALUATION":
                failure = self._make_failure(method_id, scene, status, row, failure_stage="training")
                row["failure_evidence_path"] = failure["failure_evidence_path"]
                row["failure_evidence_sha256"] = failure["failure_evidence_sha256"]
                failures[method_id] = failure
            rows.append(row)
            self.frozen_rows[(scene, method_id)] = row
        methods_payload = {
            "schema": "m3m_gcp_lidar_formal_methods_v1",
            "protocol_id": "m3m_gcp_lidar_rendered_surface_v1",
            "scene": scene, "methods": rows,
        }
        methods_payload["canonical_sha256"] = canonical_sha256(methods_payload)
        methods_path = (self.root / f"{scene}-methods.json").resolve()
        write_json(methods_path, methods_payload)
        self.methods_shas[scene] = sha256_file(methods_path)
        freeze = {
            "schema": "m3m_gcp_lidar_scene_attempt_freeze_v1",
            "protocol_id": "m3m_gcp_lidar_rendered_surface_v1", "scene": scene,
            "methods_manifest_path": str(methods_path), "methods_manifest_file_sha256": self.methods_shas[scene],
            "methods_manifest_canonical_sha256": methods_payload["canonical_sha256"],
            "frozen_method_ids": list(METHOD_CLASSES), "created_at_utc": "2026-08-21T00:00:00Z",
        }
        freeze["canonical_sha256"] = canonical_sha256(freeze)
        freeze_path = (self.root / f"{scene}-freeze.json").resolve()
        write_json(freeze_path, freeze)
        self.freeze_shas[scene] = sha256_file(freeze_path)
        return {"scene": scene, "path": str(freeze_path), "sha256": self.freeze_shas[scene]}, failures

    def result_entry(self, method_id: str, scene: str, value: float) -> dict:
        row = self.frozen_rows[(scene, method_id)]
        metrics = {field: value for field in METRIC_FIELDS}
        result = {
            "schema": "m3m_gcp_lidar_method_result_v1", "protocol_id": "m3m_gcp_lidar_rendered_surface_v1",
            "contract_file_sha256": self.contract_sha, "activation_manifest_sha256": self.activation_sha,
            "scene_execution_authorization_sha256": "x" * 64,
            "formal_methods_manifest_sha256": self.methods_shas[scene],
            "scene_attempt_freeze_sha256": self.freeze_shas[scene],
            "protocol_manifest_canonical_sha256": "p" * 64, "scene": scene, "method_id": method_id,
            "method": row["method_name"], "input_class": row["input_class"],
            "model_checkpoint_sha256": row["model_checkpoint_sha256"], "recipe_sha256": row["recipe_sha256"],
            "renderer_adapter_sha256": row["renderer_adapter_sha256"], "packet_manifest_sha256": "4" * 64,
            "surface_npz_sha256": "5" * 64, "distance_npz_sha256": "6" * 64,
            "reference_npz_sha256": "7" * 64, "evaluator_sha256": self.evaluator_sha,
            "verifier_sha256": self.verifier_sha, "artifact_schema_sha256": self.schema_sha,
            "train_view_count": 1, "reference_point_count": 1, "reconstruction_point_count": 1,
            "reconstruction_to_lidar_distance_count": 1, "lidar_to_reconstruction_distance_count": 1,
            "surface_audit": {}, "metrics": metrics, "summary_row": {},
        }
        result["canonical_sha256"] = canonical_sha256(result)
        result_path = self.root / f"{method_id}-{scene}-result.json"
        write_json(result_path, result)
        result_sha = sha256_file(result_path)
        report = {
            "schema": "m3m_gcp_lidar_formal_verification_v1", "protocol_id": "m3m_gcp_lidar_rendered_surface_v1",
            "status": "PASS_VERIFIED_FORMAL_V1", "method_id": method_id, "scene": scene,
            "method_result_sha256": result_sha, "contract_file_sha256": self.contract_sha,
            "activation_manifest_sha256": self.activation_sha,
            "scene_execution_authorization_sha256": result["scene_execution_authorization_sha256"],
            "scene_attempt_freeze_sha256": self.freeze_shas[scene],
            "formal_methods_manifest_sha256": self.methods_shas[scene],
            "artifact_schema_sha256": self.schema_sha, "evaluator_sha256": self.evaluator_sha,
            "verifier_sha256": self.verifier_sha, "surface_npz_sha256": result["surface_npz_sha256"],
            "distance_npz_sha256": result["distance_npz_sha256"], "reference_npz_sha256": result["reference_npz_sha256"],
            "reconstruction_to_lidar_distance_count": 1, "lidar_to_reconstruction_distance_count": 1,
            "errors": [], "recomputed_metrics": metrics,
        }
        report["canonical_sha256"] = canonical_sha256(report)
        report_path = self.root / f"{method_id}-{scene}-verification.json"
        write_json(report_path, report)
        return {
            "scene": scene, "status": "COMPLETE_RANKED",
            "method_result_path": str(result_path), "method_result_sha256": result_sha,
            "verification_report_path": str(report_path), "verification_report_sha256": sha256_file(report_path),
            "failure_evidence_path": None, "failure_evidence_sha256": None,
        }

    def manifest(self) -> dict:
        failures_by_scene: dict[str, dict[str, dict]] = {}
        freeze_rows = []
        for scene in SCENES:
            statuses = {method_id: "FAILED_UNRANKED" for method_id in METHOD_CLASSES if method_id not in {"3dgs_original", "2dgs", "pgsr"}}
            if scene == SCENES[-1]:
                statuses["pgsr"] = "OOM_UNRANKED"
            freeze, failures = self.scene_freeze(scene, statuses)
            freeze_rows.append(freeze)
            failures_by_scene[scene] = failures
        methods = []
        for method_id in METHOD_CLASSES:
            if method_id == "3dgs_original":
                scenes = [self.result_entry(method_id, scene, 0.8) for scene in SCENES]
            elif method_id == "2dgs":
                scenes = [self.result_entry(method_id, scene, 0.8 + 5e-10) for scene in SCENES]
            elif method_id == "pgsr":
                scenes = [self.result_entry(method_id, scene, 0.7) for scene in SCENES[:-1]] + [failures_by_scene[SCENES[-1]][method_id]]
            else:
                scenes = [failures_by_scene[scene][method_id] for scene in SCENES]
            methods.append({"method_id": method_id, "method_name": self.registry_names[method_id], "input_class": METHOD_CLASSES[method_id], "scenes": scenes})
        payload = {
            "schema": "m3m_gcp_lidar_six_scene_results_manifest_v1",
            "protocol_id": "m3m_gcp_lidar_rendered_surface_v1",
            "scene_attempt_freezes": freeze_rows, "methods": methods,
        }
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
        write_json(report_path, report)
        entry["verification_report_sha256"] = sha256_file(report_path)
        manifest["canonical_sha256"] = canonical_sha256(manifest)
        with self.assertRaisesRegex(ValueError, "not PASS"):
            self.build(manifest)

    def test_result_must_bind_exact_frozen_methods_manifest(self) -> None:
        manifest = self.manifest()
        entry = manifest["methods"][0]["scenes"][0]
        result_path = Path(entry["method_result_path"])
        result = json.loads(result_path.read_text())
        result["formal_methods_manifest_sha256"] = "9" * 64
        result["canonical_sha256"] = canonical_sha256(result)
        write_json(result_path, result)
        entry["method_result_sha256"] = sha256_file(result_path)
        report_path = Path(entry["verification_report_path"])
        report = json.loads(report_path.read_text())
        report["method_result_sha256"] = entry["method_result_sha256"]
        report["formal_methods_manifest_sha256"] = "9" * 64
        report["canonical_sha256"] = canonical_sha256(report)
        write_json(report_path, report)
        entry["verification_report_sha256"] = sha256_file(report_path)
        manifest["canonical_sha256"] = canonical_sha256(manifest)
        with self.assertRaisesRegex(ValueError, "methods-manifest SHA differs from scene freeze"):
            self.build(manifest)

    def test_frozen_ready_may_become_incomplete_without_mutating_freeze(self) -> None:
        manifest = self.manifest()
        scene, method_id = SCENES[0], "3dgs_original"
        freeze_entry = manifest["scene_attempt_freezes"][0]
        immutable_sha = freeze_entry["sha256"]
        incomplete = self._make_failure(
            method_id, scene, "INCOMPLETE_UNRANKED", self.frozen_rows[(scene, method_id)],
            failure_stage="verification", freeze_sha=immutable_sha,
        )
        manifest["methods"][0]["scenes"][0] = incomplete
        manifest["canonical_sha256"] = canonical_sha256(manifest)
        output = self.build(manifest)
        ranked = next(item for item in output["methods"] if item["method_id"] == method_id)
        self.assertEqual(ranked["scene_statuses"][scene], "INCOMPLETE_UNRANKED")
        self.assertEqual(sha256_file(Path(freeze_entry["path"])), immutable_sha)

    def test_incomplete_must_bind_original_model_and_freeze(self) -> None:
        manifest = self.manifest()
        scene, method_id = SCENES[0], "3dgs_original"
        incomplete = self._make_failure(
            method_id, scene, "INCOMPLETE_UNRANKED", self.frozen_rows[(scene, method_id)],
            failure_stage="verification", freeze_sha=self.freeze_shas[scene],
        )
        failure_path = Path(incomplete["failure_evidence_path"])
        payload = json.loads(failure_path.read_text())
        payload["model_checkpoint_sha256"] = "0" * 64
        payload["canonical_sha256"] = canonical_sha256(payload)
        write_json(failure_path, payload)
        incomplete["failure_evidence_sha256"] = sha256_file(failure_path)
        manifest["methods"][0]["scenes"][0] = incomplete
        manifest["canonical_sha256"] = canonical_sha256(manifest)
        with self.assertRaisesRegex(ValueError, "incomplete evidence model differs from freeze"):
            self.build(manifest)


if __name__ == "__main__":
    unittest.main()
