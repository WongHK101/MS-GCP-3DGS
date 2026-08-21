#!/usr/bin/env python3
"""Executable review, capacity, mutex and packet-cap tests for the 100K guard."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from m3m_gcp_lidar_artifacts import canonical_sha256
from run_m3m_gcp_100k_guarded import (
    GIB,
    METHOD_ORDER,
    PACKET_CAP_BYTES,
    acquire_packet_state,
    run_phase,
    sha256_file,
    validate_activation_and_recipe,
    validate_capacity,
    validate_packet_freeze_binding,
    validate_prepared_method_input,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


class Guarded100KTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.repo = self.root / "repo"
        self.repo.mkdir()
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        (self.repo / ".gitattributes").write_text(
            "*.py text eol=lf\n*.json text eol=lf\n", encoding="utf-8"
        )
        self.plan = self.repo / "plan.json"
        self.recipe = self.repo / "recipe.json"
        self.recipes = self.repo / "recipes.json"
        script_root = self.repo / "code" / "gcp"
        script_root.mkdir(parents=True)
        source_root = Path(__file__).resolve().parent
        shutil.copy2(source_root / "run_m3m_gcp_100k_guarded.py", script_root)
        shutil.copy2(source_root / "freeze_m3m_gcp_lidar_scene_attempts.py", script_root)
        shutil.copy2(source_root / "build_m3m_gcp_lidar_100k_activation.py", script_root)
        shutil.copy2(source_root / "build_m3m_gcp_100k_attempt_manifest.py", script_root)
        preparation = self.root / "per-method-inputs-v2.json"
        write_json(
            preparation,
            {"status": "PASS_PER_METHOD_INPUT_PREPARATION_NO_TRAINING_NO_PRIOR"},
        )
        obsolete_cleanup = self.root / "obsolete-cleanup.json"
        write_json(
            obsolete_cleanup,
            {
                "status": "PASS_OBSOLETE_FAILED_ATTEMPT_REMOVED",
                "deleted": True,
                "deleted_path": str(self.root / "does-not-exist"),
            },
        )
        contract = self.repo / "contract.json"
        artifact_schema = self.repo / "artifact-schema.json"
        write_json(contract, {"review": {"protocol_review_task_id": "phase1-review"}})
        artifact_schema.write_text("schema", encoding="utf-8")
        self.adapter = self.root / "adapter.py"
        self.adapter.write_text("adapter", encoding="utf-8")
        recipe = {
            "schema": "m3m_gcp_native_quarter_100k_execution_recipe_v1", "method_id": "2dgs",
            "scene": "gcp_100000_20260610", "seed": 0, "status": "REVIEW_CANDIDATE_NOT_EXECUTION_AUTHORIZED",
            "input_class": "rgb_colmap_only",
            "renderer_adapter_sha256": sha256_file(self.adapter),
            "authorized_packet_set_root": str((self.root / "packet-set").resolve()),
            "authorized_packet_state": str((self.root / "packet-state.json").resolve()),
            "phase_commands": {"training": [sys.executable, "-c", "print('ok')"],
                               "packet": [sys.executable, "-c", "import time; time.sleep(5)"]},
        }
        recipe["canonical_sha256"] = canonical_sha256(recipe)
        write_json(self.recipe, recipe)
        recipe_order = ["3dgs_original", "2dgs", "pgsr", "rade_gs", "qgs", "gsprior", "sof",
                        "citygaussian_v2", "citygs_x", "metrogs"]
        recipes = {"schema": "m3m_gcp_native_quarter_100k_recipe_manifest_v1",
                   "status": "REVIEW_CANDIDATE_NOT_EXECUTION_AUTHORIZED",
                   "scene": "gcp_100000_20260610", "seed": 0,
                   "method_order": recipe_order,
                   "recipes": [{"method_id": method_id, "path": "recipe.json",
                                "sha256": sha256_file(self.recipe)} for method_id in recipe_order]}
        recipes["canonical_sha256"] = canonical_sha256(recipes)
        write_json(self.recipes, recipes)
        plan = {
            "schema": "m3m_gcp_native_quarter_100k_ten_method_execution_plan_v1",
            "scene": "gcp_100000_20260610", "seed": 0,
            "status": "REVIEW_CANDIDATE_NOT_EXECUTION_AUTHORIZED", "execution_authorized": False,
            "method_order": ["3dgs_original", "2dgs", "pgsr", "rade_gs", "qgs", "gsprior",
                             "sof", "citygaussian_v2", "citygs_x", "metrogs"],
            "other_prepared_scenes_locked": ["gcp_5000_20260602", "gcp_20000_20260602",
                                              "gcp_10000_20260610", "gcp_50000_20260610"],
            "other_prepared_scene_training_rendering_or_formal_evaluation_authorized": False,
            "formal_lidar_protocol": {
                "contract": {"path": "contract.json", "sha256": sha256_file(contract)},
                "artifact_schema": {"path": "artifact-schema.json", "sha256": sha256_file(artifact_schema)},
                "execution_authorized": False,
            },
            "storage": {"minimum_free_before_prior_gib": 300,
                        "minimum_free_before_training_gib": 300,
                        "minimum_free_before_packet_export_gib": 180,
                        "packet_scratch_hard_cap_gib": 100},
            "execution_closure": {
                "activation_builder": {
                    "path": "code/gcp/build_m3m_gcp_lidar_100k_activation.py",
                    "sha256": sha256_file(script_root / "build_m3m_gcp_lidar_100k_activation.py"),
                },
                "attempt_manifest_builder": {
                    "path": "code/gcp/build_m3m_gcp_100k_attempt_manifest.py",
                    "sha256": sha256_file(script_root / "build_m3m_gcp_100k_attempt_manifest.py"),
                },
                "guarded_runner": {"path": "code/gcp/run_m3m_gcp_100k_guarded.py",
                                   "sha256": sha256_file(script_root / "run_m3m_gcp_100k_guarded.py")},
                "attempt_freezer": {"path": "code/gcp/freeze_m3m_gcp_lidar_scene_attempts.py",
                                    "sha256": sha256_file(script_root / "freeze_m3m_gcp_lidar_scene_attempts.py")},
                "exact_review_verdict_required": "PASS_100K_TIME_SPACE_EXECUTION_PLAN_V1",
            },
            "preparation": {"per_method_input_evidence": {
                "path": str(preparation), "sha256": sha256_file(preparation),
                "status_required": "PASS_PER_METHOD_INPUT_PREPARATION_NO_TRAINING_NO_PRIOR"},
                "obsolete_train_first_attempt_cleanup": {
                    "path": str(obsolete_cleanup), "sha256": sha256_file(obsolete_cleanup),
                    "status_required": "PASS_OBSOLETE_FAILED_ATTEMPT_REMOVED"}},
            "recipe_manifest": {"path": "recipes.json", "file_sha256": sha256_file(self.recipes),
                                "canonical_sha256": recipes["canonical_sha256"]},
            "review": {"task_id": "review-task"},
        }
        plan["canonical_sha256"] = canonical_sha256(plan)
        write_json(self.plan, plan)
        common_preparation = {
            "status": "PASS_COMMON_SCENE_PREPARATION_NO_TRAINING",
            "scene_count": 6,
            "contract_file_sha256": sha256_file(contract),
            "training_started": False,
            "formal_evaluation": "NOT_STARTED",
        }
        self.local_preparation = self.repo / "common-preparation-local.json"
        self.remote_preparation = self.repo / "common-preparation-remote.json"
        write_json(self.local_preparation, common_preparation)
        write_json(self.remote_preparation, common_preparation)
        subprocess.run(["git", "-C", str(self.repo), "add", "."], check=True)
        subprocess.run(["git", "-C", str(self.repo), "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture"], check=True)
        commit = subprocess.check_output(["git", "-C", str(self.repo), "rev-parse", "HEAD"], text=True).strip()
        tree = subprocess.check_output(["git", "-C", str(self.repo), "show", "-s", "--format=%T", "HEAD"], text=True).strip()
        self.activation = self.root / "activation.json"
        activation = {
            "schema": "m3m_gcp_lidar_formal_activation_v1",
            "protocol_id": "m3m_gcp_lidar_rendered_surface_v1",
            "protocol_review_task_id": "phase1-review",
            "protocol_review_verdict": "PASS_LIDAR_V1_AND_SIX_SCENE_PREPARATION_V2",
            "protocol_reviewed_commit": commit, "protocol_reviewed_tree": tree,
            "execution_plan_review_task_id": "review-task",
            "execution_plan_review_verdict": "PASS_100K_TIME_SPACE_EXECUTION_PLAN_V1",
            "execution_plan_reviewed_commit": commit, "execution_plan_reviewed_tree": tree,
            "execution_authorized": True, "execution_plan_path": "plan.json", "execution_plan_sha256": sha256_file(self.plan),
            "recipe_manifest_path": "recipes.json", "recipe_manifest_sha256": sha256_file(self.recipes),
            "contract_file_sha256": sha256_file(contract),
            "artifact_schema_sha256": sha256_file(artifact_schema),
            "common_preparation_local_path": "common-preparation-local.json",
            "common_preparation_local_sha256": sha256_file(self.local_preparation),
            "common_preparation_remote_path": "common-preparation-remote.json",
            "common_preparation_remote_sha256": sha256_file(self.remote_preparation),
            "benchmark_commit": commit, "benchmark_tree": tree,
        }
        activation["canonical_sha256"] = canonical_sha256(activation)
        write_json(self.activation, activation)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_exact_review_activation_and_recipe_pass(self) -> None:
        _, recipe = validate_activation_and_recipe(
            repo=self.repo, activation_path=self.activation, plan_path=self.plan,
            recipe_manifest_path=self.recipes, recipe_path=self.recipe, method_id="2dgs",
        )
        self.assertEqual(recipe["method_id"], "2dgs")

    def test_old_or_arbitrary_pass_verdict_rejected(self) -> None:
        payload = json.loads(self.activation.read_text())
        payload["protocol_review_verdict"] = "PASS_LIDAR_V1_AND_SIX_SCENE_PREPARATION"
        payload["canonical_sha256"] = canonical_sha256(payload)
        write_json(self.activation, payload)
        with self.assertRaisesRegex(RuntimeError, "both exact protocol/data and 100K review verdicts"):
            validate_activation_and_recipe(
                repo=self.repo, activation_path=self.activation, plan_path=self.plan,
                recipe_manifest_path=self.recipes, recipe_path=self.recipe, method_id="2dgs",
            )

    def test_executable_capacity_thresholds(self) -> None:
        validate_capacity(self.root, "training", free_bytes=300 * GIB)
        validate_capacity(self.root, "packet", free_bytes=180 * GIB)
        with self.assertRaisesRegex(RuntimeError, "300 GiB"):
            validate_capacity(self.root, "training", free_bytes=300 * GIB - 1)
        with self.assertRaisesRegex(RuntimeError, "180 GiB"):
            validate_capacity(self.root, "packet", free_bytes=180 * GIB - 1)

    def test_prepared_formal_input_rehashes_shared_all_image_sfm(self) -> None:
        dataset = self.root / "prepared-dataset"
        sparse = dataset / "sparse" / "0"
        sparse.mkdir(parents=True)
        for name, value in {
            "cameras.bin": "cam", "images.bin": "poses", "points3D.ply": "ply"
        }.items():
            (sparse / name).write_text(value, encoding="utf-8")
        all_image_root = self.root / "all-image"
        all_image_root.mkdir()
        (all_image_root / "images.bin").write_text("all-image-tracks", encoding="utf-8")
        package_audit = self.root / "package-audit.json"
        write_json(package_audit, {"status": "pass"})
        materializer = self.root / "materializer.py"
        materializer.write_text("materializer", encoding="utf-8")
        evidence = {
            "status": "PASS_PER_METHOD_INPUT_PREPARATION_NO_TRAINING_NO_PRIOR",
            "access_boundary": {"all_images_participated_in_sfm": True},
            "materializer": {
                "path": str(materializer), "sha256": sha256_file(materializer)
            },
            "shared_all_image_sfm": {
                "path": str(all_image_root), "image_count": 2510,
                "files": {"images.bin": {
                    "bytes": (all_image_root / "images.bin").stat().st_size,
                    "sha256": sha256_file(all_image_root / "images.bin"),
                }},
                "package_audit": {
                    "path": str(package_audit), "sha256": sha256_file(package_audit),
                    "status": "pass",
                },
            },
            "formal_train_view": {"path": str(sparse), "files": {}},
        }
        evidence["canonical_sha256"] = canonical_sha256(evidence)
        evidence_path = self.root / "prepared-evidence.json"
        write_json(evidence_path, evidence)
        recipe = {
            "method_id": "2dgs",
            "benchmark_required_files_sha256": {
                "code/gcp/materialize_m3m_gcp_100k_method_inputs.py": sha256_file(materializer)
            },
            "prepared_method_input_binding": {
                "evidence_path": str(evidence_path),
                "evidence_sha256": sha256_file(evidence_path),
                "dataset_root": str(dataset),
                "input_profile": "exact_formal_train_view_from_shared_all_image_sfm",
                "sparse_sha256": {
                    name: sha256_file(sparse / name)
                    for name in ("cameras.bin", "images.bin", "points3D.ply")
                },
            },
        }
        validate_prepared_method_input(recipe, dataset)
        (all_image_root / "images.bin").write_text("tampered", encoding="utf-8")
        with self.assertRaises(RuntimeError):
            validate_prepared_method_input(recipe, dataset)

    def test_packet_state_is_exclusive_and_persistent(self) -> None:
        state = self.root / "packet-state.json"
        packet_root = self.root / "packets"
        acquire_packet_state(state, "2dgs", packet_root)
        with self.assertRaises(FileExistsError):
            acquire_packet_state(state, "pgsr", self.root / "other-packets")

    def test_packet_phase_binds_immutable_ready_model_and_freeze(self) -> None:
        selected_run = (self.root / "selected-run").resolve()
        rows = []
        for method_id in METHOD_ORDER:
            asset_root = selected_run if method_id == "2dgs" else (self.root / "freeze-assets" / method_id).resolve()
            asset_root.mkdir(parents=True, exist_ok=True)
            model = asset_root / "model.ply"
            model.write_text(method_id, encoding="utf-8")
            identity = {
                "schema": "m3m_gcp_100k_model_identity_v1",
                "protocol_id": "m3m_gcp_lidar_rendered_surface_v1",
                "scene": "gcp_100000_20260610",
                "method_id": method_id,
                "run_root": str(asset_root),
                "inventory": [{
                    "path": str(model.resolve()),
                    "bytes": model.stat().st_size,
                    "sha256": sha256_file(model),
                }],
            }
            identity["canonical_sha256"] = canonical_sha256(identity)
            identity_path = asset_root / "model-identity.json"
            write_json(identity_path, identity)
            if method_id == "2dgs":
                recipe_path, adapter_path = self.recipe.resolve(), self.adapter.resolve()
            else:
                recipe_path, adapter_path = asset_root / "recipe.json", asset_root / "adapter.py"
                recipe_path.write_text(f"{method_id}-recipe", encoding="utf-8")
                adapter_path.write_text(f"{method_id}-adapter", encoding="utf-8")
            rows.append({
                "method_id": method_id, "method_name": method_id,
                "input_class": (
                    "rgb_colmap_external_geometry_prior"
                    if method_id in {"citygaussian_v2", "citygs_x", "metrogs"}
                    else "rgb_colmap_only"
                ),
                "attempt_status": "READY_FOR_EVALUATION", "run_root": str(asset_root),
                "model_checkpoint_path": str(identity_path),
                "model_checkpoint_sha256": sha256_file(identity_path),
                "recipe_path": str(recipe_path), "recipe_sha256": sha256_file(recipe_path),
                "renderer_adapter_path": str(adapter_path),
                "renderer_adapter_sha256": sha256_file(adapter_path),
                "failure_evidence_path": None, "failure_evidence_sha256": None,
            })
        methods = {
            "schema": "m3m_gcp_lidar_formal_methods_v1",
            "protocol_id": "m3m_gcp_lidar_rendered_surface_v1",
            "scene": "gcp_100000_20260610", "methods": rows,
        }
        methods["canonical_sha256"] = canonical_sha256(methods)
        methods_path = (self.root / "methods.json").resolve()
        write_json(methods_path, methods)
        freeze = {
            "schema": "m3m_gcp_lidar_scene_attempt_freeze_v1",
            "protocol_id": "m3m_gcp_lidar_rendered_surface_v1",
            "scene": "gcp_100000_20260610", "methods_manifest_path": str(methods_path),
            "methods_manifest_file_sha256": sha256_file(methods_path),
            "methods_manifest_canonical_sha256": methods["canonical_sha256"],
            "frozen_method_ids": METHOD_ORDER, "created_at_utc": "2026-08-21T00:00:00Z",
        }
        freeze["canonical_sha256"] = canonical_sha256(freeze)
        freeze_path = (self.root / "freeze.json").resolve()
        write_json(freeze_path, freeze)
        args = argparse.Namespace(
            scene_attempt_freeze=freeze_path, method_id="2dgs",
            run_root=selected_run, recipe=self.recipe,
        )
        frozen, freeze_sha = validate_packet_freeze_binding(
            args=args, recipe=json.loads(self.recipe.read_text())
        )
        self.assertEqual(
            frozen["model_checkpoint_sha256"],
            sha256_file(selected_run / "model-identity.json"),
        )
        self.assertEqual(freeze_sha, sha256_file(freeze_path))
        (selected_run / "model.ply").write_text("tampered", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "model-identity file changed"):
            validate_packet_freeze_binding(
                args=args, recipe=json.loads(self.recipe.read_text())
            )

    def test_packet_cumulative_cap_terminates_command_and_writes_failure(self) -> None:
        recipe = json.loads(self.recipe.read_text())
        failure = self.root / "evidence" / "failure.json"
        args = argparse.Namespace(
            phase="packet", capacity_root=self.root, packet_set_root=self.root / "packet-set",
            packet_state=self.root / "packet-state.json", method_id="2dgs", run_root=self.root / "run",
            failure_evidence=failure, progress_regex=None, progress_unit="views", poll_seconds=0.01,
            recipe=self.recipe, replacements={}, scene_attempt_freeze=self.root / "freeze.json",
        )
        with mock.patch("run_m3m_gcp_100k_guarded.validate_capacity"), mock.patch(
            "run_m3m_gcp_100k_guarded.directory_bytes", return_value=PACKET_CAP_BYTES + 1
        ), mock.patch(
            "run_m3m_gcp_100k_guarded.validate_packet_freeze_binding",
            return_value=({"model_checkpoint_sha256": "4" * 64}, "5" * 64),
        ):
            code = run_phase(args, recipe, [sys.executable, "-c", "import time; time.sleep(5)"])
        self.assertNotEqual(code, 0)
        evidence = json.loads(failure.read_text())
        self.assertEqual(evidence["status"], "INCOMPLETE_UNRANKED")
        self.assertEqual(evidence["failure_stage"], "packet_export")
        self.assertEqual(evidence["model_checkpoint_sha256"], "4" * 64)
        self.assertEqual(evidence["scene_attempt_freeze_sha256"], "5" * 64)
        self.assertIn("100 GiB", " ".join(evidence["errors"]))
        self.assertTrue(args.packet_state.is_file())

    def test_success_marker_is_exclusive_and_blocks_child_restart(self) -> None:
        failure = self.root / "success-evidence" / "failure.json"
        args = argparse.Namespace(
            phase="training",
            capacity_root=self.root,
            packet_set_root=None,
            packet_state=None,
            method_id="2dgs",
            run_root=self.root / "success-run",
            failure_evidence=failure,
            progress_regex=None,
            progress_unit="iterations",
            poll_seconds=0.01,
            recipe=self.recipe,
            replacements={},
            scene_attempt_freeze=None,
        )
        recipe = json.loads(self.recipe.read_text())
        command = [sys.executable, "-c", "print('ok')"]
        with mock.patch("run_m3m_gcp_100k_guarded.validate_capacity"):
            self.assertEqual(run_phase(args, recipe, command), 0)
        marker = failure.parent / "phase_success.json"
        payload = json.loads(marker.read_text())
        self.assertEqual(payload["status"], "PASS")
        self.assertEqual(payload["phase"], "training")
        with mock.patch("run_m3m_gcp_100k_guarded.validate_capacity"):
            with self.assertRaisesRegex(FileExistsError, "cannot be retried"):
                run_phase(args, recipe, command)


if __name__ == "__main__":
    unittest.main()
