#!/usr/bin/env python3
"""Executable review, capacity, mutex and packet-cap tests for the 100K guard."""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from m3m_gcp_lidar_artifacts import canonical_sha256, command_sha256
from run_m3m_gcp_100k_guarded import (
    GIB,
    METHOD_ORDER,
    PACKET_CAP_BYTES,
    REQUIRED_NOFILE_SOFT,
    acquire_packet_state,
    build_phase_product_rows,
    configure_nofile_limit,
    observe_child_nofile_limit,
    run_phase,
    sha256_file,
    validate_activation_and_recipe,
    validate_capacity,
    validate_gsprior_normalized_input,
    validate_model_identity_bundle,
    validate_packet_freeze_binding,
    validate_prepared_method_input,
    validate_prior_phase_success,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def gaussian_ply_bytes() -> bytes:
    names = [
        "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2",
        *[f"f_rest_{index}" for index in range(45)],
        "opacity", "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    ]
    header = "\n".join([
        "ply", "format binary_little_endian 1.0", "element vertex 1",
        *[f"property float {name}" for name in names], "end_header", "",
    ]).encode("ascii")
    return header + struct.pack(f"<{len(names)}f", *([0.0] * len(names)))


def write_gaussian_ply(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gaussian_ply_bytes())


def nofile_evidence() -> dict:
    return {
        "resource": "RLIMIT_NOFILE",
        "required_soft": 65536,
        "hard_minimum": 65536,
        "parent_before": {"soft": 1024, "hard": 1048576},
        "parent_after": {"soft": 65536, "hard": 1048576},
    }


def write_success_environment(
    path: Path, *, method_id: str, phase: str
) -> None:
    payload = {
        "schema": "m3m_gcp_100k_execution_environment_v2",
        "scene": "gcp_100000_20260610",
        "method_id": method_id,
        "phase": phase,
        "argv": ["fixture"],
        "python": sys.version,
        "platform": sys.platform,
        "gpu_prelaunch": {},
        "resource_limits": {
            **nofile_evidence(),
            "child_actual": {"soft": 65536, "hard": 1048576},
        },
        "started_at_utc": "2026-08-21T00:00:00Z",
    }
    payload["canonical_sha256"] = canonical_sha256(payload)
    write_json(path, payload)


class Guarded100KTest(unittest.TestCase):
    def setUp(self) -> None:
        self.configure_limit_patch = mock.patch(
            "run_m3m_gcp_100k_guarded.configure_nofile_limit",
            side_effect=lambda required=REQUIRED_NOFILE_SOFT: nofile_evidence(),
        )
        self.observe_limit_patch = mock.patch(
            "run_m3m_gcp_100k_guarded.observe_child_nofile_limit",
            return_value={"soft": 65536, "hard": 1048576},
        )
        self.configure_limit_patch.start()
        self.observe_limit_patch.start()
        self.addCleanup(self.configure_limit_patch.stop)
        self.addCleanup(self.observe_limit_patch.stop)
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
            "schema": "m3m_gcp_native_quarter_100k_execution_recipe_v2", "method_id": "2dgs",
            "scene": "gcp_100000_20260610", "seed": 0, "status": "REVIEW_CANDIDATE_NOT_EXECUTION_AUTHORIZED",
            "input_class": "rgb_colmap_only",
            "budget": {"type": "iterations", "value": 30000},
            "renderer_adapter_sha256": sha256_file(self.adapter),
            "authorized_run_root": str((self.root / "selected-run").resolve()),
            "authorized_evidence_root": str((self.root / "selected-evidence").resolve()),
            "authorized_packet_set_root": str((self.root / "packet-set").resolve()),
            "authorized_packet_state": str((self.root / "packet-state.json").resolve()),
            "fresh_run_root_policy": {
                "prior_and_training_require_absent_run_root_at_guard_admission": True,
                "prior_must_not_create_run_root": True,
                "training_guard_exclusively_creates_empty_run_root_before_child": True,
                "training_child_must_create_final_products": True,
            },
            "process_resource_limits": {
                "applies_to_phases": ["prior", "training", "packet"],
                "rlimit_nofile_hard_minimum": 65536,
                "rlimit_nofile_soft": 65536,
                "record_parent_before_after": True,
                "record_child_actual_inheritance": True,
            },
            "phase_commands": {"training": [sys.executable, "-c", "print('ok')"],
                               "packet": [sys.executable, "-c", "import time; time.sleep(5)"]},
            "phase_roots": {
                "training": {
                    "dataset_root": str((self.root / "dataset").resolve()),
                    "prior_root": str((self.root / "dataset").resolve()),
                }
            },
            "source_bindings": {
                "training": {"root": str((self.root / "source").resolve())}
            },
        }
        recipe["canonical_sha256"] = canonical_sha256(recipe)
        write_json(self.recipe, recipe)
        recipe_order = ["3dgs_original", "2dgs", "pgsr", "rade_gs", "qgs", "gsprior", "sof",
                        "citygaussian_v2", "citygs_x", "metrogs"]
        recipes = {"schema": "m3m_gcp_native_quarter_100k_recipe_manifest_v2",
                   "status": "REVIEW_CANDIDATE_NOT_EXECUTION_AUTHORIZED",
                   "scene": "gcp_100000_20260610", "seed": 0,
                   "method_order": recipe_order,
                   "recipes": [{"method_id": method_id, "path": "recipe.json",
                                "sha256": sha256_file(self.recipe)} for method_id in recipe_order]}
        recipes["canonical_sha256"] = canonical_sha256(recipes)
        write_json(self.recipes, recipes)
        superseded_files = []
        for index in range(6):
            path = (self.root / "superseded" / f"artifact-{index}.bin").resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"old-{index}".encode("ascii"))
            superseded_files.append({
                "path": str(path), "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
        supersession_receipt = self.repo / "supersession.json"
        supersession = {
            "schema": "m3m_gcp_100k_activation_supersession_v1",
            "status": "SUPERSEDED_INFRASTRUCTURE_INVALID_NOT_RANKABLE",
            "classification": {
                "algorithm_failure": False,
                "formal_retry_counted": False,
                "rankable": False,
            },
            "remote_artifacts": superseded_files,
        }
        supersession["canonical_sha256"] = canonical_sha256(supersession)
        write_json(supersession_receipt, supersession)
        plan = {
            "schema": "m3m_gcp_native_quarter_100k_ten_method_execution_plan_v2",
            "activation_manifest_path": str((self.root / "activation.json").resolve()),
            "scene": "gcp_100000_20260610", "seed": 0,
            "status": "REVIEW_CANDIDATE_NOT_EXECUTION_AUTHORIZED", "execution_authorized": False,
            "method_order": ["3dgs_original", "2dgs", "pgsr", "rade_gs", "qgs", "gsprior",
                             "sof", "citygaussian_v2", "citygs_x", "metrogs"],
            "other_prepared_scenes_locked": ["gcp_5000_20260602", "gcp_20000_20260602",
                                              "gcp_10000_20260610", "gcp_50000_20260610"],
            "other_prepared_scene_training_rendering_or_formal_evaluation_authorized": False,
            "superseded_activation": {
                "receipt": {
                    "path": "supersession.json",
                    "sha256": sha256_file(supersession_receipt),
                },
                "status_required": "SUPERSEDED_INFRASTRUCTURE_INVALID_NOT_RANKABLE",
                "algorithm_failure": False,
                "formal_retry_counted": False,
                "rankable": False,
                "remote_artifacts_must_remain_byte_identical": True,
            },
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
                "prior_and_training_require_absent_run_root_at_guard_admission": True,
                "training_child_must_create_products_inside_new_run_root": True,
                "ready_model_identity_requires_exact_phase_success_markers": True,
                "phase_success_command_rehashed_against_frozen_recipe": True,
                "rlimit_nofile_soft_required_for_child_phases": 65536,
                "rlimit_nofile_hard_minimum_prechild_gate": 65536,
                "rlimit_nofile_parent_before_after_evidence_required": True,
                "rlimit_nofile_child_actual_inheritance_evidence_required": True,
            },
            "preparation": {"per_method_input_evidence": {
                "path": str(preparation), "sha256": sha256_file(preparation),
                "status_required": "PASS_PER_METHOD_INPUT_PREPARATION_NO_TRAINING_NO_PRIOR"},
                "obsolete_train_first_attempt_cleanup": {
                    "path": str(obsolete_cleanup), "sha256": sha256_file(obsolete_cleanup),
                    "status_required": "PASS_OBSOLETE_FAILED_ATTEMPT_REMOVED"}},
            "recipe_manifest": {"path": "recipes.json", "file_sha256": sha256_file(self.recipes),
                                "canonical_sha256": recipes["canonical_sha256"]},
            "attempt_freeze": {
                "execution_plan_path": "plan.json",
                "recipe_manifest_path": "recipes.json",
                "method_registry_path": "registry.json",
                "model_identity_root": str((self.root / "model-identities").resolve()),
                "attempt_manifest_path": str((self.root / "methods.json").resolve()),
                "scene_attempt_freeze_path": str((self.root / "freeze.json").resolve()),
            },
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

    def test_superseded_v1_artifact_tamper_is_rejected(self) -> None:
        plan = json.loads(self.plan.read_text())
        receipt_path = self.repo / plan["superseded_activation"]["receipt"]["path"]
        receipt = json.loads(receipt_path.read_text())
        Path(receipt["remote_artifacts"][0]["path"]).write_text(
            "tampered", encoding="utf-8"
        )
        with self.assertRaisesRegex(RuntimeError, "superseded v1 evidence changed"):
            validate_activation_and_recipe(
                repo=self.repo, activation_path=self.activation,
                plan_path=self.plan, recipe_manifest_path=self.recipes,
                recipe_path=self.recipe, method_id="2dgs",
            )

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

    def test_low_nofile_hard_limit_rejects_before_artifacts_or_child(self) -> None:
        failure = self.root / "low-hard-evidence" / "failure.json"
        run_root = self.root / "low-hard-run"
        args = argparse.Namespace(
            phase="training", capacity_root=self.root, packet_set_root=None,
            packet_state=None, method_id="2dgs", run_root=run_root,
            dataset_root=self.root / "dataset", prior_root=self.root / "prior",
            failure_evidence=failure, progress_regex=None,
            progress_unit="iterations", poll_seconds=0.01, recipe=self.recipe,
            replacements={}, scene_attempt_freeze=None,
        )
        recipe = json.loads(self.recipe.read_text())
        with mock.patch(
            "run_m3m_gcp_100k_guarded.configure_nofile_limit",
            side_effect=RuntimeError("RLIMIT_NOFILE hard limit is below 65536"),
        ), mock.patch("subprocess.Popen") as popen:
            with self.assertRaisesRegex(RuntimeError, "hard limit"):
                run_phase(args, recipe, [sys.executable, "-c", "print('never')"])
        popen.assert_not_called()
        self.assertFalse(run_root.exists())
        self.assertFalse(failure.parent.exists())

    def test_configure_nofile_sets_exact_soft_and_preserves_hard(self) -> None:
        fake_resource = mock.Mock()
        fake_resource.RLIMIT_NOFILE = 7
        fake_resource.RLIM_INFINITY = -1
        fake_resource.getrlimit.side_effect = [
            (1024, 1048576),
            (65536, 1048576),
        ]
        with mock.patch("run_m3m_gcp_100k_guarded.resource", fake_resource):
            evidence = configure_nofile_limit(65536)
        fake_resource.setrlimit.assert_called_once_with(7, (65536, 1048576))
        self.assertEqual(evidence["parent_before"], {"soft": 1024, "hard": 1048576})
        self.assertEqual(evidence["parent_after"], {"soft": 65536, "hard": 1048576})

    def test_configure_nofile_rejects_insufficient_hard_limit(self) -> None:
        fake_resource = mock.Mock()
        fake_resource.RLIMIT_NOFILE = 7
        fake_resource.RLIM_INFINITY = -1
        fake_resource.getrlimit.return_value = (1024, 4096)
        with mock.patch("run_m3m_gcp_100k_guarded.resource", fake_resource):
            with self.assertRaisesRegex(RuntimeError, "hard limit is below"):
                configure_nofile_limit(65536)
        fake_resource.setrlimit.assert_not_called()

    @unittest.skipIf(os.name == "nt", "Linux /proc inheritance proof")
    def test_actual_child_inherits_parent_nofile_limit(self) -> None:
        configured = configure_nofile_limit(65536)
        self.assertEqual(configured["parent_after"]["soft"], 65536)
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(1)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            observed = observe_child_nofile_limit(process)
            self.assertEqual(observed["soft"], 65536)
            self.assertTrue(
                observed["hard"] == "unlimited"
                or int(observed["hard"]) >= int(observed["soft"])
            )
        finally:
            process.terminate()
            process.wait(timeout=5)

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

    def test_training_requires_exact_prior_success_and_outputs(self) -> None:
        evidence_root = self.root / "city-prior-evidence"
        dataset = self.root / "city-dataset"
        dataset.mkdir()
        recipe = {
            "method_id": "citygaussian_v2",
            "budget": {"type": "official_matrixcity_aerial_4x4"},
            "authorized_evidence_root": str(evidence_root),
            "phase_commands": {"prior": ["prior-tool", "{dataset_root}"]},
        }
        replacements = {"dataset_root": str(dataset)}
        command = ["prior-tool", str(dataset)]
        environment_path = evidence_root / "prior" / "environment.json"
        write_success_environment(
            environment_path, method_id="citygaussian_v2", phase="prior"
        )
        success = {
            "schema": "m3m_gcp_100k_phase_success_v2",
            "status": "PASS",
            "scene": "gcp_100000_20260610",
            "method_id": "citygaussian_v2",
            "phase": "prior",
            "recipe_sha256": sha256_file(self.recipe),
            "command_sha256": command_sha256(command),
            "frozen_budget": recipe["budget"],
            "environment_manifest_path": str(environment_path.resolve()),
            "environment_manifest_sha256": sha256_file(environment_path),
            "completion_evidence": {
                "progress_unit": "prior_products",
                "last_valid_progress": 2196.0,
                "required_product_postvalidation_passed": True,
            },
            "products": [],
            "ended_at_utc": "2026-08-21T00:00:00Z",
        }
        success["canonical_sha256"] = canonical_sha256(success)
        write_json(evidence_root / "prior" / "phase_success.json", success)
        with self.assertRaisesRegex(RuntimeError, "prior manifest"):
            validate_prior_phase_success(
                recipe=recipe,
                recipe_path=self.recipe,
                run_root=self.root / "run",
                dataset_root=dataset,
                prior_root=dataset,
                replacements=replacements,
            )
        names = [f"image_{index:04d}.JPG" for index in range(2196)]
        depth_rows = []
        for index, name in enumerate(names):
            path = dataset / "depths" / f"{index:04d}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"depth-{index}".encode("ascii"))
            depth_rows.append({
                "image_name": name,
                "relative_path": path.relative_to(dataset).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
        scales_path = dataset / "depth_scales.json"
        write_json(scales_path, {
            name: {"scale": 1.0, "offset": 0.0} for name in names
        })
        manifest_path = dataset / "depth_prior_v1.json"
        write_json(manifest_path, {
            "status": "PASS", "passed": True,
            "method_id": "citygaussian_v2",
            "scene": "gcp_100000_20260610",
            "protocol_id": "m3m_gcp_native_quarter_geometry_v2",
            "input_class": "rgb_colmap_external_geometry_prior",
            "access_boundary": {
                "isolated_dataset_root": str(dataset),
                "training_rgb_opened": 2196,
                "heldout_rgb_opened": 0,
                "gcp_annotations_opened": 0,
                "lidar_opened": 0,
                "only_training_rgb_and_train_only_colmap_supplied_to_prior_commands": True,
            },
            "claims": {"heldout_gcp_lidar_or_orthophoto_truth_used": False},
            "depth_outputs": depth_rows,
            "depth_scales": {
                "path": str(scales_path),
                "sha256": sha256_file(scales_path),
                "record_count": 2196,
            },
        })
        products = [manifest_path, *[dataset / row["relative_path"] for row in depth_rows], scales_path]
        success["products"] = build_phase_product_rows(products, phase="prior")
        success["canonical_sha256"] = canonical_sha256(success)
        write_json(evidence_root / "prior" / "phase_success.json", success)
        frozen_rows = [{"image_name": name} for name in names]
        with mock.patch("run_m3m_gcp_100k_guarded.load_frozen_train_rows", return_value=frozen_rows):
            validate_prior_phase_success(
                recipe=recipe,
                recipe_path=self.recipe,
                run_root=self.root / "run",
                dataset_root=dataset,
                prior_root=dataset,
                replacements=replacements,
            )
            (dataset / depth_rows[0]["relative_path"]).write_bytes(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "artifact changed"):
                validate_prior_phase_success(
                    recipe=recipe,
                    recipe_path=self.recipe,
                    run_root=self.root / "run",
                    dataset_root=dataset,
                    prior_root=dataset,
                    replacements=replacements,
                )

    @unittest.skipIf(os.name == "nt", "Windows test host lacks symlink privilege")
    def test_gsprior_normalized_input_binds_prepared_source_and_outputs(self) -> None:
        prepared = self.root / "prepared-gsprior"
        prepared_sparse = prepared / "sparse" / "0"
        prepared_images = prepared / "images"
        prepared_sparse.mkdir(parents=True)
        prepared_images.mkdir()
        binding = {"sparse_sha256": {}}
        for name, value in {
            "cameras.bin": "cam",
            "images.bin": "images",
            "points3D.ply": "points",
        }.items():
            path = prepared_sparse / name
            path.write_text(value, encoding="utf-8")
            binding["sparse_sha256"][name] = sha256_file(path)
        normalized = self.root / "normalized-gsprior"
        flat = normalized / "sparse"
        nested = flat / "0"
        nested.mkdir(parents=True)
        output_files = {}
        for name, value in {
            "cameras.bin": "cam",
            "images.bin": "normalized-images",
            "points3D.ply": "normalized-points",
        }.items():
            path = flat / name
            path.write_text(value, encoding="utf-8")
            (nested / name).symlink_to(Path("..") / name)
            output_files[name] = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        (normalized / "images").symlink_to(prepared_images, target_is_directory=True)
        manifest = {
            "schema": "m3m_gsprior_colmap_camera_normalization_v1",
            "status": "PASS",
            "source_scene": str(prepared),
            "reference_train_scene": str(prepared),
            "output_scene": str(normalized),
            "source": {
                "cameras_bin": {"sha256": binding["sparse_sha256"]["cameras.bin"]},
                "images_bin": {"sha256": binding["sparse_sha256"]["images.bin"]},
                "points3D_bin": None,
                "points3D_ply": {"sha256": binding["sparse_sha256"]["points3D.ply"]},
            },
            "output": {
                "files": output_files,
                "images_are_directory_symlink": True,
                "images_directory_target": str(prepared_images),
                "flat_and_sparse_zero_models_share_exact_files": True,
            },
            "validation": {
                "intrinsics_bytes_unchanged": True,
                "image_names_unchanged": True,
                "image_measurements_and_tracks_unchanged": True,
                "gcp_or_lidar_used": False,
                "image_pixels_resized_cropped_padded_or_reencoded": False,
            },
        }
        write_json(normalized / "normalization_manifest.json", manifest)
        validate_gsprior_normalized_input(
            prepared_root=prepared,
            normalized_root=normalized,
            binding=binding,
        )
        (flat / "images.bin").write_text("tampered", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "normalized output identity"):
            validate_gsprior_normalized_input(
                prepared_root=prepared,
                normalized_root=normalized,
                binding=binding,
            )

    def test_packet_phase_binds_immutable_ready_model_and_freeze(self) -> None:
        selected_run = (self.root / "selected-run").resolve()
        identity_root = (self.root / "model-identities").resolve()
        identity_root.mkdir()
        rows = []
        for method_id in METHOD_ORDER:
            asset_root = selected_run if method_id == "2dgs" else (self.root / "freeze-assets" / method_id).resolve()
            asset_root.mkdir(parents=True, exist_ok=True)
            model = (
                asset_root / "model" / "point_cloud" / "iteration_30000" / "point_cloud.ply"
                if method_id == "2dgs"
                else asset_root / "model.ply"
            )
            model.parent.mkdir(parents=True, exist_ok=True)
            if method_id == "2dgs":
                write_gaussian_ply(model)
            else:
                model.write_text(method_id, encoding="utf-8")
            inventory = [{
                "path": str(model.resolve()),
                "bytes": model.stat().st_size,
                "sha256": sha256_file(model),
            }]
            if method_id == "2dgs":
                recipe_payload = json.loads(self.recipe.read_text())
                marker = (
                    Path(recipe_payload["authorized_evidence_root"])
                    / "training" / "phase_success.json"
                )
                environment_path = marker.parent / "environment.json"
                write_success_environment(
                    environment_path, method_id="2dgs", phase="training"
                )
                marker_payload = {
                    "schema": "m3m_gcp_100k_phase_success_v2",
                    "status": "PASS",
                    "scene": "gcp_100000_20260610",
                    "method_id": "2dgs",
                    "phase": "training",
                    "recipe_sha256": sha256_file(self.recipe),
                    "command_sha256": command_sha256(
                        recipe_payload["phase_commands"]["training"]
                    ),
                    "frozen_budget": recipe_payload["budget"],
                    "environment_manifest_path": str(environment_path.resolve()),
                    "environment_manifest_sha256": sha256_file(environment_path),
                    "completion_evidence": {
                        "progress_unit": "iterations",
                        "last_valid_progress": 30000.0,
                        "required_product_postvalidation_passed": True,
                    },
                    "products": build_phase_product_rows(
                        [model], phase="training", method_id="2dgs"
                    ),
                    "ended_at_utc": "2026-08-21T00:00:00Z",
                }
                marker_payload["canonical_sha256"] = canonical_sha256(marker_payload)
                write_json(marker, marker_payload)
                inventory.append({
                    "path": str(marker.resolve()),
                    "bytes": marker.stat().st_size,
                    "sha256": sha256_file(marker),
                })
            identity = {
                "schema": "m3m_gcp_100k_model_identity_v1",
                "protocol_id": "m3m_gcp_lidar_rendered_surface_v1",
                "scene": "gcp_100000_20260610",
                "method_id": method_id,
                "run_root": str(asset_root),
                "inventory": sorted(inventory, key=lambda row: row["path"]),
            }
            identity["canonical_sha256"] = canonical_sha256(identity)
            identity_path = identity_root / f"{method_id}.json"
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
            run_root=selected_run, recipe=self.recipe, repo=self.repo,
        )
        frozen, freeze_sha = validate_packet_freeze_binding(
            args=args,
            recipe=json.loads(self.recipe.read_text()),
            plan=json.loads(self.plan.read_text()),
        )
        self.assertEqual(
            frozen["model_checkpoint_sha256"],
            sha256_file(identity_root / "2dgs.json"),
        )
        self.assertEqual(freeze_sha, sha256_file(freeze_path))
        model = selected_run / "model" / "point_cloud" / "iteration_30000" / "point_cloud.ply"
        model.write_text("tampered", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "model-identity file changed"):
            validate_packet_freeze_binding(
                args=args,
                recipe=json.loads(self.recipe.read_text()),
                plan=json.loads(self.plan.read_text()),
            )

    def test_packet_freeze_rejects_alternate_frozen_path(self) -> None:
        plan = json.loads(self.plan.read_text())
        alternate = self.root / "alternate-freeze.json"
        write_json(alternate, {})
        args = argparse.Namespace(
            scene_attempt_freeze=alternate,
            method_id="2dgs",
            run_root=self.root / "selected-run",
            recipe=self.recipe,
        )
        with self.assertRaisesRegex(RuntimeError, "differs from the frozen plan"):
            validate_packet_freeze_binding(
                args=args,
                recipe=json.loads(self.recipe.read_text()),
                plan=plan,
            )

    def test_model_identity_must_include_actual_final_model(self) -> None:
        run_root = (self.root / "identity-run").resolve()
        final_model = (
            run_root / "model" / "point_cloud" / "iteration_30000" / "point_cloud.ply"
        )
        write_gaussian_ply(final_model)
        decoy = run_root / "decoy.ply"
        decoy.write_bytes(b"decoy")
        payload = {
            "schema": "m3m_gcp_100k_model_identity_v1",
            "protocol_id": "m3m_gcp_lidar_rendered_surface_v1",
            "scene": "gcp_100000_20260610",
            "method_id": "2dgs",
            "run_root": str(run_root),
            "inventory": [{
                "path": str(decoy),
                "bytes": decoy.stat().st_size,
                "sha256": sha256_file(decoy),
            }],
        }
        payload["canonical_sha256"] = canonical_sha256(payload)
        identity = self.root / "decoy-identity.json"
        write_json(identity, payload)
        with self.assertRaisesRegex(RuntimeError, "omits the method's actual final model"):
            validate_model_identity_bundle(
                manifest_path=identity,
                method_id="2dgs",
                run_root=run_root,
                recipe={
                    "method_id": "2dgs",
                    "budget": {"type": "iterations", "value": 30000},
                },
                repo=self.repo,
            )

    def test_model_identity_requires_exact_training_success_marker(self) -> None:
        run_root = (self.root / "missing-marker-run").resolve()
        model = (
            run_root / "model" / "point_cloud" / "iteration_30000" / "point_cloud.ply"
        )
        write_gaussian_ply(model)
        payload = {
            "schema": "m3m_gcp_100k_model_identity_v1",
            "protocol_id": "m3m_gcp_lidar_rendered_surface_v1",
            "scene": "gcp_100000_20260610",
            "method_id": "2dgs",
            "run_root": str(run_root),
            "inventory": [{
                "path": str(model),
                "bytes": model.stat().st_size,
                "sha256": sha256_file(model),
            }],
        }
        payload["canonical_sha256"] = canonical_sha256(payload)
        identity = self.root / "missing-marker-identity.json"
        write_json(identity, payload)
        recipe = json.loads(self.recipe.read_text())
        recipe["_recipe_path"] = str(self.recipe)
        with self.assertRaisesRegex(RuntimeError, "phase-success marker inventory"):
            validate_model_identity_bundle(
                manifest_path=identity,
                method_id="2dgs",
                run_root=run_root,
                recipe=recipe,
                repo=self.repo,
            )

    def test_model_identity_rejects_wrong_training_command_hash(self) -> None:
        run_root = (self.root / "wrong-command-run").resolve()
        model = (
            run_root / "model" / "point_cloud" / "iteration_30000" / "point_cloud.ply"
        )
        write_gaussian_ply(model)
        recipe = json.loads(self.recipe.read_text())
        recipe["_recipe_path"] = str(self.recipe)
        marker = (
            Path(recipe["authorized_evidence_root"])
            / "training" / "phase_success.json"
        )
        environment_path = marker.parent / "environment.json"
        write_success_environment(
            environment_path, method_id="2dgs", phase="training"
        )
        marker_payload = {
            "schema": "m3m_gcp_100k_phase_success_v2",
            "status": "PASS",
            "scene": "gcp_100000_20260610",
            "method_id": "2dgs",
            "phase": "training",
            "recipe_sha256": sha256_file(self.recipe),
            "command_sha256": command_sha256(["wrong-command"]),
            "frozen_budget": recipe["budget"],
            "environment_manifest_path": str(environment_path.resolve()),
            "environment_manifest_sha256": sha256_file(environment_path),
            "completion_evidence": {
                "progress_unit": "iterations",
                "last_valid_progress": 30000.0,
                "required_product_postvalidation_passed": True,
            },
            "products": build_phase_product_rows(
                [model], phase="training", method_id="2dgs"
            ),
            "ended_at_utc": "2026-08-21T00:00:00Z",
        }
        marker_payload["canonical_sha256"] = canonical_sha256(marker_payload)
        write_json(marker, marker_payload)
        inventory = [
            {"path": str(path.resolve()), "bytes": path.stat().st_size,
             "sha256": sha256_file(path)}
            for path in (model, marker)
        ]
        payload = {
            "schema": "m3m_gcp_100k_model_identity_v1",
            "protocol_id": "m3m_gcp_lidar_rendered_surface_v1",
            "scene": "gcp_100000_20260610",
            "method_id": "2dgs",
            "run_root": str(run_root),
            "inventory": sorted(inventory, key=lambda row: row["path"]),
        }
        payload["canonical_sha256"] = canonical_sha256(payload)
        identity = self.root / "wrong-command-identity.json"
        write_json(identity, payload)
        with self.assertRaisesRegex(RuntimeError, "phase success identity mismatch"):
            validate_model_identity_bundle(
                manifest_path=identity,
                method_id="2dgs",
                run_root=run_root,
                recipe=recipe,
                repo=self.repo,
            )

    def test_packet_cumulative_cap_terminates_command_and_writes_failure(self) -> None:
        recipe = json.loads(self.recipe.read_text())
        failure = self.root / "evidence" / "failure.json"
        args = argparse.Namespace(
            phase="packet", capacity_root=self.root, packet_set_root=self.root / "packet-set",
            packet_state=self.root / "packet-state.json", method_id="2dgs", run_root=self.root / "run",
            failure_evidence=failure, progress_regex=None, progress_unit="views", poll_seconds=0.01,
            recipe=self.recipe, replacements={}, scene_attempt_freeze=self.root / "freeze.json",
            execution_plan={},
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
            dataset_root=self.root / "dataset",
            prior_root=self.root / "prior",
            failure_evidence=failure,
            progress_regex=None,
            progress_unit="iterations",
            poll_seconds=0.01,
            recipe=self.recipe,
            replacements={},
            scene_attempt_freeze=None,
        )
        recipe = json.loads(self.recipe.read_text())
        recipe["budget"] = {"type": "iterations", "value": 30000}
        point_cloud = (
            args.run_root / "model" / "point_cloud" / "iteration_30000" / "point_cloud.ply"
        )
        encoded_ply = base64.b64encode(gaussian_ply_bytes()).decode("ascii")
        command = [
            sys.executable,
            "-c",
            (
                "import base64; from pathlib import Path; "
                f"p=Path({str(point_cloud)!r}); p.parent.mkdir(parents=True); "
                f"p.write_bytes(base64.b64decode({encoded_ply!r}))"
            ),
        ]
        with mock.patch("run_m3m_gcp_100k_guarded.validate_capacity"):
            self.assertEqual(run_phase(args, recipe, command), 0)
        marker = failure.parent / "phase_success.json"
        payload = json.loads(marker.read_text())
        self.assertEqual(payload["status"], "PASS")
        self.assertEqual(payload["phase"], "training")
        self.assertEqual(payload["schema"], "m3m_gcp_100k_phase_success_v2")
        environment = Path(payload["environment_manifest_path"])
        self.assertEqual(sha256_file(environment), payload["environment_manifest_sha256"])
        environment_payload = json.loads(environment.read_text())
        self.assertEqual(
            environment_payload["resource_limits"]["parent_after"]["soft"], 65536
        )
        self.assertEqual(
            environment_payload["resource_limits"]["child_actual"]["soft"], 65536
        )
        with mock.patch("run_m3m_gcp_100k_guarded.validate_capacity"):
            with self.assertRaisesRegex(FileExistsError, "cannot be retried"):
                run_phase(args, recipe, command)

    def test_zero_exit_without_required_model_is_structured_failure(self) -> None:
        failure = self.root / "zero-output-evidence" / "failure.json"
        args = argparse.Namespace(
            phase="training",
            capacity_root=self.root,
            packet_set_root=None,
            packet_state=None,
            method_id="2dgs",
            run_root=self.root / "zero-output-run",
            dataset_root=self.root / "dataset",
            prior_root=self.root / "prior",
            failure_evidence=failure,
            progress_regex=None,
            progress_unit="iterations",
            poll_seconds=0.01,
            recipe=self.recipe,
            replacements={},
            scene_attempt_freeze=None,
        )
        recipe = json.loads(self.recipe.read_text())
        recipe["budget"] = {"type": "iterations", "value": 30000}
        with mock.patch("run_m3m_gcp_100k_guarded.validate_capacity"):
            code = run_phase(args, recipe, [sys.executable, "-c", "print('zero exit')"])
        self.assertNotEqual(code, 0)
        evidence = json.loads(failure.read_text())
        self.assertEqual(evidence["status"], "FAILED_UNRANKED")
        self.assertIn("required outputs did not validate", " ".join(evidence["errors"]))

    def test_zero_exit_with_fake_ply_at_frozen_final_path_is_structured_failure(self) -> None:
        failure = self.root / "fake-ply-evidence" / "failure.json"
        run_root = self.root / "fake-ply-run"
        point_cloud = run_root / "model" / "point_cloud" / "iteration_30000" / "point_cloud.ply"
        args = argparse.Namespace(
            phase="training", capacity_root=self.root, packet_set_root=None,
            packet_state=None, method_id="2dgs", run_root=run_root,
            dataset_root=self.root / "dataset", prior_root=self.root / "prior",
            failure_evidence=failure, progress_regex=None, progress_unit="iterations",
            poll_seconds=0.01, recipe=self.recipe, replacements={},
            scene_attempt_freeze=None,
        )
        recipe = json.loads(self.recipe.read_text())
        recipe["budget"] = {"type": "iterations", "value": 30000}
        command = [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; "
                f"p=Path({str(point_cloud)!r}); p.parent.mkdir(parents=True); "
                "p.write_bytes(b'not-a-ply-model')"
            ),
        ]
        with mock.patch("run_m3m_gcp_100k_guarded.validate_capacity"):
            code = run_phase(args, recipe, command)
        self.assertNotEqual(code, 0)
        evidence = json.loads(failure.read_text())
        self.assertEqual(evidence["status"], "FAILED_UNRANKED")
        self.assertIn("Gaussian PLY", " ".join(evidence["errors"]))

    def test_preexisting_training_run_root_is_rejected_before_child(self) -> None:
        run_root = self.root / "stale-run"
        point_cloud = run_root / "model" / "point_cloud" / "iteration_30000" / "point_cloud.ply"
        write_gaussian_ply(point_cloud)
        failure = self.root / "stale-evidence" / "failure.json"
        args = argparse.Namespace(
            phase="training", capacity_root=self.root, packet_set_root=None,
            packet_state=None, method_id="2dgs", run_root=run_root,
            dataset_root=self.root / "dataset", prior_root=self.root / "prior",
            failure_evidence=failure, progress_regex=None, progress_unit="iterations",
            poll_seconds=0.01, recipe=self.recipe, replacements={},
            scene_attempt_freeze=None,
        )
        recipe = json.loads(self.recipe.read_text())
        recipe["budget"] = {"type": "iterations", "value": 30000}
        with mock.patch("run_m3m_gcp_100k_guarded.validate_capacity"):
            with self.assertRaisesRegex(FileExistsError, "absent method run root"):
                run_phase(args, recipe, [sys.executable, "-c", "print('must not start')"])
        self.assertFalse(failure.exists())
        self.assertFalse((failure.parent / "command.stdout.log").exists())


if __name__ == "__main__":
    unittest.main()
