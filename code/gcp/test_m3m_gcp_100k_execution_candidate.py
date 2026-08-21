#!/usr/bin/env python3
"""Static integrity tests for the exact 100K review candidate."""

from __future__ import annotations

import hashlib
import json
import re
import unittest
from pathlib import Path, PurePosixPath
from typing import Any

from m3m_gcp_100k_source_binding_correction import (
    LINUX_HEADER_SHA,
    WINDOWS_HEADER_SHA,
    validate_source_binding_correction,
)


ROOT = Path(__file__).resolve().parents[2]
METHOD_ORDER = [
    "3dgs_original",
    "2dgs",
    "pgsr",
    "rade_gs",
    "qgs",
    "gsprior",
    "sof",
    "citygaussian_v2",
    "citygs_x",
    "metrogs",
]
LOCKED_SCENES = [
    "gcp_5000_20260602",
    "gcp_20000_20260602",
    "gcp_10000_20260610",
    "gcp_50000_20260610",
]
HEX64 = re.compile(r"[0-9a-f]{64}")
HEX40 = re.compile(r"[0-9a-f]{40}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: dict[str, Any]) -> str:
    clean = {key: value for key, value in payload.items() if key != "canonical_sha256"}
    raw = json.dumps(clean, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class ExecutionCandidateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest_path = (
            ROOT / "configs/m3m_gcp_native_quarter_100k_recipe_manifest_v3.json"
        )
        cls.manifest = json.loads(cls.manifest_path.read_text(encoding="utf-8"))
        cls.old_manifest = json.loads(
            (ROOT / "configs/m3m_gcp_native_quarter_100k_recipe_manifest_v2.json")
            .read_text(encoding="utf-8")
        )
        cls.plan = json.loads(
            (
                ROOT
                / "configs/m3m_gcp_native_quarter_100k_ten_method_execution_plan_v3.json"
            ).read_text(encoding="utf-8")
        )
        cls.recipes: dict[str, dict[str, Any]] = {}
        for row in cls.manifest["recipes"]:
            path = ROOT / row["path"]
            payload = json.loads(path.read_text(encoding="utf-8"))
            cls.recipes[row["method_id"]] = payload
            if sha256_file(path) != row["sha256"]:
                raise AssertionError(f"recipe file SHA mismatch: {row['method_id']}")
            if canonical_sha256(payload) != row["canonical_sha256"]:
                raise AssertionError(f"recipe canonical SHA mismatch: {row['method_id']}")

    def test_exact_method_pool_and_phase_cardinality(self) -> None:
        self.assertEqual(self.manifest["method_order"], METHOD_ORDER)
        self.assertEqual(
            self.manifest["schema"],
            "m3m_gcp_native_quarter_100k_recipe_manifest_v3",
        )
        self.assertEqual([row["method_id"] for row in self.manifest["recipes"]], METHOD_ORDER)
        self.assertEqual(list(self.recipes), METHOD_ORDER)
        self.assertEqual(self.manifest["canonical_sha256"], canonical_sha256(self.manifest))
        self.assertEqual(self.manifest["recipes"][1:], self.old_manifest["recipes"][1:])
        self.assertEqual(
            self.manifest["correction_scope"]["changed_method_ids"],
            ["3dgs_original"],
        )

        training = [
            method for method, recipe in self.recipes.items()
            if "training" in recipe["phase_commands"]
        ]
        packet = [
            method for method, recipe in self.recipes.items()
            if "packet" in recipe["phase_commands"]
        ]
        self.assertEqual(training, METHOD_ORDER[1:])
        self.assertEqual(packet, METHOD_ORDER)
        reuse = self.recipes["3dgs_original"]
        self.assertEqual(list(reuse["phase_commands"]), ["packet"])
        self.assertFalse(reuse["reuse_model_binding"]["retrain_allowed"])
        self.assertEqual(
            reuse["reuse_model_binding"]["point_cloud_sha256"],
            "8d92360186d268d0e20a0e328122e8c2679cddd0c2d539c27a918ee4c972e1f5",
        )

    def test_per_method_inputs_and_adapter_bindings_are_final(self) -> None:
        evidence_shas: set[str] = set()
        camera_evidence_shas: set[str] = set()
        for method in METHOD_ORDER:
            recipe = self.recipes[method]
            adapter = ROOT / recipe["renderer_adapter_path"]
            self.assertTrue(adapter.is_file(), method)
            self.assertEqual(sha256_file(adapter), recipe["renderer_adapter_sha256"], method)
            self.assertTrue(PurePosixPath(recipe["authorized_run_root"]).is_absolute(), method)
            self.assertTrue(
                PurePosixPath(recipe["authorized_evidence_root"]).is_absolute(), method
            )
            self.assertTrue(
                PurePosixPath(recipe["authorized_packet_set_root"]).is_absolute(), method
            )
            self.assertTrue(
                PurePosixPath(recipe["authorized_packet_state"]).is_absolute(), method
            )
            self.assertTrue(
                recipe["authorized_packet_set_root"].endswith(f"/{method}"), method
            )
            self.assertEqual(
                recipe["schema"],
                (
                    "m3m_gcp_native_quarter_100k_execution_recipe_v3"
                    if method == "3dgs_original"
                    else "m3m_gcp_native_quarter_100k_execution_recipe_v2"
                ),
            )
            self.assertEqual(
                recipe["process_resource_limits"],
                {
                    "applies_to_phases": ["prior", "training", "packet"],
                    "rlimit_nofile_hard_minimum": 65536,
                    "rlimit_nofile_soft": 65536,
                    "record_parent_before_after": True,
                    "record_child_actual_inheritance": True,
                },
                method,
            )
            binding = recipe["prepared_method_input_binding"]
            self.assertIsNotNone(HEX64.fullmatch(binding["evidence_sha256"]), method)
            self.assertTrue(binding["all_image_sfm_precedes_train_test_split"], method)
            evidence_shas.add(binding["evidence_sha256"])
            camera_binding = recipe["evaluation_camera_root_binding"]
            camera_evidence_shas.add(camera_binding["evidence_sha256"])
            self.assertEqual(camera_binding["view_count"], 2196)
            self.assertEqual(camera_binding["points3d_bin_point_count"], 0)
            self.assertEqual(
                set(camera_binding["sparse_sha256"]),
                {"cameras.bin", "images.bin", "points3D.bin", "points3D.ply"},
            )
            packet_command = recipe["phase_commands"]["packet"]
            self.assertEqual(
                packet_command[packet_command.index("--camera-root") + 1],
                camera_binding["root"],
            )
            self.assertIn(
                "code/gcp/materialize_m3m_gcp_100k_evaluation_camera_root.py",
                recipe["benchmark_required_files_sha256"],
            )
            expected_sparse = {"cameras.bin", "images.bin", "points3D.ply"}
            if method in {"citygaussian_v2", "citygs_x", "metrogs"}:
                expected_sparse.add("points3D.bin")
            self.assertEqual(set(binding["sparse_sha256"]), expected_sparse, method)
            for digest in binding["sparse_sha256"].values():
                self.assertIsNotNone(HEX64.fullmatch(digest), method)
            self.assertIn(
                "code/gcp/materialize_m3m_gcp_100k_method_inputs.py",
                recipe["benchmark_required_files_sha256"],
                method,
            )
            if method in {"citygaussian_v2", "citygs_x"}:
                self.assertEqual(
                    binding["input_profile"],
                    "city_train_records_with_full_all_image_sfm_points",
                )
                self.assertIn(
                    "code/gcp/materialize_colmap_train_track_compatibility_streaming.py",
                    recipe["benchmark_required_files_sha256"],
                )
            elif method == "metrogs":
                self.assertEqual(
                    binding["input_profile"],
                    "metrogs_reciprocal_train_track_closure_after_all_image_sfm",
                )
                self.assertIn(
                    "code/gcp/filter_colmap_model_to_frozen_train_streaming.py",
                    recipe["benchmark_required_files_sha256"],
                )
            else:
                self.assertEqual(
                    binding["input_profile"],
                    "exact_formal_train_view_from_shared_all_image_sfm",
                )
            if method == "gsprior":
                self.assertNotEqual(
                    binding["dataset_root"],
                    recipe["phase_roots"]["training"]["dataset_root"],
                )
                prior = recipe["phase_commands"]["prior"]
                self.assertEqual(
                    prior[prior.index("--source_scene") + 1],
                    binding["dataset_root"],
                )
                self.assertEqual(
                    prior[prior.index("--output_scene") + 1],
                    "{dataset_root}",
                )
        self.assertEqual(len(evidence_shas), 1)
        self.assertEqual(len(camera_evidence_shas), 1)
        self.assertEqual(
            len({recipe["authorized_packet_state"] for recipe in self.recipes.values()}),
            1,
        )
        citygs = self.recipes["citygs_x"]
        citygs_commands = json.dumps(citygs["phase_commands"], sort_keys=True)
        self.assertNotIn("batch-20260818", citygs_commands)
        self.assertIn("{repo}/compat/citygs_x/pytorch3d_transforms_minimal_v1", citygs_commands)
        self.assertIn(
            "compat/citygs_x/pytorch3d_transforms_minimal_v1/pytorch3d/transforms/__init__.py",
            citygs["benchmark_required_files_sha256"],
        )
        reuse = self.recipes["3dgs_original"]
        proof_relative = (
            "docs/protocol_evidence/"
            "3dgs_native_quarter_adapter_linux_identity_proof_v1.json"
        )
        self.assertIn(proof_relative, reuse["benchmark_required_files_sha256"])
        patched = reuse["source_bindings"]["packet"]["required_files_sha256"]
        self.assertEqual(len(patched), 8)
        self.assertEqual(
            patched[
                "submodules/diff-gaussian-rasterization/rasterize_points.h"
            ],
            LINUX_HEADER_SHA,
        )
        self.assertNotIn(WINDOWS_HEADER_SHA, patched.values())
        self.assertFalse(reuse["source_identity_correction"]["dual_hash_tolerance"])

    def test_training_and_prior_commands_do_not_open_benchmark_truth(self) -> None:
        forbidden = ("heldout", "lidar", "surveyed", "annotation", "gcp.json")
        for method, recipe in self.recipes.items():
            for phase in ("training", "prior"):
                command = " ".join(recipe["phase_commands"].get(phase, [])).lower()
                for token in forbidden:
                    self.assertNotIn(token, command, f"{method}:{phase}:{token}")
        truth_deny = self.plan["truth_deny"]
        self.assertFalse(truth_deny["heldout_rgb_visible_to_training"])
        self.assertFalse(truth_deny["gcp_visible_to_training_prior_or_selection"])
        self.assertFalse(truth_deny["lidar_visible_to_training_prior_or_selection"])
        self.assertFalse(truth_deny["result_driven_retry_or_recipe_change"])

    def test_all_ten_method_phase_source_status_bindings_are_complete(self) -> None:
        expected_phases = {
            "3dgs_original": {"packet"},
            "2dgs": {"training", "packet"},
            "pgsr": {"training", "packet"},
            "rade_gs": {"training", "packet"},
            "qgs": {"training", "packet"},
            "gsprior": {"prior", "training", "packet"},
            "sof": {"training", "packet"},
            "citygaussian_v2": {"prior", "training", "packet"},
            "citygs_x": {"prior", "training", "packet"},
            "metrogs": {"prior", "training", "packet"},
        }
        saw_clean = False
        saw_worktree = False
        saw_multiline = False
        for method in METHOD_ORDER:
            bindings = self.recipes[method]["source_bindings"]
            self.assertEqual(set(bindings), expected_phases[method], method)
            for phase, binding in bindings.items():
                status = binding["required_status"]
                self.assertIsInstance(status, str, f"{method}:{phase}")
                self.assertNotIn("\r", status, f"{method}:{phase}")
                self.assertEqual(status.rstrip("\r\n"), status, f"{method}:{phase}")
                saw_clean |= status == ""
                saw_worktree |= status.startswith(" M ")
                saw_multiline |= "\n" in status
                self.assertTrue(PurePosixPath(binding["root"]).is_absolute())
                self.assertIsNotNone(HEX40.fullmatch(binding["commit"]))
                self.assertIsNotNone(HEX40.fullmatch(binding["tree"]))
        self.assertTrue(saw_clean)
        self.assertTrue(saw_worktree)
        self.assertTrue(saw_multiline)

    def test_plan_scope_storage_review_and_attempt_freeze(self) -> None:
        plan = self.plan
        self.assertEqual(plan["method_order"], METHOD_ORDER)
        self.assertEqual(plan["scene"], "gcp_100000_20260610")
        self.assertEqual(
            plan["schema"],
            "m3m_gcp_native_quarter_100k_ten_method_execution_plan_v3",
        )
        self.assertEqual(
            plan["activation_manifest_path"],
            "/root/autodl-tmp/runs/m3m-gcp-native-quarter/formal-100k-v2/activation_v3.json",
        )
        self.assertEqual(plan["other_prepared_scenes_locked"], LOCKED_SCENES)
        self.assertFalse(
            plan["other_prepared_scene_training_rendering_or_formal_evaluation_authorized"]
        )
        self.assertFalse(plan["execution_authorized"])
        self.assertEqual(plan["recipe_manifest"]["file_sha256"], sha256_file(self.manifest_path))
        self.assertEqual(
            plan["recipe_manifest"]["canonical_sha256"],
            self.manifest["canonical_sha256"],
        )
        self.assertEqual(plan["preparation"]["per_method_input_evidence"]["sha256"], next(
            iter({
                self.recipes[method]["prepared_method_input_binding"]["evidence_sha256"]
                for method in METHOD_ORDER
            })
        ))
        self.assertTrue(
            plan["preparation"]["per_method_input_evidence"]
            ["all_image_sfm_precedes_train_test_split"]
        )
        camera = plan["preparation"]["evaluation_camera_root"]
        self.assertEqual(camera["view_count"], 2196)
        self.assertEqual(camera["points3d_bin_point_count"], 0)
        self.assertFalse(camera["points2d_tracks_present"])
        self.assertEqual(
            camera["evidence_sha256"],
            next(iter({
                self.recipes[method]["evaluation_camera_root_binding"]["evidence_sha256"]
                for method in METHOD_ORDER
            })),
        )
        storage = plan["storage"]
        self.assertEqual(storage["packet_scratch_hard_cap_gib"], 100)
        self.assertEqual(storage["minimum_free_before_prior_gib"], 300)
        self.assertEqual(storage["minimum_free_before_training_gib"], 300)
        self.assertEqual(storage["minimum_free_before_packet_export_gib"], 180)
        self.assertTrue(storage["all_ten_packet_sets_simultaneously_forbidden"])
        lifecycle = plan["rolling_packet_lifecycle"]
        self.assertEqual(lifecycle["simultaneous_raw_packet_sets_max"], 1)
        self.assertTrue(lifecycle["independent_verification_required_before_packet_deletion"])
        self.assertTrue(
            lifecycle["full_archive_inventory_byte_reverification_required_before_packet_deletion"]
        )
        self.assertTrue(plan["attempt_freeze"]["frozen_before_any_formal_lidar_result"])
        self.assertEqual(
            plan["attempt_freeze"]["execution_plan_path"],
            "configs/m3m_gcp_native_quarter_100k_ten_method_execution_plan_v3.json",
        )
        self.assertEqual(
            plan["attempt_freeze"]["recipe_manifest_path"],
            "configs/m3m_gcp_native_quarter_100k_recipe_manifest_v3.json",
        )
        self.assertEqual(
            plan["attempt_freeze"]["method_registry_path"],
            "configs/m3m_gcp_native_quarter_method_registry_v3.json",
        )
        closure = plan["execution_closure"]
        self.assertEqual(
            closure["phase_product_validator"]["path"],
            "code/gcp/m3m_gcp_100k_phase_products.py",
        )
        self.assertTrue(closure["zero_exit_requires_phase_product_postvalidation"])
        self.assertTrue(
            closure["prior_and_training_require_absent_run_root_at_guard_admission"]
        )
        self.assertTrue(
            closure["training_child_must_create_products_inside_new_run_root"]
        )
        self.assertTrue(
            closure["prior_phase_success_and_product_required_before_training"]
        )
        self.assertTrue(
            closure["ready_model_identity_requires_exact_phase_success_markers"]
        )
        self.assertTrue(
            closure["phase_success_command_rehashed_against_frozen_recipe"]
        )
        self.assertEqual(
            closure["rlimit_nofile_soft_required_for_child_phases"], 65536
        )
        self.assertEqual(
            closure["rlimit_nofile_hard_minimum_prechild_gate"], 65536
        )
        self.assertTrue(
            closure["rlimit_nofile_parent_before_after_evidence_required"]
        )
        self.assertTrue(
            closure["rlimit_nofile_child_actual_inheritance_evidence_required"]
        )
        self.assertEqual(
            closure["source_binding_correction_validator"]["path"],
            "code/gcp/m3m_gcp_100k_source_binding_correction.py",
        )
        correction = validate_source_binding_correction(repo=ROOT, plan=plan)
        self.assertEqual(
            correction["classification"]["type"],
            "LINUX_IDENTITY_METADATA_CORRECTION_ONLY",
        )
        continuity = plan["activation_continuity"]
        continuity_path = ROOT / continuity["receipt"]["path"]
        self.assertEqual(
            sha256_file(continuity_path), continuity["receipt"]["sha256"]
        )
        continuity_receipt = json.loads(
            continuity_path.read_text(encoding="utf-8")
        )
        self.assertEqual(
            continuity_receipt["canonical_sha256"],
            canonical_sha256(continuity_receipt),
        )
        self.assertEqual(
            continuity_receipt["status"], "SEALED_V2_TO_V3_CONTINUITY"
        )
        self.assertEqual(
            continuity["inherited_final_methods_forbidden_to_launch"], ["2dgs"]
        )
        self.assertFalse(continuity["pgsr_prechild_rejection_consumed_attempt"])
        previous_plan = ROOT / continuity["previous_execution_plan"]["path"]
        self.assertEqual(
            previous_plan.stat().st_size,
            continuity["previous_execution_plan"]["bytes"],
        )
        self.assertEqual(
            sha256_file(previous_plan),
            continuity["previous_execution_plan"]["sha256"],
        )
        self.assertTrue(plan["preparation"]["formal_training_started"])
        supersession = plan["superseded_activation"]
        receipt_path = ROOT / supersession["receipt"]["path"]
        self.assertEqual(sha256_file(receipt_path), supersession["receipt"]["sha256"])
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        self.assertEqual(receipt["canonical_sha256"], canonical_sha256(receipt))
        self.assertEqual(
            receipt["status"], "SUPERSEDED_INFRASTRUCTURE_INVALID_NOT_RANKABLE"
        )
        self.assertFalse(supersession["algorithm_failure"])
        self.assertFalse(supersession["formal_retry_counted"])
        self.assertFalse(supersession["rankable"])
        self.assertEqual(
            plan["formal_lidar_protocol"]["phase1_review"]["verdict"],
            "PASS_LIDAR_V1_AND_SIX_SCENE_PREPARATION_V2",
        )
        self.assertEqual(
            plan["review"]["required_pass_verdict"],
            "PASS_100K_TIME_SPACE_EXECUTION_PLAN_V1",
        )
        self.assertEqual(plan["review"]["verdict"], "PENDING")
        self.assertEqual(
            plan["retry_policy"]["child_started_any_exit_including_zero_progress_or_oom"],
            "final for that method",
        )
        self.assertNotIn("pre_child_or_zero_optimizer_progress", plan["retry_policy"])
        for recipe in self.recipes.values():
            self.assertIn(
                "code/gcp/m3m_gcp_100k_phase_products.py",
                recipe["benchmark_required_files_sha256"],
            )
            self.assertIn("once the child starts every exit is final", recipe["retry_policy"])
            fresh = recipe["fresh_run_root_policy"]
            self.assertTrue(
                fresh["prior_and_training_require_absent_run_root_at_guard_admission"]
            )
            self.assertTrue(fresh["prior_must_not_create_run_root"])
            self.assertTrue(
                fresh["training_guard_exclusively_creates_empty_run_root_before_child"]
            )
            self.assertTrue(fresh["training_child_must_create_final_products"])
            self.assertFalse(recipe.get("materializations", {}).get("prior", []))
            serialized = json.dumps(recipe, sort_keys=True)
            self.assertNotIn(
                "/formal-100k/gcp_100000_20260610/", serialized,
                recipe["method_id"],
            )


if __name__ == "__main__":
    unittest.main()
