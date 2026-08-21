#!/usr/bin/env python3
"""Static integrity tests for the exact 100K review candidate."""

from __future__ import annotations

import hashlib
import json
import re
import unittest
from pathlib import Path, PurePosixPath
from typing import Any


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
            ROOT / "configs/m3m_gcp_native_quarter_100k_recipe_manifest_v1.json"
        )
        cls.manifest = json.loads(cls.manifest_path.read_text(encoding="utf-8"))
        cls.plan = json.loads(
            (
                ROOT
                / "configs/m3m_gcp_native_quarter_100k_ten_method_execution_plan_v1.json"
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
        self.assertEqual([row["method_id"] for row in self.manifest["recipes"]], METHOD_ORDER)
        self.assertEqual(list(self.recipes), METHOD_ORDER)
        self.assertEqual(self.manifest["canonical_sha256"], canonical_sha256(self.manifest))

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
            binding = recipe["prepared_method_input_binding"]
            self.assertIsNotNone(HEX64.fullmatch(binding["evidence_sha256"]), method)
            self.assertTrue(binding["all_image_sfm_precedes_train_test_split"], method)
            evidence_shas.add(binding["evidence_sha256"])
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
        self.assertEqual(len(evidence_shas), 1)
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

    def test_plan_scope_storage_review_and_attempt_freeze(self) -> None:
        plan = self.plan
        self.assertEqual(plan["method_order"], METHOD_ORDER)
        self.assertEqual(plan["scene"], "gcp_100000_20260610")
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
            self.assertIn("once the child starts every exit is final", recipe["retry_policy"])


if __name__ == "__main__":
    unittest.main()
