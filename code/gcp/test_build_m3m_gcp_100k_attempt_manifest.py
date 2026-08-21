#!/usr/bin/env python3
"""Model-identity inventory tests for the frozen 100K attempt builder."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from build_m3m_gcp_100k_attempt_manifest import (
    phase_success_inventory,
    success_inventory,
    validate_frozen_attempt_paths,
)
from freeze_m3m_gcp_lidar_scene_attempts import validate_frozen_100k_paths
from m3m_gcp_lidar_artifacts import canonical_sha256, sha256_file


class AttemptManifestBuilderTest(unittest.TestCase):
    def test_frozen_attempt_and_freeze_paths_reject_alternates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory).resolve()
            configs = repo / "configs"
            configs.mkdir()
            plan_path = configs / "plan.json"
            recipe_manifest = configs / "recipes.json"
            registry = configs / "registry.json"
            identity_root = repo / "formal" / "identities"
            methods = repo / "formal" / "methods.json"
            freeze_output = repo / "formal" / "freeze.json"
            recipe_manifest.write_text("{}", encoding="utf-8")
            registry.write_text("{}", encoding="utf-8")
            plan = {
                "schema": "m3m_gcp_native_quarter_100k_ten_method_execution_plan_v1",
                "scene": "gcp_100000_20260610",
                "execution_authorized": False,
                "attempt_freeze": {
                    "execution_plan_path": "configs/plan.json",
                    "recipe_manifest_path": "configs/recipes.json",
                    "method_registry_path": "configs/registry.json",
                    "model_identity_root": str(identity_root),
                    "attempt_manifest_path": str(methods),
                    "scene_attempt_freeze_path": str(freeze_output),
                },
            }
            plan["canonical_sha256"] = canonical_sha256(plan)
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            validate_frozen_attempt_paths(
                repo=repo,
                plan_path=plan_path,
                recipe_manifest_path=recipe_manifest,
                registry_path=registry,
                identity_root=identity_root,
                output=methods,
            )
            validate_frozen_100k_paths(
                execution_plan=plan_path,
                methods_path=methods,
                output=freeze_output,
                scene="gcp_100000_20260610",
            )
            with self.assertRaisesRegex(RuntimeError, "output path differs"):
                validate_frozen_attempt_paths(
                    repo=repo,
                    plan_path=plan_path,
                    recipe_manifest_path=recipe_manifest,
                    registry_path=registry,
                    identity_root=identity_root,
                    output=repo / "alternate-methods.json",
                )
            with self.assertRaisesRegex(RuntimeError, "freeze output path differs"):
                validate_frozen_100k_paths(
                    execution_plan=plan_path,
                    methods_path=methods,
                    output=repo / "alternate-freeze.json",
                    scene="gcp_100000_20260610",
                )

    def test_phase_success_marker_is_exactly_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "phase_success.json"
            payload = {
                "schema": "m3m_gcp_100k_phase_success_v1",
                "status": "PASS",
                "scene": "gcp_100000_20260610",
                "method_id": "2dgs",
                "phase": "training",
                "recipe_sha256": "a" * 64,
                "command_sha256": "b" * 64,
                "ended_at_utc": "2026-08-21T00:00:00Z",
            }
            payload["canonical_sha256"] = canonical_sha256(payload)
            path.write_text(json.dumps(payload), encoding="utf-8")
            row = phase_success_inventory(
                path,
                method_id="2dgs",
                phase="training",
                recipe_sha256="a" * 64,
            )
            self.assertEqual(row["sha256"], sha256_file(path))

    def test_plain_gaussian_binds_ply_and_cfg_args(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory).resolve()
            point_cloud = run_root / "model" / "point_cloud" / "iteration_30000" / "point_cloud.ply"
            point_cloud.parent.mkdir(parents=True)
            point_cloud.write_bytes(b"ply")
            cfg = run_root / "model" / "cfg_args"
            cfg.write_text("cfg", encoding="utf-8")
            rows = success_inventory("2dgs", {
                "authorized_run_root": str(run_root),
                "budget": {"type": "iterations", "value": 30000},
            })
            self.assertEqual({Path(row["path"]).name for row in rows}, {"point_cloud.ply", "cfg_args"})
            self.assertEqual(next(row for row in rows if row["path"].endswith("point_cloud.ply"))["sha256"], sha256_file(point_cloud))

    def test_citygs_x_binds_all_packet_required_checkpoint_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory).resolve()
            model = run_root / "model"
            checkpoint = model / "point_cloud" / "iteration_100000"
            checkpoint.mkdir(parents=True)
            point_cloud = checkpoint / "point_cloud.ply"
            point_cloud.write_bytes(b"ply")
            (checkpoint / "additional_attributes.npz").write_bytes(b"attrs")
            (checkpoint / "checkpoints.pth").write_bytes(b"state")
            prior_root = run_root / "prior"
            prior_root.mkdir()
            prior_manifest = prior_root / "depth_and_multiview_prior_v1.json"
            prior_manifest.write_text("{}", encoding="utf-8")
            summary = {
                "method_id": "citygs_x", "status": "TRAINING_PASS", "mode": "formal",
                "iterations": 100000,
                "checkpoint": {
                    "path": str(checkpoint), "point_cloud_file": point_cloud.name,
                    "point_cloud_sha256": sha256_file(point_cloud),
                },
            }
            summary_path = model / "training_wrapper_summary.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            rows = success_inventory("citygs_x", {
                "authorized_run_root": str(run_root),
                "budget": {"type": "iterations", "value": 100000},
                "phase_roots": {"prior": {"prior_root": str(prior_root)}},
            })
            self.assertEqual(
                {Path(row["path"]).name for row in rows},
                {
                    "training_wrapper_summary.json",
                    "point_cloud.ply",
                    "additional_attributes.npz",
                    "checkpoints.pth",
                    "depth_and_multiview_prior_v1.json",
                },
            )


if __name__ == "__main__":
    unittest.main()
