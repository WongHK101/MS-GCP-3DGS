#!/usr/bin/env python3
"""Model-identity inventory tests for the frozen 100K attempt builder."""

from __future__ import annotations

import json
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path

from build_m3m_gcp_100k_attempt_manifest import (
    phase_success_inventory,
    success_inventory,
    validate_frozen_attempt_paths,
)
from freeze_m3m_gcp_lidar_scene_attempts import validate_frozen_100k_paths
from m3m_gcp_lidar_artifacts import canonical_sha256, sha256_file
from m3m_gcp_100k_phase_products import phase_product_row


def write_gaussian_ply(path: Path, *, method_id: str = "2dgs") -> None:
    if method_id == "citygs_x":
        names = [
            "x", "y", "z", "level", "extra_level", "info",
            *[f"f_offset_{index}" for index in range(30)],
            *[f"f_anchor_feat_{index}" for index in range(32)],
            "opacity", *[f"scale_{index}" for index in range(6)],
            "rot_0", "rot_1", "rot_2", "rot_3",
        ]
    else:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + struct.pack(f"<{len(names)}f", *([0.0] * len(names))))


def write_torch_checkpoint(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", b"pickle")
        archive.writestr("archive/version", b"3\n")
        archive.writestr("archive/data/0", b"0" * 2048)


def write_npz(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("attributes.npy", b"npy")


def bind_environment(
    payload: dict, root: Path, *, method_id: str = "2dgs", phase: str = "training"
) -> None:
    environment_path = root / "environment.json"
    environment = {
        "schema": "m3m_gcp_100k_execution_environment_v2",
        "scene": "gcp_100000_20260610",
        "method_id": method_id,
        "phase": phase,
        "resource_limits": {
            "resource": "RLIMIT_NOFILE",
            "required_soft": 65536,
            "hard_minimum": 65536,
            "parent_before": {"soft": 1024, "hard": 1048576},
            "parent_after": {"soft": 65536, "hard": 1048576},
            "child_actual": {"soft": 65536, "hard": 1048576},
        },
    }
    environment["canonical_sha256"] = canonical_sha256(environment)
    environment_path.write_text(json.dumps(environment), encoding="utf-8")
    payload["environment_manifest_path"] = str(environment_path.resolve())
    payload["environment_manifest_sha256"] = sha256_file(environment_path)


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
                "schema": "m3m_gcp_native_quarter_100k_ten_method_execution_plan_v2",
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
                "schema": "m3m_gcp_100k_phase_success_v2",
                "status": "PASS",
                "scene": "gcp_100000_20260610",
                "method_id": "2dgs",
                "phase": "training",
                "recipe_sha256": "a" * 64,
                "command_sha256": "b" * 64,
                "frozen_budget": {},
                "completion_evidence": {
                    "progress_unit": "iterations",
                    "last_valid_progress": 30000.0,
                    "required_product_postvalidation_passed": True,
                },
                "products": [],
                "ended_at_utc": "2026-08-21T00:00:00Z",
            }
            product = Path(directory).resolve() / "product.json"
            product.write_text("{}", encoding="utf-8")
            payload["products"] = [
                phase_product_row(product, validate_model_container=False)
            ]
            bind_environment(payload, Path(directory).resolve())
            payload["canonical_sha256"] = canonical_sha256(payload)
            path.write_text(json.dumps(payload), encoding="utf-8")
            row = phase_success_inventory(
                path,
                method_id="2dgs",
                phase="training",
                recipe_sha256="a" * 64,
                expected_command_sha256="b" * 64,
            )
            self.assertEqual(row["sha256"], sha256_file(path))
            payload["command_sha256"] = "c" * 64
            payload["canonical_sha256"] = canonical_sha256(payload)
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "phase success identity mismatch"):
                phase_success_inventory(
                    path,
                    method_id="2dgs",
                    phase="training",
                    recipe_sha256="a" * 64,
                    expected_command_sha256="b" * 64,
                )

    def test_plain_gaussian_binds_ply_and_cfg_args(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory).resolve()
            point_cloud = run_root / "model" / "point_cloud" / "iteration_30000" / "point_cloud.ply"
            point_cloud.parent.mkdir(parents=True)
            write_gaussian_ply(point_cloud)
            cfg = run_root / "model" / "cfg_args"
            cfg.write_text("cfg", encoding="utf-8")
            rows = success_inventory("2dgs", {
                "authorized_run_root": str(run_root),
                "budget": {"type": "iterations", "value": 30000},
            })
            self.assertEqual({Path(row["path"]).name for row in rows}, {"point_cloud.ply", "cfg_args"})
            self.assertEqual(next(row for row in rows if row["path"].endswith("point_cloud.ply"))["sha256"], sha256_file(point_cloud))

    def test_training_phase_success_cannot_bind_a_decoy_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            final_model = root / "point_cloud.ply"
            decoy_model = root / "decoy.ply"
            write_gaussian_ply(final_model)
            write_gaussian_ply(decoy_model)
            marker = root / "phase_success.json"
            payload = {
                "schema": "m3m_gcp_100k_phase_success_v2",
                "status": "PASS",
                "scene": "gcp_100000_20260610",
                "method_id": "2dgs",
                "phase": "training",
                "recipe_sha256": "a" * 64,
                "command_sha256": "b" * 64,
                "frozen_budget": {"type": "iterations", "value": 30000},
                "completion_evidence": {
                    "progress_unit": "iterations",
                    "last_valid_progress": 30000.0,
                    "required_product_postvalidation_passed": True,
                },
                "products": [phase_product_row(
                    decoy_model,
                    validate_model_container=True,
                    method_id="2dgs",
                )],
                "ended_at_utc": "2026-08-21T00:00:00Z",
            }
            bind_environment(payload, root)
            payload["canonical_sha256"] = canonical_sha256(payload)
            marker.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "differ from final model"):
                phase_success_inventory(
                    marker,
                    method_id="2dgs",
                    phase="training",
                    recipe_sha256="a" * 64,
                    expected_command_sha256="b" * 64,
                    frozen_budget={"type": "iterations", "value": 30000},
                    expected_product_paths=[final_model],
                )

    def test_citygs_x_binds_all_packet_required_checkpoint_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory).resolve()
            model = run_root / "model"
            checkpoint = model / "point_cloud" / "iteration_100000"
            checkpoint.mkdir(parents=True)
            point_cloud = checkpoint / "point_cloud.ply"
            write_gaussian_ply(point_cloud, method_id="citygs_x")
            write_npz(checkpoint / "additional_attributes.npz")
            write_torch_checkpoint(checkpoint / "checkpoints.pth")
            prior_root = run_root / "prior"
            prior_root.mkdir()
            prior_manifest = prior_root / "depth_and_multiview_prior_v1.json"
            prior_manifest.write_text("{}", encoding="utf-8")
            summary = {
                "method_id": "citygs_x", "status": "TRAINING_PASS", "mode": "formal",
                "scene": "gcp_100000_20260610",
                "protocol_id": "m3m_gcp_native_quarter_geometry_v2",
                "formal_result": True,
                "iterations": 100000,
                "checkpoint": {
                    "path": str(checkpoint), "point_cloud_file": point_cloud.name,
                    "point_cloud_bytes": point_cloud.stat().st_size,
                    "point_cloud_sha256": sha256_file(point_cloud),
                    "additional_attributes": {
                        "path": str(checkpoint / "additional_attributes.npz"),
                        "bytes": (checkpoint / "additional_attributes.npz").stat().st_size,
                        "sha256": sha256_file(checkpoint / "additional_attributes.npz"),
                    },
                    "optimizer_checkpoint": {
                        "path": str(checkpoint / "checkpoints.pth"),
                        "bytes": (checkpoint / "checkpoints.pth").stat().st_size,
                        "sha256": sha256_file(checkpoint / "checkpoints.pth"),
                    },
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
