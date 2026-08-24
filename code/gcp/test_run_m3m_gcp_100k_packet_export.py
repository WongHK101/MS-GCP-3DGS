#!/usr/bin/env python3
"""CPU-only invariants for the frozen 100K packet dispatcher."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import run_m3m_gcp_100k_packet_export as packet_module
from run_m3m_gcp_100k_packet_export import (
    CITYGS_X_PYTORCH3D_COMPAT_RELATIVE,
    SCENE,
    uses_geometry_camera_only,
    verify_allowlist,
    verify_camera_root,
    verify_checkpoint,
)


class PacketExport100KTest(unittest.TestCase):
    def test_geometry_camera_only_is_lidar_only_and_method_scoped(self) -> None:
        self.assertTrue(uses_geometry_camera_only("3dgs_original", "lidar"))
        self.assertTrue(uses_geometry_camera_only("rade_gs", "lidar"))
        self.assertFalse(uses_geometry_camera_only("3dgs_original", "gcp"))
        self.assertFalse(uses_geometry_camera_only("pgsr", "lidar"))
        self.assertFalse(uses_geometry_camera_only("citygs_x", "lidar"))

    @unittest.skipIf(os.name == "nt", "Windows test host lacks symlink privilege")
    def test_evaluation_camera_root_is_pose_only_and_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            formal = root / "formal-train"
            formal_images = formal / "images"
            formal_images.mkdir(parents=True)
            (formal_images / "a.jpg").write_bytes(b"jpeg")
            camera = root / "camera"
            sparse = camera / "sparse" / "0"
            sparse.mkdir(parents=True)
            (camera / "images").symlink_to(formal_images, target_is_directory=True)
            values = {
                "cameras.bin": b"cam",
                "images.bin": b"poses",
                "points3D.bin": (0).to_bytes(8, "little"),
                "points3D.ply": b"ply",
            }
            files = {}
            hashes = {}
            for name, value in values.items():
                path = sparse / name
                path.write_bytes(value)
                hashes[name] = packet_module.sha256(path)
                files[name] = {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": hashes[name],
                }
            manifest = {
                "schema": "m3m_gcp_100k_evaluation_camera_root_v1",
                "status": "PASS_EVALUATION_CAMERA_ROOT_NO_TRAINING_NO_PRIOR_NO_EVALUATION",
                "scene": SCENE,
                "output": {
                    "root": str(camera.resolve()),
                    "view_count": 1,
                    "points3d_bin_point_count": 0,
                    "files": files,
                },
                "truth_boundary": {
                    "heldout_rgb_present": False,
                    "gcp_or_lidar_used": False,
                },
            }
            manifest["canonical_sha256"] = packet_module.canonical_sha256(manifest)
            manifest_path = camera / "EVALUATION_CAMERA_ROOT_MANIFEST.json"
            manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
            profile = {
                "root": camera,
                "manifest_name": manifest_path.name,
                "manifest_sha256": packet_module.sha256(manifest_path),
                "schema": "m3m_gcp_100k_evaluation_camera_root_v1",
                "status": "PASS_EVALUATION_CAMERA_ROOT_NO_TRAINING_NO_PRIOR_NO_EVALUATION",
                "view_count": 1,
                "view_count_field": "view_count",
                "files_field": "files",
                "images_policy": "formal_train_symlink",
                "sparse_sha256": hashes,
            }
            with mock.patch.object(packet_module, "FORMAL_TRAIN_ROOT", formal):
                verify_camera_root(camera, profile)
                (sparse / "points3D.bin").write_bytes(b"not-empty")
                with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                    verify_camera_root(camera, profile)

    def test_citygs_compatibility_package_is_repository_bound(self) -> None:
        compat = Path(__file__).resolve().parents[2] / CITYGS_X_PYTORCH3D_COMPAT_RELATIVE
        self.assertTrue((compat / "pytorch3d/__init__.py").is_file())
        self.assertTrue((compat / "pytorch3d/transforms/__init__.py").is_file())
        self.assertNotIn("staging", str(CITYGS_X_PYTORCH3D_COMPAT_RELATIVE))

    def test_allowlist_requires_exact_unique_2196(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "allowlist.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["image_name"])
                writer.writeheader()
                writer.writerows(
                    {"image_name": f"image_{index:04d}.JPG"} for index in range(2196)
                )
            verify_allowlist(path, 2196)
            with path.open("a", encoding="utf-8", newline="") as handle:
                handle.write("image_0000.JPG\n")
            with self.assertRaisesRegex(RuntimeError, "2196 unique"):
                verify_allowlist(path, 2196)

    def test_gsprior_scale_is_read_only_from_frozen_prior_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_root = root / "run"
            point_cloud = run_root / "model" / "point_cloud" / "iteration_40000" / "point_cloud.ply"
            point_cloud.parent.mkdir(parents=True)
            point_cloud.write_bytes(b"ply\n")
            prior_root = root / "prior"
            prior_root.mkdir()
            (prior_root / "normalization_manifest.json").write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "scene_id": SCENE,
                        "transform": {
                            "original_colmap_units_per_normalized_unit": 12.5
                        },
                    }
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                method_id="gsprior", training_run_root=run_root, prior_root=prior_root
            )
            model_path, scale = verify_checkpoint(args, {"iteration": 40000})
            self.assertEqual(model_path, (run_root / "model").resolve())
            self.assertEqual(scale, 12.5)


if __name__ == "__main__":
    unittest.main()
