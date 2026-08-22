#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import materialize_m3m_gcp_100k_rgb_camera_root as module


class RgbEvaluationCameraRootTest(unittest.TestCase):
    @unittest.skipIf(os.name == "nt", "Windows test host lacks symlink privilege")
    def test_test_only_camera_root_has_empty_binary_and_links(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            formal = root / "formal"
            test = formal / "test"
            sparse = test / "sparse" / "0"
            images = test / "images"
            sparse.mkdir(parents=True)
            images.mkdir()
            for name, data in {
                "cameras.bin": b"cam",
                "images.bin": b"test-poses",
                "points3D.ply": b"ply",
            }.items():
                (sparse / name).write_bytes(data)
            (images / "heldout.jpg").write_bytes(b"jpeg")
            sparse_sha = {name: module.sha256_file(sparse / name) for name in module.TEST_SPARSE_SHA}
            full_sha = {
                "cameras.bin": sparse_sha["cameras.bin"],
                "images.bin": module.hashlib.sha256(b"full-images").hexdigest(),
                "points3D.bin": module.hashlib.sha256(b"full-points").hexdigest(),
                "points3D.ply": sparse_sha["points3D.ply"],
            }
            manifest = {
                "scene": module.SCENE,
                "full_view_count": 1,
                "train_view_count": 0,
                "test_view_count": 1,
                "source_model_sha256": full_sha,
                "roles": [{
                    "role": "test",
                    "image_count": 1,
                    "points2d_tracks_present": False,
                    "points3d_bin_present": False,
                    "cameras_bin_sha256": sparse_sha["cameras.bin"],
                    "images_bin_sha256": sparse_sha["images.bin"],
                    "points3d_ply_sha256": sparse_sha["points3D.ply"],
                }],
                "images": [{
                    "role": "test",
                    "image_name": "heldout.jpg",
                    "jpeg_bytes": 4,
                    "jpeg_sha256": module.sha256_file(images / "heldout.jpg"),
                }],
            }
            manifest["manifest_sha256"] = module.formal_manifest_canonical_sha256(manifest)
            manifest_path = formal / "NATIVE_QUARTER_INPUT_MANIFEST.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            output = root / "camera-root"
            evidence = root / "evidence.json"
            empty_sha = module.hashlib.sha256((0).to_bytes(8, "little")).hexdigest()
            with (
                mock.patch.object(module, "FORMAL_MANIFEST_SHA", module.sha256_file(manifest_path)),
                mock.patch.object(module, "FORMAL_MANIFEST_CANONICAL_SHA", manifest["manifest_sha256"]),
                mock.patch.object(module, "TEST_SPARSE_SHA", sparse_sha),
                mock.patch.object(module, "FULL_ALL_IMAGE_SFM_SHA", full_sha),
                mock.patch.object(module, "EMPTY_POINTS3D_SHA", empty_sha),
                mock.patch.object(module, "EXPECTED_FULL_VIEWS", 1),
                mock.patch.object(module, "EXPECTED_TRAIN_VIEWS", 0),
                mock.patch.object(module, "EXPECTED_TEST_VIEWS", 1),
            ):
                payload = module.materialize(
                    formal_scene_root=formal,
                    formal_manifest_path=manifest_path,
                    output_root=output,
                    evidence_path=evidence,
                )
            self.assertEqual(payload["status"], module.STATUS)
            self.assertEqual((output / "sparse/0/points3D.bin").read_bytes(), (0).to_bytes(8, "little"))
            self.assertTrue((output / "images").is_symlink())
            for name in module.TEST_SPARSE_SHA:
                self.assertTrue((output / "sparse/0" / name).samefile(sparse / name))
            self.assertEqual(
                module.sha256_file(output / "RGB_EVALUATION_CAMERA_ROOT_MANIFEST.json"),
                module.sha256_file(evidence),
            )
            self.assertTrue(payload["truth_boundary"]["training_or_prior_use_forbidden"])
            self.assertTrue(
                payload["truth_boundary"]["heldout_rgb_present_for_metric_and_loader_compatibility_only"]
            )


if __name__ == "__main__":
    unittest.main()
