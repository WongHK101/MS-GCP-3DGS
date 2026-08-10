import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image as PILImage


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
COLMAP_UTILS = HERE.parent / "colmap" / "utils"
if str(COLMAP_UTILS) not in sys.path:
    sys.path.insert(0, str(COLMAP_UTILS))

from materialize_gs_gcp_native_quarter_inputs import (  # noqa: E402
    HOLDOUT_SEMANTICS,
    RELEASE_DIGEST,
    canonical_sha256,
    materialize_scene,
    sha256_file,
    verify_materialization,
)
from read_write_model import Camera, Image as ColmapImage, write_cameras_binary, write_images_binary  # noqa: E402


class NativeQuarterMaterializationTest(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path]:
        source = root / "source" / "synthetic_scene"
        images_root = source / "images"
        sparse = source / "sparse" / "0"
        evidence = source / "evidence"
        images_root.mkdir(parents=True)
        sparse.mkdir(parents=True)
        evidence.mkdir(parents=True)

        camera = Camera(
            id=1,
            model="PINHOLE",
            width=8,
            height=6,
            params=np.asarray([5.0, 5.0, 4.0, 3.0], dtype=np.float64),
        )
        colmap_images = {}
        assignments = []
        sha_lines = []
        for index in range(1, 9):
            name = f"image_{index:02d}.JPG"
            pixels = np.full((6, 8, 3), index * 20, dtype=np.uint8)
            path = images_root / name
            PILImage.fromarray(pixels, mode="RGB").save(path, format="JPEG", quality=90)
            digest = sha256_file(path)
            sha_lines.append(f"{digest}  images/{name}\n")
            colmap_images[index] = ColmapImage(
                id=index,
                qvec=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
                tvec=np.asarray([float(index), 0.0, 0.0], dtype=np.float64),
                camera_id=1,
                name=name,
                xys=np.empty((0, 2), dtype=np.float64),
                point3D_ids=np.empty((0,), dtype=np.int64),
            )
            assignments.append({
                "scene": "synthetic_scene",
                "image_id": index,
                "camera_id": 1,
                "image_name": name,
                "split_role": "test" if index == 4 else "train",
                "decoded_width": 32,
                "decoded_height": 24,
                "image_bytes": path.stat().st_size,
                "image_sha256": "0" * 64,
            })
        write_cameras_binary({1: camera}, sparse / "cameras.bin")
        write_images_binary(colmap_images, sparse / "images.bin")
        (sparse / "points3D.bin").write_bytes((0).to_bytes(8, "little"))
        (sparse / "points3D.ply").write_text("ply\nformat ascii 1.0\nelement vertex 0\nend_header\n", encoding="ascii")
        image_sha_path = evidence / "images.sha256"
        image_sha_path.write_text("".join(sha_lines), encoding="utf-8")

        model_files = {
            name: {"sha256": sha256_file(sparse / name)}
            for name in ("cameras.bin", "images.bin", "points3D.bin", "points3D.ply")
        }
        audit = {
            "schema": "gs-gcp-colmap-native-quarter-package-audit-v2",
            "scene": "synthetic_scene",
            "status": "pass",
            "standard_colmap_layout": True,
            "train_ready_for_standard_colmap_loaders": True,
            "image_generation": {
                "generator": "COLMAP 4.0.4 image_undistorter",
                "max_image_size": 1414,
            },
            "counts": {"images": 8},
            "decoded_images": {"sizes": [[8, 6]]},
            "model_files": model_files,
            "image_sha256_manifest": {"sha256": sha256_file(image_sha_path)},
        }
        (evidence / "PACKAGE_AUDIT.json").write_text(json.dumps(audit), encoding="utf-8")

        split = {
            "schema": "gs_gcp_rgb_holdout_split_manifest_v1",
            "split_protocol": "gs_gcp_rgb_holdout_split_v1",
            "holdout_semantics": HOLDOUT_SEMANTICS,
            "release_root_digest": RELEASE_DIGEST,
            "scenes": [{"scene": "synthetic_scene", "assignments": assignments}],
        }
        split["manifest_sha256"] = canonical_sha256(split)
        split_path = root / "split.json"
        split_path.write_text(json.dumps(split), encoding="utf-8")
        return source, split_path

    def test_materializes_byte_preserving_disjoint_roots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, split = self._fixture(root)
            output = root / "formal"
            manifest = materialize_scene(
                split_manifest_path=split,
                scene="synthetic_scene",
                source_root=source,
                output_root=output,
                file_mode="copy",
            )
            self.assertEqual(manifest["train_view_count"], 7)
            self.assertEqual(manifest["test_view_count"], 1)
            result = verify_materialization(output)
            self.assertTrue(result["passed"], result["errors"])
            self.assertFalse((output / "train" / "sparse" / "0" / "points3D.bin").exists())
            self.assertEqual(
                sha256_file(source / "images" / "image_04.JPG"),
                sha256_file(output / "test" / "images" / "image_04.JPG"),
            )

    def test_verifier_rejects_corrupted_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, split = self._fixture(root)
            output = root / "formal"
            materialize_scene(
                split_manifest_path=split,
                scene="synthetic_scene",
                source_root=source,
                output_root=output,
                file_mode="copy",
            )
            target = output / "train" / "images" / "image_01.JPG"
            target.write_bytes(target.read_bytes() + b"corrupt")
            result = verify_materialization(output, decode_images=False)
            self.assertFalse(result["passed"])
            self.assertTrue(any("image byte count mismatch" in error for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
