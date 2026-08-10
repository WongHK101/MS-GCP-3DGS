import csv
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
    PIXEL_DOMAIN,
    RELEASE_DIGEST,
    SCHEMA as FORMAL_INPUT_SCHEMA,
    canonical_sha256,
    sha256_file,
)
from materialize_m3m_native_quarter_evaluation_subset import materialize_subset, verify_subset  # noqa: E402
from read_write_model import Camera, Image as ColmapImage, write_cameras_binary, write_images_binary  # noqa: E402


class EvaluationCameraSubsetTest(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path]:
        formal = root / "formal"
        camera = Camera(
            id=1,
            model="PINHOLE",
            width=8,
            height=6,
            params=np.asarray([5.0, 5.0, 4.0, 3.0], dtype=np.float64),
        )
        ply_bytes = b"ply\nformat ascii 1.0\nelement vertex 0\nend_header\n"
        image_rows = []
        role_rows = []
        for role, indices in (("train", (1, 2)), ("test", (3, 4))):
            images_root = formal / role / "images"
            model_root = formal / role / "sparse" / "0"
            images_root.mkdir(parents=True)
            model_root.mkdir(parents=True)
            model_images = {}
            for index in indices:
                name = f"image_{index:02d}.JPG"
                image_path = images_root / name
                PILImage.fromarray(np.full((6, 8, 3), index * 20, dtype=np.uint8), mode="RGB").save(
                    image_path,
                    format="JPEG",
                    quality=90,
                )
                model_images[index] = ColmapImage(
                    id=index,
                    qvec=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
                    tvec=np.asarray([float(index), 0.0, 0.0], dtype=np.float64),
                    camera_id=1,
                    name=name,
                    xys=np.empty((0, 2), dtype=np.float64),
                    point3D_ids=np.empty((0,), dtype=np.int64),
                )
                image_rows.append({
                    "role": role,
                    "image_id": index,
                    "camera_id": 1,
                    "image_name": name,
                    "relative_path": f"{role}/images/{name}",
                    "width": 8,
                    "height": 6,
                    "jpeg_bytes": image_path.stat().st_size,
                    "jpeg_sha256": sha256_file(image_path),
                })
            write_cameras_binary({1: camera}, model_root / "cameras.bin")
            write_images_binary(model_images, model_root / "images.bin")
            (model_root / "points3D.ply").write_bytes(ply_bytes)
            role_rows.append({
                "role": role,
                "root": role,
                "image_count": len(indices),
                "camera_count": 1,
                "cameras_bin_sha256": sha256_file(model_root / "cameras.bin"),
                "images_bin_sha256": sha256_file(model_root / "images.bin"),
                "points3d_ply_sha256": sha256_file(model_root / "points3D.ply"),
                "points2d_tracks_present": False,
                "points3d_bin_present": False,
            })
        manifest = {
            "schema": FORMAL_INPUT_SCHEMA,
            "scene": "synthetic_scene",
            "release_root_digest_sha256": RELEASE_DIGEST,
            "pixel_domain": PIXEL_DOMAIN,
            "source_model_sha256": {"points3D.ply": sha256_file(formal / "train" / "sparse" / "0" / "points3D.ply")},
            "full_view_count": 4,
            "train_view_count": 2,
            "test_view_count": 2,
            "roles": role_rows,
            "images": image_rows,
        }
        manifest["manifest_sha256"] = canonical_sha256(manifest)
        manifest_path = formal / "NATIVE_QUARTER_INPUT_MANIFEST.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        observations_path = root / "triangulation_observation_residuals.csv"
        with observations_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("scene", "point_name", "image_name"))
            writer.writeheader()
            writer.writerows([
                {"scene": "synthetic_scene", "point_name": "P1", "image_name": "image_01.JPG"},
                {"scene": "synthetic_scene", "point_name": "P1", "image_name": "image_03.JPG"},
                {"scene": "synthetic_scene", "point_name": "P2", "image_name": "image_03.JPG"},
            ])
        return manifest_path, observations_path

    def test_materializes_formal_train_and_test_cameras(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, observations_path = self._fixture(root)
            output = root / "evaluation"
            manifest = materialize_subset(
                scene="synthetic_scene",
                formal_input_manifest_path=manifest_path,
                protocol_observations_path=observations_path,
                output_root=output,
                file_mode="copy",
            )
            self.assertEqual(manifest["camera_view_count"], 2)
            self.assertEqual(manifest["observation_count"], 3)
            self.assertEqual(manifest["point_count"], 2)
            self.assertEqual({row["formal_role"] for row in manifest["images"]}, {"train", "test"})
            self.assertFalse((output / "sparse" / "0" / "points3D.bin").exists())
            result = verify_subset(output)
            self.assertTrue(result["passed"], result["errors"])

    def test_verifier_rejects_corrupted_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, observations_path = self._fixture(root)
            output = root / "evaluation"
            materialize_subset(
                scene="synthetic_scene",
                formal_input_manifest_path=manifest_path,
                protocol_observations_path=observations_path,
                output_root=output,
                file_mode="copy",
            )
            target = output / "images" / "image_01.JPG"
            target.write_bytes(target.read_bytes() + b"corrupt")
            result = verify_subset(output, decode_images=False)
            self.assertFalse(result["passed"])
            self.assertTrue(any("image identity mismatch" in error for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
