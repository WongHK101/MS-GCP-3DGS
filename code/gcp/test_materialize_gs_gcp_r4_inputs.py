#!/usr/bin/env python3
"""Tests for clean GS-GCP R4 input materialization."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from materialize_gs_gcp_r4_inputs import (
    RELEASE_DIGEST,
    canonical_sha256,
    materialize_scene,
    quarter_dimensions,
    scale_pinhole_camera,
    sha256_file,
    verify_materialization,
)

from read_write_model import (
    Camera,
    Image as ColmapImage,
    read_cameras_binary,
    read_images_binary,
    write_cameras_binary,
    write_images_binary,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _make_fixture(root: Path) -> tuple[Path, Path, Path]:
    source = root / "source" / "synthetic_scene"
    image_root = source / "images"
    sparse = source / "sparse" / "0"
    image_root.mkdir(parents=True)
    sparse.mkdir(parents=True)
    camera = Camera(
        id=1,
        model="PINHOLE",
        width=10,
        height=18,
        params=np.asarray([8.0, 14.0, 4.5, 8.5], dtype=np.float64),
    )
    write_cameras_binary({1: camera}, sparse / "cameras.bin")
    assignments = []
    images = {}
    for index, role in enumerate(("train", "test"), start=1):
        name = f"DJI_2026080700000{index}_{index:04d}_D.JPG"
        pixels = np.arange(10 * 18 * 3, dtype=np.uint16).reshape(18, 10, 3)
        pixels = ((pixels + index * 17) % 256).astype(np.uint8)
        target = image_root / name
        PILImage.fromarray(pixels, mode="RGB").save(target, format="JPEG", quality=95)
        images[index] = ColmapImage(
            id=index,
            qvec=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            tvec=np.asarray([float(index), 2.0, 3.0], dtype=np.float64),
            camera_id=1,
            name=name,
            xys=np.asarray([[1.0, 2.0]], dtype=np.float64),
            point3D_ids=np.asarray([7], dtype=np.int64),
        )
        assignments.append({
            "scene": "synthetic_scene",
            "image_id": index,
            "camera_id": 1,
            "image_name": name,
            "image_sha256": sha256_file(target),
            "image_bytes": target.stat().st_size,
            "decoded_width": 10,
            "decoded_height": 18,
            "split_role": role,
        })
    write_images_binary(images, sparse / "images.bin")
    (sparse / "points3D.ply").write_bytes(b"ply\nformat ascii 1.0\nelement vertex 0\nend_header\n")
    split = {
        "schema": "gs_gcp_rgb_holdout_split_manifest_v1",
        "split_protocol": "gs_gcp_rgb_holdout_split_v1",
        "holdout_semantics": "image_loss_holdout_under_shared_all_image_sfm_v1",
        "release_root_digest": RELEASE_DIGEST,
        "scenes": [{"scene": "synthetic_scene", "assignments": assignments}],
    }
    split["manifest_sha256"] = canonical_sha256(split)
    split_path = root / "split.json"
    _write_json(split_path, split)
    source_manifest = {
        "schema": "gs_gcp_v1_3_training_source_manifest_v1",
        "release": {"payload_root_digest_sha256": RELEASE_DIGEST},
        "scenes": {
            "synthetic_scene": {
                "cameras_bin_sha256": sha256_file(sparse / "cameras.bin"),
                "images_bin_sha256": sha256_file(sparse / "images.bin"),
                "points3d_ply_sha256": sha256_file(sparse / "points3D.ply"),
            }
        },
    }
    source_manifest_path = root / "source_manifest.json"
    _write_json(source_manifest_path, source_manifest)
    return source, split_path, source_manifest_path


def test_quarter_dimensions_follow_python_round() -> None:
    assert quarter_dimensions(5654, 4098) == (1414, 1024)
    assert quarter_dimensions(10, 18) == (2, 4)
    assert quarter_dimensions(14, 22) == (4, 6)


def test_pinhole_scaling_preserves_normalized_camera() -> None:
    source = Camera(1, "PINHOLE", 5654, 4098, np.asarray([3000.0, 2990.0, 2827.0, 2049.0]))
    scaled = scale_pinhole_camera(source, 1414, 1024)
    assert (scaled.width, scaled.height) == (1414, 1024)
    assert np.isclose(scaled.params[0] / scaled.width, source.params[0] / source.width, atol=1e-15)
    assert np.isclose(scaled.params[1] / scaled.height, source.params[1] / source.height, atol=1e-15)
    assert np.isclose(scaled.params[2] / scaled.params[0], source.params[2] / source.params[0], atol=1e-15)
    assert np.isclose(scaled.params[3] / scaled.params[1], source.params[3] / source.params[1], atol=1e-15)


def test_materialization_is_lossless_separated_and_track_free() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        source, split_path, source_manifest_path = _make_fixture(root)
        output = root / "r4"
        manifest = materialize_scene(
            split_manifest_path=split_path,
            source_manifest_path=source_manifest_path,
            scene="synthetic_scene",
            source_root=source,
            output_root=output,
        )
        assert manifest["train_view_count"] == 1
        assert manifest["test_view_count"] == 1
        assert not (output / "train" / "sparse" / "0" / "points3D.bin").exists()
        assert not (output / "test" / "sparse" / "0" / "points3D.bin").exists()
        train_images = read_images_binary(output / "train" / "sparse" / "0" / "images.bin")
        test_images = read_images_binary(output / "test" / "sparse" / "0" / "images.bin")
        assert set(row.name for row in train_images.values()).isdisjoint(
            row.name for row in test_images.values()
        )
        assert all(len(row.point3D_ids) == 0 for row in train_images.values())
        assert all(len(row.point3D_ids) == 0 for row in test_images.values())
        camera = next(iter(read_cameras_binary(output / "train" / "sparse" / "0" / "cameras.bin").values()))
        assert (camera.width, camera.height) == (2, 4)
        for row in manifest["images"]:
            source_path = source / "images" / row["source_name"]
            target_path = output / row["r4_relative_path"]
            with PILImage.open(source_path) as image:
                expected = np.asarray(image.resize((2, 4)))
            with PILImage.open(target_path) as image:
                actual = np.asarray(image)
            assert np.array_equal(actual, expected)
            assert hashlib.sha256(actual.tobytes(order="C")).hexdigest() == row["r4_rgb_uint8_sha256"]
        verification = verify_materialization(output)
        assert verification["passed"], verification["errors"]


def test_materialization_rejects_changed_split_hash() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        source, split_path, source_manifest_path = _make_fixture(root)
        split = json.loads(split_path.read_text(encoding="utf-8"))
        bad = copy.deepcopy(split)
        bad["scenes"][0]["assignments"][0]["split_role"] = "test"
        _write_json(split_path, bad)
        try:
            materialize_scene(
                split_manifest_path=split_path,
                source_manifest_path=source_manifest_path,
                scene="synthetic_scene",
                source_root=source,
                output_root=root / "r4",
            )
        except ValueError as exc:
            assert "canonical SHA" in str(exc)
        else:
            raise AssertionError("mutated split was accepted")


TESTS = [
    test_quarter_dimensions_follow_python_round,
    test_pinhole_scaling_preserves_normalized_camera,
    test_materialization_is_lossless_separated_and_track_free,
    test_materialization_rejects_changed_split_hash,
]


def main() -> int:
    for test in TESTS:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS {len(TESTS)}/{len(TESTS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
