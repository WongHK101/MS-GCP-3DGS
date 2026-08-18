#!/usr/bin/env python3
"""Materialize a train-only native-quarter COLMAP view with 2D-3D tracks."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_colmap_io(path: Path) -> ModuleType:
    path = path.resolve()
    spec = importlib.util.spec_from_file_location("m3m_colmap_io", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load COLMAP I/O module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_hash(path: Path, expected: str, label: str) -> str:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"{label} SHA256 mismatch: expected {expected}, got {actual}")
    return actual


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original_model", type=Path, required=True)
    parser.add_argument("--formal_train_model", type=Path, required=True)
    parser.add_argument("--formal_input_manifest", type=Path, required=True)
    parser.add_argument("--colmap_io", type=Path, required=True)
    parser.add_argument("--output_model", type=Path, required=True)
    parser.add_argument("--output_manifest", type=Path, required=True)
    parser.add_argument("--expected_original_cameras_sha256", required=True)
    parser.add_argument("--expected_original_images_sha256", required=True)
    parser.add_argument("--expected_original_points3d_sha256", required=True)
    parser.add_argument("--expected_formal_cameras_sha256", required=True)
    parser.add_argument("--expected_formal_images_sha256", required=True)
    args = parser.parse_args()

    original = args.original_model.resolve()
    formal = args.formal_train_model.resolve()
    formal_manifest_path = args.formal_input_manifest.resolve()
    output = args.output_model.resolve()
    output_manifest = args.output_manifest.resolve()
    if output.exists() or output_manifest.exists():
        raise FileExistsError("output already exists; overwrite/resume is forbidden")
    for required in (
        original / "cameras.bin",
        original / "images.bin",
        original / "points3D.bin",
        formal / "cameras.bin",
        formal / "images.bin",
        formal_manifest_path,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    original_hashes = {
        "cameras.bin": require_hash(
            original / "cameras.bin", args.expected_original_cameras_sha256, "original cameras"
        ),
        "images.bin": require_hash(
            original / "images.bin", args.expected_original_images_sha256, "original images"
        ),
        "points3D.bin": require_hash(
            original / "points3D.bin", args.expected_original_points3d_sha256, "original points3D"
        ),
    }
    formal_hashes = {
        "cameras.bin": require_hash(
            formal / "cameras.bin", args.expected_formal_cameras_sha256, "formal cameras"
        ),
        "images.bin": require_hash(
            formal / "images.bin", args.expected_formal_images_sha256, "formal images"
        ),
    }
    if original_hashes["cameras.bin"] != formal_hashes["cameras.bin"]:
        raise RuntimeError(
            "the frozen native-quarter 94-view and formal train camera files must be byte-identical"
        )

    colmap = load_colmap_io(args.colmap_io)
    original_cameras = colmap.read_cameras_binary(str(original / "cameras.bin"))
    original_images = colmap.read_images_binary(str(original / "images.bin"))
    formal_cameras = colmap.read_cameras_binary(str(formal / "cameras.bin"))
    formal_images = colmap.read_images_binary(str(formal / "images.bin"))
    formal_manifest = json.loads(formal_manifest_path.read_text(encoding="utf-8"))
    train_names = {
        record["image_name"]
        for record in formal_manifest["images"]
        if record["role"] == "train"
    }
    heldout_names = {
        record["image_name"]
        for record in formal_manifest["images"]
        if record["role"] == "test"
    }
    if len(train_names) != 82 or len(heldout_names) != 12:
        raise RuntimeError("frozen split must contain exactly 82 train and 12 heldout views")
    formal_by_name = {image.name: image for image in formal_images.values()}
    original_by_name = {image.name: image for image in original_images.values()}
    if set(formal_by_name) != train_names:
        raise RuntimeError("formal train model image inventory differs from frozen train names")
    if not train_names.issubset(original_by_name):
        raise RuntimeError("original COLMAP model is missing frozen training images")
    if set(original_by_name) != train_names | heldout_names:
        raise RuntimeError("original COLMAP model image inventory differs from frozen 94-view domain")
    if any(image.xys.shape[0] != 0 for image in formal_images.values()):
        raise RuntimeError("formal training images.bin unexpectedly contains 2D tracks")

    derived_images: dict[int, Any] = {}
    total_keypoints = 0
    total_observations = 0
    maximum_pose_difference = 0.0
    for name in sorted(train_names):
        source_image = original_by_name[name]
        formal_image = formal_by_name[name]
        if source_image.id != formal_image.id or source_image.camera_id != formal_image.camera_id:
            raise RuntimeError(f"image/camera identity mismatch: {name}")
        pose_difference = max(
            float(np.max(np.abs(source_image.qvec - formal_image.qvec))),
            float(np.max(np.abs(source_image.tvec - formal_image.tvec))),
        )
        maximum_pose_difference = max(maximum_pose_difference, pose_difference)
        if pose_difference != 0.0:
            raise RuntimeError(f"camera pose mismatch: {name}: {pose_difference}")

        source_camera = original_cameras[source_image.camera_id]
        target_camera = formal_cameras[formal_image.camera_id]
        if source_camera.model != "PINHOLE" or target_camera.model != "PINHOLE":
            raise RuntimeError("track-view derivation is frozen to the native-quarter PINHOLE domain")
        if (
            source_camera.id != target_camera.id
            or source_camera.width != target_camera.width
            or source_camera.height != target_camera.height
            or not np.array_equal(source_camera.params, target_camera.params)
        ):
            raise RuntimeError(f"native-quarter camera record mismatch: {name}")
        total_keypoints += int(source_image.xys.shape[0])
        valid_observations = int(np.count_nonzero(source_image.point3D_ids >= 0))
        total_observations += valid_observations
        derived_images[source_image.id] = colmap.Image(
            id=formal_image.id,
            qvec=formal_image.qvec.copy(),
            tvec=formal_image.tvec.copy(),
            camera_id=formal_image.camera_id,
            name=formal_image.name,
            xys=source_image.xys.copy(),
            point3D_ids=source_image.point3D_ids.copy(),
        )

    output.mkdir(parents=True)
    shutil.copyfile(formal / "cameras.bin", output / "cameras.bin")
    shutil.copyfile(original / "points3D.bin", output / "points3D.bin")
    colmap.write_images_binary(derived_images, str(output / "images.bin"))

    roundtrip_cameras = colmap.read_cameras_binary(str(output / "cameras.bin"))
    roundtrip_images = colmap.read_images_binary(str(output / "images.bin"))
    roundtrip_points = colmap.read_points3D_binary(str(output / "points3D.bin"))
    roundtrip_names = {image.name for image in roundtrip_images.values()}
    if roundtrip_names != train_names or roundtrip_names & heldout_names:
        raise RuntimeError("derived model contains an invalid image inventory")
    if len(roundtrip_cameras) != len(formal_cameras):
        raise RuntimeError("derived camera count mismatch")
    roundtrip_observations = sum(
        int(np.count_nonzero(image.point3D_ids >= 0)) for image in roundtrip_images.values()
    )
    if roundtrip_observations != total_observations:
        raise RuntimeError("derived observation count changed after serialization")

    evidence: dict[str, Any] = {
        "schema": "m3m_gcp_native_quarter_citygaussian_colmap_track_view_v1",
        "protocol_id": "m3m_gcp_native_quarter_geometry_v2",
        "scene": formal_manifest["scene"],
        "status": "PASS",
        "passed": True,
        "role": "derived non-authoritative training-only compatibility view for official monocular-depth scale fitting",
        "authoritative_training_camera_model_replaced": False,
        "original_model": {
            "path": str(original),
            "image_count": len(original_images),
            "point_count": len(roundtrip_points),
            "sha256": original_hashes,
        },
        "formal_train_model": {
            "path": str(formal),
            "image_count": len(formal_images),
            "effective_observation_count": 0,
            "sha256": formal_hashes,
        },
        "derived_model": {
            "path": str(output),
            "camera_count": len(roundtrip_cameras),
            "image_count": len(roundtrip_images),
            "heldout_image_record_count": len(roundtrip_names & heldout_names),
            "keypoint_count": sum(image.xys.shape[0] for image in roundtrip_images.values()),
            "effective_observation_count": roundtrip_observations,
            "point_count": len(roundtrip_points),
            "sha256": {
                "cameras.bin": sha256(output / "cameras.bin"),
                "images.bin": sha256(output / "images.bin"),
                "points3D.bin": sha256(output / "points3D.bin"),
            },
        },
        "coordinate_transform": {
            "operation": "none",
            "source_and_target_pixel_domain": "the same frozen native-quarter undistorted PINHOLE domain",
            "xys_policy": "direct copy from the 94-view native-quarter model",
            "point3D_ids_policy": "direct copy from the 94-view native-quarter model",
        },
        "validation": {
            "maximum_qvec_or_tvec_absolute_difference": maximum_pose_difference,
            "source_keypoint_count": total_keypoints,
            "source_effective_observation_count": total_observations,
            "camera_files_byte_identical": True,
            "no_pixel_rescaling_or_resampling": True,
            "train_name_set_exact": True,
            "heldout_image_records_absent": True,
            "formal_cameras_bin_byte_identical": (
                sha256(output / "cameras.bin") == formal_hashes["cameras.bin"]
            ),
            "original_points3d_bin_byte_identical": (
                sha256(output / "points3D.bin") == original_hashes["points3D.bin"]
            ),
        },
        "access_boundary": {
            "training_image_records_used": len(train_names),
            "heldout_image_records_used": 0,
            "heldout_rgb_opened": 0,
            "gcp_annotations_opened": 0,
            "lidar_opened": 0,
            "note": "The byte-identical original points3D.bin retains upstream track metadata, but official scale fitting reads only point xyz/error and selected training-image observations.",
        },
    }
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
