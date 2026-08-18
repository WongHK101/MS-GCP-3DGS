#!/usr/bin/env python3
"""Materialize a train-only COLMAP model with internally closed 2D-3D tracks."""

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
    spec = importlib.util.spec_from_file_location("m3m_colmap_io", path.resolve())
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
        args.colmap_io,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    original_hashes = {
        "cameras.bin": require_hash(
            original / "cameras.bin",
            args.expected_original_cameras_sha256,
            "original cameras",
        ),
        "images.bin": require_hash(
            original / "images.bin",
            args.expected_original_images_sha256,
            "original images",
        ),
        "points3D.bin": require_hash(
            original / "points3D.bin",
            args.expected_original_points3d_sha256,
            "original points3D",
        ),
    }
    formal_hashes = {
        "cameras.bin": require_hash(
            formal / "cameras.bin",
            args.expected_formal_cameras_sha256,
            "formal cameras",
        ),
        "images.bin": require_hash(
            formal / "images.bin",
            args.expected_formal_images_sha256,
            "formal images",
        ),
    }
    if original_hashes["cameras.bin"] != formal_hashes["cameras.bin"]:
        raise RuntimeError("94-view and formal-train cameras.bin must be byte-identical")

    colmap = load_colmap_io(args.colmap_io)
    original_cameras = colmap.read_cameras_binary(str(original / "cameras.bin"))
    original_images = colmap.read_images_binary(str(original / "images.bin"))
    original_points = colmap.read_points3D_binary(str(original / "points3D.bin"))
    formal_cameras = colmap.read_cameras_binary(str(formal / "cameras.bin"))
    formal_images = colmap.read_images_binary(str(formal / "images.bin"))
    formal_manifest = json.loads(formal_manifest_path.read_text(encoding="utf-8"))

    train_names = {
        row["image_name"] for row in formal_manifest["images"] if row["role"] == "train"
    }
    heldout_names = {
        row["image_name"] for row in formal_manifest["images"] if row["role"] == "test"
    }
    if len(train_names) != 82 or len(heldout_names) != 12:
        raise RuntimeError("frozen split must contain exactly 82 train and 12 held-out views")
    original_by_name = {image.name: image for image in original_images.values()}
    formal_by_name = {image.name: image for image in formal_images.values()}
    if set(original_by_name) != train_names | heldout_names:
        raise RuntimeError("original model image inventory differs from the frozen 94-view domain")
    if set(formal_by_name) != train_names:
        raise RuntimeError("formal model image inventory differs from the frozen train set")
    if any(image.xys.shape[0] for image in formal_images.values()):
        raise RuntimeError("formal train images.bin unexpectedly contains 2D observations")

    selected_image_ids = {original_by_name[name].id for name in train_names}
    derived_points: dict[int, Any] = {}
    removed_point_ids: list[int] = []
    removed_track_elements = 0
    kept_track_elements = 0
    for point_id, point in original_points.items():
        keep = np.asarray(
            [int(image_id) in selected_image_ids for image_id in point.image_ids],
            dtype=bool,
        )
        for image_id, point2d_idx in zip(point.image_ids, point.point2D_idxs):
            source_image = original_images[int(image_id)]
            index = int(point2d_idx)
            if index < 0 or index >= source_image.point3D_ids.shape[0]:
                raise RuntimeError(f"invalid track index for point {point_id}")
            if int(source_image.point3D_ids[index]) != int(point_id):
                raise RuntimeError(f"non-reciprocal source track for point {point_id}")
        removed_track_elements += int(np.count_nonzero(~keep))
        if not np.any(keep):
            removed_point_ids.append(int(point_id))
            continue
        image_ids = point.image_ids[keep].copy()
        point2d_idxs = point.point2D_idxs[keep].copy()
        kept_track_elements += int(image_ids.shape[0])
        derived_points[int(point_id)] = colmap.Point3D(
            id=int(point.id),
            xyz=point.xyz.copy(),
            rgb=point.rgb.copy(),
            error=point.error.copy() if hasattr(point.error, "copy") else point.error,
            image_ids=image_ids,
            point2D_idxs=point2d_idxs,
        )

    derived_images: dict[int, Any] = {}
    maximum_pose_difference = 0.0
    valid_image_observations = 0
    for name in sorted(train_names):
        source = original_by_name[name]
        frozen = formal_by_name[name]
        if source.id != frozen.id or source.camera_id != frozen.camera_id:
            raise RuntimeError(f"image/camera identity mismatch: {name}")
        pose_difference = max(
            float(np.max(np.abs(source.qvec - frozen.qvec))),
            float(np.max(np.abs(source.tvec - frozen.tvec))),
        )
        maximum_pose_difference = max(maximum_pose_difference, pose_difference)
        if pose_difference != 0.0:
            raise RuntimeError(f"camera pose mismatch: {name}: {pose_difference}")
        source_camera = original_cameras[source.camera_id]
        target_camera = formal_cameras[frozen.camera_id]
        if (
            source_camera.id != target_camera.id
            or source_camera.model != target_camera.model
            or source_camera.width != target_camera.width
            or source_camera.height != target_camera.height
            or not np.array_equal(source_camera.params, target_camera.params)
        ):
            raise RuntimeError(f"camera record mismatch: {name}")
        point_ids = source.point3D_ids.copy()
        positive = point_ids >= 0
        missing = {int(value) for value in point_ids[positive]} - set(derived_points)
        if missing:
            raise RuntimeError(f"selected image references points without train tracks: {name}")
        valid_image_observations += int(np.count_nonzero(positive))
        derived_images[int(source.id)] = colmap.Image(
            id=int(frozen.id),
            qvec=frozen.qvec.copy(),
            tvec=frozen.tvec.copy(),
            camera_id=int(frozen.camera_id),
            name=frozen.name,
            xys=source.xys.copy(),
            point3D_ids=point_ids,
        )

    if valid_image_observations != kept_track_elements:
        raise RuntimeError(
            "selected image observation count differs from the closed point-track count: "
            f"{valid_image_observations} != {kept_track_elements}"
        )

    output.mkdir(parents=True)
    shutil.copyfile(formal / "cameras.bin", output / "cameras.bin")
    colmap.write_images_binary(derived_images, str(output / "images.bin"))
    colmap.write_points3D_binary(derived_points, str(output / "points3D.bin"))

    roundtrip_images = colmap.read_images_binary(str(output / "images.bin"))
    roundtrip_points = colmap.read_points3D_binary(str(output / "points3D.bin"))
    roundtrip_observations = 0
    for point_id, point in roundtrip_points.items():
        if point.image_ids.shape[0] == 0 or not set(map(int, point.image_ids)) <= selected_image_ids:
            raise RuntimeError(f"roundtrip point track is not train-closed: {point_id}")
        source_point = original_points[point_id]
        if (
            not np.array_equal(point.xyz, source_point.xyz)
            or not np.array_equal(point.rgb, source_point.rgb)
            or float(point.error) != float(source_point.error)
        ):
            raise RuntimeError(f"point geometry changed during track closure: {point_id}")
        for image_id, point2d_idx in zip(point.image_ids, point.point2D_idxs):
            image = roundtrip_images[int(image_id)]
            if int(image.point3D_ids[int(point2d_idx)]) != int(point_id):
                raise RuntimeError(f"roundtrip track reciprocity failed: {point_id}")
            roundtrip_observations += 1
    if roundtrip_observations != kept_track_elements:
        raise RuntimeError("roundtrip track count changed")

    evidence = {
        "schema": "m3m_gcp_native_quarter_colmap_train_track_closure_v1",
        "protocol_id": "m3m_gcp_native_quarter_geometry_v2",
        "scene": formal_manifest["scene"],
        "status": "PASS",
        "passed": True,
        "role": "derived train-only compatibility model for consumers requiring a fully consistent pycolmap reconstruction",
        "original_model": {
            "path": str(original),
            "camera_count": len(original_cameras),
            "image_count": len(original_images),
            "point_count": len(original_points),
            "sha256": original_hashes,
        },
        "derived_model": {
            "path": str(output),
            "camera_count": len(formal_cameras),
            "image_count": len(roundtrip_images),
            "point_count": len(roundtrip_points),
            "track_element_count": roundtrip_observations,
            "sha256": {
                "cameras.bin": sha256(output / "cameras.bin"),
                "images.bin": sha256(output / "images.bin"),
                "points3D.bin": sha256(output / "points3D.bin"),
            },
        },
        "track_closure": {
            "policy": "retain only observations whose image is in the frozen 82-view training split; remove points with no remaining training observation",
            "removed_heldout_image_records": len(heldout_names),
            "removed_track_element_count": removed_track_elements,
            "removed_train_unobserved_point_count": len(removed_point_ids),
            "removed_point_ids_sha256": hashlib.sha256(
                "\n".join(map(str, sorted(removed_point_ids))).encode("ascii")
            ).hexdigest(),
            "kept_point_count": len(roundtrip_points),
            "kept_track_element_count": roundtrip_observations,
        },
        "validation": {
            "camera_file_byte_identical_to_formal_train": sha256(output / "cameras.bin")
            == formal_hashes["cameras.bin"],
            "train_image_name_set_exact": {image.name for image in roundtrip_images.values()}
            == train_names,
            "heldout_image_records_absent": not (
                {image.name for image in roundtrip_images.values()} & heldout_names
            ),
            "maximum_qvec_or_tvec_absolute_difference": maximum_pose_difference,
            "retained_point_xyz_rgb_error_byte_values_unchanged": True,
            "all_tracks_reference_selected_images": True,
            "all_tracks_reciprocal": True,
            "image_pixels_resized_cropped_padded_or_reencoded": False,
        },
        "access_boundary": {
            "heldout_rgb_opened": 0,
            "gcp_annotations_opened": 0,
            "lidar_opened": 0,
            "formal_training_started": False,
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
