#!/usr/bin/env python3
"""Close COLMAP point tracks to the images already present in a subset model."""

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
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_colmap_io(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("m3m_subset_colmap_io", path.resolve())
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load COLMAP I/O module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_model", type=Path, required=True)
    parser.add_argument("--output_model", type=Path, required=True)
    parser.add_argument("--output_manifest", type=Path, required=True)
    parser.add_argument("--colmap_io", type=Path, required=True)
    args = parser.parse_args()

    source = args.input_model.resolve()
    output = args.output_model.resolve()
    manifest_path = args.output_manifest.resolve()
    if output.exists() or manifest_path.exists():
        raise FileExistsError("subset track-closure output already exists")
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    if not args.colmap_io.is_file():
        raise FileNotFoundError(args.colmap_io)

    colmap = load_colmap_io(args.colmap_io)
    images = colmap.read_images_binary(str(source / "images.bin"))
    points = colmap.read_points3D_binary(str(source / "points3D.bin"))
    selected_ids = set(images)
    if not selected_ids:
        raise RuntimeError("subset model contains no images")

    closed_points: dict[int, Any] = {}
    removed_track_elements = 0
    removed_points = 0
    kept_track_elements = 0
    for point_id, point in points.items():
        keep = np.asarray(
            [int(image_id) in selected_ids for image_id in point.image_ids], dtype=bool
        )
        removed_track_elements += int(np.count_nonzero(~keep))
        if not np.any(keep):
            removed_points += 1
            continue
        image_ids = point.image_ids[keep].copy()
        point2d_idxs = point.point2D_idxs[keep].copy()
        for image_id, point2d_idx in zip(image_ids, point2d_idxs):
            image = images[int(image_id)]
            index = int(point2d_idx)
            if index < 0 or index >= image.point3D_ids.shape[0]:
                raise RuntimeError(f"invalid subset track index for point {point_id}")
            if int(image.point3D_ids[index]) != int(point_id):
                raise RuntimeError(f"non-reciprocal subset track for point {point_id}")
        kept_track_elements += int(image_ids.shape[0])
        closed_points[int(point_id)] = colmap.Point3D(
            id=int(point.id),
            xyz=point.xyz.copy(),
            rgb=point.rgb.copy(),
            error=point.error.copy() if hasattr(point.error, "copy") else point.error,
            image_ids=image_ids,
            point2D_idxs=point2d_idxs,
        )

    image_observations = 0
    for image in images.values():
        valid = image.point3D_ids >= 0
        missing = {int(value) for value in image.point3D_ids[valid]} - set(closed_points)
        if missing:
            raise RuntimeError(
                f"subset image {image.name} references points without an in-subset track"
            )
        image_observations += int(np.count_nonzero(valid))
    if image_observations != kept_track_elements:
        raise RuntimeError(
            f"subset image/point observation mismatch: {image_observations} != {kept_track_elements}"
        )

    output.mkdir(parents=True)
    shutil.copyfile(source / "cameras.bin", output / "cameras.bin")
    shutil.copyfile(source / "images.bin", output / "images.bin")
    colmap.write_points3D_binary(closed_points, str(output / "points3D.bin"))

    roundtrip_images = colmap.read_images_binary(str(output / "images.bin"))
    roundtrip_points = colmap.read_points3D_binary(str(output / "points3D.bin"))
    roundtrip_tracks = 0
    for point_id, point in roundtrip_points.items():
        if point.image_ids.shape[0] == 0 or not set(map(int, point.image_ids)) <= selected_ids:
            raise RuntimeError(f"point track is not closed after roundtrip: {point_id}")
        source_point = points[point_id]
        if (
            not np.array_equal(point.xyz, source_point.xyz)
            or not np.array_equal(point.rgb, source_point.rgb)
            or float(point.error) != float(source_point.error)
        ):
            raise RuntimeError(f"point geometry changed during subset closure: {point_id}")
        for image_id, point2d_idx in zip(point.image_ids, point.point2D_idxs):
            if int(roundtrip_images[int(image_id)].point3D_ids[int(point2d_idx)]) != int(point_id):
                raise RuntimeError(f"roundtrip subset reciprocity failed: {point_id}")
            roundtrip_tracks += 1
    if roundtrip_tracks != kept_track_elements:
        raise RuntimeError("subset track count changed during roundtrip")

    payload = {
        "schema": "m3m_gcp_colmap_subset_track_closure_v1",
        "status": "PASS",
        "passed": True,
        "input_model": str(source),
        "output_model": str(output),
        "image_count": len(roundtrip_images),
        "input_point_count": len(points),
        "output_point_count": len(roundtrip_points),
        "removed_point_count": removed_points,
        "removed_track_element_count": removed_track_elements,
        "kept_track_element_count": roundtrip_tracks,
        "input_sha256": {
            name: sha256(source / name)
            for name in ("cameras.bin", "images.bin", "points3D.bin")
        },
        "output_sha256": {
            name: sha256(output / name)
            for name in ("cameras.bin", "images.bin", "points3D.bin")
        },
        "validation": {
            "camera_file_byte_identical": sha256(source / "cameras.bin")
            == sha256(output / "cameras.bin"),
            "image_file_byte_identical": sha256(source / "images.bin")
            == sha256(output / "images.bin"),
            "retained_point_xyz_rgb_error_unchanged": True,
            "all_tracks_reference_subset_images": True,
            "all_tracks_reciprocal": True,
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
