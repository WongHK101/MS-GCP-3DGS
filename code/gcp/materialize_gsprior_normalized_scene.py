#!/usr/bin/env python3
"""Materialize a reversible GSPrior scene normalized from frozen COLMAP cameras.

GSPrior's released TSDF code fixes its volume to ``[-1, 1]^3``.  This tool
applies the repository's own NeRF++ camera normalization rule to a copy of a
COLMAP scene.  The transform is estimated from the frozen training cameras
only and can then be reused verbatim for an evaluation-camera scene.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from plyfile import PlyData, PlyElement


REPO_ROOT = Path(__file__).resolve().parents[2]
COLMAP_UTILS = REPO_ROOT / "code" / "colmap" / "utils"
if str(COLMAP_UTILS) not in sys.path:
    sys.path.insert(0, str(COLMAP_UTILS))

from read_write_model import (  # noqa: E402
    qvec2rotmat,
    read_cameras_binary,
    read_images_binary,
    read_points3D_binary,
    write_cameras_binary,
    write_images_binary,
    write_points3D_binary,
)


SCHEMA = "m3m_gsprior_colmap_camera_normalization_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_root(scene_root: Path) -> Path:
    nested = scene_root / "sparse" / "0"
    flat = scene_root / "sparse"
    if (nested / "images.bin").is_file():
        return nested
    if (flat / "images.bin").is_file():
        return flat
    raise FileNotFoundError(f"COLMAP binary model not found under {scene_root}")


def camera_center(image: Any) -> np.ndarray:
    rotation = qvec2rotmat(np.asarray(image.qvec, dtype=np.float64))
    return -rotation.T @ np.asarray(image.tvec, dtype=np.float64)


def derive_transform(reference_images: dict[int, Any]) -> dict[str, Any]:
    if not reference_images:
        raise ValueError("reference training model contains no cameras")
    ordered = [reference_images[key] for key in sorted(reference_images)]
    centers = np.stack([camera_center(image) for image in ordered], axis=0)
    center = centers.mean(axis=0)
    diagonal = float(np.linalg.norm(centers - center[None, :], axis=1).max())
    radius = diagonal * 1.1
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError(f"invalid NeRF++ radius: {radius}")
    scale = 1.0 / radius
    return {
        "rule": "upstream_gsprior_getNerfppNorm_training_camera_mean_max_distance_times_1p1",
        "reference_camera_count": len(ordered),
        "reference_camera_names_sha256": hashlib.sha256(
            "\n".join(sorted(image.name for image in ordered)).encode("utf-8")
        ).hexdigest(),
        "center_original_colmap": center.tolist(),
        "translate_original_colmap": (-center).tolist(),
        "diagonal_original_colmap": diagonal,
        "radius_original_colmap": radius,
        "normalized_units_per_original_colmap_unit": scale,
        "original_colmap_units_per_normalized_unit": radius,
        "expected_max_reference_camera_radius_normalized": 1.0 / 1.1,
    }


def validate_reused_transform(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("schema") == SCHEMA:
        value = value["transform"]
    required = {
        "center_original_colmap",
        "translate_original_colmap",
        "radius_original_colmap",
        "normalized_units_per_original_colmap_unit",
        "original_colmap_units_per_normalized_unit",
        "reference_camera_count",
        "reference_camera_names_sha256",
        "rule",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"reuse transform is missing keys: {missing}")
    scale = float(value["normalized_units_per_original_colmap_unit"])
    inverse = float(value["original_colmap_units_per_normalized_unit"])
    if not np.isclose(scale * inverse, 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("reuse transform scale and inverse scale disagree")
    return value


def transform_images(images: dict[int, Any], transform: dict[str, Any]) -> dict[int, Any]:
    translate = np.asarray(transform["translate_original_colmap"], dtype=np.float64)
    scale = float(transform["normalized_units_per_original_colmap_unit"])
    result = {}
    for key in sorted(images):
        image = images[key]
        rotation = qvec2rotmat(np.asarray(image.qvec, dtype=np.float64))
        tvec = scale * (np.asarray(image.tvec, dtype=np.float64) - rotation @ translate)
        result[key] = image._replace(tvec=tvec)
    return result


def transform_points(points: dict[int, Any], transform: dict[str, Any]) -> dict[int, Any]:
    translate = np.asarray(transform["translate_original_colmap"], dtype=np.float64)
    scale = float(transform["normalized_units_per_original_colmap_unit"])
    return {
        key: points[key]._replace(
            xyz=scale * (np.asarray(points[key].xyz, dtype=np.float64) + translate)
        )
        for key in sorted(points)
    }


def write_ply(points: dict[int, Any], path: Path) -> None:
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    rows = np.empty(len(points), dtype=dtype)
    for index, key in enumerate(sorted(points)):
        point = points[key]
        rows[index] = (*np.asarray(point.xyz, dtype=np.float32), 0.0, 0.0, 0.0, *point.rgb)
    PlyData([PlyElement.describe(rows, "vertex")], text=False).write(path)


def transform_ply(source: Path, destination: Path, transform: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    ply = PlyData.read(source)
    vertex = ply["vertex"].data.copy()
    original = np.stack(
        [np.asarray(vertex[axis], dtype=np.float64) for axis in ("x", "y", "z")],
        axis=1,
    )
    translate = np.asarray(transform["translate_original_colmap"], dtype=np.float64)
    scale = float(transform["normalized_units_per_original_colmap_unit"])
    normalized = scale * (original + translate[None, :])
    for index, axis in enumerate(("x", "y", "z")):
        vertex[axis] = normalized[:, index]
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(destination)
    written = PlyData.read(destination)["vertex"]
    written_xyz = np.stack(
        [np.asarray(written[axis], dtype=np.float64) for axis in ("x", "y", "z")],
        axis=1,
    )
    return original, written_xyz


def link_images(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    os.symlink(source.resolve(), destination, target_is_directory=True)


def identity(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_scene", type=Path, required=True)
    parser.add_argument("--reference_train_scene", type=Path, required=True)
    parser.add_argument("--output_scene", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reuse_transform_manifest", type=Path)
    parser.add_argument("--scene_id", required=True)
    parser.add_argument("--role", choices=("train", "evaluation_camera_domain"), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_scene = args.source_scene.resolve()
    reference_scene = args.reference_train_scene.resolve()
    output_scene = args.output_scene.resolve()
    manifest_path = args.manifest.resolve()
    if output_scene.exists() or manifest_path.exists():
        raise FileExistsError("output scene and manifest must not already exist")
    if output_scene == source_scene or output_scene == reference_scene:
        raise ValueError("normalization output must be a separate directory")

    source_model = model_root(source_scene)
    reference_model = model_root(reference_scene)
    source_images = read_images_binary(str(source_model / "images.bin"))
    source_cameras = read_cameras_binary(str(source_model / "cameras.bin"))
    source_points_bin = source_model / "points3D.bin"
    source_points_ply = source_model / "points3D.ply"
    if source_points_bin.is_file():
        source_points = read_points3D_binary(str(source_points_bin))
        # Evaluation-only camera roots deliberately carry an empty, eight-byte
        # points3D.bin for COLMAP-loader compatibility while retaining the
        # frozen initialization cloud in points3D.ply.  Treat that combination
        # as a PLY-backed scene instead of attempting to stack an empty dict.
        if not source_points and source_points_ply.is_file():
            source_points = None
            point_source_kind = "frozen_points3D_ply_with_empty_compatibility_bin"
        else:
            point_source_kind = "colmap_points3D_bin"
    elif source_points_ply.is_file():
        source_points = None
        point_source_kind = "frozen_points3D_ply_only"
    else:
        raise FileNotFoundError(f"neither points3D.bin nor points3D.ply exists in {source_model}")
    reference_images = read_images_binary(str(reference_model / "images.bin"))

    derived = derive_transform(reference_images)
    if args.reuse_transform_manifest:
        reused_document = json.loads(args.reuse_transform_manifest.read_text(encoding="utf-8"))
        transform = validate_reused_transform(reused_document)
        for key in (
            "reference_camera_count",
            "reference_camera_names_sha256",
            "center_original_colmap",
            "translate_original_colmap",
            "radius_original_colmap",
            "normalized_units_per_original_colmap_unit",
        ):
            if isinstance(derived[key], list):
                if not np.allclose(derived[key], transform[key], rtol=0.0, atol=1e-12):
                    raise ValueError(f"reused transform differs from reference cameras at {key}")
            elif isinstance(derived[key], float):
                if not np.isclose(derived[key], transform[key], rtol=0.0, atol=1e-12):
                    raise ValueError(f"reused transform differs from reference cameras at {key}")
            elif derived[key] != transform[key]:
                raise ValueError(f"reused transform differs from reference cameras at {key}")
        transform_origin = str(args.reuse_transform_manifest.resolve())
    else:
        transform = derived
        transform_origin = "derived_in_this_run"

    normalized_images = transform_images(source_images, transform)
    normalized_points = transform_points(source_points, transform) if source_points is not None else None

    output_scene.mkdir(parents=True)
    sparse = output_scene / "sparse"
    sparse.mkdir()
    write_cameras_binary(dict(sorted(source_cameras.items())), str(sparse / "cameras.bin"))
    write_images_binary(normalized_images, str(sparse / "images.bin"))
    if normalized_points is not None:
        write_points3D_binary(normalized_points, str(sparse / "points3D.bin"))
        write_ply(normalized_points, sparse / "points3D.ply")
        source_ply_xyz = np.stack(
            [np.asarray(source_points[key].xyz, dtype=np.float64) for key in sorted(source_points)],
            axis=0,
        )
        written_ply = PlyData.read(sparse / "points3D.ply")["vertex"]
        normalized_ply_xyz = np.stack(
            [np.asarray(written_ply[axis], dtype=np.float64) for axis in ("x", "y", "z")], axis=1
        )
    else:
        source_ply_xyz, normalized_ply_xyz = transform_ply(
            source_points_ply, sparse / "points3D.ply", transform
        )
        if source_points_bin.is_file():
            # Preserve the compatibility member so Graphdeco-family loaders
            # can read an empty COLMAP point table and then consume the already
            # materialized PLY without falling back to a missing text model.
            write_points3D_binary({}, str(sparse / "points3D.bin"))
    nested = sparse / "0"
    nested.mkdir()
    sparse_names = ["cameras.bin", "images.bin", "points3D.ply"]
    if (sparse / "points3D.bin").is_file():
        sparse_names.append("points3D.bin")
    for name in sparse_names:
        os.symlink(Path("..") / name, nested / name)
    link_images(source_scene / "images", output_scene / "images")
    for optional_name in ("split.json",):
        source_optional = source_scene / optional_name
        if source_optional.is_file():
            shutil.copy2(source_optional, output_scene / optional_name)

    check_images = read_images_binary(str(sparse / "images.bin"))
    check_points = (
        read_points3D_binary(str(sparse / "points3D.bin"))
        if (sparse / "points3D.bin").is_file()
        else None
    )
    scale = float(transform["normalized_units_per_original_colmap_unit"])
    translate = np.asarray(transform["translate_original_colmap"], dtype=np.float64)
    inverse_scale = float(transform["original_colmap_units_per_normalized_unit"])
    camera_residuals = []
    normalized_centers = []
    for key in sorted(source_images):
        expected = scale * (camera_center(source_images[key]) + translate)
        actual = camera_center(check_images[key])
        camera_residuals.append(float(np.linalg.norm(expected - actual)))
        normalized_centers.append(actual)
    restored_ply_xyz = normalized_ply_xyz * inverse_scale - translate[None, :]
    point_residuals = np.linalg.norm(restored_ply_xyz - source_ply_xyz, axis=1).tolist()

    reference_names = {image.name for image in reference_images.values()}
    source_names = {image.name for image in source_images.values()}
    normalized_centers_array = np.stack(normalized_centers, axis=0)
    output_files = {name: identity(sparse / name) for name in sparse_names}
    payload = {
        "schema": SCHEMA,
        "status": "PASS",
        "scene_id": args.scene_id,
        "role": args.role,
        "source_scene": str(source_scene),
        "source_model": str(source_model),
        "reference_train_scene": str(reference_scene),
        "reference_train_model": str(reference_model),
        "output_scene": str(output_scene),
        "transform_origin": transform_origin,
        "transform": transform,
        "source": {
            "camera_count": len(source_images),
            "point_count": len(source_ply_xyz),
            "point_source_kind": point_source_kind,
            "camera_name_set_contains_reference_train_set": reference_names <= source_names,
            "cameras_bin": identity(source_model / "cameras.bin"),
            "images_bin": identity(source_model / "images.bin"),
            "points3D_bin": identity(source_points_bin) if source_points_bin.is_file() else None,
            "points3D_ply": identity(source_points_ply) if source_points_ply.is_file() else None,
            "images_directory_target": str((source_scene / "images").resolve()),
        },
        "output": {
            "camera_count": len(check_images),
            "point_count": len(normalized_ply_xyz),
            "files": output_files,
            "images_are_directory_symlink": (output_scene / "images").is_symlink(),
            "images_directory_target": str((output_scene / "images").resolve()),
            "flat_and_sparse_zero_models_share_exact_files": True,
        },
        "validation": {
            "camera_center_transform_max_abs_residual": max(camera_residuals, default=0.0),
            "point_inverse_transform_max_abs_residual": max(point_residuals, default=0.0),
            "normalized_camera_coordinate_min": normalized_centers_array.min(axis=0).tolist(),
            "normalized_camera_coordinate_max": normalized_centers_array.max(axis=0).tolist(),
            "normalized_camera_radius_max": float(
                np.linalg.norm(normalized_centers_array, axis=1).max()
            ),
            "intrinsics_bytes_unchanged": sha256(source_model / "cameras.bin")
            == sha256(sparse / "cameras.bin"),
            "image_names_unchanged": sorted(source_names)
            == sorted(image.name for image in check_images.values()),
            "image_measurements_and_tracks_unchanged": all(
                np.array_equal(source_images[key].xys, check_images[key].xys)
                and np.array_equal(source_images[key].point3D_ids, check_images[key].point3D_ids)
                for key in source_images
            ),
            "point_tracks_and_reprojection_errors_unchanged": (
                all(
                    np.array_equal(source_points[key].image_ids, check_points[key].image_ids)
                    and np.array_equal(source_points[key].point2D_idxs, check_points[key].point2D_idxs)
                    and float(source_points[key].error) == float(check_points[key].error)
                    for key in source_points
                )
                if source_points is not None
                else "not_available_in_frozen_ply_only_release"
            ),
            "gcp_or_lidar_used": False,
            "image_pixels_resized_cropped_padded_or_reencoded": False,
        },
    }
    checks = payload["validation"]
    assert checks["camera_center_transform_max_abs_residual"] < 1e-10
    assert checks["point_inverse_transform_max_abs_residual"] < 1e-4
    assert checks["intrinsics_bytes_unchanged"]
    assert checks["image_names_unchanged"]
    assert checks["image_measurements_and_tracks_unchanged"]
    assert checks["point_tracks_and_reprojection_errors_unchanged"] is True or (
        checks["point_tracks_and_reprojection_errors_unchanged"]
        == "not_available_in_frozen_ply_only_release"
    )
    assert payload["output"]["images_are_directory_symlink"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
