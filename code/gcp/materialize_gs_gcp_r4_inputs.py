#!/usr/bin/env python3
"""Materialize leakage-safe, lossless R4 inputs for the clean GS-GCP track.

The generated training root is directly consumable by the unmodified official
3DGS loader with ``--resolution 1``.  It contains only train RGB images and
train camera poses.  The held-out RGB images and poses are written to a
separate test root and are never exposed to training.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import shutil
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np
import PIL
from PIL import Image as PILImage


COLMAP_UTILS = Path(__file__).resolve().parents[1] / "colmap" / "utils"
if str(COLMAP_UTILS) not in sys.path:
    sys.path.insert(0, str(COLMAP_UTILS))

from read_write_model import (  # noqa: E402
    Camera,
    Image as ColmapImage,
    read_cameras_binary,
    read_images_binary,
    write_cameras_binary,
    write_images_binary,
)


SCHEMA = "gs_gcp_r4_materialized_input_manifest_v1"
CONTRACT_SCHEMA = "gs_gcp_r4_input_materialization_contract_v1"
RULE_ID = "graphdeco_quarter_resolution_v1"
HOLDOUT_SEMANTICS = "image_loss_holdout_under_shared_all_image_sfm_v1"
RELEASE_DIGEST = "513f8999fe4b110f15bcbecad7932895781cee755ee9ccd7a14ff10298546d75"
REQUIRED_PILLOW_VERSION = "11.1.0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(payload: dict[str, Any]) -> bytes:
    clean = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    return json.dumps(clean, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def canonical_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def quarter_dimensions(width: int, height: int) -> tuple[int, int]:
    if not isinstance(width, int) or not isinstance(height, int):
        raise TypeError("image dimensions must be integers")
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    return round(width / 4), round(height / 4)


def _fov(focal: float, pixels: int) -> float:
    return 2.0 * math.atan(float(pixels) / (2.0 * float(focal)))


def scale_pinhole_camera(camera: Camera, target_width: int, target_height: int) -> Camera:
    """Scale a PINHOLE camera without changing its field of view or rays."""
    if camera.model != "PINHOLE":
        raise ValueError(f"clean R4 contract only admits PINHOLE cameras, got {camera.model}")
    if len(camera.params) != 4:
        raise ValueError(f"PINHOLE camera {camera.id} must have four parameters")
    scale_x = target_width / int(camera.width)
    scale_y = target_height / int(camera.height)
    fx, fy, cx, cy = (float(value) for value in camera.params)
    return Camera(
        id=int(camera.id),
        model=camera.model,
        width=int(target_width),
        height=int(target_height),
        params=np.asarray([fx * scale_x, fy * scale_y, cx * scale_x, cy * scale_y], dtype=np.float64),
    )


def camera_equivalence_error(source: Camera, scaled: Camera) -> dict[str, float]:
    source_fx, source_fy, source_cx, source_cy = (float(value) for value in source.params)
    scaled_fx, scaled_fy, scaled_cx, scaled_cy = (float(value) for value in scaled.params)
    return {
        "fov_x_abs_error_rad": abs(_fov(source_fx, int(source.width)) - _fov(scaled_fx, int(scaled.width))),
        "fov_y_abs_error_rad": abs(_fov(source_fy, int(source.height)) - _fov(scaled_fy, int(scaled.height))),
        "normalized_cx_abs_error": abs(source_cx / source_fx - scaled_cx / scaled_fx),
        "normalized_cy_abs_error": abs(source_cy / source_fy - scaled_cy / scaled_fy),
    }


def read_pose_only_images_binary(path: Path, expected_names: set[str]) -> dict[str, ColmapImage]:
    """Read COLMAP poses while skipping large POINTS2D payloads."""
    selected: dict[str, ColmapImage] = {}
    with path.open("rb") as handle:
        raw_count = handle.read(8)
        if len(raw_count) != 8:
            raise ValueError(f"truncated COLMAP images file: {path}")
        image_count = struct.unpack("<Q", raw_count)[0]
        for _ in range(image_count):
            raw_properties = handle.read(64)
            if len(raw_properties) != 64:
                raise ValueError(f"truncated COLMAP image record: {path}")
            properties = struct.unpack("<idddddddi", raw_properties)
            name_bytes = bytearray()
            while True:
                value = handle.read(1)
                if not value:
                    raise ValueError(f"unterminated COLMAP image name: {path}")
                if value == b"\x00":
                    break
                name_bytes.extend(value)
            name = name_bytes.decode("utf-8")
            raw_points = handle.read(8)
            if len(raw_points) != 8:
                raise ValueError(f"truncated COLMAP point count: {path}")
            point_count = struct.unpack("<Q", raw_points)[0]
            handle.seek(24 * point_count, os.SEEK_CUR)
            if name not in expected_names:
                continue
            if name in selected:
                raise ValueError(f"duplicate COLMAP image name: {name}")
            selected[name] = ColmapImage(
                id=int(properties[0]),
                qvec=np.asarray(properties[1:5], dtype=np.float64),
                tvec=np.asarray(properties[5:8], dtype=np.float64),
                camera_id=int(properties[8]),
                name=name,
                xys=np.empty((0, 2), dtype=np.float64),
                point3D_ids=np.empty((0,), dtype=np.int64),
            )
    missing = sorted(expected_names - set(selected))
    if missing:
        raise ValueError(f"split contains images absent from COLMAP model: {missing[:5]}")
    return selected


def _scene_row(split_manifest: dict[str, Any], scene: str) -> dict[str, Any]:
    matches = [row for row in split_manifest.get("scenes", []) if row.get("scene") == scene]
    if len(matches) != 1:
        raise ValueError(f"split manifest must contain exactly one row for {scene}")
    return matches[0]


def _validate_split_identity(split_manifest: dict[str, Any]) -> None:
    if split_manifest.get("schema") != "gs_gcp_rgb_holdout_split_manifest_v1":
        raise ValueError("unknown split manifest schema")
    if split_manifest.get("split_protocol") != "gs_gcp_rgb_holdout_split_v1":
        raise ValueError("split protocol mismatch")
    if split_manifest.get("holdout_semantics") != HOLDOUT_SEMANTICS:
        raise ValueError("holdout semantics mismatch")
    if split_manifest.get("release_root_digest") != RELEASE_DIGEST:
        raise ValueError("split release digest mismatch")
    if canonical_sha256(split_manifest) != split_manifest.get("manifest_sha256"):
        raise ValueError("split canonical SHA mismatch")


def _validate_source_identity(
    source_root: Path,
    scene: str,
    source_manifest: dict[str, Any],
) -> dict[str, str]:
    release = source_manifest.get("release", {})
    if release.get("payload_root_digest_sha256") != RELEASE_DIGEST:
        raise ValueError("training source release digest mismatch")
    row = source_manifest.get("scenes", {}).get(scene)
    if not isinstance(row, dict):
        raise ValueError(f"training source manifest has no scene {scene}")
    sparse = source_root / "sparse" / "0"
    mapping = {
        "cameras.bin": "cameras_bin_sha256",
        "images.bin": "images_bin_sha256",
        "points3D.ply": "points3d_ply_sha256",
    }
    hashes: dict[str, str] = {}
    for filename, field in mapping.items():
        path = sparse / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != row.get(field):
            raise ValueError(f"source {filename} SHA mismatch")
        hashes[filename] = actual
    return hashes


def _materialize_image(source: Path, target: Path, assignment: dict[str, Any]) -> dict[str, Any]:
    payload = source.read_bytes()
    source_sha = hashlib.sha256(payload).hexdigest()
    if source_sha != assignment.get("image_sha256"):
        raise ValueError(f"source image SHA mismatch: {source.name}")
    if len(payload) != int(assignment.get("image_bytes", -1)):
        raise ValueError(f"source image byte count mismatch: {source.name}")
    with PILImage.open(io.BytesIO(payload)) as image:
        if image.mode != "RGB":
            raise ValueError(f"official loader requires decoded RGB: {source} mode={image.mode}")
        expected_size = (int(assignment["decoded_width"]), int(assignment["decoded_height"]))
        if image.size != expected_size:
            raise ValueError(f"decoded dimensions mismatch: {source} {image.size} != {expected_size}")
        target_size = quarter_dimensions(*image.size)
        resized = image.resize(target_size)
        expected_pixels = np.asarray(resized).copy()
    target.parent.mkdir(parents=True, exist_ok=True)
    PILImage.fromarray(expected_pixels, mode="RGB").save(
        target,
        format="PNG",
        compress_level=6,
        optimize=False,
    )
    with PILImage.open(target) as reopened:
        actual_pixels = np.asarray(reopened).copy()
        if reopened.mode != "RGB" or reopened.size != target_size:
            raise ValueError(f"materialized PNG decode mismatch: {target}")
        if not np.array_equal(expected_pixels, actual_pixels):
            raise ValueError(f"lossless R4 pixel check failed: {target}")
    return {
        "source_image_sha256": source_sha,
        "source_image_bytes": len(payload),
        "decoded_width": expected_size[0],
        "decoded_height": expected_size[1],
        "r4_width": target_size[0],
        "r4_height": target_size[1],
        "r4_rgb_uint8_sha256": hashlib.sha256(expected_pixels.tobytes(order="C")).hexdigest(),
        "r4_png_sha256": sha256_file(target),
        "r4_png_bytes": target.stat().st_size,
    }


def materialize_scene(
    *,
    split_manifest_path: Path,
    source_manifest_path: Path,
    scene: str,
    source_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    if PIL.__version__ != REQUIRED_PILLOW_VERSION:
        raise RuntimeError(
            f"Pillow {REQUIRED_PILLOW_VERSION} is required, found {PIL.__version__}"
        )
    if output_root.exists():
        raise FileExistsError(output_root)
    incomplete_root = output_root

    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    _validate_split_identity(split_manifest)
    scene_row = _scene_row(split_manifest, scene)
    source_hashes = _validate_source_identity(source_root, scene, source_manifest)
    assignments = scene_row.get("assignments", [])
    if not assignments:
        raise ValueError(f"scene has no split assignments: {scene}")
    names = [str(row["image_name"]) for row in assignments]
    if len(names) != len(set(names)):
        raise ValueError("split image names are not unique")
    role_names = {
        role: [str(row["image_name"]) for row in assignments if row.get("split_role") == role]
        for role in ("train", "test")
    }
    if set(role_names["train"]) & set(role_names["test"]):
        raise ValueError("train and test image sets overlap")
    if len(role_names["test"]) != math.ceil(len(assignments) / 8):
        raise ValueError("test split count does not match ceil(N/8)")

    sparse = source_root / "sparse" / "0"
    source_cameras = read_cameras_binary(sparse / "cameras.bin")
    source_images = read_pose_only_images_binary(sparse / "images.bin", set(names))
    by_assignment = {str(row["image_name"]): row for row in assignments}
    target_dimensions_by_camera: dict[int, tuple[int, int]] = {}
    for name in names:
        pose = source_images[name]
        row = by_assignment[name]
        if int(pose.id) != int(row["image_id"]):
            raise ValueError(f"COLMAP image ID mismatch: {name}")
        camera = source_cameras.get(int(pose.camera_id))
        if camera is None:
            raise ValueError(f"missing camera {pose.camera_id} for {name}")
        decoded_size = (int(row["decoded_width"]), int(row["decoded_height"]))
        if (int(camera.width), int(camera.height)) != decoded_size:
            raise ValueError(f"camera/image dimensions differ for {name}")
        target_size = quarter_dimensions(*decoded_size)
        previous = target_dimensions_by_camera.setdefault(int(pose.camera_id), target_size)
        if previous != target_size:
            raise ValueError(f"camera {pose.camera_id} maps to multiple R4 dimensions")

    scaled_cameras = {
        camera_id: scale_pinhole_camera(source_cameras[camera_id], *target_size)
        for camera_id, target_size in sorted(target_dimensions_by_camera.items())
    }
    camera_checks = []
    for camera_id, camera in scaled_cameras.items():
        errors = camera_equivalence_error(source_cameras[camera_id], camera)
        if max(errors.values(), default=0.0) > 1e-12:
            raise ValueError(f"R4 camera equivalence failed for camera {camera_id}: {errors}")
        camera_checks.append({
            "camera_id": camera_id,
            "source_width": int(source_cameras[camera_id].width),
            "source_height": int(source_cameras[camera_id].height),
            "r4_width": int(camera.width),
            "r4_height": int(camera.height),
            **errors,
        })

    incomplete_root.mkdir(parents=True)
    incomplete_marker = incomplete_root / "MATERIALIZATION_INCOMPLETE"
    incomplete_marker.write_text(
        "This directory is not a valid formal input until R4_INPUT_MANIFEST.json exists and this marker is removed.\n",
        encoding="utf-8",
    )
    image_records: list[dict[str, Any]] = []
    role_records: list[dict[str, Any]] = []
    for role in ("train", "test"):
        role_root = incomplete_root / role
        model_root = role_root / "sparse" / "0"
        model_root.mkdir(parents=True)
        selected_cameras: dict[int, Camera] = {}
        selected_images: dict[int, ColmapImage] = {}
        for source_name in role_names[role]:
            assignment = by_assignment[source_name]
            source_pose = source_images[source_name]
            target_name = f"{Path(source_name).stem}.png"
            target_image = role_root / "images" / target_name
            image_record = _materialize_image(source_root / "images" / source_name, target_image, assignment)
            sanitized_pose = ColmapImage(
                id=int(source_pose.id),
                qvec=np.asarray(source_pose.qvec, dtype=np.float64),
                tvec=np.asarray(source_pose.tvec, dtype=np.float64),
                camera_id=int(source_pose.camera_id),
                name=target_name,
                xys=np.empty((0, 2), dtype=np.float64),
                point3D_ids=np.empty((0,), dtype=np.int64),
            )
            selected_images[int(sanitized_pose.id)] = sanitized_pose
            selected_cameras[int(sanitized_pose.camera_id)] = scaled_cameras[int(sanitized_pose.camera_id)]
            image_records.append({
                "role": role,
                "image_id": int(source_pose.id),
                "camera_id": int(source_pose.camera_id),
                "source_name": source_name,
                "r4_name": target_name,
                "r4_relative_path": target_image.relative_to(incomplete_root).as_posix(),
                **image_record,
            })
        write_cameras_binary(dict(sorted(selected_cameras.items())), model_root / "cameras.bin")
        write_images_binary(dict(sorted(selected_images.items())), model_root / "images.bin")
        shutil.copy2(sparse / "points3D.ply", model_root / "points3D.ply")
        role_records.append({
            "role": role,
            "root": role,
            "image_count": len(selected_images),
            "camera_count": len(selected_cameras),
            "cameras_bin_sha256": sha256_file(model_root / "cameras.bin"),
            "images_bin_sha256": sha256_file(model_root / "images.bin"),
            "points3d_ply_sha256": sha256_file(model_root / "points3D.ply"),
            "points2d_tracks_present": False,
            "points3d_bin_present": False,
        })

    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "scene": scene,
        "release_root_digest_sha256": RELEASE_DIGEST,
        "resolution_rule_id": RULE_ID,
        "holdout_semantics": HOLDOUT_SEMANTICS,
        "source_split_manifest_file_sha256": sha256_file(split_manifest_path),
        "source_split_manifest_canonical_sha256": split_manifest["manifest_sha256"],
        "source_training_manifest_file_sha256": sha256_file(source_manifest_path),
        "source_sparse_sha256": source_hashes,
        "generator": {
            "script": "code/gcp/materialize_gs_gcp_r4_inputs.py",
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pillow": PIL.__version__,
        },
        "image_transform": {
            "decode": "PIL.Image.open; decoded matrix orientation; ignore EXIF orientation",
            "dimension_formula": "(round(decoded_width / 4), round(decoded_height / 4))",
            "resize_call": "PIL.Image.Image.resize(size), resample omitted",
            "effective_resampling": "BICUBIC",
            "output": "lossless RGB PNG, compress_level=6, optimize=false",
        },
        "camera_transform": {
            "model": "PINHOLE",
            "dimensions": "W'=round(W/4), H'=round(H/4)",
            "intrinsics": "fx'=fx*W'/W, fy'=fy*H'/H, cx'=cx*W'/W, cy'=cy*H'/H",
            "extrinsics": "bit-preserved float64 qvec and tvec",
            "points2d_tracks": "removed from both train and test models",
            "shared_initialization": "source points3D.ply copied byte-for-byte into both roots",
        },
        "official_3dgs_binding": {
            "training_root": "train",
            "heldout_root": "test",
            "resolution_argument": 1,
            "eval": False,
            "official_training_source_modified": False,
            "reference_equivalence": "full-resolution source JPEG with official --resolution 4",
        },
        "camera_equivalence": camera_checks,
        "full_view_count": len(assignments),
        "train_view_count": len(role_names["train"]),
        "test_view_count": len(role_names["test"]),
        "roles": role_records,
        "images": image_records,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    write_json(incomplete_root / "R4_INPUT_MANIFEST.json", manifest)
    incomplete_marker.unlink()
    return manifest


def verify_materialization(input_root: Path, *, decode_pixels: bool = True) -> dict[str, Any]:
    errors: list[str] = []
    manifest_path = input_root / "R4_INPUT_MANIFEST.json"
    if not manifest_path.is_file():
        return {"schema": "gs_gcp_r4_materialization_verification_v1", "passed": False, "errors": ["manifest missing"]}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        errors.append("unknown manifest schema")
    if canonical_sha256(manifest) != manifest.get("manifest_sha256"):
        errors.append("manifest canonical SHA mismatch")
    role_names: dict[str, set[str]] = {"train": set(), "test": set()}
    for row in manifest.get("images", []):
        role = str(row.get("role"))
        if role not in role_names:
            errors.append(f"unknown image role: {role}")
            continue
        relative = Path(str(row.get("r4_relative_path", "")))
        target = input_root / relative
        if not target.is_file():
            errors.append(f"missing R4 image: {relative.as_posix()}")
            continue
        if sha256_file(target) != row.get("r4_png_sha256"):
            errors.append(f"R4 PNG SHA mismatch: {relative.as_posix()}")
        role_names[role].add(str(row.get("r4_name")))
        if decode_pixels:
            with PILImage.open(target) as image:
                if image.mode != "RGB" or image.size != (int(row["r4_width"]), int(row["r4_height"])):
                    errors.append(f"R4 PNG decode mismatch: {relative.as_posix()}")
                else:
                    pixel_sha = hashlib.sha256(np.asarray(image).tobytes(order="C")).hexdigest()
                    if pixel_sha != row.get("r4_rgb_uint8_sha256"):
                        errors.append(f"R4 pixel SHA mismatch: {relative.as_posix()}")
    if role_names["train"] & role_names["test"]:
        errors.append("train/test R4 names overlap")
    for role_row in manifest.get("roles", []):
        role = str(role_row.get("role"))
        model_root = input_root / role / "sparse" / "0"
        for filename, field in (
            ("cameras.bin", "cameras_bin_sha256"),
            ("images.bin", "images_bin_sha256"),
            ("points3D.ply", "points3d_ply_sha256"),
        ):
            path = model_root / filename
            if not path.is_file() or sha256_file(path) != role_row.get(field):
                errors.append(f"{role} {filename} identity mismatch")
        if (model_root / "points3D.bin").exists():
            errors.append(f"{role} unexpectedly contains points3D.bin")
        try:
            cameras = read_cameras_binary(model_root / "cameras.bin")
            images = read_images_binary(model_root / "images.bin")
            if len(images) != int(role_row.get("image_count", -1)):
                errors.append(f"{role} image count mismatch")
            if len(cameras) != int(role_row.get("camera_count", -1)):
                errors.append(f"{role} camera count mismatch")
            if any(len(image.point3D_ids) for image in images.values()):
                errors.append(f"{role} contains POINTS2D tracks")
        except (OSError, ValueError, struct.error) as exc:
            errors.append(f"{role} COLMAP model unreadable: {exc}")
    return {
        "schema": "gs_gcp_r4_materialization_verification_v1",
        "scene": manifest.get("scene"),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "passed": not errors,
        "error_count": len(errors),
        "errors": errors,
    }


def validate_contract(contract: dict[str, Any], repo_root: Path | None = None) -> list[str]:
    errors: list[str] = []
    required = {
        "schema": CONTRACT_SCHEMA,
        "status": "frozen_for_clean_official_3dgs_r4_qualification",
        "release_root_digest_sha256": RELEASE_DIGEST,
        "resolution_rule_id": RULE_ID,
        "pillow_version": REQUIRED_PILLOW_VERSION,
        "official_3dgs_resolution_argument": 1,
        "official_training_source_modified": False,
        "train_test_physical_separation": True,
        "points2d_tracks_present": False,
    }
    for key, expected in required.items():
        if contract.get(key) != expected:
            errors.append(f"{key} must equal {expected!r}")
    if contract.get("image_output") != "lossless_rgb_png":
        errors.append("image_output must be lossless_rgb_png")
    if contract.get("camera_model") != "PINHOLE":
        errors.append("camera_model must be PINHOLE")
    script_rel = contract.get("generator_script")
    if not isinstance(script_rel, str) or not script_rel:
        errors.append("generator_script is missing")
    elif repo_root is not None:
        script_path = (repo_root / script_rel).resolve()
        if not script_path.is_file():
            errors.append("generator script is missing")
        elif sha256_file(script_path) != contract.get("generator_script_sha256"):
            errors.append("generator script SHA mismatch")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--split_manifest", required=True, type=Path)
    materialize.add_argument("--source_manifest", required=True, type=Path)
    materialize.add_argument("--scene", required=True)
    materialize.add_argument("--source_root", required=True, type=Path)
    materialize.add_argument("--output_root", required=True, type=Path)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--input_root", required=True, type=Path)
    verify.add_argument("--skip_pixel_decode", action="store_true")
    args = parser.parse_args()
    if args.command == "materialize":
        payload = materialize_scene(
            split_manifest_path=args.split_manifest.resolve(),
            source_manifest_path=args.source_manifest.resolve(),
            scene=args.scene,
            source_root=args.source_root.resolve(),
            output_root=args.output_root.resolve(),
        )
        result = {
            "schema": "gs_gcp_r4_materialization_result_v1",
            "scene": payload["scene"],
            "manifest_sha256": payload["manifest_sha256"],
            "train_view_count": payload["train_view_count"],
            "test_view_count": payload["test_view_count"],
            "output_root": str(args.output_root.resolve()),
        }
    else:
        result = verify_materialization(args.input_root.resolve(), decode_pixels=not args.skip_pixel_decode)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
