#!/usr/bin/env python3
"""Materialize byte-preserving COLMAP-native-quarter benchmark inputs.

The source scene is the frozen, train-ready COLMAP ``image_undistorter``
output.  This tool does not resize, decode/re-encode, crop, pad, or transpose
the images.  It partitions the already-native-quarter JPEG files according to
the frozen RGB holdout manifest and writes pose-only COLMAP models for the
train and held-out roots.  Both roots share the exact frozen initial PLY.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import struct
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image as PILImage


COLMAP_UTILS = Path(__file__).resolve().parents[1] / "colmap" / "utils"
if str(COLMAP_UTILS) not in sys.path:
    sys.path.insert(0, str(COLMAP_UTILS))

from read_write_model import (  # noqa: E402
    Image as ColmapImage,
    read_cameras_binary,
    read_images_binary,
    write_cameras_binary,
    write_images_binary,
)


SCHEMA = "gs_gcp_colmap_native_quarter_materialized_input_manifest_v1"
SPLIT_SCHEMA = "gs_gcp_rgb_holdout_split_manifest_v1"
SPLIT_PROTOCOL = "gs_gcp_rgb_holdout_split_v1"
HOLDOUT_SEMANTICS = "image_loss_holdout_under_shared_all_image_sfm_v1"
RELEASE_DIGEST = "513f8999fe4b110f15bcbecad7932895781cee755ee9ccd7a14ff10298546d75"
RULE_ID = "colmap_4_0_4_image_undistorter_max_1414_v1"
PIXEL_DOMAIN = "colmap_4_0_4_image_undistorter_pinhole_max_1414"
SHA_LINE = re.compile(r"^([0-9a-f]{64})\s+\*?(.+?)\s*$")


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


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _scene_row(split_manifest: dict[str, Any], scene: str) -> dict[str, Any]:
    matches = [row for row in split_manifest.get("scenes", []) if row.get("scene") == scene]
    if len(matches) != 1:
        raise ValueError(f"split manifest must contain exactly one row for {scene}")
    return matches[0]


def _validate_split_manifest(split_manifest: dict[str, Any]) -> None:
    _require(split_manifest.get("schema") == SPLIT_SCHEMA, "unknown split manifest schema")
    _require(split_manifest.get("split_protocol") == SPLIT_PROTOCOL, "split protocol mismatch")
    _require(split_manifest.get("holdout_semantics") == HOLDOUT_SEMANTICS, "holdout semantics mismatch")
    _require(split_manifest.get("release_root_digest") == RELEASE_DIGEST, "split release digest mismatch")
    _require(canonical_sha256(split_manifest) == split_manifest.get("manifest_sha256"), "split canonical SHA mismatch")


def parse_image_sha256_manifest(path: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        match = SHA_LINE.match(raw)
        if match is None:
            raise ValueError(f"invalid image SHA line {line_number}: {raw!r}")
        digest, relative = match.groups()
        pure = PurePosixPath(relative.replace("\\", "/"))
        if len(pure.parts) != 2 or pure.parts[0] != "images":
            raise ValueError(f"image SHA path is outside images/: {relative}")
        name = pure.name
        if name in hashes:
            raise ValueError(f"duplicate image SHA entry: {name}")
        hashes[name] = digest
    if not hashes:
        raise ValueError("image SHA manifest is empty")
    return hashes


def read_pose_only_images_binary(path: Path, expected_names: set[str]) -> dict[str, ColmapImage]:
    """Read selected poses while skipping the potentially huge POINTS2D payload."""
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


def _audit_model_hashes(source_root: Path, audit: dict[str, Any]) -> dict[str, str]:
    sparse = source_root / "sparse" / "0"
    hashes: dict[str, str] = {}
    for filename in ("cameras.bin", "images.bin", "points3D.bin", "points3D.ply"):
        path = sparse / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        expected = audit.get("model_files", {}).get(filename, {}).get("sha256")
        _require(actual == expected, f"source {filename} SHA mismatch")
        hashes[filename] = actual
    return hashes


def _validate_package_audit(
    source_root: Path,
    scene: str,
    audit_path: Path,
    image_sha_path: Path,
) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    _require(audit.get("schema") == "gs-gcp-colmap-native-quarter-package-audit-v2", "unknown package audit schema")
    _require(audit.get("scene") == scene, "package audit scene mismatch")
    _require(audit.get("status") == "pass", "package audit did not pass")
    _require(audit.get("standard_colmap_layout") is True, "source is not a standard COLMAP layout")
    _require(audit.get("train_ready_for_standard_colmap_loaders") is True, "source is not train-ready")
    generation = audit.get("image_generation", {})
    _require(generation.get("generator") == "COLMAP 4.0.4 image_undistorter", "image generator mismatch")
    _require(generation.get("max_image_size") == 1414, "native-quarter max image size mismatch")
    expected_image_manifest = audit.get("image_sha256_manifest", {}).get("sha256")
    _require(sha256_file(image_sha_path) == expected_image_manifest, "image SHA manifest identity mismatch")
    image_hashes = parse_image_sha256_manifest(image_sha_path)
    _require(len(image_hashes) == audit.get("counts", {}).get("images"), "image SHA count mismatch")
    model_hashes = _audit_model_hashes(source_root, audit)
    return audit, image_hashes, model_hashes


def _materialize_file(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        os.link(source, target)
    elif mode == "copy":
        shutil.copy2(source, target)
    else:  # pragma: no cover - guarded by argparse and direct-call tests
        raise ValueError(f"unknown materialization mode: {mode}")


def materialize_scene(
    *,
    split_manifest_path: Path,
    scene: str,
    source_root: Path,
    output_root: Path,
    package_audit_path: Path | None = None,
    image_sha256_path: Path | None = None,
    file_mode: str = "copy",
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(output_root)
    package_audit_path = package_audit_path or source_root / "evidence" / "PACKAGE_AUDIT.json"
    image_sha256_path = image_sha256_path or source_root / "evidence" / "images.sha256"
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    _validate_split_manifest(split_manifest)
    scene_row = _scene_row(split_manifest, scene)
    audit, image_hashes, model_hashes = _validate_package_audit(
        source_root, scene, package_audit_path, image_sha256_path
    )

    assignments = scene_row.get("assignments", [])
    _require(bool(assignments), f"scene has no split assignments: {scene}")
    names = [str(row["image_name"]) for row in assignments]
    _require(len(names) == len(set(names)), "split image names are not unique")
    _require(set(names) == set(image_hashes), "split and native-quarter image sets differ")
    role_names = {
        role: [str(row["image_name"]) for row in assignments if row.get("split_role") == role]
        for role in ("train", "test")
    }
    _require(not (set(role_names["train"]) & set(role_names["test"])), "train and test image sets overlap")
    _require(len(role_names["test"]) == math.ceil(len(assignments) / 8), "test split count mismatch")
    _require(len(role_names["train"]) + len(role_names["test"]) == len(assignments), "unknown split role")

    sparse = source_root / "sparse" / "0"
    source_cameras = read_cameras_binary(sparse / "cameras.bin")
    source_images = read_pose_only_images_binary(sparse / "images.bin", set(names))
    by_assignment = {str(row["image_name"]): row for row in assignments}

    output_root.mkdir(parents=True)
    marker = output_root / "MATERIALIZATION_INCOMPLETE"
    marker.write_text("Remove only after NATIVE_QUARTER_INPUT_MANIFEST.json passes verification.\n", encoding="utf-8")
    image_records: list[dict[str, Any]] = []
    role_records: list[dict[str, Any]] = []
    for role in ("train", "test"):
        role_root = output_root / role
        model_root = role_root / "sparse" / "0"
        model_root.mkdir(parents=True)
        selected_cameras: dict[int, Any] = {}
        selected_images: dict[int, ColmapImage] = {}
        for name in role_names[role]:
            assignment = by_assignment[name]
            pose = source_images[name]
            _require(int(pose.id) == int(assignment["image_id"]), f"COLMAP image ID mismatch: {name}")
            _require(int(pose.camera_id) == int(assignment["camera_id"]), f"COLMAP camera ID mismatch: {name}")
            camera = source_cameras.get(int(pose.camera_id))
            if camera is None:
                raise ValueError(f"missing camera {pose.camera_id} for {name}")
            source_image = source_root / "images" / name
            if not source_image.is_file():
                raise FileNotFoundError(source_image)
            actual_sha = sha256_file(source_image)
            _require(actual_sha == image_hashes[name], f"native-quarter image SHA mismatch: {name}")
            with PILImage.open(source_image) as image:
                _require(image.mode == "RGB", f"native-quarter image is not RGB: {name}")
                width, height = image.size
            _require((int(camera.width), int(camera.height)) == (width, height), f"camera/image dimensions differ: {name}")
            target_image = role_root / "images" / name
            _materialize_file(source_image, target_image, file_mode)
            selected_cameras[int(camera.id)] = camera
            selected_images[int(pose.id)] = ColmapImage(
                id=int(pose.id),
                qvec=np.asarray(pose.qvec, dtype=np.float64),
                tvec=np.asarray(pose.tvec, dtype=np.float64),
                camera_id=int(pose.camera_id),
                name=name,
                xys=np.empty((0, 2), dtype=np.float64),
                point3D_ids=np.empty((0,), dtype=np.int64),
            )
            image_records.append({
                "role": role,
                "image_id": int(pose.id),
                "camera_id": int(pose.camera_id),
                "image_name": name,
                "relative_path": target_image.relative_to(output_root).as_posix(),
                "width": width,
                "height": height,
                "jpeg_bytes": source_image.stat().st_size,
                "jpeg_sha256": actual_sha,
            })
        write_cameras_binary(dict(sorted(selected_cameras.items())), model_root / "cameras.bin")
        write_images_binary(dict(sorted(selected_images.items())), model_root / "images.bin")
        _materialize_file(sparse / "points3D.ply", model_root / "points3D.ply", file_mode)
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

    decoded_sizes = audit.get("decoded_images", {}).get("sizes", [])
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "scene": scene,
        "release_root_digest_sha256": RELEASE_DIGEST,
        "resolution_rule_id": RULE_ID,
        "pixel_domain": PIXEL_DOMAIN,
        "holdout_semantics": HOLDOUT_SEMANTICS,
        "source_split_manifest_file_sha256": sha256_file(split_manifest_path),
        "source_split_manifest_canonical_sha256": split_manifest["manifest_sha256"],
        "source_package_audit_file_sha256": sha256_file(package_audit_path),
        "source_image_manifest_file_sha256": sha256_file(image_sha256_path),
        "source_model_sha256": model_hashes,
        "generator": {
            "script": "code/gcp/materialize_gs_gcp_native_quarter_inputs.py",
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
        "file_materialization": {
            "mode": file_mode,
            "semantic_identity": "file bytes are normative; hardlinks and copies are equivalent",
        },
        "image_transform": {
            "operation": "byte-preserving materialization of COLMAP 4.0.4 image_undistorter JPEG output",
            "resize": False,
            "crop": False,
            "pad": False,
            "exif_transpose": False,
            "reencode": False,
            "decoded_sizes": decoded_sizes,
        },
        "camera_transform": {
            "intrinsics": "byte-equivalent native camera record",
            "extrinsics": "float64 qvec/tvec preserved",
            "points2d_tracks": "removed from both train and test models",
            "shared_initialization": "source points3D.ply materialized byte-for-byte into both roots",
        },
        "official_3dgs_binding": {
            "training_root": "train",
            "heldout_root": "test",
            "resolution_argument": 1,
            "eval": False,
            "official_training_source_modified": False,
        },
        "full_view_count": len(assignments),
        "train_view_count": len(role_names["train"]),
        "test_view_count": len(role_names["test"]),
        "roles": role_records,
        "images": image_records,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    write_json(output_root / "NATIVE_QUARTER_INPUT_MANIFEST.json", manifest)
    marker.unlink()
    verification = verify_materialization(output_root, decode_images=False)
    if not verification["passed"]:
        marker.write_text(
            "Post-materialization verification failed; this directory is not a valid formal input.\n",
            encoding="utf-8",
        )
        raise ValueError(f"post-materialization verification failed: {verification['errors'][:3]}")
    return manifest


def verify_materialization(input_root: Path, *, decode_images: bool = True) -> dict[str, Any]:
    errors: list[str] = []
    manifest_path = input_root / "NATIVE_QUARTER_INPUT_MANIFEST.json"
    if not manifest_path.is_file():
        return {"schema": "gs_gcp_native_quarter_materialization_verification_v1", "passed": False, "errors": ["manifest missing"]}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        errors.append("unknown manifest schema")
    if canonical_sha256(manifest) != manifest.get("manifest_sha256"):
        errors.append("manifest canonical SHA mismatch")
    if (input_root / "MATERIALIZATION_INCOMPLETE").exists():
        errors.append("incomplete marker is present")

    role_names: dict[str, set[str]] = {"train": set(), "test": set()}
    for row in manifest.get("images", []):
        role = str(row.get("role"))
        name = str(row.get("image_name"))
        if role not in role_names:
            errors.append(f"unknown role for {name}: {role}")
            continue
        if name in role_names[role]:
            errors.append(f"duplicate manifest image: {role}/{name}")
        role_names[role].add(name)
        path = input_root / str(row.get("relative_path"))
        if not path.is_file():
            errors.append(f"missing image: {path}")
            continue
        if path.stat().st_size != int(row.get("jpeg_bytes", -1)):
            errors.append(f"image byte count mismatch: {name}")
        elif sha256_file(path) != row.get("jpeg_sha256"):
            errors.append(f"image SHA mismatch: {name}")
        if decode_images:
            try:
                with PILImage.open(path) as image:
                    if image.mode != "RGB" or image.size != (int(row["width"]), int(row["height"])):
                        errors.append(f"decoded image mismatch: {name}")
            except Exception as exc:  # pragma: no cover - exact PIL errors vary
                errors.append(f"image decode failed: {name}: {exc}")

    if role_names["train"] & role_names["test"]:
        errors.append("train/test image overlap")
    if len(role_names["train"]) != int(manifest.get("train_view_count", -1)):
        errors.append("train image count mismatch")
    if len(role_names["test"]) != int(manifest.get("test_view_count", -1)):
        errors.append("test image count mismatch")
    if len(role_names["train"] | role_names["test"]) != int(manifest.get("full_view_count", -1)):
        errors.append("full image count mismatch")

    role_rows = {str(row.get("role")): row for row in manifest.get("roles", [])}
    for role in ("train", "test"):
        row = role_rows.get(role, {})
        model_root = input_root / role / "sparse" / "0"
        for filename, field in (
            ("cameras.bin", "cameras_bin_sha256"),
            ("images.bin", "images_bin_sha256"),
            ("points3D.ply", "points3d_ply_sha256"),
        ):
            path = model_root / filename
            if not path.is_file():
                errors.append(f"missing {role} model file: {filename}")
            elif sha256_file(path) != row.get(field):
                errors.append(f"{role} {filename} SHA mismatch")
        if (model_root / "points3D.bin").exists():
            errors.append(f"{role} points3D.bin must be absent")
        try:
            cameras = read_cameras_binary(model_root / "cameras.bin")
            images = read_images_binary(model_root / "images.bin")
            if len(cameras) != int(row.get("camera_count", -1)):
                errors.append(f"{role} camera count mismatch")
            if len(images) != int(row.get("image_count", -1)):
                errors.append(f"{role} model image count mismatch")
            if {image.name for image in images.values()} != role_names[role]:
                errors.append(f"{role} model/image-directory name mismatch")
            if any(len(image.xys) or len(image.point3D_ids) for image in images.values()):
                errors.append(f"{role} POINTS2D tracks are present")
        except Exception as exc:
            errors.append(f"{role} model decode failed: {exc}")

    return {
        "schema": "gs_gcp_native_quarter_materialization_verification_v1",
        "scene": manifest.get("scene"),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "full_view_count": manifest.get("full_view_count"),
        "train_view_count": manifest.get("train_view_count"),
        "test_view_count": manifest.get("test_view_count"),
        "decoded_images_checked": decode_images,
        "passed": not errors,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify_only", type=Path)
    parser.add_argument("--skip_decode", action="store_true")
    parser.add_argument("--split_manifest", type=Path)
    parser.add_argument("--scene")
    parser.add_argument("--source_root", type=Path)
    parser.add_argument("--output_root", type=Path)
    parser.add_argument("--package_audit", type=Path)
    parser.add_argument("--image_sha256_manifest", type=Path)
    parser.add_argument("--file_mode", choices=("copy", "hardlink"), default="copy")
    args = parser.parse_args()
    if args.verify_only:
        result = verify_materialization(args.verify_only, decode_images=not args.skip_decode)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result["passed"] else 1
    for name in ("split_manifest", "scene", "source_root", "output_root"):
        if getattr(args, name) is None:
            parser.error(f"--{name} is required unless --verify_only is used")
    result = materialize_scene(
        split_manifest_path=args.split_manifest,
        scene=args.scene,
        source_root=args.source_root,
        output_root=args.output_root,
        package_audit_path=args.package_audit,
        image_sha256_path=args.image_sha256_manifest,
        file_mode=args.file_mode,
    )
    print(json.dumps({
        "status": "PASS",
        "scene": result["scene"],
        "manifest_sha256": result["manifest_sha256"],
        "train_view_count": result["train_view_count"],
        "test_view_count": result["test_view_count"],
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
