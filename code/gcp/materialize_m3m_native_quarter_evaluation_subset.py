#!/usr/bin/env python3
"""Materialize the formal cameras needed to evaluate one protocol scene.

The evaluation loader needs one COLMAP root containing every camera that has
an observation in the frozen protocol.  The formal benchmark input keeps those
cameras in disjoint ``train`` and ``test`` roots.  This tool merges only the
required camera records into an evaluation-only, pose-only model while keeping
the native-quarter JPEG files and initial PLY byte-identical to the formal
input.  It never reads a second COLMAP reconstruction or changes training data.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

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
    _materialize_file,
    canonical_sha256,
    read_pose_only_images_binary,
    sha256_file,
    write_json,
)
from read_write_model import (  # noqa: E402
    Image as ColmapImage,
    read_cameras_binary,
    read_images_binary,
    read_points3D_binary,
    write_cameras_binary,
    write_images_binary,
    write_points3D_binary,
)


SCHEMA = "m3m_gcp_native_quarter_evaluation_camera_subset_v1"
VERIFICATION_SCHEMA = "m3m_gcp_native_quarter_evaluation_subset_verification_v1"
PROTOCOL_ID = "m3m_gcp_native_quarter_geometry_v2"
FORMAL_ROLES = ("train", "test")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _observation_names(path: Path, scene: str) -> tuple[list[str], int, int]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    _require(bool(rows), "protocol observation table is empty")
    _require(all(row.get("scene") == scene for row in rows), "protocol observation scene mismatch")
    names = sorted({str(row.get("image_name", "")).strip() for row in rows if row.get("image_name", "").strip()})
    points = {str(row.get("point_name", "")).strip() for row in rows if row.get("point_name", "").strip()}
    _require(bool(names), "protocol observation table has no image names")
    _require(bool(points), "protocol observation table has no point names")
    return names, len(rows), len(points)


def _formal_role_rows(formal: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = {str(row.get("role")): row for row in formal.get("roles", [])}
    _require(set(rows) == set(FORMAL_ROLES), "formal input role inventory mismatch")
    return rows


def _validate_formal_role_model(
    formal_root: Path,
    role: str,
    role_row: dict[str, Any],
    expected_names: set[str],
) -> tuple[dict[int, Any], dict[str, ColmapImage]]:
    model_root = formal_root / role / "sparse" / "0"
    for filename, field in (
        ("cameras.bin", "cameras_bin_sha256"),
        ("images.bin", "images_bin_sha256"),
        ("points3D.ply", "points3d_ply_sha256"),
    ):
        path = model_root / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        _require(sha256_file(path) == role_row.get(field), f"formal {role} {filename} SHA mismatch")
    _require(not (model_root / "points3D.bin").exists(), f"formal {role} points3D.bin must be absent")
    cameras = read_cameras_binary(model_root / "cameras.bin")
    images = read_pose_only_images_binary(model_root / "images.bin", expected_names)
    return cameras, images


def materialize_subset(
    *,
    scene: str,
    formal_input_manifest_path: Path,
    protocol_observations_path: Path,
    output_root: Path,
    file_mode: str = "copy",
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(output_root)
    formal_root = formal_input_manifest_path.parent
    formal = json.loads(formal_input_manifest_path.read_text(encoding="utf-8"))
    _require(formal.get("schema") == FORMAL_INPUT_SCHEMA, "formal input schema mismatch")
    _require(canonical_sha256(formal) == formal.get("manifest_sha256"), "formal input canonical SHA mismatch")
    _require(formal.get("scene") == scene, "formal input scene mismatch")
    _require(formal.get("release_root_digest_sha256") == RELEASE_DIGEST, "formal input release mismatch")
    _require(formal.get("pixel_domain") == PIXEL_DOMAIN, "formal input pixel domain mismatch")
    role_rows = _formal_role_rows(formal)
    formal_images = {str(row["image_name"]): row for row in formal.get("images", [])}
    _require(len(formal_images) == int(formal.get("full_view_count", -1)), "formal input image inventory mismatch")

    names, observation_count, point_count = _observation_names(protocol_observations_path, scene)
    _require(set(names).issubset(formal_images), "protocol observation camera is absent from formal input")
    names_by_role = {
        role: {name for name in names if formal_images[name].get("role") == role}
        for role in FORMAL_ROLES
    }
    _require(set().union(*names_by_role.values()) == set(names), "protocol camera has an unknown formal role")

    formal_cameras: dict[str, dict[int, Any]] = {}
    formal_poses: dict[str, dict[str, ColmapImage]] = {}
    initial_ply_path: Path | None = None
    expected_ply_sha = str(formal.get("source_model_sha256", {}).get("points3D.ply", ""))
    _require(bool(expected_ply_sha), "formal input initial PLY SHA is missing")
    for role in FORMAL_ROLES:
        formal_cameras[role], formal_poses[role] = _validate_formal_role_model(
            formal_root,
            role,
            role_rows[role],
            names_by_role[role],
        )
        role_ply = formal_root / role / "sparse" / "0" / "points3D.ply"
        _require(sha256_file(role_ply) == expected_ply_sha, f"formal {role} initial PLY identity mismatch")
        if initial_ply_path is None:
            initial_ply_path = role_ply
    assert initial_ply_path is not None

    output_root.mkdir(parents=True)
    marker = output_root / "MATERIALIZATION_INCOMPLETE"
    marker.write_text("Remove only after the evaluation subset manifest passes verification.\n", encoding="utf-8")
    model_root = output_root / "sparse" / "0"
    model_root.mkdir(parents=True)
    selected_cameras: dict[int, Any] = {}
    selected_images: dict[int, ColmapImage] = {}
    image_rows: list[dict[str, Any]] = []
    for name in names:
        record = formal_images[name]
        role = str(record["role"])
        pose = formal_poses[role][name]
        _require(int(pose.id) == int(record["image_id"]), f"image ID mismatch: {name}")
        _require(int(pose.camera_id) == int(record["camera_id"]), f"camera ID mismatch: {name}")
        camera = formal_cameras[role].get(int(pose.camera_id))
        if camera is None:
            raise ValueError(f"missing formal {role} camera {pose.camera_id}: {name}")
        previous_camera = selected_cameras.get(int(camera.id))
        if previous_camera is not None:
            same_camera = (
                int(previous_camera.id) == int(camera.id)
                and previous_camera.model == camera.model
                and int(previous_camera.width) == int(camera.width)
                and int(previous_camera.height) == int(camera.height)
                and np.array_equal(previous_camera.params, camera.params)
            )
            _require(same_camera, f"camera record differs across formal roles: {camera.id}")
        source_image = formal_root / str(record["relative_path"])
        if not source_image.is_file():
            raise FileNotFoundError(source_image)
        actual_sha = sha256_file(source_image)
        _require(actual_sha == record.get("jpeg_sha256"), f"image SHA mismatch: {name}")
        _require(source_image.stat().st_size == int(record.get("jpeg_bytes", -1)), f"image byte count mismatch: {name}")
        with PILImage.open(source_image) as image:
            _require(image.mode == "RGB", f"image mode mismatch: {name}")
            _require(image.size == (int(record["width"]), int(record["height"])), f"image dimensions mismatch: {name}")
        _require((int(camera.width), int(camera.height)) == (int(record["width"]), int(record["height"])), f"camera dimensions mismatch: {name}")
        _materialize_file(source_image, output_root / "images" / name, file_mode)
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
        image_rows.append({
            "image_name": name,
            "image_id": int(pose.id),
            "camera_id": int(pose.camera_id),
            "formal_role": role,
            "formal_relative_path": str(record["relative_path"]),
            "width": int(record["width"]),
            "height": int(record["height"]),
            "jpeg_bytes": int(record["jpeg_bytes"]),
            "jpeg_sha256": actual_sha,
        })
    write_cameras_binary(dict(sorted(selected_cameras.items())), model_root / "cameras.bin")
    write_images_binary(dict(sorted(selected_images.items())), model_root / "images.bin")
    # COLMAP's generic read_model() requires a complete cameras/images/points3D
    # triplet.  The evaluator uses only cameras and images, so provide an
    # explicit zero-point file rather than exposing any reconstruction points.
    write_points3D_binary({}, model_root / "points3D.bin")
    _materialize_file(initial_ply_path, model_root / "points3D.ply", file_mode)

    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "scene": scene,
        "status": "PASS",
        "release_root_digest_sha256": RELEASE_DIGEST,
        "pixel_domain": PIXEL_DOMAIN,
        "purpose": "evaluation-only camera loader subset; no training or checkpoint selection",
        "formal_input_manifest_file_sha256": sha256_file(formal_input_manifest_path),
        "formal_input_manifest_canonical_sha256": formal["manifest_sha256"],
        "protocol_observations_file_sha256": sha256_file(protocol_observations_path),
        "source_role_model_sha256": {
            role: {
                field: role_rows[role][field]
                for field in ("cameras_bin_sha256", "images_bin_sha256", "points3d_ply_sha256")
            }
            for role in FORMAL_ROLES
        },
        "source_initial_ply_sha256": expected_ply_sha,
        "file_materialization": {
            "mode": file_mode,
            "semantic_identity": "file bytes are normative; hardlinks and copies are equivalent",
        },
        "image_transform": {
            "operation": "byte-preserving selection from the frozen formal native-quarter input",
            "resize": False,
            "crop": False,
            "pad": False,
            "reencode": False,
        },
        "camera_transform": {
            "intrinsics": "formal camera records merged byte-semantically",
            "extrinsics": "formal float64 qvec/tvec preserved",
            "points2d_tracks": "absent",
        },
        "observation_count": observation_count,
        "point_count": point_count,
        "camera_view_count": len(names),
        "camera_count": len(selected_cameras),
        "cameras_bin_sha256": sha256_file(model_root / "cameras.bin"),
        "images_bin_sha256": sha256_file(model_root / "images.bin"),
        "points3d_bin_sha256": sha256_file(model_root / "points3D.bin"),
        "points3d_ply_sha256": sha256_file(model_root / "points3D.ply"),
        "points2d_tracks_present": False,
        "points3d_bin_present": True,
        "points3d_bin_point_count": 0,
        "points3d_bin_purpose": "deterministic empty COLMAP compatibility file for read_model(); not geometry input",
        "images": image_rows,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    write_json(output_root / "EVALUATION_CAMERA_SUBSET_MANIFEST.json", manifest)
    marker.unlink()
    verification = verify_subset(output_root, decode_images=False)
    if not verification["passed"]:
        marker.write_text("Post-materialization verification failed; this subset is invalid.\n", encoding="utf-8")
        raise ValueError(f"evaluation subset verification failed: {verification['errors'][:3]}")
    return manifest


def verify_subset(root: Path, *, decode_images: bool = True) -> dict[str, Any]:
    errors: list[str] = []
    manifest_path = root / "EVALUATION_CAMERA_SUBSET_MANIFEST.json"
    if not manifest_path.is_file():
        return {"schema": VERIFICATION_SCHEMA, "passed": False, "errors": ["manifest missing"]}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        errors.append("manifest schema mismatch")
    if manifest.get("protocol_id") != PROTOCOL_ID:
        errors.append("protocol ID mismatch")
    if manifest.get("release_root_digest_sha256") != RELEASE_DIGEST:
        errors.append("release digest mismatch")
    if manifest.get("pixel_domain") != PIXEL_DOMAIN:
        errors.append("pixel domain mismatch")
    if manifest.get("status") != "PASS":
        errors.append("manifest status is not PASS")
    if canonical_sha256(manifest) != manifest.get("manifest_sha256"):
        errors.append("manifest canonical SHA mismatch")
    if (root / "MATERIALIZATION_INCOMPLETE").exists():
        errors.append("incomplete marker is present")

    image_entries = manifest.get("images", [])
    expected_names = {str(row.get("image_name")) for row in image_entries}
    if len(expected_names) != len(image_entries):
        errors.append("manifest image names are not unique")
    images_root = root / "images"
    actual_names = {path.name for path in images_root.iterdir() if path.is_file()} if images_root.is_dir() else set()
    if expected_names != actual_names:
        errors.append("image inventory mismatch")
    rows = {str(row.get("image_name")): row for row in image_entries}
    for name in sorted(expected_names & actual_names):
        path = images_root / name
        row = rows[name]
        if path.stat().st_size != int(row.get("jpeg_bytes", -1)) or sha256_file(path) != row.get("jpeg_sha256"):
            errors.append(f"image identity mismatch: {name}")
            continue
        if decode_images:
            try:
                with PILImage.open(path) as image:
                    if image.mode != "RGB" or image.size != (int(row["width"]), int(row["height"])):
                        errors.append(f"decoded image mismatch: {name}")
            except Exception as exc:  # pragma: no cover
                errors.append(f"image decode failed: {name}: {exc}")

    model_root = root / "sparse" / "0"
    for filename, field in (
        ("cameras.bin", "cameras_bin_sha256"),
        ("images.bin", "images_bin_sha256"),
        ("points3D.bin", "points3d_bin_sha256"),
        ("points3D.ply", "points3d_ply_sha256"),
    ):
        path = model_root / filename
        if not path.is_file() or sha256_file(path) != manifest.get(field):
            errors.append(f"model identity mismatch: {filename}")
    try:
        cameras = read_cameras_binary(model_root / "cameras.bin")
        images = read_images_binary(model_root / "images.bin")
        points3d = read_points3D_binary(model_root / "points3D.bin")
        if len(cameras) != int(manifest.get("camera_count", -1)):
            errors.append("COLMAP camera count mismatch")
        if len(images) != int(manifest.get("camera_view_count", -1)):
            errors.append("COLMAP image count mismatch")
        if {image.name for image in images.values()} != expected_names:
            errors.append("COLMAP image names mismatch")
        if any(len(image.xys) or len(image.point3D_ids) for image in images.values()):
            errors.append("COLMAP POINTS2D tracks are present")
        if points3d or int(manifest.get("points3d_bin_point_count", -1)) != 0:
            errors.append("COLMAP compatibility points3D.bin is not empty")
        for image in images.values():
            row = rows.get(image.name, {})
            if int(image.id) != int(row.get("image_id", -1)) or int(image.camera_id) != int(row.get("camera_id", -1)):
                errors.append(f"COLMAP image identity mismatch: {image.name}")
            if image.camera_id not in cameras:
                errors.append(f"COLMAP camera is missing: {image.name}")
    except Exception as exc:
        errors.append(f"COLMAP model decode failed: {exc}")
    if len(expected_names) != int(manifest.get("camera_view_count", -1)):
        errors.append("camera view count mismatch")
    return {
        "schema": VERIFICATION_SCHEMA,
        "scene": manifest.get("scene"),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "camera_view_count": manifest.get("camera_view_count"),
        "observation_count": manifest.get("observation_count"),
        "decoded_images_checked": decode_images,
        "passed": not errors,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify_only", type=Path)
    parser.add_argument("--skip_decode", action="store_true")
    parser.add_argument("--scene")
    parser.add_argument("--formal_input_manifest", type=Path)
    parser.add_argument("--protocol_observations", type=Path)
    parser.add_argument("--output_root", type=Path)
    parser.add_argument("--file_mode", choices=("copy", "hardlink"), default="copy")
    args = parser.parse_args()
    if args.verify_only:
        result = verify_subset(args.verify_only, decode_images=not args.skip_decode)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result["passed"] else 1
    for name in ("scene", "formal_input_manifest", "protocol_observations", "output_root"):
        if getattr(args, name) is None:
            parser.error(f"--{name} is required unless --verify_only is used")
    result = materialize_subset(
        scene=args.scene,
        formal_input_manifest_path=args.formal_input_manifest,
        protocol_observations_path=args.protocol_observations,
        output_root=args.output_root,
        file_mode=args.file_mode,
    )
    print(json.dumps({
        "status": "PASS",
        "scene": result["scene"],
        "camera_view_count": result["camera_view_count"],
        "observation_count": result["observation_count"],
        "manifest_sha256": result["manifest_sha256"],
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
