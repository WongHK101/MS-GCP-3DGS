#!/usr/bin/env python3
"""Create the pose-only 100K train-camera root used by every packet exporter."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from pathlib import Path
from typing import Any


SCENE = "gcp_100000_20260610"
SCHEMA = "m3m_gcp_100k_evaluation_camera_root_v1"
STATUS = "PASS_EVALUATION_CAMERA_ROOT_NO_TRAINING_NO_PRIOR_NO_EVALUATION"
EXPECTED_FULL_VIEWS = 2510
EXPECTED_TRAIN_VIEWS = 2196
EXPECTED_TEST_VIEWS = 314
FORMAL_MANIFEST_SHA = "c2cf9e951d95fee12a28d942e95c5c420df55bc364738b3f8737fed1c78bef3d"
FORMAL_MANIFEST_CANONICAL_SHA = "5b4fe34743310bd2225feb2dd236200606be933002fec19d2c9ecb9f3ba6769d"
FORMAL_SPARSE_SHA = {
    "cameras.bin": "6669584ba1ba326cf5b372b878a5abf182f8cfe0bfe0845da3a0c4f7aed8fe5e",
    "images.bin": "dfc1a5d17532aebb3da670598635baea5c8fbf999592b6b567504251a01c9f72",
    "points3D.ply": "9f653655a34c05007e58f339afec593136bd857a56b13a612c79d8e53913364e",
}
FULL_ALL_IMAGE_SFM_SHA = {
    "cameras.bin": "6669584ba1ba326cf5b372b878a5abf182f8cfe0bfe0845da3a0c4f7aed8fe5e",
    "images.bin": "57163927bceee6ca330c113c9caf06cafe1a84a7ca21ac0f055680dcbe8eff6e",
    "points3D.bin": "09fc811f32558a11a47bada7393bf7bce2585cbe68eb4872ffce72025b0fc9aa",
    "points3D.ply": "9f653655a34c05007e58f339afec593136bd857a56b13a612c79d8e53913364e",
}
EMPTY_POINTS3D_SHA = "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: dict[str, Any]) -> str:
    body = dict(payload)
    body.pop("canonical_sha256", None)
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def formal_manifest_canonical_sha256(payload: dict[str, Any]) -> str:
    body = dict(payload)
    body.pop("manifest_sha256", None)
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def require_sha(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or sha256_file(path) != expected:
        raise RuntimeError(f"{label} identity mismatch: {path}")


def materialize(
    *, formal_scene_root: Path, formal_manifest_path: Path, output_root: Path,
    evidence_path: Path,
) -> dict[str, Any]:
    formal_scene_root = formal_scene_root.resolve()
    formal_manifest_path = formal_manifest_path.resolve()
    output_root = output_root.resolve()
    evidence_path = evidence_path.resolve()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(output_root)
    if evidence_path.exists() or evidence_path.is_symlink():
        raise FileExistsError(evidence_path)
    require_sha(formal_manifest_path, FORMAL_MANIFEST_SHA, "formal manifest")
    formal = json.loads(formal_manifest_path.read_text(encoding="utf-8"))
    if (
        formal.get("scene") != SCENE
        or formal.get("manifest_sha256") != FORMAL_MANIFEST_CANONICAL_SHA
        or formal_manifest_canonical_sha256(formal) != FORMAL_MANIFEST_CANONICAL_SHA
        or formal.get("full_view_count") != EXPECTED_FULL_VIEWS
        or formal.get("train_view_count") != EXPECTED_TRAIN_VIEWS
        or formal.get("test_view_count") != EXPECTED_TEST_VIEWS
        or formal.get("source_model_sha256") != FULL_ALL_IMAGE_SFM_SHA
    ):
        raise RuntimeError("formal manifest protocol identity mismatch")
    roles = {row.get("role"): row for row in formal.get("roles", [])}
    train_role = roles.get("train", {})
    if (
        train_role.get("image_count") != EXPECTED_TRAIN_VIEWS
        or train_role.get("points2d_tracks_present") is not False
        or train_role.get("points3d_bin_present") is not False
        or {
            "cameras.bin": train_role.get("cameras_bin_sha256"),
            "images.bin": train_role.get("images_bin_sha256"),
            "points3D.ply": train_role.get("points3d_ply_sha256"),
        }
        != FORMAL_SPARSE_SHA
    ):
        raise RuntimeError("formal train role identity mismatch")
    source_root = formal_scene_root / "train"
    source_sparse = source_root / "sparse" / "0"
    source_images = source_root / "images"
    if not source_images.is_dir():
        raise FileNotFoundError(source_images)
    for name, expected in FORMAL_SPARSE_SHA.items():
        require_sha(source_sparse / name, expected, f"formal train {name}")
    if (source_sparse / "points3D.bin").exists():
        raise RuntimeError("formal train root unexpectedly exposes points3D.bin")
    train_rows = [row for row in formal.get("images", []) if row.get("role") == "train"]
    expected_names = {str(row.get("image_name", "")) for row in train_rows}
    actual_names = {path.name for path in source_images.iterdir() if path.is_file()}
    if (
        len(train_rows) != EXPECTED_TRAIN_VIEWS
        or len(expected_names) != EXPECTED_TRAIN_VIEWS
        or actual_names != expected_names
    ):
        raise RuntimeError("formal train image inventory mismatch")
    for row in train_rows:
        path = source_images / str(row["image_name"])
        if path.stat().st_size != int(row["jpeg_bytes"]) or sha256_file(path) != row["jpeg_sha256"]:
            raise RuntimeError(f"formal train RGB identity mismatch: {path.name}")

    output_sparse = output_root / "sparse" / "0"
    output_sparse.mkdir(parents=True)
    marker = output_root / "MATERIALIZATION_INCOMPLETE"
    marker.write_text("Remove only after the camera-root manifest passes.\n", encoding="utf-8")
    os.symlink(source_images, output_root / "images", target_is_directory=True)
    for name in FORMAL_SPARSE_SHA:
        os.link(source_sparse / name, output_sparse / name)
    (output_sparse / "points3D.bin").write_bytes(struct.pack("<Q", 0))
    require_sha(output_sparse / "points3D.bin", EMPTY_POINTS3D_SHA, "empty points3D")
    same_file = {
        name: (
            (source_sparse / name).stat().st_dev == (output_sparse / name).stat().st_dev
            and (source_sparse / name).stat().st_ino == (output_sparse / name).stat().st_ino
        )
        for name in FORMAL_SPARSE_SHA
    }
    if not all(same_file.values()) or not (output_root / "images").is_symlink():
        raise RuntimeError("evaluation camera root did not use required links")
    output_files = {
        name: identity(output_sparse / name)
        for name in ("cameras.bin", "images.bin", "points3D.bin", "points3D.ply")
    }
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": STATUS,
        "scene": SCENE,
        "purpose": "evaluation-only all-train-camera loader root; never a training or geometry-prior input",
        "formal_scene_root": str(formal_scene_root),
        "formal_manifest": {
            "path": str(formal_manifest_path),
            "sha256": FORMAL_MANIFEST_SHA,
            "canonical_sha256": FORMAL_MANIFEST_CANONICAL_SHA,
        },
        "shared_all_image_sfm": {
            "image_count": EXPECTED_FULL_VIEWS,
            "sha256": FULL_ALL_IMAGE_SFM_SHA,
            "all_images_participated_before_train_test_split": True,
        },
        "source_train": {
            "root": str(source_root),
            "view_count": EXPECTED_TRAIN_VIEWS,
            "sparse_sha256": FORMAL_SPARSE_SHA,
            "points2d_tracks_present": False,
            "points3d_bin_present": False,
        },
        "output": {
            "root": str(output_root),
            "view_count": EXPECTED_TRAIN_VIEWS,
            "files": output_files,
            "images_symlink_target": str((output_root / "images").resolve()),
            "source_sparse_files_are_hardlinks": same_file,
            "points3d_bin_point_count": 0,
            "points3d_bin_purpose": "deterministic empty COLMAP compatibility triplet member; no geometry or tracks",
        },
        "truth_boundary": {
            "heldout_rgb_present": False,
            "gcp_or_lidar_used": False,
            "training_or_prior_started": False,
            "formal_evaluation_started": False,
        },
        "materializer": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    payload["canonical_sha256"] = canonical_sha256(payload)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    manifest_path = output_root / "EVALUATION_CAMERA_ROOT_MANIFEST.json"
    manifest_path.write_text(encoded, encoding="utf-8")
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(encoded, encoding="utf-8")
    marker.unlink()
    if sha256_file(manifest_path) != sha256_file(evidence_path):
        raise RuntimeError("camera-root manifest/evidence bytes differ")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-scene-root", type=Path, required=True)
    parser.add_argument("--formal-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    payload = materialize(
        formal_scene_root=args.formal_scene_root,
        formal_manifest_path=args.formal_manifest,
        output_root=args.output_root,
        evidence_path=args.evidence,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
