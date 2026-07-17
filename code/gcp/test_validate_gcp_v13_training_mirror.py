#!/usr/bin/env python3
"""Focused tests for immutable v1.3 training mirrors."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path

from validate_gcp_v13_training_mirror import sha256_file, validate_mirror


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _chmod_tree(root: Path, *, writable: bool) -> None:
    file_mode = stat.S_IRUSR | (stat.S_IWUSR if writable else 0)
    dir_mode = stat.S_IRUSR | stat.S_IXUSR | (stat.S_IWUSR if writable else 0)
    for path in root.rglob("*"):
        path.chmod(dir_mode if path.is_dir() else file_mode)
    root.chmod(dir_mode)


def _fixture(root: Path) -> dict:
    payloads = {
        "images/a.jpg": b"image-a",
        "sparse/0/cameras.bin": b"camera",
        "sparse/0/images.bin": b"poses",
        "sparse/0/points3D.bin": b"points",
        "sparse/0/points3D.ply": b"ply",
    }
    for rel, data in payloads.items():
        _write(root / rel, data)
    records = [
        {
            "relative_path": rel,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "copy_strategy": "full_copy",
        }
        for rel, data in sorted(payloads.items())
    ]
    recipe = {
        "release": {"release_id": "release", "payload_root_digest_sha256": "d" * 64},
        "scenes": {
            "scene": {
                "image_count": 1,
                "image_bytes": len(payloads["images/a.jpg"]),
                "cameras_bin_sha256": records[1]["sha256"],
                "images_bin_sha256": records[2]["sha256"],
                "points3d_bin_sha256": records[3]["sha256"],
                "points3d_ply_bytes": records[4]["bytes"],
                "points3d_ply_sha256": records[4]["sha256"],
            }
        },
    }
    manifest = {
        "schema": "gs_gcp_v1_3_read_only_training_source_manifest_v1",
        "scene": "scene",
        "release_id": "release",
        "release_root_digest": "d" * 64,
        "image_count": 1,
        "image_bytes": len(payloads["images/a.jpg"]),
        "initial_point_count": 1,
        "copy_strategies": ["full_copy"],
        "independent_inode_verified": True,
        "hardlinks_used": False,
        "loader_artifact_policy": "preexisting_points3D_ply_copied_and_hash_frozen_before_training",
        "files": records,
    }
    manifest_path = root / "SOURCE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (root / "SOURCE_MANIFEST.sha256").write_text(
        f"{sha256_file(manifest_path)}  SOURCE_MANIFEST.json\n", encoding="ascii"
    )
    _chmod_tree(root, writable=False)
    return recipe


def test_valid_read_only_mirror_passes() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp) / "mirror"
        root.mkdir()
        recipe = _fixture(root)
        try:
            errors, details = validate_mirror(root, recipe)
            assert errors == []
            assert details["read_only"] is True
        finally:
            _chmod_tree(root, writable=True)


def test_rejects_writable_file() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp) / "mirror"
        root.mkdir()
        recipe = _fixture(root)
        try:
            (root / "images" / "a.jpg").chmod(stat.S_IRUSR | stat.S_IWUSR)
            errors, _ = validate_mirror(root, recipe)
            assert any("writable" in error for error in errors)
        finally:
            _chmod_tree(root, writable=True)


def test_rejects_unregistered_file() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp) / "mirror"
        root.mkdir()
        recipe = _fixture(root)
        try:
            root.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
            _write(root / "extra.bin", b"extra")
            (root / "extra.bin").chmod(stat.S_IRUSR)
            root.chmod(stat.S_IRUSR | stat.S_IXUSR)
            errors, _ = validate_mirror(root, recipe)
            assert any("file set mismatch" in error for error in errors)
        finally:
            _chmod_tree(root, writable=True)


def test_rejects_missing_precomputed_loader_ply() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp) / "mirror"
        root.mkdir()
        recipe = _fixture(root)
        try:
            _chmod_tree(root, writable=True)
            (root / "sparse" / "0" / "points3D.ply").unlink()
            _chmod_tree(root, writable=False)
            errors, _ = validate_mirror(root, recipe)
            assert any("file set mismatch" in error for error in errors)
            assert any("points3D.ply" in error for error in errors)
        finally:
            _chmod_tree(root, writable=True)


def main() -> int:
    tests = [
        test_valid_read_only_mirror_passes,
        test_rejects_writable_file,
        test_rejects_unregistered_file,
        test_rejects_missing_precomputed_loader_ply,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS {len(tests)}/{len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
