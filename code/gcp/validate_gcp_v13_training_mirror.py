#!/usr/bin/env python3
"""Validate a frozen v1.3 training mirror without trusting its stored PASS state."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any


SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_mirror(root: Path, recipe: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    manifest_path = root / "SOURCE_MANIFEST.json"
    detached_path = root / "SOURCE_MANIFEST.sha256"
    if not manifest_path.is_file() or not detached_path.is_file():
        return ["SOURCE_MANIFEST.json or detached SHA is missing"], {}
    manifest_sha = sha256_file(manifest_path)
    detached_parts = detached_path.read_text(encoding="ascii").strip().split()
    if len(detached_parts) != 2 or detached_parts[0] != manifest_sha or detached_parts[1] != "SOURCE_MANIFEST.json":
        errors.append("SOURCE_MANIFEST detached SHA mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scene = str(manifest.get("scene", ""))
    scene_spec = recipe.get("scenes", {}).get(scene)
    if not isinstance(scene_spec, dict):
        errors.append("mirror scene is not present in frozen recipe")
        scene_spec = {}
    if manifest.get("schema") != "ms_gcp_v1_3_read_only_training_source_manifest_v1":
        errors.append("unknown training source manifest schema")
    if manifest.get("release_root_digest") != recipe.get("release", {}).get("payload_root_digest_sha256"):
        errors.append("release root digest mismatch")
    if manifest.get("release_id") != recipe.get("release", {}).get("release_id"):
        errors.append("release id mismatch")
    if manifest.get("hardlinks_used") is not False or manifest.get("independent_inode_verified") is not True:
        errors.append("mirror independence/hardlink policy mismatch")
    if manifest.get("image_count") != scene_spec.get("image_count"):
        errors.append("image count mismatch")
    if manifest.get("image_bytes") != scene_spec.get("image_bytes"):
        errors.append("image byte count mismatch")

    declared: dict[str, dict[str, Any]] = {}
    for record in manifest.get("files", []):
        rel = str(record.get("relative_path", ""))
        if not rel or rel in declared or Path(rel).is_absolute() or ".." in Path(rel).parts:
            errors.append(f"invalid/duplicate manifest path: {rel}")
            continue
        declared[rel] = record
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name not in {"SOURCE_MANIFEST.json", "SOURCE_MANIFEST.sha256"}
    }
    if set(declared) != actual:
        errors.append(f"mirror file set mismatch: missing={sorted(set(declared)-actual)[:5]} extra={sorted(actual-set(declared))[:5]}")
    for rel, record in declared.items():
        path = root / rel
        expected_sha = record.get("sha256")
        if not path.is_file() or path.is_symlink():
            errors.append(f"missing or symlinked mirror file: {rel}")
            continue
        if not isinstance(expected_sha, str) or not SHA_RE.fullmatch(expected_sha):
            errors.append(f"invalid declared SHA: {rel}")
        elif sha256_file(path) != expected_sha:
            errors.append(f"file SHA mismatch: {rel}")
        if path.stat().st_size != record.get("bytes"):
            errors.append(f"file size mismatch: {rel}")
        if path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            errors.append(f"mirror file is writable: {rel}")
    for path in [root, *root.rglob("*")]:
        if path.is_symlink():
            errors.append(f"mirror symlink is forbidden: {path.relative_to(root)}")
        if path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            errors.append(f"mirror path is writable: {path.relative_to(root) if path != root else '.'}")

    expected_sparse = {
        "sparse/0/cameras.bin": scene_spec.get("cameras_bin_sha256"),
        "sparse/0/images.bin": scene_spec.get("images_bin_sha256"),
        "sparse/0/points3D.bin": scene_spec.get("points3d_bin_sha256"),
    }
    for rel, expected_sha in expected_sparse.items():
        if declared.get(rel, {}).get("sha256") != expected_sha:
            errors.append(f"frozen sparse hash mismatch: {rel}")
    details = {
        "scene": scene,
        "source_manifest_sha256": manifest_sha,
        "declared_file_count": len(declared),
        "actual_file_count": len(actual),
        "image_count": manifest.get("image_count"),
        "initial_point_count": manifest.get("initial_point_count"),
        "copy_strategies": manifest.get("copy_strategies"),
        "read_only": not any("writable" in item for item in errors),
    }
    return errors, details


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--recipe",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "configs" / "gcp_v13_original_3dgs_six_scene_recipe_v1.json",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    errors, details = validate_mirror(args.root.resolve(), recipe)
    report = {
        "schema": "ms_gcp_v1_3_training_mirror_validation_v1",
        "status": "pass" if not errors else "fail",
        "error_count": len(errors),
        "errors": errors,
        **details,
    }
    text = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
