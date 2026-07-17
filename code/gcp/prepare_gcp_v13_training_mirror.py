#!/usr/bin/env python3
"""Create a transactional, independent, read-only v1.3 scene training mirror."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import struct
import subprocess
import time
from pathlib import Path
from typing import Any


RELEASE_DIGEST = "513f8999fe4b110f15bcbecad7932895781cee755ee9ccd7a14ff10298546d75"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_independent(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    strategy = "full_copy"
    if os.name != "nt":
        result = subprocess.run(
            ["cp", "--reflink=always", "--preserve=mode,timestamps", "--", str(source), str(target)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            strategy = "copy_on_write_reflink"
        else:
            shutil.copy2(source, target)
    else:
        shutil.copy2(source, target)
    source_stat = source.stat()
    target_stat = target.stat()
    if source_stat.st_dev == target_stat.st_dev and source_stat.st_ino == target_stat.st_ino:
        raise RuntimeError(f"hardlink/same inode is forbidden: {source}")
    return strategy


def _read_point_count(path: Path) -> int:
    with path.open("rb") as handle:
        raw = handle.read(8)
    if len(raw) != 8:
        raise ValueError(f"truncated points3D.bin: {path}")
    return int(struct.unpack("<Q", raw)[0])


def _make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_symlink():
            raise RuntimeError(f"symlinks are forbidden in frozen mirror: {path}")
        if path.is_file():
            path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        elif path.is_dir():
            path.chmod(stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
    root.chmod(stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)


def _enabled(value: Any) -> bool:
    return value is True or str(value).strip().lower() == "true"


def build_mirror(
    *,
    scene: str,
    source_root: Path,
    dataset_parent: Path,
    release_root: Path,
    recipe_path: Path,
) -> dict[str, Any]:
    final_root = dataset_parent / scene
    if final_root.exists():
        raise FileExistsError(f"refusing existing dataset mirror: {final_root}")
    staging = dataset_parent / f".{scene}.staging_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    if staging.exists():
        raise FileExistsError(staging)
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    scene_spec = recipe.get("scenes", {}).get(scene)
    if not isinstance(scene_spec, dict):
        raise ValueError(f"scene is not frozen in recipe: {scene}")
    release_manifest_path = release_root / recipe["release"]["training_view_manifest"]
    release_manifest = json.loads(release_manifest_path.read_text(encoding="utf-8"))
    if release_manifest.get("release_id") != recipe["release"]["release_id"]:
        raise ValueError("training-view manifest release mismatch")
    views = [
        row for row in release_manifest.get("views", [])
        if row.get("scene") == scene and _enabled(row.get("training_view_included"))
    ]
    expected_names = [str(row["image_name"]) for row in views]
    if len(expected_names) != int(scene_spec["image_count"]) or len(set(expected_names)) != len(expected_names):
        raise ValueError("training-view count/uniqueness mismatch")
    if sum(int(row["target_image_bytes"]) for row in views) != int(scene_spec["image_bytes"]):
        raise ValueError("training-view byte-count mismatch")
    source_images = source_root / "images"
    actual_names = sorted(path.name for path in source_images.iterdir() if path.is_file())
    if actual_names != sorted(expected_names):
        missing = sorted(set(expected_names) - set(actual_names))
        extra = sorted(set(actual_names) - set(expected_names))
        raise ValueError(f"source image-list mismatch: missing={missing[:5]} extra={extra[:5]}")

    staging.mkdir(parents=True)
    file_records: list[dict[str, Any]] = []
    copy_strategies: set[str] = set()
    try:
        for index, row in enumerate(sorted(views, key=lambda item: str(item["image_name"]))):
            name = str(row["image_name"])
            source = source_images / name
            if source.is_symlink() or not source.is_file():
                raise ValueError(f"source image is missing or symlinked: {source}")
            if source.stat().st_size != int(row["target_image_bytes"]):
                raise ValueError(f"source image size mismatch: {name}")
            expected_sha = str(row["target_image_sha256"])
            actual_sha = sha256_file(source)
            if actual_sha != expected_sha:
                raise ValueError(f"source image SHA mismatch: {name}")
            target = staging / "images" / name
            strategy = _copy_independent(source, target)
            copy_strategies.add(strategy)
            if sha256_file(target) != expected_sha:
                raise ValueError(f"copied image SHA mismatch: {name}")
            file_records.append({
                "relative_path": f"images/{name}",
                "bytes": target.stat().st_size,
                "sha256": expected_sha,
                "source_path": str(source),
                "copy_strategy": strategy,
            })
            if (index + 1) % 100 == 0:
                print(f"verified/copied {index + 1}/{len(views)} images", flush=True)

        sparse_source = source_root / "sparse" / "0"
        sparse_hashes = {
            "cameras.bin": scene_spec["cameras_bin_sha256"],
            "images.bin": scene_spec["images_bin_sha256"],
            "points3D.bin": scene_spec["points3d_bin_sha256"],
        }
        for name, expected_sha in sparse_hashes.items():
            source = sparse_source / name
            if source.is_symlink() or not source.is_file() or sha256_file(source) != expected_sha:
                raise ValueError(f"source sparse file mismatch: {source}")
            target = staging / "sparse" / "0" / name
            strategy = _copy_independent(source, target)
            copy_strategies.add(strategy)
            if sha256_file(target) != expected_sha:
                raise ValueError(f"copied sparse file SHA mismatch: {name}")
            file_records.append({
                "relative_path": f"sparse/0/{name}",
                "bytes": target.stat().st_size,
                "sha256": expected_sha,
                "source_path": str(source),
                "copy_strategy": strategy,
            })

        points_count = _read_point_count(staging / "sparse" / "0" / "points3D.bin")
        manifest = {
            "schema": "ms_gcp_v1_3_read_only_training_source_manifest_v1",
            "scene": scene,
            "release_id": recipe["release"]["release_id"],
            "release_root_digest": RELEASE_DIGEST,
            "training_view_manifest_path": str(release_manifest_path),
            "training_view_manifest_sha256": sha256_file(release_manifest_path),
            "source_root": str(source_root),
            "destination_root": str(final_root),
            "image_count": len(views),
            "image_bytes": sum(int(row["target_image_bytes"]) for row in views),
            "initial_point_count": points_count,
            "copy_strategies": sorted(copy_strategies),
            "independent_inode_verified": True,
            "hardlinks_used": False,
            "source_modified": False,
            "files": sorted(file_records, key=lambda row: row["relative_path"].encode("utf-8")),
        }
        manifest_path = staging / "SOURCE_MANIFEST.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest_sha = sha256_file(manifest_path)
        (staging / "SOURCE_MANIFEST.sha256").write_text(
            f"{manifest_sha}  SOURCE_MANIFEST.json\n", encoding="ascii"
        )
        _make_read_only(staging)
        staging.rename(final_root)
        return {
            "status": "pass",
            "scene": scene,
            "final_root": str(final_root),
            "source_manifest_sha256": manifest_sha,
            "image_count": len(views),
            "initial_point_count": points_count,
            "copy_strategies": sorted(copy_strategies),
        }
    except Exception:
        if staging.exists():
            for path in staging.rglob("*"):
                try:
                    path.chmod(stat.S_IRWXU)
                except OSError:
                    pass
            shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--source_root", type=Path, required=True)
    parser.add_argument("--dataset_parent", type=Path, required=True)
    parser.add_argument("--release_root", type=Path, required=True)
    parser.add_argument(
        "--recipe",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "configs" / "gcp_v13_original_3dgs_six_scene_recipe_v1.json",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = build_mirror(
        scene=args.scene,
        source_root=args.source_root.resolve(),
        dataset_parent=args.dataset_parent.resolve(),
        release_root=args.release_root.resolve(),
        recipe_path=args.recipe.resolve(),
    )
    text = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
