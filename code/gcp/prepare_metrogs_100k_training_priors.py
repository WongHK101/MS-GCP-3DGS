#!/usr/bin/env python3
"""Prepare MetroGS's frozen 100K training-only MoGe, multi-view, and Pi3 priors."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from plyfile import PlyData


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_record(args: list[Path | str]) -> list[str]:
    return [str(value) for value in args]


def run_checked(
    args: list[Path | str], *, cwd: Path, env: dict[str, str]
) -> list[str]:
    command = command_record(args)
    print("RUN", json.dumps(command, ensure_ascii=False), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)
    return command


def git_output(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.rstrip("\r\n")


def require_sha256(path: Path, expected: str, label: str) -> str:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"{label} SHA256 mismatch: expected {expected}, got {actual}")
    return actual


def image_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def validate_training_images(
    image_dir: Path, train_records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    expected_names = {record["image_name"] for record in train_records}
    files = image_files(image_dir)
    actual_names = {path.relative_to(image_dir).as_posix() for path in files}
    if actual_names != expected_names:
        raise RuntimeError(
            "MetroGS image root differs from the frozen 2196-view training set: "
            f"missing={sorted(expected_names - actual_names)}, "
            f"extra={sorted(actual_names - expected_names)}"
        )
    by_name = {record["image_name"]: record for record in train_records}
    inventory: list[dict[str, Any]] = []
    for path in files:
        name = path.relative_to(image_dir).as_posix()
        frozen = by_name[name]
        actual_hash = require_sha256(path, frozen["jpeg_sha256"], f"training image {name}")
        if path.stat().st_size != frozen["jpeg_bytes"]:
            raise RuntimeError(f"training image byte count mismatch: {name}")
        inventory.append(
            {
                "image_name": name,
                "bytes": path.stat().st_size,
                "sha256": actual_hash,
                "width": int(frozen["width"]),
                "height": int(frozen["height"]),
            }
        )
    return inventory


def validate_depths(
    depth_dir: Path, train_records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    expected = {record["image_name"] + ".npy" for record in train_records}
    files = sorted(depth_dir.glob("*.npy"))
    actual = {path.name for path in files}
    if actual != expected:
        raise RuntimeError(
            f"MoGe depth inventory mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    by_name = {record["image_name"]: record for record in train_records}
    rows: list[dict[str, Any]] = []
    for path in files:
        image_name = path.name[: -len(".npy")]
        source = by_name[image_name]
        value = np.load(path, mmap_mode="r")
        expected_shape = (int(source["height"]), int(source["width"]), 2)
        if value.shape != expected_shape:
            raise RuntimeError(f"MoGe depth shape mismatch for {path.name}: {value.shape}")
        if not np.issubdtype(value.dtype, np.floating):
            raise RuntimeError(f"MoGe depth must be floating point: {path.name} {value.dtype}")
        inv_depth = np.asarray(value[..., 0])
        mask = np.asarray(value[..., 1]) > 0.5
        if not np.isfinite(inv_depth).all() or not np.isfinite(value[..., 1]).all():
            raise RuntimeError(f"non-finite MoGe output: {path.name}")
        valid_count = int(mask.sum())
        if valid_count == 0:
            raise RuntimeError(f"MoGe mask rejects every pixel: {path.name}")
        valid_values = inv_depth[mask]
        minimum = float(valid_values.min())
        maximum = float(valid_values.max())
        if minimum < -1e-6 or maximum > 1.0 + 1e-6 or maximum - minimum <= 1e-6:
            raise RuntimeError(
                f"invalid normalized inverse-depth range for {path.name}: {minimum}, {maximum}"
            )
        rows.append(
            {
                "image_name": image_name,
                "relative_path": path.relative_to(depth_dir.parent).as_posix(),
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "valid_pixel_count": valid_count,
                "valid_pixel_fraction": valid_count / float(mask.size),
                "valid_inverse_depth_min": minimum,
                "valid_inverse_depth_max": maximum,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return rows


def validate_scales(
    path: Path, expected_names: set[str]
) -> tuple[list[dict[str, Any]], float, list[str], list[str]]:
    values = json.loads(path.read_text(encoding="utf-8"))
    if set(values) != expected_names:
        raise RuntimeError(
            f"depth-scale inventory mismatch: missing={sorted(expected_names - set(values))}, "
            f"extra={sorted(set(values) - expected_names)}"
        )
    rows: list[dict[str, Any]] = []
    scales: list[float] = []
    for name in sorted(values):
        scale = float(values[name]["scale"])
        offset = float(values[name]["offset"])
        if not math.isfinite(scale) or scale <= 0.0 or not math.isfinite(offset):
            raise RuntimeError(f"invalid MoGe/COLMAP scale for {name}: {values[name]}")
        scales.append(scale)
        rows.append({"image_name": name, "scale": scale, "offset": offset})
    median = float(np.median(np.asarray(scales, dtype=np.float64)))
    lower, upper = 0.2 * median, 5.0 * median
    accepted: list[str] = []
    rejected: list[str] = []
    for row in rows:
        keep_depth_prior = lower <= row["scale"] <= upper
        row["official_depth_prior_accepted"] = keep_depth_prior
        (accepted if keep_depth_prior else rejected).append(row["image_name"])
    if not accepted:
        raise RuntimeError("official MetroGS depth-scale filter would reject every depth prior")
    return rows, median, accepted, rejected


def validate_multi_view(path: Path, expected_names: set[str]) -> dict[str, Any]:
    values = json.loads(path.read_text(encoding="utf-8"))
    if set(values) != expected_names:
        raise RuntimeError(
            f"multi-view inventory mismatch: missing={sorted(expected_names - set(values))}, "
            f"extra={sorted(set(values) - expected_names)}"
        )
    counts: list[int] = []
    for name, neighbors in values.items():
        if not isinstance(neighbors, list):
            raise RuntimeError(f"multi-view neighbors must be a list: {name}")
        if name in neighbors:
            raise RuntimeError(f"multi-view self-neighbor: {name}")
        if not set(neighbors) <= expected_names:
            raise RuntimeError(f"unknown multi-view neighbor for {name}")
        if len(neighbors) == 0:
            raise RuntimeError(f"training view has no MetroGS multi-view neighbor: {name}")
        counts.append(len(neighbors))
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "camera_count": len(values),
        "neighbor_count_min": min(counts),
        "neighbor_count_max": max(counts),
        "neighbor_count_mean": float(np.mean(np.asarray(counts, dtype=np.float64))),
    }


def replace_segment_rgb_copies_with_hardlinks(
    segment_root: Path, source_image_root: Path, split_num: int
) -> dict[str, Any]:
    """Replace upstream byte copies with same-filesystem links to frozen RGB."""
    sources_by_hash: dict[str, list[Path]] = {}
    source_files = image_files(source_image_root)
    for source in source_files:
        sources_by_hash.setdefault(sha256(source), []).append(source)
    for paths in sources_by_hash.values():
        paths.sort(key=lambda path: path.as_posix())

    logical_bytes = 0
    linked = 0
    for block_idx in range(split_num):
        block_images = image_files(segment_root / f"block_{block_idx}" / "images")
        if not block_images:
            raise RuntimeError(f"empty MetroGS segment before hardlinking: block_{block_idx}")
        for image in block_images:
            digest = sha256(image)
            candidates = sources_by_hash.get(digest, [])
            if not candidates:
                raise RuntimeError(f"segment RGB has no frozen byte-identical source: {image}")
            source = candidates.pop(0)
            if source.stat().st_size != image.stat().st_size:
                raise RuntimeError(f"segment/source RGB byte count mismatch: {image}")
            temporary = image.with_name(image.name + ".m3m-hardlink-tmp")
            if temporary.exists() or temporary.is_symlink():
                raise FileExistsError(temporary)
            try:
                os.link(source, temporary)
                os.replace(temporary, image)
            finally:
                temporary.unlink(missing_ok=True)
            source_stat = source.stat()
            image_stat = image.stat()
            if (
                source_stat.st_dev != image_stat.st_dev
                or source_stat.st_ino != image_stat.st_ino
                or sha256(image) != digest
            ):
                raise RuntimeError(f"segment RGB hardlink proof failed: {image}")
            logical_bytes += image_stat.st_size
            linked += 1
    leftovers = sum(len(paths) for paths in sources_by_hash.values())
    if linked != len(source_files) or leftovers != 0:
        raise RuntimeError(
            f"segment hardlink cardinality differs from frozen train RGB: "
            f"linked={linked}, sources={len(source_files)}, leftovers={leftovers}"
        )
    return {
        "policy": "same-filesystem hardlinks to byte-frozen train RGB after the official partition is fixed",
        "segment_link_count": linked,
        "source_image_count": len(source_files),
        "logical_bytes": logical_bytes,
        "additional_physical_rgb_bytes": 0,
        "all_segment_rgb_share_source_device_and_inode": True,
    }


def validate_segments(
    segment_root: Path,
    source_image_hashes: Counter[str],
    source_image_inodes: Counter[tuple[int, int]],
    split_num: int,
) -> list[dict[str, Any]]:
    block_dirs = sorted(path for path in segment_root.glob("block_*") if path.is_dir())
    if [path.name for path in block_dirs] != [f"block_{idx}" for idx in range(split_num)]:
        raise RuntimeError(f"unexpected MetroGS segment blocks: {block_dirs}")
    copied_hashes: Counter[str] = Counter()
    linked_inodes: Counter[tuple[int, int]] = Counter()
    rows: list[dict[str, Any]] = []
    for block in block_dirs:
        images = image_files(block / "images")
        if not images:
            raise RuntimeError(f"empty MetroGS segment: {block}")
        upstream_sparse = block / "sparse" / "0"
        sparse = block / "sparse_closed" / "0"
        for filename in ("cameras.bin", "images.bin", "points3D.bin"):
            if not (upstream_sparse / filename).is_file():
                raise FileNotFoundError(upstream_sparse / filename)
            if not (sparse / filename).is_file():
                raise FileNotFoundError(sparse / filename)
        closure_manifest = block / "track_closure_manifest.json"
        closure = json.loads(closure_manifest.read_text(encoding="utf-8"))
        if closure.get("status") != "PASS" or closure.get("passed") is not True:
            raise RuntimeError(f"block track closure failed: {block}")
        hashes = [sha256(path) for path in images]
        copied_hashes.update(hashes)
        linked_inodes.update((path.stat().st_dev, path.stat().st_ino) for path in images)
        rows.append(
            {
                "block": block.name,
                "image_count": len(images),
                "image_hash_aggregate_sha256": hashlib.sha256(
                    "\n".join(sorted(hashes)).encode("ascii")
                ).hexdigest(),
                "upstream_sparse": {
                    filename: {
                        "bytes": (upstream_sparse / filename).stat().st_size,
                        "sha256": sha256(upstream_sparse / filename),
                    }
                    for filename in ("cameras.bin", "images.bin", "points3D.bin")
                },
                "sparse": {
                    filename: {
                        "bytes": (sparse / filename).stat().st_size,
                        "sha256": sha256(sparse / filename),
                    }
                    for filename in ("cameras.bin", "images.bin", "points3D.bin")
                },
                "track_closure_manifest": {
                    "path": str(closure_manifest),
                    "bytes": closure_manifest.stat().st_size,
                    "sha256": sha256(closure_manifest),
                    "removed_point_count": int(closure["removed_point_count"]),
                    "removed_track_element_count": int(
                        closure["removed_track_element_count"]
                    ),
                },
            }
        )
    if copied_hashes != source_image_hashes:
        raise RuntimeError("segmented image bytes are not an exact partition of the 2196 training images")
    if linked_inodes != source_image_inodes:
        raise RuntimeError("segmented RGB files are not exact hardlinks to the frozen train images")
    return rows


def validate_ply(path: Path) -> dict[str, Any]:
    ply = PlyData.read(path)
    vertices = ply["vertex"].data
    count = len(vertices)
    if count <= 0:
        raise RuntimeError(f"empty pointmap PLY: {path}")
    xyz = np.column_stack([vertices[axis] for axis in ("x", "y", "z")])
    if not np.isfinite(xyz).all():
        raise RuntimeError(f"non-finite pointmap PLY coordinates: {path}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "vertex_count": count,
        "xyz_min": [float(value) for value in xyz.min(axis=0)],
        "xyz_max": [float(value) for value in xyz.max(axis=0)],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--formal_input_manifest", type=Path, required=True)
    parser.add_argument("--moge_path", type=Path, required=True)
    parser.add_argument("--moge_weight", type=Path, required=True)
    parser.add_argument("--pi3_weight", type=Path, required=True)
    parser.add_argument("--compatibility_patch", type=Path, required=True)
    parser.add_argument("--colmap_io", type=Path, required=True)
    parser.add_argument("--subset_track_closure_tool", type=Path, required=True)
    parser.add_argument("--additional_ply", type=Path, required=True)
    parser.add_argument("--evidence_output", type=Path, required=True)
    parser.add_argument("--expected_repo_commit", required=True)
    parser.add_argument("--expected_repo_tree", required=True)
    parser.add_argument("--expected_runtime_status", default=" M utils/get_mask_depth_scales.py")
    parser.add_argument("--expected_moge_sha256", required=True)
    parser.add_argument("--expected_pi3_sha256", required=True)
    parser.add_argument("--expected_cameras_sha256", required=True)
    parser.add_argument("--expected_images_sha256", required=True)
    parser.add_argument("--expected_points3d_sha256", required=True)
    parser.add_argument("--split_num", type=int, default=4)
    parser.add_argument("--multi_view_max_dis", type=float, default=1.5)
    args = parser.parse_args()

    repo = args.repo.resolve()
    # Preserve the virtual-environment launcher instead of resolving its
    # symlink to the system interpreter and losing the frozen packages.
    python = Path(os.path.abspath(os.fspath(args.python)))
    dataset = args.dataset.resolve()
    manifest_path = args.formal_input_manifest.resolve()
    moge_path = args.moge_path.resolve()
    moge_weight = args.moge_weight.resolve()
    pi3_weight = args.pi3_weight.resolve()
    patch = args.compatibility_patch.resolve()
    colmap_io = args.colmap_io.resolve()
    subset_track_closure_tool = args.subset_track_closure_tool.resolve()
    additional_ply = args.additional_ply.resolve()
    evidence_output = args.evidence_output.resolve()

    if args.split_num != 4:
        raise ValueError("the frozen MetroGS MatrixCity route uses exactly four Pi3 segments")
    if args.multi_view_max_dis != 1.5:
        raise ValueError(
            "the frozen MetroGS MatrixCity script uses multi_view_filter.py's "
            "unoverridden official default of 1.5"
        )
    for required in (
        python,
        manifest_path,
        moge_weight,
        pi3_weight,
        patch,
        colmap_io,
        subset_track_closure_tool,
        dataset / "images",
        dataset / "sparse" / "0" / "cameras.bin",
        dataset / "sparse" / "0" / "images.bin",
        dataset / "sparse" / "0" / "points3D.bin",
        repo / "utils" / "run_moge_v2.py",
        repo / "utils" / "get_mask_depth_scales.py",
        repo / "utils" / "multi_view_filter.py",
        repo / "pointmap" / "scene_images_segment.py",
        repo / "pointmap" / "Pi3-Align" / "X_long.py",
        repo / "pointmap" / "Pi3-Align" / "configs" / "mc_aerial.yaml",
        repo / "pointmap" / "merge_all.py",
        moge_path / "moge" / "model" / "v2.py",
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    for forbidden in (
        dataset / "estimated_mask_depths",
        dataset / "estimated_mask_depth_scales.json",
        dataset / "multi_view.json",
        dataset / "segments",
        dataset / "metrogs_pi3_config_frozen.yaml",
        additional_ply,
        evidence_output,
    ):
        if forbidden.exists():
            raise FileExistsError(f"MetroGS prior output must be fresh: {forbidden}")

    repo_commit = git_output(repo, "rev-parse", "HEAD")
    repo_tree = git_output(repo, "rev-parse", "HEAD^{tree}")
    repo_status = git_output(repo, "status", "--porcelain=v1")
    if repo_commit != args.expected_repo_commit or repo_tree != args.expected_repo_tree:
        raise RuntimeError(f"MetroGS source identity mismatch: {repo_commit} {repo_tree}")
    if repo_status != args.expected_runtime_status:
        raise RuntimeError(f"unexpected MetroGS prior-runtime diff: {repo_status!r}")
    subprocess.run(
        ["git", "-C", str(repo), "apply", "--reverse", "--check", str(patch)],
        check=True,
    )

    formal = json.loads(manifest_path.read_text(encoding="utf-8"))
    if formal["schema"] != "gs_gcp_colmap_native_quarter_materialized_input_manifest_v1":
        raise RuntimeError("unexpected formal input manifest schema")
    if formal["scene"] != "gcp_100000_20260610":
        raise RuntimeError("MetroGS preparation is frozen to the 100K scene")
    train_records = [record for record in formal["images"] if record["role"] == "train"]
    if len(train_records) != 2196 or int(formal["test_view_count"]) != 314:
        raise RuntimeError("frozen 2196/314 split mismatch")
    image_inventory = validate_training_images(dataset / "images", train_records)
    expected_names = {record["image_name"] for record in train_records}
    source_image_hashes = Counter(record["sha256"] for record in image_inventory)

    sparse_hashes = {
        "cameras.bin": require_sha256(
            dataset / "sparse" / "0" / "cameras.bin",
            args.expected_cameras_sha256,
            "cameras.bin",
        ),
        "images.bin": require_sha256(
            dataset / "sparse" / "0" / "images.bin",
            args.expected_images_sha256,
            "images.bin",
        ),
        "points3D.bin": require_sha256(
            dataset / "sparse" / "0" / "points3D.bin",
            args.expected_points3d_sha256,
            "points3D.bin",
        ),
    }
    moge_hash = require_sha256(moge_weight, args.expected_moge_sha256, "MoGe weight")
    pi3_hash = require_sha256(pi3_weight, args.expected_pi3_sha256, "Pi3 weight")

    env = dict(os.environ)
    env["PATH"] = str(python.parent) + os.pathsep + env.get("PATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONHASHSEED"] = "0"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["WANDB_MODE"] = "offline"
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(name, None)

    commands: dict[str, Any] = {}
    depth_dir = dataset / "estimated_mask_depths"
    commands["moge"] = run_checked(
        [
            python,
            repo / "utils" / "run_moge_v2.py",
            dataset / "images",
            "--output",
            depth_dir,
            "--dataset_dir",
            dataset,
            "--downsample_factor",
            "1",
            "--moge_path",
            moge_path,
        ],
        cwd=repo,
        env=env,
    )
    depth_rows = validate_depths(depth_dir, train_records)

    scale_path = dataset / "estimated_mask_depth_scales.json"
    commands["depth_scale"] = run_checked(
        [python, repo / "utils" / "get_mask_depth_scales.py", dataset],
        cwd=repo,
        env=env,
    )
    scale_rows, median_scale, scale_accepted, scale_rejected = validate_scales(
        scale_path, expected_names
    )

    multi_view_path = dataset / "multi_view.json"
    commands["multi_view"] = run_checked(
        [
            python,
            repo / "utils" / "multi_view_filter.py",
            dataset,
            "--split_mode",
            "reconstruction",
            "--downsample_factor",
            "1",
            "--multi_view_max_dis",
            str(args.multi_view_max_dis),
        ],
        cwd=repo,
        env=env,
    )
    multi_view = validate_multi_view(multi_view_path, expected_names)

    commands["segment"] = run_checked(
        [
            python,
            repo / "pointmap" / "scene_images_segment.py",
            dataset,
            "--split_num",
            str(args.split_num),
            "--split_mode",
            "reconstruction",
            "--downsample_factor",
            "1",
        ],
        cwd=repo,
        env=env,
    )
    segment_root = dataset / "segments"
    segment_rgb_storage = replace_segment_rgb_copies_with_hardlinks(
        segment_root, dataset / "images", args.split_num
    )
    commands["segment_track_closure"] = []
    for block_idx in range(args.split_num):
        block = segment_root / f"block_{block_idx}"
        commands["segment_track_closure"].append(
            run_checked(
                [
                    python,
                    subset_track_closure_tool,
                    "--input_model",
                    block / "sparse" / "0",
                    "--output_model",
                    block / "sparse_closed" / "0",
                    "--output_manifest",
                    block / "track_closure_manifest.json",
                    "--colmap_io",
                    colmap_io,
                ],
                cwd=repo,
                env=env,
            )
        )
    source_image_inodes = Counter(
        (path.stat().st_dev, path.stat().st_ino)
        for path in image_files(dataset / "images")
    )
    segment_rows = validate_segments(
        segment_root, source_image_hashes, source_image_inodes, args.split_num
    )

    pi3_template_path = repo / "pointmap" / "Pi3-Align" / "configs" / "mc_aerial.yaml"
    pi3_config = yaml.safe_load(pi3_template_path.read_text(encoding="utf-8"))
    if pi3_config["Model"]["loop_enable"] is not False or pi3_config["Model"]["useDBoW"] is not False:
        raise RuntimeError("frozen MatrixCity Pi3 route unexpectedly enables loop retrieval")
    pi3_config["Weights"]["Pi3"] = str(pi3_weight)
    pi3_config_path = dataset / "metrogs_pi3_config_frozen.yaml"
    pi3_config_path.write_text(
        yaml.safe_dump(pi3_config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    block_commands: list[list[str]] = []
    block_pointmaps: list[dict[str, Any]] = []
    for block_idx in range(args.split_num):
        block = segment_root / f"block_{block_idx}"
        command = run_checked(
            [
                python,
                repo / "pointmap" / "Pi3-Align" / "X_long.py",
                "--image_dir",
                block / "images",
                "--sparse_dir",
                block / "sparse_closed",
                "--save_dir",
                block / "output",
                "--config",
                pi3_config_path,
                "--Xname",
                "Pi3",
            ],
            cwd=repo / "pointmap" / "Pi3-Align",
            env=env,
        )
        block_commands.append(command)
        block_pointmaps.append(validate_ply(block / "output" / "pcd" / "combined_pcd.ply"))
    commands["pi3_blocks"] = block_commands

    additional_ply.parent.mkdir(parents=True, exist_ok=True)
    commands["merge_pointmaps"] = run_checked(
        [
            python,
            repo / "pointmap" / "merge_all.py",
            "--base_dir",
            dataset,
            "--output",
            additional_ply,
        ],
        cwd=repo,
        env=env,
    )
    merged_pointmap = validate_ply(additional_ply)

    if git_output(repo, "status", "--porcelain=v1") != args.expected_runtime_status:
        raise RuntimeError("MetroGS source changed while preparing priors")

    evidence: dict[str, Any] = {
        "schema": "m3m_gcp_native_quarter_metrogs_training_priors_v1",
        "protocol_id": "m3m_gcp_native_quarter_geometry_v2",
        "method_id": "metrogs",
        "scene": formal["scene"],
        "status": "PASS",
        "passed": True,
        "input_class": "rgb_colmap_external_geometry_prior",
        "source": {
            "repository_commit": repo_commit,
            "repository_tree": repo_tree,
            "runtime_status": repo_status,
            "compatibility_patch": str(patch),
            "compatibility_patch_sha256": sha256(patch),
            "subset_track_closure_tool": str(subset_track_closure_tool),
            "subset_track_closure_tool_sha256": sha256(subset_track_closure_tool),
            "patched_depth_scale_script_sha256": sha256(
                repo / "utils" / "get_mask_depth_scales.py"
            ),
        },
        "input": {
            "dataset": str(dataset),
            "formal_input_manifest_sha256": sha256(manifest_path),
            "train_view_count": len(image_inventory),
            "heldout_view_count": int(formal["test_view_count"]),
            "image_transform": "none; byte-identical frozen native-quarter JPEGs",
            "image_hash_aggregate_sha256": hashlib.sha256(
                "\n".join(
                    f'{row["image_name"]},{row["sha256"]},{row["bytes"]}'
                    for row in image_inventory
                ).encode("utf-8")
            ).hexdigest(),
            "sparse_hashes": sparse_hashes,
        },
        "moge": {
            "weight": str(moge_weight),
            "weight_bytes": moge_weight.stat().st_size,
            "weight_sha256": moge_hash,
            "runtime_source": str(moge_path),
            "depth_count": len(depth_rows),
            "depth_hash_aggregate_sha256": hashlib.sha256(
                "\n".join(
                    f'{row["image_name"]},{row["sha256"]},{row["bytes"]}'
                    for row in depth_rows
                ).encode("utf-8")
            ).hexdigest(),
            "depth_valid_fraction_min": min(row["valid_pixel_fraction"] for row in depth_rows),
            "depth_valid_fraction_max": max(row["valid_pixel_fraction"] for row in depth_rows),
            "scale_manifest": str(scale_path),
            "scale_manifest_sha256": sha256(scale_path),
            "scale_count": len(scale_rows),
            "scale_median": median_scale,
            "official_scale_bound_lower_factor": 0.2,
            "official_scale_bound_upper_factor": 5.0,
            "official_scale_bound_survivor_count": len(scale_accepted),
            "official_scale_bound_rejected_count": len(scale_rejected),
            "official_scale_bound_survivor_images": scale_accepted,
            "official_scale_bound_rejected_images": scale_rejected,
            "official_filter_semantics": (
                "all 2196 RGB views remain in the training set; only the depth prior is "
                "left unattached for out-of-bound views, exactly as the frozen upstream "
                "MetroGS dataparser implements"
            ),
        },
        "multi_view": {
            **multi_view,
            "max_distance": args.multi_view_max_dis,
            "rule": "exact official MatrixCity script route: multi_view_filter.py is called without a distance override, so its default maximum distance is 1.5",
        },
        "pi3": {
            "weight": str(pi3_weight),
            "weight_bytes": pi3_weight.stat().st_size,
            "weight_sha256": pi3_hash,
            "config_template": str(pi3_template_path),
            "config_template_sha256": sha256(pi3_template_path),
            "resolved_config": str(pi3_config_path),
            "resolved_config_sha256": sha256(pi3_config_path),
            "split_num": args.split_num,
            "segment_image_counts": [row["image_count"] for row in segment_rows],
            "segment_rgb_storage": segment_rgb_storage,
            "segments": segment_rows,
            "block_pointmaps": block_pointmaps,
            "merged_pointmap": merged_pointmap,
            "loop_enable": False,
            "use_dbow": False,
        },
        "commands": commands,
        "claims": {
            "training_rgb_only": True,
            "training_colmap_only": True,
            "external_geometry_priors": ["MoGe-2", "Pi3-Align"],
            "heldout_rgb_read": False,
            "gcp_truth_read": False,
            "lidar_read": False,
            "image_pixels_changed": False,
            "segment_rgb_additional_physical_bytes": 0,
            "rgb_training_views_removed_by_depth_scale_filter": False,
            "formal_training_started": False,
        },
        "runtime": {
            "python": sys.version,
            "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"],
        },
    }
    evidence_output.parent.mkdir(parents=True, exist_ok=True)
    evidence_output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
