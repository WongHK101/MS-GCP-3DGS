#!/usr/bin/env python3
"""Prepare CityGS-X's frozen training-only DAv2 and multi-view prior.

The historical entry point name is retained for compatibility.  Scene identity
and split counts are explicit guards so the already-qualified route can be
reused without embedding 3K/20K/100K inventory constants in the algorithm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_record(args: list[Path | str]) -> list[str]:
    return [str(value) for value in args]


def run_checked(args: list[Path | str], *, cwd: Path, env: dict[str, str]) -> None:
    command = command_record(args)
    print("RUN", json.dumps(command, ensure_ascii=False), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def require_sha256(path: Path, expected: str, label: str) -> str:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"{label} SHA256 mismatch: expected {expected}, got {actual}")
    return actual


def require_image_inventory(
    image_dir: Path, train_records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    expected_names = {record["image_name"] for record in train_records}
    files = sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    actual_names = {path.relative_to(image_dir).as_posix() for path in files}
    if actual_names != expected_names:
        raise RuntimeError(
            "isolated CityGS-X image root differs from the frozen "
            f"{len(expected_names)}-view train set: "
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
                "width": frozen["width"],
                "height": frozen["height"],
            }
        )
    return inventory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city_repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--formal_input_manifest", type=Path, required=True)
    parser.add_argument("--da2_root", type=Path, required=True)
    parser.add_argument("--pytorch3d_compat", type=Path, required=True)
    parser.add_argument("--manifest_output", type=Path, required=True)
    parser.add_argument("--expected_city_commit", required=True)
    parser.add_argument("--expected_camera_utils_sha256", required=True)
    parser.add_argument("--expected_dataset_readers_sha256", required=True)
    parser.add_argument("--expected_da2_commit", required=True)
    parser.add_argument("--expected_da2_run_sha256", required=True)
    parser.add_argument("--expected_da2_weight_sha256", required=True)
    parser.add_argument("--expected_cameras_sha256", required=True)
    parser.add_argument("--expected_images_sha256", required=True)
    parser.add_argument("--expected_points3d_sha256", required=True)
    parser.add_argument("--expected_scene", default="gcp_100000_20260610")
    parser.add_argument("--expected_train_count", type=int, default=2196)
    parser.add_argument("--expected_heldout_count", type=int, default=314)
    parser.add_argument("--input_size", type=int, default=518)
    parser.add_argument("--resolution", type=int, default=1)
    parser.add_argument("--pixel_thred", type=float, default=1.0)
    parser.add_argument("--multi_view_num", type=int, default=8)
    parser.add_argument("--multi_view_max_angle", type=int, default=15)
    parser.add_argument("--multi_view_min_dis", type=float, default=0.01)
    parser.add_argument("--multi_view_max_dis", type=float, default=25.0)
    args = parser.parse_args()

    city_repo = args.city_repo.resolve()
    # Keep the virtual-environment launcher path lexical. Resolving its symlink
    # to /usr/bin/python bypasses the environment's pyvenv.cfg and packages.
    python = Path(os.path.abspath(os.fspath(args.python)))
    dataset = args.dataset.resolve()
    formal_manifest_path = args.formal_input_manifest.resolve()
    da2_root = args.da2_root.resolve()
    pytorch3d_compat = args.pytorch3d_compat.resolve()
    output_manifest = args.manifest_output.resolve()
    image_dir = dataset / "images"
    depth_dir = dataset / "depth"
    mask_dir = dataset / "mask"
    sparse_dir = dataset / "sparse" / "0"
    depth_params_path = sparse_dir / "depth_params.json"
    da2_run = da2_root / "run.py"
    weight_path = da2_root / "checkpoints" / "depth_anything_v2_vitl.pth"

    if args.input_size != 518:
        raise ValueError("the frozen CityGS-X route requires DAv2 input_size=518")
    if args.resolution != 1:
        raise ValueError("the native-quarter CityGS-X route requires resolution=1")
    if not math.isclose(args.pixel_thred, 1.0, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("the frozen CityGS-X multi-view mask route requires pixel_thred=1")
    neighbor_route = {
        "multi_view_num": args.multi_view_num,
        "multi_view_max_angle_deg": args.multi_view_max_angle,
        "multi_view_min_dis": args.multi_view_min_dis,
        "multi_view_max_dis": args.multi_view_max_dis,
    }
    expected_neighbor_route = {
        "multi_view_num": 8,
        "multi_view_max_angle_deg": 15,
        "multi_view_min_dis": 0.01,
        "multi_view_max_dis": 25.0,
    }
    if neighbor_route != expected_neighbor_route:
        raise ValueError(
            "the frozen CityGS-X aerial neighbor route must equal "
            f"{expected_neighbor_route}, got {neighbor_route}"
        )
    path_parts = {part.lower() for part in dataset.parts}
    if "train" not in path_parts or "matrixcity" not in str(dataset).lower():
        raise RuntimeError(
            "CityGS-X compatibility root must contain a train path component and MatrixCity marker"
        )
    if depth_dir.exists() or mask_dir.exists() or depth_params_path.exists() or output_manifest.exists():
        raise FileExistsError("prior outputs already exist; overwrite/resume is forbidden")
    for required in (
        city_repo / ".git",
        city_repo / "utils" / "make_depth_scale.py",
        city_repo / "multi_view_precess.py",
        city_repo / "utils" / "camera_utils.py",
        city_repo / "scene" / "dataset_readers.py",
        python,
        formal_manifest_path,
        image_dir,
        sparse_dir / "cameras.bin",
        sparse_dir / "images.bin",
        sparse_dir / "points3D.bin",
        sparse_dir / "points3D.ply",
        da2_run,
        da2_root / "depth_anything_v2" / "dpt.py",
        weight_path,
        pytorch3d_compat / "pytorch3d" / "transforms" / "__init__.py",
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    city_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=city_repo, text=True
    ).strip()
    if city_commit != args.expected_city_commit:
        raise RuntimeError(
            f"CityGS-X commit mismatch: expected {args.expected_city_commit}, got {city_commit}"
        )
    city_status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=city_repo,
        text=True,
    ).splitlines()
    if city_status != [" M scene/dataset_readers.py", " M utils/camera_utils.py"]:
        raise RuntimeError(f"unexpected CityGS-X training-runtime diff: {city_status}")
    camera_utils_hash = require_sha256(
        city_repo / "utils" / "camera_utils.py",
        args.expected_camera_utils_sha256,
        "CityGS-X case-insensitive prior-suffix compatibility file",
    )
    dataset_readers_hash = require_sha256(
        city_repo / "scene" / "dataset_readers.py",
        args.expected_dataset_readers_sha256,
        "CityGS-X preserve-existing-image-suffix compatibility file",
    )
    sparse_hashes = {
        "cameras.bin": require_sha256(
            sparse_dir / "cameras.bin", args.expected_cameras_sha256, "training cameras"
        ),
        "images.bin": require_sha256(
            sparse_dir / "images.bin", args.expected_images_sha256, "training images model"
        ),
        "points3D.bin": require_sha256(
            sparse_dir / "points3D.bin", args.expected_points3d_sha256, "training points3D"
        ),
    }
    da2_run_hash = require_sha256(
        da2_run, args.expected_da2_run_sha256, "Depth Anything V2 run.py"
    )
    weight_hash = require_sha256(
        weight_path, args.expected_da2_weight_sha256, "Depth Anything V2 Large weight"
    )

    formal = json.loads(formal_manifest_path.read_text(encoding="utf-8"))
    if formal.get("scene") != args.expected_scene:
        raise RuntimeError(f"unexpected scene: {formal.get('scene')}")
    train_records = [record for record in formal["images"] if record["role"] == "train"]
    heldout_records = [record for record in formal["images"] if record["role"] == "test"]
    if (
        len(train_records) != args.expected_train_count
        or len(heldout_records) != args.expected_heldout_count
    ):
        raise RuntimeError(
            "frozen split must contain "
            f"{args.expected_train_count} train and "
            f"{args.expected_heldout_count} heldout views, got "
            f"{len(train_records)} and {len(heldout_records)}"
        )
    input_records = require_image_inventory(image_dir, train_records)
    expected_stems = {Path(record["image_name"]).stem for record in train_records}
    if len(expected_stems) != len(train_records):
        raise RuntimeError("training image stems are not unique")

    env = dict(os.environ)
    env["PATH"] = str(python.parent) + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(pytorch3d_compat), str(city_repo), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["WANDB_MODE"] = "offline"
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(name, None)

    depth_command: list[Path | str] = [
        python,
        da2_run,
        "--encoder",
        "vitl",
        "--input-size",
        str(args.input_size),
        "--pred-only",
        "--grayscale",
        "--img-path",
        image_dir,
        "--outdir",
        depth_dir,
    ]
    run_checked(depth_command, cwd=da2_root, env=env)

    depth_files = sorted(depth_dir.glob("*.png"))
    actual_depth_stems = {path.stem for path in depth_files}
    if actual_depth_stems != expected_stems:
        raise RuntimeError(
            f"depth inventory mismatch: missing={sorted(expected_stems - actual_depth_stems)}, "
            f"extra={sorted(actual_depth_stems - expected_stems)}"
        )
    by_stem = {Path(record["image_name"]).stem: record for record in train_records}
    depth_records: list[dict[str, Any]] = []
    for path in depth_files:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"cannot read generated depth image: {path}")
        source = by_stem[path.stem]
        if image.shape != (source["height"], source["width"], 3):
            raise RuntimeError(f"depth shape mismatch for {path.name}: {image.shape}")
        if image.dtype != np.uint8:
            raise RuntimeError(f"depth dtype mismatch for {path.name}: {image.dtype}")
        if not (np.array_equal(image[..., 0], image[..., 1]) and np.array_equal(image[..., 1], image[..., 2])):
            raise RuntimeError(f"official grayscale DAv2 output has unequal channels: {path.name}")
        depth_records.append(
            {
                "image_stem": path.stem,
                "relative_path": path.relative_to(dataset).as_posix(),
                "shape": list(image.shape),
                "dtype": str(image.dtype),
                "minimum": int(image.min()),
                "maximum": int(image.max()),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    scale_command: list[Path | str] = [
        python,
        city_repo / "utils" / "make_depth_scale.py",
        "--base_dir",
        dataset,
        "--depths_dir",
        depth_dir,
    ]
    run_checked(scale_command, cwd=city_repo, env=env)
    depth_params = json.loads(depth_params_path.read_text(encoding="utf-8"))
    if set(depth_params) != expected_stems:
        raise RuntimeError(
            f"depth-scale inventory mismatch: missing={sorted(expected_stems - set(depth_params))}, "
            f"extra={sorted(set(depth_params) - expected_stems)}"
        )
    invalid_params = [
        name
        for name, values in depth_params.items()
        if not all(math.isfinite(float(values[key])) for key in ("scale", "offset"))
        or float(values["scale"]) <= 0.0
    ]
    if invalid_params:
        raise RuntimeError(f"non-finite or non-positive depth scales: {invalid_params}")

    mask_command: list[Path | str] = [
        python,
        city_repo / "multi_view_precess.py",
        "-s",
        dataset,
        "--resolution",
        str(args.resolution),
        "--model_path",
        mask_dir,
        "--images",
        "images",
        "--pixel_thred",
        str(args.pixel_thred),
        "--multi_view_num",
        str(args.multi_view_num),
        "--multi_view_max_angle",
        str(args.multi_view_max_angle),
        "--multi_view_min_dis",
        str(args.multi_view_min_dis),
        "--multi_view_max_dis",
        str(args.multi_view_max_dis),
    ]
    run_checked(mask_command, cwd=city_repo, env=env)

    mask_files = sorted(mask_dir.glob("*.png"))
    actual_mask_stems = {path.stem for path in mask_files}
    if actual_mask_stems != expected_stems:
        raise RuntimeError(
            f"mask inventory mismatch: missing={sorted(expected_stems - actual_mask_stems)}, "
            f"extra={sorted(actual_mask_stems - expected_stems)}"
        )
    mask_records: list[dict[str, Any]] = []
    for path in mask_files:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"cannot read generated mask image: {path}")
        source = by_stem[path.stem]
        if image.shape != (source["height"], source["width"], 3):
            raise RuntimeError(f"mask shape mismatch for {path.name}: {image.shape}")
        valid = np.any(image != 0, axis=-1)
        valid_count = int(valid.sum())
        if valid_count == 0:
            raise RuntimeError(f"multi-view mask rejects every pixel: {path.name}")
        mask_records.append(
            {
                "image_stem": path.stem,
                "relative_path": path.relative_to(dataset).as_posix(),
                "shape": list(image.shape),
                "dtype": str(image.dtype),
                "valid_pixel_count": valid_count,
                "valid_pixel_fraction": valid_count / float(valid.size),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    evidence: dict[str, Any] = {
        "schema": "m3m_gcp_native_quarter_citygs_x_depth_prior_v1",
        "protocol_id": "m3m_gcp_native_quarter_geometry_v2",
        "method_id": "citygs_x",
        "scene": formal["scene"],
        "status": "PASS",
        "passed": True,
        "input_class": "rgb_colmap_external_geometry_prior",
        "citygs_x": {
            "repository_commit": city_commit,
            "camera_utils_sha256": camera_utils_hash,
            "dataset_readers_sha256": dataset_readers_hash,
            "make_depth_scale_sha256": sha256(city_repo / "utils" / "make_depth_scale.py"),
            "multi_view_precess_sha256": sha256(city_repo / "multi_view_precess.py"),
            "resolution": args.resolution,
            "pixel_thred": args.pixel_thred,
            "multi_view_neighbor_selection": neighbor_route,
            "neighbor_selection_basis": "upstream MatrixCity aerial command frozen before any CityGS-X result; uses training camera geometry only",
            "suffix_compatibility_scope": "I/O-only preservation of an existing frozen upper-case JPEG path plus mapping to official PNG prior filenames",
        },
        "depth_anything_v2": {
            "repository_commit": args.expected_da2_commit,
            "run_py_sha256": da2_run_hash,
            "encoder": "vitl",
            "input_size": args.input_size,
            "pred_only": True,
            "grayscale": True,
            "weight_path": str(weight_path),
            "weight_bytes": weight_path.stat().st_size,
            "weight_sha256": weight_hash,
            "output_semantics": "official per-image normalized uint8 inverse-depth proxy",
        },
        "dataset": {
            "path": str(dataset),
            "sparse_sha256": sparse_hashes,
            "training_images": input_records,
            "depth_outputs": depth_records,
            "depth_params": {
                "path": str(depth_params_path),
                "sha256": sha256(depth_params_path),
                "record_count": len(depth_params),
                "zero_or_negative_scale_count": 0,
            },
            "multi_view_masks": mask_records,
        },
        "formal_input_manifest": {
            "path": str(formal_manifest_path),
            "file_sha256": sha256(formal_manifest_path),
            "canonical_sha256": formal["manifest_sha256"],
            "train_view_count": len(train_records),
            "heldout_view_count": len(heldout_records),
        },
        "access_boundary": {
            "training_rgb_opened": len(input_records),
            "heldout_rgb_opened": 0,
            "gcp_annotations_opened": 0,
            "lidar_opened": 0,
            "only_training_rgb_and_train_only_colmap_supplied_to_prior_commands": True,
        },
        "commands": {
            "depth_generation": command_record(depth_command),
            "colmap_scale_fit": command_record(scale_command),
            "multi_view_mask": command_record(mask_command),
        },
        "claims": {
            "heldout_gcp_lidar_or_orthophoto_truth_used": False,
            "result_driven_prior_or_hyperparameter_selection": False,
            "physical_metric_depth_claim": False,
        },
    }
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
