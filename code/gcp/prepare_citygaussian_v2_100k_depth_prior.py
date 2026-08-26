#!/usr/bin/env python3
"""Prepare CityGaussianV2's frozen training-only Depth Anything V2 prior."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city_repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--formal_input_manifest", type=Path, required=True)
    parser.add_argument("--da2_root", type=Path, required=True)
    parser.add_argument("--manifest_output", type=Path, required=True)
    parser.add_argument("--expected_city_commit", required=True)
    parser.add_argument("--expected_da2_commit", required=True)
    parser.add_argument("--expected_da2_weight_sha256", required=True)
    parser.add_argument("--expected_scene", default="gcp_100000_20260610")
    parser.add_argument("--expected_train_count", type=int, default=2196)
    parser.add_argument("--expected_heldout_count", type=int, default=314)
    parser.add_argument("--input_size", type=int, default=518)
    parser.add_argument("--point_max_error", type=float, default=1.5)
    args = parser.parse_args()

    city_repo = args.city_repo.resolve()
    # Keep the virtual-environment launcher path lexical. Resolving its symlink
    # to /usr/bin/python bypasses the environment's pyvenv.cfg and packages.
    python = Path(os.path.abspath(os.fspath(args.python)))
    dataset = args.dataset.resolve()
    formal_manifest_path = args.formal_input_manifest.resolve()
    da2_root = args.da2_root.resolve()
    output_manifest = args.manifest_output.resolve()
    image_dir = dataset / "images"
    depth_dir = dataset / "estimated_depths"
    scales_path = dataset / "estimated_depth_scales.json"
    weight_path = da2_root / "checkpoints" / "depth_anything_v2_vitl.pth"

    if args.input_size != 518:
        raise ValueError("the frozen CityGaussianV2 route requires DAv2 input_size=518")
    if not math.isclose(args.point_max_error, 1.5, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("the frozen CityGaussianV2 route requires point_max_error=1.5")
    if depth_dir.exists() or scales_path.exists() or output_manifest.exists():
        raise FileExistsError("depth prior outputs already exist; overwrite/resume is forbidden")
    for required in (
        city_repo / ".git",
        city_repo / "utils" / "run_depth_anything_v2.py",
        city_repo / "utils" / "get_depth_scales.py",
        python,
        formal_manifest_path,
        image_dir,
        dataset / "sparse" / "0" / "cameras.bin",
        dataset / "sparse" / "0" / "images.bin",
        dataset / "sparse" / "0" / "points3D.bin",
        da2_root / "depth_anything_v2" / "dpt.py",
        weight_path,
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    city_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=city_repo, text=True
    ).strip()
    if city_commit != args.expected_city_commit:
        raise RuntimeError(
            f"CityGaussianV2 commit mismatch: expected {args.expected_city_commit}, got {city_commit}"
        )
    city_status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=city_repo, text=True
    )
    if city_status:
        raise RuntimeError("depth prior generation requires a clean official CityGaussianV2 tree")

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
            f"frozen split must contain {args.expected_train_count} train and "
            f"{args.expected_heldout_count} heldout views, got "
            f"{len(train_records)} and {len(heldout_records)}"
        )

    expected_names = {record["image_name"] for record in train_records}
    actual_files = sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    actual_names = {path.relative_to(image_dir).as_posix() for path in actual_files}
    if actual_names != expected_names:
        raise RuntimeError(
            "isolated CityGaussianV2 image root differs from the frozen training set: "
            f"missing={sorted(expected_names - actual_names)}, "
            f"extra={sorted(actual_names - expected_names)}"
        )

    input_records: list[dict[str, Any]] = []
    by_name = {record["image_name"]: record for record in train_records}
    for path in actual_files:
        name = path.relative_to(image_dir).as_posix()
        frozen = by_name[name]
        actual_hash = require_sha256(path, frozen["jpeg_sha256"], f"training image {name}")
        if path.stat().st_size != frozen["jpeg_bytes"]:
            raise RuntimeError(f"training image byte count mismatch: {name}")
        input_records.append(
            {
                "image_name": name,
                "bytes": path.stat().st_size,
                "sha256": actual_hash,
                "width": frozen["width"],
                "height": frozen["height"],
            }
        )

    weight_hash = require_sha256(
        weight_path, args.expected_da2_weight_sha256, "Depth Anything V2 Large weight"
    )
    env = dict(os.environ)
    env["PATH"] = str(python.parent) + os.pathsep + env.get("PATH", "")
    env["PYTHONUNBUFFERED"] = "1"

    depth_command: list[Path | str] = [
        python,
        "utils/run_depth_anything_v2.py",
        image_dir,
        "--input_size",
        str(args.input_size),
        "--output",
        depth_dir,
        "--downsample_factor",
        "1",
        "--encoder",
        "vitl",
        "--da2_path",
        da2_root,
    ]
    run_checked(depth_command, cwd=city_repo, env=env)

    expected_depth_names = {f"{name}.npy" for name in expected_names}
    depth_files = sorted(path for path in depth_dir.rglob("*.npy") if path.is_file())
    actual_depth_names = {path.relative_to(depth_dir).as_posix() for path in depth_files}
    if actual_depth_names != expected_depth_names:
        raise RuntimeError(
            f"depth output inventory mismatch: missing={sorted(expected_depth_names - actual_depth_names)}, "
            f"extra={sorted(actual_depth_names - expected_depth_names)}"
        )

    depth_records: list[dict[str, Any]] = []
    for path in depth_files:
        array = np.load(path, allow_pickle=False)
        source_name = path.relative_to(depth_dir).as_posix()[: -len(".npy")]
        source = by_name[source_name]
        if array.shape != (source["height"], source["width"]):
            raise RuntimeError(
                f"depth shape mismatch for {source_name}: {array.shape} != "
                f"{(source['height'], source['width'])}"
            )
        if array.dtype != np.float32:
            raise RuntimeError(f"depth dtype mismatch for {source_name}: {array.dtype}")
        if not np.isfinite(array).all():
            raise RuntimeError(f"non-finite depth values for {source_name}")
        minimum = float(array.min())
        maximum = float(array.max())
        if minimum < -1e-6 or maximum > 1.000001:
            raise RuntimeError(
                f"normalized inverse-depth range violation for {source_name}: {minimum}, {maximum}"
            )
        depth_records.append(
            {
                "image_name": source_name,
                "relative_path": path.relative_to(dataset).as_posix(),
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "minimum": minimum,
                "maximum": maximum,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    scale_command: list[Path | str] = [
        python,
        "utils/get_depth_scales.py",
        dataset,
        "--depth_dir",
        depth_dir,
        "--output",
        scales_path,
        "--point-max-error",
        str(args.point_max_error),
    ]
    run_checked(scale_command, cwd=city_repo, env=env)

    scales = json.loads(scales_path.read_text(encoding="utf-8"))
    if set(scales) != expected_names:
        raise RuntimeError(
            f"depth-scale inventory mismatch: missing={sorted(expected_names - set(scales))}, "
            f"extra={sorted(set(scales) - expected_names)}"
        )
    invalid_scale_names = [
        name
        for name, values in scales.items()
        if not all(math.isfinite(float(values[key])) for key in ("scale", "offset"))
    ]
    if invalid_scale_names:
        raise RuntimeError(f"non-finite depth scales: {invalid_scale_names}")
    zero_scale_names = [
        name for name, values in scales.items() if float(values["scale"]) == 0.0
    ]
    if zero_scale_names:
        raise RuntimeError(f"zero depth scales are forbidden: {zero_scale_names}")
    scale_values = np.asarray(
        [float(values["scale"]) for values in scales.values()], dtype=np.float64
    )
    median_scale = float(np.median(scale_values))
    if not math.isfinite(median_scale) or median_scale <= 0.0:
        raise RuntimeError(f"invalid median depth scale: {median_scale}")
    enabled_depth_names = sorted(
        name
        for name, values in scales.items()
        if 0.2 * median_scale <= float(values["scale"]) <= 5.0 * median_scale
    )
    rejected_depth_names = sorted(set(scales) - set(enabled_depth_names))
    if not enabled_depth_names:
        raise RuntimeError("official 0.2x-5x median scale bounds reject every depth map")

    evidence: dict[str, Any] = {
        "schema": "m3m_gcp_native_quarter_citygaussian_v2_depth_prior_v1",
        "protocol_id": "m3m_gcp_native_quarter_geometry_v2",
        "method_id": "citygaussian_v2",
        "scene": formal["scene"],
        "status": "PASS",
        "passed": True,
        "input_class": "rgb_colmap_external_geometry_prior",
        "citygaussian_v2": {
            "repository_commit": city_commit,
            "run_depth_script_sha256": sha256(
                city_repo / "utils" / "run_depth_anything_v2.py"
            ),
            "get_depth_scales_script_sha256": sha256(
                city_repo / "utils" / "get_depth_scales.py"
            ),
        },
        "depth_anything_v2": {
            "repository_commit": args.expected_da2_commit,
            "encoder": "vitl",
            "input_size": args.input_size,
            "weight_path": str(weight_path),
            "weight_bytes": weight_path.stat().st_size,
            "weight_sha256": weight_hash,
            "per_image_normalization": "official CityGaussianV2 wrapper min-max to [0,1]",
            "output_semantics": "normalized monocular inverse-depth proxy used by official CityGaussianV2 depth-scale fitting",
        },
        "access_boundary": {
            "isolated_dataset_root": str(dataset),
            "training_rgb_opened": len(input_records),
            "heldout_rgb_opened": 0,
            "gcp_annotations_opened": 0,
            "lidar_opened": 0,
            "only_training_rgb_and_train_only_colmap_supplied_to_prior_commands": True,
        },
        "formal_input_manifest": {
            "path": str(formal_manifest_path),
            "file_sha256": sha256(formal_manifest_path),
            "canonical_sha256": formal["manifest_sha256"],
            "train_view_count": len(train_records),
            "heldout_view_count": len(heldout_records),
        },
        "training_images": input_records,
        "depth_outputs": depth_records,
        "depth_scales": {
            "path": str(scales_path),
            "sha256": sha256(scales_path),
            "point_max_error": args.point_max_error,
            "record_count": len(scales),
            "zero_scale_count": 0,
            "median_scale": median_scale,
            "official_lower_bound_factor": 0.2,
            "official_upper_bound_factor": 5.0,
            "enabled_depth_count": len(enabled_depth_names),
            "enabled_depth_names": enabled_depth_names,
            "rejected_depth_count": len(rejected_depth_names),
            "rejected_depth_names": rejected_depth_names,
            "thresholds_selected_from_results": False,
        },
        "commands": {
            "depth_generation": command_record(depth_command),
            "colmap_scale_fit": command_record(scale_command),
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
