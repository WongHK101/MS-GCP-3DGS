#!/usr/bin/env python3
"""Run the frozen single-GPU CityGS-X training route with immutable checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


CITYGS_X_COMMIT = "27617f2486505e3b6fe75345edf7c2b11161bc2a"
CITYGS_X_TREE = "f8b1b5148c1f47420ab698fd069bdb78acf901ab"
CAMERA_UTILS_COMPAT_SHA256 = "9326e6571685177543e34c903823b207b75258e96489d9398b08672637f5c9e3"
FORMAL_MANIFEST_FILE_SHA256 = "ae29817198f54f04e4133a7b5fd03df679dd6f259b2d1ef4125e825cbb8e422e"
FORMAL_MANIFEST_CANONICAL_SHA256 = "4ae07aad9278e2eb5af2f04268f3301df56c6f6ada9ee51c6f125fdbb29e7ec8"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def build_command(
    *,
    python: Path,
    repo: Path,
    dataset: Path,
    model_path: Path,
    mode: str,
    iterations: int,
) -> list[str]:
    if mode not in {"qualification", "formal"}:
        raise ValueError(f"unsupported mode: {mode}")
    if iterations <= 1:
        raise ValueError("CityGS-X training requires at least two iterations")
    if mode == "formal" and iterations != 100_000:
        raise ValueError("formal CityGS-X route is frozen to 100,000 iterations")

    command = [
        str(python),
        "-B",
        "train.py",
        "--bsz",
        "1",
        "-s",
        str(dataset),
        "--resolution",
        "1",
        "--model_path",
        str(model_path),
        "--iterations",
        str(iterations),
        "--images",
        "images",
        "--single_view_weight_from_iter",
        "10000" if mode == "formal" else "0",
        "--depth_l1_weight_final",
        "0.01",
        "--depth_l1_weight_init",
        "0.5",
        "--dpt_loss_from_iter",
        "10000" if mode == "formal" else "0",
        "--multi_view_weight_from_iter",
        "30000" if mode == "formal" else "0",
        "--default_voxel_size",
        "0.001",
        "--dpt_end_iter",
        "30000" if mode == "formal" else str(iterations),
        "--multi_view_patch_size",
        "3",
        "--test_iterations",
        str(iterations),
        "--save_iterations",
        str(iterations),
        "--quiet",
    ]
    del repo  # The caller supplies it as subprocess.cwd; keep it explicit in this API.
    return command


def verify_inputs(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    python = args.python.resolve()
    dataset = args.dataset.resolve()
    model_path = args.model_path.resolve()
    prior_path = args.prior_manifest.resolve()
    compat = args.pytorch3d_compat.resolve()
    if model_path.exists():
        raise FileExistsError(f"model path already exists: {model_path}")
    for required in (
        repo / ".git",
        repo / "train.py",
        repo / "utils" / "camera_utils.py",
        python,
        dataset / "images",
        dataset / "depth",
        dataset / "mask",
        dataset / "sparse" / "0" / "cameras.bin",
        dataset / "sparse" / "0" / "images.bin",
        dataset / "sparse" / "0" / "points3D.bin",
        dataset / "sparse" / "0" / "depth_params.json",
        prior_path,
        compat / "pytorch3d" / "transforms" / "__init__.py",
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    if git_output(repo, "rev-parse", "HEAD") != CITYGS_X_COMMIT:
        raise RuntimeError("CityGS-X repository commit mismatch")
    if git_output(repo, "rev-parse", "HEAD^{tree}") != CITYGS_X_TREE:
        raise RuntimeError("CityGS-X repository tree mismatch")
    if git_output(repo, "diff", "--name-only") != "utils/camera_utils.py":
        raise RuntimeError("unexpected CityGS-X training-runtime source diff")
    camera_hash = sha256(repo / "utils" / "camera_utils.py")
    if camera_hash != CAMERA_UTILS_COMPAT_SHA256:
        raise RuntimeError(f"CityGS-X camera_utils compatibility hash mismatch: {camera_hash}")

    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    if prior.get("status") != "PASS" or prior.get("passed") is not True:
        raise RuntimeError("CityGS-X prior evidence is not PASS")
    if prior.get("method_id") != "citygs_x":
        raise RuntimeError("unexpected prior method identity")
    if prior.get("citygs_x", {}).get("repository_commit") != CITYGS_X_COMMIT:
        raise RuntimeError("prior CityGS-X commit mismatch")
    if Path(prior.get("dataset", {}).get("path", "")).resolve() != dataset:
        raise RuntimeError("prior dataset path differs from training dataset")
    formal_manifest = prior.get("formal_input_manifest", {})
    if formal_manifest.get("file_sha256") != FORMAL_MANIFEST_FILE_SHA256:
        raise RuntimeError("formal input manifest file identity mismatch")
    if formal_manifest.get("canonical_sha256") != FORMAL_MANIFEST_CANONICAL_SHA256:
        raise RuntimeError("formal input manifest canonical identity mismatch")
    access = prior.get("access_boundary", {})
    expected_access = {
        "training_rgb_opened": 82,
        "heldout_rgb_opened": 0,
        "gcp_annotations_opened": 0,
        "lidar_opened": 0,
    }
    for key, expected in expected_access.items():
        if access.get(key) != expected:
            raise RuntimeError(f"prior access boundary mismatch for {key}: {access.get(key)}")
    image_count = sum(
        1
        for path in (dataset / "images").iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if image_count != 82:
        raise RuntimeError(f"CityGS-X training root must expose exactly 82 images, got {image_count}")
    return {
        "prior_manifest": str(prior_path),
        "prior_manifest_sha256": sha256(prior_path),
        "camera_utils_sha256": camera_hash,
        "training_image_count": image_count,
        "access_boundary": access,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--prior_manifest", type=Path, required=True)
    parser.add_argument("--pytorch3d_compat", type=Path, required=True)
    parser.add_argument("--mode", choices=("qualification", "formal"), required=True)
    parser.add_argument("--iterations", type=int, required=True)
    args = parser.parse_args()

    verified = verify_inputs(args)
    repo = args.repo.resolve()
    python = args.python.resolve()
    dataset = args.dataset.resolve()
    model_path = args.model_path.resolve()
    compat = args.pytorch3d_compat.resolve()
    command = build_command(
        python=python,
        repo=repo,
        dataset=dataset,
        model_path=model_path,
        mode=args.mode,
        iterations=args.iterations,
    )
    env = dict(os.environ)
    env["PATH"] = str(python.parent) + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(compat), str(repo), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONHASHSEED"] = "0"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["WANDB_MODE"] = "offline"
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(name, None)

    print("RUN", json.dumps(command, ensure_ascii=False), flush=True)
    subprocess.run(command, cwd=repo, env=env, check=True)
    checkpoint = model_path / "point_cloud" / f"iteration_{args.iterations}"
    for required in (
        model_path / "cfg_args",
        checkpoint / "point_cloud.ply",
        checkpoint / "additional_attributes.npz",
        checkpoint / "checkpoints.pth",
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    payload = {
        "schema": "m3m_gcp_native_quarter_citygs_x_training_run_v1",
        "protocol_id": "m3m_gcp_native_quarter_geometry_v2",
        "method_id": "citygs_x",
        "status": "TRAINING_PASS",
        "mode": args.mode,
        "formal_result": args.mode == "formal",
        "seed": 0,
        "iterations": args.iterations,
        "source": {
            "commit": CITYGS_X_COMMIT,
            "tree": CITYGS_X_TREE,
            "training_runtime_diff": ["utils/camera_utils.py"],
            "camera_utils_sha256": verified["camera_utils_sha256"],
        },
        "input": verified,
        "model_path": str(model_path),
        "checkpoint": {
            "path": str(checkpoint),
            "point_cloud_sha256": sha256(checkpoint / "point_cloud.ply"),
        },
        "command": command,
        "route": {
            "single_gpu": True,
            "bsz": 1,
            "resolution": 1,
            "images": "images",
            "single_view_weight_from_iter": 10_000 if args.mode == "formal" else 0,
            "dpt_loss_from_iter": 10_000 if args.mode == "formal" else 0,
            "multi_view_weight_from_iter": 30_000 if args.mode == "formal" else 0,
            "dpt_end_iter": 30_000 if args.mode == "formal" else args.iterations,
            "depth_l1_weight_init": 0.5,
            "depth_l1_weight_final": 0.01,
            "default_voxel_size": 0.001,
            "multi_view_patch_size": 3,
        },
    }
    (model_path / "training_wrapper_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
