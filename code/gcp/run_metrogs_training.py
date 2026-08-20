#!/usr/bin/env python3
"""Run the frozen single-GPU MetroGS native-quarter training route."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml


METROGS_COMMIT = "8cf9ac13c0c34b65c1a935d181c4634909e60f3f"
METROGS_TREE = "7e92b13095cf4a031d7eb8593e10616db154abbf"
OFFICIAL_CONFIG_RELATIVE = Path("configs/metrogs/train/MatrixCity-Aerial.yaml")
OFFICIAL_CONFIG_SHA256 = "86e8cbc1e44f177c80e69151f06c08782cab1b50ec145e4e6fdd44797ef814e3"
MAIN_BSZ_SHA256 = "3bb5fb6de6d62cd9bb74fcfc4b8fcfd85afc839b39542dd79e702402963edbf1"
MERGE_SCRIPT_SHA256 = "90a5d7b56e605cecb2238514449d53deb4dc097be4eed84e9f897b847e7a6075"
CKPT2PLY_SHA256 = "fe3a6f540cba658696d9c2e64d13ad76deb8df1401f81820bf79ddedd23231e9"
FORMAL_MANIFEST_FILE_SHA256 = "ae29817198f54f04e4133a7b5fd03df679dd6f259b2d1ef4125e825cbb8e422e"
FORMAL_MANIFEST_CANONICAL_SHA256 = "4ae07aad9278e2eb5af2f04268f3301df56c6f6ada9ee51c6f125fdbb29e7ec8"
MOGE_WEIGHT_SHA256 = "280741fd09bc3f403ccff9967784c2a391b52d2c0742ae3efdb21d9f90cc1a01"
PI3_WEIGHT_SHA256 = "33580e4702ac671558aedeab1148fd08118f7ce45bdbeb99f3e3cf340062875d"
SPARSE_SHA256 = {
    "cameras.bin": "a627e4ecd29ea1afe44937b56719d0cb5f3f4d20b8b368542db64a395306567f",
    "images.bin": "c0ce229da0adbe69f4796d749ed071e1cb5a87d50774c81d22dc1c369590199d",
    "points3D.bin": "e0f7bb4d9e39ad433fb9778c1bd86bc76bb0fc303b5389791a448f743f1cb955",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True
    ).rstrip("\r\n")


def validate_budget(mode: str, iterations: int) -> None:
    expected = 8 if mode == "qualification" else 150_000
    if mode not in {"qualification", "formal"}:
        raise ValueError(f"unsupported mode: {mode}")
    if iterations != expected:
        raise ValueError(
            f"{mode} MetroGS route is frozen to {expected} effective iterations"
        )


def require_official_semantics(config: dict[str, Any]) -> None:
    metric = config["model"]["metric"]["init_args"]
    renderer = config["model"]["renderer"]["init_args"]
    density = config["model"]["density"]["init_args"]
    parser = config["data"]["parser"]["init_args"]
    checks = {
        "model.gaussian.class_path": config["model"]["gaussian"]["class_path"]
        == "internal.models.metrogs.Gaussian2D",
        "model.metric.class_path": config["model"]["metric"]["class_path"]
        == "internal.metrics.metrogs_metrics.DistributedMetrics",
        "model.renderer.class_path": config["model"]["renderer"]["class_path"]
        == "internal.renderers.metrogs_renderer.DistributedRenderer",
        "metric.single_view_from": metric["single_view_from"] == 0,
        "metric.multi_view_from": metric["multi_view_from"] == 50_000,
        "metric.depth_loss": metric["depth_loss_type"] == "l1+ssim",
        "metric.depth_weight_init": metric["depth_loss_weight"]["init"] == 0.5,
        "metric.depth_weight_final": metric["depth_loss_weight"]["final_factor"]
        == 0.005,
        "renderer.use_app": renderer["use_app"] is True,
        "renderer.matrixcity_aabb": renderer["aabb"]
        == [-12, -8, -1, 11, 9, 6],
        "density.voxel_size": density["voxel_size"] == 0.1,
        "density.densify_until": density["densify_until_iter"] == 50_000,
        "trainer.strategy": config["trainer"]["strategy"]["class_path"]
        == "internal.mp_strategy.MPStrategy",
        "trainer.devices": config["trainer"]["devices"] == [0, 1, 2, 3],
        "trainer.max_steps": config["trainer"]["max_steps"] == 150_000,
        "data.use_multi_view": config["data"]["use_multi_view"] is True,
        "data.batch_size": config["data"]["batch_size"] == 4,
        "data.parser": config["data"]["parser"]["class_path"]
        == "internal.dataparsers.estimated_mask_depth_colmap_dataparser.EstimatedDepthColmap",
        "data.downsample": parser["down_sample_factor"] == 1.2,
        "save_iterations": config["save_iterations"] == [150_000],
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"official MetroGS MatrixCity-Aerial semantics changed: {failed}")


def build_resolved_config(
    *,
    official: dict[str, Any],
    dataset: Path,
    additional_ply: Path,
    model_path: Path,
    mode: str,
    iterations: int,
) -> dict[str, Any]:
    validate_budget(mode, iterations)
    require_official_semantics(official)
    config = copy.deepcopy(official)

    # Common benchmark adaptations: seed 0, one available GPU, and the already
    # materialized native-quarter camera/image domain. Batch size and all method
    # optimization semantics remain at the official MatrixCity-Aerial values.
    config["seed_everything"] = 0
    config["trainer"]["devices"] = [0]
    config["trainer"]["max_steps"] = iterations
    config["data"]["path"] = str(dataset)
    config["data"]["batch_size"] = 4
    config["data"]["use_multi_view"] = True
    parser = config["data"]["parser"]["init_args"]
    parser["additional_ply_path"] = str(additional_ply)
    parser["down_sample_factor"] = 1.0

    # The published numeric AABB is specific to MatrixCity. MetroGS already
    # supplies a robust MAD estimator when this value is absent; use that
    # upstream path for the custom M3M scene rather than importing foreign
    # scene coordinates.
    config["model"]["renderer"]["init_args"]["aabb"] = None
    config["save_iterations"] = [iterations]
    config["save_val"] = False
    config["output"] = str(model_path.parent)
    config["name"] = model_path.name
    # MetroGS' frozen upstream training loop unconditionally emits the
    # Gaussian-count/learning-rate metrics through ``self.logger`` on the
    # first optimizer step.  Use Lightning's local TensorBoard logger so the
    # official loop remains untouched and no network service is required.
    config["logger"] = "tensorboard"

    # Qualification covers one official single-view depth step and one
    # multi-view geometry step (effective iterations 0 and 4 at batch size 4).
    if mode == "qualification":
        config["model"]["metric"]["init_args"]["multi_view_from"] = 4
    return config


def build_commands(
    *, python: Path, repo: Path, model_path: Path, resolved_config: Path
) -> tuple[list[str], list[str], list[str]]:
    train = [
        str(python),
        "-B",
        str(repo / "main_bsz.py"),
        "fit",
        "--config",
        str(resolved_config),
    ]
    merge = [
        str(python),
        "-B",
        str(repo / "utils" / "merge_distributed_ckpts.py"),
        str(model_path),
    ]
    convert = [
        str(python),
        "-B",
        str(repo / "utils" / "ckpt2ply.py"),
        str(model_path),
    ]
    return train, merge, convert


def build_subprocess_envs(*, python: Path, repo: Path) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Build isolated environments for train, merge, and checkpoint conversion.

    PyTorch 2.6's legacy checkpoint opt-out is required only by the frozen
    upstream ``ckpt2ply.py`` conversion utility.  It must never leak into the
    fresh training or checkpoint-merge subprocesses, even when inherited from
    the wrapper's parent environment.
    """
    base = dict(os.environ)
    base["PATH"] = str(python.parent) + os.pathsep + base.get("PATH", "")
    base["PYTHONPATH"] = str(repo) + os.pathsep + base.get("PYTHONPATH", "")
    base["PYTHONUNBUFFERED"] = "1"
    base["PYTHONDONTWRITEBYTECODE"] = "1"
    base["PYTHONHASHSEED"] = "0"
    base["CUDA_VISIBLE_DEVICES"] = "0"
    base["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    base["NCCL_SHM_DISABLE"] = "1"
    base["WANDB_MODE"] = "offline"
    base.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        base.pop(name, None)

    train_env = dict(base)
    merge_env = dict(base)
    convert_env = dict(base)
    # The frozen converter reads the self-produced Lightning checkpoint and
    # requires the pre-2.6 torch.load behavior.  This trust exception is
    # deliberately confined to this one post-training subprocess.
    convert_env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    return train_env, merge_env, convert_env


def verify_inputs(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    # Preserve the virtual-environment launcher instead of resolving its
    # symlink to the system interpreter and losing the frozen packages.
    python = Path(os.path.abspath(os.fspath(args.python)))
    dataset = args.dataset.resolve()
    model_path = args.model_path.resolve()
    prior_path = args.prior_manifest.resolve()
    pass_marker = args.prior_pass_marker.resolve()
    additional_ply = args.additional_ply.resolve()
    official_config = repo / OFFICIAL_CONFIG_RELATIVE

    if model_path.exists():
        raise FileExistsError(f"model path already exists: {model_path}")
    for required in (
        repo / ".git",
        repo / "main_bsz.py",
        repo / "utils" / "merge_distributed_ckpts.py",
        official_config,
        python,
        dataset / "images",
        dataset / "sparse" / "0" / "cameras.bin",
        dataset / "sparse" / "0" / "images.bin",
        dataset / "sparse" / "0" / "points3D.bin",
        dataset / "estimated_mask_depths",
        dataset / "estimated_mask_depth_scales.json",
        dataset / "multi_view.json",
        dataset / "NATIVE_QUARTER_INPUT_MANIFEST.json",
        additional_ply,
        prior_path,
        pass_marker,
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    if git_output(repo, "rev-parse", "HEAD") != METROGS_COMMIT:
        raise RuntimeError("MetroGS repository commit mismatch")
    if git_output(repo, "rev-parse", "HEAD^{tree}") != METROGS_TREE:
        raise RuntimeError("MetroGS repository tree mismatch")
    if git_output(repo, "status", "--porcelain=v1"):
        raise RuntimeError("MetroGS formal training runtime is not clean")
    source_hashes = {
        "official_config": sha256(official_config),
        "main_bsz.py": sha256(repo / "main_bsz.py"),
        "merge_distributed_ckpts.py": sha256(
            repo / "utils" / "merge_distributed_ckpts.py"
        ),
        "ckpt2ply.py": sha256(repo / "utils" / "ckpt2ply.py"),
    }
    expected_source_hashes = {
        "official_config": OFFICIAL_CONFIG_SHA256,
        "main_bsz.py": MAIN_BSZ_SHA256,
        "merge_distributed_ckpts.py": MERGE_SCRIPT_SHA256,
        "ckpt2ply.py": CKPT2PLY_SHA256,
    }
    if source_hashes != expected_source_hashes:
        raise RuntimeError(
            f"MetroGS frozen source-file identity mismatch: {source_hashes}"
        )

    formal_path = dataset / "NATIVE_QUARTER_INPUT_MANIFEST.json"
    if sha256(formal_path) != FORMAL_MANIFEST_FILE_SHA256:
        raise RuntimeError("formal input manifest file identity mismatch")
    formal = json.loads(formal_path.read_text(encoding="utf-8"))
    if formal.get("manifest_sha256") != FORMAL_MANIFEST_CANONICAL_SHA256:
        raise RuntimeError("formal input manifest canonical identity mismatch")
    if formal.get("scene") != "gcp_3000_20260602":
        raise RuntimeError("MetroGS route is frozen to the 3K scene")
    if int(formal.get("train_view_count", -1)) != 82:
        raise RuntimeError("formal train-view count mismatch")
    if int(formal.get("test_view_count", -1)) != 12:
        raise RuntimeError("formal heldout-view count mismatch")

    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    if prior.get("status") != "PASS" or prior.get("passed") is not True:
        raise RuntimeError("MetroGS training-prior evidence is not PASS")
    if prior.get("method_id") != "metrogs":
        raise RuntimeError("unexpected MetroGS prior method identity")
    if prior.get("protocol_id") != "m3m_gcp_native_quarter_geometry_v2":
        raise RuntimeError("unexpected MetroGS prior protocol identity")
    if prior.get("scene") != "gcp_3000_20260602":
        raise RuntimeError("unexpected MetroGS prior scene identity")
    source = prior.get("source", {})
    if source.get("repository_commit") != METROGS_COMMIT:
        raise RuntimeError("MetroGS prior commit mismatch")
    if source.get("repository_tree") != METROGS_TREE:
        raise RuntimeError("MetroGS prior tree mismatch")
    prior_input = prior.get("input", {})
    if Path(prior_input.get("dataset", "")).resolve() != dataset:
        raise RuntimeError("MetroGS prior dataset differs from training dataset")
    if prior_input.get("formal_input_manifest_sha256") != FORMAL_MANIFEST_FILE_SHA256:
        raise RuntimeError("MetroGS prior formal-manifest identity mismatch")
    if prior_input.get("train_view_count") != 82:
        raise RuntimeError("MetroGS prior train-view count mismatch")
    if prior_input.get("heldout_view_count") != 12:
        raise RuntimeError("MetroGS prior heldout-view count mismatch")
    if not str(prior_input.get("image_transform", "")).startswith("none;"):
        raise RuntimeError("MetroGS prior did not preserve native-quarter pixels")
    if prior_input.get("sparse_hashes") != SPARSE_SHA256:
        raise RuntimeError("MetroGS track-closed sparse identity mismatch")
    if prior.get("moge", {}).get("weight_sha256") != MOGE_WEIGHT_SHA256:
        raise RuntimeError("MetroGS MoGe weight identity mismatch")
    moge = prior.get("moge", {})
    if moge.get("depth_count") != 82 or moge.get("scale_count") != 82:
        raise RuntimeError("MetroGS MoGe inventory must cover all 82 RGB training views")
    survivor_count = int(moge.get("official_scale_bound_survivor_count", -1))
    rejected_count = int(moge.get("official_scale_bound_rejected_count", -1))
    rejected_images = moge.get("official_scale_bound_rejected_images")
    if survivor_count <= 0 or survivor_count + rejected_count != 82:
        raise RuntimeError("MetroGS official depth-prior filter accounting mismatch")
    if not isinstance(rejected_images, list) or len(rejected_images) != rejected_count:
        raise RuntimeError("MetroGS rejected depth-prior inventory mismatch")
    if prior.get("pi3", {}).get("weight_sha256") != PI3_WEIGHT_SHA256:
        raise RuntimeError("MetroGS Pi3 weight identity mismatch")
    merged = prior.get("pi3", {}).get("merged_pointmap", {})
    if Path(merged.get("path", "")).resolve() != additional_ply:
        raise RuntimeError("MetroGS additional pointmap path mismatch")
    actual_additional_hash = sha256(additional_ply)
    if merged.get("sha256") != actual_additional_hash:
        raise RuntimeError("MetroGS additional pointmap content mismatch")
    if int(merged.get("vertex_count", 0)) <= 0:
        raise RuntimeError("MetroGS additional pointmap is empty")
    claims = prior.get("claims", {})
    expected_claims = {
        "heldout_rgb_read": False,
        "gcp_truth_read": False,
        "lidar_read": False,
        "image_pixels_changed": False,
        "rgb_training_views_removed_by_depth_scale_filter": False,
        "formal_training_started": False,
    }
    for name, expected in expected_claims.items():
        if claims.get(name) is not expected:
            raise RuntimeError(f"MetroGS prior access claim mismatch for {name}")

    image_count = sum(
        1
        for path in (dataset / "images").rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    depth_count = sum(
        1 for path in (dataset / "estimated_mask_depths").glob("*.npy") if path.is_file()
    )
    if image_count != 82 or depth_count != 82:
        raise RuntimeError(
            f"MetroGS training-only inventory mismatch: images={image_count}, depths={depth_count}"
        )
    actual_sparse = {
        name: sha256(dataset / "sparse" / "0" / name) for name in SPARSE_SHA256
    }
    if actual_sparse != SPARSE_SHA256:
        raise RuntimeError("MetroGS sparse files changed after prior qualification")

    official = yaml.safe_load(official_config.read_text(encoding="utf-8"))
    require_official_semantics(official)
    return {
        "prior_manifest": str(prior_path),
        "prior_manifest_sha256": sha256(prior_path),
        "prior_pass_marker": str(pass_marker),
        "formal_input_manifest": str(formal_path),
        "formal_input_manifest_file_sha256": FORMAL_MANIFEST_FILE_SHA256,
        "formal_input_manifest_canonical_sha256": FORMAL_MANIFEST_CANONICAL_SHA256,
        "training_image_count": image_count,
        "training_depth_count": depth_count,
        "training_depth_prior_attached_count": survivor_count,
        "training_depth_prior_skipped_count": rejected_count,
        "training_depth_prior_skipped_images": rejected_images,
        "sparse_sha256": actual_sparse,
        "additional_pointmap": {
            "path": str(additional_ply),
            "bytes": additional_ply.stat().st_size,
            "sha256": actual_additional_hash,
            "vertex_count": int(merged["vertex_count"]),
        },
        "source_file_sha256": source_hashes,
        "official_config": official,
        "access_boundary": expected_claims,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--prior_manifest", type=Path, required=True)
    parser.add_argument("--prior_pass_marker", type=Path, required=True)
    parser.add_argument("--additional_ply", type=Path, required=True)
    parser.add_argument("--mode", choices=("qualification", "formal"), required=True)
    parser.add_argument("--iterations", type=int, required=True)
    args = parser.parse_args()

    validate_budget(args.mode, args.iterations)
    verified = verify_inputs(args)
    repo = args.repo.resolve()
    python = Path(os.path.abspath(os.fspath(args.python)))
    dataset = args.dataset.resolve()
    model_path = args.model_path.resolve()
    additional_ply = args.additional_ply.resolve()
    resolved_config_path = model_path.parent / "metrogs_frozen_training_config.yaml"
    if resolved_config_path.exists():
        raise FileExistsError(f"resolved config already exists: {resolved_config_path}")
    resolved = build_resolved_config(
        official=verified.pop("official_config"),
        dataset=dataset,
        additional_ply=additional_ply,
        model_path=model_path,
        mode=args.mode,
        iterations=args.iterations,
    )
    resolved_config_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_config_path.write_text(
        yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    resolved_config_sha256 = sha256(resolved_config_path)
    train_command, merge_command, convert_command = build_commands(
        python=python,
        repo=repo,
        model_path=model_path,
        resolved_config=resolved_config_path,
    )

    train_env, merge_env, convert_env = build_subprocess_envs(
        python=python, repo=repo
    )

    print("RUN", json.dumps(train_command, ensure_ascii=False), flush=True)
    subprocess.run(train_command, cwd=repo, env=train_env, check=True)
    print("RUN", json.dumps(merge_command, ensure_ascii=False), flush=True)
    subprocess.run(merge_command, cwd=repo, env=merge_env, check=True)
    print("RUN", json.dumps(convert_command, ensure_ascii=False), flush=True)
    subprocess.run(convert_command, cwd=repo, env=convert_env, check=True)

    checkpoint_dir = model_path / "checkpoints"
    point_cloud = model_path / "point_cloud" / f"iteration_{args.iterations}" / "point_cloud.ply"
    rank_checkpoints = sorted(
        checkpoint_dir.glob(f"*-step={args.iterations}-rank=0.ckpt")
    )
    merged_checkpoints = sorted(
        path
        for path in checkpoint_dir.glob(f"*-step={args.iterations}.ckpt")
        if "-rank=" not in path.name
    )
    for required in (
        model_path / "cfg_args",
        model_path / "cameras.json",
        model_path / "input.ply",
        point_cloud,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    if len(rank_checkpoints) != 1 or len(merged_checkpoints) != 1:
        raise RuntimeError(
            "MetroGS checkpoint inventory mismatch: "
            f"rank={rank_checkpoints}, merged={merged_checkpoints}"
        )
    cameras = json.loads((model_path / "cameras.json").read_text(encoding="utf-8"))
    if len(cameras) != 82:
        raise RuntimeError(f"MetroGS output camera count mismatch: {len(cameras)}")

    payload = {
        "schema": "m3m_gcp_native_quarter_metrogs_training_run_v1",
        "protocol_id": "m3m_gcp_native_quarter_geometry_v2",
        "method_id": "metrogs",
        "scene": "gcp_3000_20260602",
        "status": "TRAINING_PASS",
        "mode": args.mode,
        "formal_result": args.mode == "formal",
        "seed": 0,
        "effective_iterations": args.iterations,
        "optimizer_steps": math.ceil(args.iterations / 4),
        "source": {
            "commit": METROGS_COMMIT,
            "tree": METROGS_TREE,
            "runtime_diff": [],
            "official_recipe": "MatrixCity-Aerial",
            "official_config_sha256": OFFICIAL_CONFIG_SHA256,
            "source_file_sha256": verified["source_file_sha256"],
        },
        "input": verified,
        "model_path": str(model_path),
        "resolved_config": {
            "path": str(resolved_config_path),
            "sha256": resolved_config_sha256,
        },
        "checkpoint": {
            "rank_path": str(rank_checkpoints[0]),
            "rank_sha256": sha256(rank_checkpoints[0]),
            "merged_path": str(merged_checkpoints[0]),
            "merged_sha256": sha256(merged_checkpoints[0]),
            "point_cloud_path": str(point_cloud),
            "point_cloud_sha256": sha256(point_cloud),
        },
        "commands": {
            "train": train_command,
            "merge": merge_command,
            "convert_checkpoint_to_ply": convert_command,
        },
        "route": {
            "single_gpu": True,
            "batch_size": 4,
            "native_quarter_down_sample_factor": 1.0,
            "use_multi_view": True,
            "single_view_from": 0,
            "multi_view_from": 50_000 if args.mode == "formal" else 4,
            "appearance_model": True,
            "aabb": "upstream MAD estimator for custom scene",
            "external_geometry_priors": ["MoGe-2", "Pi3-Align"],
        },
    }
    summary_path = model_path / "training_wrapper_summary.json"
    summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
