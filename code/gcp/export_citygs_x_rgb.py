#!/usr/bin/env python3
"""Render frozen CityGS-X heldout RGB views without method-specific metrics."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_citygs_x_depth_maps import (  # noqa: E402
    build_frozen_cameras,
    load_citygs_x_runtime,
    load_model,
    resolve_sparse_model,
)
from rgb_quality_contract import (  # noqa: E402
    RgbRenderWriter,
    git_identity,
    sha256_file,
)


def export(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
    camera_root = args.camera_root.expanduser().resolve()
    sparse_model = resolve_sparse_model(camera_root)
    writer = RgbRenderWriter(
        contract_path=args.rgb_contract,
        input_manifest_path=args.input_manifest,
        scene=args.scene,
        method_id="citygs_x",
        output_dir=args.output_dir,
        manifest_path=args.manifest_path,
    )
    runtime_log = writer.manifest_path.parent / "citygs_x_rgb_runtime.log"
    old_cwd = Path.cwd()
    os.chdir(repo)
    try:
        with runtime_log.open("w", encoding="utf-8", buffering=1) as log:
            runtime = load_citygs_x_runtime(
                repo,
                model_path,
                args.iteration,
                log,
                args.pytorch3d_compat,
            )
            torch = runtime["torch"]
            if not torch.cuda.is_available():
                raise RuntimeError("formal CityGS-X RGB export requires CUDA")
            cameras = build_frozen_cameras(
                runtime, sparse_model, camera_root, writer.expected_names
            )
            gaussians = load_model(runtime, cameras, args.iteration)
            pipeline = runtime["pipeline_group"].extract(runtime["args"])
            background = torch.zeros(3, dtype=torch.float32, device="cuda")
            strategy_dataset = SimpleNamespace(
                cameras=[camera for _name, camera, _image_id in cameras]
            )
            strategy_history = runtime["DivisionStrategyHistoryFinal"](
                strategy_dataset,
                runtime["utils"].DEFAULT_GROUP.size(),
                runtime["utils"].DEFAULT_GROUP.rank(),
            )
            with torch.no_grad():
                for image_name, camera, source_image_id in cameras:
                    batched_cameras = [camera]
                    strategies, _tasks = runtime["start_strategy_final"](
                        batched_cameras, strategy_history
                    )
                    gaussians.set_anchor_mask(camera.camera_center, args.iteration, 1)
                    visible_mask = runtime["prefilter_voxel"](
                        camera, gaussians, pipeline, background
                    )
                    screenspace = runtime["preprocess"](
                        batched_cameras,
                        gaussians,
                        pipeline,
                        background,
                        batched_voxel_mask=[visible_mask],
                        batched_nearest_cameras=[None],
                        batched_nearest_voxel_mask=[None],
                        batched_strategies=strategies,
                        mode="test",
                        return_plane=True,
                        iterations=args.iteration,
                    )
                    rendered = runtime["render_final"](
                        batched_cameras, screenspace, strategies
                    )
                    rgb = rendered[0][0]
                    if rgb is None:
                        raise RuntimeError(f"CityGS-X returned no RGB for {image_name}")
                    writer.save(
                        image_name,
                        rgb,
                        camera_record={
                            "source_colmap_image_id": int(source_image_id),
                            "appearance_policy": args.appearance_policy,
                            "appearance_dim": int(runtime["args"].appearance_dim),
                        },
                    )
    finally:
        os.chdir(old_cwd)

    checkpoint_dir = model_path / "point_cloud" / f"iteration_{args.iteration}"
    checkpoint_files = [
        checkpoint_dir / "point_cloud_rk0_ws1.ply",
        checkpoint_dir / "additional_attributes.npz",
        checkpoint_dir / "checkpoints.pth",
    ]
    for path in checkpoint_files:
        if not path.is_file():
            raise FileNotFoundError(path)
    renderer_source = repo / "gaussian_renderer" / "__init__.py"
    provenance = {
        "adapter_kind": "citygs_x_rgb_v1",
        "adapter_path": str(Path(__file__).resolve()),
        "adapter_sha256": sha256_file(Path(__file__).resolve()),
        "renderer_repository": git_identity(repo),
        "renderer_source_path": str(renderer_source),
        "renderer_source_sha256": sha256_file(renderer_source),
        "model_root": str(model_path),
        "formal_model_files_sha256": {
            path.name: sha256_file(path) for path in checkpoint_files
        },
        "cfg_args_sha256": sha256_file(model_path / "cfg_args"),
        "camera_source_root": str(camera_root),
        "frozen_sparse_model": str(sparse_model),
        "frozen_sparse_model_sha256": {
            name: sha256_file(sparse_model / name)
            for name in ("cameras.bin", "images.bin", "points3D.bin")
        },
        "iteration": int(args.iteration),
        "white_background": False,
        "appearance_policy": args.appearance_policy,
        "appearance_dim": 0,
        "heldout_rgb_used_by_adapter": False,
        "test_time_optimization": False,
        "runtime": {
            "python": sys.version,
            "torch": str(runtime["torch"].__version__),
            "torch_cuda": str(runtime["torch"].version.cuda),
            "device_name": runtime["torch"].cuda.get_device_name(0),
        },
    }
    return writer.finalize(provenance=provenance)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--pytorch3d_compat", type=Path, required=True)
    parser.add_argument("--camera_root", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--rgb_contract", type=Path, required=True)
    parser.add_argument("--input_manifest", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--manifest_path", type=Path)
    parser.add_argument("--appearance_policy", default="appearance_dim_0")
    return parser


def main() -> int:
    manifest = export(build_parser().parse_args())
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
