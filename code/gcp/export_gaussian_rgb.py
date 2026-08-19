#!/usr/bin/env python3
"""Render frozen heldout RGB views for Graphdeco-style Gaussian methods."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_gaussian_depth_maps import (  # noqa: E402
    collect_views,
    load_gaussian_runtime,
    parse_train_repo,
)
from rgb_quality_contract import (  # noqa: E402
    RgbRenderWriter,
    git_identity,
    sha256_file,
)


def export_rgb(
    args: argparse.Namespace,
    dataset: Any,
    pipeline: Any,
    runtime: Dict[str, Any],
) -> dict[str, Any]:
    train_repo = Path(runtime["train_repo"]).expanduser().resolve()
    torch = runtime["torch"]
    writer = RgbRenderWriter(
        contract_path=Path(args.rgb_contract),
        input_manifest_path=Path(args.input_manifest),
        scene=args.scene,
        method_id=args.method_id,
        output_dir=Path(args.output_dir),
        manifest_path=Path(args.manifest_path) if args.manifest_path else None,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("formal Gaussian RGB export requires CUDA")

    GaussianModel = runtime["GaussianModel"]
    Scene = runtime["Scene"]
    render = runtime["render"]
    render_parameters = set(inspect.signature(render).parameters)
    splat_args = None
    splatting_config_path = None
    if getattr(args, "splatting_config_path", ""):
        splatting_config_path = Path(args.splatting_config_path).expanduser().resolve()
        if not splatting_config_path.is_file():
            raise FileNotFoundError(splatting_config_path)
        if "splat_args" not in render_parameters:
            raise RuntimeError("renderer does not accept the frozen splatting configuration")
        rasterizer = importlib.import_module("diff_gaussian_rasterization")
        splat_args = rasterizer.ExtendedSettings.from_json(str(splatting_config_path))
    old_cwd = Path.cwd()
    os.chdir(train_repo)
    try:
        with torch.no_grad():
            gaussians = GaussianModel(dataset.sh_degree)
            scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
            background = torch.tensor(
                [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
                dtype=torch.float32,
                device="cuda",
            )
            views = collect_views(
                scene,
                args.camera_sets,
                allowlist=writer.allowlist_map(),
            )
            upstream_decoded_heldout_rgb_count = 0
            for split, view, image_name in views:
                # Graphdeco-family Scene loaders construct Camera objects by
                # decoding image bytes even though official render() functions
                # do not consume them.  Remove the tensor before the renderer
                # boundary so a future source drift fails instead of silently
                # gaining access to heldout truth.
                if getattr(view, "original_image", None) is not None:
                    upstream_decoded_heldout_rgb_count += 1
                    view.original_image = None
                if getattr(view, "original_image", None) is not None:
                    raise RuntimeError(f"failed to detach heldout RGB for {image_name}")
                kwargs: dict[str, Any] = {}
                if "use_trained_exp" in render_parameters:
                    kwargs["use_trained_exp"] = bool(
                        getattr(dataset, "train_test_exp", False)
                    )
                if "separate_sh" in render_parameters:
                    kwargs["separate_sh"] = bool(runtime["sparse_adam_available"])
                if "kernel_size" in render_parameters:
                    if not hasattr(dataset, "kernel_size"):
                        raise RuntimeError("renderer requires kernel_size but the frozen model config has none")
                    kwargs["kernel_size"] = float(dataset.kernel_size)
                if splat_args is not None:
                    kwargs["splat_args"] = splat_args
                payload = render(view, gaussians, pipeline, background, **kwargs)
                if not isinstance(payload, dict) or "render" not in payload:
                    raise RuntimeError(f"renderer did not return RGB for {image_name}")
                rgb = payload["render"]
                if bool(getattr(dataset, "train_test_exp", False)):
                    rgb = rgb[..., rgb.shape[-1] // 2 :]
                writer.save(
                    image_name,
                    rgb,
                    camera_record={
                        "renderer_split": split,
                        "camera_uid": int(getattr(view, "uid", -1)),
                        "appearance_policy": args.appearance_policy,
                    },
                )
    finally:
        os.chdir(old_cwd)

    model_root = Path(dataset.model_path).expanduser().resolve()
    ply = model_root / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply"
    if not ply.is_file():
        raise FileNotFoundError(ply)
    source_root = Path(dataset.source_path).expanduser().resolve()
    cfg_args = model_root / "cfg_args"
    method_config_path = (
        Path(args.qgs_config_path).expanduser().resolve()
        if getattr(args, "qgs_config_path", "")
        else None
    )
    renderer_file = train_repo / "gaussian_renderer" / "__init__.py"
    entrypoint = Path(getattr(args, "adapter_path", __file__)).expanduser().resolve()
    provenance = {
        "adapter_kind": getattr(args, "adapter_kind", "graphdeco_style_gaussian_rgb_v1"),
        "adapter_path": str(entrypoint),
        "adapter_sha256": sha256_file(entrypoint),
        "shared_graphdeco_adapter_path": str(Path(__file__).resolve()),
        "shared_graphdeco_adapter_sha256": sha256_file(Path(__file__).resolve()),
        "renderer_repository": git_identity(train_repo),
        "renderer_source_path": str(renderer_file),
        "renderer_source_sha256": sha256_file(renderer_file),
        "model_root": str(model_root),
        "formal_model_path": str(ply),
        "formal_model_sha256": sha256_file(ply),
        "cfg_args_sha256": sha256_file(cfg_args) if cfg_args.is_file() else None,
        "method_config": (
            {
                "path": str(method_config_path),
                "sha256": sha256_file(method_config_path),
            }
            if method_config_path is not None
            else None
        ),
        "camera_source_root": str(source_root),
        "iteration": int(args.iteration),
        "sh_degree": int(dataset.sh_degree),
        "white_background": bool(dataset.white_background),
        "kernel_size": (
            float(dataset.kernel_size) if hasattr(dataset, "kernel_size") else None
        ),
        "appearance_policy": args.appearance_policy,
        "splatting_config": (
            {
                "path": str(splatting_config_path),
                "sha256": sha256_file(splatting_config_path),
            }
            if splatting_config_path is not None
            else None
        ),
        "upstream_camera_loader_decoded_heldout_rgb_count": int(
            upstream_decoded_heldout_rgb_count
        ),
        "heldout_rgb_detached_before_renderer": True,
        "heldout_rgb_consumed_by_renderer_or_policy": False,
        "heldout_rgb_used_by_adapter": False,
        "test_time_optimization": False,
        "runtime": {
            "python": sys.version,
            "torch": str(torch.__version__),
            "torch_cuda": str(torch.version.cuda),
            "device_name": torch.cuda.get_device_name(0),
        },
    }
    return writer.finalize(provenance=provenance)


def build_parser(runtime: Dict[str, Any]) -> tuple[argparse.ArgumentParser, Any, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    model = runtime["ModelParams"](parser, sentinel=True)
    pipeline = runtime["PipelineParams"](parser)
    parser.add_argument("--train_repo", default=str(runtime["train_repo"]))
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--camera_sets", choices=["all", "train", "test"], default="train")
    parser.add_argument("--rgb_contract", required=True)
    parser.add_argument("--input_manifest", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--method_id", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--manifest_path", default="")
    parser.add_argument("--appearance_policy", required=True)
    parser.add_argument(
        "--splatting_config_path",
        default="",
        help="Optional frozen ExtendedSettings JSON required by renderers such as SOF.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser, model, pipeline


def main() -> int:
    train_repo = parse_train_repo(sys.argv[1:])
    runtime = load_gaussian_runtime(train_repo)
    parser, model_group, pipeline_group = build_parser(runtime)
    args = runtime["get_combined_args"](parser)
    runtime["safe_state"](args.quiet)
    dataset = model_group.extract(args)
    pipeline = pipeline_group.extract(args)
    manifest = export_rgb(args, dataset, pipeline, runtime)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
