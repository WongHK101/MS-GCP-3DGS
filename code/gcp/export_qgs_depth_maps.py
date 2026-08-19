#!/usr/bin/env python3
"""Export protocol-v2 metric-depth packets from a frozen QGS checkpoint."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_gaussian_depth_maps import export_depths  # noqa: E402
from metric_depth_packet import (  # noqa: E402
    DEFAULT_NORMALIZATION_EPSILON,
    DEFAULT_NUMERICAL_SUPPORT_FLOOR,
    DEFAULT_VARIANCE_CLAMP_TOLERANCE,
)


def load_config(qgs_repo: Path, config_path: Path) -> Any:
    from omegaconf import OmegaConf

    config = OmegaConf.merge(
        OmegaConf.load(qgs_repo / "config/base.yaml"),
        OmegaConf.load(config_path),
    )
    OmegaConf.resolve(config)
    config.model_path = str(config.model_path).replace(" ", "").replace("\n", "")
    return config


def load_qgs_runtime(qgs_repo: Path, config: Any) -> Dict[str, Any]:
    if not qgs_repo.is_dir():
        raise FileNotFoundError(f"QGS repository not found: {qgs_repo}")
    sys.path.insert(0, str(qgs_repo))

    import torch
    from gaussian_renderer import render as qgs_render
    from scene import Scene as QGSScene
    from scene.gaussian_model import GaussianModel as QGSGaussianModel
    from utils.general_utils import safe_state

    def gaussian_model(_sh_degree: int) -> Any:
        return QGSGaussianModel(config.gs_model)

    def scene(_dataset: Any, gaussians: Any, load_iteration: int, shuffle: bool) -> Any:
        return QGSScene(config, gaussians, load_iteration=load_iteration, shuffle=shuffle)

    def render(
        view: Any,
        gaussians: Any,
        pipeline: Any,
        background: Any,
        return_raw_metric_depth_accumulators: bool = False,
    ) -> Dict[str, Any]:
        return call_qgs_render(
            qgs_render,
            view,
            gaussians,
            pipeline,
            config.optimizer,
            background,
            kernel_size=config.dataset.kernel_size,
            return_raw_metric_depth_accumulators=return_raw_metric_depth_accumulators,
        )

    return {
        "train_repo": qgs_repo,
        "torch": torch,
        "GaussianModel": gaussian_model,
        "Scene": scene,
        "render": render,
        "safe_state": safe_state,
        "sparse_adam_available": False,
        "render_parameters": set(inspect.signature(render).parameters),
    }


def call_qgs_render(
    qgs_render: Any,
    view: Any,
    gaussians: Any,
    pipeline: Any,
    optimizer: Any,
    background: Any,
    *,
    kernel_size: float,
    return_raw_metric_depth_accumulators: bool = False,
) -> Dict[str, Any]:
    """Call either the official RGB renderer or the additive depth-patched renderer.

    The official frozen QGS source has no raw-accumulator keyword.  RGB export
    must therefore omit that depth-only extension.  A geometry caller that
    actually requests raw accumulators still fails closed unless the evaluated
    renderer explicitly exposes the extension.
    """

    parameters = set(inspect.signature(qgs_render).parameters)
    kwargs: Dict[str, Any] = {
        "kernel_size": kernel_size,
        "return_depth": True,
        "return_normal": True,
    }
    raw_keyword = "return_raw_metric_depth_accumulators"
    if raw_keyword in parameters:
        kwargs[raw_keyword] = bool(return_raw_metric_depth_accumulators)
    elif return_raw_metric_depth_accumulators:
        raise RuntimeError(
            "QGS renderer does not expose the required raw metric-depth accumulators"
        )
    return qgs_render(
        view,
        gaussians,
        pipeline,
        optimizer,
        background,
        **kwargs,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_repo", required=True, help="Evaluation-only patched QGS repository.")
    parser.add_argument("--qgs_config_path", required=True, help="Resolved or mergeable QGS training configuration.")
    parser.add_argument(
        "--camera_source_path",
        default="",
        help=(
            "Optional read-only COLMAP scene used only to construct evaluation cameras. "
            "The checkpoint and all training settings still come from qgs_config_path."
        ),
    )
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--camera_sets", default="all", choices=["all", "train", "test"])
    parser.add_argument("--depth_output_dir", required=True)
    parser.add_argument("--manifest_path", default="")
    parser.add_argument("--mapping_csv", default="")
    parser.add_argument("--numerical_support_floor", type=float, default=DEFAULT_NUMERICAL_SUPPORT_FLOOR)
    parser.add_argument("--normalization_epsilon", type=float, default=DEFAULT_NORMALIZATION_EPSILON)
    parser.add_argument("--variance_clamp_tolerance", type=float, default=DEFAULT_VARIANCE_CLAMP_TOLERANCE)
    parser.add_argument("--image_list_csv", default="")
    parser.add_argument("--image_name_column", default="image_name")
    parser.add_argument("--image_list_status_column", default="")
    parser.add_argument("--image_list_status_values", default="")
    parser.add_argument("--image_domain", default="rendered_colmap_camera_domain")
    parser.add_argument(
        "--pixel_coordinate_convention",
        default="zero_indexed_pixel_centers",
        choices=["zero_indexed_pixel_centers", "zero_based_pixel_centers"],
    )
    parser.add_argument("--protocol_id", default="")
    parser.add_argument("--protocol_scene", default="")
    parser.add_argument("--source_data_release_root_digest_sha256", default="")
    parser.add_argument("--camera_z_unit_contract", default="")
    parser.add_argument("--adapter_conformance_status", default="")
    parser.add_argument("--adapter_conformance_report", default="")
    parser.add_argument("--adapter_conformance_report_sha256", default="")
    parser.add_argument("--renderer_adapter_patch", default="")
    parser.add_argument("--rasterizer_adapter_patch", default="")
    parser.add_argument("--rasterizer_repo", default="submodules/diff-quadratic-rasterization")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    qgs_repo = Path(args.train_repo).expanduser().resolve()
    config_path = Path(args.qgs_config_path).expanduser().resolve()
    config = load_config(qgs_repo, config_path)
    if args.camera_source_path:
        camera_source = Path(args.camera_source_path).expanduser().resolve()
        if not (camera_source / "sparse" / "0").is_dir():
            raise FileNotFoundError(f"COLMAP evaluation camera model not found: {camera_source / 'sparse' / '0'}")
        config.root_dir = str(camera_source)
        config.gs_model.root_dir = str(camera_source)
    runtime = load_qgs_runtime(qgs_repo, config)
    runtime["safe_state"](args.quiet)

    dataset = SimpleNamespace(
        sh_degree=int(config.gs_model.sh_degree),
        white_background=bool(config.dataset.white_background),
        model_path=str(config.model_path),
        source_path=str(config.root_dir),
        train_test_exp=False,
    )
    old_cwd = Path.cwd()
    os.chdir(qgs_repo)
    try:
        manifest = export_depths(args, dataset, config.pipeline, runtime)
    finally:
        os.chdir(old_cwd)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
