#!/usr/bin/env python3
"""Export CityGS-X raw A/M1/M2/H packets on frozen COLMAP cameras."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_citygaussian_v2_depth_maps import (  # noqa: E402
    MAPPING_FIELDS,
    git_status_porcelain,
    read_allowlist,
    resolve_sparse_model,
    save_packet,
    source_trace,
)
from export_gaussian_depth_maps import (  # noqa: E402
    RAW_ACCUMULATOR_TENSOR_NAMES,
    convert_raw_camera_z_units,
    git_tree_hash,
    write_csv,
)
from gcp_packet_camera_compatibility import parse_cfg_args  # noqa: E402
from metric_depth_packet import (  # noqa: E402
    DEFAULT_ALPHA_CUTOFF,
    DEFAULT_EARLY_TERMINATION_THRESHOLD,
    DEFAULT_NORMALIZATION_EPSILON,
    DEFAULT_NUMERICAL_SUPPORT_FLOOR,
    DEFAULT_VARIANCE_CLAMP_TOLERANCE,
    DIAGNOSTIC_VARIANCE_TENSOR,
    DIAGNOSTIC_VARIANCE_VALID_MASK_TENSOR,
    HISTORICAL_INVALID_TENSOR,
    METRIC_PACKET_MANIFEST_SCHEMA,
    METRIC_PACKET_SCHEMA,
    METRIC_PACKET_TENSOR_NAMES,
    PRIMARY_DEPTH_SEMANTICS,
    PRIMARY_DEPTH_TENSOR,
    directory_tree_hash,
    file_sha256,
    git_commit,
    packet_manifest_tensor_formulas,
    variance_validation_manifest_fields,
)


METHOD_ID = "citygs_x"
EXPECTED_RAW_SCALE = 1.0


def build_official_defaults(
    repo: Path,
    pytorch3d_compat: Path | None = None,
) -> tuple[argparse.Namespace, dict[str, Any]]:
    if pytorch3d_compat is not None:
        compat = pytorch3d_compat.expanduser().resolve()
        if not (compat / "pytorch3d" / "transforms" / "__init__.py").is_file():
            raise FileNotFoundError(f"invalid minimal PyTorch3D compatibility root: {compat}")
        sys.path.insert(0, str(compat))
    sys.path.insert(0, str(repo))
    from arguments import (  # noqa: WPS433
        AuxiliaryParams,
        BenchmarkParams,
        DebugParams,
        DistributionParams,
        ModelParams,
        OptimizationParams,
        PipelineParams,
    )

    parser = argparse.ArgumentParser(add_help=False)
    groups = {
        "model": ModelParams(parser),
        "pipeline": PipelineParams(parser),
    }
    AuxiliaryParams(parser)
    OptimizationParams(parser)
    DistributionParams(parser)
    BenchmarkParams(parser)
    DebugParams(parser)
    return parser.parse_args([]), groups


def load_citygs_x_runtime(
    repo: Path,
    model_path: Path,
    iteration: int,
    log_file: Any,
    pytorch3d_compat: Path,
) -> dict[str, Any]:
    repo = repo.expanduser().resolve()
    model_path = model_path.expanduser().resolve()
    if not repo.is_dir():
        raise FileNotFoundError(repo)
    cfg_path = model_path / "cfg_args"
    if not cfg_path.is_file():
        raise FileNotFoundError(cfg_path)
    defaults, parameter_groups = build_official_defaults(repo, pytorch3d_compat)
    values = vars(defaults).copy()
    values.update(parse_cfg_args(cfg_path))
    args = argparse.Namespace(**values)
    args.model_path = str(model_path)
    args.iteration = int(iteration)
    args.bsz = 1
    args.gaussians_distribution = False
    args.image_distribution = False
    args.image_distribution_mode = ""
    args.distributed_dataset_storage = False
    args.distributed_save = False
    args.local_sampling = False
    args.preload_dataset_to_gpu = False
    args.multiprocesses_image_loading = False
    args.enable_timer = False
    args.check_gpu_memory = False
    args.check_cpu_memory = False
    args.zhx_time = False

    import torch  # noqa: WPS433
    import utils.general_utils as utils  # noqa: WPS433
    from arguments import init_args, print_all_args  # noqa: WPS433
    from gaussian_renderer import (  # noqa: WPS433
        distributed_preprocess3dgs_and_all2all_final,
        prefilter_voxel,
        render_final,
    )
    from gaussian_renderer.workload_division import (  # noqa: WPS433
        DivisionStrategyHistoryFinal,
        start_strategy_final,
    )
    from scene.cameras import Camera  # noqa: WPS433
    from scene.colmap_loader import (  # noqa: WPS433
        qvec2rotmat,
        read_extrinsics_binary,
        read_intrinsics_binary,
    )
    from scene.gaussian_model import GaussianModel  # noqa: WPS433
    from utils.graphics_utils import focal2fov  # noqa: WPS433

    if str(os.environ.get("WORLD_SIZE", "1")) != "1":
        raise RuntimeError("formal CityGS-X packet export is frozen to one GPU")
    utils.init_distributed(args)
    init_args(args)
    utils.set_args(args)
    utils.set_log_file(log_file)
    utils.set_cur_iter(int(iteration))
    print_all_args(args, log_file)
    return {
        "repo": repo,
        "model_path": model_path,
        "args": args,
        "torch": torch,
        "utils": utils,
        "model_group": parameter_groups["model"],
        "pipeline_group": parameter_groups["pipeline"],
        "Camera": Camera,
        "GaussianModel": GaussianModel,
        "qvec2rotmat": qvec2rotmat,
        "read_extrinsics_binary": read_extrinsics_binary,
        "read_intrinsics_binary": read_intrinsics_binary,
        "focal2fov": focal2fov,
        "prefilter_voxel": prefilter_voxel,
        "preprocess": distributed_preprocess3dgs_and_all2all_final,
        "render_final": render_final,
        "DivisionStrategyHistoryFinal": DivisionStrategyHistoryFinal,
        "start_strategy_final": start_strategy_final,
    }


def build_frozen_cameras(
    runtime: dict[str, Any],
    sparse_model: Path,
    camera_root: Path,
    allowlisted_names: list[str],
) -> list[tuple[str, Any, int]]:
    extrinsics = runtime["read_extrinsics_binary"](str(sparse_model / "images.bin"))
    intrinsics = runtime["read_intrinsics_binary"](str(sparse_model / "cameras.bin"))
    images_by_name = {Path(image.name).name: image for image in extrinsics.values()}
    missing = sorted(set(allowlisted_names) - set(images_by_name))
    if missing:
        raise ValueError(f"allowlisted cameras absent from frozen COLMAP model: {missing[:12]}")

    cameras: list[tuple[str, Any, int]] = []
    shapes: set[tuple[int, int]] = set()
    for uid, image_name in enumerate(allowlisted_names):
        image = images_by_name[image_name]
        intrinsic = intrinsics[image.camera_id]
        model = str(intrinsic.model).upper()
        params = np.asarray(intrinsic.params, dtype=np.float64)
        if model == "PINHOLE" and params.size == 4:
            fx, fy, cx, cy = (float(value) for value in params)
        elif model == "SIMPLE_PINHOLE" and params.size == 3:
            fx = fy = float(params[0])
            cx, cy = float(params[1]), float(params[2])
        else:
            raise ValueError(f"unsupported frozen camera model for {image_name}: {intrinsic.model}")
        width, height = int(intrinsic.width), int(intrinsic.height)
        # CityGS-X's official Camera uses the image centre rather than carrying
        # principal-point fields.  The frozen COLMAP model is exactly centred,
        # so this is an identity condition rather than an approximation.
        if not math.isclose(cx, width / 2.0, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError(f"CityGS-X cannot represent off-centre cx for {image_name}: {cx} vs {width / 2.0}")
        if not math.isclose(cy, height / 2.0, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError(f"CityGS-X cannot represent off-centre cy for {image_name}: {cy} vs {height / 2.0}")
        camera = runtime["Camera"](
            colmap_id=int(image.id),
            R=np.transpose(runtime["qvec2rotmat"](image.qvec)),
            T=np.asarray(image.tvec),
            FoVx=runtime["focal2fov"](fx, width),
            FoVy=runtime["focal2fov"](fy, height),
            image=None,
            gt_alpha_mask=None,
            image_name=Path(image_name).stem,
            image_height=height,
            image_width=width,
            uid=uid,
            image_path=str(camera_root / "images" / image_name),
        )
        camera.original_image_backup = None
        shapes.add((width, height))
        cameras.append((image_name, camera, int(image.id)))
    if len(shapes) != 1:
        raise ValueError(f"CityGS-X single-batch renderer requires one frozen image shape, got {sorted(shapes)}")
    width, height = next(iter(shapes))
    runtime["utils"].set_img_size(height, width)
    return cameras


def load_model(runtime: dict[str, Any], cameras: list[tuple[str, Any, int]], iteration: int) -> Any:
    args = runtime["args"]
    dataset = runtime["model_group"].extract(args)
    if int(dataset.resolution) != 1:
        raise ValueError(f"formal CityGS-X route requires resolution=1, got {dataset.resolution}")
    if int(dataset.appearance_dim) != 0:
        raise ValueError("formal CityGS-X route requires appearance_dim=0 for unseen-camera evaluation")
    gaussians = runtime["GaussianModel"](
        dataset.feat_dim,
        dataset.n_offsets,
        dataset.fork,
        dataset.use_feat_bank,
        dataset.appearance_dim,
        dataset.add_opacity_dist,
        dataset.add_cov_dist,
        dataset.add_color_dist,
        dataset.add_level,
        dataset.visible_threshold,
        dataset.dist2level,
        dataset.base_layer,
        dataset.progressive,
        dataset.extend,
    )
    gaussians.set_appearance(len(cameras))
    checkpoint_dir = runtime["model_path"] / "point_cloud" / f"iteration_{iteration}"
    canonical_point_cloud = checkpoint_dir / "point_cloud.ply"
    distributed_point_clouds = sorted(checkpoint_dir.glob("point_cloud_rk*_ws*.ply"))
    if not canonical_point_cloud.is_file() and [path.name for path in distributed_point_clouds] != [
        "point_cloud_rk0_ws1.ply"
    ]:
        raise RuntimeError(
            "unexpected CityGS-X checkpoint point-cloud layout: "
            f"{[path.name for path in distributed_point_clouds]}"
        )
    for required in (checkpoint_dir / "additional_attributes.npz", checkpoint_dir / "checkpoints.pth"):
        if not required.is_file():
            raise FileNotFoundError(required)
    gaussians.load_ply(str(checkpoint_dir))
    gaussians.get_camer_info([camera for _name, camera, _image_id in cameras], [1.0])
    gaussians.load_mlp_checkpoints(str(checkpoint_dir))
    gaussians.eval()
    return gaussians


def render_packets(
    runtime: dict[str, Any],
    cameras: list[tuple[str, Any, int]],
    gaussians: Any,
    out_dir: Path,
    iteration: int,
    numerical_support_floor: float,
    variance_clamp_tolerance: float,
    raw_scale: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    torch = runtime["torch"]
    args = runtime["args"]
    pipeline = runtime["pipeline_group"].extract(args)
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    strategy_dataset = SimpleNamespace(cameras=[camera for _name, camera, _image_id in cameras])
    strategy_history = runtime["DivisionStrategyHistoryFinal"](
        strategy_dataset,
        runtime["utils"].DEFAULT_GROUP.size(),
        runtime["utils"].DEFAULT_GROUP.rank(),
    )
    rows: list[dict[str, Any]] = []
    camera_records: list[dict[str, Any]] = []
    # The official render path explicitly creates screen-space tensors with
    # requires_grad=True even during evaluation.
    with torch.no_grad():
        for index, (image_name, camera, source_image_id) in enumerate(
            tqdm(cameras, desc="Exporting CityGS-X depth")
        ):
            batched_cameras = [camera]
            strategies, _tasks = runtime["start_strategy_final"](batched_cameras, strategy_history)
            gaussians.set_anchor_mask(camera.camera_center, iteration, 1)
            visible_mask = runtime["prefilter_voxel"](camera, gaussians, pipeline, background)
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
                iterations=iteration,
            )
            rendered = runtime["render_final"](batched_cameras, screenspace, strategies)
            render_pkg = rendered[5][0]
            if not isinstance(render_pkg, dict) or "raw_metric_depth_accumulators" not in render_pkg:
                raise RuntimeError(f"CityGS-X renderer did not return raw accumulators for {image_name}")
            raw = render_pkg["raw_metric_depth_accumulators"].detach().squeeze().cpu().numpy().astype(np.float32)
            raw = convert_raw_camera_z_units(raw, camera_z_to_protocol_scale=raw_scale)
            rows.append(
                save_packet(
                    raw,
                    image_name=image_name,
                    index=index,
                    out_dir=out_dir,
                    numerical_support_floor=numerical_support_floor,
                    variance_clamp_tolerance=variance_clamp_tolerance,
                )
            )
            camera_records.append(
                {
                    "image_name": image_name,
                    "source_colmap_image_id": source_image_id,
                    "width": int(camera.image_width),
                    "height": int(camera.image_height),
                    "principal_point_identity_check": True,
                }
            )
    return rows, camera_records


def export(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
    rasterizer_repo = args.rasterizer_repo.expanduser().resolve()
    camera_root = args.camera_root.expanduser().resolve()
    sparse_model = resolve_sparse_model(camera_root)
    out_dir = args.depth_output_dir.expanduser().resolve()
    manifest_path = args.manifest_path.expanduser().resolve() if args.manifest_path else out_dir / "depth_export_manifest.json"
    mapping_path = args.mapping_csv.expanduser().resolve() if args.mapping_csv else out_dir / "depth_map_index.csv"
    conformance_report = args.adapter_conformance_report.expanduser().resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty packet directory: {out_dir}")
    for required in (repo, model_path, rasterizer_repo, conformance_report):
        if not required.exists():
            raise FileNotFoundError(required)
    if not math.isclose(args.raw_camera_z_to_protocol_scale, EXPECTED_RAW_SCALE, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("CityGS-X formal route has no scene rescaling; raw camera-z scale must be exactly 1.0")
    out_dir.mkdir(parents=True)

    allowlisted_names = read_allowlist(
        args.image_list_csv,
        image_name_column=args.image_name_column,
        status_column=args.image_list_status_column,
        status_values=args.image_list_status_values,
    )
    runtime_log_path = out_dir / "citygs_x_export_runtime.log"
    old_cwd = Path.cwd()
    os.chdir(repo)
    try:
        with runtime_log_path.open("w", encoding="utf-8", buffering=1) as runtime_log:
            runtime = load_citygs_x_runtime(
                repo,
                model_path,
                args.iteration,
                runtime_log,
                args.pytorch3d_compat,
            )
            if not runtime["torch"].cuda.is_available():
                raise RuntimeError("CityGS-X packet export requires CUDA")
            cameras = build_frozen_cameras(runtime, sparse_model, camera_root, allowlisted_names)
            gaussians = load_model(runtime, cameras, args.iteration)
            rows, camera_records = render_packets(
                runtime,
                cameras,
                gaussians,
                out_dir,
                args.iteration,
                args.numerical_support_floor,
                args.variance_clamp_tolerance,
                args.raw_camera_z_to_protocol_scale,
            )
    finally:
        os.chdir(old_cwd)

    write_csv(mapping_path, rows, MAPPING_FIELDS)
    checkpoint_dir = model_path / "point_cloud" / f"iteration_{args.iteration}"
    renderer_sources = [repo / "gaussian_renderer" / "__init__.py"]
    rasterizer_sources = [
        rasterizer_repo / "diff_gaussian_rasterization" / "__init__.py",
        rasterizer_repo / "cuda_rasterizer" / "config.h",
        rasterizer_repo / "cuda_rasterizer" / "forward.cu",
    ]
    adapter_patch_files = [path.expanduser().resolve() for path in args.adapter_patch]
    torch = runtime["torch"]
    manifest: dict[str, Any] = {
        "schema": METRIC_PACKET_MANIFEST_SCHEMA,
        "packet_schema": METRIC_PACKET_SCHEMA,
        "created_at": datetime.now().astimezone().isoformat(),
        "purpose": "Metric depth packet for P1 Gaussian GCP geometry evaluator; not a visualization artifact.",
        "method_id": METHOD_ID,
        "train_repo": str(repo),
        "renderer_repository": str(repo),
        "renderer_commit": git_commit(repo),
        "renderer_git_status": git_status_porcelain(repo),
        "rasterizer_repository": str(rasterizer_repo),
        "rasterizer_commit": git_commit(rasterizer_repo),
        "rasterizer_tree_hash": git_tree_hash(rasterizer_repo),
        "rasterizer_git_status": git_status_porcelain(rasterizer_repo),
        "exporter_repository": str(Path(__file__).resolve().parents[2]),
        "exporter_commit": git_commit(Path(__file__).resolve().parents[2]),
        "source_path": str(camera_root),
        "frozen_sparse_model": str(sparse_model),
        "frozen_sparse_model_sha256": {
            name: file_sha256(sparse_model / name) for name in ("cameras.bin", "images.bin", "points3D.bin")
        },
        "model_path": str(model_path),
        "model_content_hash": directory_tree_hash(checkpoint_dir),
        "model_content_hash_algorithm": "sha256_directory_tree_v1",
        "iteration": int(args.iteration),
        "camera_sets": "frozen_evaluation_allowlist",
        "depth_output_dir": str(out_dir),
        "mapping_csv": str(mapping_path),
        "depth_file_format": "compressed numpy .npz metric depth packet",
        "primary_depth_tensor": PRIMARY_DEPTH_TENSOR,
        "primary_depth_semantics": PRIMARY_DEPTH_SEMANTICS,
        "formal_depth_formula": "M1/A",
        "depth_semantics": PRIMARY_DEPTH_SEMANTICS,
        "depth_units": "frozen_colmap_model_camera_z_units",
        "tensor_names": METRIC_PACKET_TENSOR_NAMES + [HISTORICAL_INVALID_TENSOR],
        "tensor_formulas": packet_manifest_tensor_formulas(),
        "diagnostic_tensor_names": [DIAGNOSTIC_VARIANCE_TENSOR, DIAGNOSTIC_VARIANCE_VALID_MASK_TENSOR],
        "dtype": "float32",
        "image_domain": args.image_domain,
        "distorted_or_undistorted": "frozen_native_quarter_colmap_camera_domain",
        "pixel_coordinate_convention": args.pixel_coordinate_convention,
        "protocol_id": args.protocol_id,
        "scene": args.protocol_scene,
        "source_data_release_root_digest_sha256": args.source_data_release_root_digest_sha256,
        "camera_z_unit_contract": args.camera_z_unit_contract,
        "adapter_conformance_status": args.adapter_conformance_status,
        "adapter_conformance_report": str(conformance_report),
        "adapter_conformance_report_sha256": file_sha256(conformance_report),
        "camera_model_source": "frozen protocol COLMAP cameras.bin/images.bin, reconstructed at resolution=1",
        "alpha_cutoff": DEFAULT_ALPHA_CUTOFF,
        "early_termination_threshold": DEFAULT_EARLY_TERMINATION_THRESHOLD,
        "numerical_support_floor": float(args.numerical_support_floor),
        "normalization_epsilon": float(args.normalization_epsilon),
        "variance_clamp_tolerance": float(args.variance_clamp_tolerance),
        **variance_validation_manifest_fields(),
        "depth_semantics_note": (
            "Primary formal P1 depth is alpha_normalized_expected_camera_z=M1/A. "
            "The formal CityGS-X route uses resolution=1 with unscaled COLMAP poses; frozen principal points "
            "are exactly image-centred and therefore identical to the official Camera representation."
        ),
        "alpha_map_available": True,
        "depth_second_moment_available": True,
        "depth_scale_for_evaluator": 1.0,
        "depth_offset_for_evaluator": 0.0,
        "raw_camera_z_to_protocol_scale": float(args.raw_camera_z_to_protocol_scale),
        "raw_camera_z_unit_conversion": {
            "A": "unchanged",
            "M1": "multiply_by_scale",
            "M2": "multiply_by_scale_squared",
            "H": "divide_by_scale",
            "packets_are_in_protocol_colmap_units": True,
        },
        "rendered_view_count": len(rows),
        "depth_index": rows,
        "packet_index": rows,
        "camera_records": camera_records,
        "renderer_source_trace": source_trace(renderer_sources),
        "rasterizer_source_trace": source_trace(rasterizer_sources),
        "image_list_csv": str(args.image_list_csv.expanduser().resolve()),
        "image_name_column": args.image_name_column,
        "image_list_status_column": args.image_list_status_column,
        "image_list_status_values": args.image_list_status_values,
        "renderer_adapter_api": "raw_metric_depth_accumulators_v1",
        "raw_accumulator_tensor_names": list(RAW_ACCUMULATOR_TENSOR_NAMES),
        "derived_packet_computed_on_cpu": True,
        "adapter_patch_files": [
            {"path": str(path), "sha256": file_sha256(path)} for path in adapter_patch_files
        ],
        "pytorch3d_compatibility": {
            "path": str(args.pytorch3d_compat.expanduser().resolve()),
            "tree_sha256": directory_tree_hash(args.pytorch3d_compat.expanduser().resolve()),
            "provided_api": "pytorch3d.transforms.quaternion_to_matrix",
            "scope": "dependency compatibility only; no renderer or model arithmetic change",
        },
        "historical_invalid_tensor_source": "weighted_inverse_camera_z_sum raw H alias",
        "uses_alpha_map": True,
        "uses_depth_second_moment": True,
        "runtime_log": str(runtime_log_path),
        "runtime": {
            "python": sys.version,
            "torch": str(torch.__version__),
            "torch_cuda": str(torch.version.cuda),
            "cuda_available": bool(torch.cuda.is_available()),
            "device_name": torch.cuda.get_device_name(0),
        },
        "notes": [
            "No checkpoint mutation, retraining, support modification, camera resize, or method-specific registration.",
            "CityGS-X source, patches, environment, and weights are internal-only and redistribution-blocked.",
        ],
    }
    if not manifest["rasterizer_commit"] or not manifest["rasterizer_tree_hash"]:
        raise RuntimeError("rasterizer Git provenance is incomplete")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--rasterizer_repo", type=Path, required=True)
    parser.add_argument("--pytorch3d_compat", type=Path, required=True)
    parser.add_argument("--camera_root", type=Path, required=True)
    parser.add_argument("--image_list_csv", type=Path, required=True)
    parser.add_argument("--depth_output_dir", type=Path, required=True)
    parser.add_argument("--manifest_path", type=Path)
    parser.add_argument("--mapping_csv", type=Path)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--numerical_support_floor", type=float, default=DEFAULT_NUMERICAL_SUPPORT_FLOOR)
    parser.add_argument("--normalization_epsilon", type=float, default=DEFAULT_NORMALIZATION_EPSILON)
    parser.add_argument("--variance_clamp_tolerance", type=float, default=DEFAULT_VARIANCE_CLAMP_TOLERANCE)
    parser.add_argument("--raw_camera_z_to_protocol_scale", type=float, default=EXPECTED_RAW_SCALE)
    parser.add_argument("--image_name_column", default="image_name")
    parser.add_argument("--image_list_status_column", default="")
    parser.add_argument("--image_list_status_values", default="")
    parser.add_argument("--image_domain", default="colmap_4_0_4_image_undistorter_pinhole_max_1414")
    parser.add_argument("--pixel_coordinate_convention", default="zero_based_pixel_centers")
    parser.add_argument("--protocol_id", required=True)
    parser.add_argument("--protocol_scene", required=True)
    parser.add_argument("--source_data_release_root_digest_sha256", required=True)
    parser.add_argument("--camera_z_unit_contract", default="frozen_colmap_model_camera_z_units")
    parser.add_argument("--adapter_conformance_status", default="PASS")
    parser.add_argument("--adapter_conformance_report", type=Path, required=True)
    parser.add_argument("--adapter_patch", type=Path, action="append", default=[])
    return parser


def main() -> int:
    manifest = export(build_parser().parse_args())
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
