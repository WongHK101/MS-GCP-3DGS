#!/usr/bin/env python3
"""Export MetroGS raw A/M1/M2/H packets on frozen COLMAP cameras.

The adapter is evaluation-only. It loads the merged Lightning checkpoint,
constructs cameras directly from the frozen native-quarter COLMAP model, and
derives protocol tensors on CPU without changing native render support.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_citygaussian_v2_depth_maps import (  # noqa: E402
    MAPPING_FIELDS,
    build_frozen_cameras,
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
    file_sha256,
    git_commit,
    packet_manifest_tensor_formulas,
    variance_validation_manifest_fields,
)


METHOD_ID = "metrogs"
EXPECTED_RENDERER_CLASS = "DistributedRendererImpl"
EXPECTED_RAW_SCALE = 1.0


def git_status_porcelain(repo: Path) -> str:
    try:
        return subprocess.check_output(
            [
                "git",
                "-c",
                f"safe.directory={repo.as_posix()}",
                "-C",
                str(repo),
                "status",
                "--porcelain=v1",
            ],
            text=True,
        ).strip()
    except Exception as exc:  # noqa: BLE001
        return f"ERROR:{type(exc).__name__}:{exc}"


def load_metrogs_runtime(repo: Path) -> dict[str, Any]:
    repo = repo.expanduser().resolve()
    if not repo.is_dir():
        raise FileNotFoundError(repo)
    sys.path.insert(0, str(repo))

    import torch  # noqa: WPS433
    from internal.cameras.cameras import Cameras  # noqa: WPS433
    from internal.utils import colmap as colmap_utils  # noqa: WPS433
    from internal.utils.gaussian_model_loader import GaussianModelLoader  # noqa: WPS433

    return {
        "repo": repo,
        "torch": torch,
        "Cameras": Cameras,
        "colmap_utils": colmap_utils,
        "GaussianModelLoader": GaussianModelLoader,
    }


def export(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    rasterizer_repo = args.rasterizer_repo.expanduser().resolve()
    sparse_model = resolve_sparse_model(args.camera_root)
    out_dir = args.depth_output_dir.expanduser().resolve()
    manifest_path = (
        args.manifest_path.expanduser().resolve()
        if args.manifest_path
        else out_dir / "depth_export_manifest.json"
    )
    mapping_path = (
        args.mapping_csv.expanduser().resolve()
        if args.mapping_csv
        else out_dir / "depth_map_index.csv"
    )
    conformance_report = args.adapter_conformance_report.expanduser().resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty packet directory: {out_dir}")
    for required in (checkpoint, conformance_report, rasterizer_repo):
        if not required.exists():
            raise FileNotFoundError(required)
    if not math.isclose(
        args.raw_camera_z_to_protocol_scale,
        EXPECTED_RAW_SCALE,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError(
            "MetroGS native-quarter route has no scene rescaling; raw camera-z scale must be 1.0"
        )

    out_dir.mkdir(parents=True)
    allowlisted_names = read_allowlist(
        args.image_list_csv,
        image_name_column=args.image_name_column,
        status_column=args.image_list_status_column,
        status_values=args.image_list_status_values,
    )
    runtime = load_metrogs_runtime(repo)
    torch = runtime["torch"]
    if not torch.cuda.is_available():
        raise RuntimeError("MetroGS packet export requires CUDA")
    frozen_cameras = build_frozen_cameras(runtime, sparse_model, allowlisted_names)

    old_cwd = Path.cwd()
    os.chdir(repo)
    try:
        model, renderer, checkpoint_payload = runtime[
            "GaussianModelLoader"
        ].initialize_model_and_renderer_from_checkpoint_file(
            str(checkpoint), device="cuda", eval_mode=True, pre_activate=True
        )
        if renderer.__class__.__name__ != EXPECTED_RENDERER_CLASS:
            raise RuntimeError(
                f"expected {EXPECTED_RENDERER_CLASS}, got "
                f"{renderer.__class__.__module__}.{renderer.__class__.__name__}"
            )
        forward_parameters = set(inspect.signature(renderer.forward).parameters)
        if "return_raw_metric_depth_accumulators" not in forward_parameters:
            raise RuntimeError(f"renderer lacks raw-moment API: {sorted(forward_parameters)}")

        background = torch.zeros(3, dtype=torch.float32, device="cuda")
        rows: list[dict[str, Any]] = []
        camera_records: list[dict[str, Any]] = []
        with torch.no_grad():
            for index, (image_name, camera, source_image_id) in enumerate(
                tqdm(frozen_cameras, desc="Exporting MetroGS depth")
            ):
                camera = camera.to_device("cuda")
                payload = renderer(
                    camera,
                    model,
                    background,
                    return_raw_metric_depth_accumulators=True,
                )
                if not isinstance(payload, dict) or "raw_metric_depth_accumulators" not in payload:
                    raise RuntimeError(f"renderer did not return raw accumulators for {image_name}")
                raw = (
                    payload["raw_metric_depth_accumulators"]
                    .detach()
                    .squeeze()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                raw = convert_raw_camera_z_units(
                    raw,
                    camera_z_to_protocol_scale=args.raw_camera_z_to_protocol_scale,
                )
                rows.append(
                    save_packet(
                        raw,
                        image_name=image_name,
                        index=index,
                        out_dir=out_dir,
                        numerical_support_floor=args.numerical_support_floor,
                        variance_clamp_tolerance=args.variance_clamp_tolerance,
                    )
                )
                camera_records.append(
                    {
                        "image_name": image_name,
                        "source_colmap_image_id": source_image_id,
                        "width": int(camera.width.item()),
                        "height": int(camera.height.item()),
                        "appearance_id": int(camera.appearance_id.item()),
                    }
                )
        del checkpoint_payload
    finally:
        os.chdir(old_cwd)

    write_csv(mapping_path, rows, MAPPING_FIELDS)
    renderer_sources = [repo / "internal" / "renderers" / "metrogs_renderer.py"]
    rasterizer_sources = [
        rasterizer_repo / "dist_2dgs" / "__init__.py",
        rasterizer_repo / "cuda_rasterizer" / "auxiliary.h",
        rasterizer_repo / "cuda_rasterizer" / "forward.cu",
        rasterizer_repo / "rasterize_points.cu",
    ]
    adapter_patch_files = [path.expanduser().resolve() for path in args.adapter_patch]
    manifest: dict[str, Any] = {
        "schema": METRIC_PACKET_MANIFEST_SCHEMA,
        "packet_schema": METRIC_PACKET_SCHEMA,
        "created_at": datetime.now().astimezone().isoformat(),
        "purpose": "Metric depth packet for P1 Gaussian GCP geometry evaluator; not a visualization artifact.",
        "method_id": METHOD_ID,
        "train_repo": str(repo),
        "renderer_repository": str(repo),
        "renderer_commit": git_commit(repo),
        "renderer_tree_hash": git_tree_hash(repo),
        "renderer_git_status": git_status_porcelain(repo),
        "renderer_class": EXPECTED_RENDERER_CLASS,
        "rasterizer_repository": str(rasterizer_repo),
        "rasterizer_commit": git_commit(rasterizer_repo),
        "rasterizer_tree_hash": git_tree_hash(rasterizer_repo),
        "rasterizer_git_status": git_status_porcelain(rasterizer_repo),
        "exporter_repository": str(Path(__file__).resolve().parents[2]),
        "exporter_commit": git_commit(Path(__file__).resolve().parents[2]),
        "source_path": str(args.camera_root.expanduser().resolve()),
        "frozen_sparse_model": str(sparse_model),
        "frozen_sparse_model_sha256": {
            name: file_sha256(sparse_model / name)
            for name in ("cameras.bin", "images.bin", "points3D.bin")
        },
        "model_path": str(checkpoint),
        "model_content_hash": file_sha256(checkpoint),
        "model_content_hash_algorithm": "sha256_file",
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
        "diagnostic_tensor_names": [
            DIAGNOSTIC_VARIANCE_TENSOR,
            DIAGNOSTIC_VARIANCE_VALID_MASK_TENSOR,
        ],
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
        "camera_model_source": "frozen protocol COLMAP cameras.bin/images.bin; no image resize",
        "alpha_cutoff": DEFAULT_ALPHA_CUTOFF,
        "early_termination_threshold": DEFAULT_EARLY_TERMINATION_THRESHOLD,
        "numerical_support_floor": float(args.numerical_support_floor),
        "normalization_epsilon": float(args.normalization_epsilon),
        "variance_clamp_tolerance": float(args.variance_clamp_tolerance),
        **variance_validation_manifest_fields(),
        "depth_semantics_note": (
            "Primary P1 depth is alpha-normalized expected per-ray 2D-surfel-intersection camera z. "
            "The frozen MetroGS route uses the original COLMAP coordinates, scene_scale=1, "
            "reorient=false, and native-quarter down_sample_factor=1."
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
            {"path": str(path), "sha256": file_sha256(path)}
            for path in adapter_patch_files
        ],
        "historical_invalid_tensor_source": "weighted_inverse_camera_z_sum raw H alias",
        "uses_alpha_map": True,
        "uses_depth_second_moment": True,
        "runtime": {
            "python": sys.version,
            "torch": str(torch.__version__),
            "torch_cuda": str(torch.version.cuda),
            "cuda_available": bool(torch.cuda.is_available()),
            "device_name": torch.cuda.get_device_name(0),
        },
        "notes": [
            "Evaluation-only renderer/rasterizer copy; official training source and checkpoint are unchanged.",
            "No support modification, camera resize, retraining, or method-specific registration.",
        ],
    }
    if not manifest["renderer_commit"] or not manifest["rasterizer_commit"]:
        raise RuntimeError("MetroGS renderer/rasterizer Git provenance is incomplete")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rasterizer_repo", type=Path, required=True)
    parser.add_argument("--camera_root", type=Path, required=True)
    parser.add_argument("--image_list_csv", type=Path, required=True)
    parser.add_argument("--depth_output_dir", type=Path, required=True)
    parser.add_argument("--manifest_path", type=Path)
    parser.add_argument("--mapping_csv", type=Path)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument(
        "--numerical_support_floor",
        type=float,
        default=DEFAULT_NUMERICAL_SUPPORT_FLOOR,
    )
    parser.add_argument(
        "--normalization_epsilon",
        type=float,
        default=DEFAULT_NORMALIZATION_EPSILON,
    )
    parser.add_argument(
        "--variance_clamp_tolerance",
        type=float,
        default=DEFAULT_VARIANCE_CLAMP_TOLERANCE,
    )
    parser.add_argument(
        "--raw_camera_z_to_protocol_scale", type=float, default=EXPECTED_RAW_SCALE
    )
    parser.add_argument("--image_name_column", default="image_name")
    parser.add_argument("--image_list_status_column", default="")
    parser.add_argument("--image_list_status_values", default="")
    parser.add_argument(
        "--image_domain", default="colmap_4_0_4_image_undistorter_pinhole_max_1414"
    )
    parser.add_argument("--pixel_coordinate_convention", default="zero_based_pixel_centers")
    parser.add_argument("--protocol_id", required=True)
    parser.add_argument("--protocol_scene", required=True)
    parser.add_argument("--source_data_release_root_digest_sha256", required=True)
    parser.add_argument(
        "--camera_z_unit_contract", default="frozen_colmap_model_camera_z_units"
    )
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
