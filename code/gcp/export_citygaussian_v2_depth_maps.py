#!/usr/bin/env python3
"""Export CityGaussianV2 raw A/M1/M2/H packets on frozen COLMAP cameras.

This adapter is evaluation-only.  It loads the merged Lightning checkpoint,
constructs cameras directly from the frozen native-quarter COLMAP model, and
derives the protocol packet tensors on CPU without changing renderer support.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_gaussian_depth_maps import (  # noqa: E402
    RAW_ACCUMULATOR_TENSOR_NAMES,
    convert_raw_camera_z_units,
    derive_packet_from_raw_accumulators,
    git_tree_hash,
    packet_filename,
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
    directory_tree_hash,
    file_sha256,
    git_commit,
    packet_manifest_tensor_formulas,
    recompute_and_compare_packet,
    variance_validation_manifest_fields,
)


METHOD_ID = "citygaussian_v2"
EXPECTED_RENDERER_CLASS = "SepDepthTrim2DGSRenderer"
EXPECTED_RAW_SCALE = 1.0
MAPPING_FIELDS = (
    "index",
    "split",
    "image_name",
    "packet_path",
    "depth_path",
    "packet_sha256",
    "packet_bytes",
    "height",
    "width",
    "dtype",
    "primary_depth_tensor",
    "primary_depth_semantics",
    "tensor_names",
    "valid_pixel_count",
    "accumulated_alpha_min",
    "accumulated_alpha_max",
    "expected_camera_z_finite_count",
    "packet_recompute_passed",
    "variance_validation_policy",
    "variance_validation_max_abs_error",
    "variance_validation_max_allowed_error",
    "variance_validation_max_error_to_bound_ratio",
    "variance_packet_ref_abs_error",
    "variance_packet_ref_allowed_error",
    "variance_packet_ref_consistency_ratio",
    "variance_packet_negative_to_bound_ratio",
    "variance_ref_negative_to_bound_ratio",
    "variance_consistency_fail_count",
    "variance_nonnegativity_unresolved_count",
    "variance_diagnostic_valid_ratio",
    "variance_diagnostic_invalid_count",
    "variance_validation_failing_pixel_count",
    "variance_raw_negative_count",
    "variance_cancellation_accepted_count",
    "variance_cancellation_rejected_count",
    "variance_min_raw",
    "variance_max_negative_magnitude",
    "variance_max_negative_to_bound_ratio",
    "variance_diagnostic_zero_clamped_count",
    "variance_max_connected_component_size",
)


def git_status_porcelain(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={repo.as_posix()}", "-C", str(repo), "status", "--porcelain=v1"],
            text=True,
        ).strip()
    except Exception as exc:  # noqa: BLE001
        return f"ERROR:{type(exc).__name__}:{exc}"


def resolve_sparse_model(path: Path) -> Path:
    path = path.expanduser().resolve()
    candidates = (path / "sparse" / "0", path / "sparse", path)
    matched = [
        candidate
        for candidate in candidates
        if all((candidate / name).is_file() for name in ("cameras.bin", "images.bin", "points3D.bin"))
    ]
    if not matched:
        raise FileNotFoundError(f"no complete COLMAP model below {path}")
    return matched[0]


def read_allowlist(
    path: Path,
    *,
    image_name_column: str,
    status_column: str,
    status_values: str,
) -> list[str]:
    accepted = {value.strip() for value in status_values.split(",") if value.strip()}
    names: list[str] = []
    seen: set[str] = set()
    with path.expanduser().resolve().open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if status_column and accepted and str(row.get(status_column, "")).strip() not in accepted:
                continue
            name = Path(str(row.get(image_name_column, "")).strip()).name
            if not name:
                continue
            if name in seen:
                raise ValueError(f"duplicate allowlisted image name: {name}")
            seen.add(name)
            names.append(name)
    if not names:
        raise ValueError(f"allowlist selected no images: {path}")
    return names


def load_citygaussian_runtime(repo: Path) -> dict[str, Any]:
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


def build_frozen_cameras(
    runtime: dict[str, Any],
    sparse_model: Path,
    allowlisted_names: Sequence[str],
) -> list[tuple[str, Any, int]]:
    torch = runtime["torch"]
    Cameras = runtime["Cameras"]
    colmap_utils = runtime["colmap_utils"]
    cameras = colmap_utils.read_cameras_binary(str(sparse_model / "cameras.bin"))
    images = colmap_utils.read_images_binary(str(sparse_model / "images.bin"))
    images_by_name = {Path(image.name).name: image for image in images.values()}
    requested = list(allowlisted_names)
    missing = sorted(set(requested) - set(images_by_name))
    if missing:
        raise ValueError(f"allowlisted cameras absent from frozen COLMAP model: {missing[:12]}")

    # This reproduces CityGaussian's default appearance-group rule.  The
    # selected CityGaussianV2 renderer has no appearance module, but retaining
    # the IDs makes the camera container faithful to the official parser.
    camera_ids = sorted({int(image.camera_id) for image in images.values()})
    appearance_by_camera = {camera_id: index for index, camera_id in enumerate(camera_ids)}
    denominator = float(len(camera_ids))

    r_list: list[np.ndarray] = []
    t_list: list[np.ndarray] = []
    fx_list: list[float] = []
    fy_list: list[float] = []
    cx_list: list[float] = []
    cy_list: list[float] = []
    width_list: list[int] = []
    height_list: list[int] = []
    appearance_list: list[int] = []
    normalized_appearance_list: list[float] = []
    source_image_ids: list[int] = []

    for name in requested:
        image = images_by_name[name]
        camera = cameras[image.camera_id]
        model = str(camera.model).upper()
        params = np.asarray(camera.params, dtype=np.float64)
        if model == "PINHOLE" and params.size == 4:
            fx, fy, cx, cy = (float(value) for value in params)
        elif model == "SIMPLE_PINHOLE" and params.size == 3:
            fx = fy = float(params[0])
            cx, cy = float(params[1]), float(params[2])
        else:
            raise ValueError(f"unsupported frozen camera model for {name}: {camera.model} {params.tolist()}")
        appearance_id = appearance_by_camera[int(image.camera_id)]
        r_list.append(np.asarray(image.qvec2rotmat(), dtype=np.float32))
        t_list.append(np.asarray(image.tvec, dtype=np.float32))
        fx_list.append(fx)
        fy_list.append(fy)
        cx_list.append(cx)
        cy_list.append(cy)
        width_list.append(int(camera.width))
        height_list.append(int(camera.height))
        appearance_list.append(appearance_id)
        normalized_appearance_list.append(appearance_id / denominator)
        source_image_ids.append(int(image.id))

    widths = torch.tensor(width_list, dtype=torch.int32)
    camera_batch = Cameras(
        R=torch.tensor(np.stack(r_list), dtype=torch.float32),
        T=torch.tensor(np.stack(t_list), dtype=torch.float32),
        fx=torch.tensor(fx_list, dtype=torch.float32),
        fy=torch.tensor(fy_list, dtype=torch.float32),
        cx=torch.tensor(cx_list, dtype=torch.float32),
        cy=torch.tensor(cy_list, dtype=torch.float32),
        width=widths,
        height=torch.tensor(height_list, dtype=torch.int32),
        appearance_id=torch.tensor(appearance_list, dtype=torch.int32),
        normalized_appearance_id=torch.tensor(normalized_appearance_list, dtype=torch.float32),
        distortion_params=None,
        camera_type=torch.zeros_like(widths, dtype=torch.int8),
    )
    return [(name, camera_batch[index], source_image_ids[index]) for index, name in enumerate(requested)]


def save_packet(
    raw_accumulators: np.ndarray,
    *,
    image_name: str,
    index: int,
    out_dir: Path,
    numerical_support_floor: float,
    variance_clamp_tolerance: float,
) -> dict[str, Any]:
    packet_payload = derive_packet_from_raw_accumulators(
        raw_accumulators,
        numerical_support_floor=numerical_support_floor,
        variance_clamp_tolerance=variance_clamp_tolerance,
    )
    packet_payload["metric_depth_valid_mask"] = packet_payload["metric_depth_valid_mask"] > 0.5
    packet_payload[HISTORICAL_INVALID_TENSOR] = packet_payload["weighted_inverse_camera_z_sum"].copy()
    packet_path = out_dir / packet_filename(image_name)
    np.savez_compressed(packet_path, **packet_payload)

    recompute = recompute_and_compare_packet(
        packet_payload,
        numerical_support_floor=numerical_support_floor,
        variance_clamp_tolerance=variance_clamp_tolerance,
        **variance_validation_manifest_fields(),
    )
    if not recompute["passed"]:
        raise RuntimeError(f"derived tensor recomputation failed for {image_name}: {recompute}")
    variance_row = next(row for row in recompute["rows"] if row["tensor"] == "camera_z_variance")
    alpha = packet_payload["accumulated_alpha"]
    return {
        "index": index,
        "split": "frozen_evaluation",
        "image_name": image_name,
        "packet_path": str(packet_path),
        "depth_path": str(packet_path),
        "packet_sha256": file_sha256(packet_path),
        "packet_bytes": packet_path.stat().st_size,
        "height": int(alpha.shape[0]),
        "width": int(alpha.shape[1]),
        "dtype": "float32",
        "primary_depth_tensor": PRIMARY_DEPTH_TENSOR,
        "primary_depth_semantics": PRIMARY_DEPTH_SEMANTICS,
        "tensor_names": "|".join(METRIC_PACKET_TENSOR_NAMES + [HISTORICAL_INVALID_TENSOR]),
        "valid_pixel_count": int(np.count_nonzero(packet_payload["metric_depth_valid_mask"])),
        "accumulated_alpha_min": float(np.nanmin(alpha)),
        "accumulated_alpha_max": float(np.nanmax(alpha)),
        "expected_camera_z_finite_count": int(np.isfinite(packet_payload[PRIMARY_DEPTH_TENSOR]).sum()),
        "packet_recompute_passed": True,
        "variance_validation_policy": variance_row["variance_validation_policy"],
        "variance_validation_max_abs_error": variance_row["max_abs_error"],
        "variance_validation_max_allowed_error": variance_row["max_allowed_error"],
        "variance_validation_max_error_to_bound_ratio": variance_row["max_error_to_bound_ratio"],
        "variance_packet_ref_abs_error": variance_row["packet_ref_abs_error"],
        "variance_packet_ref_allowed_error": variance_row["packet_ref_allowed_error"],
        "variance_packet_ref_consistency_ratio": variance_row["packet_ref_consistency_ratio"],
        "variance_packet_negative_to_bound_ratio": variance_row["variance_packet_negative_max_ratio"],
        "variance_ref_negative_to_bound_ratio": variance_row["variance_ref_negative_max_ratio"],
        "variance_consistency_fail_count": variance_row["variance_consistency_fail_count"],
        "variance_nonnegativity_unresolved_count": variance_row["variance_nonnegativity_unresolved_count"],
        "variance_diagnostic_valid_ratio": variance_row["variance_diagnostic_valid_ratio"],
        "variance_diagnostic_invalid_count": variance_row["variance_diagnostic_invalid_count"],
        "variance_validation_failing_pixel_count": variance_row["failing_pixel_count"],
        "variance_raw_negative_count": variance_row["raw_negative_variance_count"],
        "variance_cancellation_accepted_count": variance_row["cancellation_accepted_count"],
        "variance_cancellation_rejected_count": variance_row["cancellation_rejected_count"],
        "variance_min_raw": variance_row["min_raw_variance"],
        "variance_max_negative_magnitude": variance_row["max_negative_magnitude"],
        "variance_max_negative_to_bound_ratio": variance_row["max_negative_to_bound_ratio"],
        "variance_diagnostic_zero_clamped_count": variance_row["diagnostic_zero_clamped_count"],
        "variance_max_connected_component_size": variance_row["accepted_negative_spatial_distribution"][
            "max_connected_component_size"
        ],
    }


def source_trace(paths: Iterable[Path]) -> list[dict[str, Any]]:
    return [
        {"path": str(path), "sha256": file_sha256(path) if path.is_file() else "", "exists": path.is_file()}
        for path in paths
    ]


def export(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    rasterizer_repo = args.rasterizer_repo.expanduser().resolve()
    sparse_model = resolve_sparse_model(args.camera_root)
    out_dir = args.depth_output_dir.expanduser().resolve()
    manifest_path = args.manifest_path.expanduser().resolve() if args.manifest_path else out_dir / "depth_export_manifest.json"
    mapping_path = args.mapping_csv.expanduser().resolve() if args.mapping_csv else out_dir / "depth_map_index.csv"
    conformance_report = args.adapter_conformance_report.expanduser().resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty packet directory: {out_dir}")
    for required in (checkpoint, conformance_report, rasterizer_repo):
        if not required.exists():
            raise FileNotFoundError(required)
    if not math.isclose(args.raw_camera_z_to_protocol_scale, EXPECTED_RAW_SCALE, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("CityGaussianV2 formal route has no scene rescaling; raw camera-z scale must be exactly 1.0")

    out_dir.mkdir(parents=True)
    allowlisted_names = read_allowlist(
        args.image_list_csv,
        image_name_column=args.image_name_column,
        status_column=args.image_list_status_column,
        status_values=args.image_list_status_values,
    )
    runtime = load_citygaussian_runtime(repo)
    torch = runtime["torch"]
    if not torch.cuda.is_available():
        raise RuntimeError("CityGaussianV2 packet export requires CUDA")
    frozen_cameras = build_frozen_cameras(runtime, sparse_model, allowlisted_names)

    old_cwd = Path.cwd()
    os.chdir(repo)
    try:
        model, renderer, checkpoint_payload = runtime["GaussianModelLoader"].initialize_model_and_renderer_from_checkpoint_file(
            str(checkpoint), device="cuda", eval_mode=True, pre_activate=True
        )
        if renderer.__class__.__name__ != EXPECTED_RENDERER_CLASS:
            raise RuntimeError(
                f"expected {EXPECTED_RENDERER_CLASS}, got {renderer.__class__.__module__}.{renderer.__class__.__name__}"
            )
        forward_parameters = set(inspect.signature(renderer.forward).parameters)
        if "return_raw_metric_depth_accumulators" not in forward_parameters:
            raise RuntimeError(f"renderer lacks raw-moment API: {sorted(forward_parameters)}")
        background = torch.zeros(3, dtype=torch.float32, device="cuda")
        rows: list[dict[str, Any]] = []
        camera_records: list[dict[str, Any]] = []
        # Match the official renderer's evaluation context.  Its forward path
        # explicitly creates screen-space tensors with requires_grad=True, so
        # inference_mode is intentionally not used here.
        with torch.no_grad():
            for index, (image_name, camera, source_image_id) in enumerate(
                tqdm(frozen_cameras, desc="Exporting CityGaussianV2 depth")
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
                raw = payload["raw_metric_depth_accumulators"].detach().squeeze().cpu().numpy().astype(np.float32)
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
                    }
                )
        del checkpoint_payload
    finally:
        os.chdir(old_cwd)

    write_csv(mapping_path, rows, MAPPING_FIELDS)
    rasterizer_sources = [
        rasterizer_repo / "diff_trim_surfel_rasterization" / "__init__.py",
        rasterizer_repo / "cuda_rasterizer" / "forward.cu",
        rasterizer_repo / "cuda_rasterizer" / "auxiliary.h",
        rasterizer_repo / "rasterize_points.cu",
    ]
    renderer_sources = [repo / "internal" / "renderers" / "sep_depth_trim_2dgs_renderer.py"]
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
        "renderer_git_status": git_status_porcelain(repo),
        "rasterizer_repository": str(rasterizer_repo),
        "rasterizer_commit": git_commit(rasterizer_repo),
        "rasterizer_tree_hash": git_tree_hash(rasterizer_repo),
        "rasterizer_git_status": git_status_porcelain(rasterizer_repo),
        "exporter_repository": str(Path(__file__).resolve().parents[2]),
        "exporter_commit": git_commit(Path(__file__).resolve().parents[2]),
        "source_path": str(args.camera_root.expanduser().resolve()),
        "frozen_sparse_model": str(sparse_model),
        "frozen_sparse_model_sha256": {
            name: file_sha256(sparse_model / name) for name in ("cameras.bin", "images.bin", "points3D.bin")
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
        "camera_model_source": "frozen protocol COLMAP cameras.bin/images.bin, reconstructed without resize",
        "alpha_cutoff": DEFAULT_ALPHA_CUTOFF,
        "early_termination_threshold": DEFAULT_EARLY_TERMINATION_THRESHOLD,
        "numerical_support_floor": float(args.numerical_support_floor),
        "normalization_epsilon": float(args.normalization_epsilon),
        "variance_clamp_tolerance": float(args.variance_clamp_tolerance),
        **variance_validation_manifest_fields(),
        "depth_semantics_note": (
            "Primary formal P1 depth is alpha_normalized_expected_camera_z=M1/A. "
            "The official CityGaussianV2 route uses down_sample_factor=1, scene_scale=1, reorient=false, "
            "and normalize=false, so raw camera-z is already in frozen COLMAP units."
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
            "No checkpoint mutation, retraining, support modification, camera resize, or method-specific registration.",
            "Packet arrays are linear float/bool data; visualization files must not enter metric evaluation.",
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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rasterizer_repo", type=Path, required=True)
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
