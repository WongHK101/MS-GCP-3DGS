from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
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
    derive_metric_depth_packet,
    file_sha256,
    git_commit,
    packet_manifest_tensor_formulas,
    recompute_and_compare_packet,
    tensor_stats,
    variance_validation_manifest_fields,
)

DEFAULT_RASTERIZER_DEPTH_SEMANTICS = "alpha_weighted_unnormalized_inverse_camera_z"
RAW_ACCUMULATOR_TENSOR_NAMES = tuple(METRIC_PACKET_TENSOR_NAMES[:4])


def git_tree_hash(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "show", "-s", "--format=%T", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def resolve_rasterizer_repo(train_repo: Path, requested: str = "") -> Path:
    resolved_train = train_repo.resolve()
    if requested:
        candidate = Path(requested).expanduser()
        if not candidate.is_absolute():
            candidate = resolved_train / candidate
        candidate = candidate.resolve()
        if not candidate.is_relative_to(resolved_train):
            raise ValueError(f"rasterizer repository must be inside train_repo: {candidate}")
        if not candidate.is_dir():
            raise FileNotFoundError(f"rasterizer repository not found: {candidate}")
        return candidate

    candidates = [
        resolved_train / "submodules" / "diff-gaussian-rasterization",
        resolved_train / "submodules" / "diff-surfel-rasterization",
    ]
    existing = [path for path in candidates if path.is_dir()]
    if len(existing) != 1:
        raise RuntimeError(
            "could not infer one rasterizer repository; pass --rasterizer_repo explicitly: "
            f"{[str(path) for path in existing]}"
        )
    return existing[0]


def parse_train_repo(argv: Sequence[str]) -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--train_repo", required=True)
    known, _unknown = parser.parse_known_args(argv)
    return Path(known.train_repo).expanduser().resolve()


def load_gaussian_runtime(train_repo: Path) -> Dict[str, Any]:
    if not train_repo.exists():
        raise FileNotFoundError(f"training repository not found: {train_repo}")
    sys.path.insert(0, str(train_repo))

    import torch  # noqa: WPS433
    from arguments import ModelParams, PipelineParams, get_combined_args  # noqa: WPS433
    from gaussian_renderer import render  # noqa: WPS433
    from scene import Scene  # noqa: WPS433
    from scene.gaussian_model import GaussianModel  # noqa: WPS433
    from utils.general_utils import safe_state  # noqa: WPS433

    try:
        from diff_gaussian_rasterization import SparseGaussianAdam  # noqa: F401,WPS433

        sparse_adam_available = True
    except Exception:
        sparse_adam_available = False

    return {
        "train_repo": train_repo,
        "torch": torch,
        "ModelParams": ModelParams,
        "PipelineParams": PipelineParams,
        "get_combined_args": get_combined_args,
        "GaussianModel": GaussianModel,
        "render": render,
        "Scene": Scene,
        "safe_state": safe_state,
        "sparse_adam_available": sparse_adam_available,
        "render_parameters": set(inspect.signature(render).parameters),
    }


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_allowlist(args: argparse.Namespace) -> Dict[str, str] | None:
    if not args.image_list_csv:
        return None
    path = Path(args.image_list_csv).expanduser().resolve()
    names: Dict[str, str] = {}
    accepted_values = {
        value.strip()
        for value in str(args.image_list_status_values).split(",")
        if value.strip()
    }
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if args.image_list_status_column:
                value = str(row.get(args.image_list_status_column, "")).strip()
                if accepted_values and value not in accepted_values:
                    continue
            name = str(row.get(args.image_name_column, "")).strip()
            if name:
                canonical = Path(name).name
                for alias in {name, canonical, Path(canonical).stem}:
                    previous = names.get(alias)
                    if previous is not None and previous != canonical:
                        raise ValueError(f"ambiguous image-list alias {alias!r}: {previous!r} vs {canonical!r}")
                    names[alias] = canonical
    return names


def camera_name(view: Any) -> str:
    name = str(getattr(view, "image_name", "")).strip()
    if not name:
        raise ValueError(f"camera has no image_name: {view}")
    return name


def depth_filename(image_name: str) -> str:
    stem = Path(image_name).stem
    if not stem:
        raise ValueError(f"invalid image_name for depth filename: {image_name!r}")
    return f"{stem}.npy"


def packet_filename(image_name: str) -> str:
    stem = Path(image_name).stem
    if not stem:
        raise ValueError(f"invalid image_name for packet filename: {image_name!r}")
    return f"{stem}_metric_depth_packet.npz"


def derive_packet_from_raw_accumulators(
    raw_accumulators: np.ndarray,
    *,
    numerical_support_floor: float,
    variance_clamp_tolerance: float,
) -> Dict[str, np.ndarray]:
    raw = np.asarray(raw_accumulators, dtype=np.float32)
    if raw.ndim != 3 or raw.shape[0] != len(RAW_ACCUMULATOR_TENSOR_NAMES):
        raise ValueError(
            "raw_metric_depth_accumulators must have shape "
            f"({len(RAW_ACCUMULATOR_TENSOR_NAMES)}, H, W), got {raw.shape}"
        )
    return derive_metric_depth_packet(
        *(raw[index] for index in range(len(RAW_ACCUMULATOR_TENSOR_NAMES))),
        numerical_support_floor=float(numerical_support_floor),
        variance_clamp_tolerance=float(variance_clamp_tolerance),
    )


def collect_views(
    scene: Any,
    camera_sets: str,
    allowlist: Dict[str, str] | None = None,
) -> List[tuple[str, Any, str]]:
    views: List[tuple[str, Any]] = []
    if camera_sets in {"train", "all"}:
        views.extend(("train", view) for view in scene.getTrainCameras())
    if camera_sets in {"test", "all"}:
        views.extend(("test", view) for view in scene.getTestCameras())

    seen: set[str] = set()
    unique: List[tuple[str, Any, str]] = []
    for split, view in views:
        name = camera_name(view)
        if allowlist is None:
            canonical = Path(name).name
        else:
            canonical = allowlist.get(name) or allowlist.get(Path(name).name) or allowlist.get(Path(name).stem)
            if canonical is None:
                continue
        if canonical in seen:
            continue
        seen.add(canonical)
        unique.append((split, view, canonical))
    if allowlist is not None:
        expected = set(allowlist.values())
        resolved = {canonical for _split, _view, canonical in unique}
        if resolved != expected:
            raise ValueError(
                "renderer camera list does not exactly match requested image list: "
                f"missing={sorted(expected - resolved)[:12]} extra={sorted(resolved - expected)[:12]}"
            )
    return unique


def export_depths(args: argparse.Namespace, dataset: Any, pipeline: Any, runtime: Dict[str, Any]) -> Dict[str, Any]:
    out_dir = Path(args.depth_output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest_path).expanduser().resolve() if args.manifest_path else out_dir / "depth_export_manifest.json"
    mapping_path = Path(args.mapping_csv).expanduser().resolve() if args.mapping_csv else out_dir / "depth_map_index.csv"

    train_repo = Path(runtime["train_repo"])
    torch = runtime["torch"]
    GaussianModel = runtime["GaussianModel"]
    Scene = runtime["Scene"]
    render = runtime["render"]
    sparse_adam_available = bool(runtime["sparse_adam_available"])
    render_parameters = set(runtime["render_parameters"])
    if "return_raw_metric_depth_accumulators" in render_parameters:
        adapter_api = "raw_metric_depth_accumulators_v1"
    elif "return_metric_depth_packet" in render_parameters:
        adapter_api = "legacy_metric_depth_packet_v2"
    else:
        raise RuntimeError(
            "renderer exposes neither the required raw-accumulator API nor the legacy packet API; "
            f"available={sorted(render_parameters)}"
        )

    old_cwd = Path.cwd()
    os.chdir(train_repo)
    try:
        with torch.no_grad():
            gaussians = GaussianModel(dataset.sh_degree)
            scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
            bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

            rows: List[Dict[str, Any]] = []
            allowlist = read_allowlist(args)
            views = collect_views(scene, args.camera_sets, allowlist=allowlist)
            for index, (split, view, image_name) in enumerate(tqdm(views, desc="Exporting Gaussian depth")):
                render_kwargs: Dict[str, Any] = {}
                if adapter_api == "raw_metric_depth_accumulators_v1":
                    render_kwargs["return_raw_metric_depth_accumulators"] = True
                else:
                    render_kwargs.update(
                        {
                            "return_metric_depth_packet": True,
                            "numerical_support_floor": float(args.numerical_support_floor),
                            "normalization_epsilon": float(args.normalization_epsilon),
                            "variance_clamp_tolerance": float(args.variance_clamp_tolerance),
                        }
                    )
                if "use_trained_exp" in render_parameters:
                    render_kwargs["use_trained_exp"] = bool(getattr(dataset, "train_test_exp", False))
                if "separate_sh" in render_parameters:
                    render_kwargs["separate_sh"] = sparse_adam_available
                unsupported = sorted(set(render_kwargs) - render_parameters)
                if unsupported:
                    raise RuntimeError(
                        f"renderer does not expose the required metric-depth API: {unsupported}; "
                        f"available={sorted(render_parameters)}"
                    )
                payload = render(
                    view,
                    gaussians,
                    pipeline,
                    background,
                    **render_kwargs,
                )
                if adapter_api == "raw_metric_depth_accumulators_v1":
                    raw_packet = payload["raw_metric_depth_accumulators"]
                    if bool(getattr(dataset, "train_test_exp", False)):
                        raw_packet = raw_packet[..., raw_packet.shape[-1] // 2 :]
                    raw_packet_np = raw_packet.detach().squeeze().cpu().numpy().astype(np.float32)
                    packet_payload = derive_packet_from_raw_accumulators(
                        raw_packet_np,
                        numerical_support_floor=float(args.numerical_support_floor),
                        variance_clamp_tolerance=float(args.variance_clamp_tolerance),
                    )
                else:
                    metric_packet = payload["metric_depth_packet"]
                    if bool(getattr(dataset, "train_test_exp", False)):
                        metric_packet = metric_packet[..., metric_packet.shape[-1] // 2 :]
                    packet_np = metric_packet.detach().squeeze().cpu().numpy().astype(np.float32)
                    if packet_np.shape[0] != len(METRIC_PACKET_TENSOR_NAMES):
                        raise RuntimeError(
                            f"metric_depth_packet expected {len(METRIC_PACKET_TENSOR_NAMES)} tensors, "
                            f"got shape {packet_np.shape}"
                        )
                    packet_payload = {
                        name: packet_np[i].astype(np.float32)
                        for i, name in enumerate(METRIC_PACKET_TENSOR_NAMES)
                    }
                packet_path = out_dir / packet_filename(image_name)
                packet_payload["metric_depth_valid_mask"] = packet_payload["metric_depth_valid_mask"] > 0.5
                packet_payload[HISTORICAL_INVALID_TENSOR] = packet_payload[
                    "weighted_inverse_camera_z_sum"
                ].copy()
                np.savez_compressed(packet_path, **packet_payload)
                packet_hash = file_sha256(packet_path)
                packet_size = packet_path.stat().st_size
                recompute = recompute_and_compare_packet(
                    packet_payload,
                    numerical_support_floor=float(args.numerical_support_floor),
                    variance_clamp_tolerance=float(args.variance_clamp_tolerance),
                    **variance_validation_manifest_fields(),
                )
                if not recompute["passed"]:
                    raise RuntimeError(f"Derived tensor recomputation failed for {image_name}: {recompute}")
                variance_row = next(row for row in recompute["rows"] if row["tensor"] == "camera_z_variance")
                rows.append(
                    {
                        "index": index,
                        "split": split,
                        "image_name": image_name,
                        "packet_path": str(packet_path),
                        "depth_path": str(packet_path),
                        "packet_sha256": packet_hash,
                        "packet_bytes": packet_size,
                        "height": int(packet_payload["accumulated_alpha"].shape[0]),
                        "width": int(packet_payload["accumulated_alpha"].shape[1]),
                        "dtype": "float32",
                        "primary_depth_tensor": PRIMARY_DEPTH_TENSOR,
                        "primary_depth_semantics": PRIMARY_DEPTH_SEMANTICS,
                        "tensor_names": "|".join(METRIC_PACKET_TENSOR_NAMES + [HISTORICAL_INVALID_TENSOR]),
                        "valid_pixel_count": int(np.count_nonzero(packet_payload["metric_depth_valid_mask"])),
                        "accumulated_alpha_min": float(np.nanmin(packet_payload["accumulated_alpha"])),
                        "accumulated_alpha_max": float(np.nanmax(packet_payload["accumulated_alpha"])),
                        "expected_camera_z_finite_count": int(np.isfinite(packet_payload[PRIMARY_DEPTH_TENSOR]).sum()),
                        "packet_recompute_passed": bool(recompute["passed"]),
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
                        "variance_max_connected_component_size": variance_row["accepted_negative_spatial_distribution"]["max_connected_component_size"],
                    }
                )
    finally:
        os.chdir(old_cwd)

    write_csv(
        mapping_path,
        rows,
        [
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
        ],
    )

    rasterizer_repo = resolve_rasterizer_repo(train_repo, args.rasterizer_repo)
    rasterizer_sources = sorted(rasterizer_repo.glob("*_rasterization/__init__.py"))
    rasterizer_sources.extend(
        path
        for path in [rasterizer_repo / "cuda_rasterizer" / "forward.cu"]
        if path.is_file()
    )
    renderer_sources = [train_repo / "gaussian_renderer" / "__init__.py", *rasterizer_sources]
    rasterizer_commit = git_commit(rasterizer_repo)
    rasterizer_tree_hash = git_tree_hash(rasterizer_repo)
    if not rasterizer_commit or not rasterizer_tree_hash:
        raise RuntimeError(f"rasterizer Git provenance is incomplete: {rasterizer_repo}")
    model_tree_hash = directory_tree_hash(Path(dataset.model_path))
    manifest: Dict[str, Any] = {
        "schema": METRIC_PACKET_MANIFEST_SCHEMA,
        "packet_schema": METRIC_PACKET_SCHEMA,
        "created_at": datetime.now().astimezone().isoformat(),
        "purpose": "Metric depth packet for P1 Gaussian GCP geometry evaluator; not a visualization artifact.",
        "train_repo": str(train_repo),
        "renderer_repository": str(train_repo),
        "renderer_commit": git_commit(train_repo),
        "rasterizer_repository": str(rasterizer_repo),
        "rasterizer_commit": rasterizer_commit,
        "rasterizer_tree_hash": rasterizer_tree_hash,
        "exporter_repository": str(Path(__file__).resolve().parents[2]),
        "exporter_commit": git_commit(Path(__file__).resolve().parents[2]),
        "source_path": str(dataset.source_path),
        "model_path": str(dataset.model_path),
        "model_content_hash": model_tree_hash,
        "iteration": int(args.iteration),
        "camera_sets": args.camera_sets,
        "depth_output_dir": str(out_dir),
        "mapping_csv": str(mapping_path),
        "depth_file_format": "compressed numpy .npz metric depth packet",
        "primary_depth_tensor": PRIMARY_DEPTH_TENSOR,
        "primary_depth_semantics": PRIMARY_DEPTH_SEMANTICS,
        "formal_depth_formula": "M1/A",
        "depth_semantics": PRIMARY_DEPTH_SEMANTICS,
        "depth_units": "model_coordinate_units_before_sim3",
        "tensor_names": METRIC_PACKET_TENSOR_NAMES + [HISTORICAL_INVALID_TENSOR],
        "tensor_formulas": packet_manifest_tensor_formulas(),
        "diagnostic_tensor_names": [DIAGNOSTIC_VARIANCE_TENSOR, DIAGNOSTIC_VARIANCE_VALID_MASK_TENSOR],
        "dtype": "float32",
        "image_domain": args.image_domain,
        "distorted_or_undistorted": "same_as_gaussian_render_camera",
        "pixel_coordinate_convention": args.pixel_coordinate_convention,
        "protocol_id": args.protocol_id,
        "scene": args.protocol_scene,
        "source_data_release_root_digest_sha256": args.source_data_release_root_digest_sha256,
        "camera_z_unit_contract": args.camera_z_unit_contract,
        "adapter_conformance_status": args.adapter_conformance_status,
        "adapter_conformance_report": args.adapter_conformance_report,
        "adapter_conformance_report_sha256": args.adapter_conformance_report_sha256,
        "camera_model_source": "Gaussian Scene/COLMAP camera loaded by training repository",
        "alpha_cutoff": DEFAULT_ALPHA_CUTOFF,
        "early_termination_threshold": DEFAULT_EARLY_TERMINATION_THRESHOLD,
        "numerical_support_floor": float(args.numerical_support_floor),
        "normalization_epsilon": float(args.normalization_epsilon),
        "variance_clamp_tolerance": float(args.variance_clamp_tolerance),
        **variance_validation_manifest_fields(),
        "depth_semantics_note": (
            "Primary formal P1 depth is alpha_normalized_expected_camera_z=M1/A for valid A. "
            "The old renderer payload depth is preserved only as historical_invalid_unnormalized_inverse_depth. "
            "alpha_cutoff and early_termination_threshold record fixed rasterizer behavior in this protocol; "
            "they are not exporter CLI knobs."
        ),
        "alpha_map_available": True,
        "depth_second_moment_available": True,
        "depth_scale_for_evaluator": 1.0,
        "depth_offset_for_evaluator": 0.0,
        "rendered_view_count": len(rows),
        "depth_index": rows,
        "packet_index": rows,
        "renderer_source_trace": [
            {
                "path": str(path),
                "sha256": file_sha256(path) if path.exists() else "",
                "exists": path.exists(),
            }
            for path in renderer_sources
        ],
        "rasterizer_source_trace": [
            {
                "path": str(path),
                "sha256": file_sha256(path) if path.exists() else "",
                "exists": path.exists(),
            }
            for path in rasterizer_sources
        ],
        "image_list_csv": str(Path(args.image_list_csv).expanduser().resolve()) if args.image_list_csv else "",
        "image_name_column": args.image_name_column,
        "image_list_status_column": args.image_list_status_column,
        "image_list_status_values": args.image_list_status_values,
        "sparse_adam_available": bool(sparse_adam_available),
        "renderer_render_parameters": sorted(render_parameters),
        "renderer_adapter_api": adapter_api,
        "raw_accumulator_tensor_names": list(RAW_ACCUMULATOR_TENSOR_NAMES),
        "derived_packet_computed_on_cpu": adapter_api == "raw_metric_depth_accumulators_v1",
        "adapter_patch_files": [
            {
                "path": str(Path(path).expanduser().resolve()),
                "sha256": file_sha256(Path(path).expanduser().resolve()),
            }
            for path in (args.renderer_adapter_patch, args.rasterizer_adapter_patch)
            if path
        ],
        "historical_invalid_tensor_source": "weighted_inverse_camera_z_sum raw H alias",
        "uses_alpha_map": True,
        "uses_depth_second_moment": True,
        "runtime": {
            "python": sys.version,
            "torch": str(getattr(torch, "__version__", "")),
            "torch_cuda": str(getattr(getattr(torch, "version", None), "cuda", "")),
            "cuda_available": bool(torch.cuda.is_available()),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        },
        "notes": [
            "No checkpoint mutation, no retraining, and no support modification.",
            "Packet arrays are saved as linear float/bool data; PNG displays must not be used for metric evaluation.",
            "historical_invalid_unnormalized_inverse_depth must not enter formal P1 ranking or camera_z=1/depth backprojection.",
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def build_parser(runtime: Dict[str, Any]) -> tuple[argparse.ArgumentParser, Any, Any]:
    parser = argparse.ArgumentParser(description="Export metric-depth packets for the GS-GCP evaluator.")
    ModelParams = runtime["ModelParams"]
    PipelineParams = runtime["PipelineParams"]
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--train_repo", default=str(runtime["train_repo"]), help="Path to the Gaussian training/rendering repository.")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--camera_sets", default="all", choices=["all", "train", "test"])
    parser.add_argument("--depth_output_dir", required=True)
    parser.add_argument("--manifest_path", default="")
    parser.add_argument("--mapping_csv", default="")
    parser.add_argument("--numerical_support_floor", type=float, default=DEFAULT_NUMERICAL_SUPPORT_FLOOR)
    parser.add_argument("--normalization_epsilon", type=float, default=DEFAULT_NORMALIZATION_EPSILON)
    parser.add_argument("--variance_clamp_tolerance", type=float, default=DEFAULT_VARIANCE_CLAMP_TOLERANCE)
    parser.add_argument("--image_list_csv", default="", help="Optional CSV that restricts export to listed image names.")
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
    parser.add_argument(
        "--rasterizer_repo",
        default="",
        help="Rasterizer Git repository inside train_repo; inferred for official 3DGS or 2DGS layouts when omitted.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser, model, pipeline


def main() -> None:
    train_repo = parse_train_repo(sys.argv[1:])
    runtime = load_gaussian_runtime(train_repo)
    parser, model, pipeline = build_parser(runtime)
    args = runtime["get_combined_args"](parser)
    runtime["safe_state"](args.quiet)
    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    manifest = export_depths(args, dataset, pipe, runtime)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
