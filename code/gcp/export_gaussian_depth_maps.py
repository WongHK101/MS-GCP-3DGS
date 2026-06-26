from __future__ import annotations

import argparse
import csv
import json
import os
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
    tensor_stats,
    variance_validation_manifest_fields,
)

DEFAULT_TRAIN_REPO = r"E:\Multispectral" if Path(r"E:\Multispectral").exists() else "/root/autodl-tmp/Multispectral"
DEFAULT_RASTERIZER_DEPTH_SEMANTICS = "alpha_weighted_unnormalized_inverse_camera_z"


def parse_train_repo(argv: Sequence[str]) -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--train_repo", default=DEFAULT_TRAIN_REPO)
    known, _unknown = parser.parse_known_args(argv)
    return Path(known.train_repo).expanduser().resolve()


def load_gaussian_runtime(train_repo: Path) -> Dict[str, Any]:
    if not train_repo.exists():
        raise FileNotFoundError(f"training repository not found: {train_repo}")
    sys.path.insert(0, str(train_repo))

    import torch  # noqa: WPS433
    from arguments import ModelParams, PipelineParams, get_combined_args  # noqa: WPS433
    from gaussian_renderer import GaussianModel, render  # noqa: WPS433
    from scene import Scene  # noqa: WPS433
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
    }


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_allowlist(args: argparse.Namespace) -> set[str] | None:
    if not args.image_list_csv:
        return None
    path = Path(args.image_list_csv).expanduser().resolve()
    names: set[str] = set()
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
                names.add(name)
                names.add(Path(name).name)
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


def collect_views(scene: Any, camera_sets: str, allowlist: set[str] | None = None) -> List[tuple[str, Any]]:
    views: List[tuple[str, Any]] = []
    if camera_sets in {"train", "all"}:
        views.extend(("train", view) for view in scene.getTrainCameras())
    if camera_sets in {"test", "all"}:
        views.extend(("test", view) for view in scene.getTestCameras())

    seen: set[str] = set()
    unique: List[tuple[str, Any]] = []
    for split, view in views:
        name = camera_name(view)
        if allowlist is not None and name not in allowlist and Path(name).name not in allowlist:
            continue
        if name in seen:
            continue
        seen.add(name)
        unique.append((split, view))
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
            for index, (split, view) in enumerate(tqdm(views, desc="Exporting Gaussian depth")):
                payload = render(
                    view,
                    gaussians,
                    pipeline,
                    background,
                    use_trained_exp=dataset.train_test_exp,
                    separate_sh=sparse_adam_available,
                    return_metric_depth_packet=True,
                    numerical_support_floor=float(args.numerical_support_floor),
                    normalization_epsilon=float(args.normalization_epsilon),
                    variance_clamp_tolerance=float(args.variance_clamp_tolerance),
                )
                historical_depth = payload["depth"]
                metric_packet = payload["metric_depth_packet"]
                if dataset.train_test_exp:
                    historical_depth = historical_depth[..., historical_depth.shape[-1] // 2 :]
                    metric_packet = metric_packet[..., metric_packet.shape[-1] // 2 :]
                historical_depth_np = historical_depth.detach().squeeze().cpu().numpy().astype(np.float32)
                packet_np = metric_packet.detach().squeeze().cpu().numpy().astype(np.float32)
                if packet_np.shape[0] != len(METRIC_PACKET_TENSOR_NAMES):
                    raise RuntimeError(
                        f"metric_depth_packet expected {len(METRIC_PACKET_TENSOR_NAMES)} tensors, "
                        f"got shape {packet_np.shape}"
                    )
                image_name = camera_name(view)
                packet_path = out_dir / packet_filename(image_name)
                packet_payload = {
                    name: packet_np[i].astype(np.float32)
                    for i, name in enumerate(METRIC_PACKET_TENSOR_NAMES)
                }
                packet_payload["metric_depth_valid_mask"] = packet_payload["metric_depth_valid_mask"] > 0.5
                packet_payload[HISTORICAL_INVALID_TENSOR] = historical_depth_np.astype(np.float32)
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
                        "height": int(packet_np.shape[1]),
                        "width": int(packet_np.shape[2]),
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
                        "variance_validation_failing_pixel_count": variance_row["failing_pixel_count"],
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
            "variance_validation_failing_pixel_count",
        ],
    )

    renderer_sources = [
        train_repo / "gaussian_renderer" / "__init__.py",
        train_repo / "submodules" / "diff-gaussian-rasterization" / "diff_gaussian_rasterization" / "__init__.py",
        train_repo / "submodules" / "diff-gaussian-rasterization" / "cuda_rasterizer" / "forward.cu",
    ]
    rasterizer_repo = train_repo / "submodules" / "diff-gaussian-rasterization"
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
        "rasterizer_commit": git_commit(rasterizer_repo) or git_commit(train_repo),
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
        "depth_semantics": PRIMARY_DEPTH_SEMANTICS,
        "tensor_names": METRIC_PACKET_TENSOR_NAMES + [HISTORICAL_INVALID_TENSOR],
        "tensor_formulas": packet_manifest_tensor_formulas(),
        "dtype": "float32",
        "image_domain": "rendered_colmap_camera_domain",
        "distorted_or_undistorted": "same_as_gaussian_render_camera",
        "pixel_coordinate_convention": "zero_indexed_pixel_centers",
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
        "image_list_csv": str(Path(args.image_list_csv).expanduser().resolve()) if args.image_list_csv else "",
        "image_name_column": args.image_name_column,
        "image_list_status_column": args.image_list_status_column,
        "image_list_status_values": args.image_list_status_values,
        "sparse_adam_available": bool(sparse_adam_available),
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
    parser = argparse.ArgumentParser(description="Export float Gaussian-rendered depth maps for MS-GCP-3DGS P1 evaluator.")
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
