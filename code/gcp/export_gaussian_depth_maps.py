from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
from tqdm import tqdm


DEFAULT_TRAIN_REPO = r"E:\Multispectral" if Path(r"E:\Multispectral").exists() else "/root/autodl-tmp/Multispectral"
DEFAULT_RASTERIZER_DEPTH_SEMANTICS = "alpha_weighted_unnormalized_inverse_camera_z"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


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
                )
                depth = payload["depth"]
                if dataset.train_test_exp:
                    depth = depth[..., depth.shape[-1] // 2 :]
                depth_np = depth.detach().squeeze().cpu().numpy().astype(np.float32)
                image_name = camera_name(view)
                depth_path = out_dir / depth_filename(image_name)
                np.save(depth_path, depth_np)
                rows.append(
                    {
                        "index": index,
                        "split": split,
                        "image_name": image_name,
                        "depth_path": str(depth_path),
                        "height": int(depth_np.shape[0]),
                        "width": int(depth_np.shape[1]),
                        "finite_count": int(np.isfinite(depth_np).sum()),
                        "min_depth": float(np.nanmin(depth_np)) if np.isfinite(depth_np).any() else "",
                        "max_depth": float(np.nanmax(depth_np)) if np.isfinite(depth_np).any() else "",
                        "median_depth": float(np.nanmedian(depth_np)) if np.isfinite(depth_np).any() else "",
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
            "depth_path",
            "height",
            "width",
            "finite_count",
            "min_depth",
            "max_depth",
            "median_depth",
        ],
    )

    renderer_sources = [
        train_repo / "gaussian_renderer" / "__init__.py",
        train_repo / "submodules" / "diff-gaussian-rasterization" / "diff_gaussian_rasterization" / "__init__.py",
        train_repo / "submodules" / "diff-gaussian-rasterization" / "cuda_rasterizer" / "forward.cu",
    ]
    manifest: Dict[str, Any] = {
        "schema": "ms_gcp_gaussian_depth_export_v1_1",
        "created_at": datetime.now().astimezone().isoformat(),
        "purpose": "Depth-only P1 Gaussian GCP geometry evaluator input; not a visualization artifact.",
        "train_repo": str(train_repo),
        "renderer_repository": str(train_repo),
        "renderer_commit": git_commit(train_repo),
        "source_path": str(dataset.source_path),
        "model_path": str(dataset.model_path),
        "iteration": int(args.iteration),
        "camera_sets": args.camera_sets,
        "depth_output_dir": str(out_dir),
        "mapping_csv": str(mapping_path),
        "depth_file_format": "float32 numpy .npy",
        "depth_semantics": args.depth_semantics,
        "depth_units": (
            "alpha_transmittance_weighted_1/metre"
            if args.depth_semantics == DEFAULT_RASTERIZER_DEPTH_SEMANTICS
            else ("1/metre" if args.depth_semantics.endswith("inverse_camera_z") else "metre")
        ),
        "image_domain": "rendered_colmap_camera_domain",
        "depth_semantics_note": (
            "The current Graphdeco-derived rasterizer output used by gaussian_renderer.render(...)[\"depth\"] "
            "is the CUDA variable expected_invdepth = sum_j (1 / camera_z_j) * alpha_j * T_j. "
            "It is not normalized by accumulated opacity/weight. Formal GCP evaluation must not convert "
            "this artifact with camera_z=1/depth unless a future exporter also provides and applies the "
            "corresponding alpha/weight normalization, or exports camera_z/ray_distance directly."
        ),
        "depth_accumulation_formula": "D(p)=sum_j alpha_j(p) T_j(p) / z_j, where z_j is camera-space z",
        "alpha_map_available": False,
        "depth_second_moment_available": False,
        "depth_scale_for_evaluator": 1.0,
        "depth_offset_for_evaluator": 0.0,
        "rendered_view_count": len(rows),
        "depth_index": rows,
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
        "uses_alpha_map": False,
        "uses_depth_second_moment": False,
        "notes": [
            "No checkpoint mutation, no retraining, and no support modification.",
            "Depth arrays are saved as linear float data; PNG displays must not be used for metric evaluation.",
            "This manifest intentionally labels the current renderer output as unsupported for formal depth-only GCP evaluation.",
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
    parser.add_argument(
        "--depth_semantics",
        default=DEFAULT_RASTERIZER_DEPTH_SEMANTICS,
        choices=[
            "camera_z",
            "ray_distance",
            "inverse_camera_z",
            "inverse_ray_distance",
            DEFAULT_RASTERIZER_DEPTH_SEMANTICS,
        ],
    )
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
