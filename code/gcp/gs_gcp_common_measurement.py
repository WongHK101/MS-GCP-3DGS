#!/usr/bin/env python3
"""Post-hoc GS-GCP RGB metrics, render timing, and representation inspection."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from statistics import median
from types import SimpleNamespace
from typing import Any, Callable, Iterable


SUITE_ID = "gs_gcp_common_measurement_suite_v1"
RESOLUTION_PROTOCOL = "graphdeco_quarter_resolution_v1"
HOLDOUT_SEMANTICS = "image_loss_holdout_under_shared_all_image_sfm_v1"
REFERENCE_GPU_UUID = "GPU-b5804bf4-ec7a-06c5-eda6-a896ab721251"
EXPECTED_VGG16_SHA256 = "397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0"
EXPECTED_LPIPS_SHA256 = "a78928a0af1e5f0fcb1f3b9e8f8c3a2a5a3de244d830ad5c1feddc79b8432868"
TIMED_ROUNDS = 5
WARMUP_ROUNDS = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def _percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo = math.floor(position)
    hi = math.ceil(position)
    if lo == hi:
        return ordered[lo]
    weight = position - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def _image_map(path: Path) -> dict[str, Path]:
    if not path.is_dir():
        raise FileNotFoundError(path)
    result = {}
    for item in path.iterdir():
        if not item.is_file() or item.suffix.lower() != ".png":
            continue
        key = item.stem
        if key in result:
            raise ValueError(f"duplicate PNG stem: {key}")
        result[key] = item
    return result


def validate_rgb_image_set(render_dir: Path, gt_manifest: dict[str, Any], gt_root: Path) -> dict[str, Any]:
    expected = {Path(row["image_name"]).stem: row for row in gt_manifest.get("images", [])}
    actual = _image_map(render_dir)
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    gt_errors = []
    for key, row in expected.items():
        gt_path = gt_root / row["gt_relative_path"]
        if not gt_path.is_file() or sha256_file(gt_path) != row["gt_png_sha256"]:
            gt_errors.append(key)
    return {
        "passed": not missing and not extra and not gt_errors,
        "expected_count": len(expected),
        "render_count": len(actual),
        "missing_render_names": missing,
        "extra_render_names": extra,
        "invalid_gt_names": gt_errors,
    }


def run_rgb_metrics(
    *,
    render_dir: Path,
    gt_root: Path,
    gt_manifest_path: Path,
    method_root: Path,
    output_dir: Path,
    device: str,
    vgg16_weights: Path,
    lpips_weights: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if sha256_file(vgg16_weights) != EXPECTED_VGG16_SHA256:
        raise ValueError("VGG16 weight SHA mismatch")
    if sha256_file(lpips_weights) != EXPECTED_LPIPS_SHA256:
        raise ValueError("LPIPS linear weight SHA mismatch")
    gt_manifest = json.loads(gt_manifest_path.read_text(encoding="utf-8"))
    image_set = validate_rgb_image_set(render_dir, gt_manifest, gt_root)
    output_dir.mkdir(parents=True)
    if not image_set["passed"]:
        summary = {
            "schema": "gs_gcp_rgb_metric_summary_v1",
            "status": "INCOMPLETE_COVERAGE",
            "suite_id": SUITE_ID,
            "image_set": image_set,
            "primary_scene_mean_available": False,
        }
        write_json(output_dir / "rgb_metric_summary.json", summary)
        return summary
    sys.path.insert(0, str(method_root.resolve()))
    try:
        import torch
        from PIL import Image
        from torchvision.transforms.functional import to_tensor
        from utils.image_utils import psnr
        from utils.loss_utils import ssim
        from lpipsPyTorch import lpips
    finally:
        pass
    expected = {Path(row["image_name"]).stem: row for row in gt_manifest["images"]}
    renders = _image_map(render_dir)
    per_image = []
    failures = []
    with torch.no_grad():
        for name in sorted(expected):
            row = expected[name]
            render_path = renders[name]
            gt_path = gt_root / row["gt_relative_path"]
            try:
                with Image.open(render_path) as image:
                    render = to_tensor(image).unsqueeze(0)[:, :3, :, :].to(device)
                with Image.open(gt_path) as image:
                    gt = to_tensor(image).unsqueeze(0)[:, :3, :, :].to(device)
                if tuple(render.shape) != tuple(gt.shape):
                    raise ValueError(f"shape mismatch {tuple(render.shape)} != {tuple(gt.shape)}")
                values = {
                    "PSNR": float(psnr(render, gt).mean().item()),
                    "SSIM": float(ssim(render, gt).item()),
                    "LPIPS_VGG": float(lpips(render, gt, net_type="vgg").item()),
                }
                if not all(math.isfinite(value) for value in values.values()):
                    raise ValueError("metric is non-finite")
                per_image.append({
                    "image_name": row["image_name"],
                    "render_sha256": sha256_file(render_path),
                    "gt_sha256": row["gt_png_sha256"],
                    "width": int(render.shape[-1]),
                    "height": int(render.shape[-2]),
                    **values,
                })
            except Exception as exc:  # noqa: BLE001
                failures.append({"image_name": row["image_name"], "reason": f"{type(exc).__name__}: {exc}"})
    metrics = ("PSNR", "SSIM", "LPIPS_VGG")
    successful_mean = {
        key: math.fsum(float(row[key]) for row in per_image) / len(per_image) if per_image else None
        for key in metrics
    }
    complete = len(per_image) == len(expected) and not failures
    summary = {
        "schema": "gs_gcp_rgb_metric_summary_v1",
        "status": "PASS" if complete else "INCOMPLETE_COVERAGE",
        "suite_id": SUITE_ID,
        "resolution_protocol": RESOLUTION_PROTOCOL,
        "holdout_semantics": HOLDOUT_SEMANTICS,
        "rgb_domain": "sRGB float [0,1] decoded from benchmark-owned PNG",
        "expected_count": len(expected),
        "successful_count": len(per_image),
        "failed_count": len(failures),
        "primary_scene_mean_available": complete,
        "primary_scene_mean": successful_mean if complete else None,
        "successful_only_diagnostic_mean": successful_mean,
        "failures": failures,
        "vgg16_weight_sha256": EXPECTED_VGG16_SHA256,
        "lpips_vgg_linear_weight_sha256": EXPECTED_LPIPS_SHA256,
        "device": device,
    }
    write_json(output_dir / "per_image_metrics.json", per_image)
    write_json(output_dir / "rgb_metric_summary.json", summary)
    return summary


def _nvidia_gpu_uuid(index: int) -> str:
    output = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu=uuid", "--format=csv,noheader", "-i", str(index)],
        text=True,
    )
    return output.strip()


def _load_cfg_args(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8").strip()
    expression = ast.parse(text, mode="eval").body
    if isinstance(expression, ast.Call) and isinstance(expression.func, ast.Name) and expression.func.id == "Namespace":
        if expression.args:
            raise ValueError("cfg_args Namespace must use keyword arguments")
        return {keyword.arg: ast.literal_eval(keyword.value) for keyword in expression.keywords}
    value = ast.literal_eval(expression)
    if isinstance(value, dict):
        return dict(value)
    raise ValueError("unsupported cfg_args payload")


def _load_original_3dgs_scene(method_root: Path, model_path: Path, test_source: Path, iteration: int):
    sys.path.insert(0, str(method_root.resolve()))
    import torch
    from gaussian_renderer import render
    from scene import GaussianModel, Scene
    cfg = _load_cfg_args(model_path / "cfg_args")
    dataset = SimpleNamespace(
        sh_degree=int(cfg.get("sh_degree", 3)), source_path=str(test_source.resolve()),
        model_path=str(model_path.resolve()), images="images", resolution=4,
        white_background=bool(cfg.get("white_background", False)), data_device="cuda", eval=False,
    )
    pipeline = SimpleNamespace(
        convert_SHs_python=bool(cfg.get("convert_SHs_python", False)),
        compute_cov3D_python=bool(cfg.get("compute_cov3D_python", False)), debug=False,
    )
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    cameras = scene.getTrainCameras()
    if not cameras:
        raise ValueError("test camera subset is empty")
    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32, device="cuda",
    )
    return torch, render, cameras, gaussians, pipeline, background


def render_original_3dgs_holdout(
    *, method_root: Path, model_path: Path, test_source: Path, iteration: int, output_dir: Path
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    render_dir = output_dir / "renders"
    render_dir.mkdir()
    torch, render, cameras, gaussians, pipeline, background = _load_original_3dgs_scene(
        method_root, model_path, test_source, iteration
    )
    from torchvision.utils import save_image
    records = []
    with torch.no_grad():
        for camera in cameras:
            image = render(camera, gaussians, pipeline, background)["render"]
            if not torch.isfinite(image).all():
                raise ValueError(f"non-finite render: {camera.image_name}")
            target = render_dir / f"{camera.image_name}.png"
            save_image(image, target)
            records.append({
                "image_name": f"{camera.image_name}.JPG",
                "render_relative_path": target.relative_to(output_dir).as_posix(),
                "render_sha256": sha256_file(target), "render_bytes": target.stat().st_size,
                "decoded_width": int(image.shape[-1]), "decoded_height": int(image.shape[-2]),
            })
    manifest = {
        "schema": "gs_gcp_original_3dgs_holdout_render_manifest_v1", "status": "PASS",
        "suite_id": SUITE_ID, "resolution_protocol": RESOLUTION_PROTOCOL,
        "holdout_semantics": HOLDOUT_SEMANTICS, "iteration": iteration,
        "image_count": len(records), "images": records,
        "png_write": "torchvision.utils.save_image", "secondary_resize_or_crop": False,
    }
    write_json(output_dir / "RENDER_MANIFEST.json", manifest)
    return manifest


def prepare_original_3dgs_evaluation_model(
    trained_model: Path, evaluation_model: Path, full_source: Path
) -> dict[str, Any]:
    if evaluation_model.exists():
        raise FileExistsError(evaluation_model)
    point_cloud = trained_model / "point_cloud"
    cfg_path = trained_model / "cfg_args"
    if not point_cloud.is_dir() or not cfg_path.is_file():
        raise FileNotFoundError("trained model lacks point_cloud or cfg_args")
    evaluation_model.mkdir(parents=True)
    if os.name == "nt":
        shutil.copytree(point_cloud, evaluation_model / "point_cloud")
        link_mode = "byte_copy_for_windows_test_portability"
    else:
        (evaluation_model / "point_cloud").symlink_to(point_cloud.resolve(), target_is_directory=True)
        link_mode = "read_only_symlink"
    cfg = _load_cfg_args(cfg_path)
    cfg.update({
        "source_path": str(full_source.resolve()),
        "model_path": str(evaluation_model.resolve()),
        "images": "images",
        "resolution": 4,
        "data_device": "cuda",
        "eval": False,
    })
    serialized = "Namespace(" + ", ".join(f"{key}={cfg[key]!r}" for key in sorted(cfg)) + ")"
    (evaluation_model / "cfg_args").write_text(serialized, encoding="utf-8", newline="\n")
    manifest = {
        "schema": "gs_gcp_original_3dgs_read_only_evaluation_model_adapter_v1",
        "trained_model": str(trained_model.resolve()),
        "trained_point_cloud_sha256": sha256_file(point_cloud / "iteration_30000" / "point_cloud.ply"),
        "evaluation_model": str(evaluation_model.resolve()),
        "full_camera_source": str(full_source.resolve()),
        "resolution": 4,
        "point_cloud_link_target": str((evaluation_model / "point_cloud").resolve()),
        "point_cloud_materialization": link_mode,
        "checkpoint_mutated": False,
        "cameras_json_expected_to_be_generated_from_full_source_by_export_adapter": True,
        "cfg_args_sha256": sha256_file(evaluation_model / "cfg_args"),
    }
    write_json(evaluation_model / "EVALUATION_ADAPTER_MANIFEST.json", manifest)
    return manifest


def benchmark_original_3dgs_render(
    *, method_root: Path, model_path: Path, test_source: Path, iteration: int, gpu_index: int, output: Path
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    if _nvidia_gpu_uuid(gpu_index) != REFERENCE_GPU_UUID:
        raise RuntimeError("single-GPU reference render must use the frozen 901 GPU UUID")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    torch, render, cameras, gaussians, pipeline, background = _load_original_3dgs_scene(
        method_root, model_path, test_source, iteration
    )
    with torch.no_grad():
        completion = []
        for _ in range(WARMUP_ROUNDS):
            for camera in cameras:
                render(camera, gaussians, pipeline, background)["render"]
            torch.cuda.synchronize()
        per_round = []
        all_latencies = []
        for round_index in range(TIMED_ROUNDS):
            events = []
            torch.cuda.synchronize()
            round_start = time.perf_counter()
            for camera in cameras:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                rendered = render(camera, gaussians, pipeline, background)["render"]
                end.record()
                if not torch.isfinite(rendered).all():
                    raise ValueError(f"non-finite render: {camera.image_name}")
                events.append((camera.image_name, start, end))
            torch.cuda.synchronize()
            round_wall = time.perf_counter() - round_start
            latencies = [float(start.elapsed_time(end)) for _, start, end in events]
            all_latencies.extend(latencies)
            completion.extend(name for name, _, _ in events)
            per_round.append({
                "round": round_index,
                "view_count": len(events),
                "wall_seconds": round_wall,
                "event_sum_seconds": math.fsum(latencies) / 1000.0,
                "throughput_fps": len(events) / (math.fsum(latencies) / 1000.0),
                "median_latency_ms": median(latencies),
                "p90_latency_ms": _percentile(latencies, 0.9),
            })
    event_sum_seconds = math.fsum(all_latencies) / 1000.0
    report = {
        "schema": "gs_gcp_single_gpu_reference_render_v1",
        "status": "PASS",
        "suite_id": SUITE_ID,
        "mode": "single_gpu_reference_render",
        "gpu_uuid": REFERENCE_GPU_UUID,
        "gpu_count": 1,
        "batch_size": 1,
        "warmup_rounds": WARMUP_ROUNDS,
        "timed_rounds": TIMED_ROUNDS,
        "camera_count": len(cameras),
        "timed_render_count": len(all_latencies),
        "completion_count": len(completion),
        "median_latency_ms": median(all_latencies),
        "p90_latency_ms": _percentile(all_latencies, 0.9),
        "throughput_fps": len(all_latencies) / event_sum_seconds,
        "rounds": per_round,
        "model_load_included": False,
        "disk_io_included": False,
    }
    write_json(output, report)
    return report


def inspect_original_3dgs_representation(model_path: Path, iteration: int) -> dict[str, Any]:
    ply = model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    if not ply.is_file():
        raise FileNotFoundError(ply)
    header = []
    with ply.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                raise ValueError("PLY header is truncated")
            decoded = line.decode("ascii").rstrip("\r\n")
            header.append(decoded)
            if decoded == "end_header":
                break
    vertex_count = None
    properties = []
    in_vertex = False
    for line in header:
        if line.startswith("element vertex "):
            vertex_count = int(line.split()[-1])
            in_vertex = True
        elif line.startswith("element "):
            in_vertex = False
        elif in_vertex and line.startswith("property "):
            properties.append(line)
    if vertex_count is None:
        raise ValueError("PLY has no vertex element")
    runtime_files = []
    for path in sorted(item for item in model_path.rglob("*") if item.is_file()):
        runtime_files.append({
            "relative_path": path.relative_to(model_path).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    exposure = model_path / "exposure.json"
    exposure_bytes = exposure.stat().st_size if exposure.is_file() else 0
    report = {
        "schema": "gs_gcp_original_3dgs_representation_summary_v1",
        "status": "PASS",
        "primitive_type": "3D Gaussian",
        "gaussian_count": vertex_count,
        "gaussian_order": "serialized PLY vertex order",
        "property_schema": properties,
        "serialized_scalar_count": vertex_count * len(properties),
        "sh_degree": 3,
        "point_cloud_ply": ply.relative_to(model_path).as_posix(),
        "point_cloud_ply_bytes": ply.stat().st_size,
        "point_cloud_ply_sha256": sha256_file(ply),
        "exposure_bytes": exposure_bytes,
        "deployable_model_bytes": sum(row["bytes"] for row in runtime_files),
        "runtime_file_count": len(runtime_files),
        "runtime_files": runtime_files,
        "serializer_patch_commit": "db8deebca67e8d5e1507e67c98de603eca0dfd85",
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    metrics = sub.add_parser("rgb-metrics")
    metrics.add_argument("--render_dir", type=Path, required=True)
    metrics.add_argument("--gt_root", type=Path, required=True)
    metrics.add_argument("--gt_manifest", type=Path, required=True)
    metrics.add_argument("--method_root", type=Path, required=True)
    metrics.add_argument("--output_dir", type=Path, required=True)
    metrics.add_argument("--device", default="cuda")
    metrics.add_argument("--vgg16_weights", type=Path, required=True)
    metrics.add_argument("--lpips_weights", type=Path, required=True)
    benchmark = sub.add_parser("render-benchmark")
    benchmark.add_argument("--method_root", type=Path, required=True)
    benchmark.add_argument("--model_path", type=Path, required=True)
    benchmark.add_argument("--test_source", type=Path, required=True)
    benchmark.add_argument("--iteration", type=int, default=30000)
    benchmark.add_argument("--gpu_index", type=int, default=0)
    benchmark.add_argument("--output", type=Path, required=True)
    heldout = sub.add_parser("render-heldout")
    heldout.add_argument("--method_root", type=Path, required=True)
    heldout.add_argument("--model_path", type=Path, required=True)
    heldout.add_argument("--test_source", type=Path, required=True)
    heldout.add_argument("--iteration", type=int, default=30000)
    heldout.add_argument("--output_dir", type=Path, required=True)
    inspect = sub.add_parser("inspect-3dgs")
    inspect.add_argument("--model_path", type=Path, required=True)
    inspect.add_argument("--iteration", type=int, default=30000)
    inspect.add_argument("--output", type=Path, required=True)
    prepare = sub.add_parser("prepare-eval-model")
    prepare.add_argument("--trained_model", type=Path, required=True)
    prepare.add_argument("--evaluation_model", type=Path, required=True)
    prepare.add_argument("--full_source", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "rgb-metrics":
        payload = run_rgb_metrics(
            render_dir=args.render_dir.resolve(), gt_root=args.gt_root.resolve(),
            gt_manifest_path=args.gt_manifest.resolve(), method_root=args.method_root.resolve(),
            output_dir=args.output_dir.resolve(), device=args.device,
            vgg16_weights=args.vgg16_weights.resolve(), lpips_weights=args.lpips_weights.resolve(),
        )
    elif args.command == "render-benchmark":
        payload = benchmark_original_3dgs_render(
            method_root=args.method_root.resolve(), model_path=args.model_path.resolve(),
            test_source=args.test_source.resolve(), iteration=args.iteration,
            gpu_index=args.gpu_index, output=args.output.resolve(),
        )
    elif args.command == "render-heldout":
        payload = render_original_3dgs_holdout(
            method_root=args.method_root.resolve(), model_path=args.model_path.resolve(),
            test_source=args.test_source.resolve(), iteration=args.iteration,
            output_dir=args.output_dir.resolve(),
        )
    elif args.command == "prepare-eval-model":
        payload = prepare_original_3dgs_evaluation_model(
            args.trained_model.resolve(), args.evaluation_model.resolve(), args.full_source.resolve()
        )
    else:
        payload = inspect_original_3dgs_representation(args.model_path.resolve(), args.iteration)
        if args.output.exists():
            raise FileExistsError(args.output)
        write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
