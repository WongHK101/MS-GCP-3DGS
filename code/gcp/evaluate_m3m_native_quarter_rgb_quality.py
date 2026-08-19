#!/usr/bin/env python3
"""Compute one shared PSNR/SSIM/LPIPS-VGG suite from adapter PNGs."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rgb_quality_contract import (  # noqa: E402
    RENDER_MANIFEST_SCHEMA,
    SUITE_ID,
    arithmetic_mean,
    load_bound_input_manifest,
    load_contract,
    role_rows,
    sha256_file,
    write_json,
)


SUMMARY_SCHEMA = "m3m_gcp_native_quarter_rgb_quality_summary_v1"
EVALUATOR_MANIFEST_SCHEMA = "m3m_gcp_native_quarter_rgb_evaluator_manifest_v1"


def _safe_relative(base: Path, relative: str) -> Path:
    path = (base / relative).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError(f"path escapes manifest root: {relative}")
    return path


def validate_render_and_ground_truth(
    *,
    contract_path: Path,
    input_manifest_path: Path,
    input_root: Path,
    render_manifest_path: Path,
    scene: str,
    method_id: str,
    allow_review_candidate: bool,
) -> dict[str, Any]:
    contract_path = contract_path.expanduser().resolve()
    input_manifest_path = input_manifest_path.expanduser().resolve()
    input_root = input_root.expanduser().resolve()
    render_manifest_path = render_manifest_path.expanduser().resolve()
    contract = load_contract(
        contract_path, allow_review_candidate=allow_review_candidate
    )
    if not allow_review_candidate and contract["status"] != "ACTIVE_FROZEN":
        raise ValueError("formal RGB evaluation requires ACTIVE_FROZEN contract")
    input_manifest = load_bound_input_manifest(contract, input_manifest_path, scene)
    render_manifest = json.loads(render_manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []

    expected_header = {
        "schema": RENDER_MANIFEST_SCHEMA,
        "suite_id": SUITE_ID,
        "contract_status": contract["status"],
        "scene": scene,
        "method_id": method_id,
        "contract_file_sha256": sha256_file(contract_path),
        "input_manifest_file_sha256": sha256_file(input_manifest_path),
        "input_manifest_canonical_sha256": input_manifest["manifest_sha256"],
        "source_data_release_root_digest_sha256": input_manifest[
            "release_root_digest_sha256"
        ],
        "pixel_domain": input_manifest["pixel_domain"],
        "holdout_semantics": input_manifest["holdout_semantics"],
        "complete_test_coverage": True,
        "prediction_encoding": contract["prediction_contract"],
    }
    for key, expected in expected_header.items():
        if render_manifest.get(key) != expected:
            errors.append(
                f"render manifest {key} mismatch: {render_manifest.get(key)!r} != {expected!r}"
            )

    expected_rows = role_rows(input_manifest, "test")
    expected_by_name = {str(row["image_name"]): row for row in expected_rows}
    rendered_rows = render_manifest.get("renders", [])
    actual_by_name: dict[str, dict[str, Any]] = {}
    for row in rendered_rows:
        name = str(row.get("image_name", ""))
        if name in actual_by_name:
            errors.append(f"duplicate render manifest image: {name}")
        actual_by_name[name] = row
    missing = sorted(set(expected_by_name) - set(actual_by_name))
    extra = sorted(set(actual_by_name) - set(expected_by_name))
    if missing:
        errors.append(f"missing renders: {missing}")
    if extra:
        errors.append(f"extra renders: {extra}")
    if render_manifest.get("required_test_view_count") != len(expected_rows):
        errors.append("required_test_view_count mismatch")
    if render_manifest.get("rendered_test_view_count") != len(rendered_rows):
        errors.append("rendered_test_view_count mismatch")

    validated: list[dict[str, Any]] = []
    for expected in expected_rows:
        name = str(expected["image_name"])
        row = actual_by_name.get(name)
        if row is None:
            continue
        try:
            prediction = _safe_relative(
                render_manifest_path.parent, str(row["prediction_relative_path"])
            )
            ground_truth = _safe_relative(input_root, str(expected["relative_path"]))
            if not prediction.is_file():
                raise FileNotFoundError(prediction)
            if not ground_truth.is_file():
                raise FileNotFoundError(ground_truth)
            prediction_sha = sha256_file(prediction)
            gt_sha = sha256_file(ground_truth)
            if prediction_sha != row.get("prediction_png_sha256"):
                raise ValueError("prediction PNG SHA mismatch")
            if gt_sha != expected["jpeg_sha256"]:
                raise ValueError("ground-truth JPEG SHA mismatch")
            if row.get("ground_truth_jpeg_sha256") != expected["jpeg_sha256"]:
                raise ValueError("render manifest ground-truth SHA mismatch")
            with Image.open(prediction) as image:
                prediction_mode = image.mode
                prediction_size = image.size
                prediction_format = image.format
            with Image.open(ground_truth) as image:
                gt_size = image.size
                gt_format = image.format
            expected_size = (int(expected["width"]), int(expected["height"]))
            if prediction_mode != "RGB":
                raise ValueError(f"prediction mode is {prediction_mode}, expected RGB")
            if prediction_format != "PNG":
                raise ValueError(f"prediction format is {prediction_format}, expected PNG")
            if gt_format != "JPEG":
                raise ValueError(f"ground-truth format is {gt_format}, expected JPEG")
            if prediction_size != expected_size or gt_size != expected_size:
                raise ValueError(
                    f"image size mismatch pred={prediction_size} gt={gt_size} expected={expected_size}"
                )
            if int(row.get("width", -1)) != expected_size[0] or int(
                row.get("height", -1)
            ) != expected_size[1]:
                raise ValueError("render manifest dimensions mismatch")
            validated.append(
                {
                    "image_name": name,
                    "prediction_path": str(prediction),
                    "prediction_sha256": prediction_sha,
                    "ground_truth_path": str(ground_truth),
                    "ground_truth_sha256": gt_sha,
                    "width": expected_size[0],
                    "height": expected_size[1],
                }
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    return {
        "passed": not errors and len(validated) == len(expected_rows),
        "scene": scene,
        "method_id": method_id,
        "expected_count": len(expected_rows),
        "validated_count": len(validated),
        "missing_names": missing,
        "extra_names": extra,
        "errors": errors,
        "validated_rows": validated,
        "contract": contract,
        "input_manifest": input_manifest,
        "render_manifest": render_manifest,
    }


def validate_metric_reference(
    *,
    contract: dict[str, Any],
    metric_reference_root: Path,
    vgg16_weights: Path,
    lpips_vgg_weights: Path,
) -> dict[str, Any]:
    metric_reference_root = metric_reference_root.expanduser().resolve()
    expected_files = contract["metric_reference"]["files_sha256"]
    files: dict[str, dict[str, Any]] = {}
    for relative, expected_sha in expected_files.items():
        path = metric_reference_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != expected_sha:
            raise ValueError(f"metric reference SHA mismatch for {relative}: {actual}")
        files[relative] = {"path": str(path), "sha256": actual}
    expected_weights = contract["metric_reference"]["weights_sha256"]
    weight_paths = {
        "vgg16-397923af.pth": vgg16_weights.expanduser().resolve(),
        "vgg.pth": lpips_vgg_weights.expanduser().resolve(),
    }
    weights: dict[str, dict[str, Any]] = {}
    for name, path in weight_paths.items():
        if path.name != name:
            raise ValueError(f"metric weight filename mismatch for {name}: {path.name}")
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != expected_weights[name]:
            raise ValueError(f"metric weight SHA mismatch for {name}: {actual}")
        weights[name] = {"path": str(path), "sha256": actual}
    if weight_paths["vgg16-397923af.pth"].parent != weight_paths["vgg.pth"].parent:
        raise ValueError("metric weights must share the pinned torch-hub checkpoints directory")
    return {
        "metric_reference_root": str(metric_reference_root),
        "files": files,
        "weights": weights,
    }


def _image_tensor(path: Path, torch: Any, device: str) -> Any:
    with Image.open(path) as image:
        array = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor.to(device=device, dtype=torch.float32).div_(255.0).unsqueeze(0)


def _json_metric(value: float) -> float | str:
    if math.isinf(value) and value > 0:
        return "Infinity"
    if not math.isfinite(value):
        raise ValueError(f"unsupported non-finite metric: {value}")
    return float(value)


def compute_metrics(
    *,
    rows: list[dict[str, Any]],
    contract: dict[str, Any],
    metric_reference_root: Path,
    vgg16_weights: Path,
    device: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch  # noqa: WPS433
    import torchvision  # noqa: WPS433

    expected_runtime = contract["metric_reference"]["runtime"]
    actual_runtime = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "Pillow": importlib.metadata.version("Pillow"),
        "numpy": importlib.metadata.version("numpy"),
    }
    if actual_runtime != expected_runtime:
        raise ValueError(f"metric runtime mismatch: {actual_runtime} != {expected_runtime}")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA metric device requested but unavailable")

    # Both pinned files live in <torch-hub-dir>/checkpoints.  Point torch.hub
    # at the parent so torchvision and load_state_dict_from_url resolve those
    # exact verified cache entries without a network fetch.
    torch.hub.set_dir(str(vgg16_weights.expanduser().resolve().parent.parent))
    sys.path.insert(0, str(metric_reference_root.expanduser().resolve()))
    from lpipsPyTorch.modules.lpips import LPIPS  # noqa: WPS433
    from utils.image_utils import psnr  # noqa: WPS433
    from utils.loss_utils import ssim  # noqa: WPS433

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    criterion = LPIPS("vgg", "0.1").eval().to(device)

    output: list[dict[str, Any]] = []
    with torch.no_grad():
        for row in rows:
            prediction = _image_tensor(Path(row["prediction_path"]), torch, device)
            ground_truth = _image_tensor(Path(row["ground_truth_path"]), torch, device)
            if tuple(prediction.shape) != tuple(ground_truth.shape):
                raise ValueError(f"tensor shape mismatch for {row['image_name']}")
            psnr_value = float(psnr(prediction, ground_truth).mean().item())
            ssim_value = float(ssim(prediction, ground_truth).item())
            lpips_value = float(criterion(prediction, ground_truth).item())
            if not math.isfinite(ssim_value) or not math.isfinite(lpips_value):
                raise ValueError(f"non-finite SSIM/LPIPS for {row['image_name']}")
            output.append(
                {
                    **row,
                    "PSNR": _json_metric(psnr_value),
                    "SSIM": ssim_value,
                    "LPIPS_VGG": lpips_value,
                    "psnr_is_positive_infinity": math.isinf(psnr_value)
                    and psnr_value > 0,
                }
            )
            del prediction, ground_truth
    numeric_psnr = [
        math.inf if row["PSNR"] == "Infinity" else float(row["PSNR"])
        for row in output
    ]
    means = {
        "PSNR": _json_metric(arithmetic_mean(numeric_psnr)),
        "SSIM": arithmetic_mean(float(row["SSIM"]) for row in output),
        "LPIPS_VGG": arithmetic_mean(float(row["LPIPS_VGG"]) for row in output),
    }
    runtime = {
        **actual_runtime,
        "device": device,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_name": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
        "deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
    }
    return output, {"mean": means, "runtime": runtime}


def write_per_view_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "image_name",
        "PSNR",
        "SSIM",
        "LPIPS_VGG",
        "psnr_is_positive_infinity",
        "prediction_sha256",
        "ground_truth_sha256",
        "width",
        "height",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            converted = dict(row)
            if converted.get("PSNR") == "Infinity":
                converted["PSNR"] = "inf"
            writer.writerow(converted)


def evaluate(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    started_at = datetime.now(timezone.utc).isoformat()
    validation = validate_render_and_ground_truth(
        contract_path=args.rgb_contract,
        input_manifest_path=args.input_manifest,
        input_root=args.input_root,
        render_manifest_path=args.render_manifest,
        scene=args.scene,
        method_id=args.method_id,
        allow_review_candidate=args.allow_review_candidate,
    )
    public_validation = {
        key: value
        for key, value in validation.items()
        if key not in {"contract", "input_manifest", "render_manifest", "validated_rows"}
    }
    write_json(output_dir / "input_render_validation.json", public_validation)
    if not validation["passed"]:
        summary = {
            "schema": SUMMARY_SCHEMA,
            "suite_id": SUITE_ID,
            "status": "INCOMPLETE_UNRANKED",
            "scene": args.scene,
            "method_id": args.method_id,
            "expected_test_view_count": validation["expected_count"],
            "evaluated_test_view_count": validation["validated_count"],
            "complete_test_coverage": False,
            "ranking_eligible": False,
            "formal_execution": False,
            "contract_status": validation["contract"]["status"],
            "primary_scene_mean_available": False,
            "unranked_reason": "input/render validation did not achieve complete frozen coverage",
            "validation": public_validation,
        }
        write_json(output_dir / "rgb_quality_summary.json", summary)
        return summary, 2

    metric_reference = validate_metric_reference(
        contract=validation["contract"],
        metric_reference_root=args.metric_reference_root,
        vgg16_weights=args.vgg16_weights,
        lpips_vgg_weights=args.lpips_vgg_weights,
    )
    per_view, computed = compute_metrics(
        rows=validation["validated_rows"],
        contract=validation["contract"],
        metric_reference_root=args.metric_reference_root,
        vgg16_weights=args.vgg16_weights,
        device=args.device,
    )
    write_json(output_dir / "per_view_metrics.json", per_view)
    write_per_view_csv(output_dir / "per_view_metrics.csv", per_view)
    finished_at = datetime.now(timezone.utc).isoformat()
    technical_smoke = bool(args.allow_review_candidate)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "suite_id": SUITE_ID,
        "status": (
            "TECHNICAL_SMOKE_COMPLETE_UNRANKED"
            if technical_smoke
            else "COMPLETE_RANKED"
        ),
        "scene": args.scene,
        "method_id": args.method_id,
        "expected_test_view_count": validation["expected_count"],
        "evaluated_test_view_count": len(per_view),
        "complete_test_coverage": True,
        "ranking_eligible": not technical_smoke,
        "formal_execution": not technical_smoke,
        "contract_status": validation["contract"]["status"],
        "primary_scene_mean_available": not technical_smoke,
        "aggregation": validation["contract"]["coverage_and_aggregation"],
        "metric_domain": validation["contract"]["metric_domain"],
        "test_time_optimization": False,
        "started_at": started_at,
        "finished_at": finished_at,
    }
    if technical_smoke:
        summary["diagnostic_scene_mean"] = computed["mean"]
        summary["unranked_reason"] = "allow_review_candidate technical-smoke mode"
    else:
        summary["primary_scene_mean"] = computed["mean"]
    write_json(output_dir / "rgb_quality_summary.json", summary)
    evaluator_manifest = {
        "schema": EVALUATOR_MANIFEST_SCHEMA,
        "suite_id": SUITE_ID,
        "status": "PASS_TECHNICAL_SMOKE" if technical_smoke else "PASS_FORMAL",
        "ranking_eligible": not technical_smoke,
        "formal_execution": not technical_smoke,
        "scene": args.scene,
        "method_id": args.method_id,
        "evaluator_path": str(Path(__file__).resolve()),
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
        "contract_path": str(args.rgb_contract.expanduser().resolve()),
        "contract_sha256": sha256_file(args.rgb_contract.expanduser().resolve()),
        "input_manifest_path": str(args.input_manifest.expanduser().resolve()),
        "input_manifest_sha256": sha256_file(args.input_manifest.expanduser().resolve()),
        "render_manifest_path": str(args.render_manifest.expanduser().resolve()),
        "render_manifest_sha256": sha256_file(args.render_manifest.expanduser().resolve()),
        "metric_reference": metric_reference,
        "runtime": computed["runtime"],
        "outputs_sha256": {
            name: sha256_file(output_dir / name)
            for name in (
                "input_render_validation.json",
                "per_view_metrics.json",
                "per_view_metrics.csv",
                "rgb_quality_summary.json",
            )
        },
    }
    write_json(output_dir / "evaluator_manifest.json", evaluator_manifest)
    return summary, 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb_contract", type=Path, required=True)
    parser.add_argument("--input_manifest", type=Path, required=True)
    parser.add_argument("--input_root", type=Path, required=True)
    parser.add_argument("--render_manifest", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--method_id", required=True)
    parser.add_argument("--metric_reference_root", type=Path, required=True)
    parser.add_argument("--vgg16_weights", type=Path, required=True)
    parser.add_argument("--lpips_vgg_weights", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--allow_review_candidate",
        action="store_true",
        help="Technical smoke only; formal results require ACTIVE_FROZEN.",
    )
    return parser


def main() -> int:
    summary, exit_code = evaluate(build_parser().parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
