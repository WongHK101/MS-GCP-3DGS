#!/usr/bin/env python3
"""Compute one shared PSNR/SSIM/LPIPS-VGG suite from adapter PNGs."""

from __future__ import annotations

import argparse
import csv
import hashlib
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
    directory_content_identity,
    git_identity,
    load_bound_input_manifest,
    load_contract,
    role_rows,
    sha256_file,
    sparse_model_sha256,
    validate_benchmark_checkout,
    write_json,
)


SUMMARY_SCHEMA = "m3m_gcp_native_quarter_rgb_quality_summary_v1"
EVALUATOR_MANIFEST_SCHEMA = "m3m_gcp_native_quarter_rgb_evaluator_manifest_v1"


def _safe_relative(base: Path, relative: str) -> Path:
    path = (base / relative).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError(f"path escapes manifest root: {relative}")
    return path


def _same_resolved_path(actual: Any, expected: Any) -> bool:
    try:
        return Path(str(actual)).expanduser().resolve() == Path(str(expected)).expanduser().resolve()
    except Exception:  # noqa: BLE001
        return False


def _registered_camera_root(method: dict[str, Any], shared: dict[str, Any]) -> Path:
    value = str(method["camera_root"])
    aliases = {
        "shared.default_camera_root": "default_camera_root",
        "shared.graphdeco_camera_root": "graphdeco_camera_root",
    }
    if value in aliases:
        value = str(shared[aliases[value]])
    return Path(value).expanduser().resolve()


def _registered_camera_sparse_sha256(
    method: dict[str, Any], shared: dict[str, Any]
) -> dict[str, str]:
    value = str(method["camera_root"])
    aliases = {
        "shared.default_camera_root": "default_camera_sparse_sha256",
        "shared.graphdeco_camera_root": "graphdeco_camera_sparse_sha256",
    }
    if value in aliases:
        return dict(shared[aliases[value]])
    return dict(method["camera_sparse_sha256"])


def validate_registered_provenance(
    *,
    contract: dict[str, Any],
    registry_path: Path | None,
    render_manifest: dict[str, Any],
    scene: str,
    method_id: str,
    allow_review_candidate: bool,
    benchmark_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind a formal render to the frozen registry, source tree, and model bytes."""

    if registry_path is None:
        if allow_review_candidate:
            return {
                "passed": True,
                "binding_mode": "technical_smoke_unbound_unranked",
                "registry_path": None,
                "registry_sha256": None,
                "checks": [],
                "errors": [],
            }
        return {
            "passed": False,
            "binding_mode": "formal_registry_required",
            "registry_path": None,
            "registry_sha256": None,
            "checks": [],
            "errors": ["formal evaluation requires --registry"],
        }

    registry_path = registry_path.expanduser().resolve()
    errors: list[str] = []
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})
        if not passed:
            errors.append(f"{name}: {detail}")

    if not registry_path.is_file():
        return {
            "passed": False,
            "binding_mode": "frozen_registry",
            "registry_path": str(registry_path),
            "registry_sha256": None,
            "checks": [],
            "errors": [f"registry is missing: {registry_path}"],
        }
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry_sha = sha256_file(registry_path)
    record("registry_suite_id", registry.get("suite_id") == SUITE_ID, registry.get("suite_id"))
    record("registry_scene", registry.get("scene") == scene, registry.get("scene"))
    record(
        "registry_status",
        registry.get("status") == contract.get("status"),
        {"registry": registry.get("status"), "contract": contract.get("status")},
    )
    if not allow_review_candidate:
        record("formal_registry_active", registry.get("status") == "ACTIVE_FROZEN", registry.get("status"))
    candidates = [
        method
        for method in registry.get("methods", [])
        if isinstance(method, dict) and method.get("method_id") == method_id
    ]
    record("registered_method_unique", len(candidates) == 1, len(candidates))
    if len(candidates) != 1:
        return {
            "passed": False,
            "binding_mode": "frozen_registry",
            "registry_path": str(registry_path),
            "registry_sha256": registry_sha,
            "checks": checks,
            "errors": errors,
        }

    method = candidates[0]
    shared = registry["shared"]
    provenance = render_manifest.get("provenance", {})
    if not isinstance(provenance, dict):
        provenance = {}
    if not allow_review_candidate:
        record(
            "benchmark_identity_supplied",
            benchmark_identity is not None,
            benchmark_identity,
        )
        rendered_benchmark = provenance.get("benchmark_repository")
        if not isinstance(rendered_benchmark, dict):
            rendered_benchmark = {}
        current_benchmark = benchmark_identity or {}
        for key in (
            "path",
            "commit",
            "tree",
            "tracked_diff_sha256",
            "tracked_modified_files_sha256",
            "unexpected_untracked_files",
        ):
            record(
                f"benchmark_repository:{key}",
                rendered_benchmark.get(key) == current_benchmark.get(key),
                rendered_benchmark.get(key),
            )
    adapter_path = Path(__file__).resolve().parent / str(method["adapter"])
    adapter_sha = sha256_file(adapter_path) if adapter_path.is_file() else "MISSING"
    record("adapter_kind", provenance.get("adapter_kind") == method.get("adapter_kind"), provenance.get("adapter_kind"))
    record("adapter_path", _same_resolved_path(provenance.get("adapter_path"), adapter_path), provenance.get("adapter_path"))
    record("adapter_sha256", provenance.get("adapter_sha256") == adapter_sha, provenance.get("adapter_sha256"))
    if method.get("adapter") == "export_qgs_rgb.py":
        shared_adapter = Path(__file__).resolve().parent / "export_gaussian_rgb.py"
        record(
            "shared_graphdeco_adapter_path",
            _same_resolved_path(provenance.get("shared_graphdeco_adapter_path"), shared_adapter),
            provenance.get("shared_graphdeco_adapter_path"),
        )
        record(
            "shared_graphdeco_adapter_sha256",
            provenance.get("shared_graphdeco_adapter_sha256") == sha256_file(shared_adapter),
            provenance.get("shared_graphdeco_adapter_sha256"),
        )

    source_root = Path(str(method["source_root"])).expanduser().resolve()
    source_identity = git_identity(source_root)
    rendered_source = provenance.get("renderer_repository", {})
    if not isinstance(rendered_source, dict):
        rendered_source = {}
    expected_worktree = method.get("source_worktree") or {}
    expected_diff = expected_worktree.get(
        "expected_tracked_diff_sha256", hashlib.sha256(b"").hexdigest()
    )
    expected_modified = expected_worktree.get("expected_tracked_files_sha256", {})
    record("source_path", _same_resolved_path(rendered_source.get("path"), source_root), rendered_source.get("path"))
    record("source_commit", source_identity.get("commit") == method.get("source_commit"), source_identity.get("commit"))
    record("source_manifest_commit", rendered_source.get("commit") == source_identity.get("commit"), rendered_source.get("commit"))
    record("source_diff", source_identity.get("tracked_diff_sha256") == expected_diff, source_identity.get("tracked_diff_sha256"))
    record(
        "source_manifest_diff",
        rendered_source.get("tracked_diff_sha256") == source_identity.get("tracked_diff_sha256"),
        rendered_source.get("tracked_diff_sha256"),
    )
    record(
        "source_modified_files",
        source_identity.get("tracked_modified_files_sha256") == expected_modified,
        source_identity.get("tracked_modified_files_sha256"),
    )
    record(
        "source_manifest_modified_files",
        rendered_source.get("tracked_modified_files_sha256")
        == source_identity.get("tracked_modified_files_sha256"),
        rendered_source.get("tracked_modified_files_sha256"),
    )
    record(
        "source_untracked_policy",
        source_identity.get("unexpected_untracked_files") == [],
        source_identity.get("unexpected_untracked_files"),
    )
    record(
        "source_manifest_untracked_policy",
        rendered_source.get("unexpected_untracked_files")
        == source_identity.get("unexpected_untracked_files"),
        rendered_source.get("unexpected_untracked_files"),
    )

    renderer_source = Path(str(provenance.get("renderer_source_path", ""))).expanduser().resolve()
    try:
        renderer_inside_source = renderer_source.is_relative_to(source_root)
    except ValueError:
        renderer_inside_source = False
    record("renderer_source_inside_frozen_repo", renderer_inside_source, str(renderer_source))
    renderer_sha = sha256_file(renderer_source) if renderer_source.is_file() else "MISSING"
    record(
        "renderer_source_sha256",
        provenance.get("renderer_source_sha256") == renderer_sha,
        provenance.get("renderer_source_sha256"),
    )
    camera_root = _registered_camera_root(method, shared)
    record(
        "camera_source_root",
        _same_resolved_path(provenance.get("camera_source_root"), camera_root),
        provenance.get("camera_source_root"),
    )
    expected_camera_sha = _registered_camera_sparse_sha256(method, shared)
    actual_camera_sha = sparse_model_sha256(camera_root)
    record(
        "camera_sparse_model_sha256",
        actual_camera_sha == expected_camera_sha,
        actual_camera_sha,
    )
    record(
        "camera_manifest_sparse_model_sha256",
        provenance.get("camera_sparse_model_sha256") == actual_camera_sha,
        provenance.get("camera_sparse_model_sha256"),
    )
    record("iteration", provenance.get("iteration") == method.get("iteration"), provenance.get("iteration"))
    record(
        "appearance_policy",
        provenance.get("appearance_policy") == method.get("appearance_policy"),
        provenance.get("appearance_policy"),
    )
    record("white_background", provenance.get("white_background") is False, provenance.get("white_background"))
    record(
        "heldout_rgb_not_consumed",
        provenance.get("heldout_rgb_used_by_adapter") is False
        and provenance.get("heldout_rgb_consumed_by_renderer_or_policy", False) is False,
        {
            "used_by_adapter": provenance.get("heldout_rgb_used_by_adapter"),
            "consumed_by_renderer_or_policy": provenance.get(
                "heldout_rgb_consumed_by_renderer_or_policy"
            ),
        },
    )
    if method.get("adapter_kind") in {"graphdeco_style_gaussian_rgb_v1", "qgs_rgb_v1"}:
        record(
            "heldout_rgb_detached_before_renderer",
            provenance.get("heldout_rgb_detached_before_renderer") is True,
            provenance.get("heldout_rgb_detached_before_renderer"),
        )
    record("test_time_optimization", provenance.get("test_time_optimization") is False, provenance.get("test_time_optimization"))

    if "formal_checkpoint" in method:
        model_path = Path(str(method["formal_checkpoint"])).expanduser().resolve()
    else:
        model_path = (
            Path(str(method["model_root"])).expanduser().resolve()
            / str(method["formal_model_relative_path"])
        )
    actual_model_sha = sha256_file(model_path) if model_path.is_file() else "MISSING"
    record("formal_model_sha256", actual_model_sha == method.get("formal_model_sha256"), actual_model_sha)
    if method_id == "citygs_x":
        record("model_root", _same_resolved_path(provenance.get("model_root"), method.get("model_root")), provenance.get("model_root"))
        model_files = provenance.get("formal_model_files_sha256", {})
        record("formal_model_manifest_sha256", model_files.get(model_path.name) == actual_model_sha, model_files.get(model_path.name))
        for name, expected_sha in method.get("formal_model_aux_sha256", {}).items():
            aux_path = model_path.parent / name
            actual_sha = sha256_file(aux_path) if aux_path.is_file() else "MISSING"
            record(f"formal_model_aux:{name}", actual_sha == expected_sha, actual_sha)
            record(f"formal_model_manifest_aux:{name}", model_files.get(name) == actual_sha, model_files.get(name))
    else:
        record("formal_model_path", _same_resolved_path(provenance.get("formal_model_path"), model_path), provenance.get("formal_model_path"))
        record("formal_model_manifest_sha256", provenance.get("formal_model_sha256") == actual_model_sha, provenance.get("formal_model_sha256"))
        if "model_root" in method:
            record("model_root", _same_resolved_path(provenance.get("model_root"), method.get("model_root")), provenance.get("model_root"))

    if "cfg_args_sha256" in method:
        record("cfg_args_sha256", provenance.get("cfg_args_sha256") == method["cfg_args_sha256"], provenance.get("cfg_args_sha256"))
    if "config_path" in method:
        method_config = provenance.get("method_config", {})
        record("method_config_path", _same_resolved_path(method_config.get("path"), method["config_path"]), method_config.get("path"))
        record("method_config_sha256", method_config.get("sha256") == method.get("config_sha256"), method_config.get("sha256"))
    if "splatting_config_path" in method:
        splatting = provenance.get("splatting_config", {})
        record("splatting_config_path", _same_resolved_path(splatting.get("path"), method["splatting_config_path"]), splatting.get("path"))
        record("splatting_config_sha256", splatting.get("sha256") == method.get("splatting_config_sha256"), splatting.get("sha256"))

    expected_pythonpath = method.get("pythonpath_content_identity", [])
    actual_pythonpath = [
        directory_content_identity(Path(str(row["path"]))) for row in expected_pythonpath
    ]
    record("runtime_pythonpath_identity", actual_pythonpath == expected_pythonpath, actual_pythonpath)
    record(
        "runtime_pythonpath_manifest_identity",
        provenance.get("runtime_pythonpath_identity") == actual_pythonpath,
        provenance.get("runtime_pythonpath_identity"),
    )
    if "training_cameras_json" in method:
        training_path = Path(str(method["training_cameras_json"])).expanduser().resolve()
        actual_training_sha = sha256_file(training_path) if training_path.is_file() else "MISSING"
        record(
            "training_cameras_json_sha256",
            actual_training_sha == method.get("training_cameras_json_sha256"),
            actual_training_sha,
        )
        rendered_training = provenance.get("training_cameras_json") or {}
        record(
            "training_cameras_json_path",
            _same_resolved_path(rendered_training.get("path"), training_path),
            rendered_training.get("path"),
        )
        record(
            "training_cameras_json_manifest_sha256",
            rendered_training.get("sha256") == actual_training_sha,
            rendered_training.get("sha256"),
        )
    if method_id == "citygs_x":
        record("citygs_x_runtime_appearance_dim", provenance.get("appearance_dim") == 0, provenance.get("appearance_dim"))

    return {
        "passed": not errors,
        "binding_mode": "frozen_registry",
        "registry_path": str(registry_path),
        "registry_sha256": registry_sha,
        "checks": checks,
        "errors": errors,
    }


def validate_render_and_ground_truth(
    *,
    contract_path: Path,
    input_manifest_path: Path,
    input_root: Path,
    render_manifest_path: Path,
    scene: str,
    method_id: str,
    allow_review_candidate: bool,
    registry_path: Path | None = None,
    benchmark_identity: dict[str, Any] | None = None,
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
    provenance_validation = validate_registered_provenance(
        contract=contract,
        registry_path=registry_path,
        render_manifest=render_manifest,
        scene=scene,
        method_id=method_id,
        allow_review_candidate=allow_review_candidate,
        benchmark_identity=benchmark_identity,
    )
    errors.extend(
        f"provenance: {message}"
        for message in provenance_validation.get("errors", [])
    )

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
        "provenance_validation": provenance_validation,
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
    benchmark_identity: dict[str, Any] | None = None
    if args.allow_review_candidate:
        if args.technical_smoke_root is None:
            raise ValueError("technical smoke requires --technical_smoke_root")
        try:
            output_dir.relative_to(args.technical_smoke_root.expanduser().resolve())
        except ValueError as exc:
            raise ValueError("technical-smoke metric output is outside its frozen root") from exc
    else:
        if (
            args.benchmark_repo is None
            or args.benchmark_commit is None
            or args.benchmark_tree is None
        ):
            raise ValueError("formal evaluator requires frozen benchmark checkout identity")
        benchmark_identity = validate_benchmark_checkout(
            benchmark_repo=args.benchmark_repo,
            expected_commit=args.benchmark_commit,
            expected_tree=args.benchmark_tree,
            entrypoint=Path(__file__).resolve(),
        )
    started_at = datetime.now(timezone.utc).isoformat()
    validation = validate_render_and_ground_truth(
        contract_path=args.rgb_contract,
        input_manifest_path=args.input_manifest,
        input_root=args.input_root,
        render_manifest_path=args.render_manifest,
        scene=args.scene,
        method_id=args.method_id,
        allow_review_candidate=args.allow_review_candidate,
        registry_path=args.registry,
        benchmark_identity=benchmark_identity,
    )
    public_validation = {
        key: value
        for key, value in validation.items()
        if key not in {"contract", "input_manifest", "render_manifest", "validated_rows"}
    }
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
        return summary, 2

    metric_reference = validate_metric_reference(
        contract=validation["contract"],
        metric_reference_root=args.metric_reference_root,
        vgg16_weights=args.vgg16_weights,
        lpips_vgg_weights=args.lpips_vgg_weights,
    )
    # All contract, checkout, provenance, data and metric-reference gates pass
    # before the immutable metric directory is created.
    output_dir.mkdir(parents=True)
    write_json(output_dir / "input_render_validation.json", public_validation)
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
        "benchmark_repository": benchmark_identity,
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
        "registry_path": (
            str(args.registry.expanduser().resolve()) if args.registry is not None else None
        ),
        "registry_sha256": (
            sha256_file(args.registry.expanduser().resolve())
            if args.registry is not None
            else None
        ),
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
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--benchmark_repo", type=Path)
    parser.add_argument("--benchmark_commit")
    parser.add_argument("--benchmark_tree")
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
    parser.add_argument("--technical_smoke_root", type=Path)
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
