#!/usr/bin/env python3
"""Promote terminal 100K outcomes into one success-only evaluation registry.

This is the intentionally small successor to the retired activation-v4 chain.
It records every terminal training outcome, promotes only immutable successful
models, and writes the RGB contract/registry used by the common evaluators.
No metric value participates in promotion.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from m3m_gcp_lidar_artifacts import canonical_sha256, sha256_file
from rgb_quality_contract import (
    directory_content_identity,
    git_identity,
    sparse_model_sha256,
    validate_benchmark_checkout,
)


SCENE = "gcp_100000_20260610"
SUITE = "m3m_gcp_native_quarter_rgb_quality_v1"
QUALIFICATION_ROOT = Path(
    f"/root/autodl-tmp/runs/m3m-gcp-native-quarter/qualification-100k-v1/{SCENE}"
)
FORMAL_ROOT = Path(
    f"/root/autodl-tmp/datasets/M3M-GCP-colmap-native-quarter-v1/formal_inputs/{SCENE}"
)
RGB_CAMERA_ROOT = Path(
    f"/root/autodl-tmp/datasets/M3M-GCP-100K-rgb-evaluation-camera-root-v1/{SCENE}"
)
GCP_CAMERA_ROOT = Path(
    f"/root/autodl-tmp/datasets/M3M-GCP-100K-gcp-evaluation-camera-root-v1/{SCENE}"
)
LIDAR_CAMERA_ROOT = Path(
    f"/root/autodl-tmp/datasets/M3M-GCP-100K-evaluation-camera-root-v1/{SCENE}"
)
GSPRIOR_NORMALIZED_ROOT = Path(
    f"/root/autodl-tmp/datasets/M3M-GCP-gsprior-normalized-v1/{SCENE}"
)


METHODS: dict[str, dict[str, Any]] = {
    "3dgs_original": {
        "display_name": "3D Gaussian Splatting",
        "outcome": "SUCCESS_MODEL_FIXED",
        "run_root": Path(
            "/root/autodl-tmp/runs/m3m-gcp-native-quarter/3dgs-original/"
            f"{SCENE}/seed0-30k-20260810T175634Z"
        ),
        "model_root_relative": "model",
        "model_relative": "point_cloud/iteration_30000/point_cloud.ply",
        "iteration": 30000,
        "source_root": Path(
            "/root/autodl-tmp/worktrees/m3m-gcp-native-quarter/3dgs-original/"
            "2eee0e26d2d5fd00ec462df47752223952f6bf4e/official-train"
        ),
        "source_commit": "2eee0e26d2d5fd00ec462df47752223952f6bf4e",
        "environment": Path(
            "/root/autodl-tmp/envs/m3m-gcp-native-quarter/3dgs-original/"
            "py310-torch2.7.1-cu128-v1"
        ),
        "adapter": "export_gaussian_rgb.py",
        "adapter_kind": "graphdeco_style_gaussian_rgb_v1",
        "appearance_policy": "none",
        "camera_root": "shared.graphdeco_camera_root",
        "input_class": "rgb_colmap_only",
    },
    "pgsr": {
        "display_name": "PGSR",
        "outcome": "SUCCESS_MODEL_FIXED",
        "run_root": QUALIFICATION_ROOT / "pgsr/attempt-20260822T171431Z-flex1",
        "model_root_relative": "model",
        "model_relative": "point_cloud/iteration_30000/point_cloud.ply",
        "iteration": 30000,
        "source_root": Path(
            "/root/autodl-tmp/worktrees/m3m-gcp-native-quarter/pgsr/"
            "de24f1a38b350387e8d8fe381b2cd70c1ae946e7/train-loader-compat-v1"
        ),
        "source_commit": "de24f1a38b350387e8d8fe381b2cd70c1ae946e7",
        "environment": Path(
            "/root/autodl-tmp/envs/m3m-gcp-native-quarter/pgsr/"
            "py310-torch2.7.1-cu128-v1"
        ),
        "adapter": "export_gaussian_rgb.py",
        "adapter_kind": "graphdeco_style_gaussian_rgb_v1",
        "appearance_policy": "exposure_compensation_false_use_render_not_app_image",
        "camera_root": "shared.graphdeco_camera_root",
        "input_class": "rgb_colmap_only",
    },
    "rade_gs": {
        "display_name": "RaDe-GS",
        "outcome": "SUCCESS_MODEL_FIXED",
        "run_root": QUALIFICATION_ROOT / "rade_gs/attempt-20260823T113221Z-rescue1",
        "model_root_relative": "model",
        "model_relative": "point_cloud/iteration_30000/point_cloud.ply",
        "iteration": 30000,
        "source_root": Path(
            "/root/autodl-tmp/worktrees/m3m-gcp-native-quarter/rade_gs/"
            "d72f20792005ae1d6555a82aa2d15345f247604e/official-train"
        ),
        "source_commit": "d72f20792005ae1d6555a82aa2d15345f247604e",
        "environment": Path(
            "/root/autodl-tmp/envs/m3m-gcp-native-quarter/rade_gs/"
            "py310-torch2.7.1-cu128-v1"
        ),
        "adapter": "export_gaussian_rgb.py",
        "adapter_kind": "graphdeco_style_gaussian_rgb_v1",
        "appearance_policy": "canonical_base_render_training_only_pgsr_appearance_no_test_fit",
        "camera_root": "shared.graphdeco_camera_root",
        "input_class": "rgb_colmap_only",
        "extra_cli": ["--kernel_size", "0.0", "--use_decoupled_appearance", "0"],
    },
    "citygs_x": {
        "display_name": "CityGS-X",
        "outcome": "SUCCESS_MODEL_FIXED",
        "run_root": QUALIFICATION_ROOT / "citygs_x/attempt-20260823T062554Z-infra1",
        "model_root_relative": "model",
        "model_relative": "point_cloud/iteration_100000/point_cloud_rk0_ws1.ply",
        "model_aux": ["additional_attributes.npz", "checkpoints.pth"],
        "iteration": 100000,
        "source_root": Path(
            "/root/autodl-tmp/worktrees/m3m-gcp-native-quarter/citygs_x/"
            "27617f2486505e3b6fe75345edf7c2b11161bc2a/train-runtime-v1"
        ),
        "source_commit": "27617f2486505e3b6fe75345edf7c2b11161bc2a",
        "environment": Path(
            "/root/autodl-tmp/envs/m3m-gcp-native-quarter/citygs_x/"
            "py310-torch2.7.1-cu128-v1"
        ),
        "adapter": "export_citygs_x_rgb.py",
        "adapter_kind": "citygs_x_rgb_v1",
        "appearance_policy": "appearance_dim_0",
        "camera_root": "shared.default_camera_root",
        "input_class": "rgb_colmap_external_geometry_prior",
    },
    "metrogs": {
        "display_name": "MetroGS",
        "outcome": "SUCCESS_MODEL_FIXED",
        "run_root": QUALIFICATION_ROOT / "metrogs/attempt-20260823T130924Z-sparsefix1",
        "model_root_relative": "model",
        "model_relative": "point_cloud/iteration_150000/point_cloud.ply",
        "checkpoint_relative": "model/checkpoints/epoch=69-step=150000.ckpt",
        "iteration": 150000,
        "source_root": Path(
            "/root/autodl-tmp/worktrees/m3m-gcp-native-quarter/metrogs/"
            "8cf9ac13c0c34b65c1a935d181c4634909e60f3f/official-train"
        ),
        "source_commit": "8cf9ac13c0c34b65c1a935d181c4634909e60f3f",
        "environment": Path(
            "/root/autodl-tmp/envs/m3m-gcp-native-quarter/metrogs/"
            "py310-torch2.7.1-cu128-v1"
        ),
        "adapter": "export_lightning_gaussian_rgb.py",
        "adapter_kind": "lightning_gaussian_rgb_v1",
        "appearance_policy": "official_nearest_training_camera_geometry_alpha_0_7_v1",
        "camera_root": "shared.default_camera_root",
        "input_class": "rgb_colmap_external_geometry_prior",
        "environment_variables": {"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"},
        "appearance_training_camera_root": str(FORMAL_ROOT / "train"),
    },
    "gsprior": {
        "display_name": "GSPrior",
        "outcome": "SUCCESS_MODEL_FIXED",
        "run_root": QUALIFICATION_ROOT / "gsprior/attempt-20260823T191522Z-nvccpath1",
        "model_root_relative": "model",
        "model_relative": "point_cloud/iteration_40000/point_cloud.ply",
        "iteration": 40000,
        "source_root": Path(
            "/root/autodl-tmp/worktrees/m3m-gcp-native-quarter/gsprior/"
            "dcb7c89fb6b60f068b440de45d064ecc7fbcba55/train-compat-v1"
        ),
        "source_commit": "dcb7c89fb6b60f068b440de45d064ecc7fbcba55",
        "environment": Path(
            "/root/autodl-tmp/envs/m3m-gcp-native-quarter/gsprior/"
            "py310-torch2.7.1-cu128-v1"
        ),
        "adapter": "export_gaussian_rgb.py",
        "adapter_kind": "graphdeco_style_gaussian_rgb_v1",
        "appearance_policy": "exposure_compensation_false_use_render_not_app_image",
        "camera_root": str(GSPRIOR_NORMALIZED_ROOT / "rgb_evaluation"),
        "input_class": "rgb_colmap_only",
    },
    "2dgs": {
        "display_name": "2D Gaussian Splatting",
        "outcome": "RESOURCE_FAILURE_HOST_MEMORY",
        "run_root": QUALIFICATION_ROOT / "2dgs/attempt-20260822T170333Z-flex1",
        "failure_iteration": 7000,
        "failure_note": "return code 137 near the 118111600640-byte host cgroup cap; no final model",
    },
    "qgs": {
        "display_name": "QGS",
        "outcome": "RESOURCE_FAILURE_CUDA_OOM",
        "run_root": QUALIFICATION_ROOT / "qgs/attempt-20260823T111143Z-flex1",
        "failure_iteration": 7600,
        "failure_note": "CUDA OOM at 14917637 points with 96325 MiB observed peak; no final model",
    },
    "sof": {
        "display_name": "Scaffold-on-Forests",
        "outcome": "RESOURCE_FAILURE_HOST_MEMORY",
        "run_root": QUALIFICATION_ROOT / "sof/attempt-20260823T112919Z-flex1",
        "failure_iteration": None,
        "failure_note": "return code 137 during pre-optimizer loading at the host cgroup cap; no final model",
    },
    "citygaussian_v2": {
        "display_name": "CityGaussianV2",
        "outcome": "NR_TIME",
        "run_root": None,
        "failure_iteration": None,
        "failure_note": "not launched: estimated 75-85 single-GPU hours under the common formal input",
    },
}

SUCCESS_ORDER = [
    "3dgs_original",
    "pgsr",
    "rade_gs",
    "citygs_x",
    "metrogs",
    "gsprior",
]
FULL_ORDER = [
    "3dgs_original",
    "2dgs",
    "pgsr",
    "rade_gs",
    "qgs",
    "gsprior",
    "sof",
    "citygaussian_v2",
    "citygs_x",
    "metrogs",
]


def identity(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def optional_identity(path: Path) -> dict[str, Any] | None:
    return identity(path) if path.is_file() else None


def write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(
            descriptor,
            (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
                "utf-8"
            ),
        )
    finally:
        os.close(descriptor)


def write_allowlist_exclusive(path: Path, image_names: list[str]) -> None:
    if len(image_names) != 211 or len(set(image_names)) != 211 or any(
        not name for name in image_names
    ):
        raise RuntimeError("GCP packet allowlist must contain 211 unique image names")
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["image_name"])
        writer.writerows([name] for name in image_names)


def source_binding(spec: dict[str, Any]) -> dict[str, Any]:
    source_root = Path(spec["source_root"]).resolve()
    source = git_identity(source_root)
    if source.get("commit") != spec["source_commit"]:
        raise RuntimeError(f"source commit mismatch: {source_root}")
    if source.get("unexpected_untracked_files") != []:
        raise RuntimeError(f"unexpected source files: {source_root}")
    return {
        "source_root": str(source_root),
        "source_commit": spec["source_commit"],
        "source_worktree": {
            "expected_tracked_diff_sha256": source["tracked_diff_sha256"],
            "expected_tracked_files_sha256": source["tracked_modified_files_sha256"],
            "untracked_policy": "generated_pycache_only",
        },
    }


def success_model_identity(method_id: str, spec: dict[str, Any]) -> dict[str, Any]:
    run_root = Path(spec["run_root"]).resolve()
    model_root = run_root / str(spec["model_root_relative"])
    model_path = model_root / str(spec["model_relative"])
    row: dict[str, Any] = {
        "run_root": str(run_root),
        "model_root": str(model_root),
        "formal_model": identity(model_path),
        "iteration": int(spec["iteration"]),
        "cfg_args": optional_identity(model_root / "cfg_args"),
        "training_cameras_json": optional_identity(model_root / "cameras.json"),
    }
    aux = {
        name: identity(model_path.parent / name)
        for name in spec.get("model_aux", [])
    }
    if aux:
        row["formal_model_aux"] = aux
    if spec.get("checkpoint_relative"):
        row["formal_checkpoint"] = identity(
            run_root / str(spec["checkpoint_relative"])
        )
    for name in (
        "qualification_launch_receipt.json",
        "qualification_terminal_receipt.json",
        "telemetry/resource_summary.json",
        "model/training_wrapper_summary.json",
    ):
        evidence = optional_identity(run_root / name)
        if evidence is not None:
            row.setdefault("evidence", {})[name] = evidence
    return row


def build_inventory(benchmark_identity: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for method_id in FULL_ORDER:
        spec = METHODS[method_id]
        row: dict[str, Any] = {
            "method_id": method_id,
            "display_name": spec["display_name"],
            "outcome": spec["outcome"],
            "ranking_eligible_tracks": (
                ["rgb", "gcp", "lidar"]
                if spec["outcome"] == "SUCCESS_MODEL_FIXED"
                else []
            ),
        }
        if spec["outcome"] == "SUCCESS_MODEL_FIXED":
            row["model_identity"] = success_model_identity(method_id, spec)
            row["renderer_source"] = source_binding(spec)
            row["environment"] = str(Path(spec["environment"]).resolve())
        else:
            row["run_root"] = (
                str(Path(spec["run_root"]).resolve()) if spec["run_root"] else None
            )
            row["failure_iteration"] = spec["failure_iteration"]
            row["note"] = spec["failure_note"]
            if spec["run_root"]:
                run_root = Path(spec["run_root"]).resolve()
                row["evidence"] = {
                    name: optional_identity(run_root / name)
                    for name in (
                        "qualification_launch_receipt.json",
                        "qualification_terminal_receipt.json",
                        "telemetry/resource_summary.json",
                        "telemetry/stderr.log",
                    )
                }
        rows.append(row)
    payload: dict[str, Any] = {
        "schema": "m3m_gcp_100k_qualification_outcome_inventory_v1",
        "status": "COMPLETE_TERMINAL_OUTCOMES",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scene": SCENE,
        "method_order": FULL_ORDER,
        "success_method_ids": SUCCESS_ORDER,
        "success_count": len(SUCCESS_ORDER),
        "terminal_non_success_count": len(FULL_ORDER) - len(SUCCESS_ORDER),
        "promotion_uses_metric_values": False,
        "model_bytes_mutated_by_promotion": False,
        "input_manifest": identity(FORMAL_ROOT / "NATIVE_QUARTER_INPUT_MANIFEST.json"),
        "benchmark_builder_identity": benchmark_identity,
        "methods": rows,
    }
    payload["canonical_sha256"] = canonical_sha256(payload)
    return payload


def build_contract(source: dict[str, Any], source_path: Path) -> dict[str, Any]:
    contract = deepcopy(source)
    contract["derivation"] = {
        **contract.get("derivation", {}),
        "source_contract_path": str(source_path.resolve()),
        "source_contract_sha256": sha256_file(source_path),
        "scientific_metric_prediction_aggregation_and_appearance_fields_byte_semantically_unchanged": True,
        "formal_use_requires_three_track_addendum_activation": False,
        "operational_successor": "100K terminal-outcome success-subset promotion v1",
    }
    contract["formal_gate"] = {
        "required_status_for_formal_run": "ACTIVE_FROZEN",
        "required_registry": "m3m_gcp_native_quarter_rgb_quality_100k_success_registry_v1",
        "required_outcome_inventory": "m3m_gcp_100k_qualification_outcome_inventory_v1",
        "benchmark_checkout_identity": "exact clean commit and tree rechecked by every adapter and evaluator",
        "render_manifest_required": True,
        "per_view_metrics_required": True,
        "source_weight_model_and_input_hashes_required": True,
        "no_overwrite": True,
        "legacy_activation_v4_required": False,
        "review_mode": "one combined red-line review before launch and one combined review after all methods",
    }
    return contract


def registry_method(
    method_id: str,
    spec: dict[str, Any],
    model: dict[str, Any],
    benchmark_repo: Path,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "method_id": method_id,
        "scene": SCENE,
        "display_name": spec["display_name"],
        "input_class": spec["input_class"],
        "adapter": spec["adapter"],
        "adapter_kind": spec["adapter_kind"],
        **source_binding(spec),
        "environment": str(Path(spec["environment"]).resolve()),
        "run_root": model["run_root"],
        "iteration": spec["iteration"],
        "camera_root": spec["camera_root"],
        "appearance_policy": spec["appearance_policy"],
        "extra_cli": list(spec.get("extra_cli", [])),
        "formal_output_root": str(
            Path(model["run_root"]) / "formal_evaluation/rgb_quality_100k_success_v1"
        ),
    }
    if method_id == "metrogs":
        row["formal_checkpoint"] = model["formal_checkpoint"]["path"]
        row["formal_model_sha256"] = model["formal_checkpoint"]["sha256"]
        row["training_cameras_json"] = model["training_cameras_json"]["path"]
        row["training_cameras_json_sha256"] = model["training_cameras_json"]["sha256"]
        row["appearance_training_camera_root"] = spec[
            "appearance_training_camera_root"
        ]
        row["environment_variables"] = dict(spec["environment_variables"])
    else:
        row["model_root"] = model["model_root"]
        row["formal_model_relative_path"] = spec["model_relative"]
        row["formal_model_sha256"] = model["formal_model"]["sha256"]
        if model.get("cfg_args"):
            row["cfg_args_sha256"] = model["cfg_args"]["sha256"]
    if method_id == "citygs_x":
        row["formal_model_aux_sha256"] = {
            name: value["sha256"]
            for name, value in model.get("formal_model_aux", {}).items()
        }
        compat = benchmark_repo / "compat/citygs_x/pytorch3d_transforms_minimal_v1"
        row["pytorch3d_compat"] = str(compat)
        row["pythonpath"] = [str(compat)]
    elif method_id in {"pgsr", "gsprior"}:
        compat = benchmark_repo / f"compat/{method_id}/pytorch3d_transforms_minimal_v1"
        row["pythonpath"] = [str(compat)]
    if row.get("pythonpath"):
        row["pythonpath_content_identity"] = [
            directory_content_identity(Path(path)) for path in row["pythonpath"]
        ]
    if method_id == "gsprior":
        row["camera_sparse_sha256"] = sparse_model_sha256(
            GSPRIOR_NORMALIZED_ROOT / "rgb_evaluation"
        )
        row["camera_coordinate_policy"] = (
            "frozen training-only GSPrior similarity transform; pixels and intrinsics unchanged"
        )
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-repo", type=Path, required=True)
    parser.add_argument("--benchmark-commit", required=True)
    parser.add_argument("--benchmark-tree", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    benchmark_repo = args.benchmark_repo.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(output_root)
    benchmark_identity = validate_benchmark_checkout(
        benchmark_repo=benchmark_repo,
        expected_commit=args.benchmark_commit,
        expected_tree=args.benchmark_tree,
        entrypoint=Path(__file__).resolve(),
    )
    for root, manifest_name, expected_status in (
        (RGB_CAMERA_ROOT, "RGB_EVALUATION_CAMERA_ROOT_MANIFEST.json", "PASS_RGB_EVALUATION_CAMERA_ROOT"),
        (GCP_CAMERA_ROOT, "GCP_EVALUATION_CAMERA_ROOT_MANIFEST.json", "PASS_GCP_EVALUATION_CAMERA_ROOT_NO_RGB_PIXELS"),
        (LIDAR_CAMERA_ROOT, "EVALUATION_CAMERA_ROOT_MANIFEST.json", "PASS_EVALUATION_CAMERA_ROOT_NO_TRAINING_NO_PRIOR_NO_EVALUATION"),
    ):
        manifest = json.loads((root / manifest_name).read_text(encoding="utf-8"))
        if manifest.get("status") != expected_status or manifest.get("scene") != SCENE:
            raise RuntimeError(f"camera manifest mismatch: {root}")
    for role, expected_count in (("rgb_evaluation", 314), ("gcp_evaluation", 211)):
        manifest = json.loads(
            (GSPRIOR_NORMALIZED_ROOT / role / "normalization_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        if manifest.get("status") != "PASS" or manifest.get("output", {}).get("camera_count") != expected_count:
            raise RuntimeError(f"GSPrior normalized camera root mismatch: {role}")

    inventory = build_inventory(benchmark_identity)
    inventory_path = output_root / "qualification_outcome_inventory.json"
    output_root.mkdir(parents=True)
    write_exclusive(inventory_path, inventory)

    gcp_manifest = json.loads(
        (GCP_CAMERA_ROOT / "GCP_EVALUATION_CAMERA_ROOT_MANIFEST.json").read_text(
            encoding="utf-8"
        )
    )
    gcp_allowlist_path = output_root / "gcp_211_view_allowlist.csv"
    write_allowlist_exclusive(
        gcp_allowlist_path,
        [str(name) for name in gcp_manifest["output"]["image_names"]],
    )

    promotion: dict[str, Any] = {
        "schema": "m3m_gcp_100k_success_subset_promotion_receipt_v1",
        "status": "PROMOTED_FOR_EVALUATION",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scene": SCENE,
        "outcome_inventory": identity(inventory_path),
        "eligible_method_ids": SUCCESS_ORDER,
        "ineligible_method_ids": [name for name in FULL_ORDER if name not in SUCCESS_ORDER],
        "promotion_basis": "terminal model existence and resource outcome only",
        "metric_values_used_for_promotion": False,
        "model_bytes_changed": False,
        "legacy_activation_v4_used": False,
    }
    promotion["canonical_sha256"] = canonical_sha256(promotion)
    promotion_path = output_root / "success_subset_promotion_receipt.json"
    write_exclusive(promotion_path, promotion)

    source_contract_path = benchmark_repo / "configs/m3m_gcp_native_quarter_rgb_quality_100k_v1.json"
    source_contract = json.loads(source_contract_path.read_text(encoding="utf-8"))
    contract = build_contract(source_contract, source_contract_path)
    contract_path = output_root / "rgb_quality_100k_success_contract.json"
    write_exclusive(contract_path, contract)

    inventory_by_id = {row["method_id"]: row for row in inventory["methods"]}
    rgb_manifest = json.loads(
        (RGB_CAMERA_ROOT / "RGB_EVALUATION_CAMERA_ROOT_MANIFEST.json").read_text(
            encoding="utf-8"
        )
    )
    shared_sparse = {
        name: value["sha256"]
        for name, value in rgb_manifest["output"]["files"].items()
        if name in {"cameras.bin", "images.bin", "points3D.bin"}
    }
    registry: dict[str, Any] = {
        "schema": "m3m_gcp_native_quarter_rgb_quality_100k_success_registry_v1",
        "suite_id": SUITE,
        "status": "ACTIVE_FROZEN",
        "scene": SCENE,
        "server": "AutoDL-901",
        "active_method_count": len(SUCCESS_ORDER),
        "ready_method_ids": SUCCESS_ORDER,
        "outcome_inventory": identity(inventory_path),
        "promotion_receipt": identity(promotion_path),
        "shared": {
            "benchmark_repo": str(benchmark_repo),
            "contract_path": str(contract_path),
            "registry_path": str(output_root / "rgb_success_registry.json"),
            "input_manifest": str(FORMAL_ROOT / "NATIVE_QUARTER_INPUT_MANIFEST.json"),
            "input_root": str(FORMAL_ROOT),
            "default_camera_root": str(RGB_CAMERA_ROOT),
            "default_camera_sparse_sha256": shared_sparse,
            "graphdeco_camera_root": str(RGB_CAMERA_ROOT),
            "graphdeco_camera_sparse_sha256": shared_sparse,
            "metric_environment": str(METHODS["3dgs_original"]["environment"]),
            "metric_reference_root": str(METHODS["3dgs_original"]["source_root"]),
            "vgg16_weights": "/root/.cache/torch/hub/checkpoints/vgg16-397923af.pth",
            "lpips_vgg_weights": "/root/.cache/torch/hub/checkpoints/vgg.pth",
            "metric_device": "cuda:0",
            "rgb_view_count": 314,
            "gcp_view_count": 211,
            "lidar_view_count": 2196,
            "gcp_packet_allowlist": identity(gcp_allowlist_path),
            "lidar_packet_allowlist": identity(
                benchmark_repo
                / "configs/m3m_gcp_lidar_train_view_allowlists_v1"
                / f"{SCENE}.csv"
            ),
        },
        "methods": [
            registry_method(
                method_id,
                METHODS[method_id],
                inventory_by_id[method_id]["model_identity"],
                benchmark_repo,
            )
            for method_id in SUCCESS_ORDER
        ],
    }
    registry["canonical_sha256"] = canonical_sha256(registry)
    registry_path = output_root / "rgb_success_registry.json"
    write_exclusive(registry_path, registry)

    summary = {
        "status": "PASS_100K_SUCCESS_EVALUATION_RUNTIME",
        "output_root": str(output_root),
        "inventory": identity(inventory_path),
        "promotion": identity(promotion_path),
        "rgb_contract": identity(contract_path),
        "rgb_registry": identity(registry_path),
        "gcp_packet_allowlist": identity(gcp_allowlist_path),
        "success_method_ids": SUCCESS_ORDER,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
