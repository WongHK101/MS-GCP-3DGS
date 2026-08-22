#!/usr/bin/env python3
"""Build flexible-but-scientifically-fixed 100K qualification recipes."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from build_m3m_gcp_100k_execution_recipes import (
    CITY_IMAGES_SHA,
    ENV,
    FORMAL,
    FORMAL_CAMERAS_SHA,
    FORMAL_IMAGES_SHA,
    FORMAL_MANIFEST,
    FULL_POINTS_SHA,
    INITIAL_PLY_SHA,
    METHOD_INPUT_EVIDENCE,
    METHOD_INPUT_EVIDENCE_SHA,
    METHODS,
    METRO_POINTS_SHA,
    REUSE_3DGS,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "configs" / "m3m_gcp_native_quarter_100k_qualification_recipes_v1"
SCENE = "gcp_100000_20260610"
PLAN_ID = "m3m_gcp_100k_qualification_flex_v1"
PLAN_DOC = "docs/M3M_GCP_100K_QUALIFICATION_FLEX_V1.md"
RUN_ROOT = (
    "/root/autodl-tmp/runs/m3m-gcp-native-quarter/qualification-100k-v1/"
    f"{SCENE}"
)
BENCHMARK_REPO = "/root/autodl-tmp/code/GS-GCP-Benchmark"
CITYGAUSSIAN_RESUME = (
    "/root/autodl-tmp/runs/m3m-gcp-native-quarter/formal-100k-v2/"
    f"{SCENE}/citygaussian_v2/seed0-v2/pipeline"
)
REUSED_3DGS = (
    "/root/autodl-tmp/runs/m3m-gcp-native-quarter/3dgs-original/"
    f"{SCENE}/seed0-30k-20260810T175634Z"
)
GNU_TIME = (
    "/root/autodl-tmp/tools/gs-gcp-v13/gnu-time/"
    "ubuntu-jammy-time-1.9-v1/root/usr/bin/time"
)
CITYGAUSSIAN_RESUME_MANIFEST = (
    "{repo}/configs/m3m_gcp_native_quarter_"
    "citygaussian_v2_100k_resume_v1.json"
)

COMMON_ENV = {
    "CUDA_VISIBLE_DEVICES": "0",
    "PYTHONHASHSEED": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONUNBUFFERED": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "WANDB_MODE": "offline",
}

METHOD_ORDER = [
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

CORRECTIONS = {
    "3dgs_original": ["reuse_validated_model_no_retrain"],
    "2dgs": ["defer_in_training_test_until_after_final_save"],
    "pgsr": ["restore_minimal_pytorch3d_transforms_pythonpath"],
    "rade_gs": [
        "defer_in_training_test_until_after_final_save",
        "checkpoint_and_exact_filter_sidecar_before_final_ply_serialization",
    ],
    "qgs": ["common_expandable_cuda_allocator"],
    "gsprior": [
        "restore_minimal_pytorch3d_and_benchmark_helper_pythonpath",
        "lazy_rgb_residency_to_avoid_duplicate_tsdf_scene_preload",
        "bounded_total_wall_clock_12h",
    ],
    "sof": ["clean_attempt_with_complete_resource_telemetry"],
    "citygaussian_v2": [
        "replace_broken_free_gpu_polling_with_official_sequential_block_commands",
        "hardlink_completed_coarse_and_blocks_into_fresh_attempt",
    ],
    "citygs_x": ["save_final_checkpoint_before_offline_evaluation"],
    "metrogs": ["bind_authoritative_formal_manifest_outside_derived_prior_root"],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(payload: dict[str, Any]) -> str:
    clean = {key: value for key, value in payload.items() if key != "canonical_sha256"}
    raw = json.dumps(clean, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def set_option_values(command: list[str], option: str, values: list[str]) -> None:
    index = command.index(option)
    end = index + 1
    while end < len(command) and not command[end].startswith("-"):
        end += 1
    command[index + 1 : end] = values


def corrected_command(method: str, spec: dict[str, Any]) -> list[str]:
    command = copy.deepcopy(spec["command"])
    if method in {"2dgs", "rade_gs"}:
        set_option_values(command, "--test_iterations", ["30001"])
        if method == "rade_gs":
            set_option_values(
                command, "--checkpoint_iterations", ["15000", "30000"]
            )
            command = [
                command[0],
                "-B",
                "{repo}/code/gcp/run_rade_gs_checkpoint_first_rescue.py",
                "--source-root",
                "{source_root}",
                "--expected-train-sha256",
                spec["files"]["train.py"],
                "--",
                *command[3:],
            ]
    elif method == "citygaussian_v2":
        command.extend(
            [
                "--sequential_blocks",
                "--resume_from",
                CITYGAUSSIAN_RESUME,
                "--resume_manifest",
                CITYGAUSSIAN_RESUME_MANIFEST,
            ]
        )
    elif method == "citygs_x":
        command.append("--defer_evaluation")
    elif method == "metrogs":
        command.extend(["--formal_input_manifest", FORMAL_MANIFEST])
    elif method == "gsprior":
        command = [
            command[0],
            "-B",
            "{repo}/code/gcp/run_gsprior_lazy_preload_rescue.py",
            "--source-root",
            "{source_root}",
            "--expected-train-sha256",
            spec["files"]["train.py"],
            "--",
            *command[3:],
        ]
    return command


def training_environment(method: str) -> dict[str, str]:
    env = dict(COMMON_ENV)
    if method == "pgsr":
        env["PYTHONPATH"] = (
            "{repo}/compat/pgsr/pytorch3d_transforms_minimal_v1"
        )
    elif method == "gsprior":
        env["PYTHONPATH"] = (
            "{repo}/compat/gsprior/pytorch3d_transforms_minimal_v1:"
            "{repo}/code/gcp"
        )
    return env


def source_root(method: str, spec: dict[str, Any]) -> str:
    return (
        "/root/autodl-tmp/worktrees/m3m-gcp-native-quarter/"
        f"{method}/{spec['commit']}/{spec['sub']}"
    )


def prepared_input_binding(
    method: str, spec: dict[str, Any], dataset: str
) -> dict[str, Any]:
    if method in {"citygaussian_v2", "citygs_x"}:
        profile = "city_train_records_with_full_all_image_sfm_points"
        sparse = {
            "cameras.bin": FORMAL_CAMERAS_SHA,
            "images.bin": CITY_IMAGES_SHA,
            "points3D.bin": FULL_POINTS_SHA,
            "points3D.ply": INITIAL_PLY_SHA,
        }
    elif method == "metrogs":
        profile = "metrogs_reciprocal_train_track_closure_after_all_image_sfm"
        sparse = {
            "cameras.bin": FORMAL_CAMERAS_SHA,
            "images.bin": CITY_IMAGES_SHA,
            "points3D.bin": METRO_POINTS_SHA,
            "points3D.ply": INITIAL_PLY_SHA,
        }
    else:
        profile = "exact_formal_train_view_from_shared_all_image_sfm"
        sparse = {
            "cameras.bin": FORMAL_CAMERAS_SHA,
            "images.bin": FORMAL_IMAGES_SHA,
            "points3D.ply": INITIAL_PLY_SHA,
        }
    return {
        "evidence_path": METHOD_INPUT_EVIDENCE,
        "evidence_sha256": METHOD_INPUT_EVIDENCE_SHA,
        "dataset_root": FORMAL if method == "gsprior" else dataset,
        "input_profile": profile,
        "sparse_sha256": sparse,
        "all_image_sfm_precedes_train_test_split": True,
    }


def input_validation_dependencies(method: str) -> dict[str, str]:
    paths = ["code/gcp/materialize_m3m_gcp_100k_method_inputs.py"]
    if method == "rade_gs":
        paths.extend(
            [
                "code/gcp/run_rade_gs_checkpoint_first_rescue.py",
                "code/gcp/materialize_rade_gs_ply_from_checkpoint.py",
                "code/gcp/smoke_rade_gs_serialization_rescue.py",
                "docs/protocol_evidence/m3m_gcp_100k_rade_gs_final_serialization_rescue_v1.json",
            ]
        )
    elif method == "gsprior":
        paths.extend(
            [
                "code/gcp/run_gsprior_lazy_preload_rescue.py",
                "code/gcp/smoke_gsprior_lazy_preload_equivalence.py",
                "docs/protocol_evidence/m3m_gcp_100k_gsprior_lazy_preload_rescue_v1.json",
                "docs/protocol_evidence/m3m_gcp_100k_gsprior_lazy_preload_smoke_v1.json",
            ]
        )
    if method in {"citygaussian_v2", "citygs_x"}:
        paths.append(
            "code/gcp/materialize_colmap_train_track_compatibility_streaming.py"
        )
    elif method == "metrogs":
        paths.append("code/gcp/filter_colmap_model_to_frozen_train_streaming.py")
    if method == "citygaussian_v2":
        paths.append(
            "configs/m3m_gcp_native_quarter_citygaussian_v2_100k_resume_v1.json"
        )
    return {path: sha256(ROOT / path) for path in paths}


def build_recipe(method: str) -> dict[str, Any]:
    if method == "3dgs_original":
        payload: dict[str, Any] = {
            "schema": "m3m_gcp_100k_qualification_recipe_v1",
            "status": "REVIEW_CANDIDATE_NOT_EXECUTION_AUTHORIZED",
            "plan_id": PLAN_ID,
            "method_id": method,
            "scene": SCENE,
            "seed": 0,
            "input_class": REUSE_3DGS["input_class"],
            "budget": REUSE_3DGS["budget"],
            "corrections": CORRECTIONS[method],
            "training": None,
            "reuse_model": {
                "run_root": REUSED_3DGS,
                "source_commit": REUSE_3DGS["commit"],
                "source_tree": REUSE_3DGS["tree"],
                "point_cloud_relative_path": (
                    "model/point_cloud/iteration_30000/point_cloud.ply"
                ),
                "bytes": 2_340_432_588,
                "sha256": (
                    "8d92360186d268d0e20a0e328122e8c2679cddd0c2d539c27a918ee4c972e1f5"
                ),
                "retrain_allowed": False,
            },
        }
    else:
        spec = METHODS[method]
        dataset = spec.get(
            "dataset",
            f"/root/autodl-tmp/datasets/M3M-GCP-colmap-native-quarter-v1/"
            f"formal_inputs/{SCENE}/train",
        )
        payload = {
            "schema": "m3m_gcp_100k_qualification_recipe_v1",
            "status": "REVIEW_CANDIDATE_NOT_EXECUTION_AUTHORIZED",
            "plan_id": PLAN_ID,
            "method_id": method,
            "scene": SCENE,
            "seed": 0,
            "input_class": spec["input_class"],
            "budget": spec["budget"],
            "corrections": CORRECTIONS[method],
            "training": {
                "attempt_root_template": f"{RUN_ROOT}/{method}/attempt-{{attempt_id}}",
                "benchmark_repo": BENCHMARK_REPO,
                "source_root": source_root(method, spec),
                "dataset_root": dataset,
                "prior_root": dataset,
                "working_directory": source_root(method, spec),
                "interpreter": ENV.format(method=method),
                "environment": training_environment(method),
                "command": corrected_command(method, spec),
                "source_binding": {
                    "commit": spec["commit"],
                    "tree": spec["tree"],
                    "required_status": spec["status"],
                    "required_files_sha256": spec["files"],
                },
                "resource_probe": {
                    "script": "{repo}/code/gcp/run_with_resource_probe_v2.py",
                    "contract": "{repo}/configs/gs_gcp_resource_probe_contract_v2.json",
                    "gpu_indices": "0",
                    "time_binary": GNU_TIME,
                    "enforce_contract_gates": False,
                    **({"timeout_seconds": 43_200} if method == "gsprior" else {}),
                },
                "materializations": (
                    [
                        {
                            "relative_path": "formal_training_config.yaml",
                            "content": spec["materialization"],
                        }
                    ]
                    if "materialization" in spec
                    else []
                ),
            },
        }

    payload["scientific_contract"] = {
        "protocol_id": "m3m_gcp_native_quarter_geometry_v2",
        "formal_input_manifest": {
            "path": FORMAL_MANIFEST,
            "file_sha256": (
                "c2cf9e951d95fee12a28d942e95c5c420df55bc364738b3f8737fed1c78bef3d"
            ),
            "canonical_sha256": (
                "5b4fe34743310bd2225feb2dd236200606be933002fec19d2c9ecb9f3ba6769d"
            ),
            "train_views": 2196,
            "heldout_views": 314,
        },
        "truth_access": {"gcp_training": False, "lidar_training": False},
        "result_driven_tuning": "FORBIDDEN",
        "evaluation_after_saved_model": True,
    }
    input_spec = REUSE_3DGS if method == "3dgs_original" else METHODS[method]
    input_dataset = input_spec.get("dataset", FORMAL)
    payload["formal_input_manifest"] = dict(
        payload["scientific_contract"]["formal_input_manifest"]
    )
    payload["prepared_method_input_binding"] = prepared_input_binding(
        method, input_spec, input_dataset
    )
    payload["benchmark_required_files_sha256"] = input_validation_dependencies(
        method
    )
    payload["phase_commands"] = (
        {"prior": copy.deepcopy(METHODS[method]["prior_command"])}
        if method == "gsprior"
        else {}
    )
    payload["retry_policy"] = {
        "engineering_failure": "correct_and_use_new_attempt_id_without_per_method_audit",
        "deterministic_oom_with_telemetry": "terminal_on_current_hardware_and_config",
        "ambiguous_sigkill_without_telemetry": "one_diagnostic_repeat_with_telemetry",
        "evaluation_only_failure": "repair_adapter_and_reuse_saved_model",
        "scientific_contract_change": "stop_before_execution_and_batch_review",
        "metric_based_attempt_selection": "FORBIDDEN",
    }
    payload["promotion_policy"] = {
        "eligible_after_post_batch_review": True,
        "requires_retraining": False,
        "requires_scientific_contract_unchanged": True,
        "metric_based_promotion": "FORBIDDEN",
    }
    payload["historical_attempts"] = {
        "status": (
            "VALIDATED_MODEL_REUSE_SOURCE"
            if method == "3dgs_original"
            else "DIAGNOSTIC_SUPERSEDED_FOR_EXECUTION"
        ),
        "delete": False,
        "rank": method == "3dgs_original",
    }
    payload["canonical_sha256"] = canonical(payload)
    return payload


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for method in METHOD_ORDER:
        payload = build_recipe(method)
        path = OUT / f"{method}.json"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        rows.append(
            {
                "method_id": method,
                "path": path.relative_to(ROOT).as_posix(),
                "file_sha256": sha256(path),
                "canonical_sha256": payload["canonical_sha256"],
            }
        )

    plan = {
        "schema": "m3m_gcp_100k_qualification_plan_v1",
        "status": "REVIEW_CANDIDATE_NOT_EXECUTION_AUTHORIZED",
        "plan_id": PLAN_ID,
        "scene": SCENE,
        "seed": 0,
        "plan_document": PLAN_DOC,
        "plan_document_sha256": sha256(ROOT / PLAN_DOC),
        "method_order": METHOD_ORDER,
        "recipes": rows,
        "audit_schedule": [
            "one_prelaunch_batch_review",
            "one_post_100k_outcome_batch_review",
            "one_final_six_scene_protocol_freeze",
        ],
        "per_method_audit": "NOT_REQUIRED_UNLESS_SCIENTIFIC_RED_LINE",
        "old_recipe_status": "HISTORICAL_DIAGNOSTIC_NOT_CURRENT_EXECUTION_POINTER",
    }
    plan["canonical_sha256"] = canonical(plan)
    output = ROOT / "configs" / "m3m_gcp_native_quarter_100k_qualification_v1.json"
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
