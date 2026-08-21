#!/usr/bin/env python3
"""Build the exact review-candidate 100K time/space execution plan."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from m3m_gcp_100k_source_binding_correction import (
    validate_source_binding_correction,
)


ROOT = Path(__file__).resolve().parents[2]
SCENE = "gcp_100000_20260610"
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
LOCKED_SCENES = [
    "gcp_5000_20260602",
    "gcp_20000_20260602",
    "gcp_10000_20260610",
    "gcp_50000_20260610",
]
METHOD_INPUT_PREPARATION_EVIDENCE = (
    "/root/autodl-tmp/runs/m3m-gcp-native-quarter/preparation/"
    "gcp_100000_20260610/per-method-inputs-v2.json"
)
METHOD_INPUT_PREPARATION_EVIDENCE_SHA = "080a1ef97ab5caadca70420d6e34b57681d793f874b2a43511480fbc09b30ab1"
EVALUATION_CAMERA_EVIDENCE = (
    "/root/autodl-tmp/runs/m3m-gcp-native-quarter/preparation/"
    "gcp_100000_20260610/evaluation-camera-root-v1.json"
)
EVALUATION_CAMERA_EVIDENCE_SHA = "6b31e460ba80b17e85ac284c55165bfbc6c6b3a85411ad88e785ed8fe6645aac"
OBSOLETE_ATTEMPT_CLEANUP = (
    "/root/autodl-tmp/runs/m3m-gcp-native-quarter/preparation/"
    "gcp_100000_20260610/obsolete-train-first-undistorter-cleanup-v1.json"
)
OBSOLETE_ATTEMPT_CLEANUP_SHA = "44b6722ff586d1c17aa8bdfd57fd5e926acb2cbd961a220d739386acc3242c51"
PHASE1_REVIEW_COMMIT = "e9c3414b808b374bd8632a45ee965e3f6acc1ac0"
PHASE1_REVIEW_TREE = "d1d6c73852e42bc02c519d0853e26c114dcb1f8f"
PHASE1_REVIEW_VERDICT = "PASS_LIDAR_V1_AND_SIX_SCENE_PREPARATION_V2"
SUPERSESSION_RECEIPT = (
    "docs/protocol_evidence/"
    "m3m_gcp_100k_activation_v1_infrastructure_supersession.json"
)
CONTINUITY_RECEIPT = (
    "docs/protocol_evidence/"
    "m3m_gcp_100k_activation_v2_to_v3_continuity.json"
)
SOURCE_BINDING_CORRECTION_RECEIPT = (
    "docs/protocol_evidence/"
    "m3m_gcp_100k_3dgs_linux_source_binding_correction_v1.json"
)
PREVIOUS_PLAN = "configs/m3m_gcp_native_quarter_100k_ten_method_execution_plan_v2.json"
RECIPE_MANIFEST = "configs/m3m_gcp_native_quarter_100k_recipe_manifest_v3.json"
REQUIRED_NOFILE_SOFT = 65536


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(payload: dict[str, Any]) -> str:
    clean = {key: value for key, value in payload.items() if key != "canonical_sha256"}
    raw = json.dumps(clean, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def repo_file(path: str) -> dict[str, str]:
    return {"path": path, "sha256": sha(ROOT / path)}


def main() -> int:
    recipe_manifest_path = ROOT / RECIPE_MANIFEST
    recipe_manifest = json.loads(recipe_manifest_path.read_text(encoding="utf-8"))
    if recipe_manifest.get("method_order") != METHOD_ORDER:
        raise RuntimeError("ten-recipe manifest order differs from frozen active pool")
    supersession_path = ROOT / SUPERSESSION_RECEIPT
    supersession = json.loads(supersession_path.read_text(encoding="utf-8"))
    if (
        supersession.get("status")
        != "SUPERSEDED_INFRASTRUCTURE_INVALID_NOT_RANKABLE"
        or supersession.get("classification", {}).get("algorithm_failure") is not False
        or supersession.get("classification", {}).get("formal_retry_counted") is not False
        or supersession.get("classification", {}).get("rankable") is not False
        or supersession.get("canonical_sha256") != canonical(supersession)
    ):
        raise RuntimeError("v1 infrastructure supersession receipt is not sealed")
    continuity_path = ROOT / CONTINUITY_RECEIPT
    continuity = json.loads(continuity_path.read_text(encoding="utf-8"))
    previous_plan_path = ROOT / PREVIOUS_PLAN
    previous_plan = json.loads(previous_plan_path.read_text(encoding="utf-8"))
    if (
        continuity.get("schema") != "m3m_gcp_100k_activation_continuity_v1"
        or continuity.get("status") != "SEALED_V2_TO_V3_CONTINUITY"
        or continuity.get("canonical_sha256") != canonical(continuity)
        or previous_plan.get("schema")
        != "m3m_gcp_native_quarter_100k_ten_method_execution_plan_v2"
        or previous_plan.get("canonical_sha256") != canonical(previous_plan)
    ):
        raise RuntimeError("v2-to-v3 continuity receipt is not sealed")
    payload: dict[str, Any] = {
        "schema": "m3m_gcp_native_quarter_100k_ten_method_execution_plan_v3",
        "plan_id": "m3m-gcp-native-quarter-100k-ten-method-seed0-v3-continuation",
        "status": "REVIEW_CANDIDATE_NOT_EXECUTION_AUTHORIZED",
        "execution_authorized": False,
        "activation_manifest_path": (
            "/root/autodl-tmp/runs/m3m-gcp-native-quarter/"
            "formal-100k-v2/activation_v3.json"
        ),
        "scene": SCENE,
        "seed": 0,
        "train_view_count": 2196,
        "heldout_view_count": 314,
        "method_order": METHOD_ORDER,
        "other_prepared_scenes_locked": LOCKED_SCENES,
        "other_prepared_scene_training_rendering_or_formal_evaluation_authorized": False,
        "execution_revision_note": repo_file(
            "docs/M3M_GCP_100K_TEN_METHOD_TIME_SPACE_EXECUTION_PLAN_V3.md"
        ),
        "activation_continuity": {
            "receipt": repo_file(CONTINUITY_RECEIPT),
            "status_required": "SEALED_V2_TO_V3_CONTINUITY",
            "previous_execution_plan": {
                "path": PREVIOUS_PLAN,
                "bytes": previous_plan_path.stat().st_size,
                "sha256": sha(previous_plan_path),
                "canonical_sha256": previous_plan["canonical_sha256"],
            },
            "execution_plan_v2_bytes_unchanged": True,
            "recipe_manifest_v2_bytes_unchanged": True,
            "remote_artifacts_must_remain_byte_identical": True,
            "inherited_final_methods_forbidden_to_launch": ["2dgs"],
            "pgsr_prechild_rejection_consumed_attempt": False,
            "continued_run_namespace": (
                "/root/autodl-tmp/runs/m3m-gcp-native-quarter/formal-100k-v2"
            ),
            "final_attempt_freeze_authorization": "activation_v3_only",
        },
        "source_binding_correction": {
            "receipt": repo_file(SOURCE_BINDING_CORRECTION_RECEIPT),
            "status_required": "SEALED_LINUX_IDENTITY_METADATA_CORRECTION",
            "type": "LINUX_IDENTITY_METADATA_CORRECTION_ONLY",
            "source_modified": False,
            "child_started": False,
            "attempt_consumed": False,
            "dual_hash_tolerance": False,
            "recipe_manifest": {
                "path": RECIPE_MANIFEST,
                "sha256": sha(recipe_manifest_path),
                "canonical_sha256": recipe_manifest["canonical_sha256"],
            },
        },
        "superseded_activation": {
            "receipt": repo_file(SUPERSESSION_RECEIPT),
            "status_required": "SUPERSEDED_INFRASTRUCTURE_INVALID_NOT_RANKABLE",
            "algorithm_failure": False,
            "formal_retry_counted": False,
            "rankable": False,
            "remote_artifacts_must_remain_byte_identical": True,
        },
        "formal_lidar_protocol": {
            "contract": repo_file("configs/m3m_gcp_lidar_formal_v1.json"),
            "artifact_schema": repo_file(
                "configs/m3m_gcp_lidar_formal_artifact_schema_v1.json"
            ),
            "execution_authorized": False,
            "phase1_review": {
                "task_id": "019ff12c-cb29-7cb2-8fb6-1d82c5f8c54b",
                "verdict": PHASE1_REVIEW_VERDICT,
                "reviewed_commit": PHASE1_REVIEW_COMMIT,
                "reviewed_tree": PHASE1_REVIEW_TREE,
                "protocol_pass_alone_authorizes_execution": False,
            },
        },
        "reuse": {
            "method_id": "3dgs_original",
            "run_root": "/root/autodl-tmp/runs/m3m-gcp-native-quarter/3dgs-original/gcp_100000_20260610/seed0-30k-20260810T175634Z",
            "model_checkpoint_relative_path": "model/point_cloud/iteration_30000/point_cloud.ply",
            "model_checkpoint_bytes": 2340432588,
            "model_checkpoint_sha256": "8d92360186d268d0e20a0e328122e8c2679cddd0c2d539c27a918ee4c972e1f5",
            "retrain_allowed": False,
            "packet_export_recipe_required": True,
        },
        "budgets": {
            "2dgs": {"type": "iterations", "value": 30000},
            "pgsr": {"type": "iterations", "value": 30000},
            "rade_gs": {"type": "iterations", "value": 30000},
            "qgs": {"type": "iterations", "value": 30000},
            "gsprior": {"type": "iterations", "value": 40000},
            "sof": {"type": "iterations", "value": 30000},
            "citygaussian_v2": {
                "type": "official_matrixcity_aerial_4x4_two_stage",
                "coarse_steps": 30000,
                "fine_steps": 60000,
            },
            "citygs_x": {"type": "iterations", "value": 100000},
            "metrogs": {
                "type": "effective_image_iterations",
                "value": 150000,
                "optimizer_steps": 37500,
            },
        },
        "preparation": {
            "six_scene_common_remote_evidence": repo_file(
                "docs/protocol_evidence/m3m_gcp_six_scene_common_preparation_remote_v2.json"
            ),
            "six_scene_common_local_evidence": repo_file(
                "docs/protocol_evidence/m3m_gcp_six_scene_common_preparation_local_v2.json"
            ),
            "train_view_allowlist_manifest": repo_file(
                "configs/m3m_gcp_lidar_train_view_allowlists_v1.json"
            ),
            "shared_all_image_sfm": {
                "path": "/root/autodl-tmp/datasets/M3M-GCP-colmap-native-quarter-full-model-v1/gcp_100000_20260610/sparse/0",
                "image_count": 2510,
                "point_count": 1262896,
                "cameras_sha256": "6669584ba1ba326cf5b372b878a5abf182f8cfe0bfe0845da3a0c4f7aed8fe5e",
                "images_sha256": "57163927bceee6ca330c113c9caf06cafe1a84a7ca21ac0f055680dcbe8eff6e",
                "points3d_sha256": "09fc811f32558a11a47bada7393bf7bce2585cbe68eb4872ffce72025b0fc9aa",
                "split_applied_after_sfm": True,
            },
            "per_method_input_profiles": {
                "formal_train_view": [
                    "3dgs_original", "2dgs", "pgsr", "rade_gs", "qgs", "sof"
                ],
                "gsprior_formal_train_view_then_deterministic_camera_normalization": [
                    "gsprior"
                ],
                "city_train_records_with_full_all_image_sfm_points": [
                    "citygaussian_v2", "citygs_x"
                ],
                "metrogs_reciprocal_train_track_closure_after_all_image_sfm": [
                    "metrogs"
                ],
            },
            "per_method_input_evidence": {
                "path": METHOD_INPUT_PREPARATION_EVIDENCE,
                "sha256": METHOD_INPUT_PREPARATION_EVIDENCE_SHA,
                "status_required": "PASS_PER_METHOD_INPUT_PREPARATION_NO_TRAINING_NO_PRIOR",
                "all_image_sfm_precedes_train_test_split": True,
            },
            "evaluation_camera_root": {
                "path": "/root/autodl-tmp/datasets/M3M-GCP-100K-evaluation-camera-root-v1/gcp_100000_20260610",
                "evidence_path": EVALUATION_CAMERA_EVIDENCE,
                "evidence_sha256": EVALUATION_CAMERA_EVIDENCE_SHA,
                "status_required": "PASS_EVALUATION_CAMERA_ROOT_NO_TRAINING_NO_PRIOR_NO_EVALUATION",
                "view_count": 2196,
                "points2d_tracks_present": False,
                "points3d_bin_point_count": 0,
                "points3d_bin_sha256": "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc",
                "purpose": "evaluation-only all-train camera loader; never training or prior input",
            },
            "obsolete_train_first_attempt_cleanup": {
                "path": OBSOLETE_ATTEMPT_CLEANUP,
                "sha256": OBSOLETE_ATTEMPT_CLEANUP_SHA,
                "status_required": "PASS_OBSOLETE_FAILED_ATTEMPT_REMOVED",
            },
            "formal_training_started": True,
            "external_prior_generation_started": False,
            "formal_lidar_evaluation_started": False,
        },
        "recipe_manifest": {
            "path": RECIPE_MANIFEST,
            "file_sha256": sha(recipe_manifest_path),
            "canonical_sha256": recipe_manifest["canonical_sha256"],
            "method_order": METHOD_ORDER,
            "recipe_count": 10,
            "new_training_recipes": 9,
            "reused_model_packet_only_recipes": 1,
        },
        "execution_closure": {
            "activation_builder": repo_file(
                "code/gcp/build_m3m_gcp_lidar_100k_activation.py"
            ),
            "attempt_manifest_builder": repo_file(
                "code/gcp/build_m3m_gcp_100k_attempt_manifest.py"
            ),
            "activation_continuity_validator": repo_file(
                "code/gcp/m3m_gcp_100k_continuity.py"
            ),
            "source_binding_correction_validator": repo_file(
                "code/gcp/m3m_gcp_100k_source_binding_correction.py"
            ),
            "recipe_manifest_v3_builder": repo_file(
                "code/gcp/build_m3m_gcp_100k_recipe_manifest_v3.py"
            ),
            "guarded_runner": repo_file("code/gcp/run_m3m_gcp_100k_guarded.py"),
            "phase_product_validator": repo_file(
                "code/gcp/m3m_gcp_100k_phase_products.py"
            ),
            "attempt_freezer": repo_file(
                "code/gcp/freeze_m3m_gcp_lidar_scene_attempts.py"
            ),
            "method_input_materializer": repo_file(
                "code/gcp/materialize_m3m_gcp_100k_method_inputs.py"
            ),
            "evaluation_camera_root_materializer": repo_file(
                "code/gcp/materialize_m3m_gcp_100k_evaluation_camera_root.py"
            ),
            "packet_export_dispatcher": repo_file(
                "code/gcp/run_m3m_gcp_100k_packet_export.py"
            ),
            "post_sfm_track_compatibility_materializers": [
                repo_file("code/gcp/filter_colmap_model_to_frozen_train_streaming.py"),
                repo_file(
                    "code/gcp/materialize_colmap_train_track_compatibility_streaming.py"
                ),
            ],
            "exact_review_verdict_required": "PASS_100K_TIME_SPACE_EXECUTION_PLAN_V1",
            "exact_reviewed_commit_and_tree_required": True,
            "foreign_gpu_process_gate": "no compute process may exist before prior/training/packet launch",
            "progress_monitor_bound_by_each_recipe": True,
            "structured_failure_evidence_required": True,
            "prior_and_training_require_absent_run_root_at_guard_admission": True,
            "training_child_must_create_products_inside_new_run_root": True,
            "zero_exit_requires_phase_product_postvalidation": True,
            "prior_phase_success_and_product_required_before_training": True,
            "ready_model_identity_requires_exact_phase_success_markers": True,
            "phase_success_command_rehashed_against_frozen_recipe": True,
            "rlimit_nofile_soft_required_for_child_phases": REQUIRED_NOFILE_SOFT,
            "rlimit_nofile_hard_minimum_prechild_gate": REQUIRED_NOFILE_SOFT,
            "rlimit_nofile_parent_before_after_evidence_required": True,
            "rlimit_nofile_child_actual_inheritance_evidence_required": True,
        },
        "attempt_freeze": {
            "execution_plan_path": "configs/m3m_gcp_native_quarter_100k_ten_method_execution_plan_v3.json",
            "recipe_manifest_path": RECIPE_MANIFEST,
            "method_registry_path": "configs/m3m_gcp_native_quarter_method_registry_v3.json",
            "attempt_manifest_path": "/root/autodl-tmp/runs/m3m-gcp-native-quarter/formal-100k-v2/scene_attempts_v3.json",
            "scene_attempt_freeze_path": "/root/autodl-tmp/runs/m3m-gcp-native-quarter/formal-100k-v2/scene_attempt_freeze_v3.json",
            "model_identity_root": "/root/autodl-tmp/runs/m3m-gcp-native-quarter/formal-100k-v2/model-identities-v3",
            "exclusive_create_no_replace": True,
            "frozen_before_any_formal_lidar_result": True,
            "shared_freeze_sha_required_by_authorization_result_verifier_ranker_and_archive": True,
            "attempt_statuses": [
                "READY_FOR_EVALUATION",
                "OOM_UNRANKED",
                "FAILED_UNRANKED",
            ],
        },
        "rolling_packet_lifecycle": {
            "simultaneous_raw_packet_sets_max": 1,
            "persistent_exclusive_packet_mutex": True,
            "selected_method_all_train_views_required": True,
            "per_method_pre_result_authorization_required": True,
            "independent_verification_required_before_packet_deletion": True,
            "full_archive_inventory_byte_reverification_required_before_packet_deletion": True,
            "final_models_and_distance_arrays_retained_on_901": True,
        },
        "storage": {
            "packet_scratch_hard_cap_gib": 100,
            "minimum_free_before_prior_gib": 300,
            "minimum_free_before_training_gib": 300,
            "minimum_free_before_packet_export_gib": 180,
            "all_ten_packet_sets_simultaneously_forbidden": True,
            "duplicate_rgb_policy": "symlink/hardlink only; no per-method RGB copy",
            "prior_generation_policy": "just in time immediately before its method",
            "transient_checkpoint_deletion_requires_final_artifact_hash_inventory": True,
        },
        "time": {
            "existing_3dgs_packet_seconds_per_view": 3.510931,
            "successful_method_packet_export_hours_range": [2.0, 3.5],
            "whole_queue_wall_days_range": [5.5, 12],
            "working_expectation_days_range": [7, 10],
            "estimate_is_not_early_stop_or_budget": True,
        },
        "retry_policy": {
            "pre_child_guard_rejection": "not an attempt; may relaunch only after the exact guard cause is corrected",
            "child_started_any_exit_including_zero_progress_or_oom": "final for that method",
            "superseded_v1_infrastructure_event": "not an algorithm attempt and never rankable; only the reviewer-authorized fresh v2 namespace may restart at 2dgs",
            "activation_v2_to_v3_continuity": "2dgs FAILED_UNRANKED is inherited as final; PGSR pre-child rejection consumed no attempt; activation_v3 forbids 2dgs relaunch",
            "recipe_resolution_partition_loss_checkpoint_or_budget_change": "FORBIDDEN",
            "result_driven_retry": "FORBIDDEN",
        },
        "truth_deny": {
            "heldout_rgb_visible_to_training": False,
            "gcp_visible_to_training_prior_or_selection": False,
            "lidar_visible_to_training_prior_or_selection": False,
            "result_driven_retry_or_recipe_change": False,
        },
        "review": {
            "task_id": "019ff12c-cb29-7cb2-8fb6-1d82c5f8c54b",
            "required_pass_verdict": "PASS_100K_TIME_SPACE_EXECUTION_PLAN_V1",
            "verdict": "PENDING",
            "exact_clean_commit_and_tree_binding_required": True,
        },
    }
    validate_source_binding_correction(repo=ROOT, plan=payload)
    payload["canonical_sha256"] = canonical(payload)
    output = ROOT / "configs" / "m3m_gcp_native_quarter_100k_ten_method_execution_plan_v3.json"
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
