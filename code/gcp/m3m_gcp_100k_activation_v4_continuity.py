#!/usr/bin/env python3
"""Validate the immutable activation-v3 to activation-v4 continuity chain."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from m3m_gcp_lidar_artifacts import canonical_sha256, sha256_file
from m3m_gcp_100k_continuity import validate_activation_continuity
from m3m_gcp_100k_postattempt_closure import validate_postattempt_closure


SCENE = "gcp_100000_20260610"
RECEIPT_SCHEMA = "m3m_gcp_100k_activation_continuity_v2"
RECEIPT_STATUS = "SEALED_V3_TO_V4_GUARD_CONTINUITY"
PREVIOUS_COMMIT = "e33368db9333f826a3e808ff00c437c1a6c63b82"
PREVIOUS_TREE = "4620a434bd081af9274fdfc37dbb0d673636edfc"
PREVIOUS_PLAN = "configs/m3m_gcp_native_quarter_100k_ten_method_execution_plan_v3.json"
PREVIOUS_RECIPES = "configs/m3m_gcp_native_quarter_100k_recipe_manifest_v3.json"
FAILED_OUTCOMES = {
    "2dgs": "FAILED_UNRANKED",
    "pgsr": "FAILED_UNRANKED",
    "rade_gs": "FAILED_UNRANKED",
    "qgs": "OOM_UNRANKED",
    "gsprior": "FAILED_UNRANKED",
    "sof": "FAILED_UNRANKED",
    "citygaussian_v2": "FAILED_UNRANKED",
    "citygs_x": "FAILED_UNRANKED",
}
REMOTE_ROLES = {
    "activation_v3",
    "3dgs_reused_model",
    *{f"{method_id}_failure" for method_id in FAILED_OUTCOMES},
    "metrogs_prior_phase_success",
    "metrogs_prior_environment",
    "metrogs_training_priors",
    "metrogs_prior_pass_marker",
    "metrogs_prior_merged_ply",
    "metrogs_training_prechild_guard_console",
}
REPOSITORY_ROLES = {
    "execution_plan_v3",
    "recipe_manifest_v3",
    "execution_note_v3",
}
PRELAUNCH_FRESH = "PRELAUNCH_FRESH"
POSTATTEMPT_TERMINAL = "POSTATTEMPT_TERMINAL"
CONTINUITY_MODES = {PRELAUNCH_FRESH, POSTATTEMPT_TERMINAL}


def _bound_path(repo: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else repo / path


def _role_map(
    rows: object, expected_roles: set[str], label: str
) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list):
        raise RuntimeError(f"{label} inventory is not a list")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("role"), str):
            raise RuntimeError(f"{label} inventory row is invalid")
        role = str(row["role"])
        if role in result:
            raise RuntimeError(f"{label} inventory contains duplicate role: {role}")
        result[role] = row
    if set(result) != expected_roles:
        raise RuntimeError(f"{label} inventory role mismatch")
    return result


def _validate_file(path: Path, row: dict[str, Any], label: str) -> None:
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != int(row.get("bytes", -1))
        or sha256_file(path) != row.get("sha256")
    ):
        raise RuntimeError(f"{label} changed or disappeared: {path}")


def _validate_failure(
    method_id: str, expected_status: str, row: dict[str, Any]
) -> None:
    path = Path(str(row.get("path", "")))
    _validate_file(path, row, f"inherited {method_id} failure")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != "m3m_gcp_lidar_failure_evidence_v1"
        or payload.get("scene") != SCENE
        or payload.get("method_id") != method_id
        or payload.get("status") != expected_status
        or payload.get("failure_stage") != "training"
        or payload.get("canonical_sha256") != canonical_sha256(payload)
    ):
        raise RuntimeError(f"inherited {method_id} terminal outcome mismatch")
    if expected_status == "OOM_UNRANKED":
        if payload.get("oom_signal") != "CUDA_OUT_OF_MEMORY":
            raise RuntimeError("inherited QGS OOM classification mismatch")
    elif payload.get("oom_signal") is not None:
        raise RuntimeError(f"inherited {method_id} unexpected OOM classification")
    for path_key, sha_key, label in (
        ("environment_manifest_path", "environment_manifest_sha256", "environment"),
        ("stdout_path", "stdout_sha256", "stdout"),
        ("stderr_path", "stderr_sha256", "stderr"),
    ):
        evidence_path = Path(str(payload.get(path_key, "")))
        if (
            not evidence_path.is_absolute()
            or not evidence_path.is_file()
            or evidence_path.is_symlink()
            or sha256_file(evidence_path) != payload.get(sha_key)
        ):
            raise RuntimeError(f"inherited {method_id} {label} binding mismatch")


def validate_prelaunch_fresh_state(rejection: dict[str, Any]) -> None:
    """Require the exact still-unused MetroGS attempt authorized by activation-v4."""

    run_root = Path(str(rejection.get("run_root", "")))
    failure_path = Path(str(rejection.get("failure_path", "")))
    phase_success_path = failure_path.parent / "phase_success.json"
    if (
        not run_root.is_absolute()
        or not failure_path.is_absolute()
        or run_root.exists()
        or failure_path.exists()
        or phase_success_path.exists()
    ):
        raise RuntimeError("MetroGS pre-child rejection did not preserve a fresh attempt")


def validate_activation_v4_continuity(
    *, repo: Path, plan: dict[str, Any], method_id: str | None = None,
    phase: str | None = None, mode: str,
    postattempt_receipt: Path | None = None,
) -> dict[str, Any]:
    """Validate the historical chain in one explicit lifecycle mode."""

    repo = repo.resolve()
    if mode not in CONTINUITY_MODES:
        raise RuntimeError("100K continuity mode must be explicit and recognized")
    binding = plan.get("activation_continuity", {})
    receipt_row = binding.get("receipt", {})
    receipt_path = _bound_path(repo, receipt_row.get("path", "")).resolve()
    if (
        not receipt_path.is_file()
        or receipt_path.is_symlink()
        or sha256_file(receipt_path) != receipt_row.get("sha256")
    ):
        raise RuntimeError("v3-to-v4 continuity receipt identity mismatch")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("schema") != RECEIPT_SCHEMA
        or receipt.get("status") != RECEIPT_STATUS
        or receipt.get("status") != binding.get("status_required")
        or receipt.get("scene") != SCENE
        or receipt.get("canonical_sha256") != canonical_sha256(receipt)
        or binding.get("remote_artifacts_must_remain_byte_identical") is not True
    ):
        raise RuntimeError("v3-to-v4 continuity receipt classification mismatch")

    repository_rows = _role_map(
        receipt.get("repository_artifacts"), REPOSITORY_ROLES, "repository continuity"
    )
    repository_paths: dict[str, Path] = {}
    for role, row in repository_rows.items():
        relative = Path(str(row.get("path", "")))
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise RuntimeError(f"unsafe repository continuity path: {relative}")
        path = (repo / relative).resolve()
        _validate_file(path, row, f"repository continuity artifact {role}")
        repository_paths[role] = path
        expected_canonical = row.get("canonical_sha256")
        if expected_canonical is not None:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (
                payload.get("canonical_sha256") != expected_canonical
                or canonical_sha256(payload) != expected_canonical
            ):
                raise RuntimeError(f"repository continuity canonical mismatch: {role}")

    previous_plan_path = repository_paths["execution_plan_v3"]
    previous_recipes_path = repository_paths["recipe_manifest_v3"]
    if (
        previous_plan_path != (repo / PREVIOUS_PLAN).resolve()
        or previous_recipes_path != (repo / PREVIOUS_RECIPES).resolve()
    ):
        raise RuntimeError("v3 repository artifact path mismatch")
    previous_plan = json.loads(previous_plan_path.read_text(encoding="utf-8"))
    previous_recipes = json.loads(previous_recipes_path.read_text(encoding="utf-8"))
    if (
        previous_plan.get("schema")
        != "m3m_gcp_native_quarter_100k_ten_method_execution_plan_v3"
        or previous_recipes.get("schema")
        != "m3m_gcp_native_quarter_100k_recipe_manifest_v3"
    ):
        raise RuntimeError("v3 repository artifact schema mismatch")
    validate_activation_continuity(repo=repo, plan=previous_plan)

    remote_rows = _role_map(
        receipt.get("remote_artifacts"), REMOTE_ROLES, "remote continuity"
    )
    for role, row in remote_rows.items():
        path = Path(str(row.get("path", "")))
        if not path.is_absolute():
            raise RuntimeError(f"remote continuity path is not absolute: {role}")
        _validate_file(path, row, f"remote continuity artifact {role}")

    activation = json.loads(
        Path(str(remote_rows["activation_v3"]["path"])).read_text(encoding="utf-8")
    )
    if (
        activation.get("schema") != "m3m_gcp_lidar_formal_activation_v1"
        or activation.get("benchmark_commit") != PREVIOUS_COMMIT
        or activation.get("benchmark_tree") != PREVIOUS_TREE
        or activation.get("execution_plan_reviewed_commit") != PREVIOUS_COMMIT
        or activation.get("execution_plan_reviewed_tree") != PREVIOUS_TREE
        or activation.get("execution_plan_path") != PREVIOUS_PLAN
        or activation.get("execution_plan_sha256")
        != repository_rows["execution_plan_v3"].get("sha256")
        or activation.get("recipe_manifest_path") != PREVIOUS_RECIPES
        or activation.get("recipe_manifest_sha256")
        != repository_rows["recipe_manifest_v3"].get("sha256")
        or activation.get("canonical_sha256") != canonical_sha256(activation)
    ):
        raise RuntimeError("activation-v3 continuity identity mismatch")

    ready = receipt.get("inherited_ready_model", {})
    model_row = remote_rows["3dgs_reused_model"]
    if (
        ready.get("method_id") != "3dgs_original"
        or ready.get("formal_status") != "READY_FOR_EVALUATION"
        or ready.get("retrain_allowed") is not False
        or ready.get("model_sha256") != model_row.get("sha256")
        or ready.get("model_bytes") != model_row.get("bytes")
    ):
        raise RuntimeError("inherited 3DGS ready-model identity mismatch")

    outcomes = receipt.get("inherited_terminal_outcomes")
    if not isinstance(outcomes, list) or len(outcomes) != len(FAILED_OUTCOMES):
        raise RuntimeError("inherited terminal outcome inventory mismatch")
    by_method = {str(row.get("method_id")): row for row in outcomes}
    if set(by_method) != set(FAILED_OUTCOMES):
        raise RuntimeError("inherited terminal method inventory mismatch")
    for failed_method, expected_status in FAILED_OUTCOMES.items():
        outcome = by_method[failed_method]
        failure_row = remote_rows[f"{failed_method}_failure"]
        if (
            outcome.get("formal_status") != expected_status
            or outcome.get("formal_attempt_consumed") is not True
            or outcome.get("retry_allowed") is not False
            or outcome.get("failure_sha256") != failure_row.get("sha256")
        ):
            raise RuntimeError(f"inherited {failed_method} outcome policy mismatch")
        _validate_failure(failed_method, expected_status, failure_row)

    prior = receipt.get("metrogs_prior", {})
    prior_phase_path = Path(str(remote_rows["metrogs_prior_phase_success"]["path"]))
    prior_phase = json.loads(prior_phase_path.read_text(encoding="utf-8"))
    if (
        prior.get("status") != "PASS"
        or prior.get("rerun_allowed") is not False
        or prior.get("phase_success_sha256")
        != remote_rows["metrogs_prior_phase_success"].get("sha256")
        or prior_phase.get("schema") != "m3m_gcp_100k_phase_success_v2"
        or prior_phase.get("status") != "PASS"
        or prior_phase.get("scene") != SCENE
        or prior_phase.get("method_id") != "metrogs"
        or prior_phase.get("phase") != "prior"
        or prior_phase.get("canonical_sha256") != canonical_sha256(prior_phase)
        or prior_phase.get("environment_manifest_sha256")
        != remote_rows["metrogs_prior_environment"].get("sha256")
    ):
        raise RuntimeError("inherited MetroGS prior PASS mismatch")
    prior_products = {
        str(Path(str(row.get("path", ""))).resolve()): row
        for row in prior_phase.get("products", [])
        if isinstance(row, dict)
    }
    for role in (
        "metrogs_training_priors",
        "metrogs_prior_pass_marker",
        "metrogs_prior_merged_ply",
    ):
        row = remote_rows[role]
        product = prior_products.get(str(Path(str(row.get("path", ""))).resolve()))
        if (
            not isinstance(product, dict)
            or product.get("bytes") != row.get("bytes")
            or product.get("sha256") != row.get("sha256")
        ):
            raise RuntimeError(f"MetroGS prior product binding mismatch: {role}")

    rejection = receipt.get("metrogs_training_prechild_rejection", {})
    console_row = remote_rows["metrogs_training_prechild_guard_console"]
    console = Path(str(console_row["path"])).read_text(
        encoding="utf-8", errors="replace"
    )
    if (
        rejection.get("child_started") is not False
        or rejection.get("run_root_created") is not False
        or rejection.get("formal_attempt_consumed") is not False
        or rejection.get("retry_allowed_only_after_guard_fix_and_successor_review")
        is not True
        or rejection.get("console_sha256") != console_row.get("sha256")
        or "training requires the exact successful prior phase" not in console
    ):
        raise RuntimeError("MetroGS training pre-child rejection mismatch")
    transition = receipt.get("transition_policy", {})
    forbidden = sorted(FAILED_OUTCOMES)
    if (
        transition.get("activation_v3_immutable") is not True
        or transition.get("activation_v4_path") != plan.get("activation_manifest_path")
        or transition.get("continued_run_namespace")
        != "/root/autodl-tmp/runs/m3m-gcp-native-quarter/formal-100k-v2"
        or transition.get("inherited_failed_methods_forbidden_to_launch") != forbidden
        or transition.get("3dgs_retraining_forbidden") is not True
        or transition.get("metrogs_prior_rerun_forbidden") is not True
        or transition.get("metrogs_training_is_only_unfinished_attempt") is not True
        or transition.get("final_attempt_freeze_authorization")
        != "activation_v4_only"
        or transition.get("manual_guard_bypass_forbidden") is not True
    ):
        raise RuntimeError("activation-v4 transition policy mismatch")

    if mode == PRELAUNCH_FRESH:
        if postattempt_receipt is not None:
            raise RuntimeError("prelaunch continuity cannot consume a post-attempt receipt")
        validate_prelaunch_fresh_state(rejection)
        if method_id in FAILED_OUTCOMES:
            raise RuntimeError(
                f"activation-v4 forbids relaunch of terminal method: {method_id}"
            )
        if method_id == "3dgs_original" and phase != "packet":
            raise RuntimeError("activation-v4 permits reused 3DGS only for packet export")
        if method_id == "metrogs" and phase == "prior":
            raise RuntimeError("activation-v4 forbids rerunning the inherited MetroGS prior")
        return receipt

    if method_id is not None or phase is not None:
        raise RuntimeError("post-attempt closure cannot authorize a method phase")
    if postattempt_receipt is None:
        raise RuntimeError("post-attempt continuity requires the exact closure receipt")
    return validate_postattempt_closure(
        repo=repo,
        plan=plan,
        receipt_path=postattempt_receipt,
    )


def validate_continuity_for_plan(
    *, repo: Path, plan: dict[str, Any], method_id: str | None = None,
    phase: str | None = None, require_pgsr_absent: bool = False,
    mode: str, postattempt_receipt: Path | None = None,
) -> dict[str, Any]:
    """Dispatch continuity validation without weakening either frozen generation."""

    schema = plan.get("schema")
    if schema == "m3m_gcp_native_quarter_100k_ten_method_execution_plan_v4":
        return validate_activation_v4_continuity(
            repo=repo,
            plan=plan,
            method_id=method_id,
            phase=phase,
            mode=mode,
            postattempt_receipt=postattempt_receipt,
        )
    if schema == "m3m_gcp_native_quarter_100k_ten_method_execution_plan_v3":
        if mode != PRELAUNCH_FRESH or postattempt_receipt is not None:
            raise RuntimeError("superseded plan-v3 has no post-attempt closure mode")
        return validate_activation_continuity(
            repo=repo,
            plan=plan,
            method_id=method_id,
            require_pgsr_absent=require_pgsr_absent,
        )
    raise RuntimeError("unsupported 100K execution-plan continuity generation")
