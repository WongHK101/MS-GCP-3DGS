#!/usr/bin/env python3
"""Validate the immutable activation-v2 to activation-v3 continuity chain."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from m3m_gcp_lidar_artifacts import canonical_sha256, sha256_file


SCENE = "gcp_100000_20260610"
RECEIPT_SCHEMA = "m3m_gcp_100k_activation_continuity_v1"
RECEIPT_STATUS = "SEALED_V2_TO_V3_CONTINUITY"
PREVIOUS_COMMIT = "a64752b5f7375d79b0e9d82ca1f0e782ac6f0f86"
PREVIOUS_TREE = "9cbc07527c87614bf74cc3239360fe4a53519ef8"
REPOSITORY_ROLES = {
    "execution_plan_v2",
    "recipe_manifest_v2",
    "execution_note_v2",
}
REMOTE_ROLES = {
    "activation_v2",
    "2dgs_failure",
    "2dgs_classification_supplement",
    "2dgs_environment",
    "2dgs_stdout",
    "2dgs_stderr",
    "pgsr_prechild_guard_console",
}


def _bound_path(repo: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else repo / path


def _role_map(rows: object, expected_roles: set[str], label: str) -> dict[str, dict[str, Any]]:
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


def validate_activation_continuity(
    *,
    repo: Path,
    plan: dict[str, Any],
    method_id: str | None = None,
    require_pgsr_absent: bool = False,
) -> dict[str, Any]:
    """Rehash the complete continuity receipt and enforce inherited attempt state."""

    repo = repo.resolve()
    binding = plan.get("activation_continuity", {})
    receipt_row = binding.get("receipt", {})
    receipt_path = _bound_path(repo, receipt_row.get("path", "")).resolve()
    if (
        not receipt_path.is_file()
        or receipt_path.is_symlink()
        or sha256_file(receipt_path) != receipt_row.get("sha256")
    ):
        raise RuntimeError("v2-to-v3 continuity receipt identity mismatch")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("schema") != RECEIPT_SCHEMA
        or receipt.get("status") != RECEIPT_STATUS
        or receipt.get("status") != binding.get("status_required")
        or receipt.get("scene") != SCENE
        or receipt.get("canonical_sha256") != canonical_sha256(receipt)
        or binding.get("remote_artifacts_must_remain_byte_identical") is not True
        or binding.get("recipe_manifest_v2_bytes_unchanged") is not True
        or binding.get("execution_plan_v2_bytes_unchanged") is not True
    ):
        raise RuntimeError("v2-to-v3 continuity receipt classification mismatch")

    previous = receipt.get("previous_reviewed_checkout", {})
    if (
        previous.get("commit") != PREVIOUS_COMMIT
        or previous.get("tree") != PREVIOUS_TREE
        or previous.get("review_task_id")
        != "019ff12c-cb29-7cb2-8fb6-1d82c5f8c54b"
        or previous.get("review_verdict")
        != "PASS_100K_TIME_SPACE_EXECUTION_PLAN_V1"
    ):
        raise RuntimeError("previous reviewed activation identity mismatch")

    repository_rows = _role_map(
        receipt.get("repository_artifacts"), REPOSITORY_ROLES, "repository continuity"
    )
    for role, row in repository_rows.items():
        relative = Path(str(row.get("path", "")))
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise RuntimeError(f"unsafe repository continuity path: {relative}")
        path = (repo / relative).resolve()
        _validate_file(path, row, f"repository continuity artifact {role}")
        expected_canonical = row.get("canonical_sha256")
        if expected_canonical is not None:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (
                payload.get("canonical_sha256") != expected_canonical
                or canonical_sha256(payload) != expected_canonical
            ):
                raise RuntimeError(f"repository continuity canonical mismatch: {role}")

    remote_rows = _role_map(
        receipt.get("remote_artifacts"), REMOTE_ROLES, "remote continuity"
    )
    remote_paths: dict[str, Path] = {}
    for role, row in remote_rows.items():
        path = Path(str(row.get("path", "")))
        if not path.is_absolute():
            raise RuntimeError(f"remote continuity path is not absolute: {role}")
        _validate_file(path, row, f"remote continuity artifact {role}")
        remote_paths[role] = path

    old_plan = repository_rows["execution_plan_v2"]
    old_recipes = repository_rows["recipe_manifest_v2"]
    activation = json.loads(remote_paths["activation_v2"].read_text(encoding="utf-8"))
    if (
        activation.get("schema") != "m3m_gcp_lidar_formal_activation_v1"
        or activation.get("benchmark_commit") != PREVIOUS_COMMIT
        or activation.get("benchmark_tree") != PREVIOUS_TREE
        or activation.get("execution_plan_reviewed_commit") != PREVIOUS_COMMIT
        or activation.get("execution_plan_reviewed_tree") != PREVIOUS_TREE
        or activation.get("execution_plan_path") != old_plan.get("path")
        or activation.get("execution_plan_sha256") != old_plan.get("sha256")
        or activation.get("recipe_manifest_path") != old_recipes.get("path")
        or activation.get("recipe_manifest_sha256") != old_recipes.get("sha256")
        or activation.get("canonical_sha256") != canonical_sha256(activation)
    ):
        raise RuntimeError("activation-v2 continuity identity mismatch")

    outcomes = receipt.get("inherited_method_outcomes")
    if not isinstance(outcomes, list) or len(outcomes) != 1:
        raise RuntimeError("inherited method outcome inventory mismatch")
    outcome = outcomes[0]
    failure = json.loads(remote_paths["2dgs_failure"].read_text(encoding="utf-8"))
    supplement = json.loads(
        remote_paths["2dgs_classification_supplement"].read_text(encoding="utf-8")
    )
    if (
        outcome.get("method_id") != "2dgs"
        or outcome.get("formal_status") != "FAILED_UNRANKED"
        or outcome.get("formal_oom_signal") is not None
        or outcome.get("formal_attempt_consumed") is not True
        or outcome.get("retry_allowed") is not False
        or outcome.get("failure_sha256") != remote_rows["2dgs_failure"].get("sha256")
        or outcome.get("classification_supplement_sha256")
        != remote_rows["2dgs_classification_supplement"].get("sha256")
        or failure.get("schema") != "m3m_gcp_lidar_failure_evidence_v1"
        or failure.get("scene") != SCENE
        or failure.get("method_id") != "2dgs"
        or failure.get("status") != "FAILED_UNRANKED"
        or failure.get("oom_signal") is not None
        or failure.get("canonical_sha256") != canonical_sha256(failure)
        or failure.get("environment_manifest_sha256")
        != remote_rows["2dgs_environment"].get("sha256")
        or failure.get("stdout_sha256") != remote_rows["2dgs_stdout"].get("sha256")
        or failure.get("stderr_sha256") != remote_rows["2dgs_stderr"].get("sha256")
        or supplement.get("formal_status_unchanged") != "FAILED_UNRANKED"
        or supplement.get("formal_attempt_consumed") is not True
        or supplement.get("retry_allowed") is not False
        or supplement.get("bound_evidence", {}).get("failure_file_sha256")
        != remote_rows["2dgs_failure"].get("sha256")
    ):
        raise RuntimeError("inherited 2DGS final outcome mismatch")

    rejections = receipt.get("pre_child_guard_rejections")
    if not isinstance(rejections, list) or len(rejections) != 1:
        raise RuntimeError("pre-child rejection inventory mismatch")
    rejection = rejections[0]
    console = remote_paths["pgsr_prechild_guard_console"].read_text(
        encoding="utf-8", errors="replace"
    )
    if (
        rejection.get("method_id") != "pgsr"
        or rejection.get("child_started") is not False
        or rejection.get("run_root_created") is not False
        or rejection.get("evidence_root_created") is not False
        or rejection.get("formal_attempt_consumed") is not False
        or rejection.get("retry_allowed_only_after_exact_guard_fix_and_new_review")
        is not True
        or rejection.get("console_sha256")
        != remote_rows["pgsr_prechild_guard_console"].get("sha256")
        or "method source runtime status mismatch" not in console
    ):
        raise RuntimeError("PGSR pre-child guard rejection mismatch")

    transition = receipt.get("transition_policy", {})
    forbidden = transition.get("inherited_final_methods_forbidden_to_launch")
    if (
        transition.get("activation_v2_immutable") is not True
        or transition.get("activation_v3_path")
        != plan.get("activation_manifest_path")
        or transition.get("continued_run_namespace")
        != "/root/autodl-tmp/runs/m3m-gcp-native-quarter/formal-100k-v2"
        or transition.get("recipe_manifest_v2_bytes_unchanged") is not True
        or transition.get("execution_plan_v2_bytes_unchanged") is not True
        or transition.get("final_attempt_freeze_authorization")
        != "activation_v3_only"
        or transition.get("remote_artifacts_must_remain_byte_identical") is not True
        or transition.get("manual_guard_bypass_forbidden") is not True
        or forbidden != ["2dgs"]
        or binding.get("inherited_final_methods_forbidden_to_launch") != ["2dgs"]
    ):
        raise RuntimeError("activation continuity transition policy mismatch")
    if method_id in forbidden:
        raise RuntimeError(f"activation-v3 forbids relaunch of inherited final method: {method_id}")

    if require_pgsr_absent:
        for field in ("run_root", "evidence_root"):
            path = Path(str(rejection.get(field, "")))
            if not path.is_absolute() or path.exists():
                raise RuntimeError(
                    f"PGSR pre-child continuity requires absent {field}: {path}"
                )
    return receipt
