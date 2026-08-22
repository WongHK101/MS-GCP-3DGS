#!/usr/bin/env python3
"""Validate the non-executable 100K MetroGS post-attempt closure receipt."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from m3m_gcp_lidar_artifacts import (
    PROTOCOL_ID,
    canonical_sha256,
    command_sha256,
    sha256_file,
    validate_failure_evidence_file,
)


SCENE = "gcp_100000_20260610"
METHOD_ID = "metrogs"
REVIEW_TASK_ID = "019ff12c-cb29-7cb2-8fb6-1d82c5f8c54b"
REVIEW_VERDICT = "APPROVE_POSTATTEMPT_LIFECYCLE_CLOSURE_NO_ACTIVATION_V5"
RECEIPT_SCHEMA = "m3m_gcp_100k_postattempt_closure_v1"
RECEIPT_STATUS = "SEALED_METROGS_FAILED_UNRANKED_NO_RETRY"
ACTIVATION_V4_COMMIT = "04f453dbf0d438addaa087b1402f7b1acdfc987d"
ACTIVATION_V4_TREE = "f84727f89620e8679049863d1bdbf6d8aaf2c491"
FORMAL_EXECUTION_REPO = Path("/root/autodl-tmp/code/GS-GCP-Benchmark")
FORMAL_RUN_ROOT = Path(
    "/root/autodl-tmp/runs/m3m-gcp-native-quarter/formal-100k-v2"
)
ACTIVATION_V4_PATH = FORMAL_RUN_ROOT / "activation_v4.json"
ACTIVATION_V5_PATH = FORMAL_RUN_ROOT / "activation_v5.json"
METRO_EVIDENCE_ROOT = (
    FORMAL_RUN_ROOT / SCENE / METHOD_ID / "evidence"
)
METRO_FAILURE_PATH = METRO_EVIDENCE_ROOT / "training" / "failure.json"
METRO_GUARD_CONSOLE_PATH = (
    METRO_EVIDENCE_ROOT / "training" / "guard-console-v4.log"
)
METRO_PRIOR_SUCCESS_PATH = METRO_EVIDENCE_ROOT / "prior" / "phase_success.json"
POSTATTEMPT_RECEIPT_RELATIVE = Path(
    "docs/protocol_evidence/m3m_gcp_100k_postattempt_closure_v1.json"
)
PLAN_RELATIVE = Path(
    "configs/m3m_gcp_native_quarter_100k_ten_method_execution_plan_v4.json"
)
RECIPE_MANIFEST_RELATIVE = Path(
    "configs/m3m_gcp_native_quarter_100k_recipe_manifest_v3.json"
)
METRO_RECIPE_RELATIVE = Path(
    "configs/m3m_gcp_native_quarter_100k_recipes_v2/metrogs.json"
)
OLD_CONTINUITY_RELATIVE = Path(
    "docs/protocol_evidence/m3m_gcp_100k_activation_v3_to_v4_continuity.json"
)
EXPECTED_PRIOR_PRODUCT_COUNT = 2205
EXPECTED_PRIOR_DEPTH_COUNT = 2196
REPOSITORY_ROLE_PATHS = {
    "execution_plan_v4": PLAN_RELATIVE,
    "recipe_manifest_v3": RECIPE_MANIFEST_RELATIVE,
    "metrogs_recipe_v2": METRO_RECIPE_RELATIVE,
    "activation_v3_to_v4_continuity": OLD_CONTINUITY_RELATIVE,
    "postattempt_closure_validator": Path(
        "code/gcp/m3m_gcp_100k_postattempt_closure.py"
    ),
    "postattempt_closure_builder": Path(
        "code/gcp/build_m3m_gcp_100k_postattempt_closure.py"
    ),
    "activation_v4_continuity_validator": Path(
        "code/gcp/m3m_gcp_100k_activation_v4_continuity.py"
    ),
    "attempt_manifest_builder": Path(
        "code/gcp/build_m3m_gcp_100k_attempt_manifest.py"
    ),
    "attempt_freezer": Path("code/gcp/freeze_m3m_gcp_lidar_scene_attempts.py"),
    "guarded_runner": Path("code/gcp/run_m3m_gcp_100k_guarded.py"),
    "activation_v4_builder": Path(
        "code/gcp/build_m3m_gcp_lidar_100k_activation_v4.py"
    ),
}
REMOTE_ROLE_PATHS = {
    "activation_v4": ACTIVATION_V4_PATH,
    "metrogs_training_failure": METRO_FAILURE_PATH,
    "metrogs_training_environment": METRO_EVIDENCE_ROOT
    / "training"
    / "environment.json",
    "metrogs_training_stdout": METRO_EVIDENCE_ROOT
    / "training"
    / "command.stdout.log",
    "metrogs_training_stderr": METRO_EVIDENCE_ROOT
    / "training"
    / "command.stderr.log",
    "metrogs_training_guard_console_v4": METRO_GUARD_CONSOLE_PATH,
    "metrogs_prior_phase_success": METRO_PRIOR_SUCCESS_PATH,
    "metrogs_prior_environment": METRO_EVIDENCE_ROOT
    / "prior"
    / "environment.json",
}


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON artifact is not an object: {path}")
    return payload


def _record(
    role: str, path: Path, *, recorded_path: str | None = None
) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"closure artifact is missing or symlinked: {role}: {path}")
    payload: dict[str, Any] | None = None
    if path.suffix == ".json":
        payload = _json(path)
    return {
        "role": role,
        "path": recorded_path if recorded_path is not None else str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "canonical_sha256": (
            payload.get("canonical_sha256") if payload is not None else None
        ),
    }


def _role_map(
    rows: object, expected_roles: set[str], label: str
) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list):
        raise RuntimeError(f"{label} inventory is not a list")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "role",
            "path",
            "bytes",
            "sha256",
            "canonical_sha256",
        }:
            raise RuntimeError(f"{label} inventory row is invalid")
        role = str(row.get("role", ""))
        if role in result:
            raise RuntimeError(f"{label} inventory contains duplicate role: {role}")
        result[role] = row
    if set(result) != expected_roles:
        raise RuntimeError(f"{label} inventory role mismatch")
    return result


def _validate_file_record(
    path: Path, row: dict[str, Any], label: str
) -> dict[str, Any] | None:
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != int(row.get("bytes", -1))
        or sha256_file(path) != row.get("sha256")
    ):
        raise RuntimeError(f"{label} changed or disappeared: {path}")
    expected_canonical = row.get("canonical_sha256")
    if path.suffix != ".json":
        if expected_canonical is not None:
            raise RuntimeError(f"non-JSON {label} carries a canonical SHA")
        return None
    payload = _json(path)
    if expected_canonical is None:
        raise RuntimeError(f"JSON {label} lacks a canonical SHA binding")
    if (
        payload.get("canonical_sha256") != expected_canonical
        or canonical_sha256(payload) != expected_canonical
    ):
        raise RuntimeError(f"{label} canonical SHA mismatch")
    return payload


def _git_bytes(repo: Path, relative: Path) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repo), "show", f"HEAD:{relative.as_posix()}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"post-attempt receipt is not tracked at HEAD: {relative}: "
            f"{result.stderr.decode('utf-8', errors='replace').strip()}"
        )
    return result.stdout


def _git_value(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def _validate_tracked_receipt(repo: Path, receipt_path: Path) -> None:
    expected = (repo / POSTATTEMPT_RECEIPT_RELATIVE).resolve()
    if receipt_path.resolve() != expected:
        raise RuntimeError("post-attempt closure receipt path mismatch")
    relative = receipt_path.resolve().relative_to(repo.resolve())
    tracked = _git_bytes(repo, relative)
    if tracked != receipt_path.read_bytes():
        raise RuntimeError("post-attempt closure receipt differs from tracked HEAD")
    status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain=v1", "--", relative.as_posix()],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if status.returncode != 0 or status.stdout:
        raise RuntimeError("post-attempt closure receipt checkout is not clean")


def _validate_prior_products(
    phase_success: dict[str, Any], *, recipe_sha256: str
) -> None:
    if (
        phase_success.get("schema") != "m3m_gcp_100k_phase_success_v2"
        or phase_success.get("status") != "PASS"
        or phase_success.get("scene") != SCENE
        or phase_success.get("method_id") != METHOD_ID
        or phase_success.get("phase") != "prior"
        or phase_success.get("recipe_sha256") != recipe_sha256
        or phase_success.get("canonical_sha256") != canonical_sha256(phase_success)
    ):
        raise RuntimeError("MetroGS prior phase-success identity mismatch")
    products = phase_success.get("products")
    if not isinstance(products, list) or len(products) != EXPECTED_PRIOR_PRODUCT_COUNT:
        raise RuntimeError("MetroGS prior product count mismatch")
    seen: set[Path] = set()
    depth_count = 0
    for row in products:
        if not isinstance(row, dict):
            raise RuntimeError("MetroGS prior product row is invalid")
        path = Path(str(row.get("path", "")))
        if not path.is_absolute() or path in seen:
            raise RuntimeError("MetroGS prior product path inventory mismatch")
        seen.add(path)
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != int(row.get("bytes", -1))
            or sha256_file(path) != row.get("sha256")
            or row.get("validation") != {"kind": "hash_bound_file_v1"}
        ):
            raise RuntimeError(f"MetroGS prior product changed: {path}")
        if path.suffix == ".npy":
            depth_count += 1
    if depth_count != EXPECTED_PRIOR_DEPTH_COUNT:
        raise RuntimeError("MetroGS prior depth product count mismatch")


def _expected_training_command(
    recipe: dict[str, Any], *, repo_path: Path
) -> list[str]:
    roots = recipe.get("phase_roots", {}).get("training", {})
    source = recipe.get("source_bindings", {}).get("training", {})
    replacements = {
        "repo": str(repo_path.resolve()),
        "source_root": str(Path(str(source.get("root", ""))).resolve()),
        "dataset_root": str(Path(str(roots.get("dataset_root", ""))).resolve()),
        "prior_root": str(Path(str(roots.get("prior_root", ""))).resolve()),
        "run_root": str(Path(str(recipe.get("authorized_run_root", ""))).resolve()),
        "packet_set_root": str(
            Path(str(recipe.get("authorized_packet_set_root", ""))).resolve()
        ),
    }
    template = recipe.get("phase_commands", {}).get("training")
    if not isinstance(template, list) or not template:
        raise RuntimeError("MetroGS frozen training command is missing")
    return [str(item).format(**replacements) for item in template]


def _validate_live_terminal(
    *,
    repo: Path,
    plan: dict[str, Any],
    repository_rows: dict[str, dict[str, Any]],
    remote_rows: dict[str, dict[str, Any]],
    terminal: dict[str, Any],
    prior: dict[str, Any],
) -> None:
    plan_path = (repo / PLAN_RELATIVE).resolve()
    if plan.get("schema") != "m3m_gcp_native_quarter_100k_ten_method_execution_plan_v4":
        raise RuntimeError("post-attempt closure requires execution plan v4")
    if (
        plan.get("canonical_sha256") != canonical_sha256(plan)
        or plan != _json(plan_path)
    ):
        raise RuntimeError("post-attempt execution plan canonical SHA mismatch")
    if plan_path.stat().st_size != repository_rows["execution_plan_v4"]["bytes"]:
        raise RuntimeError("post-attempt execution plan size mismatch")

    recipe_manifest = _validate_file_record(
        repo / RECIPE_MANIFEST_RELATIVE,
        repository_rows["recipe_manifest_v3"],
        "recipe manifest v3",
    )
    recipe = _validate_file_record(
        repo / METRO_RECIPE_RELATIVE,
        repository_rows["metrogs_recipe_v2"],
        "MetroGS recipe v2",
    )
    if not isinstance(recipe_manifest, dict) or not isinstance(recipe, dict):
        raise RuntimeError("post-attempt recipe bindings are not JSON")
    manifest_rows = {
        str(row.get("method_id")): row
        for row in recipe_manifest.get("recipes", [])
        if isinstance(row, dict)
    }
    metro_manifest = manifest_rows.get(METHOD_ID)
    if (
        not isinstance(metro_manifest, dict)
        or metro_manifest.get("path") != METRO_RECIPE_RELATIVE.as_posix()
        or metro_manifest.get("sha256") != repository_rows["metrogs_recipe_v2"]["sha256"]
        or recipe.get("method_id") != METHOD_ID
        or recipe.get("scene") != SCENE
        or recipe.get("canonical_sha256") != canonical_sha256(recipe)
    ):
        raise RuntimeError("MetroGS frozen recipe identity mismatch")

    activation = _validate_file_record(
        ACTIVATION_V4_PATH,
        remote_rows["activation_v4"],
        "activation v4",
    )
    if not isinstance(activation, dict) or (
        activation.get("schema") != "m3m_gcp_lidar_formal_activation_v1"
        or activation.get("execution_authorized") is not True
        or activation.get("benchmark_commit") != ACTIVATION_V4_COMMIT
        or activation.get("benchmark_tree") != ACTIVATION_V4_TREE
        or activation.get("execution_plan_reviewed_commit") != ACTIVATION_V4_COMMIT
        or activation.get("execution_plan_reviewed_tree") != ACTIVATION_V4_TREE
        or activation.get("execution_plan_path") != PLAN_RELATIVE.as_posix()
        or activation.get("execution_plan_sha256")
        != repository_rows["execution_plan_v4"]["sha256"]
        or activation.get("recipe_manifest_path")
        != RECIPE_MANIFEST_RELATIVE.as_posix()
        or activation.get("recipe_manifest_sha256")
        != repository_rows["recipe_manifest_v3"]["sha256"]
    ):
        raise RuntimeError("activation-v4 post-attempt identity mismatch")

    failure = _validate_file_record(
        METRO_FAILURE_PATH,
        remote_rows["metrogs_training_failure"],
        "MetroGS terminal failure",
    )
    if not isinstance(failure, dict):
        raise RuntimeError("MetroGS failure is not JSON")
    errors = validate_failure_evidence_file(
        METRO_FAILURE_PATH,
        expected_sha256=str(remote_rows["metrogs_training_failure"]["sha256"]),
        expected_scene=SCENE,
        expected_method_id=METHOD_ID,
        expected_status="FAILED_UNRANKED",
    )
    if errors:
        raise RuntimeError(f"MetroGS terminal failure invalid: {'; '.join(errors)}")
    authorized_run_root = Path(str(recipe.get("authorized_run_root", ""))).resolve()
    expected_command = _expected_training_command(recipe, repo_path=FORMAL_EXECUTION_REPO)
    if (
        _git_value(FORMAL_EXECUTION_REPO, "rev-parse", "HEAD")
        != ACTIVATION_V4_COMMIT
        or _git_value(FORMAL_EXECUTION_REPO, "rev-parse", "HEAD^{tree}")
        != ACTIVATION_V4_TREE
        or _git_value(FORMAL_EXECUTION_REPO, "status", "--porcelain=v1")
    ):
        raise RuntimeError("formal activation-v4 execution checkout changed")
    if (
        failure.get("failure_stage") != "training"
        or failure.get("run_root") != str(authorized_run_root)
        or failure.get("command_argv") != expected_command
        or failure.get("command_sha256") != command_sha256(expected_command)
        or failure.get("recipe_sha256") != repository_rows["metrogs_recipe_v2"]["sha256"]
        or failure.get("renderer_adapter_sha256")
        != recipe.get("renderer_adapter_sha256")
        or failure.get("exit_code") != 1
        or failure.get("last_valid_progress")
        != {"unit": "optimizer_steps", "value": 0.0}
        or float(failure.get("peak_gpu_memory_mib", -1)) != 0.0
        or failure.get("oom_signal") is not None
    ):
        raise RuntimeError("MetroGS terminal failure semantics mismatch")

    environment = _validate_file_record(
        Path(str(failure.get("environment_manifest_path", ""))),
        remote_rows["metrogs_training_environment"],
        "MetroGS training environment",
    )
    if not isinstance(environment, dict) or (
        failure.get("environment_manifest_sha256")
        != remote_rows["metrogs_training_environment"]["sha256"]
        or environment.get("schema") != "m3m_gcp_100k_execution_environment_v2"
        or environment.get("scene") != SCENE
        or environment.get("method_id") != METHOD_ID
        or environment.get("phase") != "training"
        or environment.get("argv") != expected_command
        or environment.get("started_at_utc") != failure.get("started_at_utc")
        or environment.get("canonical_sha256") != canonical_sha256(environment)
        or environment.get("resource_limits", {}).get("child_actual", {}).get("soft")
        != 65536
    ):
        raise RuntimeError("MetroGS child-start environment mismatch")

    stdout_path = Path(str(failure.get("stdout_path", "")))
    stderr_path = Path(str(failure.get("stderr_path", "")))
    _validate_file_record(
        stdout_path, remote_rows["metrogs_training_stdout"], "MetroGS training stdout"
    )
    _validate_file_record(
        stderr_path, remote_rows["metrogs_training_stderr"], "MetroGS training stderr"
    )
    _validate_file_record(
        METRO_GUARD_CONSOLE_PATH,
        remote_rows["metrogs_training_guard_console_v4"],
        "MetroGS guard console v4",
    )
    missing_path = Path(str(terminal.get("missing_required_path", "")))
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
    if (
        stdout_path.stat().st_size != 0
        or f"FileNotFoundError: {missing_path}" not in stderr_text
        or METRO_GUARD_CONSOLE_PATH.read_text(encoding="utf-8", errors="replace").strip()
        != json.dumps(
            {"status": "FAILED_UNRANKED", "failure_evidence": str(METRO_FAILURE_PATH)},
            ensure_ascii=False,
        )
    ):
        raise RuntimeError("MetroGS frozen prepared-input failure classification mismatch")

    formal_manifest = recipe.get("formal_input_manifest", {})
    source_manifest = Path(str(formal_manifest.get("path", "")))
    training_dataset = Path(
        str(recipe.get("phase_roots", {}).get("training", {}).get("dataset_root", ""))
    ).resolve()
    expected_missing = training_dataset / "NATIVE_QUARTER_INPUT_MANIFEST.json"
    if (
        missing_path != expected_missing
        or missing_path.exists()
        or not source_manifest.is_file()
        or source_manifest.is_symlink()
        or sha256_file(source_manifest) != formal_manifest.get("file_sha256")
    ):
        raise RuntimeError("MetroGS missing prepared-input manifest evidence mismatch")

    if (
        not authorized_run_root.is_dir()
        or authorized_run_root.is_symlink()
        or any(authorized_run_root.iterdir())
    ):
        raise RuntimeError("MetroGS terminal run root is not the exact empty attempt root")
    training_success = METRO_EVIDENCE_ROOT / "training" / "phase_success.json"
    absent_products = [
        training_success,
        authorized_run_root / "model",
        authorized_run_root / "metrogs_frozen_training_config.yaml",
    ]
    if any(path.exists() for path in absent_products):
        raise RuntimeError("MetroGS terminal failure unexpectedly has success/model products")

    prior_success = _validate_file_record(
        METRO_PRIOR_SUCCESS_PATH,
        remote_rows["metrogs_prior_phase_success"],
        "MetroGS prior phase success",
    )
    if not isinstance(prior_success, dict):
        raise RuntimeError("MetroGS prior phase-success is not JSON")
    prior_environment_path = Path(str(prior_success.get("environment_manifest_path", "")))
    _validate_file_record(
        prior_environment_path,
        remote_rows["metrogs_prior_environment"],
        "MetroGS prior environment",
    )
    if (
        prior_success.get("environment_manifest_sha256")
        != remote_rows["metrogs_prior_environment"]["sha256"]
    ):
        raise RuntimeError("MetroGS prior environment binding mismatch")
    _validate_prior_products(
        prior_success,
        recipe_sha256=repository_rows["metrogs_recipe_v2"]["sha256"],
    )

    if set(terminal) != {
        "method_id",
        "formal_status",
        "failure_stage",
        "failure_path",
        "failure_sha256",
        "authorized_run_root",
        "child_started",
        "formal_attempt_consumed",
        "retry_allowed",
        "classification",
        "missing_required_path",
        "algorithm_correctness_not_evaluated",
        "evaluation_eligible",
        "training_phase_success_absent",
        "final_model_absent",
    } or (
        terminal.get("method_id") != METHOD_ID
        or terminal.get("formal_status") != "FAILED_UNRANKED"
        or terminal.get("failure_stage") != "training"
        or terminal.get("failure_path") != str(METRO_FAILURE_PATH)
        or terminal.get("failure_sha256")
        != remote_rows["metrogs_training_failure"]["sha256"]
        or terminal.get("authorized_run_root") != str(authorized_run_root)
        or terminal.get("child_started") is not True
        or terminal.get("formal_attempt_consumed") is not True
        or terminal.get("retry_allowed") is not False
        or terminal.get("classification")
        != "FROZEN_PREPARED_INPUT_MANIFEST_MISSING_PRE_GPU"
        or terminal.get("algorithm_correctness_not_evaluated") is not True
        or terminal.get("evaluation_eligible") is not False
        or terminal.get("training_phase_success_absent") is not True
        or terminal.get("final_model_absent") is not True
    ):
        raise RuntimeError("MetroGS terminal closure policy mismatch")
    if set(prior) != {
        "status",
        "rerun_allowed",
        "phase_success_path",
        "phase_success_sha256",
        "product_count",
        "depth_product_count",
        "all_products_rehashed_required",
    } or (
        prior.get("status") != "PASS"
        or prior.get("rerun_allowed") is not False
        or prior.get("phase_success_path") != str(METRO_PRIOR_SUCCESS_PATH)
        or prior.get("phase_success_sha256")
        != remote_rows["metrogs_prior_phase_success"]["sha256"]
        or prior.get("product_count") != EXPECTED_PRIOR_PRODUCT_COUNT
        or prior.get("depth_product_count") != EXPECTED_PRIOR_DEPTH_COUNT
        or prior.get("all_products_rehashed_required") is not True
    ):
        raise RuntimeError("MetroGS inherited prior closure policy mismatch")


def _validate_payload(
    *, repo: Path, plan: dict[str, Any], payload: dict[str, Any]
) -> dict[str, Any]:
    if set(payload) != {
        "schema",
        "status",
        "scene",
        "review",
        "repository_artifacts",
        "remote_artifacts",
        "metrogs_terminal",
        "metrogs_prior",
        "closure_policy",
        "canonical_sha256",
    } or (
        payload.get("schema") != RECEIPT_SCHEMA
        or payload.get("status") != RECEIPT_STATUS
        or payload.get("scene") != SCENE
        or payload.get("canonical_sha256") != canonical_sha256(payload)
    ):
        raise RuntimeError("post-attempt closure receipt classification mismatch")
    review = payload.get("review", {})
    if set(review) != {
        "task_id",
        "verdict",
        "activation_v4_remains_only_training_authority",
    } or (
        review.get("task_id") != REVIEW_TASK_ID
        or review.get("verdict") != REVIEW_VERDICT
        or review.get("activation_v4_remains_only_training_authority") is not True
    ):
        raise RuntimeError("post-attempt closure review binding mismatch")
    policy = payload.get("closure_policy", {})
    if set(policy) != {
        "closure_grants_training_authority",
        "closure_grants_prior_authority",
        "closure_grants_packet_authority",
        "activation_v4_immutable",
        "activation_v5_forbidden",
        "activation_v5_path",
        "attempt_manifest_and_scene_freeze_only",
        "other_terminal_methods_unchanged",
        "3dgs_ready_identity_unchanged",
    } or (
        policy.get("closure_grants_training_authority") is not False
        or policy.get("closure_grants_prior_authority") is not False
        or policy.get("closure_grants_packet_authority") is not False
        or policy.get("activation_v4_immutable") is not True
        or policy.get("activation_v5_forbidden") is not True
        or policy.get("activation_v5_path") != str(ACTIVATION_V5_PATH)
        or policy.get("attempt_manifest_and_scene_freeze_only") is not True
        or policy.get("other_terminal_methods_unchanged") is not True
        or policy.get("3dgs_ready_identity_unchanged") is not True
        or ACTIVATION_V5_PATH.exists()
    ):
        raise RuntimeError("post-attempt closure execution boundary mismatch")

    repository_rows = _role_map(
        payload.get("repository_artifacts"),
        set(REPOSITORY_ROLE_PATHS),
        "post-attempt repository",
    )
    for role, relative in REPOSITORY_ROLE_PATHS.items():
        row = repository_rows[role]
        if row.get("path") != relative.as_posix():
            raise RuntimeError(f"post-attempt repository path mismatch: {role}")
        _validate_file_record(repo / relative, row, f"post-attempt repository {role}")
    remote_rows = _role_map(
        payload.get("remote_artifacts"),
        set(REMOTE_ROLE_PATHS),
        "post-attempt remote",
    )
    for role, path in REMOTE_ROLE_PATHS.items():
        row = remote_rows[role]
        if Path(str(row.get("path", ""))).resolve() != path.resolve():
            raise RuntimeError(f"post-attempt remote path mismatch: {role}")
        _validate_file_record(path, row, f"post-attempt remote {role}")
    _validate_live_terminal(
        repo=repo,
        plan=plan,
        repository_rows=repository_rows,
        remote_rows=remote_rows,
        terminal=payload.get("metrogs_terminal", {}),
        prior=payload.get("metrogs_prior", {}),
    )
    return payload


def build_postattempt_closure_payload(
    *, repo: Path, plan: dict[str, Any]
) -> dict[str, Any]:
    """Build and live-validate the one non-executable post-attempt receipt."""

    repo = repo.resolve()
    repository_artifacts = [
        _record(role, repo / relative, recorded_path=relative.as_posix())
        for role, relative in REPOSITORY_ROLE_PATHS.items()
    ]
    remote_artifacts = [
        _record(role, path)
        for role, path in REMOTE_ROLE_PATHS.items()
    ]
    remote_by_role = {row["role"]: row for row in remote_artifacts}
    recipe = _json(repo / METRO_RECIPE_RELATIVE)
    run_root = Path(str(recipe["authorized_run_root"])).resolve()
    dataset_root = Path(
        str(recipe["phase_roots"]["training"]["dataset_root"])
    ).resolve()
    payload: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "status": RECEIPT_STATUS,
        "scene": SCENE,
        "review": {
            "task_id": REVIEW_TASK_ID,
            "verdict": REVIEW_VERDICT,
            "activation_v4_remains_only_training_authority": True,
        },
        "repository_artifacts": repository_artifacts,
        "remote_artifacts": remote_artifacts,
        "metrogs_terminal": {
            "method_id": METHOD_ID,
            "formal_status": "FAILED_UNRANKED",
            "failure_stage": "training",
            "failure_path": str(METRO_FAILURE_PATH),
            "failure_sha256": remote_by_role["metrogs_training_failure"]["sha256"],
            "authorized_run_root": str(run_root),
            "child_started": True,
            "formal_attempt_consumed": True,
            "retry_allowed": False,
            "classification": "FROZEN_PREPARED_INPUT_MANIFEST_MISSING_PRE_GPU",
            "missing_required_path": str(
                dataset_root / "NATIVE_QUARTER_INPUT_MANIFEST.json"
            ),
            "algorithm_correctness_not_evaluated": True,
            "evaluation_eligible": False,
            "training_phase_success_absent": True,
            "final_model_absent": True,
        },
        "metrogs_prior": {
            "status": "PASS",
            "rerun_allowed": False,
            "phase_success_path": str(METRO_PRIOR_SUCCESS_PATH),
            "phase_success_sha256": remote_by_role["metrogs_prior_phase_success"][
                "sha256"
            ],
            "product_count": EXPECTED_PRIOR_PRODUCT_COUNT,
            "depth_product_count": EXPECTED_PRIOR_DEPTH_COUNT,
            "all_products_rehashed_required": True,
        },
        "closure_policy": {
            "closure_grants_training_authority": False,
            "closure_grants_prior_authority": False,
            "closure_grants_packet_authority": False,
            "activation_v4_immutable": True,
            "activation_v5_forbidden": True,
            "activation_v5_path": str(ACTIVATION_V5_PATH),
            "attempt_manifest_and_scene_freeze_only": True,
            "other_terminal_methods_unchanged": True,
            "3dgs_ready_identity_unchanged": True,
        },
    }
    payload["canonical_sha256"] = canonical_sha256(payload)
    return _validate_payload(repo=repo, plan=plan, payload=payload)


def validate_postattempt_closure(
    *, repo: Path, plan: dict[str, Any], receipt_path: Path
) -> dict[str, Any]:
    """Validate a tracked closure receipt and every bound local/remote artifact."""

    repo = repo.resolve()
    receipt_path = receipt_path.resolve()
    _validate_tracked_receipt(repo, receipt_path)
    payload = _json(receipt_path)
    return _validate_payload(repo=repo, plan=plan, payload=payload)
