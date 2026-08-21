#!/usr/bin/env python3
"""Execute one frozen 100K phase behind review, disk and packet-lifecycle gates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from m3m_gcp_lidar_artifacts import (
    canonical_sha256,
    command_sha256,
    sha256_file,
    validate_scene_attempt_freeze,
)
from m3m_gcp_100k_phase_products import (
    phase_product_row,
    revalidate_phase_product_row,
    validate_gaussian_ply,
    validate_npz,
    validate_torch_checkpoint,
)
from m3m_gcp_100k_continuity import validate_activation_continuity
from m3m_gcp_100k_source_binding_correction import (
    validate_source_binding_correction,
)
from verify_m3m_gcp_lidar_formal_v1 import validate_archive_manifest

try:
    import resource
except ImportError:  # pragma: no cover - Windows authoring host
    resource = None


REQUIRED_VERDICT = "PASS_100K_TIME_SPACE_EXECUTION_PLAN_V1"
REQUIRED_PROTOCOL_VERDICT = "PASS_LIDAR_V1_AND_SIX_SCENE_PREPARATION_V2"
PROTOCOL_ID = "m3m_gcp_lidar_rendered_surface_v1"
SCENE = "gcp_100000_20260610"
METHOD_ORDER = [
    "3dgs_original", "2dgs", "pgsr", "rade_gs", "qgs", "gsprior", "sof",
    "citygaussian_v2", "citygs_x", "metrogs",
]
LOCKED_SCENES = [
    "gcp_5000_20260602", "gcp_20000_20260602", "gcp_10000_20260610",
    "gcp_50000_20260610",
]
GIB = 1024**3
MIN_FREE_GIB = {"prior": 300, "training": 300, "packet": 180}
PACKET_CAP_BYTES = 100 * GIB
REQUIRED_NOFILE_SOFT = 65536
ACTIVATION_FIELDS = {
    "schema", "protocol_id", "protocol_review_task_id",
    "protocol_review_verdict", "protocol_reviewed_commit",
    "protocol_reviewed_tree", "execution_plan_review_task_id",
    "execution_plan_review_verdict", "execution_plan_reviewed_commit",
    "execution_plan_reviewed_tree", "execution_authorized",
    "contract_file_sha256", "artifact_schema_sha256",
    "common_preparation_local_path", "common_preparation_local_sha256",
    "common_preparation_remote_path", "common_preparation_remote_sha256",
    "execution_plan_path", "execution_plan_sha256", "recipe_manifest_path",
    "recipe_manifest_sha256", "benchmark_commit", "benchmark_tree",
    "canonical_sha256",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _limit_to_json(value: int) -> int | str:
    if resource is not None and value == resource.RLIM_INFINITY:
        return "unlimited"
    return int(value)


def _limit_meets_minimum(value: int, minimum: int) -> bool:
    return bool(
        resource is not None
        and (value == resource.RLIM_INFINITY or int(value) >= minimum)
    )


def configure_nofile_limit(required_soft: int = REQUIRED_NOFILE_SOFT) -> dict[str, Any]:
    """Set the exact parent soft limit before any formal child or artifact exists."""
    if resource is None:
        raise RuntimeError("formal child launch requires POSIX RLIMIT_NOFILE support")
    before_soft, before_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if not _limit_meets_minimum(before_hard, required_soft):
        raise RuntimeError(
            "RLIMIT_NOFILE hard limit is below the required pre-child minimum: "
            f"hard={_limit_to_json(before_hard)} required={required_soft}"
        )
    resource.setrlimit(resource.RLIMIT_NOFILE, (required_soft, before_hard))
    after_soft, after_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if int(after_soft) != required_soft or after_hard != before_hard:
        raise RuntimeError(
            "failed to establish the exact RLIMIT_NOFILE child-launch contract"
        )
    return {
        "resource": "RLIMIT_NOFILE",
        "required_soft": required_soft,
        "hard_minimum": required_soft,
        "parent_before": {
            "soft": _limit_to_json(before_soft),
            "hard": _limit_to_json(before_hard),
        },
        "parent_after": {
            "soft": _limit_to_json(after_soft),
            "hard": _limit_to_json(after_hard),
        },
    }


def read_child_nofile_limit(pid: int) -> dict[str, int | str]:
    """Read the actual inherited limit from Linux procfs after child creation."""
    limits_path = Path(f"/proc/{pid}/limits")
    text = limits_path.read_text(encoding="utf-8", errors="strict")
    for line in text.splitlines():
        if line.startswith("Max open files"):
            parts = line.split()
            if len(parts) < 5:
                break

            def parse(raw: str) -> int | str:
                return "unlimited" if raw == "unlimited" else int(raw)

            return {"soft": parse(parts[3]), "hard": parse(parts[4])}
    raise RuntimeError(f"Max open files row missing from {limits_path}")


def observe_child_nofile_limit(
    process: subprocess.Popen[Any], *, timeout_seconds: float = 2.0
) -> dict[str, int | str]:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            return read_child_nofile_limit(process.pid)
        except (FileNotFoundError, ProcessLookupError, RuntimeError, ValueError) as exc:
            last_error = exc
            if process.poll() is not None:
                break
            time.sleep(0.01)
    raise RuntimeError(
        f"could not observe child RLIMIT_NOFILE inheritance for pid {process.pid}: "
        f"{last_error}"
    )


def write_environment_manifest(path: Path, payload: dict[str, Any]) -> None:
    payload.pop("canonical_sha256", None)
    payload["canonical_sha256"] = canonical_sha256(payload)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def git_value(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, encoding="utf-8"
    ).strip()


def git_porcelain_status(repo: Path) -> str:
    """Return porcelain-v1 bytes without destroying its significant XY columns."""

    payload = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain=v1", "--untracked-files=no"]
    )
    return payload.rstrip(b"\r\n").decode("utf-8")


def git_blob_sha256(repo: Path, commit: str, relative_path: str) -> str:
    payload = subprocess.check_output(
        ["git", "-C", str(repo), "show", f"{commit}:{relative_path}"]
    )
    return hashlib.sha256(payload).hexdigest()


def directory_bytes(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def cgroup_memory_events() -> dict[str, int]:
    path = Path("/sys/fs/cgroup/memory.events")
    values = {"oom": 0, "oom_kill": 0, "max": 0}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] in values:
            values[parts[0]] = int(parts[1])
    return values


def memory_event_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {key: max(0, int(after.get(key, 0)) - int(before.get(key, 0))) for key in ("oom", "oom_kill", "max")}


def gpu_memory_for_pid(pid: int) -> float:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return 0.0
    total = 0.0
    for line in output.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) == pid:
            total += float(parts[1])
    return total


def require_idle_gpu() -> dict[str, Any]:
    try:
        devices = subprocess.check_output(
            ["nvidia-smi", "-L"], text=True, encoding="utf-8", stderr=subprocess.STDOUT, timeout=10
        ).strip()
        compute = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            encoding="utf-8",
            stderr=subprocess.STDOUT,
            timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"formal 100K phase requires an observable NVIDIA GPU: {exc}") from exc
    if not devices:
        raise RuntimeError("formal 100K phase requires an NVIDIA GPU")
    if compute:
        raise RuntimeError(f"foreign or stale GPU compute process present before launch: {compute}")
    return {"devices": devices.splitlines(), "compute_processes_before_launch": []}


def validate_capacity(capacity_root: Path, phase: str, *, free_bytes: int | None = None) -> None:
    required = MIN_FREE_GIB[phase] * GIB
    actual = shutil.disk_usage(capacity_root).free if free_bytes is None else free_bytes
    if actual < required:
        raise RuntimeError(
            f"{phase} requires at least {MIN_FREE_GIB[phase]} GiB free; found {actual / GIB:.3f} GiB"
        )


def bound_path(repo: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo / path


def validate_superseded_activation_receipt(
    *, repo: Path, plan: dict[str, Any]
) -> dict[str, Any]:
    binding = plan.get("superseded_activation", {})
    receipt_row = binding.get("receipt", {})
    receipt_path = bound_path(repo, str(receipt_row.get("path", ""))).resolve()
    if (
        not receipt_path.is_file()
        or receipt_path.is_symlink()
        or sha256_file(receipt_path) != receipt_row.get("sha256")
    ):
        raise RuntimeError("v1 supersession receipt identity mismatch")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    classification = receipt.get("classification", {})
    if (
        receipt.get("schema") != "m3m_gcp_100k_activation_supersession_v1"
        or receipt.get("status") != binding.get("status_required")
        or receipt.get("canonical_sha256") != canonical_sha256(receipt)
        or classification.get("algorithm_failure") is not False
        or classification.get("formal_retry_counted") is not False
        or classification.get("rankable") is not False
        or binding.get("algorithm_failure") is not False
        or binding.get("formal_retry_counted") is not False
        or binding.get("rankable") is not False
        or binding.get("remote_artifacts_must_remain_byte_identical") is not True
    ):
        raise RuntimeError("v1 supersession receipt classification mismatch")
    artifacts = receipt.get("remote_artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 6:
        raise RuntimeError("v1 supersession remote artifact inventory mismatch")
    for row in artifacts:
        path = Path(str(row.get("path", "")))
        if (
            not path.is_absolute()
            or not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != int(row.get("bytes", -1))
            or sha256_file(path) != row.get("sha256")
        ):
            raise RuntimeError(f"superseded v1 evidence changed or disappeared: {path}")
    return receipt


def validate_activation_and_recipe(
    *,
    repo: Path,
    activation_path: Path,
    plan_path: Path,
    recipe_manifest_path: Path,
    recipe_path: Path,
    method_id: str,
    phase: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    activation = json.loads(activation_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    recipe_manifest = json.loads(recipe_manifest_path.read_text(encoding="utf-8"))
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    if set(activation) != ACTIVATION_FIELDS:
        raise RuntimeError("activation field inventory mismatch")
    if activation.get("schema") != "m3m_gcp_lidar_formal_activation_v1":
        raise RuntimeError("activation schema mismatch")
    if activation.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("activation protocol mismatch")
    if (
        activation.get("protocol_review_verdict") != REQUIRED_PROTOCOL_VERDICT
        or activation.get("execution_plan_review_verdict") != REQUIRED_VERDICT
        or activation.get("execution_authorized") is not True
    ):
        raise RuntimeError("both exact protocol/data and 100K review verdicts are required")
    if activation.get("canonical_sha256") != canonical_sha256(activation):
        raise RuntimeError("activation canonical SHA mismatch")
    if sha256_file(plan_path) != activation.get("execution_plan_sha256"):
        raise RuntimeError("activation execution-plan SHA mismatch")
    if sha256_file(recipe_manifest_path) != activation.get("recipe_manifest_sha256"):
        raise RuntimeError("activation recipe-manifest SHA mismatch")
    if bound_path(repo, str(activation.get("execution_plan_path", ""))).resolve() != plan_path.resolve():
        raise RuntimeError("activation execution-plan path mismatch")
    if bound_path(repo, str(activation.get("recipe_manifest_path", ""))).resolve() != recipe_manifest_path.resolve():
        raise RuntimeError("activation recipe-manifest path mismatch")
    preparation_paths: list[Path] = []
    for path_field, sha_field in (
        ("common_preparation_local_path", "common_preparation_local_sha256"),
        ("common_preparation_remote_path", "common_preparation_remote_sha256"),
    ):
        path = bound_path(repo, str(activation.get(path_field, ""))).resolve()
        if not path.is_file() or sha256_file(path) != activation.get(sha_field):
            raise RuntimeError(f"activation common-preparation identity mismatch: {path_field}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("status") != "PASS_COMMON_SCENE_PREPARATION_NO_TRAINING"
            or payload.get("scene_count") != 6
            or payload.get("training_started") is not False
            or payload.get("formal_evaluation") != "NOT_STARTED"
            or payload.get("contract_file_sha256")
            != activation.get("contract_file_sha256")
        ):
            raise RuntimeError(f"activation common-preparation did not pass: {path_field}")
        preparation_paths.append(path)
    if sha256_file(preparation_paths[0]) != sha256_file(preparation_paths[1]):
        raise RuntimeError("local/remote common-preparation evidence bytes differ")
    head = git_value(repo, "rev-parse", "HEAD")
    tree = git_value(repo, "show", "-s", "--format=%T", "HEAD")
    if head != activation.get("benchmark_commit") or head != activation.get("execution_plan_reviewed_commit"):
        raise RuntimeError("checkout is not the exact reviewed commit")
    if tree != activation.get("benchmark_tree") or tree != activation.get("execution_plan_reviewed_tree"):
        raise RuntimeError("checkout is not the exact reviewed tree")
    if git_value(repo, "status", "--porcelain"):
        raise RuntimeError("reviewed checkout is dirty")
    if plan.get("schema") != "m3m_gcp_native_quarter_100k_ten_method_execution_plan_v3":
        raise RuntimeError("execution plan schema mismatch")
    expected_activation_path = Path(
        str(plan.get("activation_manifest_path", ""))
    ).resolve()
    if activation_path.resolve() != expected_activation_path:
        raise RuntimeError("activation path differs from the frozen v3 continuation")
    if plan.get("scene") != SCENE or plan.get("seed") != 0:
        raise RuntimeError("execution plan is not frozen 100K seed0")
    if plan.get("status") != "REVIEW_CANDIDATE_NOT_EXECUTION_AUTHORIZED" or plan.get("execution_authorized") is not False:
        raise RuntimeError("reviewed execution-plan candidate identity changed")
    if plan.get("canonical_sha256") != canonical_sha256(plan):
        raise RuntimeError("execution plan canonical SHA mismatch")
    validate_superseded_activation_receipt(repo=repo, plan=plan)
    validate_activation_continuity(
        repo=repo,
        plan=plan,
        method_id=method_id,
        require_pgsr_absent=(method_id == "pgsr" and phase == "training"),
    )
    if plan.get("method_order") != METHOD_ORDER:
        raise RuntimeError("execution plan method order mismatch")
    if activation.get("execution_plan_review_task_id") != plan.get("review", {}).get("task_id"):
        raise RuntimeError("activation review task differs from execution plan")
    formal_protocol = plan.get("formal_lidar_protocol", {})
    for label, activation_field in (
        ("contract", "contract_file_sha256"),
        ("artifact_schema", "artifact_schema_sha256"),
    ):
        row = formal_protocol.get(label, {})
        path = bound_path(repo, str(row.get("path", "")))
        if not path.is_file() or sha256_file(path) != row.get("sha256"):
            raise RuntimeError(f"execution plan formal {label} identity mismatch")
        if activation.get(activation_field) != row.get("sha256"):
            raise RuntimeError(f"activation formal {label} SHA mismatch")
    contract_path = bound_path(
        repo, str(formal_protocol.get("contract", {}).get("path", ""))
    ).resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if (
        activation.get("protocol_review_task_id")
        != contract.get("review", {}).get("protocol_review_task_id")
    ):
        raise RuntimeError("activation protocol review task differs from contract")
    protocol_commit = str(activation.get("protocol_reviewed_commit", ""))
    if (
        git_value(repo, "show", "-s", "--format=%T", protocol_commit)
        != activation.get("protocol_reviewed_tree")
    ):
        raise RuntimeError("activation protocol reviewed commit/tree mismatch")
    protocol_files = (
        (contract_path, "contract_file_sha256"),
        (
            bound_path(repo, str(formal_protocol.get("artifact_schema", {}).get("path", ""))).resolve(),
            "artifact_schema_sha256",
        ),
        (preparation_paths[0], "common_preparation_local_sha256"),
        (preparation_paths[1], "common_preparation_remote_sha256"),
    )
    for path, activation_field in protocol_files:
        try:
            relative = path.relative_to(repo.resolve()).as_posix()
        except ValueError as exc:
            raise RuntimeError("protocol-reviewed artifact is outside benchmark repository") from exc
        if git_blob_sha256(repo, protocol_commit, relative) != activation.get(activation_field):
            raise RuntimeError(f"protocol-reviewed commit blob mismatch: {relative}")
    if formal_protocol.get("execution_authorized") is not False:
        raise RuntimeError("execution-plan candidate protocol state mismatch")
    if plan.get("other_prepared_scenes_locked") != LOCKED_SCENES or plan.get(
        "other_prepared_scene_training_rendering_or_formal_evaluation_authorized"
    ) is not False:
        raise RuntimeError("execution plan does not lock the other four scenes")
    storage = plan.get("storage", {})
    if (
        storage.get("minimum_free_before_prior_gib") != 300
        or storage.get("minimum_free_before_training_gib") != 300
        or storage.get("minimum_free_before_packet_export_gib") != 180
        or storage.get("packet_scratch_hard_cap_gib") != 100
    ):
        raise RuntimeError("execution plan capacity gates mismatch")
    closure = plan.get("execution_closure", {})
    for label, expected_path in (
        (
            "activation_builder",
            repo / "code/gcp/build_m3m_gcp_lidar_100k_activation.py",
        ),
        (
            "attempt_manifest_builder",
            repo / "code/gcp/build_m3m_gcp_100k_attempt_manifest.py",
        ),
        (
            "activation_continuity_validator",
            repo / "code/gcp/m3m_gcp_100k_continuity.py",
        ),
        (
            "source_binding_correction_validator",
            repo / "code/gcp/m3m_gcp_100k_source_binding_correction.py",
        ),
        (
            "recipe_manifest_v3_builder",
            repo / "code/gcp/build_m3m_gcp_100k_recipe_manifest_v3.py",
        ),
        ("guarded_runner", repo / "code/gcp/run_m3m_gcp_100k_guarded.py"),
        (
            "attempt_freezer",
            repo / "code/gcp/freeze_m3m_gcp_lidar_scene_attempts.py",
        ),
    ):
        row = closure.get(label, {})
        path = bound_path(repo, str(row.get("path", ""))).resolve()
        if path != expected_path or not path.is_file() or sha256_file(path) != row.get("sha256"):
            raise RuntimeError(f"execution plan {label} identity mismatch")
    if closure.get("exact_review_verdict_required") != REQUIRED_VERDICT:
        raise RuntimeError("execution plan exact review verdict mismatch")
    if (
        closure.get("prior_and_training_require_absent_run_root_at_guard_admission")
        is not True
        or closure.get("training_child_must_create_products_inside_new_run_root")
        is not True
        or closure.get("ready_model_identity_requires_exact_phase_success_markers")
        is not True
        or closure.get("phase_success_command_rehashed_against_frozen_recipe")
        is not True
        or closure.get("rlimit_nofile_soft_required_for_child_phases")
        != REQUIRED_NOFILE_SOFT
        or closure.get("rlimit_nofile_hard_minimum_prechild_gate")
        != REQUIRED_NOFILE_SOFT
        or closure.get("rlimit_nofile_parent_before_after_evidence_required")
        is not True
        or closure.get("rlimit_nofile_child_actual_inheritance_evidence_required")
        is not True
    ):
        raise RuntimeError("execution plan fresh-run-root closure mismatch")
    preparation = plan.get("preparation", {}).get("per_method_input_evidence", {})
    preparation_path = Path(str(preparation.get("path", "")))
    if not preparation_path.is_absolute() or not preparation_path.is_file():
        raise RuntimeError("100K prepared per-method input evidence missing")
    if sha256_file(preparation_path) != preparation.get("sha256"):
        raise RuntimeError("100K prepared per-method input evidence SHA mismatch")
    preparation_payload = json.loads(preparation_path.read_text(encoding="utf-8"))
    if preparation_payload.get("status") != preparation.get("status_required"):
        raise RuntimeError("100K prepared per-method input evidence status mismatch")
    cleanup = plan.get("preparation", {}).get("obsolete_train_first_attempt_cleanup", {})
    cleanup_path = Path(str(cleanup.get("path", "")))
    if not cleanup_path.is_absolute() or not cleanup_path.is_file():
        raise RuntimeError("obsolete 100K attempt cleanup receipt missing")
    if sha256_file(cleanup_path) != cleanup.get("sha256"):
        raise RuntimeError("obsolete 100K attempt cleanup receipt SHA mismatch")
    cleanup_payload = json.loads(cleanup_path.read_text(encoding="utf-8"))
    if (
        cleanup_payload.get("status") != cleanup.get("status_required")
        or cleanup_payload.get("deleted") is not True
        or Path(str(cleanup_payload.get("deleted_path", ""))).exists()
    ):
        raise RuntimeError("obsolete 100K attempt was not safely removed")
    validate_source_binding_correction(
        repo=repo,
        plan=plan,
        require_live_sources=True,
    )
    if recipe_manifest.get("schema") != "m3m_gcp_native_quarter_100k_recipe_manifest_v3":
        raise RuntimeError("recipe manifest schema mismatch")
    if recipe_manifest.get("canonical_sha256") != canonical_sha256(recipe_manifest):
        raise RuntimeError("recipe manifest canonical SHA mismatch")
    if recipe_manifest.get("status") != "REVIEW_CANDIDATE_NOT_EXECUTION_AUTHORIZED":
        raise RuntimeError("recipe manifest candidate identity changed")
    if recipe_manifest.get("scene") != SCENE or recipe_manifest.get("seed") != 0:
        raise RuntimeError("recipe manifest identity mismatch")
    if recipe_manifest.get("method_order") != METHOD_ORDER:
        raise RuntimeError("recipe manifest method order mismatch")
    manifest_rows = recipe_manifest.get("recipes", [])
    if [row.get("method_id") for row in manifest_rows] != METHOD_ORDER:
        raise RuntimeError("recipe manifest row order or cardinality mismatch")
    plan_recipe = plan.get("recipe_manifest", {})
    if (
        bound_path(repo, str(plan_recipe.get("path", ""))).resolve()
        != recipe_manifest_path.resolve()
        or plan_recipe.get("file_sha256") != sha256_file(recipe_manifest_path)
        or plan_recipe.get("canonical_sha256") != recipe_manifest.get("canonical_sha256")
    ):
        raise RuntimeError("execution plan recipe-manifest binding mismatch")
    rows = [row for row in manifest_rows if row.get("method_id") == method_id]
    if len(rows) != 1:
        raise RuntimeError("method does not have exactly one frozen recipe")
    row = rows[0]
    if bound_path(repo, str(row.get("path", ""))).resolve() != recipe_path.resolve():
        raise RuntimeError("recipe path differs from recipe manifest")
    if sha256_file(recipe_path) != row.get("sha256"):
        raise RuntimeError("recipe SHA differs from recipe manifest")
    if recipe.get("canonical_sha256") != canonical_sha256(recipe):
        raise RuntimeError("recipe canonical SHA mismatch")
    expected_recipe_schema = (
        "m3m_gcp_native_quarter_100k_execution_recipe_v3"
        if method_id == "3dgs_original"
        else "m3m_gcp_native_quarter_100k_execution_recipe_v2"
    )
    if recipe.get("schema") != expected_recipe_schema:
        raise RuntimeError("recipe schema mismatch")
    if recipe.get("method_id") != method_id or recipe.get("scene") != SCENE or recipe.get("seed") != 0:
        raise RuntimeError("recipe identity mismatch")
    if recipe.get("status") != "REVIEW_CANDIDATE_NOT_EXECUTION_AUTHORIZED":
        raise RuntimeError("reviewed recipe candidate identity changed")
    if recipe.get("fresh_run_root_policy") != {
        "prior_and_training_require_absent_run_root_at_guard_admission": True,
        "prior_must_not_create_run_root": True,
        "training_guard_exclusively_creates_empty_run_root_before_child": True,
        "training_child_must_create_final_products": True,
    }:
        raise RuntimeError("recipe fresh-run-root policy mismatch")
    if recipe.get("process_resource_limits") != {
        "applies_to_phases": ["prior", "training", "packet"],
        "rlimit_nofile_hard_minimum": REQUIRED_NOFILE_SOFT,
        "rlimit_nofile_soft": REQUIRED_NOFILE_SOFT,
        "record_parent_before_after": True,
        "record_child_actual_inheritance": True,
    }:
        raise RuntimeError("recipe process resource-limit contract mismatch")
    for relative, expected_sha in recipe.get("benchmark_required_files_sha256", {}).items():
        path = bound_path(repo, str(relative))
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise RuntimeError(f"benchmark recipe dependency identity mismatch: {relative}")
    return plan, recipe


def create_exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o444)
    try:
        os.write(descriptor, (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8"))
    finally:
        os.close(descriptor)


def acquire_packet_state(state_path: Path, method_id: str, packet_set_root: Path) -> None:
    payload = {
        "schema": "m3m_gcp_100k_single_packet_state_v1",
        "scene": SCENE,
        "method_id": method_id,
        "packet_set_root": str(packet_set_root),
        "created_at_utc": utc_now(),
    }
    create_exclusive_json(state_path, payload)


def materialize_phase_files(
    recipe: dict[str, Any], *, phase: str, run_root: Path, replacements: dict[str, str]
) -> None:
    for row in recipe.get("materializations", {}).get(phase, []):
        relative = Path(str(row.get("relative_path", "")))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise RuntimeError("unsafe recipe materialization path")
        target = run_root / relative
        if target.exists():
            raise RuntimeError(f"refusing to overwrite recipe materialization: {target}")
        content = str(row.get("content", "")).format(**replacements)
        expected = row.get("rendered_sha256")
        actual = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if expected is not None and actual != expected:
            raise RuntimeError("recipe materialization rendered SHA mismatch")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def validate_source_binding(recipe: dict[str, Any], source_root: Path, phase: str) -> None:
    binding = recipe.get("source_bindings", {}).get(phase)
    if not isinstance(binding, dict):
        binding = recipe.get("source_binding", {})
    expected_root = binding.get("root")
    if expected_root and source_root.resolve() != Path(str(expected_root)).resolve():
        raise RuntimeError("method source root mismatch")
    if git_value(source_root, "rev-parse", "HEAD") != binding.get("commit"):
        raise RuntimeError("method source commit mismatch")
    if git_value(source_root, "rev-parse", "HEAD^{tree}") != binding.get("tree"):
        raise RuntimeError("method source tree mismatch")
    if git_porcelain_status(source_root) != binding.get("required_status", ""):
        raise RuntimeError("method source runtime status mismatch")
    for relative, expected_sha in binding.get("required_files_sha256", {}).items():
        path = source_root / relative
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise RuntimeError(f"method source file identity mismatch: {relative}")


def validate_phase_roots(recipe: dict[str, Any], *, phase: str, dataset_root: Path, prior_root: Path) -> None:
    roots = recipe.get("phase_roots", {}).get(phase, {})
    expected_dataset = roots.get("dataset_root")
    expected_prior = roots.get("prior_root")
    if expected_dataset and dataset_root.resolve() != Path(str(expected_dataset)).resolve():
        raise RuntimeError("phase dataset root mismatch")
    if expected_prior and prior_root.resolve() != Path(str(expected_prior)).resolve():
        raise RuntimeError("phase prior root mismatch")


def validate_external_files(recipe: dict[str, Any], phase: str) -> None:
    for raw_path, expected_sha in recipe.get(
        "phase_external_required_files_sha256", {}
    ).get(phase, {}).items():
        path = Path(str(raw_path))
        if not path.is_absolute() or not path.is_file():
            raise RuntimeError(f"phase external dependency missing: {raw_path}")
        if sha256_file(path) != expected_sha:
            raise RuntimeError(f"phase external dependency SHA mismatch: {raw_path}")


def load_frozen_train_rows(recipe: dict[str, Any]) -> list[dict[str, Any]]:
    binding = recipe.get("formal_input_manifest", {})
    manifest_path = Path(str(binding.get("path", "")))
    if not manifest_path.is_absolute() or not manifest_path.is_file():
        raise RuntimeError("formal input manifest is missing")
    if sha256_file(manifest_path) != binding.get("file_sha256"):
        raise RuntimeError("formal input manifest file SHA mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("manifest_sha256") != binding.get("canonical_sha256"):
        raise RuntimeError("formal input manifest canonical SHA mismatch")
    rows = [row for row in manifest.get("images", []) if row.get("role") == "train"]
    if len(rows) != int(binding.get("train_views", -1)):
        raise RuntimeError("formal train-view count mismatch")
    names = [str(row.get("image_name", "")) for row in rows]
    if len(set(names)) != len(rows) or any(not name for name in names):
        raise RuntimeError("formal train-view identity inventory mismatch")
    return rows


def validate_frozen_training_images(recipe: dict[str, Any], dataset_root: Path) -> None:
    rows = load_frozen_train_rows(recipe)
    image_root = dataset_root / "images"
    if not image_root.is_dir():
        raise RuntimeError("phase dataset has no images directory")
    files = {path.name: path for path in image_root.iterdir() if path.is_file()}
    if set(files) != {str(row.get("image_name")) for row in rows}:
        raise RuntimeError("phase dataset image-name inventory differs from frozen train split")
    for row in rows:
        name = str(row["image_name"])
        path = files[name]
        if path.stat().st_size != int(row["jpeg_bytes"]):
            raise RuntimeError(f"frozen training image byte count mismatch: {name}")
        if sha256_file(path) != row["jpeg_sha256"]:
            raise RuntimeError(f"frozen training image SHA mismatch: {name}")


def validate_prepared_method_input(recipe: dict[str, Any], dataset_root: Path) -> None:
    binding = recipe.get("prepared_method_input_binding")
    if not isinstance(binding, dict):
        raise RuntimeError("prepared per-method input binding is missing")
    evidence_path = Path(str(binding.get("evidence_path", "")))
    if not evidence_path.is_absolute() or not evidence_path.is_file():
        raise RuntimeError("prepared per-method input evidence is missing")
    if sha256_file(evidence_path) != binding.get("evidence_sha256"):
        raise RuntimeError("prepared per-method input evidence SHA mismatch")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if (
        evidence.get("status")
        != "PASS_PER_METHOD_INPUT_PREPARATION_NO_TRAINING_NO_PRIOR"
        or evidence.get("canonical_sha256") != canonical_sha256(evidence)
    ):
        raise RuntimeError("prepared per-method input evidence did not pass")
    if evidence.get("access_boundary", {}).get("all_images_participated_in_sfm") is not True:
        raise RuntimeError("prepared input does not bind all-image SfM before split")
    dependencies = recipe.get("benchmark_required_files_sha256", {})
    materializer = evidence.get("materializer", {})
    materializer_path = Path(str(materializer.get("path", "")))
    expected_materializer_sha = dependencies.get(
        "code/gcp/materialize_m3m_gcp_100k_method_inputs.py"
    )
    if (
        not materializer_path.is_absolute()
        or not materializer_path.is_file()
        or sha256_file(materializer_path) != materializer.get("sha256")
        or materializer.get("sha256") != expected_materializer_sha
    ):
        raise RuntimeError("prepared per-method input materializer identity mismatch")
    all_image = evidence.get("shared_all_image_sfm", {})
    if all_image.get("image_count") != 2510:
        raise RuntimeError("shared all-image SfM count mismatch")
    all_image_root = Path(str(all_image.get("path", "")))
    for name, row in all_image.get("files", {}).items():
        path = all_image_root / name
        if not path.is_file() or path.stat().st_size != row.get("bytes") or sha256_file(path) != row.get("sha256"):
            raise RuntimeError(f"shared all-image SfM identity mismatch: {name}")
    package_audit = all_image.get("package_audit", {})
    package_audit_path = Path(str(package_audit.get("path", "")))
    if (
        package_audit.get("status") != "pass"
        or not package_audit_path.is_absolute()
        or not package_audit_path.is_file()
        or sha256_file(package_audit_path) != package_audit.get("sha256")
    ):
        raise RuntimeError("shared all-image SfM package audit identity mismatch")
    profile = binding.get("input_profile")
    if profile == "city_train_records_with_full_all_image_sfm_points":
        block = evidence.get("city_track_compatibility", {})
        helper_dependency = "code/gcp/materialize_colmap_train_track_compatibility_streaming.py"
    elif profile == "metrogs_reciprocal_train_track_closure_after_all_image_sfm":
        block = evidence.get("metrogs_track_closure", {})
        helper_dependency = "code/gcp/filter_colmap_model_to_frozen_train_streaming.py"
    elif profile == "exact_formal_train_view_from_shared_all_image_sfm":
        block = evidence.get("formal_train_view", {})
        helper_dependency = None
    else:
        raise RuntimeError("unknown prepared input profile")
    if helper_dependency is not None:
        helper = block.get("evidence", {})
        helper_path = Path(str(helper.get("path", "")))
        if not helper_path.is_absolute() or not helper_path.is_file() or sha256_file(helper_path) != helper.get("sha256"):
            raise RuntimeError("prepared track helper evidence identity mismatch")
        helper_payload = json.loads(helper_path.read_text(encoding="utf-8"))
        helper_materializer = helper_payload.get("materializer", {})
        helper_materializer_path = Path(str(helper_materializer.get("path", "")))
        if (
            helper_payload.get("status") != "PASS"
            or helper_payload.get("passed") is not True
            or not helper_materializer_path.is_absolute()
            or not helper_materializer_path.is_file()
            or sha256_file(helper_materializer_path) != helper_materializer.get("sha256")
            or helper_materializer.get("sha256") != dependencies.get(helper_dependency)
        ):
            raise RuntimeError("prepared track helper materializer identity mismatch")
    prepared_root = Path(str(binding.get("dataset_root", "")))
    if not prepared_root.is_absolute() or not prepared_root.is_dir():
        raise RuntimeError("prepared per-method input dataset root is missing")
    phase_root = dataset_root.resolve()
    prepared_root = prepared_root.resolve()
    if prepared_root != phase_root:
        if (
            recipe.get("method_id") != "gsprior"
            or profile != "exact_formal_train_view_from_shared_all_image_sfm"
        ):
            raise RuntimeError("prepared per-method input dataset root mismatch")
        prior_command = recipe.get("phase_commands", {}).get("prior", [])
        if (
            "--source_scene" not in prior_command
            or prior_command[prior_command.index("--source_scene") + 1]
            != str(prepared_root)
            or "--output_scene" not in prior_command
            or prior_command[prior_command.index("--output_scene") + 1]
            != "{dataset_root}"
        ):
            raise RuntimeError("GSPrior normalization lineage differs from prepared input")
        validate_gsprior_normalized_input(
            prepared_root=prepared_root,
            normalized_root=phase_root,
            binding=binding,
        )
    sparse = prepared_root / "sparse" / "0"
    for name, expected_sha in binding.get("sparse_sha256", {}).items():
        path = sparse / name
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise RuntimeError(f"prepared per-method input sparse SHA mismatch: {name}")


def validate_gsprior_normalized_input(
    *, prepared_root: Path, normalized_root: Path, binding: dict[str, Any]
) -> None:
    manifest_path = normalized_root / "normalization_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("GSPrior normalized-input manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != "m3m_gsprior_colmap_camera_normalization_v1"
        or manifest.get("status") != "PASS"
        or Path(str(manifest.get("source_scene", ""))).resolve() != prepared_root
        or Path(str(manifest.get("reference_train_scene", ""))).resolve()
        != prepared_root
        or Path(str(manifest.get("output_scene", ""))).resolve() != normalized_root
    ):
        raise RuntimeError("GSPrior normalized-input manifest identity mismatch")
    source = manifest.get("source", {})
    source_key_by_name = {
        "cameras.bin": "cameras_bin",
        "images.bin": "images_bin",
        "points3D.bin": "points3D_bin",
        "points3D.ply": "points3D_ply",
    }
    for name, expected_sha in binding.get("sparse_sha256", {}).items():
        row = source.get(source_key_by_name[name])
        if not isinstance(row, dict) or row.get("sha256") != expected_sha:
            raise RuntimeError(f"GSPrior normalization source SHA mismatch: {name}")
    output = manifest.get("output", {})
    image_link = normalized_root / "images"
    if (
        output.get("images_are_directory_symlink") is not True
        or output.get("flat_and_sparse_zero_models_share_exact_files") is not True
        or not image_link.is_symlink()
        or image_link.resolve() != (prepared_root / "images").resolve()
        or Path(str(output.get("images_directory_target", ""))).resolve()
        != (prepared_root / "images").resolve()
    ):
        raise RuntimeError("GSPrior normalized-input image or sparse layout mismatch")
    output_files = output.get("files", {})
    if set(output_files) != set(binding.get("sparse_sha256", {})):
        raise RuntimeError("GSPrior normalized sparse-file inventory mismatch")
    for name, row in output_files.items():
        path = normalized_root / "sparse" / name
        nested = normalized_root / "sparse" / "0" / name
        if (
            not isinstance(row, dict)
            or not path.is_file()
            or path.stat().st_size != row.get("bytes")
            or sha256_file(path) != row.get("sha256")
            or not nested.is_symlink()
            or nested.resolve() != path.resolve()
        ):
            raise RuntimeError(f"GSPrior normalized output identity mismatch: {name}")
    validation = manifest.get("validation", {})
    if (
        validation.get("intrinsics_bytes_unchanged") is not True
        or validation.get("image_names_unchanged") is not True
        or validation.get("image_measurements_and_tracks_unchanged") is not True
        or validation.get("gcp_or_lidar_used") is not False
        or validation.get("image_pixels_resized_cropped_padded_or_reencoded") is not False
    ):
        raise RuntimeError("GSPrior normalized-input validation did not pass")


def validate_evaluation_camera_root(recipe: dict[str, Any]) -> None:
    binding = recipe.get("evaluation_camera_root_binding")
    if not isinstance(binding, dict):
        raise RuntimeError("evaluation camera-root binding is missing")
    evidence_path = Path(str(binding.get("evidence_path", "")))
    if (
        not evidence_path.is_absolute()
        or not evidence_path.is_file()
        or sha256_file(evidence_path) != binding.get("evidence_sha256")
    ):
        raise RuntimeError("evaluation camera-root evidence identity mismatch")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if (
        evidence.get("schema") != "m3m_gcp_100k_evaluation_camera_root_v1"
        or evidence.get("status") != binding.get("status_required")
        or evidence.get("canonical_sha256") != canonical_sha256(evidence)
        or evidence.get("scene") != SCENE
    ):
        raise RuntimeError("evaluation camera-root evidence did not pass")
    materializer = evidence.get("materializer", {})
    materializer_path = Path(str(materializer.get("path", "")))
    expected_materializer_sha = recipe.get("benchmark_required_files_sha256", {}).get(
        "code/gcp/materialize_m3m_gcp_100k_evaluation_camera_root.py"
    )
    if (
        not materializer_path.is_absolute()
        or not materializer_path.is_file()
        or sha256_file(materializer_path) != materializer.get("sha256")
        or materializer.get("sha256") != expected_materializer_sha
    ):
        raise RuntimeError("evaluation camera-root materializer identity mismatch")
    root = Path(str(binding.get("root", "")))
    output = evidence.get("output", {})
    if (
        not root.is_absolute()
        or not root.is_dir()
        or Path(str(output.get("root", ""))).resolve() != root.resolve()
        or output.get("view_count") != binding.get("view_count")
        or output.get("points3d_bin_point_count") != 0
        or binding.get("points3d_bin_point_count") != 0
        or evidence.get("truth_boundary", {}).get("heldout_rgb_present") is not False
        or evidence.get("truth_boundary", {}).get("gcp_or_lidar_used") is not False
    ):
        raise RuntimeError("evaluation camera-root boundary mismatch")
    packet_command = recipe.get("phase_commands", {}).get("packet", [])
    if (
        "--camera-root" not in packet_command
        or packet_command[packet_command.index("--camera-root") + 1] != str(root)
    ):
        raise RuntimeError("packet command does not bind the evaluation camera root")
    manifest_path = root / "EVALUATION_CAMERA_ROOT_MANIFEST.json"
    if not manifest_path.is_file() or sha256_file(manifest_path) != binding.get("evidence_sha256"):
        raise RuntimeError("evaluation camera-root manifest/evidence mismatch")
    images = root / "images"
    expected_target = Path(str(evidence.get("source_train", {}).get("root", ""))) / "images"
    if (
        not images.is_symlink()
        or images.resolve() != expected_target.resolve()
        or len([path for path in images.iterdir() if path.is_file()])
        != int(binding.get("view_count", -1))
    ):
        raise RuntimeError("evaluation camera-root RGB link mismatch")
    sparse = root / "sparse" / "0"
    output_files = output.get("files", {})
    expected_files = binding.get("sparse_sha256", {})
    if set(output_files) != set(expected_files):
        raise RuntimeError("evaluation camera-root sparse inventory mismatch")
    for name, expected_sha in expected_files.items():
        path = sparse / name
        row = output_files.get(name, {})
        if (
            not path.is_file()
            or sha256_file(path) != expected_sha
            or row.get("sha256") != expected_sha
            or row.get("bytes") != path.stat().st_size
        ):
            raise RuntimeError(f"evaluation camera-root sparse identity mismatch: {name}")
    if (sparse / "points3D.bin").read_bytes() != (0).to_bytes(8, "little"):
        raise RuntimeError("evaluation camera-root compatibility points3D.bin is not empty")


def read_bound_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"{label} is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} is not a JSON object")
    return payload


def path_inside(root: Path, value: object, *, label: str) -> Path:
    root = root.resolve()
    path = Path(str(value))
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"{label} escapes the frozen prior root: {path}") from exc
    return path


def validate_relative_prior_rows(
    rows: object, *, root: Path, expected_ids: set[str], id_field: str, label: str
) -> list[Path]:
    if not isinstance(rows, list) or len(rows) != len(expected_ids):
        raise RuntimeError(f"{label} row count mismatch")
    actual_ids = [str(row.get(id_field, "")) for row in rows if isinstance(row, dict)]
    if len(actual_ids) != len(rows) or len(set(actual_ids)) != len(rows) or set(actual_ids) != expected_ids:
        raise RuntimeError(f"{label} identity inventory mismatch")
    paths: list[Path] = []
    for row in rows:
        path = path_inside(root, row.get("relative_path", ""), label=label)
        if (
            not path.is_file()
            or path.stat().st_size != row.get("bytes")
            or sha256_file(path) != row.get("sha256")
        ):
            raise RuntimeError(f"{label} artifact changed: {path}")
        paths.append(path)
    if len(set(paths)) != len(paths):
        raise RuntimeError(f"{label} paths are not unique")
    return paths


def validate_scale_json(
    *, path: Path, expected_sha: object, expected_ids: set[str], positive: bool
) -> None:
    if not path.is_file() or sha256_file(path) != expected_sha:
        raise RuntimeError(f"depth-scale artifact changed: {path}")
    values = read_bound_json(path, label="depth-scale artifact")
    if set(values) != expected_ids:
        raise RuntimeError("depth-scale identity inventory mismatch")
    for name, row in values.items():
        if not isinstance(row, dict):
            raise RuntimeError(f"depth-scale row is invalid: {name}")
        try:
            scale = float(row["scale"])
            offset = float(row["offset"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"depth-scale row is invalid: {name}") from exc
        if not math.isfinite(scale) or not math.isfinite(offset) or (
            scale <= 0.0 if positive else scale == 0.0
        ):
            raise RuntimeError(f"depth-scale row is invalid: {name}")


def validate_bound_file_record(
    row: object, *, root: Path, label: str, require_vertex_count: bool = False
) -> Path:
    if not isinstance(row, dict):
        raise RuntimeError(f"{label} record is missing")
    path = path_inside(root, row.get("path", ""), label=label)
    if (
        not path.is_file()
        or path.stat().st_size != row.get("bytes")
        or sha256_file(path) != row.get("sha256")
    ):
        raise RuntimeError(f"{label} artifact changed: {path}")
    if require_vertex_count and int(row.get("vertex_count", 0)) <= 0:
        raise RuntimeError(f"{label} vertex count is invalid")
    return path


def validate_prior_outputs(
    recipe: dict[str, Any], *, dataset_root: Path, prior_root: Path
) -> list[Path]:
    method_id = str(recipe.get("method_id", ""))
    if method_id == "gsprior":
        binding = recipe.get("prepared_method_input_binding", {})
        prepared_root = Path(str(binding.get("dataset_root", ""))).resolve()
        validate_gsprior_normalized_input(
            prepared_root=prepared_root,
            normalized_root=dataset_root.resolve(),
            binding=binding,
        )
        return [prior_root / "normalization_manifest.json"]
    manifest_name = {
        "citygaussian_v2": "depth_prior_v1.json",
        "citygs_x": "depth_and_multiview_prior_v1.json",
        "metrogs": "training_priors.json",
    }.get(method_id)
    if manifest_name is None:
        raise RuntimeError(f"{method_id}: unexpected prior phase")
    manifest_path = prior_root / manifest_name
    payload = read_bound_json(manifest_path, label=f"{method_id} prior manifest")
    if (
        payload.get("status") != "PASS"
        or payload.get("passed") is not True
        or payload.get("method_id") != method_id
        or payload.get("scene") != SCENE
        or payload.get("protocol_id") != "m3m_gcp_native_quarter_geometry_v2"
    ):
        raise RuntimeError(f"{method_id} prior manifest did not pass")
    frozen_rows = load_frozen_train_rows(recipe)
    expected_names = {str(row.get("image_name", "")) for row in frozen_rows}
    if len(frozen_rows) != 2196 or len(expected_names) != 2196 or "" in expected_names:
        raise RuntimeError("external-prior validation lacks the exact 2196-view train inventory")
    if payload.get("input_class") != "rgb_colmap_external_geometry_prior":
        raise RuntimeError(f"{method_id} prior input-class boundary mismatch")
    outputs = [manifest_path]
    if method_id == "citygaussian_v2":
        boundary = payload.get("access_boundary", {})
        claims = payload.get("claims", {})
        if (
            boundary.get("training_rgb_opened") != 2196
            or boundary.get("heldout_rgb_opened") != 0
            or boundary.get("gcp_annotations_opened") != 0
            or boundary.get("lidar_opened") != 0
            or boundary.get("only_training_rgb_and_train_only_colmap_supplied_to_prior_commands") is not True
            or claims.get("heldout_gcp_lidar_or_orthophoto_truth_used") is not False
        ):
            raise RuntimeError("CityGaussianV2 prior truth boundary mismatch")
        if Path(str(payload.get("access_boundary", {}).get("isolated_dataset_root", ""))).resolve() != prior_root.resolve():
            raise RuntimeError("CityGaussianV2 prior root mismatch")
        outputs.extend(validate_relative_prior_rows(
            payload.get("depth_outputs"), root=prior_root,
            expected_ids=expected_names, id_field="image_name",
            label="CityGaussianV2 depth prior",
        ))
        scales = payload.get("depth_scales", {})
        scale_path = path_inside(prior_root, scales.get("path", ""), label="CityGaussianV2 scales")
        if scales.get("record_count") != 2196:
            raise RuntimeError("CityGaussianV2 scale count mismatch")
        validate_scale_json(
            path=scale_path, expected_sha=scales.get("sha256"),
            expected_ids=expected_names, positive=False,
        )
        outputs.append(scale_path)
    elif method_id == "citygs_x":
        boundary = payload.get("access_boundary", {})
        claims = payload.get("claims", {})
        if (
            boundary.get("training_rgb_opened") != 2196
            or boundary.get("heldout_rgb_opened") != 0
            or boundary.get("gcp_annotations_opened") != 0
            or boundary.get("lidar_opened") != 0
            or boundary.get("only_training_rgb_and_train_only_colmap_supplied_to_prior_commands") is not True
            or claims.get("heldout_gcp_lidar_or_orthophoto_truth_used") is not False
        ):
            raise RuntimeError("CityGS-X prior truth boundary mismatch")
        dataset = payload.get("dataset", {})
        if Path(str(dataset.get("path", ""))).resolve() != prior_root.resolve():
            raise RuntimeError("CityGS-X prior root mismatch")
        expected_stems = {Path(name).stem for name in expected_names}
        if len(expected_stems) != 2196:
            raise RuntimeError("CityGS-X frozen image stems are not unique")
        outputs.extend(validate_relative_prior_rows(
            dataset.get("depth_outputs"), root=prior_root,
            expected_ids=expected_stems, id_field="image_stem",
            label="CityGS-X depth prior",
        ))
        outputs.extend(validate_relative_prior_rows(
            dataset.get("multi_view_masks"), root=prior_root,
            expected_ids=expected_stems, id_field="image_stem",
            label="CityGS-X multi-view mask",
        ))
        params = dataset.get("depth_params", {})
        params_path = path_inside(prior_root, params.get("path", ""), label="CityGS-X scales")
        if params.get("record_count") != 2196:
            raise RuntimeError("CityGS-X scale count mismatch")
        validate_scale_json(
            path=params_path, expected_sha=params.get("sha256"),
            expected_ids=expected_stems, positive=True,
        )
        outputs.append(params_path)
    elif method_id == "metrogs":
        claims = payload.get("claims", {})
        if (
            claims.get("heldout_rgb_read") is not False
            or claims.get("gcp_truth_read") is not False
            or claims.get("lidar_read") is not False
            or claims.get("training_rgb_only") is not True
            or claims.get("training_colmap_only") is not True
            or claims.get("formal_training_started") is not False
        ):
            raise RuntimeError("MetroGS prior truth boundary mismatch")
        if Path(str(payload.get("input", {}).get("dataset", ""))).resolve() != prior_root.resolve():
            raise RuntimeError("MetroGS prior root mismatch")
        moge = payload.get("moge", {})
        if moge.get("depth_count") != 2196 or moge.get("scale_count") != 2196:
            raise RuntimeError("MetroGS depth/scale count mismatch")
        outputs.extend(validate_relative_prior_rows(
            moge.get("depth_outputs"), root=prior_root,
            expected_ids=expected_names, id_field="image_name",
            label="MetroGS MoGe depth/mask prior",
        ))
        scale_path = path_inside(prior_root, moge.get("scale_manifest", ""), label="MetroGS scales")
        validate_scale_json(
            path=scale_path, expected_sha=moge.get("scale_manifest_sha256"),
            expected_ids=expected_names, positive=True,
        )
        outputs.append(scale_path)
        multi_view = payload.get("multi_view", {})
        multi_view_path = path_inside(prior_root, multi_view.get("path", ""), label="MetroGS multi-view")
        if (
            not multi_view_path.is_file()
            or multi_view_path.stat().st_size != multi_view.get("bytes")
            or sha256_file(multi_view_path) != multi_view.get("sha256")
            or multi_view.get("camera_count") != 2196
        ):
            raise RuntimeError("MetroGS multi-view artifact changed")
        neighbors = read_bound_json(multi_view_path, label="MetroGS multi-view artifact")
        if set(neighbors) != expected_names:
            raise RuntimeError("MetroGS multi-view identity inventory mismatch")
        for name, values in neighbors.items():
            if (
                not isinstance(values, list) or not values or name in values
                or not set(values) <= expected_names
            ):
                raise RuntimeError(f"MetroGS multi-view row is invalid: {name}")
        outputs.append(multi_view_path)
        pi3 = payload.get("pi3", {})
        pointmaps = pi3.get("block_pointmaps")
        if not isinstance(pointmaps, list) or len(pointmaps) != 4:
            raise RuntimeError("MetroGS Pi3 block pointmap count mismatch")
        for index, row in enumerate(pointmaps):
            outputs.append(validate_bound_file_record(
                row, root=prior_root, label=f"MetroGS Pi3 block pointmap {index}",
                require_vertex_count=True,
            ))
        outputs.append(validate_bound_file_record(
            pi3.get("merged_pointmap"), root=prior_root,
            label="MetroGS merged Pi3 pointmap", require_vertex_count=True,
        ))
        marker_path = prior_root / "TRAINING_PRIORS_PASS"
        marker = read_bound_json(marker_path, label="MetroGS prior PASS marker")
        if (
            marker.get("schema") != "m3m_gcp_100k_metrogs_prior_pass_v1"
            or marker.get("status") != "PASS"
            or marker.get("scene") != SCENE
            or marker.get("method_id") != method_id
            or Path(str(marker.get("prior_evidence_path", ""))).resolve()
            != manifest_path.resolve()
            or marker.get("prior_evidence_sha256") != sha256_file(manifest_path)
        ):
            raise RuntimeError("MetroGS prior PASS marker identity mismatch")
        outputs.append(marker_path)
    if len(outputs) != len(set(path.resolve() for path in outputs)):
        raise RuntimeError(f"{method_id} prior product paths are not unique")
    return sorted((path.resolve() for path in outputs), key=str)


def validate_training_outputs(recipe: dict[str, Any], *, run_root: Path) -> list[Path]:
    method_id = str(recipe.get("method_id", ""))
    budget = recipe.get("budget", {})
    if method_id in {"2dgs", "pgsr", "rade_gs", "qgs", "gsprior", "sof"}:
        iteration = int(budget.get("value", -1))
        paths = [
            run_root / "model" / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
        ]
    elif method_id == "3dgs_original":
        reuse = recipe.get("reuse_model_binding", {})
        paths = [run_root / str(reuse.get("point_cloud_relative_path", ""))]
    elif method_id == "citygaussian_v2":
        summary_path = run_root / "pipeline" / "pipeline_summary.json"
        summary = read_bound_json(summary_path, label="CityGaussianV2 training summary")
        checkpoint = summary.get("merged_checkpoint", {})
        checkpoint_path = path_inside(
            run_root, checkpoint.get("path", ""), label="CityGaussianV2 checkpoint"
        )
        if (
            summary.get("method_id") != method_id
            or summary.get("scene") != SCENE
            or summary.get("protocol_id") != "m3m_gcp_native_quarter_geometry_v2"
            or summary.get("status") != "PIPELINE_PASS"
            or summary.get("mode") != "formal"
            or summary.get("formal_result") is not True
            or summary.get("coarse_steps") != 30000
            or summary.get("fine_steps") != 60000
            or checkpoint_path.stat().st_size != checkpoint.get("bytes")
        ):
            raise RuntimeError("CityGaussianV2 training summary did not pass")
        if sha256_file(checkpoint_path) != checkpoint.get("sha256"):
            raise RuntimeError("CityGaussianV2 merged checkpoint identity mismatch")
        paths = [summary_path, checkpoint_path]
    elif method_id == "citygs_x":
        summary_path = run_root / "model" / "training_wrapper_summary.json"
        summary = read_bound_json(summary_path, label="CityGS-X training summary")
        checkpoint = summary.get("checkpoint", {})
        checkpoint_root = path_inside(
            run_root, checkpoint.get("path", ""), label="CityGS-X checkpoint"
        )
        point_cloud = checkpoint_root / str(checkpoint.get("point_cloud_file", ""))
        attributes = checkpoint.get("additional_attributes", {})
        optimizer = checkpoint.get("optimizer_checkpoint", {})
        attributes_path = Path(str(attributes.get("path", ""))).resolve()
        optimizer_path = Path(str(optimizer.get("path", ""))).resolve()
        if (
            summary.get("method_id") != method_id
            or summary.get("scene") != SCENE
            or summary.get("protocol_id") != "m3m_gcp_native_quarter_geometry_v2"
            or summary.get("status") != "TRAINING_PASS"
            or summary.get("mode") != "formal"
            or summary.get("formal_result") is not True
            or summary.get("iterations") != 100000
            or point_cloud.stat().st_size != checkpoint.get("point_cloud_bytes")
            or sha256_file(point_cloud) != checkpoint.get("point_cloud_sha256")
            or attributes_path != checkpoint_root / "additional_attributes.npz"
            or attributes_path.stat().st_size != attributes.get("bytes")
            or sha256_file(attributes_path) != attributes.get("sha256")
            or optimizer_path != checkpoint_root / "checkpoints.pth"
            or optimizer_path.stat().st_size != optimizer.get("bytes")
            or sha256_file(optimizer_path) != optimizer.get("sha256")
        ):
            raise RuntimeError("CityGS-X final checkpoint identity mismatch")
        paths = [
            summary_path,
            point_cloud,
            attributes_path,
            optimizer_path,
        ]
    elif method_id == "metrogs":
        summary_path = run_root / "model" / "training_wrapper_summary.json"
        summary = read_bound_json(summary_path, label="MetroGS training summary")
        checkpoint = summary.get("checkpoint", {})
        cleanup = summary.get("rank_checkpoint_cleanup", {})
        merged = path_inside(
            run_root, checkpoint.get("merged_path", ""), label="MetroGS merged checkpoint"
        )
        point_cloud = path_inside(
            run_root, checkpoint.get("point_cloud_path", ""), label="MetroGS point cloud"
        )
        cleanup_inventory = path_inside(
            run_root, cleanup.get("inventory_path", ""), label="MetroGS cleanup inventory"
        )
        if (
            summary.get("method_id") != method_id
            or summary.get("scene") != SCENE
            or summary.get("protocol_id") != "m3m_gcp_native_quarter_geometry_v2"
            or summary.get("status") != "TRAINING_PASS"
            or summary.get("mode") != "formal"
            or summary.get("effective_iterations") != 150000
            or summary.get("optimizer_steps") != 37500
            or merged.stat().st_size != checkpoint.get("merged_bytes")
            or sha256_file(merged) != checkpoint.get("merged_sha256")
            or point_cloud.stat().st_size != checkpoint.get("point_cloud_bytes")
            or sha256_file(point_cloud) != checkpoint.get("point_cloud_sha256")
            or sha256_file(cleanup_inventory) != cleanup.get("inventory_sha256")
        ):
            raise RuntimeError("MetroGS final checkpoint identity mismatch")
        paths = [summary_path, merged, point_cloud, cleanup_inventory]
    else:  # pragma: no cover - frozen pool is exhaustive
        raise RuntimeError(f"unsupported method final-output validation: {method_id}")
    for path in paths:
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"{method_id} required final output is missing or empty: {path}")
    if method_id in {"2dgs", "pgsr", "rade_gs", "qgs", "gsprior", "sof", "3dgs_original"}:
        validate_gaussian_ply(paths[0], method_id=method_id)
    elif method_id == "citygaussian_v2":
        validate_torch_checkpoint(paths[1])
    elif method_id == "citygs_x":
        validate_gaussian_ply(paths[1], method_id=method_id)
        validate_npz(paths[2])
        validate_torch_checkpoint(paths[3])
    elif method_id == "metrogs":
        validate_torch_checkpoint(paths[1])
        validate_gaussian_ply(paths[2], method_id=method_id)
    return paths


def validate_packet_outputs(recipe: dict[str, Any], *, packet_root: Path) -> list[Path]:
    manifest_path = packet_root / "depth_export_manifest.json"
    mapping_path = packet_root / "depth_map_index.csv"
    manifest = read_bound_json(manifest_path, label="packet manifest")
    rows = manifest.get("depth_index")
    if not isinstance(rows, list):
        rows = manifest.get("packet_index")
    if (
        manifest.get("scene") != SCENE
        or manifest.get("protocol_id") != "m3m_gcp_native_quarter_geometry_v2"
        or manifest.get("rendered_view_count") != 2196
        or not isinstance(rows, list)
        or len(rows) != 2196
    ):
        raise RuntimeError("packet manifest did not cover all 2196 frozen train views")
    expected_names = {
        str(row.get("image_name", "")) for row in load_frozen_train_rows(recipe)
    }
    actual_names = [str(row.get("image_name", "")) for row in rows]
    if len(set(actual_names)) != 2196 or set(actual_names) != expected_names:
        raise RuntimeError("packet manifest image inventory differs from frozen train split")
    packet_paths: list[Path] = []
    packet_root_resolved = packet_root.resolve()
    for row in rows:
        path = Path(str(row.get("packet_path", ""))).resolve()
        if (
            path.parent != packet_root_resolved
            or not path.is_file()
            or path.stat().st_size != int(row.get("packet_bytes", -1))
            or sha256_file(path) != row.get("packet_sha256")
            or row.get("packet_recompute_passed") is not True
        ):
            raise RuntimeError(f"packet file identity/postcondition mismatch: {path}")
        packet_paths.append(path)
    if len(set(packet_paths)) != 2196:
        raise RuntimeError("packet paths are not unique")
    if not mapping_path.is_file():
        raise RuntimeError("packet mapping CSV is missing")
    with mapping_path.open("r", encoding="utf-8-sig", newline="") as handle:
        mapping = list(csv.DictReader(handle))
    if (
        len(mapping) != 2196
        or [row.get("image_name") for row in mapping] != actual_names
        or [Path(str(row.get("packet_path", ""))).resolve() for row in mapping]
        != packet_paths
    ):
        raise RuntimeError("packet mapping CSV differs from the manifest")
    return [manifest_path, mapping_path, *packet_paths]


def validate_phase_postconditions(
    recipe: dict[str, Any], *, phase: str, run_root: Path, dataset_root: Path,
    prior_root: Path, packet_root: Path | None,
) -> list[Path]:
    if phase == "prior":
        return validate_prior_outputs(
            recipe, dataset_root=dataset_root, prior_root=prior_root
        )
    if phase == "training":
        return validate_training_outputs(recipe, run_root=run_root)
    if phase == "packet":
        if packet_root is None:
            raise RuntimeError("packet postcondition has no packet root")
        return validate_packet_outputs(recipe, packet_root=packet_root)
    raise RuntimeError(f"unsupported phase postcondition: {phase}")


def build_phase_product_rows(
    paths: list[Path], *, phase: str, method_id: str | None = None
) -> list[dict[str, Any]]:
    resolved = sorted({path.resolve() for path in paths}, key=str)
    if len(resolved) != len(paths):
        raise RuntimeError("phase postcondition returned duplicate product paths")
    return [
        phase_product_row(
            path,
            validate_model_container=phase == "training",
            method_id=method_id,
        )
        for path in resolved
    ]


def validate_phase_success_products(
    payload: dict[str, Any], *, expected_paths: list[Path], phase: str,
    frozen_budget: dict[str, Any],
) -> None:
    if payload.get("frozen_budget") != frozen_budget:
        raise RuntimeError("phase success frozen budget mismatch")
    completion = payload.get("completion_evidence")
    if (
        not isinstance(completion, dict)
        or completion.get("required_product_postvalidation_passed") is not True
        or not isinstance(completion.get("progress_unit"), str)
        or not isinstance(completion.get("last_valid_progress"), (int, float))
        or not math.isfinite(float(completion.get("last_valid_progress", float("nan"))))
    ):
        raise RuntimeError("phase success completion evidence mismatch")
    rows = payload.get("products")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("phase success product inventory is empty")
    actual_paths = [revalidate_phase_product_row(row) for row in rows]
    if actual_paths != sorted(actual_paths, key=str) or len(actual_paths) != len(set(actual_paths)):
        raise RuntimeError("phase success product inventory order/cardinality mismatch")
    expected_rows = build_phase_product_rows(
        expected_paths, phase=phase, method_id=str(payload.get("method_id", ""))
    )
    if rows != expected_rows:
        raise RuntimeError("phase success products differ from current method postconditions")


def validate_phase_success_environment(payload: dict[str, Any]) -> Path:
    path = Path(str(payload.get("environment_manifest_path", "")))
    if (
        not path.is_absolute()
        or not path.is_file()
        or path.is_symlink()
        or sha256_file(path) != payload.get("environment_manifest_sha256")
    ):
        raise RuntimeError("phase success environment-manifest binding mismatch")
    environment = json.loads(path.read_text(encoding="utf-8"))
    limits = environment.get("resource_limits", {})
    child = limits.get("child_actual", {})
    parent_after = limits.get("parent_after", {})

    def hard_ok(value: object) -> bool:
        return value == "unlimited" or (
            isinstance(value, int) and value >= REQUIRED_NOFILE_SOFT
        )

    if (
        environment.get("schema") != "m3m_gcp_100k_execution_environment_v2"
        or environment.get("scene") != payload.get("scene")
        or environment.get("method_id") != payload.get("method_id")
        or environment.get("phase") != payload.get("phase")
        or environment.get("canonical_sha256") != canonical_sha256(environment)
        or limits.get("required_soft") != REQUIRED_NOFILE_SOFT
        or limits.get("hard_minimum") != REQUIRED_NOFILE_SOFT
        or parent_after.get("soft") != REQUIRED_NOFILE_SOFT
        or not hard_ok(parent_after.get("hard"))
        or child.get("soft") != REQUIRED_NOFILE_SOFT
        or not hard_ok(child.get("hard"))
    ):
        raise RuntimeError("phase success RLIMIT_NOFILE environment evidence mismatch")
    return path


def validate_prior_phase_success(
    *, recipe: dict[str, Any], recipe_path: Path, run_root: Path,
    dataset_root: Path, prior_root: Path, replacements: dict[str, str],
) -> None:
    if "prior" not in recipe.get("phase_commands", {}):
        return
    path = (
        Path(str(recipe.get("authorized_evidence_root", "")))
        / "prior"
        / "phase_success.json"
    )
    payload = read_bound_json(path, label="prior phase success")
    expected_command = [
        item.format(**replacements)
        for item in recipe.get("phase_commands", {}).get("prior", [])
    ]
    expected_products = validate_prior_outputs(
        recipe, dataset_root=dataset_root, prior_root=prior_root
    )
    if (
        payload.get("schema") != "m3m_gcp_100k_phase_success_v2"
        or payload.get("status") != "PASS"
        or payload.get("scene") != SCENE
        or payload.get("method_id") != recipe.get("method_id")
        or payload.get("phase") != "prior"
        or payload.get("recipe_sha256") != sha256_file(recipe_path)
        or payload.get("command_sha256") != command_sha256(expected_command)
        or payload.get("canonical_sha256") != canonical_sha256(payload)
    ):
        raise RuntimeError("training requires the exact successful prior phase")
    validate_phase_success_environment(payload)
    validate_phase_success_products(
        payload,
        expected_paths=expected_products,
        phase="prior",
        frozen_budget=recipe.get("budget", {}),
    )


def classify_oom(stderr_text: str, delta: dict[str, int]) -> tuple[str, str | None]:
    lower = stderr_text.lower()
    if "cuda" in lower and "out of memory" in lower:
        return "OOM_UNRANKED", "CUDA_OUT_OF_MEMORY"
    if delta["oom_kill"] > 0:
        return "OOM_UNRANKED", "CGROUP_OOM_KILL"
    if delta["oom"] > 0 or "out of memory" in lower:
        return "OOM_UNRANKED", "HOST_OOM"
    return "FAILED_UNRANKED", None


def validate_model_identity_bundle(
    *, manifest_path: Path, method_id: str, run_root: Path, recipe: dict[str, Any],
    repo: Path,
) -> dict[str, Any]:
    """Rehash every file in the frozen 100K model-identity bundle."""
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if set(payload) != {
        "schema",
        "protocol_id",
        "scene",
        "method_id",
        "run_root",
        "inventory",
        "canonical_sha256",
    }:
        raise RuntimeError("100K model-identity field inventory mismatch")
    if (
        payload.get("schema") != "m3m_gcp_100k_model_identity_v1"
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("scene") != SCENE
        or payload.get("method_id") != method_id
        or Path(str(payload.get("run_root", ""))).resolve() != run_root.resolve()
        or payload.get("canonical_sha256") != canonical_sha256(payload)
    ):
        raise RuntimeError("100K model-identity metadata mismatch")
    inventory = payload.get("inventory")
    if not isinstance(inventory, list) or not inventory:
        raise RuntimeError("100K model-identity inventory is empty")
    paths: list[str] = []
    for row in inventory:
        if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}:
            raise RuntimeError("100K model-identity inventory row mismatch")
        path = Path(str(row.get("path", "")))
        if not path.is_absolute() or not path.is_file():
            raise RuntimeError(f"100K model-identity file missing: {path}")
        if path.stat().st_size != row.get("bytes") or sha256_file(path) != row.get("sha256"):
            raise RuntimeError(f"100K model-identity file changed: {path}")
        paths.append(str(path))
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise RuntimeError("100K model-identity inventory order/cardinality mismatch")
    required_final_paths = {
        str(path.resolve())
        for path in validate_training_outputs(recipe, run_root=run_root)
    }
    if not required_final_paths.issubset(set(paths)):
        missing = sorted(required_final_paths - set(paths))
        raise RuntimeError(
            "100K model identity omits the method's actual final model: "
            + ", ".join(missing)
        )
    required_phases = [
        phase
        for phase in ("prior", "training")
        if phase in recipe.get("phase_commands", {})
    ]
    evidence_root = Path(str(recipe.get("authorized_evidence_root", ""))).resolve()
    expected_marker_paths = {
        phase: (evidence_root / phase / "phase_success.json").resolve()
        for phase in required_phases
    }
    phase_markers = {
        Path(path).resolve()
        for path in paths
        if Path(path).name == "phase_success.json"
    }
    if phase_markers != set(expected_marker_paths.values()):
        raise RuntimeError(
            "100K model identity phase-success marker inventory mismatch"
        )
    for phase, marker_path in expected_marker_paths.items():
        marker = read_bound_json(marker_path, label="frozen phase success")
        roots = recipe.get("phase_roots", {}).get(phase, {})
        source = recipe.get("source_bindings", {}).get(phase, {})
        replacements = {
            "repo": str(repo.resolve()),
            "dataset_root": str(
                Path(str(roots.get("dataset_root", ""))).resolve()
            ),
            "source_root": str(Path(str(source.get("root", ""))).resolve()),
            "prior_root": str(Path(str(roots.get("prior_root", ""))).resolve()),
            "run_root": str(run_root.resolve()),
            "packet_set_root": str(
                Path(str(recipe.get("authorized_packet_set_root", ""))).resolve()
            ),
        }
        template = recipe.get("phase_commands", {}).get(phase, [])
        expected_command = [str(item).format(**replacements) for item in template]
        if (
            marker.get("schema") != "m3m_gcp_100k_phase_success_v2"
            or marker.get("status") != "PASS"
            or marker.get("scene") != SCENE
            or marker.get("method_id") != method_id
            or marker.get("phase") != phase
            or marker.get("recipe_sha256") != sha256_file(Path(recipe["_recipe_path"]))
            or marker.get("command_sha256") != command_sha256(expected_command)
            or marker.get("canonical_sha256") != canonical_sha256(marker)
        ):
            raise RuntimeError("frozen phase success identity mismatch")
        validate_phase_success_environment(marker)
        if phase == "training":
            expected_products = validate_training_outputs(recipe, run_root=run_root)
        elif phase == "prior":
            roots = recipe.get("phase_roots", {}).get("prior", {})
            expected_products = validate_prior_outputs(
                recipe,
                dataset_root=Path(str(roots.get("dataset_root", ""))).resolve(),
                prior_root=Path(str(roots.get("prior_root", ""))).resolve(),
            )
        else:
            raise RuntimeError(f"unexpected frozen phase success marker: {phase}")
        validate_phase_success_products(
            marker,
            expected_paths=expected_products,
            phase=phase,
            frozen_budget=recipe.get("budget", {}),
        )
    return payload


def validate_packet_freeze_binding(
    *, args: argparse.Namespace, recipe: dict[str, Any], plan: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    freeze_path = args.scene_attempt_freeze
    frozen_paths = plan.get("attempt_freeze", {})
    expected_freeze = Path(str(frozen_paths.get("scene_attempt_freeze_path", ""))).resolve()
    expected_methods = Path(str(frozen_paths.get("attempt_manifest_path", ""))).resolve()
    expected_identity_root = Path(str(frozen_paths.get("model_identity_root", ""))).resolve()
    if freeze_path is None or freeze_path.resolve() != expected_freeze:
        raise RuntimeError("packet scene-attempt freeze path differs from the frozen plan")
    if not freeze_path.is_file():
        raise RuntimeError("packet phase requires an immutable scene-attempt freeze")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    errors, methods = validate_scene_attempt_freeze(
        freeze, freeze_path=freeze_path, expected_scene=SCENE
    )
    if errors or methods is None:
        raise RuntimeError("packet scene-attempt freeze is invalid: " + "; ".join(errors))
    if Path(str(freeze.get("methods_manifest_path", ""))).resolve() != expected_methods:
        raise RuntimeError("packet methods manifest path differs from the frozen plan")
    rows = [
        row for row in methods.get("methods", [])
        if row.get("method_id") == args.method_id
    ]
    if len(rows) != 1 or rows[0].get("attempt_status") != "READY_FOR_EVALUATION":
        raise RuntimeError("packet method is not frozen READY_FOR_EVALUATION")
    row = rows[0]
    if Path(str(row.get("run_root", ""))).resolve() != args.run_root.resolve():
        raise RuntimeError("packet run root differs from frozen method row")
    if row.get("recipe_sha256") != sha256_file(args.recipe):
        raise RuntimeError("packet recipe differs from frozen method row")
    if row.get("renderer_adapter_sha256") != recipe.get("renderer_adapter_sha256"):
        raise RuntimeError("packet renderer adapter differs from frozen method row")
    identity_path = Path(str(row.get("model_checkpoint_path", ""))).resolve()
    if identity_path != expected_identity_root / f"{args.method_id}.json":
        raise RuntimeError("packet model-identity path differs from the frozen plan")
    bound_recipe = dict(recipe)
    bound_recipe["_recipe_path"] = str(args.recipe.resolve())
    validate_model_identity_bundle(
        manifest_path=identity_path,
        method_id=args.method_id,
        run_root=args.run_root,
        recipe=bound_recipe,
        repo=args.repo,
    )
    return row, sha256_file(freeze_path)


def write_failure_evidence(
    *,
    path: Path,
    recipe: dict[str, Any],
    argv: list[str],
    run_root: Path,
    environment_path: Path,
    stdout_path: Path,
    stderr_path: Path,
    started_at: str,
    ended_at: str,
    exit_code: int,
    last_progress: float,
    progress_unit: str,
    peak_gpu_mib: float,
    maximum_rss_kib: int,
    events_delta: dict[str, int],
    extra_error: str | None,
    failure_stage: str,
    model_checkpoint_sha256: str | None,
    scene_attempt_freeze_sha256: str | None,
) -> dict[str, Any]:
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
    status, oom_signal = classify_oom(stderr_text, events_delta)
    errors = [f"command exited with code {exit_code}"]
    if extra_error:
        errors.append(extra_error)
        if "100 GiB" in extra_error:
            status, oom_signal = "FAILED_UNRANKED", None
    if failure_stage == "packet_export":
        status = "INCOMPLETE_UNRANKED"
    payload = {
        "schema": "m3m_gcp_lidar_failure_evidence_v1",
        "protocol_id": "m3m_gcp_lidar_rendered_surface_v1",
        "scene": SCENE,
        "method_id": recipe["method_id"],
        "input_class": recipe["input_class"],
        "seed": 0,
        "status": status,
        "failure_stage": failure_stage,
        "run_root": str(run_root),
        "model_checkpoint_sha256": model_checkpoint_sha256,
        "scene_attempt_freeze_sha256": scene_attempt_freeze_sha256,
        "command_argv": argv,
        "command_sha256": command_sha256(argv),
        "environment_manifest_path": str(environment_path),
        "environment_manifest_sha256": sha256_file(environment_path),
        "recipe_sha256": sha256_file(Path(recipe["_recipe_path"])),
        "renderer_adapter_sha256": recipe["renderer_adapter_sha256"],
        "started_at_utc": started_at,
        "ended_at_utc": ended_at,
        "exit_code": int(exit_code),
        "last_valid_progress": {"unit": progress_unit, "value": float(last_progress)},
        "peak_gpu_memory_mib": float(peak_gpu_mib),
        "process_maximum_rss_kib": int(maximum_rss_kib),
        "cgroup_memory_events_delta": events_delta,
        "oom_signal": oom_signal,
        "stdout_path": str(stdout_path),
        "stdout_sha256": sha256_file(stdout_path),
        "stderr_path": str(stderr_path),
        "stderr_sha256": sha256_file(stderr_path),
        "errors": errors,
    }
    payload["canonical_sha256"] = canonical_sha256(payload)
    create_exclusive_json(path, payload)
    return payload


def run_phase(args: argparse.Namespace, recipe: dict[str, Any], argv: list[str]) -> int:
    limit_policy = recipe.get("process_resource_limits", {})
    if (
        args.phase not in limit_policy.get("applies_to_phases", [])
        or limit_policy.get("rlimit_nofile_soft") != REQUIRED_NOFILE_SOFT
        or limit_policy.get("rlimit_nofile_hard_minimum") != REQUIRED_NOFILE_SOFT
    ):
        raise RuntimeError("phase is not covered by the frozen RLIMIT_NOFILE contract")
    nofile_evidence = configure_nofile_limit(REQUIRED_NOFILE_SOFT)
    validate_capacity(args.capacity_root, args.phase)
    evidence_dir = args.failure_evidence.parent
    stdout_path = evidence_dir / "command.stdout.log"
    stderr_path = evidence_dir / "command.stderr.log"
    environment_path = evidence_dir / "environment.json"
    success_path = evidence_dir / "phase_success.json"
    immutable_outputs = (
        args.failure_evidence,
        stdout_path,
        stderr_path,
        environment_path,
        success_path,
    )
    if any(path.exists() or path.is_symlink() for path in immutable_outputs):
        raise FileExistsError(
            "phase evidence already exists; child-started attempts are immutable and cannot be retried"
        )
    if args.phase in {"prior", "training"} and (
        args.run_root.exists() or args.run_root.is_symlink()
    ):
        raise FileExistsError(
            "prior/training requires an absent method run root at guard admission"
        )
    frozen_method: dict[str, Any] | None = None
    freeze_sha: str | None = None
    if args.phase == "packet":
        if args.packet_set_root is None or args.packet_state is None:
            raise RuntimeError("packet phase requires --packet-set-root and --packet-state")
        if args.packet_set_root.exists():
            raise RuntimeError("packet set root already exists")
        frozen_method, freeze_sha = validate_packet_freeze_binding(
            args=args, recipe=recipe, plan=args.execution_plan
        )
        acquire_packet_state(args.packet_state, args.method_id, args.packet_set_root)
    if args.phase == "training":
        args.run_root.mkdir(parents=True, exist_ok=False)
    materialize_phase_files(
        recipe, phase=args.phase, run_root=args.run_root, replacements=args.replacements
    )
    if args.phase == "prior" and (
        args.run_root.exists() or args.run_root.is_symlink()
    ):
        raise RuntimeError("prior setup created the forbidden training run root")
    evidence_dir.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    environment = {
        "schema": "m3m_gcp_100k_execution_environment_v2",
        "scene": SCENE,
        "method_id": args.method_id,
        "phase": args.phase,
        "argv": argv,
        "python": sys.version,
        "platform": platform.platform(),
        "gpu_prelaunch": getattr(args, "gpu_prelaunch", {}),
        "resource_limits": nofile_evidence,
        "started_at_utc": started_at,
    }
    write_environment_manifest(environment_path, environment)
    before = cgroup_memory_events()
    last_progress = 0.0
    peak_gpu = 0.0
    cap_error: str | None = None
    inheritance_error: str | None = None
    pattern = re.compile(args.progress_regex) if args.progress_regex else None
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
        process = subprocess.Popen(argv, stdout=stdout_handle, stderr=stderr_handle, text=True)
        try:
            child_actual = observe_child_nofile_limit(process)
            nofile_evidence["child_actual"] = child_actual
            if (
                child_actual.get("soft") != REQUIRED_NOFILE_SOFT
                or (
                    child_actual.get("hard") != "unlimited"
                    and int(child_actual.get("hard", -1)) < REQUIRED_NOFILE_SOFT
                )
            ):
                raise RuntimeError(
                    "child did not inherit the frozen RLIMIT_NOFILE contract: "
                    f"{child_actual}"
                )
        except Exception as exc:  # noqa: BLE001 - immutable child-started failure
            inheritance_error = f"child RLIMIT_NOFILE evidence failed: {type(exc).__name__}: {exc}"
            nofile_evidence["child_actual_error"] = inheritance_error
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
        write_environment_manifest(environment_path, environment)
        while process.poll() is None:
            peak_gpu = max(peak_gpu, gpu_memory_for_pid(process.pid))
            if pattern:
                observed: list[float] = []
                for progress_path in (stdout_path, stderr_path):
                    if progress_path.is_file():
                        text = progress_path.read_text(encoding="utf-8", errors="replace")
                        observed.extend(float(match.group(1)) for match in pattern.finditer(text))
                if observed:
                    last_progress = max(last_progress, max(observed))
            if args.phase == "packet" and directory_bytes(args.packet_set_root) > PACKET_CAP_BYTES:
                cap_error = "packet scratch exceeded the executable 100 GiB cumulative cap"
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                break
            time.sleep(args.poll_seconds)
        exit_code = int(process.wait())
    after = cgroup_memory_events()
    ended_at = utc_now()
    if cap_error and exit_code == 0:
        exit_code = 70
    if inheritance_error:
        exit_code = 70
    postcondition_error: str | None = None
    validated_products: list[Path] = []
    if exit_code == 0:
        try:
            if args.phase == "prior" and (
                args.run_root.exists() or args.run_root.is_symlink()
            ):
                raise RuntimeError("prior child created the forbidden training run root")
            validated_products = validate_phase_postconditions(
                recipe,
                phase=args.phase,
                run_root=args.run_root,
                dataset_root=args.dataset_root,
                prior_root=args.prior_root,
                packet_root=args.packet_set_root,
            )
        except Exception as exc:  # noqa: BLE001 - converted to immutable evidence
            postcondition_error = (
                f"phase child exited zero but required outputs did not validate: "
                f"{type(exc).__name__}: {exc}"
            )
            exit_code = 70
    if exit_code != 0:
        recipe["_recipe_path"] = str(args.recipe)
        payload = write_failure_evidence(
            path=args.failure_evidence,
            recipe=recipe,
            argv=argv,
            run_root=args.run_root,
            environment_path=environment_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            started_at=started_at,
            ended_at=ended_at,
            exit_code=exit_code,
            last_progress=last_progress,
            progress_unit=args.progress_unit,
            peak_gpu_mib=peak_gpu,
            maximum_rss_kib=(
                resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
                if resource is not None else 0
            ),
            events_delta=memory_event_delta(before, after),
            extra_error="; ".join(
                error
                for error in (cap_error, inheritance_error, postcondition_error)
                if error
            ) or None,
            failure_stage=("packet_export" if args.phase == "packet" else args.phase),
            model_checkpoint_sha256=(
                str(frozen_method["model_checkpoint_sha256"])
                if frozen_method is not None else None
            ),
            scene_attempt_freeze_sha256=freeze_sha,
        )
        print(json.dumps({"status": payload["status"], "failure_evidence": str(args.failure_evidence)}))
        return exit_code
    if args.failure_evidence.exists():
        raise RuntimeError("successful phase cannot reuse an existing failure-evidence path")
    if args.phase == "packet":
        packet_bytes = directory_bytes(args.packet_set_root)
        if packet_bytes > PACKET_CAP_BYTES:
            raise RuntimeError("packet scratch exceeded 100 GiB after process exit")
    success = {
        "schema": "m3m_gcp_100k_phase_success_v2",
        "status": "PASS",
        "scene": SCENE,
        "method_id": args.method_id,
        "phase": args.phase,
        "recipe_sha256": sha256_file(args.recipe),
        "command_sha256": command_sha256(argv),
        "frozen_budget": recipe.get("budget", {}),
        "environment_manifest_path": str(environment_path),
        "environment_manifest_sha256": sha256_file(environment_path),
        "completion_evidence": {
            "progress_unit": args.progress_unit,
            "last_valid_progress": float(last_progress),
            "required_product_postvalidation_passed": True,
        },
        "products": build_phase_product_rows(
            validated_products, phase=args.phase, method_id=args.method_id
        ),
        "ended_at_utc": utc_now(),
    }
    success["canonical_sha256"] = canonical_sha256(success)
    create_exclusive_json(success_path, success)
    print(json.dumps({"status": "PASS", "phase": args.phase, "method_id": args.method_id}))
    return 0


def release_packet(args: argparse.Namespace) -> int:
    if args.packet_state is None or args.packet_set_root is None:
        raise RuntimeError("packet-release requires --packet-state and --packet-set-root")
    if args.packet_state.is_symlink() or args.packet_set_root.is_symlink():
        raise RuntimeError("packet release refuses symlinked state or packet roots")
    state = json.loads(args.packet_state.read_text(encoding="utf-8"))
    if state.get("scene") != SCENE or state.get("method_id") != args.method_id:
        raise RuntimeError("packet state identity mismatch")
    if Path(os.path.abspath(str(state.get("packet_set_root")))) != args.packet_set_root:
        raise RuntimeError("packet state root mismatch")
    report = json.loads(args.verification_report.read_text(encoding="utf-8"))
    archive = json.loads(args.archive_manifest.read_text(encoding="utf-8"))
    if report.get("status") != "PASS_VERIFIED_FORMAL_V1" or report.get("canonical_sha256") != canonical_sha256(report):
        raise RuntimeError("packet release lacks independent verification PASS")
    if report.get("scene") != SCENE or report.get("method_id") != args.method_id:
        raise RuntimeError("packet release verification identity mismatch")
    archive_errors = validate_archive_manifest(
        args.archive_manifest,
        args.archive_manifest.parent,
        expected_scene_attempt_freeze_sha256=str(
            report.get("scene_attempt_freeze_sha256", "")
        ),
    )
    if archive_errors:
        raise RuntimeError(
            "packet release archive manifest is invalid: " + "; ".join(archive_errors)
        )
    if archive.get("scene") != SCENE or archive.get("method_id") != args.method_id:
        raise RuntimeError("packet release archive identity mismatch")
    if archive.get("scene_attempt_freeze_sha256") != report.get("scene_attempt_freeze_sha256"):
        raise RuntimeError("packet release archive/freeze binding mismatch")
    if args.packet_set_root.exists():
        shutil.rmtree(args.packet_set_root)
    args.packet_state.unlink()
    print(json.dumps({"status": "PASS_PACKET_RELEASED", "method_id": args.method_id}))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--recipe-manifest", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--method-id", required=True)
    parser.add_argument("--phase", choices=("prior", "training", "packet", "packet-release"), required=True)
    parser.add_argument("--capacity-root", type=Path, default=Path("/root/autodl-tmp"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--prior-root", type=Path, required=True)
    parser.add_argument("--failure-evidence", type=Path, required=True)
    parser.add_argument("--packet-state", type=Path)
    parser.add_argument("--packet-set-root", type=Path)
    parser.add_argument("--scene-attempt-freeze", type=Path)
    parser.add_argument("--verification-report", type=Path)
    parser.add_argument("--archive-manifest", type=Path)
    parser.add_argument("--progress-regex")
    parser.add_argument("--progress-unit", default="iterations")
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    return args


def main() -> int:
    args = parse_args()
    for field in ("repo", "activation", "plan", "recipe_manifest", "recipe", "capacity_root", "run_root", "dataset_root", "source_root", "prior_root", "failure_evidence"):
        setattr(args, field, getattr(args, field).resolve())
    if args.packet_state is not None:
        args.packet_state = Path(os.path.abspath(args.packet_state))
    if args.packet_set_root is not None:
        args.packet_set_root = Path(os.path.abspath(args.packet_set_root))
    if args.scene_attempt_freeze is not None:
        args.scene_attempt_freeze = args.scene_attempt_freeze.resolve()
    plan, recipe = validate_activation_and_recipe(
        repo=args.repo,
        activation_path=args.activation,
        plan_path=args.plan,
        recipe_manifest_path=args.recipe_manifest,
        recipe_path=args.recipe,
        method_id=args.method_id,
        phase=args.phase,
    )
    args.execution_plan = plan
    expected_run_root = Path(str(recipe.get("authorized_run_root", ""))).resolve()
    if args.run_root != expected_run_root:
        raise RuntimeError("run root differs from the frozen method recipe")
    if args.phase in {"packet", "packet-release"}:
        if args.packet_state is None or args.packet_set_root is None:
            raise RuntimeError("packet lifecycle requires frozen state and packet roots")
        expected_packet_state = Path(os.path.abspath(
            str(recipe.get("authorized_packet_state", ""))
        ))
        expected_packet_root = Path(os.path.abspath(
            str(recipe.get("authorized_packet_set_root", ""))
        ))
        if args.packet_state != expected_packet_state:
            raise RuntimeError("packet-state path differs from the frozen method recipe")
        if args.packet_set_root != expected_packet_root:
            raise RuntimeError("packet-set root differs from the frozen method recipe")
    if args.phase != "packet-release":
        expected_failure = (
            Path(str(recipe.get("authorized_evidence_root", "")))
            / args.phase
            / "failure.json"
        ).resolve()
        if args.failure_evidence != expected_failure:
            raise RuntimeError("failure-evidence path differs from the frozen method recipe")
    if args.phase == "packet-release":
        if args.verification_report is None or args.archive_manifest is None:
            raise RuntimeError("packet-release requires verification and archive manifests")
        return release_packet(args)
    if args.method_id == "3dgs_original":
        if args.phase != "packet":
            raise RuntimeError("frozen reused 3DGS 100K model permits packet export only")
        reuse = recipe.get("reuse_model_binding", {})
        expected_run = Path(str(reuse.get("run_root", ""))).resolve()
        if args.run_root != expected_run or reuse.get("retrain_allowed") is not False:
            raise RuntimeError("reused 3DGS run-root/retrain binding mismatch")
        point_cloud = args.run_root / str(reuse.get("point_cloud_relative_path", ""))
        if (
            not point_cloud.is_file()
            or point_cloud.stat().st_size != int(reuse.get("point_cloud_bytes", -1))
            or sha256_file(point_cloud) != reuse.get("point_cloud_sha256")
        ):
            raise RuntimeError("reused 3DGS point-cloud identity mismatch")
    if args.phase == "packet":
        validate_evaluation_camera_root(recipe)
    templates = recipe.get("phase_commands", {})
    template = templates.get(args.phase)
    if not isinstance(template, list) or not template or any(not isinstance(item, str) for item in template):
        raise RuntimeError(f"recipe has no exact {args.phase} command")
    replacements = {
        "repo": str(args.repo),
        "dataset_root": str(args.dataset_root),
        "source_root": str(args.source_root),
        "prior_root": str(args.prior_root),
        "run_root": str(args.run_root),
        "packet_set_root": str(args.packet_set_root) if args.packet_set_root else "",
    }
    args.replacements = replacements
    if args.phase == "training":
        validate_prior_phase_success(
            recipe=recipe,
            recipe_path=args.recipe,
            run_root=args.run_root,
            dataset_root=args.dataset_root,
            prior_root=args.prior_root,
            replacements=replacements,
        )
    args.gpu_prelaunch = require_idle_gpu()
    validate_source_binding(recipe, args.source_root, args.phase)
    validate_phase_roots(
        recipe,
        phase=args.phase,
        dataset_root=args.dataset_root,
        prior_root=args.prior_root,
    )
    validate_external_files(recipe, args.phase)
    if args.phase in recipe.get("input_validation_phases", []):
        validate_frozen_training_images(recipe, args.dataset_root)
        validate_prepared_method_input(recipe, args.dataset_root)
    monitor = recipe.get("progress_monitors", {}).get(
        args.phase, recipe.get("progress_monitor", {})
    )
    frozen_regex = monitor.get("regex")
    frozen_unit = monitor.get("unit")
    if args.progress_regex is not None and args.progress_regex != frozen_regex:
        raise RuntimeError("CLI progress regex differs from the frozen recipe")
    if args.progress_unit != "iterations" and args.progress_unit != frozen_unit:
        raise RuntimeError("CLI progress unit differs from the frozen recipe")
    args.progress_regex = frozen_regex
    args.progress_unit = str(frozen_unit)
    expected_argv = [item.format(**replacements) for item in template]
    if args.command and list(args.command) != expected_argv:
        raise RuntimeError("CLI command differs from the exact frozen recipe command")
    return run_phase(args, recipe, expected_argv)


if __name__ == "__main__":
    raise SystemExit(main())
