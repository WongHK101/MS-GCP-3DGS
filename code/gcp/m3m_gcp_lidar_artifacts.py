#!/usr/bin/env python3
"""Shared fail-closed validators for frozen M3M-GCP LiDAR artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


PROTOCOL_ID = "m3m_gcp_lidar_rendered_surface_v1"
METHOD_IDS = (
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
)
METHOD_CLASSES = {
    method_id: (
        "rgb_colmap_external_geometry_prior"
        if method_id in {"citygaussian_v2", "citygs_x", "metrogs"}
        else "rgb_colmap_only"
    )
    for method_id in METHOD_IDS
}
FAILURE_STATUSES = {"OOM_UNRANKED", "FAILED_UNRANKED", "INCOMPLETE_UNRANKED"}
OOM_SIGNALS = {"CUDA_OUT_OF_MEMORY", "CGROUP_OOM_KILL", "HOST_OOM"}
PRE_FREEZE_FAILURE_STAGES = {"prior", "training"}
POST_FREEZE_FAILURE_STAGES = {
    "packet_export",
    "formal_evaluation",
    "verification",
    "archive",
}
FAILURE_FIELDS = {
    "schema",
    "protocol_id",
    "scene",
    "method_id",
    "input_class",
    "seed",
    "status",
    "failure_stage",
    "run_root",
    "model_checkpoint_sha256",
    "scene_attempt_freeze_sha256",
    "command_argv",
    "command_sha256",
    "environment_manifest_path",
    "environment_manifest_sha256",
    "recipe_sha256",
    "renderer_adapter_sha256",
    "started_at_utc",
    "ended_at_utc",
    "exit_code",
    "last_valid_progress",
    "peak_gpu_memory_mib",
    "process_maximum_rss_kib",
    "cgroup_memory_events_delta",
    "oom_signal",
    "stdout_path",
    "stdout_sha256",
    "stderr_path",
    "stderr_sha256",
    "errors",
    "canonical_sha256",
}
FREEZE_FIELDS = {
    "schema",
    "protocol_id",
    "scene",
    "methods_manifest_path",
    "methods_manifest_file_sha256",
    "methods_manifest_canonical_sha256",
    "frozen_method_ids",
    "created_at_utc",
    "canonical_sha256",
}
METHOD_ROW_FIELDS = {
    "method_id",
    "method_name",
    "input_class",
    "attempt_status",
    "run_root",
    "model_checkpoint_path",
    "model_checkpoint_sha256",
    "recipe_path",
    "recipe_sha256",
    "renderer_adapter_path",
    "renderer_adapter_sha256",
    "failure_evidence_path",
    "failure_evidence_sha256",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: Any, *, self_field: str = "canonical_sha256") -> str:
    clean = dict(payload)
    clean.pop(self_field, None)
    raw = json.dumps(clean, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def command_sha256(argv: list[str]) -> str:
    raw = json.dumps(argv, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _validate_bound_file(path_value: Any, sha_value: Any, label: str) -> list[str]:
    path = Path(str(path_value))
    if not path.is_absolute():
        return [f"{label} path is not absolute"]
    if not path.is_file():
        return [f"{label} file missing"]
    if sha256_file(path) != sha_value:
        return [f"{label} SHA mismatch"]
    return []


def validate_failure_evidence(
    payload: dict[str, Any],
    *,
    expected_scene: str | None = None,
    expected_method_id: str | None = None,
    expected_status: str | None = None,
) -> list[str]:
    """Validate immutable, independently auditable OOM/failure evidence."""
    errors: list[str] = []
    if set(payload) != FAILURE_FIELDS:
        errors.append("failure evidence field inventory mismatch")
    if payload.get("schema") != "m3m_gcp_lidar_failure_evidence_v1":
        errors.append("failure evidence schema mismatch")
    if payload.get("protocol_id") != PROTOCOL_ID:
        errors.append("failure evidence protocol mismatch")
    if payload.get("canonical_sha256") != canonical_sha256(payload):
        errors.append("failure evidence canonical SHA mismatch")
    if expected_scene is not None and payload.get("scene") != expected_scene:
        errors.append("failure evidence scene mismatch")
    if expected_method_id is not None and payload.get("method_id") != expected_method_id:
        errors.append("failure evidence method mismatch")
    status = str(payload.get("status", ""))
    if status not in FAILURE_STATUSES:
        errors.append("failure evidence status mismatch")
    if expected_status is not None and status != expected_status:
        errors.append("failure evidence status differs from attempt")
    if payload.get("method_id") not in METHOD_IDS:
        errors.append("failure evidence method is outside frozen pool")
    elif payload.get("input_class") != METHOD_CLASSES[payload["method_id"]]:
        errors.append("failure evidence input class mismatch")
    if payload.get("seed") != 0:
        errors.append("failure evidence seed is not zero")
    failure_stage = str(payload.get("failure_stage", ""))
    model_sha = payload.get("model_checkpoint_sha256")
    freeze_sha = payload.get("scene_attempt_freeze_sha256")
    if status in {"OOM_UNRANKED", "FAILED_UNRANKED"}:
        if failure_stage not in PRE_FREEZE_FAILURE_STAGES:
            errors.append("pre-freeze failure stage mismatch")
        if model_sha is not None or freeze_sha is not None:
            errors.append("pre-freeze failure carries model/freeze binding")
    elif status == "INCOMPLETE_UNRANKED":
        if failure_stage not in POST_FREEZE_FAILURE_STAGES:
            errors.append("post-freeze failure stage mismatch")
        for field, value in (
            ("model_checkpoint_sha256", model_sha),
            ("scene_attempt_freeze_sha256", freeze_sha),
        ):
            if not isinstance(value, str) or len(value) != 64:
                errors.append(f"post-freeze failure {field} is invalid")
    if not isinstance(payload.get("run_root"), str) or not Path(payload["run_root"]).is_absolute():
        errors.append("failure evidence run root is not absolute")
    argv = payload.get("command_argv")
    if not isinstance(argv, list) or not argv or any(not isinstance(item, str) or not item for item in argv):
        errors.append("failure evidence command argv is invalid")
    elif payload.get("command_sha256") != command_sha256(argv):
        errors.append("failure evidence command SHA mismatch")
    for field in ("recipe_sha256", "renderer_adapter_sha256"):
        value = payload.get(field)
        if not isinstance(value, str) or len(value) != 64:
            errors.append(f"failure evidence {field} is invalid")
    for prefix in ("environment_manifest", "stdout", "stderr"):
        errors.extend(
            _validate_bound_file(
                payload.get(f"{prefix}_path"), payload.get(f"{prefix}_sha256"), prefix
            )
        )
    if not isinstance(payload.get("started_at_utc"), str) or not payload.get("started_at_utc"):
        errors.append("failure evidence start timestamp missing")
    if not isinstance(payload.get("ended_at_utc"), str) or not payload.get("ended_at_utc"):
        errors.append("failure evidence end timestamp missing")
    if not isinstance(payload.get("exit_code"), int):
        errors.append("failure evidence exit code is invalid")
    elif status in {"OOM_UNRANKED", "FAILED_UNRANKED"} and payload["exit_code"] == 0:
        errors.append("OOM/failed evidence requires a non-zero exit code")
    progress = payload.get("last_valid_progress")
    if (
        not isinstance(progress, dict)
        or set(progress) != {"unit", "value"}
        or not isinstance(progress.get("unit"), str)
        or not isinstance(progress.get("value"), (int, float))
        or not math.isfinite(float(progress["value"]))
        or float(progress["value"]) < 0
    ):
        errors.append("failure evidence last progress is invalid")
    for field in ("peak_gpu_memory_mib", "process_maximum_rss_kib"):
        value = payload.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0:
            errors.append(f"failure evidence {field} is invalid")
    events = payload.get("cgroup_memory_events_delta")
    if (
        not isinstance(events, dict)
        or set(events) != {"oom", "oom_kill", "max"}
        or any(not isinstance(events.get(key), int) or events[key] < 0 for key in ("oom", "oom_kill", "max"))
    ):
        errors.append("failure evidence cgroup counters are invalid")
        events = {"oom": 0, "oom_kill": 0, "max": 0}
    oom_signal = payload.get("oom_signal")
    if status == "OOM_UNRANKED" or (
        status == "INCOMPLETE_UNRANKED" and oom_signal is not None
    ):
        if oom_signal not in OOM_SIGNALS:
            errors.append("OOM failure lacks a recognized OOM signal")
        stderr_text = ""
        stderr_path = Path(str(payload.get("stderr_path", "")))
        if stderr_path.is_file():
            stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace").lower()
        marker = any(token in stderr_text for token in ("out of memory", "cuda oom", "cuda error: out of memory"))
        if oom_signal == "CUDA_OUT_OF_MEMORY" and not marker:
            errors.append("CUDA OOM signal lacks a matching stderr marker")
        elif oom_signal == "CGROUP_OOM_KILL" and int(events.get("oom_kill", 0)) == 0:
            errors.append("cgroup OOM-kill signal lacks an oom_kill counter delta")
        elif oom_signal == "HOST_OOM" and int(events.get("oom", 0)) == 0 and int(events.get("oom_kill", 0)) == 0:
            errors.append("host OOM signal lacks a cgroup OOM counter delta")
    elif oom_signal is not None:
        errors.append("non-OOM failure carries an OOM signal")
    if not isinstance(payload.get("errors"), list) or not payload.get("errors") or any(
        not isinstance(item, str) or not item for item in payload.get("errors", [])
    ):
        errors.append("failure evidence errors list is invalid")
    return errors


def validate_failure_evidence_file(
    path: Path,
    *,
    expected_sha256: str,
    expected_scene: str,
    expected_method_id: str,
    expected_status: str,
) -> list[str]:
    if not path.is_file():
        return ["failure evidence file missing"]
    errors: list[str] = []
    if sha256_file(path) != expected_sha256:
        errors.append("failure evidence file SHA mismatch")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return errors + [f"failure evidence JSON unreadable: {exc}"]
    return errors + validate_failure_evidence(
        payload,
        expected_scene=expected_scene,
        expected_method_id=expected_method_id,
        expected_status=expected_status,
    )


def validate_scene_attempt_freeze(
    payload: dict[str, Any],
    *,
    freeze_path: Path,
    expected_scene: str,
) -> tuple[list[str], dict[str, Any] | None]:
    """Validate the one immutable ten-method attempt snapshot for a scene."""
    errors: list[str] = []
    if set(payload) != FREEZE_FIELDS:
        errors.append("scene attempt freeze field inventory mismatch")
    if payload.get("schema") != "m3m_gcp_lidar_scene_attempt_freeze_v1":
        errors.append("scene attempt freeze schema mismatch")
    if payload.get("protocol_id") != PROTOCOL_ID:
        errors.append("scene attempt freeze protocol mismatch")
    if payload.get("scene") != expected_scene:
        errors.append("scene attempt freeze scene mismatch")
    if payload.get("frozen_method_ids") != list(METHOD_IDS):
        errors.append("scene attempt freeze method order mismatch")
    if payload.get("canonical_sha256") != canonical_sha256(payload):
        errors.append("scene attempt freeze canonical SHA mismatch")
    methods_path = Path(str(payload.get("methods_manifest_path", "")))
    if not methods_path.is_absolute():
        errors.append("scene attempt freeze methods path is not absolute")
        return errors, None
    if not methods_path.is_file():
        errors.append("scene attempt freeze methods file missing")
        return errors, None
    if sha256_file(methods_path) != payload.get("methods_manifest_file_sha256"):
        errors.append("scene attempt freeze methods file SHA mismatch")
    try:
        methods = json.loads(methods_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return errors + [f"scene attempt methods JSON unreadable: {exc}"], None
    if methods.get("canonical_sha256") != canonical_sha256(methods):
        errors.append("scene attempt methods canonical SHA mismatch")
    if methods.get("canonical_sha256") != payload.get("methods_manifest_canonical_sha256"):
        errors.append("scene attempt freeze methods canonical binding mismatch")
    if set(methods) != {"schema", "protocol_id", "scene", "methods", "canonical_sha256"}:
        errors.append("scene attempt methods field inventory mismatch")
    if methods.get("schema") != "m3m_gcp_lidar_formal_methods_v1":
        errors.append("scene attempt methods schema mismatch")
    if methods.get("protocol_id") != PROTOCOL_ID:
        errors.append("scene attempt methods protocol mismatch")
    if methods.get("scene") != expected_scene:
        errors.append("scene attempt methods scene mismatch")
    rows = methods.get("methods", [])
    if [row.get("method_id") for row in rows] != list(METHOD_IDS):
        errors.append("scene attempt methods pool mismatch")
    for row in rows:
        method_id = str(row.get("method_id", ""))
        if set(row) != METHOD_ROW_FIELDS:
            errors.append(f"{method_id}: frozen method row field inventory mismatch")
            continue
        if row.get("input_class") != METHOD_CLASSES.get(method_id):
            errors.append(f"{method_id}: frozen method input class mismatch")
        if not isinstance(row.get("method_name"), str) or not row.get("method_name"):
            errors.append(f"{method_id}: frozen method name missing")
        if not Path(str(row.get("run_root", ""))).is_absolute():
            errors.append(f"{method_id}: frozen run root is not absolute")
        for path_field, sha_field, label in (
            ("recipe_path", "recipe_sha256", "recipe"),
            ("renderer_adapter_path", "renderer_adapter_sha256", "renderer adapter"),
        ):
            errors.extend(
                f"{method_id}: {item}"
                for item in _validate_bound_file(
                    row.get(path_field), row.get(sha_field), label
                )
            )
        status = row.get("attempt_status")
        if status == "READY_FOR_EVALUATION":
            errors.extend(
                f"{method_id}: {item}"
                for item in _validate_bound_file(
                    row.get("model_checkpoint_path"),
                    row.get("model_checkpoint_sha256"),
                    "model checkpoint",
                )
            )
            if row.get("failure_evidence_path") is not None or row.get(
                "failure_evidence_sha256"
            ) is not None:
                errors.append(f"{method_id}: ready row carries failure evidence")
        elif status in {"OOM_UNRANKED", "FAILED_UNRANKED"}:
            if row.get("model_checkpoint_path") is not None or row.get(
                "model_checkpoint_sha256"
            ) is not None:
                errors.append(f"{method_id}: failed row carries model checkpoint")
            errors.extend(
                f"{method_id}: {item}"
                for item in validate_failure_evidence_file(
                    Path(str(row.get("failure_evidence_path", ""))),
                    expected_sha256=str(row.get("failure_evidence_sha256", "")),
                    expected_scene=expected_scene,
                    expected_method_id=method_id,
                    expected_status=str(status),
                )
            )
        else:
            errors.append(f"{method_id}: frozen attempt status mismatch")
    if not freeze_path.is_absolute():
        errors.append("scene attempt freeze path is not absolute")
    return errors, methods
