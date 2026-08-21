#!/usr/bin/env python3
"""Execute one frozen 100K phase behind review, disk and packet-lifecycle gates."""

from __future__ import annotations

import argparse
import hashlib
import json
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


def git_value(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, encoding="utf-8"
    ).strip()


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


def validate_activation_and_recipe(
    *,
    repo: Path,
    activation_path: Path,
    plan_path: Path,
    recipe_manifest_path: Path,
    recipe_path: Path,
    method_id: str,
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
    if plan.get("schema") != "m3m_gcp_native_quarter_100k_ten_method_execution_plan_v1":
        raise RuntimeError("execution plan schema mismatch")
    if plan.get("scene") != SCENE or plan.get("seed") != 0:
        raise RuntimeError("execution plan is not frozen 100K seed0")
    if plan.get("status") != "REVIEW_CANDIDATE_NOT_EXECUTION_AUTHORIZED" or plan.get("execution_authorized") is not False:
        raise RuntimeError("reviewed execution-plan candidate identity changed")
    if plan.get("canonical_sha256") != canonical_sha256(plan):
        raise RuntimeError("execution plan canonical SHA mismatch")
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
    if recipe_manifest.get("schema") != "m3m_gcp_native_quarter_100k_recipe_manifest_v1":
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
    if recipe.get("schema") != "m3m_gcp_native_quarter_100k_execution_recipe_v1":
        raise RuntimeError("recipe schema mismatch")
    if recipe.get("method_id") != method_id or recipe.get("scene") != SCENE or recipe.get("seed") != 0:
        raise RuntimeError("recipe identity mismatch")
    if recipe.get("status") != "REVIEW_CANDIDATE_NOT_EXECUTION_AUTHORIZED":
        raise RuntimeError("reviewed recipe candidate identity changed")
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
    if git_value(source_root, "status", "--porcelain=v1", "--untracked-files=no") != binding.get("required_status", ""):
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


def validate_frozen_training_images(recipe: dict[str, Any], dataset_root: Path) -> None:
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
    if Path(str(binding.get("dataset_root", ""))).resolve() != dataset_root.resolve():
        raise RuntimeError("prepared per-method input dataset root mismatch")
    sparse = dataset_root / "sparse" / "0"
    for name, expected_sha in binding.get("sparse_sha256", {}).items():
        path = sparse / name
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise RuntimeError(f"prepared per-method input sparse SHA mismatch: {name}")


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
    *, manifest_path: Path, method_id: str, run_root: Path
) -> None:
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


def validate_packet_freeze_binding(
    *, args: argparse.Namespace, recipe: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    freeze_path = args.scene_attempt_freeze
    if freeze_path is None or not freeze_path.is_file():
        raise RuntimeError("packet phase requires an immutable scene-attempt freeze")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    errors, methods = validate_scene_attempt_freeze(
        freeze, freeze_path=freeze_path, expected_scene=SCENE
    )
    if errors or methods is None:
        raise RuntimeError("packet scene-attempt freeze is invalid: " + "; ".join(errors))
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
    validate_model_identity_bundle(
        manifest_path=Path(str(row.get("model_checkpoint_path", ""))),
        method_id=args.method_id,
        run_root=args.run_root,
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
    frozen_method: dict[str, Any] | None = None
    freeze_sha: str | None = None
    if args.phase == "packet":
        if args.packet_set_root is None or args.packet_state is None:
            raise RuntimeError("packet phase requires --packet-set-root and --packet-state")
        if args.packet_set_root.exists():
            raise RuntimeError("packet set root already exists")
        frozen_method, freeze_sha = validate_packet_freeze_binding(
            args=args, recipe=recipe
        )
        acquire_packet_state(args.packet_state, args.method_id, args.packet_set_root)
    args.run_root.mkdir(parents=True, exist_ok=True)
    materialize_phase_files(
        recipe, phase=args.phase, run_root=args.run_root, replacements=args.replacements
    )
    evidence_dir.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    environment = {
        "schema": "m3m_gcp_100k_execution_environment_v1",
        "scene": SCENE,
        "method_id": args.method_id,
        "phase": args.phase,
        "argv": argv,
        "python": sys.version,
        "platform": platform.platform(),
        "gpu_prelaunch": getattr(args, "gpu_prelaunch", {}),
        "started_at_utc": started_at,
    }
    environment["canonical_sha256"] = canonical_sha256(environment)
    environment_path.write_text(json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    before = cgroup_memory_events()
    last_progress = 0.0
    peak_gpu = 0.0
    cap_error: str | None = None
    pattern = re.compile(args.progress_regex) if args.progress_regex else None
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
        process = subprocess.Popen(argv, stdout=stdout_handle, stderr=stderr_handle, text=True)
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
            extra_error=cap_error,
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
        "schema": "m3m_gcp_100k_phase_success_v1",
        "status": "PASS",
        "scene": SCENE,
        "method_id": args.method_id,
        "phase": args.phase,
        "recipe_sha256": sha256_file(args.recipe),
        "command_sha256": command_sha256(argv),
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
    _, recipe = validate_activation_and_recipe(
        repo=args.repo,
        activation_path=args.activation,
        plan_path=args.plan,
        recipe_manifest_path=args.recipe_manifest,
        recipe_path=args.recipe,
        method_id=args.method_id,
    )
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
    args.gpu_prelaunch = require_idle_gpu()
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
