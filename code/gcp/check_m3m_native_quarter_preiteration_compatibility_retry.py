#!/usr/bin/env python3
"""Authorize one same-run retry after a proven pre-iteration host-memory kill."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any


PROTOCOL_ID = "m3m_gcp_native_quarter_geometry_v2"
POLICY_ID = "glibc_malloc_trim_threshold_zero_v1"
FROZEN_RUN_ROOT = PurePosixPath(
    "/root/autodl-tmp/runs/m3m-gcp-native-quarter/3dgs-original/"
    "gcp_100000_20260610/seed0-30k-20260810T175634Z"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: dict[str, Any]) -> str:
    clean = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    encoded = json.dumps(clean, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def check_retry(
    registry: dict[str, Any],
    repo_root: Path,
    input_release_root: Path,
    run_root: Path,
) -> dict[str, Any]:
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(registry.get("protocol_id") == PROTOCOL_ID, "registry protocol mismatch")
    require(registry.get("global_training_allowed") is False, "global training must remain locked")
    require(registry.get("per_method_training_allowed_methods") == [], "3K allowlist must remain empty")
    scope = registry.get("preliminary_evidence_scope", {})
    require(scope.get("six_scene_matrix_status") == "LOCKED", "six-scene matrix must remain locked")
    require(scope.get("multi_seed_status") == "NOT_AUTHORIZED", "multi-seed execution must remain locked")

    initial = registry.get("explicit_scene_run_authorization", {})
    require(initial.get("status") == "ATTEMPTED_PREITERATION_HOST_MEMORY_KILL_RELOCKED", "initial authorization is not relocked")
    require(initial.get("single_fresh_run_allowed") is False, "initial fresh-run authorization remains active")
    require(initial.get("formal_iteration_reached") == 0, "initial attempt reached a formal iteration")
    require(initial.get("checkpoint_file_count") == 0, "initial attempt created a checkpoint")

    entry = registry.get("explicit_scene_compatibility_retry_authorization", {})
    require(entry.get("status") == "AUTHORIZED_NOT_STARTED", "compatibility retry is not active")
    require(entry.get("method_id") == "3dgs_original", "retry method mismatch")
    require(entry.get("scene") == "gcp_100000_20260610", "retry scene mismatch")
    require(entry.get("seed") == 0 and entry.get("iterations") == 30000, "retry recipe mismatch")
    require(entry.get("compatibility_policy") == POLICY_ID, "compatibility policy mismatch")
    require(entry.get("environment") == {"MALLOC_TRIM_THRESHOLD_": "0"}, "allocator environment mismatch")
    require(entry.get("single_preiteration_retry_allowed") is True, "single retry is not allowed")
    require(entry.get("new_run_root_allowed") is False, "new run root was allowed")
    require(entry.get("resume_allowed") is False, "resume was allowed")
    require(entry.get("other_methods_scenes_seeds_retries_allowed") is False, "retry scope is not isolated")
    require(entry.get("run_root") == run_root.as_posix(), "retry run root mismatch")

    authorization: dict[str, Any] = {}
    auth_path: Path | None = None
    relative = entry.get("authorization")
    if isinstance(relative, str) and relative:
        auth_path = (repo_root / relative).resolve()
        require(auth_path.is_relative_to(repo_root.resolve()), "retry authorization escapes repository")
        require(auth_path.is_file(), "retry authorization is missing")
        if auth_path.is_file():
            require(file_sha256(auth_path) == entry.get("authorization_sha256"), "retry authorization SHA mismatch")
            authorization = json.loads(auth_path.read_text(encoding="utf-8"))
    else:
        errors.append("retry authorization path is missing")

    require(authorization.get("protocol_id") == PROTOCOL_ID, "retry protocol mismatch")
    require(authorization.get("authorization_id") == entry.get("authorization_id"), "retry authorization ID mismatch")
    require(authorization.get("status") == "AUTHORIZED_NOT_STARTED", "retry file is not launchable")
    method = authorization.get("method", {})
    require(method.get("method_id") == "3dgs_original", "retry file method mismatch")
    require(method.get("repository_commit") == "2eee0e26d2d5fd00ec462df47752223952f6bf4e", "source commit mismatch")
    require(method.get("repository_tree") == "5eee127dfc0942bf83d9fdd72328e03ec0cbf6c4", "source tree mismatch")
    policy = authorization.get("compatibility_policy", {})
    require(policy.get("policy_id") == POLICY_ID, "retry file policy mismatch")
    require(policy.get("environment") == {"MALLOC_TRIM_THRESHOLD_": "0"}, "retry file environment mismatch")
    for key in ("training_tensor_semantics_changed", "rng_semantics_changed", "camera_order_changed", "image_pixels_changed"):
        require(policy.get(key) is False, f"compatibility policy changes semantics: {key}")
    policy_contract = repo_root / str(policy.get("source_contract", ""))
    require(policy_contract.is_file(), "compatibility policy contract is missing")
    if policy_contract.is_file():
        require(file_sha256(policy_contract) == policy.get("source_contract_sha256"), "compatibility policy contract SHA mismatch")

    auth_input = authorization.get("input", {})
    manifest_path = (input_release_root / str(auth_input.get("formal_input_manifest", ""))).resolve()
    require(manifest_path.is_relative_to(input_release_root.resolve()), "formal input manifest escapes release root")
    require(manifest_path.is_file(), "formal input manifest is missing")
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        require(file_sha256(manifest_path) == auth_input.get("formal_input_manifest_file_sha256"), "formal input manifest file SHA mismatch")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(canonical_sha256(manifest) == auth_input.get("formal_input_manifest_canonical_sha256"), "formal input canonical SHA mismatch")
    require(manifest.get("scene") == "gcp_100000_20260610", "formal input scene mismatch")
    require(manifest.get("train_view_count") == 2196 and manifest.get("test_view_count") == 314, "formal input view counts mismatch")

    pure_run = PurePosixPath(run_root.as_posix())
    require(pure_run == FROZEN_RUN_ROOT, "run root is not the frozen failed attempt")
    require(run_root.is_dir(), "failed run root is missing")
    require((run_root / "state.txt").is_file() and (run_root / "state.txt").read_text().strip() == "TRAINING_FAILED", "failed run state mismatch")
    require((run_root / "retry1_probe_exit_code.txt").is_file() and (run_root / "retry1_probe_exit_code.txt").read_text().strip() == "137", "failed child exit code mismatch")

    initial_attempt = authorization.get("initial_attempt", {})
    summary_path = run_root / str(initial_attempt.get("resource_summary", ""))
    require(summary_path.is_file(), "failed resource summary is missing")
    summary: dict[str, Any] = {}
    if summary_path.is_file():
        require(file_sha256(summary_path) == initial_attempt.get("resource_summary_sha256"), "failed resource summary SHA mismatch")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    require(summary.get("outer_probe_exit_code") == 137, "failed resource exit code mismatch")
    require(summary.get("status") == "METHOD_FAILURE", "failed resource status mismatch")
    require(summary.get("max_gpu_utilization_percent") == 1.0, "failure did not occur during camera loading")
    require(summary.get("peak_gpu_memory_mib") == 39523.0, "failed GPU peak mismatch")
    require(summary.get("process_maximum_rss_kib") == 110096564, "failed host RSS mismatch")
    require(summary.get("memory_events_delta", {}).get("oom") == 0, "failure recorded cgroup OOM")
    require(summary.get("memory_events_delta", {}).get("oom_kill") == 0, "failure recorded cgroup OOM kill")
    gnu_time = summary_path.parent / "gnu_time.txt"
    require(gnu_time.is_file() and file_sha256(gnu_time) == initial_attempt.get("gnu_time_sha256"), "failed GNU-time evidence mismatch")

    model_root = run_root / "model"
    require(model_root.is_dir(), "failed model root is missing")
    expected_model_files = set(initial_attempt.get("model_files_to_preserve", []))
    actual_model_files = {path.relative_to(model_root).as_posix() for path in model_root.rglob("*") if path.is_file()}
    require(actual_model_files == expected_model_files, "failed model root contains unexpected files")
    require(not any((model_root / "point_cloud").rglob("*") if (model_root / "point_cloud").exists() else []), "failed model has point-cloud output")
    retry = authorization.get("retry", {})
    require(retry.get("single_retry_allowed") is True and retry.get("resume_allowed") is False, "retry execution flags mismatch")
    require(not (run_root / str(retry.get("failed_model_archive", ""))).exists(), "failed-model archive target already exists")
    require(not (run_root / str(retry.get("resource_probe_root", ""))).exists(), "retry resource probe target already exists")
    locks = authorization.get("scope_locks", {})
    require(all(locks.get(key) is False for key in locks), "retry scope lock is open")

    return {
        "schema": "m3m_gcp_native_quarter_preiteration_compatibility_retry_gate_v1",
        "protocol_id": PROTOCOL_ID,
        "authorization_id": entry.get("authorization_id"),
        "method_id": entry.get("method_id"),
        "scene": entry.get("scene"),
        "seed": entry.get("seed"),
        "iterations": entry.get("iterations"),
        "run_root": run_root.as_posix(),
        "compatibility_policy": POLICY_ID,
        "environment": {"MALLOC_TRIM_THRESHOLD_": "0"},
        "passed": not errors,
        "status": "AUTHORIZED" if not errors else "DENIED",
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--registry", type=Path, default=default_root / "configs" / "m3m_gcp_native_quarter_method_registry_v2.json")
    parser.add_argument("--repo_root", type=Path, default=default_root)
    parser.add_argument("--input_release_root", required=True, type=Path)
    parser.add_argument("--run_root", required=True, type=Path)
    args = parser.parse_args()
    result = check_retry(
        json.loads(args.registry.read_text(encoding="utf-8")),
        args.repo_root.resolve(),
        args.input_release_root.resolve(),
        args.run_root.resolve(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
