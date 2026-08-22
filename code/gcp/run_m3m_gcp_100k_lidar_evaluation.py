#!/usr/bin/env python3
"""Run one activation-bound 100K LiDAR evaluator/verifier pair without retry."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from m3m_gcp_100k_raw_packet_state import validate_active_raw_packet_state
from m3m_gcp_100k_three_track_runtime import validate_addendum_runtime
from m3m_gcp_lidar_artifacts import canonical_sha256, command_sha256, sha256_file
from run_m3m_gcp_100k_guarded import cgroup_memory_events, memory_event_delta


SCENE = "gcp_100000_20260610"


def require_json(path: Path, expected_sha: str | None = None) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    if expected_sha is not None and sha256_file(path) != expected_sha:
        raise RuntimeError(f"file SHA mismatch: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(
            descriptor,
            (
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n"
            ).encode("utf-8"),
        )
    finally:
        os.close(descriptor)


def build_failure(
    *,
    method_id: str,
    activation_path: Path,
    candidate: dict[str, Any],
    dispatch_path: Path,
    authorization_path: Path,
    global_state_path: Path,
    stage: str,
    exit_code: int,
    evaluator_command: list[str],
    verifier_command: list[str],
    logs: list[Path],
    memory_delta: dict[str, int],
    error: str,
) -> dict[str, Any]:
    stderr = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in logs
        if path.name.endswith("stderr.log") and path.is_file()
    ).lower()
    oom = (
        ("out of memory" in stderr and ("cuda" in stderr or "memory" in stderr))
        or memory_delta.get("oom", 0) > 0
        or memory_delta.get("oom_kill", 0) > 0
    )
    payload: dict[str, Any] = {
        "schema": "m3m_gcp_100k_lidar_evaluation_failure_v1",
        "status": "OOM_UNRANKED" if oom else "FAILED_UNRANKED",
        "ranking_status": "INCOMPLETE_UNRANKED",
        "scene": SCENE,
        "method_id": method_id,
        "failure_stage": stage,
        "three_track_activation_sha256": sha256_file(activation_path),
        "scene_attempt_freeze_sha256": candidate["scene_attempt_freeze"]["sha256"],
        "methods_manifest_sha256": candidate["methods_manifest"]["sha256"],
        "lidar_dispatch_receipt_sha256": sha256_file(dispatch_path),
        "scene_authorization_sha256": sha256_file(authorization_path),
        "global_raw_packet_state_sha256": sha256_file(global_state_path),
        "evaluator_command_sha256": command_sha256(evaluator_command),
        "verifier_command_sha256": command_sha256(verifier_command),
        "logs": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in logs
            if path.is_file()
        ],
        "exit_code": int(exit_code),
        "cgroup_memory_events_delta": memory_delta,
        "error": error,
        "retry_forbidden_after_evaluator_or_verifier_child_start": True,
    }
    payload["canonical_sha256"] = canonical_sha256(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument(
        "--method-id", choices=("3dgs_original", "citygs_x", "metrogs"), required=True
    )
    parser.add_argument("--lidar-dispatch-receipt", type=Path, required=True)
    parser.add_argument("--scene-authorization", type=Path, required=True)
    parser.add_argument("--geometry-release-root", type=Path, required=True)
    parser.add_argument("--formal-input-root", type=Path, required=True)
    parser.add_argument("--lidar-inventory", type=Path, required=True)
    parser.add_argument("--lidar-root", type=Path, required=True)
    parser.add_argument("--global-packet-state", type=Path, required=True)
    parser.add_argument("--execution-root", type=Path, required=True)
    args = parser.parse_args()

    activation_path = args.activation.resolve()
    activation = require_json(activation_path)
    if (
        activation.get("schema") != "m3m_gcp_100k_three_track_activation_v1"
        or activation.get("status") != "ACTIVE_FROZEN"
        or activation.get("execution_authorized") is not True
        or activation.get("scene") != SCENE
        or activation.get("canonical_sha256") != canonical_sha256(activation)
    ):
        raise RuntimeError("three-track activation mismatch")
    candidate_path = Path(str(activation["candidate_manifest_path"])).resolve()
    candidate = require_json(candidate_path, str(activation["candidate_manifest_sha256"]))
    registry_row = candidate["rgb_registry"]
    registry = require_json(Path(str(registry_row["path"])), str(registry_row["sha256"]))
    _addendum_repo, addendum_config = validate_addendum_runtime(
        activation=activation,
        candidate=candidate,
        registry=registry,
        executing_file=Path(__file__),
    )
    methods = {str(row["method_id"]): row for row in registry.get("methods", [])}
    if args.method_id not in registry.get("ready_method_ids", []) or args.method_id not in methods:
        raise RuntimeError("LiDAR evaluation method is not activated READY")
    method = methods[args.method_id]
    recipe_path = Path(str(method["recipe_path"])).resolve()
    require_json(recipe_path, str(method["recipe_sha256"]))

    runtime_root = Path(str(candidate["candidate_output_root"])).resolve().parent
    formal_root = Path(str(candidate["formal_results_root"])).resolve()
    dispatch_path = args.lidar_dispatch_receipt.resolve()
    authorization_path = args.scene_authorization.resolve()
    global_state_path = args.global_packet_state.resolve()
    execution_root = args.execution_root.resolve()
    output_root = formal_root / "lidar" / args.method_id
    expected_dispatch = (
        runtime_root / "lidar-packet-dispatch" / args.method_id / "dispatch_receipt.json"
    )
    expected_authorization = (
        runtime_root / "lidar-authorizations" / args.method_id / "scene_authorization.json"
    )
    if (
        dispatch_path != expected_dispatch
        or authorization_path != expected_authorization
        or execution_root != runtime_root / "lidar-execution" / args.method_id
        or output_root.exists()
        or output_root.is_symlink()
        or execution_root.exists()
        or execution_root.is_symlink()
    ):
        raise RuntimeError("LiDAR evaluation path/freshness mismatch")
    dispatch = require_json(dispatch_path)
    authorization = require_json(authorization_path)
    packet_root = Path(str(dispatch["packet_set_root"])).resolve()
    packet_state = Path(str(dispatch["packet_state_path"])).resolve()
    packet_manifest = Path(str(dispatch["packet_manifest_path"])).resolve()
    validate_active_raw_packet_state(
        global_state_path,
        activation_path=activation_path,
        candidate=candidate,
        method_id=args.method_id,
        track="lidar",
        recipe_sha256=sha256_file(recipe_path),
        attempt_model_identity_sha256=method["attempt_model_identity_sha256"],
        packet_set_root=packet_root,
        track_packet_state_path=packet_state,
    )
    if (
        dispatch.get("status") != "PASS_LIDAR_PACKET_2196_DISPATCHED"
        or dispatch.get("packet_manifest_sha256") != sha256_file(packet_manifest)
        or dispatch.get("global_raw_packet_state_sha256")
        != sha256_file(global_state_path)
        or dispatch.get("canonical_sha256") != canonical_sha256(dispatch)
        or authorization.get("selected_method_id") != args.method_id
        or authorization.get("packet_manifest_sha256") != sha256_file(packet_manifest)
        or Path(str(authorization.get("authorized_output_root", ""))).resolve()
        != output_root
        or authorization.get("canonical_sha256") != canonical_sha256(authorization)
    ):
        raise RuntimeError("LiDAR dispatch/scene authorization mismatch")

    base_repo = Path(str(candidate["base_checkout"]["path"])).resolve()
    base_activation = Path(str(candidate["base_activation"]["path"])).resolve()
    contract = base_repo / "configs" / "m3m_gcp_lidar_formal_v1.json"
    schema = base_repo / "configs" / "m3m_gcp_lidar_formal_artifact_schema_v1.json"
    split = base_repo / "configs" / "gs_gcp_rgb_holdout_split_manifest_v1.json"
    method_registry = base_repo / "configs" / "m3m_gcp_native_quarter_method_registry_v3.json"
    methods_manifest = Path(str(candidate["methods_manifest"]["path"])).resolve()
    freeze = Path(str(candidate["scene_attempt_freeze"]["path"])).resolve()
    formal_input_root = args.formal_input_root.resolve()
    geometry_root = args.geometry_release_root.resolve()
    lidar_root = args.lidar_root.resolve()
    lidar_inventory = args.lidar_inventory.resolve()
    evaluator_script = base_repo / "code" / "gcp" / "evaluate_m3m_gcp_lidar_formal_v1.py"
    verifier_script = base_repo / "code" / "gcp" / "verify_m3m_gcp_lidar_formal_v1.py"
    configured_python = addendum_config.get("tracks", {}).get("lidar", {}).get(
        "environment_python"
    )
    if not configured_python:
        raise RuntimeError("reviewed addendum lacks a frozen LiDAR evaluation Python")
    lidar_python = Path(str(configured_python)).resolve()
    if not lidar_python.is_file() or lidar_python.is_symlink():
        raise RuntimeError("frozen LiDAR evaluation Python is missing or symlinked")
    result_path = output_root / "methods" / args.method_id / "metrics.json"
    reference_path = output_root / "reference_voxel_centres_local_metric.npz"
    verification_path = output_root / "independent_verification.json"
    evaluator_command = [
        str(lidar_python), "-B", str(evaluator_script),
        "--repo", str(base_repo),
        "--contract", str(contract),
        "--activation", str(base_activation),
        "--artifact-schema", str(schema),
        "--split", str(split),
        "--registry", str(method_registry),
        "--geometry-release-root", str(geometry_root),
        "--formal-input-root", str(formal_input_root),
        "--lidar-inventory", str(lidar_inventory),
        "--scene", SCENE,
        "--lidar-root", str(lidar_root),
        "--colmap-model", str(formal_input_root / "train" / "sparse" / "0"),
        "--gcp-csv", str(geometry_root / "benchmark" / "source_release_v1_3_0" / "gcp_points_cgcs2000_cm108_v1_3_0.csv"),
        "--sim3-json", str(geometry_root / "scenes" / SCENE / "common_sim3.json"),
        "--methods-json", str(methods_manifest),
        "--scene-attempt-freeze", str(freeze),
        "--scene-authorization", str(authorization_path),
        "--output-root", str(output_root),
        "--method-id", args.method_id,
    ]
    verifier_command = [
        str(lidar_python), "-B", str(verifier_script),
        "--result", str(result_path),
        "--reference", str(reference_path),
        "--artifact-schema", str(schema),
        "--contract", str(contract),
        "--activation", str(base_activation),
        "--scene-authorization", str(authorization_path),
        "--scene-attempt-freeze", str(freeze),
        "--methods", str(methods_manifest),
        "--output", str(verification_path),
    ]
    execution_root.mkdir(parents=True, exist_ok=False)
    eval_stdout = execution_root / "evaluator.stdout.log"
    eval_stderr = execution_root / "evaluator.stderr.log"
    verify_stdout = execution_root / "verifier.stdout.log"
    verify_stderr = execution_root / "verifier.stderr.log"
    failure_path = execution_root / "failure.json"
    receipt_path = execution_root / "execution_receipt.json"
    before = cgroup_memory_events()
    with eval_stdout.open("xb") as stdout_handle, eval_stderr.open("xb") as stderr_handle:
        evaluator = subprocess.run(
            evaluator_command, stdout=stdout_handle, stderr=stderr_handle, check=False
        )
    if evaluator.returncode != 0:
        delta = memory_event_delta(before, cgroup_memory_events())
        failure = build_failure(
            method_id=args.method_id,
            activation_path=activation_path,
            candidate=candidate,
            dispatch_path=dispatch_path,
            authorization_path=authorization_path,
            global_state_path=global_state_path,
            stage="lidar_evaluator",
            exit_code=evaluator.returncode,
            evaluator_command=evaluator_command,
            verifier_command=verifier_command,
            logs=[eval_stdout, eval_stderr],
            memory_delta=delta,
            error=f"LiDAR evaluator exited with code {evaluator.returncode}",
        )
        write_exclusive(failure_path, failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2, sort_keys=True))
        return int(evaluator.returncode or 1)
    with verify_stdout.open("xb") as stdout_handle, verify_stderr.open("xb") as stderr_handle:
        verifier = subprocess.run(
            verifier_command, stdout=stdout_handle, stderr=stderr_handle, check=False
        )
    if verifier.returncode != 0:
        delta = memory_event_delta(before, cgroup_memory_events())
        failure = build_failure(
            method_id=args.method_id,
            activation_path=activation_path,
            candidate=candidate,
            dispatch_path=dispatch_path,
            authorization_path=authorization_path,
            global_state_path=global_state_path,
            stage="lidar_independent_verifier",
            exit_code=verifier.returncode,
            evaluator_command=evaluator_command,
            verifier_command=verifier_command,
            logs=[eval_stdout, eval_stderr, verify_stdout, verify_stderr],
            memory_delta=delta,
            error=f"LiDAR verifier exited with code {verifier.returncode}",
        )
        write_exclusive(failure_path, failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2, sort_keys=True))
        return int(verifier.returncode or 1)
    try:
        result = require_json(result_path)
        verification = require_json(verification_path)
        if (
            result.get("schema") != "m3m_gcp_lidar_method_result_v1"
            or result.get("scene") != SCENE
            or result.get("method_id") != args.method_id
            or result.get("train_view_count") != 2196
            or result.get("summary_row", {}).get("status") != "COMPLETE_RANKED"
            or result.get("packet_manifest_sha256") != sha256_file(packet_manifest)
            or result.get("canonical_sha256") != canonical_sha256(result)
            or verification.get("status") != "PASS_VERIFIED_FORMAL_V1"
            or verification.get("method_id") != args.method_id
            or verification.get("method_result_sha256") != sha256_file(result_path)
            or verification.get("canonical_sha256") != canonical_sha256(verification)
        ):
            raise RuntimeError("LiDAR evaluator/verifier postcondition mismatch")
    except Exception as exc:
        delta = memory_event_delta(before, cgroup_memory_events())
        failure = build_failure(
            method_id=args.method_id,
            activation_path=activation_path,
            candidate=candidate,
            dispatch_path=dispatch_path,
            authorization_path=authorization_path,
            global_state_path=global_state_path,
            stage="lidar_postvalidation",
            exit_code=0,
            evaluator_command=evaluator_command,
            verifier_command=verifier_command,
            logs=[eval_stdout, eval_stderr, verify_stdout, verify_stderr],
            memory_delta=delta,
            error=f"{type(exc).__name__}: {exc}",
        )
        write_exclusive(failure_path, failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2, sort_keys=True))
        return 1
    receipt: dict[str, Any] = {
        "schema": "m3m_gcp_100k_lidar_evaluation_execution_receipt_v1",
        "status": "PASS_LIDAR_EVALUATOR_AND_INDEPENDENT_VERIFIER",
        "scene": SCENE,
        "method_id": args.method_id,
        "three_track_activation_sha256": sha256_file(activation_path),
        "scene_attempt_freeze_sha256": candidate["scene_attempt_freeze"]["sha256"],
        "methods_manifest_sha256": candidate["methods_manifest"]["sha256"],
        "lidar_dispatch_receipt_sha256": sha256_file(dispatch_path),
        "scene_authorization_sha256": sha256_file(authorization_path),
        "global_raw_packet_state_sha256": sha256_file(global_state_path),
        "evaluator_command": evaluator_command,
        "evaluator_command_sha256": command_sha256(evaluator_command),
        "verifier_command": verifier_command,
        "verifier_command_sha256": command_sha256(verifier_command),
        "result_path": str(result_path),
        "result_sha256": sha256_file(result_path),
        "verification_path": str(verification_path),
        "verification_sha256": sha256_file(verification_path),
        "evaluator_stdout_sha256": sha256_file(eval_stdout),
        "evaluator_stderr_sha256": sha256_file(eval_stderr),
        "verifier_stdout_sha256": sha256_file(verify_stdout),
        "verifier_stderr_sha256": sha256_file(verify_stderr),
        "cgroup_memory_events_delta": memory_event_delta(before, cgroup_memory_events()),
    }
    receipt["canonical_sha256"] = canonical_sha256(receipt)
    write_exclusive(receipt_path, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
