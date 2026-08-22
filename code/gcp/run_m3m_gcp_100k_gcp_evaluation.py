#!/usr/bin/env python3
"""Execute exactly one activation-bound GCP evaluator and independent verifier pair."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from m3m_gcp_100k_three_track_runtime import (
    validate_addendum_runtime,
    validate_frozen_gcp_evaluation_runtime,
)
from m3m_gcp_100k_raw_packet_state import validate_active_raw_packet_state
from m3m_gcp_lidar_artifacts import canonical_sha256, command_sha256, sha256_file


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
            (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
                "utf-8"
            ),
        )
    finally:
        os.close(descriptor)


def failure_payload(
    *, method_id: str, activation_sha: str, authorization_sha: str,
    global_state_path: Path, stage: str, exit_code: int, logs: list[Path],
    error: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "m3m_gcp_100k_gcp_evaluation_failure_v1",
        "status": "INCOMPLETE_UNRANKED",
        "scene": SCENE,
        "method_id": method_id,
        "failure_stage": stage,
        "three_track_activation_sha256": activation_sha,
        "gcp_authorization_sha256": authorization_sha,
        "global_raw_packet_state_sha256": sha256_file(global_state_path),
        "exit_code": exit_code,
        "logs": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in logs
            if path.is_file()
        ],
        "retry_forbidden_after_child_start": True,
        "error": error,
    }
    payload["canonical_sha256"] = canonical_sha256(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--method-id", choices=("citygs_x", "metrogs"), required=True)
    parser.add_argument("--gcp-authorization", type=Path, required=True)
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
    if args.method_id not in registry.get("ready_method_ids", []):
        raise RuntimeError("GCP evaluation method is not activated READY")
    methods = {str(row["method_id"]): row for row in registry.get("methods", [])}
    method = methods[args.method_id]
    recipe_path = Path(str(method["recipe_path"])).resolve()

    authorization_path = args.gcp_authorization.resolve()
    authorization = require_json(authorization_path)
    global_state_path = args.global_packet_state.resolve()
    validate_active_raw_packet_state(
        global_state_path,
        activation_path=activation_path,
        candidate=candidate,
        method_id=args.method_id,
        track="gcp",
        recipe_sha256=sha256_file(recipe_path),
        attempt_model_identity_sha256=method["attempt_model_identity_sha256"],
        packet_set_root=Path(str(authorization["packet_manifest_path"])).resolve().parent,
        track_packet_state_path=Path(str(authorization["packet_state_path"])).resolve(),
    )
    execution_root = args.execution_root.resolve()
    runtime_root = Path(str(candidate["candidate_output_root"])).resolve().parent
    expected_execution = runtime_root / "gcp-execution" / args.method_id
    if execution_root != expected_execution:
        raise RuntimeError("GCP execution root differs from activated namespace")
    if execution_root.exists() or execution_root.is_symlink():
        raise FileExistsError(execution_root)
    if (
        authorization.get("schema") != "m3m_gcp_100k_gcp_execution_authorization_v1"
        or authorization.get("status") != "ACTIVE_FROZEN"
        or authorization.get("execution_authorized") is not True
        or authorization.get("scene") != SCENE
        or authorization.get("method_id") != args.method_id
        or authorization.get("three_track_activation_sha256") != sha256_file(activation_path)
        or authorization.get("scene_attempt_freeze_sha256")
        != candidate["scene_attempt_freeze"]["sha256"]
        or authorization.get("methods_manifest_sha256")
        != candidate["methods_manifest"]["sha256"]
        or authorization.get("global_raw_packet_state_path") != str(global_state_path)
        or authorization.get("global_raw_packet_state_sha256")
        != sha256_file(global_state_path)
        or authorization.get("canonical_sha256") != canonical_sha256(authorization)
    ):
        raise RuntimeError("GCP execution authorization binding mismatch")
    evaluator_command = [str(value) for value in authorization.get("evaluator_command", [])]
    verifier_command = [str(value) for value in authorization.get("verifier_command", [])]
    configured_python = Path(str(authorization.get("gcp_evaluation_python_path", "")))
    runtime_identity, evaluation_environment = validate_frozen_gcp_evaluation_runtime(
        addendum_config.get("tracks", {}).get("gcp", {}),
        requested_python=configured_python,
    )
    if (
        command_sha256(evaluator_command) != authorization.get("evaluator_command_sha256")
        or command_sha256(verifier_command) != authorization.get("verifier_command_sha256")
        or not evaluator_command
        or not verifier_command
        or evaluator_command[0] != str(configured_python)
        or verifier_command[0] != str(configured_python)
        or authorization.get("gcp_evaluation_runtime_identity") != runtime_identity
        or authorization.get("gcp_evaluation_runtime_identity_canonical_sha256")
        != canonical_sha256(runtime_identity)
        or authorization.get("gcp_evaluation_subprocess_environment")
        != evaluation_environment
    ):
        raise RuntimeError("GCP evaluator/verifier command or frozen environment mismatch")
    output_root = Path(str(authorization["authorized_output_root"])).resolve()
    verification_path = Path(str(authorization["authorized_verification_output"])).resolve()
    if output_root.exists() or output_root.is_symlink() or verification_path.exists() or verification_path.is_symlink():
        raise FileExistsError("GCP evaluator/verifier formal outputs must be fresh")

    execution_root.mkdir(parents=True, exist_ok=False)
    eval_stdout = execution_root / "evaluator.stdout.log"
    eval_stderr = execution_root / "evaluator.stderr.log"
    verify_stdout = execution_root / "verifier.stdout.log"
    verify_stderr = execution_root / "verifier.stderr.log"
    failure_path = execution_root / "failure.json"
    receipt_path = execution_root / "execution_receipt.json"
    with eval_stdout.open("xb") as stdout_handle, eval_stderr.open("xb") as stderr_handle:
        evaluator = subprocess.run(
            evaluator_command,
            stdout=stdout_handle,
            stderr=stderr_handle,
            env=evaluation_environment,
            check=False,
        )
    if evaluator.returncode != 0:
        payload = failure_payload(
            method_id=args.method_id,
            activation_sha=sha256_file(activation_path),
            authorization_sha=sha256_file(authorization_path),
            global_state_path=global_state_path,
            stage="gcp_evaluator",
            exit_code=evaluator.returncode,
            logs=[eval_stdout, eval_stderr],
            error=f"GCP evaluator exited with code {evaluator.returncode}",
        )
        write_exclusive(failure_path, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return int(evaluator.returncode or 1)
    with verify_stdout.open("xb") as stdout_handle, verify_stderr.open("xb") as stderr_handle:
        verifier = subprocess.run(
            verifier_command,
            stdout=stdout_handle,
            stderr=stderr_handle,
            env=evaluation_environment,
            check=False,
        )
    if verifier.returncode != 0:
        payload = failure_payload(
            method_id=args.method_id,
            activation_sha=sha256_file(activation_path),
            authorization_sha=sha256_file(authorization_path),
            global_state_path=global_state_path,
            stage="gcp_independent_verifier",
            exit_code=verifier.returncode,
            logs=[eval_stdout, eval_stderr, verify_stdout, verify_stderr],
            error=f"GCP verifier exited with code {verifier.returncode}",
        )
        write_exclusive(failure_path, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return int(verifier.returncode or 1)

    summary_path = output_root / "evaluation_summary.json"
    manifest_path = output_root / "evaluator_manifest.json"
    packet_sha = authorization["packet_manifest_sha256"]
    try:
        summary = require_json(summary_path)
        manifest = require_json(manifest_path)
        verification = require_json(verification_path)
        if (
            summary.get("scene") != SCENE
            or summary.get("method_id") != args.method_id
            or summary.get("packet_manifest_sha256") != packet_sha
            or summary.get("status") not in {"COMPLETE_RANKED", "INCOMPLETE_UNRANKED"}
            or manifest.get("packet_manifest_sha256") != packet_sha
            or verification.get("status") != "PASS"
            or verification.get("passed") is not True
            or verification.get("scene") != SCENE
            or verification.get("method_id") != args.method_id
            or verification.get("ranking_status") != summary.get("status")
        ):
            raise RuntimeError("GCP evaluator/verifier postcondition mismatch")
    except Exception as exc:
        payload = failure_payload(
            method_id=args.method_id,
            activation_sha=sha256_file(activation_path),
            authorization_sha=sha256_file(authorization_path),
            global_state_path=global_state_path,
            stage="gcp_postvalidation",
            exit_code=0,
            logs=[eval_stdout, eval_stderr, verify_stdout, verify_stderr],
            error=f"{type(exc).__name__}: {exc}",
        )
        write_exclusive(failure_path, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 1
    receipt: dict[str, Any] = {
        "schema": "m3m_gcp_100k_gcp_evaluation_execution_receipt_v1",
        "status": "PASS_GCP_EVALUATOR_AND_INDEPENDENT_VERIFIER",
        "scene": SCENE,
        "method_id": args.method_id,
        "three_track_activation_sha256": sha256_file(activation_path),
        "scene_attempt_freeze_sha256": candidate["scene_attempt_freeze"]["sha256"],
        "methods_manifest_sha256": candidate["methods_manifest"]["sha256"],
        "gcp_authorization_path": str(authorization_path),
        "gcp_authorization_sha256": sha256_file(authorization_path),
        "packet_manifest_sha256": packet_sha,
        "global_raw_packet_state_sha256": sha256_file(global_state_path),
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "evaluator_manifest_path": str(manifest_path),
        "evaluator_manifest_sha256": sha256_file(manifest_path),
        "verification_path": str(verification_path),
        "verification_sha256": sha256_file(verification_path),
        "gcp_evaluation_runtime_identity_canonical_sha256": canonical_sha256(
            runtime_identity
        ),
        "evaluator_stdout_sha256": sha256_file(eval_stdout),
        "evaluator_stderr_sha256": sha256_file(eval_stderr),
        "verifier_stdout_sha256": sha256_file(verify_stdout),
        "verifier_stderr_sha256": sha256_file(verify_stderr),
    }
    receipt["canonical_sha256"] = canonical_sha256(receipt)
    write_exclusive(receipt_path, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
