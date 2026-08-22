#!/usr/bin/env python3
"""Dispatch the unchanged frozen 2,196-train-view LiDAR packet phase after the GCP gate."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from m3m_gcp_lidar_artifacts import canonical_sha256, command_sha256, sha256_file
from m3m_gcp_100k_three_track_runtime import validate_addendum_runtime
from run_m3m_gcp_100k_guarded import validate_model_identity_bundle


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


def validate_gcp_precondition(
    *, method_id: str, activation_path: Path, activation: dict[str, Any], candidate: dict[str, Any], receipt: Path | None
) -> dict[str, Any]:
    if method_id == "3dgs_original":
        row = candidate["legacy_3dgs_gcp_adoption"]
        path = Path(str(row["path"])).resolve()
        payload = require_json(path, str(row["sha256"]))
        if (
            payload.get("status") != "PASS_LEGACY_GCP_ADOPTION_CANDIDATE"
            or payload.get("method_id") != method_id
            or payload.get("scene_attempt_freeze_sha256")
            != candidate["scene_attempt_freeze"]["sha256"]
            or payload.get("canonical_sha256") != row["canonical_sha256"]
            or canonical_sha256(payload) != row["canonical_sha256"]
            or activation.get("legacy_3dgs_gcp_adoption_sha256") != sha256_file(path)
        ):
            raise RuntimeError("legacy 3DGS GCP adoption precondition mismatch")
        return {"kind": "LEGACY_3DGS_GCP_ADOPTION", "path": str(path), "sha256": sha256_file(path)}
    if receipt is None:
        raise RuntimeError("new method LiDAR packet requires a GCP deletion receipt")
    path = receipt.resolve()
    payload = require_json(path)
    runtime_root = Path(str(candidate["candidate_output_root"])).resolve().parent
    expected_path = runtime_root / "gcp-packet-release" / method_id / "deletion_receipt.json"
    packet_root = runtime_root / "gcp-packet-scratch" / method_id
    state_path = runtime_root / "gcp-packet-scratch" / "ACTIVE_GCP_PACKET_STATE.json"
    if (
        path != expected_path
        or payload.get("schema") != "m3m_gcp_100k_gcp_packet_deletion_receipt_v1"
        or payload.get("status") != "PASS_GCP_PACKET_DELETED"
        or payload.get("method_id") != method_id
        or payload.get("three_track_activation_sha256") != sha256_file(activation_path)
        or payload.get("scene_attempt_freeze_sha256")
        != candidate["scene_attempt_freeze"]["sha256"]
        or payload.get("methods_manifest_sha256") != candidate["methods_manifest"]["sha256"]
        or payload.get("packet_set_root") != str(packet_root)
        or payload.get("packet_state_path") != str(state_path)
        or payload.get("packet_set_root_absent") is not True
        or payload.get("packet_state_absent") is not True
        or payload.get("canonical_sha256") != canonical_sha256(payload)
        or packet_root.exists()
        or packet_root.is_symlink()
        or state_path.exists()
        or state_path.is_symlink()
    ):
        raise RuntimeError("GCP deletion receipt/current-state precondition mismatch")
    return {"kind": "GCP_PACKET_DELETION_RECEIPT", "path": str(path), "sha256": sha256_file(path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--method-id", choices=("3dgs_original", "citygs_x", "metrogs"), required=True)
    parser.add_argument("--gcp-deletion-receipt", type=Path)
    parser.add_argument("--dispatch-root", type=Path, required=True)
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
    if (
        candidate.get("canonical_sha256") != activation["candidate_manifest_canonical_sha256"]
        or canonical_sha256(candidate) != activation["candidate_manifest_canonical_sha256"]
    ):
        raise RuntimeError("activation/candidate binding mismatch")
    registry_row = candidate["rgb_registry"]
    registry = require_json(Path(str(registry_row["path"])), str(registry_row["sha256"]))
    methods = {str(row["method_id"]): row for row in registry.get("methods", [])}
    if args.method_id not in registry.get("ready_method_ids", []) or args.method_id not in methods:
        raise RuntimeError("LiDAR packet method is not activated READY")
    method = methods[args.method_id]
    validate_addendum_runtime(
        activation=activation,
        candidate=candidate,
        registry=registry,
        executing_file=Path(__file__),
    )

    base_repo = Path(str(candidate["base_checkout"]["path"])).resolve()
    base = candidate["base_checkout"]
    if (
        subprocess.check_output(["git", "-C", str(base_repo), "rev-parse", "HEAD"], text=True).strip()
        != base["commit"]
        or subprocess.check_output(
            ["git", "-C", str(base_repo), "show", "-s", "--format=%T", "HEAD"], text=True
        ).strip()
        != base["tree"]
        or subprocess.check_output(["git", "-C", str(base_repo), "status", "--porcelain"], text=True).strip()
    ):
        raise RuntimeError("frozen base checkout identity mismatch")
    recipe_path = Path(str(method["recipe_path"])).resolve()
    recipe = require_json(recipe_path, str(method["recipe_sha256"]))
    identity_path = Path(str(method["attempt_model_identity_path"])).resolve()
    bound_recipe = dict(recipe)
    bound_recipe["_recipe_path"] = str(recipe_path)
    identity = validate_model_identity_bundle(
        manifest_path=identity_path,
        method_id=args.method_id,
        run_root=Path(str(method["run_root"])).resolve(),
        recipe=bound_recipe,
        repo=base_repo,
    )
    if (
        sha256_file(identity_path) != method["attempt_model_identity_sha256"]
        or identity["canonical_sha256"]
        != method["attempt_model_identity_canonical_sha256"]
    ):
        raise RuntimeError("LiDAR packet frozen model identity mismatch")
    gcp_gate = validate_gcp_precondition(
        method_id=args.method_id,
        activation_path=activation_path,
        activation=activation,
        candidate=candidate,
        receipt=args.gcp_deletion_receipt,
    )

    runtime_root = Path(str(candidate["candidate_output_root"])).resolve().parent
    dispatch_root = args.dispatch_root.resolve()
    expected_dispatch = runtime_root / "lidar-packet-dispatch" / args.method_id
    packet_root = Path(str(recipe["authorized_packet_set_root"])).resolve()
    packet_state = Path(str(recipe["authorized_packet_state"])).resolve()
    if dispatch_root != expected_dispatch:
        raise RuntimeError("LiDAR packet dispatch root differs from activated namespace")
    for path in (dispatch_root, packet_root, packet_state):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"LiDAR packet dispatch requires fresh path: {path}")
    # A GCP scratch state from any new method is forbidden while a LiDAR packet is active.
    gcp_scratch = runtime_root / "gcp-packet-scratch"
    if gcp_scratch.exists() and any(gcp_scratch.iterdir()):
        raise RuntimeError("GCP packet scratch is not empty before LiDAR packet export")

    base_activation = Path(str(candidate["base_activation"]["path"])).resolve()
    plan = base_repo / "configs" / "m3m_gcp_native_quarter_100k_ten_method_execution_plan_v3.json"
    recipe_manifest = base_repo / "configs" / "m3m_gcp_native_quarter_100k_recipe_manifest_v3.json"
    roots = recipe["phase_roots"]["packet"]
    source = recipe["source_bindings"]["packet"]
    failure = Path(str(recipe["authorized_evidence_root"])).resolve() / "packet" / "failure.json"
    guard_python = str(recipe["phase_commands"]["packet"][0])
    guard = base_repo / "code" / "gcp" / "run_m3m_gcp_100k_guarded.py"
    command = [
        guard_python,
        "-B",
        str(guard),
        "--repo",
        str(base_repo),
        "--activation",
        str(base_activation),
        "--plan",
        str(plan),
        "--recipe-manifest",
        str(recipe_manifest),
        "--recipe",
        str(recipe_path),
        "--method-id",
        args.method_id,
        "--phase",
        "packet",
        "--run-root",
        str(Path(str(method["run_root"])).resolve()),
        "--dataset-root",
        str(Path(str(roots["dataset_root"])).resolve()),
        "--source-root",
        str(Path(str(source["root"])).resolve()),
        "--prior-root",
        str(Path(str(roots["prior_root"])).resolve()),
        "--failure-evidence",
        str(failure),
        "--packet-state",
        str(packet_state),
        "--packet-set-root",
        str(packet_root),
        "--scene-attempt-freeze",
        str(Path(str(candidate["scene_attempt_freeze"]["path"])).resolve()),
    ]
    dispatch_root.mkdir(parents=True, exist_ok=False)
    stdout_path = dispatch_root / "base_guard.stdout.log"
    stderr_path = dispatch_root / "base_guard.stderr.log"
    environment_path = dispatch_root / "dispatch_environment.json"
    success_path = dispatch_root / "dispatch_receipt.json"
    failure_path = dispatch_root / "failure.json"
    environment: dict[str, Any] = {
        "schema": "m3m_gcp_100k_lidar_packet_dispatch_environment_v1",
        "scene": SCENE,
        "method_id": args.method_id,
        "three_track_activation_sha256": sha256_file(activation_path),
        "scene_attempt_freeze_sha256": candidate["scene_attempt_freeze"]["sha256"],
        "attempt_model_identity_sha256": sha256_file(identity_path),
        "gcp_precondition": gcp_gate,
        "command": command,
        "command_sha256": command_sha256(command),
        "python": sys.version,
    }
    environment["canonical_sha256"] = canonical_sha256(environment)
    write_exclusive(environment_path, environment)
    with stdout_path.open("xb") as stdout_handle, stderr_path.open("xb") as stderr_handle:
        process = subprocess.run(command, stdout=stdout_handle, stderr=stderr_handle, check=False)
    if process.returncode != 0:
        failure_payload: dict[str, Any] = {
            "schema": "m3m_gcp_100k_lidar_packet_dispatch_failure_v1",
            "status": "FAILED_UNRANKED",
            "scene": SCENE,
            "method_id": args.method_id,
            "three_track_activation_sha256": sha256_file(activation_path),
            "scene_attempt_freeze_sha256": candidate["scene_attempt_freeze"]["sha256"],
            "attempt_model_identity_sha256": sha256_file(identity_path),
            "environment_sha256": sha256_file(environment_path),
            "stdout_sha256": sha256_file(stdout_path),
            "stderr_sha256": sha256_file(stderr_path),
            "exit_code": process.returncode,
            "retry_forbidden_after_base_guard_child_start": True,
        }
        failure_payload["canonical_sha256"] = canonical_sha256(failure_payload)
        write_exclusive(failure_path, failure_payload)
        print(json.dumps(failure_payload, ensure_ascii=False, indent=2, sort_keys=True))
        return int(process.returncode or 1)

    phase_path = Path(str(recipe["authorized_evidence_root"])).resolve() / "packet" / "phase_success.json"
    phase = require_json(phase_path)
    state = require_json(packet_state)
    packet_manifest_path = packet_root / "depth_export_manifest.json"
    packet = require_json(packet_manifest_path)
    depth_rows = packet.get("depth_index", [])
    if (
        phase.get("schema") != "m3m_gcp_100k_phase_success_v2"
        or phase.get("status") != "PASS"
        or phase.get("scene") != SCENE
        or phase.get("method_id") != args.method_id
        or phase.get("phase") != "packet"
        or phase.get("recipe_sha256") != sha256_file(recipe_path)
        or phase.get("canonical_sha256") != canonical_sha256(phase)
        or state.get("schema") != "m3m_gcp_100k_single_packet_state_v1"
        or state.get("method_id") != args.method_id
        or Path(str(state.get("packet_set_root", ""))).resolve() != packet_root
        or packet.get("scene") != SCENE
        or packet.get("rendered_view_count") != 2196
        or len(depth_rows) != 2196
        or len(packet.get("packet_index", [])) != 2196
        or len({str(row.get("image_name", "")) for row in depth_rows}) != 2196
        or any(row.get("split") != "train" for row in depth_rows)
    ):
        raise RuntimeError("base LiDAR packet phase postvalidation mismatch")
    receipt: dict[str, Any] = {
        "schema": "m3m_gcp_100k_lidar_packet_dispatch_receipt_v1",
        "status": "PASS_LIDAR_PACKET_2196_DISPATCHED",
        "scene": SCENE,
        "method_id": args.method_id,
        "three_track_activation_path": str(activation_path),
        "three_track_activation_sha256": sha256_file(activation_path),
        "candidate_manifest_sha256": sha256_file(candidate_path),
        "scene_attempt_freeze_sha256": candidate["scene_attempt_freeze"]["sha256"],
        "methods_manifest_sha256": candidate["methods_manifest"]["sha256"],
        "recipe_path": str(recipe_path),
        "recipe_sha256": sha256_file(recipe_path),
        "attempt_model_identity_path": str(identity_path),
        "attempt_model_identity_sha256": sha256_file(identity_path),
        "attempt_model_identity_canonical_sha256": identity["canonical_sha256"],
        "gcp_precondition": gcp_gate,
        "packet_set_root": str(packet_root),
        "packet_state_path": str(packet_state),
        "packet_state_sha256": sha256_file(packet_state),
        "packet_manifest_path": str(packet_manifest_path),
        "packet_manifest_sha256": sha256_file(packet_manifest_path),
        "packet_view_count": 2196,
        "phase_success_path": str(phase_path),
        "phase_success_sha256": sha256_file(phase_path),
        "environment_sha256": sha256_file(environment_path),
        "stdout_sha256": sha256_file(stdout_path),
        "stderr_sha256": sha256_file(stderr_path),
    }
    receipt["canonical_sha256"] = canonical_sha256(receipt)
    write_exclusive(success_path, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
