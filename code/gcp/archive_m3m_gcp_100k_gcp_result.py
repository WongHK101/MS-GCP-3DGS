#!/usr/bin/env python3
"""Create the lightweight GCP archive required before deleting one 211-view packet set."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from m3m_gcp_lidar_artifacts import canonical_sha256, sha256_file
from m3m_gcp_100k_raw_packet_state import (
    active_raw_packet_state_path,
    validate_active_raw_packet_state,
)
from m3m_gcp_100k_three_track_runtime import validate_addendum_runtime


SCENE = "gcp_100000_20260610"
PROTOCOL_ID = "m3m_gcp_native_quarter_geometry_v2"
EVAL_FILES = (
    "observation_samples.csv",
    "point_results.csv",
    "evaluation_summary.json",
    "evaluator_manifest.json",
)


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


def copy_exclusive(source: Path, destination: Path) -> dict[str, Any]:
    source = source.resolve()
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as input_handle, destination.open("xb") as output_handle:
        shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
    os.chmod(destination, 0o444)
    if sha256_file(destination) != sha256_file(source):
        raise RuntimeError(f"archive byte verification failed: {source}")
    return {
        "source_path": str(source),
        "archive_path": str(destination.resolve()),
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
    }


def load_activation(path: Path) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    path = path.resolve()
    activation = require_json(path)
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
        candidate.get("canonical_sha256")
        != activation["candidate_manifest_canonical_sha256"]
        or canonical_sha256(candidate) != activation["candidate_manifest_canonical_sha256"]
    ):
        raise RuntimeError("activation/candidate binding mismatch")
    registry_row = candidate["rgb_registry"]
    registry = require_json(Path(str(registry_row["path"])), str(registry_row["sha256"]))
    if (
        registry.get("canonical_sha256") != registry_row["canonical_sha256"]
        or canonical_sha256(registry) != registry_row["canonical_sha256"]
    ):
        raise RuntimeError("activated registry mismatch")
    return path, activation, candidate, registry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--method-id", choices=("citygs_x", "metrogs"), required=True)
    parser.add_argument("--gcp-authorization", type=Path, required=True)
    parser.add_argument("--gcp-packet-phase-success", type=Path, required=True)
    parser.add_argument("--gcp-execution-receipt", type=Path, required=True)
    parser.add_argument("--packet-state", type=Path, required=True)
    parser.add_argument("--global-packet-state", type=Path, required=True)
    parser.add_argument("--packet-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    args = parser.parse_args()

    activation_path, activation, candidate, registry = load_activation(args.activation)
    validate_addendum_runtime(
        activation=activation,
        candidate=candidate,
        registry=registry,
        executing_file=Path(__file__),
    )
    if args.method_id not in registry.get("ready_method_ids", []):
        raise RuntimeError("GCP archive method is not activated READY")
    runtime_root = Path(str(candidate["candidate_output_root"])).resolve().parent
    evaluation_root = args.evaluation_root.resolve()
    archive_root = args.archive_root.resolve()
    expected_eval = Path(str(candidate["formal_results_root"])).resolve() / "gcp" / args.method_id
    expected_archive = (
        Path(str(candidate["formal_results_root"])).resolve()
        / "gcp-lightweight-archives"
        / args.method_id
    )
    if evaluation_root != expected_eval or archive_root != expected_archive:
        raise RuntimeError("GCP result/archive root differs from activated namespace")
    if archive_root.exists() or archive_root.is_symlink():
        raise FileExistsError(archive_root)

    authorization_path = args.gcp_authorization.resolve()
    authorization = require_json(authorization_path)
    phase_path = args.gcp_packet_phase_success.resolve()
    phase = require_json(phase_path)
    state_path = args.packet_state.resolve()
    state = require_json(state_path)
    global_state_path = args.global_packet_state.resolve()
    packet_path = args.packet_manifest.resolve()
    packet = require_json(packet_path)
    verification_path = args.verification.resolve()
    verification = require_json(verification_path)
    execution_receipt_path = args.gcp_execution_receipt.resolve()
    execution_receipt = require_json(execution_receipt_path)
    summary_path = evaluation_root / "evaluation_summary.json"
    summary = require_json(summary_path)
    evaluator_manifest_path = evaluation_root / "evaluator_manifest.json"
    evaluator_manifest = require_json(evaluator_manifest_path)

    methods = {str(row["method_id"]): row for row in registry.get("methods", [])}
    method = methods[args.method_id]
    recipe_path = Path(str(method["recipe_path"])).resolve()
    validate_active_raw_packet_state(
        global_state_path,
        activation_path=activation_path,
        candidate=candidate,
        method_id=args.method_id,
        track="gcp",
        recipe_sha256=sha256_file(recipe_path),
        attempt_model_identity_sha256=method["attempt_model_identity_sha256"],
        packet_set_root=packet_path.parent,
        track_packet_state_path=state_path,
    )
    if global_state_path != active_raw_packet_state_path(candidate):
        raise RuntimeError("GCP archive global raw-packet state path mismatch")

    if (
        authorization.get("schema") != "m3m_gcp_100k_gcp_execution_authorization_v1"
        or authorization.get("status") != "ACTIVE_FROZEN"
        or authorization.get("execution_authorized") is not True
        or authorization.get("scene") != SCENE
        or authorization.get("method_id") != args.method_id
        or authorization.get("three_track_activation_sha256")
        != sha256_file(activation_path)
        or authorization.get("scene_attempt_freeze_sha256")
        != candidate["scene_attempt_freeze"]["sha256"]
        or authorization.get("methods_manifest_sha256")
        != candidate["methods_manifest"]["sha256"]
        or authorization.get("gcp_packet_phase_success_sha256") != sha256_file(phase_path)
        or authorization.get("packet_state_sha256") != sha256_file(state_path)
        or authorization.get("global_raw_packet_state_path") != str(global_state_path)
        or authorization.get("global_raw_packet_state_sha256")
        != sha256_file(global_state_path)
        or authorization.get("packet_manifest_sha256") != sha256_file(packet_path)
        or authorization.get("authorized_output_root") != str(evaluation_root)
        or authorization.get("authorized_verification_output") != str(verification_path)
        or authorization.get("canonical_sha256") != canonical_sha256(authorization)
    ):
        raise RuntimeError("GCP authorization/current activation binding mismatch")
    if (
        phase.get("status") != "PASS_GCP_PACKET_211"
        or phase.get("method_id") != args.method_id
        or phase.get("three_track_activation_sha256") != sha256_file(activation_path)
        or phase.get("packet_state_sha256") != sha256_file(state_path)
        or phase.get("canonical_sha256") != canonical_sha256(phase)
        or state.get("method_id") != args.method_id
        or state.get("three_track_activation_sha256") != sha256_file(activation_path)
        or state.get("canonical_sha256") != canonical_sha256(state)
        or packet.get("scene") != SCENE
        or packet.get("protocol_id") != PROTOCOL_ID
        or packet.get("rendered_view_count") != 211
        or len(packet.get("depth_index", [])) != 211
        or len(packet.get("packet_index", [])) != 211
    ):
        raise RuntimeError("GCP packet evidence mismatch")
    if (
        summary.get("scene") != SCENE
        or summary.get("method_id") != args.method_id
        or summary.get("protocol_id") != PROTOCOL_ID
        or summary.get("packet_manifest_sha256") != sha256_file(packet_path)
        or summary.get("status") not in {"COMPLETE_RANKED", "INCOMPLETE_UNRANKED"}
        or evaluator_manifest.get("packet_manifest_sha256") != sha256_file(packet_path)
        or verification.get("schema")
        != "m3m_gcp_native_quarter_evaluator_output_independent_verification_v1"
        or verification.get("status") != "PASS"
        or verification.get("passed") is not True
        or verification.get("scene") != SCENE
        or verification.get("method_id") != args.method_id
        or verification.get("ranking_status") != summary.get("status")
        or verification.get("recomputed_residual_statistics")
        != summary.get("residual_statistics")
    ):
        raise RuntimeError("GCP evaluator/independent-verifier gate mismatch")
    if (
        execution_receipt.get("schema")
        != "m3m_gcp_100k_gcp_evaluation_execution_receipt_v1"
        or execution_receipt.get("status")
        != "PASS_GCP_EVALUATOR_AND_INDEPENDENT_VERIFIER"
        or execution_receipt.get("method_id") != args.method_id
        or execution_receipt.get("three_track_activation_sha256")
        != sha256_file(activation_path)
        or execution_receipt.get("gcp_authorization_sha256")
        != sha256_file(authorization_path)
        or execution_receipt.get("packet_manifest_sha256") != sha256_file(packet_path)
        or execution_receipt.get("global_raw_packet_state_sha256")
        != sha256_file(global_state_path)
        or execution_receipt.get("summary_sha256") != sha256_file(summary_path)
        or execution_receipt.get("verification_sha256") != sha256_file(verification_path)
        or execution_receipt.get("canonical_sha256")
        != canonical_sha256(execution_receipt)
    ):
        raise RuntimeError("GCP formal execution receipt mismatch")
    for name in EVAL_FILES[:3]:
        source = evaluation_root / name
        if not source.is_file() or sha256_file(source) != evaluator_manifest.get("outputs", {}).get(name):
            raise RuntimeError(f"GCP evaluator output identity mismatch: {name}")

    sources = [
        (authorization_path, "gcp_execution_authorization.json"),
        (phase_path, "gcp_packet_phase_success.json"),
        (state_path, "gcp_packet_state.json"),
        (global_state_path, "active_raw_packet_state.json"),
        (packet_path, "depth_export_manifest.json"),
        (execution_receipt_path, "gcp_evaluation_execution_receipt.json"),
        *[(evaluation_root / name, f"evaluation/{name}") for name in EVAL_FILES],
        (verification_path, "independent_verification.json"),
    ]
    archive_root.mkdir(parents=True, exist_ok=False)
    files = [copy_exclusive(source, archive_root / relative) for source, relative in sources]
    payload: dict[str, Any] = {
        "schema": "m3m_gcp_100k_gcp_lightweight_archive_v1",
        "status": "PASS_GCP_LIGHTWEIGHT_ARCHIVE_BYTE_VERIFIED",
        "scene": SCENE,
        "method_id": args.method_id,
        "three_track_activation_path": str(activation_path),
        "three_track_activation_sha256": sha256_file(activation_path),
        "scene_attempt_freeze_sha256": candidate["scene_attempt_freeze"]["sha256"],
        "methods_manifest_sha256": candidate["methods_manifest"]["sha256"],
        "packet_manifest_sha256": sha256_file(packet_path),
        "gcp_authorization_sha256": sha256_file(authorization_path),
        "gcp_packet_phase_success_sha256": sha256_file(phase_path),
        "packet_state_sha256": sha256_file(state_path),
        "global_raw_packet_state_sha256": sha256_file(global_state_path),
        "evaluation_summary_sha256": sha256_file(summary_path),
        "gcp_execution_receipt_sha256": sha256_file(execution_receipt_path),
        "verification_sha256": sha256_file(verification_path),
        "archive_root": str(archive_root),
        "files": files,
        "raw_metric_depth_packet_files_archived": False,
        "source_and_archive_bytes_reverified": True,
        "packet_release_authorized_by_archive_alone": False,
        "runtime_root": str(runtime_root),
    }
    payload["canonical_sha256"] = canonical_sha256(payload)
    manifest_path = archive_root / "archive_manifest.json"
    write_exclusive(manifest_path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
