#!/usr/bin/env python3
"""Fail-closed cleanup of one failed raw packet without authorizing a retry."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from m3m_gcp_100k_raw_packet_state import (
    active_raw_packet_state_path,
    validate_active_raw_packet_state,
)
from m3m_gcp_100k_three_track_runtime import validate_addendum_runtime
from m3m_gcp_lidar_artifacts import canonical_sha256, sha256_file
from metric_depth_packet import directory_tree_hash


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


def validate_failure_archive(path: Path, root: Path) -> dict[str, Any]:
    payload = require_json(path)
    rows = payload.get("files", [])
    relatives = [str(row.get("relative_path", "")) for row in rows]
    if (
        path.resolve() != root.resolve() / "archive_manifest.json"
        or payload.get("schema")
        != "m3m_gcp_100k_failed_packet_lightweight_archive_v1"
        or payload.get("status") != "PASS_FAILURE_EVIDENCE_ARCHIVED"
        or payload.get("canonical_sha256") != canonical_sha256(payload)
        or not rows
        or "failure_evidence.json" not in relatives
        or "active_raw_packet_state.json" not in relatives
        or len(relatives) != len(set(relatives))
    ):
        raise RuntimeError("failed-packet archive identity mismatch")
    for row in rows:
        if set(row) != {"relative_path", "bytes", "sha256"}:
            raise RuntimeError("failed-packet archive row fields mismatch")
        relative = Path(str(row["relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("unsafe failed-packet archive relative path")
        archived = root.resolve() / relative
        if (
            not archived.is_file()
            or archived.is_symlink()
            or archived.stat().st_size != int(row["bytes"])
            or sha256_file(archived) != row["sha256"]
        ):
            raise RuntimeError(f"failed-packet archive changed: {archived}")
    return payload


def _copy(source: Path, destination: Path) -> None:
    source = source.resolve()
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as input_handle, destination.open("xb") as output_handle:
        shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
    os.chmod(destination, 0o444)
    if sha256_file(destination) != sha256_file(source):
        raise RuntimeError(f"failed-packet archive copy mismatch: {source}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument(
        "--method-id", choices=("3dgs_original", "citygs_x", "metrogs"), required=True
    )
    parser.add_argument("--track", choices=("gcp", "lidar"), required=True)
    parser.add_argument("--failure-evidence", type=Path, required=True)
    parser.add_argument("--packet-set-root", type=Path, required=True)
    parser.add_argument("--packet-state", type=Path, required=True)
    parser.add_argument("--global-packet-state", type=Path, required=True)
    parser.add_argument("--cleanup-root", type=Path, required=True)
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
    validate_addendum_runtime(
        activation=activation,
        candidate=candidate,
        registry=registry,
        executing_file=Path(__file__),
    )
    methods = {str(row["method_id"]): row for row in registry.get("methods", [])}
    if args.method_id not in registry.get("ready_method_ids", []) or args.method_id not in methods:
        raise RuntimeError("failed-packet cleanup method is not activated READY")
    if args.track == "gcp" and args.method_id == "3dgs_original":
        raise RuntimeError("legacy 3DGS has no new GCP raw packet")
    method = methods[args.method_id]
    recipe_path = Path(str(method["recipe_path"])).resolve()
    recipe = require_json(recipe_path, str(method["recipe_sha256"]))
    runtime_root = Path(str(candidate["candidate_output_root"])).resolve().parent
    packet_root = args.packet_set_root.resolve()
    state_path = args.packet_state.resolve()
    global_state_path = args.global_packet_state.resolve()
    cleanup_root = args.cleanup_root.resolve()
    if args.track == "gcp":
        expected_packet = runtime_root / "gcp-packet-scratch" / args.method_id
        expected_state = runtime_root / "gcp-packet-scratch" / "ACTIVE_GCP_PACKET_STATE.json"
        allowed_failures = {
            runtime_root / "gcp-packet-evidence" / args.method_id / "failure.json",
            runtime_root / "gcp-execution" / args.method_id / "failure.json",
        }
    else:
        expected_packet = Path(str(recipe["authorized_packet_set_root"])).resolve()
        expected_state = Path(str(recipe["authorized_packet_state"])).resolve()
        allowed_failures = {
            runtime_root / "lidar-packet-dispatch" / args.method_id / "failure.json",
            runtime_root / "lidar-execution" / args.method_id / "failure.json",
        }
    expected_cleanup = runtime_root / "failed-packet-cleanup" / args.track / args.method_id
    failure_path = args.failure_evidence.resolve()
    if (
        packet_root != expected_packet
        or state_path != expected_state
        or global_state_path != active_raw_packet_state_path(candidate)
        or cleanup_root != expected_cleanup
        or failure_path not in allowed_failures
    ):
        raise RuntimeError("failed-packet cleanup path differs from activated namespace")
    intent_path = cleanup_root / "cleanup_intent.json"
    receipt_path = cleanup_root / "cleanup_receipt.json"
    if receipt_path.exists() or receipt_path.is_symlink():
        raise FileExistsError(receipt_path)
    if intent_path.is_file():
        intent = require_json(intent_path)
        archive_manifest = Path(str(intent.get("failure_archive_manifest_path", ""))).resolve()
        if (
            intent.get("schema") != "m3m_gcp_100k_failed_packet_cleanup_intent_v1"
            or intent.get("status") != "AUTHORIZED_TO_DELETE_FAILED_RAW_PACKET_ONLY"
            or intent.get("ranking_status") != "INCOMPLETE_UNRANKED"
            or intent.get("scene") != SCENE
            or intent.get("method_id") != args.method_id
            or intent.get("track") != args.track
            or intent.get("three_track_activation_sha256") != sha256_file(activation_path)
            or intent.get("scene_attempt_freeze_sha256")
            != candidate["scene_attempt_freeze"]["sha256"]
            or intent.get("methods_manifest_sha256")
            != candidate["methods_manifest"]["sha256"]
            or intent.get("attempt_model_identity_sha256")
            != method["attempt_model_identity_sha256"]
            or intent.get("recipe_sha256") != sha256_file(recipe_path)
            or intent.get("failure_evidence_path") != str(failure_path)
            or intent.get("failure_evidence_sha256") != sha256_file(failure_path)
            or intent.get("packet_set_root") != str(packet_root)
            or intent.get("track_packet_state_path") != str(state_path)
            or intent.get("global_raw_packet_state_path") != str(global_state_path)
            or intent.get("authorized_targets_exact")
            != [str(packet_root), str(state_path), str(global_state_path)]
            or intent.get("retry_forbidden") is not True
            or intent.get("failure_archive_manifest_sha256")
            != sha256_file(archive_manifest)
            or intent.get("canonical_sha256") != canonical_sha256(intent)
        ):
            raise RuntimeError("existing failed-packet cleanup intent mismatch")
        validate_failure_archive(archive_manifest, archive_manifest.parent)
        if not global_state_path.exists() and (
            packet_root.exists()
            or packet_root.is_symlink()
            or state_path.exists()
            or state_path.is_symlink()
        ):
            raise RuntimeError("failed cleanup continuation lost the global mutex early")
        if global_state_path.exists() or global_state_path.is_symlink():
            validate_active_raw_packet_state(
                global_state_path,
                activation_path=activation_path,
                candidate=candidate,
                method_id=args.method_id,
                track=args.track,
                recipe_sha256=sha256_file(recipe_path),
                attempt_model_identity_sha256=method["attempt_model_identity_sha256"],
                packet_set_root=packet_root,
                track_packet_state_path=state_path,
            )
            if sha256_file(global_state_path) != intent.get(
                "global_raw_packet_state_sha256"
            ):
                raise RuntimeError("failed cleanup continuation global mutex mismatch")
        if state_path.exists() or state_path.is_symlink():
            if (
                state_path.is_symlink()
                or not state_path.is_file()
                or sha256_file(state_path) != intent.get("track_packet_state_sha256")
            ):
                raise RuntimeError("failed cleanup continuation track state mismatch")
        if packet_root.exists() or packet_root.is_symlink():
            if packet_root.is_symlink() or any(
                path.is_symlink() for path in packet_root.rglob("*")
            ):
                raise RuntimeError("failed cleanup continuation refuses packet symlinks")
            original_rows = {
                str(row["path"]): row
                for row in intent.get("packet_tree_before_cleanup", {}).get("files", [])
            }
            current_rows = directory_tree_hash(packet_root).get("files", [])
            if any(
                str(row["path"]) not in original_rows
                or row.get("sha256") != original_rows[str(row["path"])].get("sha256")
                or row.get("bytes") != original_rows[str(row["path"])].get("bytes")
                for row in current_rows
            ):
                raise RuntimeError("failed cleanup continuation packet subset mismatch")
            shutil.rmtree(packet_root)
        if state_path.exists():
            state_path.unlink()
        if global_state_path.exists():
            global_state_path.unlink()
        if any(
            path.exists() or path.is_symlink()
            for path in (packet_root, state_path, global_state_path)
        ):
            raise RuntimeError("failed cleanup continuation deletion postcondition failed")
        receipt: dict[str, Any] = {
            **intent,
            "schema": "m3m_gcp_100k_failed_packet_cleanup_receipt_v1",
            "status": "PASS_FAILED_RAW_PACKET_DELETED_INCOMPLETE_UNRANKED",
            "cleanup_intent_path": str(intent_path),
            "cleanup_intent_sha256": sha256_file(intent_path),
            "packet_set_root_absent": True,
            "track_packet_state_absent": True,
            "global_raw_packet_state_absent": True,
            "recovered_after_cleanup_interruption": True,
        }
        receipt.pop("canonical_sha256", None)
        receipt["canonical_sha256"] = canonical_sha256(receipt)
        write_exclusive(receipt_path, receipt)
        print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if cleanup_root.exists() or cleanup_root.is_symlink():
        raise FileExistsError(cleanup_root)
    global_state = validate_active_raw_packet_state(
        global_state_path,
        activation_path=activation_path,
        candidate=candidate,
        method_id=args.method_id,
        track=args.track,
        recipe_sha256=sha256_file(recipe_path),
        attempt_model_identity_sha256=method["attempt_model_identity_sha256"],
        packet_set_root=packet_root,
        track_packet_state_path=state_path,
    )
    failure = require_json(failure_path)
    allowed_schemas = {
        "gcp": {
            "m3m_gcp_100k_gcp_packet_failure_v1",
            "m3m_gcp_100k_gcp_evaluation_failure_v1",
        },
        "lidar": {
            "m3m_gcp_100k_lidar_packet_dispatch_failure_v1",
            "m3m_gcp_100k_lidar_evaluation_failure_v1",
        },
    }
    retry_forbidden = bool(
        failure.get("retry_forbidden_after_child_start")
        or failure.get("retry_forbidden_after_base_guard_child_start")
        or failure.get("retry_forbidden_after_evaluator_or_verifier_child_start")
        or failure.get("retry_forbidden_after_export_child_start")
    )
    if (
        failure.get("schema") not in allowed_schemas[args.track]
        or failure.get("status")
        not in {"OOM_UNRANKED", "FAILED_UNRANKED", "INCOMPLETE_UNRANKED"}
        or failure.get("scene") != SCENE
        or failure.get("method_id") != args.method_id
        or failure.get("three_track_activation_sha256") != sha256_file(activation_path)
        or failure.get("global_raw_packet_state_sha256")
        != sha256_file(global_state_path)
        or failure.get("canonical_sha256") != canonical_sha256(failure)
        or not retry_forbidden
    ):
        raise RuntimeError("failed-packet cleanup evidence is not immutable/no-retry")
    for target in (packet_root, state_path, global_state_path):
        if target.is_symlink():
            raise RuntimeError(f"failed-packet cleanup refuses symlinked target: {target}")
    if packet_root.exists() and any(path.is_symlink() for path in packet_root.rglob("*")):
        raise RuntimeError("failed-packet cleanup refuses symlinks inside packet root")

    cleanup_root.mkdir(parents=True, exist_ok=False)
    archive_root = cleanup_root / "failure-archive"
    archive_root.mkdir()
    copies = {
        "failure_evidence.json": failure_path,
        "active_raw_packet_state.json": global_state_path,
    }
    if state_path.is_file():
        copies["track_packet_state.json"] = state_path
    packet_manifest = packet_root / "depth_export_manifest.json"
    if packet_manifest.is_file() and not packet_manifest.is_symlink():
        copies["depth_export_manifest.json"] = packet_manifest
    for index, row in enumerate(failure.get("logs", [])):
        log_path = Path(str(row.get("path", ""))).resolve()
        if (
            log_path.is_file()
            and not log_path.is_symlink()
            and log_path.stat().st_size == row.get("bytes")
            and sha256_file(log_path) == row.get("sha256")
        ):
            copies[f"logs/{index:02d}_{log_path.name}"] = log_path
    for relative, source in copies.items():
        _copy(source, archive_root / relative)
    archive_payload: dict[str, Any] = {
        "schema": "m3m_gcp_100k_failed_packet_lightweight_archive_v1",
        "status": "PASS_FAILURE_EVIDENCE_ARCHIVED",
        "scene": SCENE,
        "method_id": args.method_id,
        "track": args.track,
        "failure_evidence_sha256": sha256_file(failure_path),
        "global_raw_packet_state_sha256": sha256_file(global_state_path),
        "files": [
            {
                "relative_path": path.relative_to(archive_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(archive_root.rglob("*"))
            if path.is_file()
        ],
    }
    archive_payload["canonical_sha256"] = canonical_sha256(archive_payload)
    archive_manifest = archive_root / "archive_manifest.json"
    write_exclusive(archive_manifest, archive_payload)
    validate_failure_archive(archive_manifest, archive_root)

    packet_tree = directory_tree_hash(packet_root)
    intent: dict[str, Any] = {
        "schema": "m3m_gcp_100k_failed_packet_cleanup_intent_v1",
        "status": "AUTHORIZED_TO_DELETE_FAILED_RAW_PACKET_ONLY",
        "ranking_status": "INCOMPLETE_UNRANKED",
        "scene": SCENE,
        "method_id": args.method_id,
        "track": args.track,
        "three_track_activation_sha256": sha256_file(activation_path),
        "scene_attempt_freeze_sha256": candidate["scene_attempt_freeze"]["sha256"],
        "methods_manifest_sha256": candidate["methods_manifest"]["sha256"],
        "attempt_model_identity_sha256": method["attempt_model_identity_sha256"],
        "recipe_sha256": sha256_file(recipe_path),
        "failure_evidence_path": str(failure_path),
        "failure_evidence_sha256": sha256_file(failure_path),
        "failure_archive_manifest_path": str(archive_manifest),
        "failure_archive_manifest_sha256": sha256_file(archive_manifest),
        "packet_set_root": str(packet_root),
        "packet_tree_before_cleanup": packet_tree,
        "track_packet_state_path": str(state_path),
        "track_packet_state_sha256": sha256_file(state_path) if state_path.is_file() else None,
        "global_raw_packet_state_path": str(global_state_path),
        "global_raw_packet_state_sha256": sha256_file(global_state_path),
        "global_raw_packet_state_canonical_sha256": global_state["canonical_sha256"],
        "authorized_targets_exact": [str(packet_root), str(state_path), str(global_state_path)],
        "retry_forbidden": True,
    }
    intent["canonical_sha256"] = canonical_sha256(intent)
    write_exclusive(intent_path, intent)
    if packet_root.exists():
        shutil.rmtree(packet_root)
    if state_path.exists():
        state_path.unlink()
    global_state_path.unlink()
    if any(path.exists() or path.is_symlink() for path in (packet_root, state_path, global_state_path)):
        raise RuntimeError("failed-packet cleanup deletion postcondition failed")
    receipt: dict[str, Any] = {
        **intent,
        "schema": "m3m_gcp_100k_failed_packet_cleanup_receipt_v1",
        "status": "PASS_FAILED_RAW_PACKET_DELETED_INCOMPLETE_UNRANKED",
        "cleanup_intent_path": str(intent_path),
        "cleanup_intent_sha256": sha256_file(intent_path),
        "packet_set_root_absent": True,
        "track_packet_state_absent": True,
        "global_raw_packet_state_absent": True,
        "recovered_after_cleanup_interruption": False,
    }
    receipt.pop("canonical_sha256", None)
    receipt["canonical_sha256"] = canonical_sha256(receipt)
    write_exclusive(receipt_path, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
