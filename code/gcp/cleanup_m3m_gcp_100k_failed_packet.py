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
ALLOWED_FAILURE_SCHEMAS = {
    "gcp": {
        "m3m_gcp_100k_gcp_packet_failure_v1",
        "m3m_gcp_100k_gcp_evaluation_failure_v1",
    },
    "lidar": {
        "m3m_gcp_100k_lidar_packet_dispatch_failure_v1",
        "m3m_gcp_100k_lidar_evaluation_failure_v1",
    },
}
FAILURE_ARCHIVE_FIELDS = {
    "schema",
    "status",
    "scene",
    "method_id",
    "track",
    "three_track_activation_sha256",
    "failure_evidence_sha256",
    "global_raw_packet_state_sha256",
    "declared_log_count",
    "declared_log_archive_map",
    "files",
    "canonical_sha256",
}


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


def declared_failure_log_archive_map(failure: dict[str, Any]) -> list[dict[str, Any]]:
    logs = failure.get("logs")
    if not isinstance(logs, list):
        raise RuntimeError("failure evidence logs must be an explicit list")
    result: list[dict[str, Any]] = []
    source_paths: set[str] = set()
    for index, row in enumerate(logs):
        if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}:
            raise RuntimeError("failure evidence log row fields mismatch")
        source_text = str(row["path"])
        source = Path(source_text)
        if (
            not source.is_absolute()
            or not source.name
            or source_text in source_paths
            or not isinstance(row["bytes"], int)
            or row["bytes"] < 0
            or not isinstance(row["sha256"], str)
            or len(row["sha256"]) != 64
        ):
            raise RuntimeError("failure evidence log identity mismatch")
        source_paths.add(source_text)
        result.append(
            {
                "source_path": source_text,
                "relative_path": f"logs/{index:02d}_{source.name}",
                "bytes": row["bytes"],
                "sha256": row["sha256"],
            }
        )
    return result


def validated_declared_failure_log_sources(
    failure: dict[str, Any],
) -> dict[str, Path]:
    copies: dict[str, Path] = {}
    for row in declared_failure_log_archive_map(failure):
        source = Path(str(row["source_path"])).resolve()
        if (
            not source.is_file()
            or source.is_symlink()
            or source.stat().st_size != row["bytes"]
            or sha256_file(source) != row["sha256"]
        ):
            raise RuntimeError(f"declared failure log is missing or changed: {source}")
        copies[str(row["relative_path"])] = source
    return copies


def validate_failure_archive(
    path: Path,
    root: Path,
    *,
    expected_scene: str,
    expected_method_id: str,
    expected_track: str,
    expected_activation_sha256: str,
    expected_failure_evidence_sha256: str,
    expected_global_state_sha256: str,
) -> dict[str, Any]:
    root = root.resolve()
    payload = require_json(path)
    rows = payload.get("files", [])
    relatives = [str(row.get("relative_path", "")) for row in rows]
    if (
        path.resolve() != root / "archive_manifest.json"
        or set(payload) != FAILURE_ARCHIVE_FIELDS
        or payload.get("schema")
        != "m3m_gcp_100k_failed_packet_lightweight_archive_v1"
        or payload.get("status") != "PASS_FAILURE_EVIDENCE_ARCHIVED"
        or payload.get("scene") != expected_scene
        or payload.get("method_id") != expected_method_id
        or payload.get("track") != expected_track
        or payload.get("three_track_activation_sha256")
        != expected_activation_sha256
        or payload.get("failure_evidence_sha256")
        != expected_failure_evidence_sha256
        or payload.get("global_raw_packet_state_sha256")
        != expected_global_state_sha256
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
        archived = root / relative
        if (
            not archived.is_file()
            or archived.is_symlink()
            or archived.stat().st_size != int(row["bytes"])
            or sha256_file(archived) != row["sha256"]
        ):
            raise RuntimeError(f"failed-packet archive changed: {archived}")
    actual_relatives = {
        item.relative_to(root).as_posix()
        for item in root.rglob("*")
        if item.is_file() and item.resolve() != path.resolve()
    }
    if actual_relatives != set(relatives):
        raise RuntimeError("failed-packet archive inventory is not exact")
    by_relative = {str(row["relative_path"]): row for row in rows}
    if (
        by_relative["failure_evidence.json"]["sha256"]
        != payload["failure_evidence_sha256"]
        or by_relative["active_raw_packet_state.json"]["sha256"]
        != payload["global_raw_packet_state_sha256"]
    ):
        raise RuntimeError("failed-packet archive primary-file SHA binding mismatch")
    archived_failure = require_json(
        root / "failure_evidence.json", payload["failure_evidence_sha256"]
    )
    archived_global = require_json(
        root / "active_raw_packet_state.json",
        payload["global_raw_packet_state_sha256"],
    )
    retry_forbidden = bool(
        archived_failure.get("retry_forbidden_after_child_start")
        or archived_failure.get("retry_forbidden_after_base_guard_child_start")
        or archived_failure.get("retry_forbidden_after_evaluator_or_verifier_child_start")
        or archived_failure.get("retry_forbidden_after_export_child_start")
    )
    if (
        archived_failure.get("schema") not in ALLOWED_FAILURE_SCHEMAS[expected_track]
        or archived_failure.get("status")
        not in {"OOM_UNRANKED", "FAILED_UNRANKED", "INCOMPLETE_UNRANKED"}
        or archived_failure.get("scene") != expected_scene
        or archived_failure.get("method_id") != expected_method_id
        or archived_failure.get("three_track_activation_sha256")
        != expected_activation_sha256
        or archived_failure.get("global_raw_packet_state_sha256")
        != expected_global_state_sha256
        or archived_failure.get("canonical_sha256") != canonical_sha256(archived_failure)
        or not retry_forbidden
        or archived_global.get("schema") != "m3m_gcp_100k_active_raw_packet_state_v1"
        or archived_global.get("status") != "ACTIVE_EXCLUSIVE_RAW_PACKET"
        or archived_global.get("scene") != expected_scene
        or archived_global.get("method_id") != expected_method_id
        or archived_global.get("track") != expected_track
        or archived_global.get("three_track_activation_sha256")
        != expected_activation_sha256
        or archived_global.get("canonical_sha256") != canonical_sha256(archived_global)
    ):
        raise RuntimeError("failed-packet archive semantic identity mismatch")
    expected_logs = declared_failure_log_archive_map(archived_failure)
    expected_log_relatives = {str(row["relative_path"]) for row in expected_logs}
    allowed_relatives = {
        "failure_evidence.json",
        "active_raw_packet_state.json",
        "track_packet_state.json",
        "depth_export_manifest.json",
        *expected_log_relatives,
    }
    if (
        payload.get("declared_log_count") != len(expected_logs)
        or payload.get("declared_log_archive_map") != expected_logs
        or not set(relatives).issubset(allowed_relatives)
        or {relative for relative in relatives if relative.startswith("logs/")}
        != expected_log_relatives
    ):
        raise RuntimeError("failed-packet archive declared-log coverage mismatch")
    for log in expected_logs:
        row = by_relative.get(str(log["relative_path"]), {})
        if row.get("bytes") != log["bytes"] or row.get("sha256") != log["sha256"]:
            raise RuntimeError("failed-packet archived log identity mismatch")
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
        validate_failure_archive(
            archive_manifest,
            archive_manifest.parent,
            expected_scene=SCENE,
            expected_method_id=args.method_id,
            expected_track=args.track,
            expected_activation_sha256=sha256_file(activation_path),
            expected_failure_evidence_sha256=str(
                intent["failure_evidence_sha256"]
            ),
            expected_global_state_sha256=str(
                intent["global_raw_packet_state_sha256"]
            ),
        )
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
    retry_forbidden = bool(
        failure.get("retry_forbidden_after_child_start")
        or failure.get("retry_forbidden_after_base_guard_child_start")
        or failure.get("retry_forbidden_after_evaluator_or_verifier_child_start")
        or failure.get("retry_forbidden_after_export_child_start")
    )
    if (
        failure.get("schema") not in ALLOWED_FAILURE_SCHEMAS[args.track]
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
    declared_log_copies = validated_declared_failure_log_sources(failure)

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
    copies.update(declared_log_copies)
    for relative, source in copies.items():
        _copy(source, archive_root / relative)
    archive_payload: dict[str, Any] = {
        "schema": "m3m_gcp_100k_failed_packet_lightweight_archive_v1",
        "status": "PASS_FAILURE_EVIDENCE_ARCHIVED",
        "scene": SCENE,
        "method_id": args.method_id,
        "track": args.track,
        "three_track_activation_sha256": sha256_file(activation_path),
        "failure_evidence_sha256": sha256_file(failure_path),
        "global_raw_packet_state_sha256": sha256_file(global_state_path),
        "declared_log_count": len(declared_log_copies),
        "declared_log_archive_map": declared_failure_log_archive_map(failure),
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
    validate_failure_archive(
        archive_manifest,
        archive_root,
        expected_scene=SCENE,
        expected_method_id=args.method_id,
        expected_track=args.track,
        expected_activation_sha256=sha256_file(activation_path),
        expected_failure_evidence_sha256=sha256_file(failure_path),
        expected_global_state_sha256=sha256_file(global_state_path),
    )

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
