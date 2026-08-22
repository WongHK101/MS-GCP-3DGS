#!/usr/bin/env python3
"""Validate both GCP gates, delete one exact 211-view packet set, and seal a receipt."""

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
from metric_depth_packet import directory_tree_hash
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


def validate_archive(
    path: Path, expected_root: Path, *, require_sources: bool = True
) -> dict[str, Any]:
    payload = require_json(path)
    expected_relatives = {
        "gcp_execution_authorization.json",
        "gcp_packet_phase_success.json",
        "gcp_packet_state.json",
        "active_raw_packet_state.json",
        "depth_export_manifest.json",
        "gcp_evaluation_execution_receipt.json",
        "evaluation/observation_samples.csv",
        "evaluation/point_results.csv",
        "evaluation/evaluation_summary.json",
        "evaluation/evaluator_manifest.json",
        "independent_verification.json",
    }
    rows = payload.get("files", [])
    if (
        path.resolve() != expected_root / "archive_manifest.json"
        or payload.get("schema") != "m3m_gcp_100k_gcp_lightweight_archive_v1"
        or payload.get("status") != "PASS_GCP_LIGHTWEIGHT_ARCHIVE_BYTE_VERIFIED"
        or Path(str(payload.get("archive_root", ""))).resolve() != expected_root
        or payload.get("canonical_sha256") != canonical_sha256(payload)
        or payload.get("source_and_archive_bytes_reverified") is not True
        or payload.get("raw_metric_depth_packet_files_archived") is not False
        or not isinstance(rows, list)
        or len(rows) != len(expected_relatives)
    ):
        raise RuntimeError("GCP lightweight archive identity mismatch")
    observed_relatives: set[str] = set()
    for row in rows:
        archived = Path(str(row.get("archive_path", ""))).resolve()
        try:
            archived.relative_to(expected_root)
        except ValueError as exc:
            raise RuntimeError("GCP archive file escapes archive root") from exc
        relative = archived.relative_to(expected_root).as_posix()
        observed_relatives.add(relative)
        source = Path(str(row.get("source_path", ""))).resolve()
        if (
            not archived.is_file()
            or archived.is_symlink()
            or archived.stat().st_size != row.get("bytes")
            or sha256_file(archived) != row.get("sha256")
        ):
            raise RuntimeError(f"GCP archive file changed: {archived}")
        if require_sources and (
            not source.is_file()
            or source.is_symlink()
            or source.stat().st_size != row.get("bytes")
            or sha256_file(source) != row.get("sha256")
        ):
            raise RuntimeError(f"GCP archive source file changed: {source}")
    if observed_relatives != expected_relatives:
        raise RuntimeError("GCP archive exact file inventory mismatch")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--method-id", choices=("citygs_x", "metrogs"), required=True)
    parser.add_argument("--packet-set-root", type=Path, required=True)
    parser.add_argument("--packet-state", type=Path, required=True)
    parser.add_argument("--global-packet-state", type=Path, required=True)
    parser.add_argument("--gcp-packet-phase-success", type=Path, required=True)
    parser.add_argument("--gcp-authorization", type=Path, required=True)
    parser.add_argument("--gcp-evaluation-root", type=Path, required=True)
    parser.add_argument("--gcp-verification", type=Path, required=True)
    parser.add_argument("--gcp-archive-manifest", type=Path, required=True)
    parser.add_argument("--release-intent", type=Path, required=True)
    parser.add_argument("--deletion-receipt", type=Path, required=True)
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
    methods = {str(row["method_id"]): row for row in registry.get("methods", [])}
    if args.method_id not in registry.get("ready_method_ids", []) or args.method_id not in methods:
        raise RuntimeError("GCP release method is not activated READY")
    method = methods[args.method_id]
    _addendum_repo, addendum_config = validate_addendum_runtime(
        activation=activation,
        candidate=candidate,
        registry=registry,
        executing_file=Path(__file__),
    )

    base_repo = Path(str(candidate["base_checkout"]["path"])).resolve()
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
        raise RuntimeError("GCP release frozen model identity mismatch")

    runtime_root = Path(str(candidate["candidate_output_root"])).resolve().parent
    packet_root = args.packet_set_root.resolve()
    state_path = args.packet_state.resolve()
    global_state_path = args.global_packet_state.resolve()
    phase_path = args.gcp_packet_phase_success.resolve()
    authorization_path = args.gcp_authorization.resolve()
    evaluation_root = args.gcp_evaluation_root.resolve()
    verification_path = args.gcp_verification.resolve()
    archive_path = args.gcp_archive_manifest.resolve()
    intent_path = args.release_intent.resolve()
    receipt_path = args.deletion_receipt.resolve()
    expected_packet_root = runtime_root / "gcp-packet-scratch" / args.method_id
    expected_state = runtime_root / "gcp-packet-scratch" / "ACTIVE_GCP_PACKET_STATE.json"
    expected_global_state = active_raw_packet_state_path(candidate)
    expected_eval = Path(str(candidate["formal_results_root"])).resolve() / "gcp" / args.method_id
    expected_archive_root = (
        Path(str(candidate["formal_results_root"])).resolve()
        / "gcp-lightweight-archives"
        / args.method_id
    )
    expected_release_root = runtime_root / "gcp-packet-release" / args.method_id
    summary_path = evaluation_root / "evaluation_summary.json"
    archived_execution_receipt_path = (
        expected_archive_root / "gcp_evaluation_execution_receipt.json"
    )
    if (
        packet_root != expected_packet_root
        or state_path != expected_state
        or global_state_path != expected_global_state
        or evaluation_root != expected_eval
        or archive_path.parent != expected_archive_root
        or intent_path != expected_release_root / "release_intent.json"
        or receipt_path != expected_release_root / "deletion_receipt.json"
    ):
        raise RuntimeError("GCP release path differs from activated namespace")
    if receipt_path.exists() or receipt_path.is_symlink():
        raise FileExistsError(receipt_path)

    # Crash-safe completion: an immutable intent plus all three targets absent is sufficient
    # to seal the receipt after a process interruption between deletion and receipt write.
    if intent_path.is_file():
        intent = require_json(intent_path)
        if (
            intent.get("schema") != "m3m_gcp_100k_gcp_packet_release_intent_v1"
            or intent.get("status") != "AUTHORIZED_TO_DELETE_EXACT_GCP_PACKET"
            or intent.get("method_id") != args.method_id
            or intent.get("three_track_activation_sha256") != sha256_file(activation_path)
            or intent.get("candidate_manifest_sha256") != sha256_file(candidate_path)
            or intent.get("scene_attempt_freeze_sha256")
            != candidate["scene_attempt_freeze"]["sha256"]
            or intent.get("methods_manifest_sha256")
            != candidate["methods_manifest"]["sha256"]
            or intent.get("recipe_sha256") != sha256_file(recipe_path)
            or intent.get("attempt_model_identity_sha256") != sha256_file(identity_path)
            or intent.get("packet_set_root") != str(packet_root)
            or intent.get("packet_state_path") != str(state_path)
            or intent.get("global_raw_packet_state_path") != str(global_state_path)
            or intent.get("authorized_targets_exact")
            != [str(packet_root), str(state_path), str(global_state_path)]
            or intent.get("gcp_packet_phase_success_path") != str(phase_path)
            or intent.get("gcp_packet_phase_success_sha256") != sha256_file(phase_path)
            or intent.get("gcp_authorization_path") != str(authorization_path)
            or intent.get("gcp_authorization_sha256") != sha256_file(authorization_path)
            or intent.get("gcp_evaluation_summary_path") != str(summary_path)
            or intent.get("gcp_evaluation_summary_sha256") != sha256_file(summary_path)
            or intent.get("gcp_verification_path") != str(verification_path)
            or intent.get("gcp_verification_sha256") != sha256_file(verification_path)
            or intent.get("gcp_archive_manifest_path") != str(archive_path)
            or intent.get("gcp_archive_manifest_sha256") != sha256_file(archive_path)
            or intent.get("gcp_execution_receipt_path")
            != str(archived_execution_receipt_path)
            or intent.get("gcp_execution_receipt_sha256")
            != sha256_file(archived_execution_receipt_path)
            or intent.get("canonical_sha256") != canonical_sha256(intent)
        ):
            raise RuntimeError("existing GCP release intent mismatch")
        if not global_state_path.exists() and (
            packet_root.exists()
            or packet_root.is_symlink()
            or state_path.exists()
            or state_path.is_symlink()
        ):
            raise RuntimeError("GCP release continuation lost the global mutex early")
        if global_state_path.exists() or global_state_path.is_symlink():
            validate_active_raw_packet_state(
                global_state_path,
                activation_path=activation_path,
                candidate=candidate,
                method_id=args.method_id,
                track="gcp",
                recipe_sha256=sha256_file(recipe_path),
                attempt_model_identity_sha256=sha256_file(identity_path),
                packet_set_root=packet_root,
                track_packet_state_path=state_path,
            )
            if sha256_file(global_state_path) != intent.get(
                "global_raw_packet_state_sha256"
            ):
                raise RuntimeError("GCP release continuation global mutex SHA mismatch")
        validate_archive(
            archive_path,
            expected_archive_root,
            require_sources=False,
        )
        recovery_summary = require_json(summary_path)
        recovery_verification = require_json(verification_path)
        recovery_execution_receipt = require_json(archived_execution_receipt_path)
        if (
            recovery_summary.get("scene") != SCENE
            or recovery_summary.get("method_id") != args.method_id
            or recovery_summary.get("status")
            not in {"COMPLETE_RANKED", "INCOMPLETE_UNRANKED"}
            or recovery_verification.get("status") != "PASS"
            or recovery_verification.get("passed") is not True
            or recovery_verification.get("scene") != SCENE
            or recovery_verification.get("method_id") != args.method_id
            or recovery_verification.get("ranking_status")
            != recovery_summary.get("status")
            or recovery_execution_receipt.get("status")
            != "PASS_GCP_EVALUATOR_AND_INDEPENDENT_VERIFIER"
            or recovery_execution_receipt.get("canonical_sha256")
            != canonical_sha256(recovery_execution_receipt)
        ):
            raise RuntimeError("existing GCP release intent result/archive gate mismatch")
        if packet_root.exists() or packet_root.is_symlink():
            if packet_root.is_symlink():
                raise RuntimeError("GCP release continuation refuses symlinked packet root")
            if any(path.is_symlink() for path in packet_root.rglob("*")):
                raise RuntimeError("GCP release continuation packet identity mismatch")
            original_rows = {
                str(row["path"]): row
                for row in intent.get("packet_tree_before_delete", {}).get("files", [])
            }
            current_rows = directory_tree_hash(packet_root).get("files", [])
            if any(
                str(row["path"]) not in original_rows
                or row.get("sha256") != original_rows[str(row["path"])].get("sha256")
                or row.get("bytes") != original_rows[str(row["path"])].get("bytes")
                for row in current_rows
            ):
                raise RuntimeError("GCP release continuation packet subset mismatch")
            shutil.rmtree(packet_root)
        if state_path.exists() or state_path.is_symlink():
            if (
                state_path.is_symlink()
                or not state_path.is_file()
                or sha256_file(state_path) != intent.get("packet_state_sha256")
            ):
                raise RuntimeError("GCP release continuation state identity mismatch")
            state_path.unlink()
        if global_state_path.exists() or global_state_path.is_symlink():
            if global_state_path.is_symlink():
                raise RuntimeError("GCP release continuation refuses symlinked global mutex")
            global_state_path.unlink()
        if (
            packet_root.exists()
            or packet_root.is_symlink()
            or state_path.exists()
            or state_path.is_symlink()
            or global_state_path.exists()
            or global_state_path.is_symlink()
        ):
            raise RuntimeError("GCP release continuation deletion postcondition failed")
        receipt: dict[str, Any] = {
            **intent,
            "schema": "m3m_gcp_100k_gcp_packet_deletion_receipt_v1",
            "status": "PASS_GCP_PACKET_DELETED",
            "release_intent_path": str(intent_path),
            "release_intent_sha256": sha256_file(intent_path),
            "packet_set_root_absent": True,
            "packet_state_absent": True,
            "global_raw_packet_state_absent": True,
            "recovered_after_post_delete_interruption": True,
        }
        receipt.pop("canonical_sha256", None)
        receipt["canonical_sha256"] = canonical_sha256(receipt)
        write_exclusive(receipt_path, receipt)
        print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    for path in (
        packet_root,
        state_path,
        global_state_path,
        phase_path,
        authorization_path,
        verification_path,
        archive_path,
    ):
        if path.is_symlink():
            raise RuntimeError(f"GCP release refuses symlinked evidence/target: {path}")
    state = require_json(state_path)
    validate_active_raw_packet_state(
        global_state_path,
        activation_path=activation_path,
        candidate=candidate,
        method_id=args.method_id,
        track="gcp",
        recipe_sha256=sha256_file(recipe_path),
        attempt_model_identity_sha256=sha256_file(identity_path),
        packet_set_root=packet_root,
        track_packet_state_path=state_path,
    )
    phase = require_json(phase_path)
    authorization = require_json(authorization_path)
    gcp_config = addendum_config["tracks"]["gcp"]
    verification = require_json(verification_path)
    archive = validate_archive(archive_path, expected_archive_root)
    archived_execution_receipt = require_json(archived_execution_receipt_path)
    packet_manifest_path = packet_root / "depth_export_manifest.json"
    packet = require_json(packet_manifest_path)
    summary = require_json(summary_path)
    if args.method_id == "citygs_x":
        expected_model_content: Any = directory_tree_hash(
            (
                Path(str(method["model_root"])).resolve()
                / str(method["formal_model_relative_path"])
            ).resolve().parent
        )
    else:
        checkpoint = Path(str(method["formal_checkpoint"])).resolve()
        if sha256_file(checkpoint) != method["formal_model_sha256"]:
            raise RuntimeError("MetroGS release checkpoint identity mismatch")
        expected_model_content = method["formal_model_sha256"]
    if any(path.is_symlink() for path in packet_root.rglob("*")):
        raise RuntimeError("GCP release refuses a packet tree containing symlinks")
    if (
        state.get("method_id") != args.method_id
        or Path(str(state.get("packet_set_root", ""))).resolve() != packet_root
        or state.get("three_track_activation_sha256") != sha256_file(activation_path)
        or state.get("attempt_model_identity_sha256")
        != method["attempt_model_identity_sha256"]
        or state.get("canonical_sha256") != canonical_sha256(state)
        or phase.get("status") != "PASS_GCP_PACKET_211"
        or phase.get("method_id") != args.method_id
        or phase.get("packet_state_sha256") != sha256_file(state_path)
        or phase.get("attempt_model_identity_sha256")
        != method["attempt_model_identity_sha256"]
        or phase.get("canonical_sha256") != canonical_sha256(phase)
        or packet.get("rendered_view_count") != 211
        or len(packet.get("depth_index", [])) != 211
        or len(packet.get("packet_index", [])) != 211
        or packet.get("model_content_hash") != expected_model_content
        or authorization.get("method_id") != args.method_id
        or authorization.get("three_track_activation_sha256")
        != sha256_file(activation_path)
        or authorization.get("attempt_model_identity_sha256")
        != method["attempt_model_identity_sha256"]
        or authorization.get("expected_model_content") != expected_model_content
        or authorization.get("observed_packet_model_content") != expected_model_content
        or authorization.get("packet_manifest_sha256") != sha256_file(packet_manifest_path)
        or authorization.get("global_raw_packet_state_path") != str(global_state_path)
        or authorization.get("global_raw_packet_state_sha256")
        != sha256_file(global_state_path)
        or authorization.get("canonical_sha256") != canonical_sha256(authorization)
        or authorization.get("evaluator_sha256") != gcp_config["evaluator_sha256"]
        or authorization.get("verifier_sha256") != gcp_config["verifier_sha256"]
        or summary.get("packet_manifest_sha256") != sha256_file(packet_manifest_path)
        or verification.get("status") != "PASS"
        or verification.get("passed") is not True
        or verification.get("method_id") != args.method_id
        or archive.get("method_id") != args.method_id
        or archive.get("three_track_activation_sha256") != sha256_file(activation_path)
        or archive.get("scene_attempt_freeze_sha256")
        != candidate["scene_attempt_freeze"]["sha256"]
        or archive.get("methods_manifest_sha256") != candidate["methods_manifest"]["sha256"]
        or archive.get("packet_manifest_sha256") != sha256_file(packet_manifest_path)
        or archive.get("gcp_authorization_sha256") != sha256_file(authorization_path)
        or archive.get("gcp_packet_phase_success_sha256") != sha256_file(phase_path)
        or archive.get("packet_state_sha256") != sha256_file(state_path)
        or archive.get("global_raw_packet_state_sha256")
        != sha256_file(global_state_path)
        or archive.get("evaluation_summary_sha256") != sha256_file(summary_path)
        or archive.get("verification_sha256") != sha256_file(verification_path)
        or archive.get("gcp_execution_receipt_sha256")
        != sha256_file(archived_execution_receipt_path)
        or archived_execution_receipt.get("schema")
        != "m3m_gcp_100k_gcp_evaluation_execution_receipt_v1"
        or archived_execution_receipt.get("status")
        != "PASS_GCP_EVALUATOR_AND_INDEPENDENT_VERIFIER"
        or archived_execution_receipt.get("method_id") != args.method_id
        or archived_execution_receipt.get("three_track_activation_sha256")
        != sha256_file(activation_path)
        or archived_execution_receipt.get("gcp_authorization_sha256")
        != sha256_file(authorization_path)
        or archived_execution_receipt.get("packet_manifest_sha256")
        != sha256_file(packet_manifest_path)
        or archived_execution_receipt.get("global_raw_packet_state_sha256")
        != sha256_file(global_state_path)
        or archived_execution_receipt.get("summary_sha256") != sha256_file(summary_path)
        or archived_execution_receipt.get("verification_sha256")
        != sha256_file(verification_path)
        or archived_execution_receipt.get("canonical_sha256")
        != canonical_sha256(archived_execution_receipt)
    ):
        raise RuntimeError("GCP release dual-gate/current identity binding mismatch")

    file_count = sum(1 for path in packet_root.rglob("*") if path.is_file())
    byte_count = sum(path.stat().st_size for path in packet_root.rglob("*") if path.is_file())
    intent: dict[str, Any] = {
        "schema": "m3m_gcp_100k_gcp_packet_release_intent_v1",
        "status": "AUTHORIZED_TO_DELETE_EXACT_GCP_PACKET",
        "scene": SCENE,
        "method_id": args.method_id,
        "three_track_activation_path": str(activation_path),
        "three_track_activation_sha256": sha256_file(activation_path),
        "candidate_manifest_sha256": sha256_file(candidate_path),
        "scene_attempt_freeze_sha256": candidate["scene_attempt_freeze"]["sha256"],
        "methods_manifest_sha256": candidate["methods_manifest"]["sha256"],
        "recipe_sha256": sha256_file(recipe_path),
        "attempt_model_identity_sha256": sha256_file(identity_path),
        "packet_set_root": str(packet_root),
        "packet_state_path": str(state_path),
        "packet_state_sha256": sha256_file(state_path),
        "global_raw_packet_state_path": str(global_state_path),
        "global_raw_packet_state_sha256": sha256_file(global_state_path),
        "packet_manifest_sha256": sha256_file(packet_manifest_path),
        "packet_tree_before_delete": directory_tree_hash(packet_root),
        "packet_file_count_before_delete": file_count,
        "packet_bytes_before_delete": byte_count,
        "gcp_packet_phase_success_sha256": sha256_file(phase_path),
        "gcp_packet_phase_success_path": str(phase_path),
        "gcp_authorization_sha256": sha256_file(authorization_path),
        "gcp_authorization_path": str(authorization_path),
        "gcp_evaluation_summary_sha256": sha256_file(summary_path),
        "gcp_evaluation_summary_path": str(summary_path),
        "gcp_execution_receipt_sha256": sha256_file(archived_execution_receipt_path),
        "gcp_execution_receipt_path": str(archived_execution_receipt_path),
        "gcp_verification_sha256": sha256_file(verification_path),
        "gcp_verification_path": str(verification_path),
        "gcp_archive_manifest_sha256": sha256_file(archive_path),
        "gcp_archive_manifest_path": str(archive_path),
        "authorized_targets_exact": [
            str(packet_root),
            str(state_path),
            str(global_state_path),
        ],
    }
    intent["canonical_sha256"] = canonical_sha256(intent)
    write_exclusive(intent_path, intent)

    shutil.rmtree(packet_root)
    state_path.unlink()
    global_state_path.unlink()
    if (
        packet_root.exists()
        or packet_root.is_symlink()
        or state_path.exists()
        or state_path.is_symlink()
        or global_state_path.exists()
        or global_state_path.is_symlink()
    ):
        raise RuntimeError("GCP packet/state deletion postcondition failed")
    receipt: dict[str, Any] = {
        **intent,
        "schema": "m3m_gcp_100k_gcp_packet_deletion_receipt_v1",
        "status": "PASS_GCP_PACKET_DELETED",
        "release_intent_path": str(intent_path),
        "release_intent_sha256": sha256_file(intent_path),
        "packet_set_root_absent": True,
        "packet_state_absent": True,
        "global_raw_packet_state_absent": True,
        "recovered_after_post_delete_interruption": False,
    }
    receipt.pop("canonical_sha256", None)
    receipt["canonical_sha256"] = canonical_sha256(receipt)
    write_exclusive(receipt_path, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
