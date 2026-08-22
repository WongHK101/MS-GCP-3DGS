#!/usr/bin/env python3
"""Authorize base-plan packet deletion only after the added GCP and LiDAR gates pass."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from m3m_gcp_lidar_artifacts import canonical_sha256, sha256_file
from verify_m3m_gcp_lidar_formal_v1 import validate_archive_manifest


SCENE = "gcp_100000_20260610"


def require_json(path: Path, expected_sha: str | None = None) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
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
            (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
    finally:
        os.close(descriptor)


def candidate_from_activation(activation_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
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
    return activation, candidate


def validate_gcp_gate(
    *,
    method_id: str,
    activation: dict[str, Any],
    candidate: dict[str, Any],
    packet_sha: str,
    authorization_path: Path | None,
    summary_path: Path | None,
    verification_path: Path | None,
) -> dict[str, Any]:
    if method_id == "3dgs_original":
        row = candidate["legacy_3dgs_gcp_adoption"]
        path = Path(str(row["path"])).resolve()
        receipt = require_json(path, str(row["sha256"]))
        if (
            receipt.get("canonical_sha256") != row["canonical_sha256"]
            or canonical_sha256(receipt) != row["canonical_sha256"]
            or receipt.get("status") != "PASS_LEGACY_GCP_ADOPTION_CANDIDATE"
            or receipt.get("method_id") != method_id
            or receipt.get("scene_attempt_freeze_sha256") != candidate["scene_attempt_freeze"]["sha256"]
            or activation.get("legacy_3dgs_gcp_adoption_sha256") != sha256_file(path)
        ):
            raise RuntimeError("legacy 3DGS GCP adoption gate mismatch")
        return {
            "gate": "LEGACY_GCP_ADOPTION_PASS",
            "path": str(path),
            "sha256": sha256_file(path),
            "active_packet_manifest_sha256_required_to_match": False,
        }

    if authorization_path is None or summary_path is None or verification_path is None:
        raise RuntimeError("new GCP packet-release gate requires authorization, summary and verification")
    authorization_path = authorization_path.resolve()
    summary_path = summary_path.resolve()
    verification_path = verification_path.resolve()
    authorization = require_json(authorization_path)
    summary = require_json(summary_path)
    verification = require_json(verification_path)
    if (
        authorization.get("schema") != "m3m_gcp_100k_gcp_execution_authorization_v1"
        or authorization.get("status") != "ACTIVE_FROZEN"
        or authorization.get("execution_authorized") is not True
        or authorization.get("method_id") != method_id
        or authorization.get("scene") != SCENE
    ):
        raise RuntimeError("new GCP authorization mismatch")
    # File SHA, rather than canonical SHA, is the activation binding used by the authorization.
    activation_path = Path(str(authorization["three_track_activation_path"])).resolve()
    if (
        sha256_file(activation_path) != authorization.get("three_track_activation_sha256")
        or authorization.get("packet_manifest_sha256") != packet_sha
        or authorization.get("authorized_output_root") != str(summary_path.parent.resolve())
        or authorization.get("authorized_verification_output") != str(verification_path)
        or authorization.get("canonical_sha256") != canonical_sha256(authorization)
    ):
        raise RuntimeError("new GCP authorization file/packet/output binding mismatch")
    if (
        summary.get("scene") != SCENE
        or summary.get("method_id") != method_id
        or summary.get("protocol_id") != "m3m_gcp_native_quarter_geometry_v2"
        or summary.get("packet_manifest_sha256") != packet_sha
        or summary.get("status") not in {"COMPLETE_RANKED", "INCOMPLETE_UNRANKED"}
        or verification.get("schema")
        != "m3m_gcp_native_quarter_evaluator_output_independent_verification_v1"
        or verification.get("status") != "PASS"
        or verification.get("passed") is not True
        or verification.get("scene") != SCENE
        or verification.get("method_id") != method_id
        or verification.get("ranking_status") != summary.get("status")
        or verification.get("recomputed_residual_statistics") != summary.get("residual_statistics")
    ):
        raise RuntimeError("new GCP evaluator/verifier gate mismatch")
    return {
        "gate": "NEW_GCP_VERIFIER_PASS",
        "authorization_path": str(authorization_path),
        "authorization_sha256": sha256_file(authorization_path),
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "verification_path": str(verification_path),
        "verification_sha256": sha256_file(verification_path),
        "active_packet_manifest_sha256_required_to_match": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--method-id", required=True)
    parser.add_argument("--packet-manifest", type=Path, required=True)
    parser.add_argument("--packet-set-root", type=Path, required=True)
    parser.add_argument("--packet-state", type=Path, required=True)
    parser.add_argument("--gcp-authorization", type=Path)
    parser.add_argument("--gcp-summary", type=Path)
    parser.add_argument("--gcp-verification", type=Path)
    parser.add_argument("--lidar-method-result", type=Path, required=True)
    parser.add_argument("--lidar-verification", type=Path, required=True)
    parser.add_argument("--lidar-archive-manifest", type=Path, required=True)
    parser.add_argument("--lidar-archive-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    activation_path = args.activation.resolve()
    activation, candidate = candidate_from_activation(activation_path)
    registry_path = Path(str(candidate["rgb_registry"]["path"])).resolve()
    registry = require_json(registry_path, str(candidate["rgb_registry"]["sha256"]))
    if args.method_id not in registry.get("ready_method_ids", []):
        raise RuntimeError("packet-release method is not READY in the activated registry")

    packet_manifest_path = args.packet_manifest.resolve()
    packet_manifest = require_json(packet_manifest_path)
    packet_sha = sha256_file(packet_manifest_path)
    packet_set_root = args.packet_set_root.resolve()
    packet_state_path = args.packet_state.resolve()
    packet_state = require_json(packet_state_path)
    if (
        packet_manifest.get("scene") != SCENE
        or packet_manifest.get("rendered_view_count") != 2196
        or Path(str(packet_manifest.get("depth_output_dir", ""))).resolve() != packet_set_root
        or packet_state.get("method_id") != args.method_id
        or Path(str(packet_state.get("packet_set_root", ""))).resolve() != packet_set_root
    ):
        raise RuntimeError("active packet state/root/manifest mismatch")

    gcp_gate = validate_gcp_gate(
        method_id=args.method_id,
        activation=activation,
        candidate=candidate,
        packet_sha=packet_sha,
        authorization_path=args.gcp_authorization,
        summary_path=args.gcp_summary,
        verification_path=args.gcp_verification,
    )

    lidar_result_path = args.lidar_method_result.resolve()
    lidar_verification_path = args.lidar_verification.resolve()
    lidar_archive_path = args.lidar_archive_manifest.resolve()
    lidar_archive_root = args.lidar_archive_root.resolve()
    lidar_result = require_json(lidar_result_path)
    lidar_verification = require_json(lidar_verification_path)
    if (
        lidar_result.get("scene") != SCENE
        or lidar_result.get("method_id") != args.method_id
        or lidar_result.get("packet_manifest_sha256") != packet_sha
        or lidar_verification.get("schema") != "m3m_gcp_lidar_formal_verification_v1"
        or lidar_verification.get("status") != "PASS_VERIFIED_FORMAL_V1"
        or lidar_verification.get("scene") != SCENE
        or lidar_verification.get("method_id") != args.method_id
        or lidar_verification.get("method_result_sha256") != sha256_file(lidar_result_path)
        or lidar_verification.get("scene_attempt_freeze_sha256")
        != candidate["scene_attempt_freeze"]["sha256"]
        or lidar_verification.get("canonical_sha256") != canonical_sha256(lidar_verification)
    ):
        raise RuntimeError("LiDAR result/verifier gate mismatch")
    archive_errors = validate_archive_manifest(
        lidar_archive_path,
        lidar_archive_root,
        expected_scene_attempt_freeze_sha256=candidate["scene_attempt_freeze"]["sha256"],
    )
    archive = require_json(lidar_archive_path)
    if archive_errors or archive.get("method_id") != args.method_id or archive.get("scene") != SCENE:
        raise RuntimeError("LiDAR archive gate mismatch: " + "; ".join(archive_errors))

    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    payload: dict[str, Any] = {
        "schema": "m3m_gcp_100k_three_track_packet_release_authorization_v1",
        "status": "PASS_THREE_TRACK_PACKET_RELEASE_AUTHORIZED",
        "scene": SCENE,
        "method_id": args.method_id,
        "three_track_activation_path": str(activation_path),
        "three_track_activation_sha256": sha256_file(activation_path),
        "scene_attempt_freeze_sha256": candidate["scene_attempt_freeze"]["sha256"],
        "packet_manifest_path": str(packet_manifest_path),
        "packet_manifest_sha256": packet_sha,
        "packet_set_root": str(packet_set_root),
        "packet_state_path": str(packet_state_path),
        "packet_state_sha256": sha256_file(packet_state_path),
        "gcp_gate": gcp_gate,
        "lidar_gate": {
            "method_result_path": str(lidar_result_path),
            "method_result_sha256": sha256_file(lidar_result_path),
            "verification_path": str(lidar_verification_path),
            "verification_sha256": sha256_file(lidar_verification_path),
            "archive_manifest_path": str(lidar_archive_path),
            "archive_manifest_sha256": sha256_file(lidar_archive_path),
            "archive_root": str(lidar_archive_root),
        },
        "next_authorized_action": "invoke the unchanged base-plan packet-release guard on these exact paths",
        "raw_packets_deleted_by_this_authorization": False,
    }
    payload["canonical_sha256"] = canonical_sha256(payload)
    write_exclusive(output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "path": str(output),
                "sha256": sha256_file(output),
                "canonical_sha256": payload["canonical_sha256"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
