#!/usr/bin/env python3
"""One atomic addendum-wide mutex for every active 100K raw depth packet."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from m3m_gcp_lidar_artifacts import canonical_sha256, sha256_file


SCENE = "gcp_100000_20260610"
SCHEMA = "m3m_gcp_100k_active_raw_packet_state_v1"
FIELDS = {
    "schema",
    "status",
    "scene",
    "method_id",
    "track",
    "three_track_activation_path",
    "three_track_activation_sha256",
    "candidate_manifest_sha256",
    "scene_attempt_freeze_sha256",
    "methods_manifest_sha256",
    "recipe_sha256",
    "attempt_model_identity_sha256",
    "packet_set_root",
    "track_packet_state_path",
    "owner_evidence_root",
    "created_at_utc",
    "canonical_sha256",
}


def active_raw_packet_state_path(candidate: dict[str, Any]) -> Path:
    runtime_root = Path(str(candidate["candidate_output_root"])).resolve().parent
    return runtime_root / "raw-packet-lifecycle" / "ACTIVE_RAW_PACKET_STATE.json"


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
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


def collect_raw_packet_targets(
    *, candidate: dict[str, Any], registry: dict[str, Any]
) -> set[Path]:
    runtime_root = Path(str(candidate["candidate_output_root"])).resolve().parent
    targets = {
        runtime_root / "gcp-packet-scratch" / "ACTIVE_GCP_PACKET_STATE.json"
    }
    for method in registry.get("methods", []):
        method_id = str(method["method_id"])
        if method_id != "3dgs_original":
            targets.add(runtime_root / "gcp-packet-scratch" / method_id)
        recipe_path = Path(str(method["recipe_path"])).resolve()
        if not recipe_path.is_file() or recipe_path.is_symlink():
            raise FileNotFoundError(recipe_path)
        if sha256_file(recipe_path) != method["recipe_sha256"]:
            raise RuntimeError(f"recipe SHA mismatch while collecting packet roots: {method_id}")
        recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
        targets.add(Path(str(recipe["authorized_packet_set_root"])).resolve())
        targets.add(Path(str(recipe["authorized_packet_state"])).resolve())
    return targets


def acquire_active_raw_packet_state(
    *,
    activation_path: Path,
    candidate: dict[str, Any],
    registry: dict[str, Any],
    method_id: str,
    track: str,
    recipe_sha256: str,
    attempt_model_identity_sha256: str,
    packet_set_root: Path,
    track_packet_state_path: Path,
    owner_evidence_root: Path,
) -> tuple[Path, dict[str, Any]]:
    if track not in {"gcp", "lidar"}:
        raise ValueError(f"invalid raw-packet track: {track}")
    state_path = active_raw_packet_state_path(candidate)
    if state_path.exists() or state_path.is_symlink():
        raise FileExistsError(f"another raw packet owns the global mutex: {state_path}")
    stale = sorted(
        str(path)
        for path in collect_raw_packet_targets(candidate=candidate, registry=registry)
        if path.exists() or path.is_symlink()
    )
    if stale:
        raise RuntimeError(f"raw packet target exists without the global mutex: {stale}")
    activation_path = activation_path.resolve()
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "ACTIVE_EXCLUSIVE_RAW_PACKET",
        "scene": SCENE,
        "method_id": method_id,
        "track": track,
        "three_track_activation_path": str(activation_path),
        "three_track_activation_sha256": sha256_file(activation_path),
        "candidate_manifest_sha256": sha256_file(
            Path(str(candidate["candidate_output_root"])).resolve()
            / "three_track_candidate_manifest_v1.json"
        ),
        "scene_attempt_freeze_sha256": candidate["scene_attempt_freeze"]["sha256"],
        "methods_manifest_sha256": candidate["methods_manifest"]["sha256"],
        "recipe_sha256": recipe_sha256,
        "attempt_model_identity_sha256": attempt_model_identity_sha256,
        "packet_set_root": str(packet_set_root.resolve()),
        "track_packet_state_path": str(track_packet_state_path.resolve()),
        "owner_evidence_root": str(owner_evidence_root.resolve()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    payload["canonical_sha256"] = canonical_sha256(payload)
    # O_EXCL is the actual cross-track serialization point.
    _write_exclusive(state_path, payload)
    return state_path, payload


def validate_active_raw_packet_state(
    path: Path,
    *,
    activation_path: Path,
    candidate: dict[str, Any],
    method_id: str,
    track: str,
    recipe_sha256: str,
    attempt_model_identity_sha256: str,
    packet_set_root: Path,
    track_packet_state_path: Path,
) -> dict[str, Any]:
    path = path.resolve()
    if path != active_raw_packet_state_path(candidate):
        raise RuntimeError("global raw-packet state path differs from activated namespace")
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema": SCHEMA,
        "status": "ACTIVE_EXCLUSIVE_RAW_PACKET",
        "scene": SCENE,
        "method_id": method_id,
        "track": track,
        "three_track_activation_path": str(activation_path.resolve()),
        "three_track_activation_sha256": sha256_file(activation_path.resolve()),
        "scene_attempt_freeze_sha256": candidate["scene_attempt_freeze"]["sha256"],
        "methods_manifest_sha256": candidate["methods_manifest"]["sha256"],
        "recipe_sha256": recipe_sha256,
        "attempt_model_identity_sha256": attempt_model_identity_sha256,
        "packet_set_root": str(packet_set_root.resolve()),
        "track_packet_state_path": str(track_packet_state_path.resolve()),
    }
    if (
        set(payload) != FIELDS
        or any(payload.get(key) != value for key, value in expected.items())
        or payload.get("candidate_manifest_sha256")
        != sha256_file(
            Path(str(candidate["candidate_output_root"])).resolve()
            / "three_track_candidate_manifest_v1.json"
        )
        or not str(payload.get("owner_evidence_root", ""))
        or payload.get("canonical_sha256") != canonical_sha256(payload)
    ):
        raise RuntimeError("global raw-packet state identity mismatch")
    return payload
