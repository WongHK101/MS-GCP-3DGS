#!/usr/bin/env python3
"""Exact lightweight-archive contract for one successful 100K LiDAR method."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any

from m3m_gcp_lidar_artifacts import canonical_sha256, sha256_file
from verify_m3m_gcp_lidar_formal_v1 import validate_archive_manifest


SCENE = "gcp_100000_20260610"


def expected_lidar_archive_relatives(method_id: str) -> set[str]:
    return {
        "contract/m3m_gcp_lidar_formal_v1.json",
        "activation/activation_v3.json",
        "scene/scene_attempt_freeze_v3.json",
        "scene/formal_methods_manifest.json",
        "scene/scene_authorization.json",
        "packet/depth_export_manifest.json",
        "packet/dispatch_receipt.json",
        "packet/active_raw_packet_state.json",
        "evaluation/protocol_manifest.json",
        f"evaluation/methods/{method_id}/metrics.json",
        f"evaluation/methods/{method_id}/nearest_neighbor_distances.npz",
        "evaluation/batch_result.json",
        "evaluation/independent_verification.json",
        "protocol/train_view_allowlist.csv",
        "implementation/evaluate_m3m_gcp_lidar_formal_v1.py",
        "implementation/verify_m3m_gcp_lidar_formal_v1.py",
        "implementation/m3m_gcp_lidar_formal_artifact_schema_v1.json",
        "execution/evaluator.stdout.log",
        "execution/evaluator.stderr.log",
        "execution/verifier.stdout.log",
        "execution/verifier.stderr.log",
        "execution/execution_receipt.json",
        "source_bindings.json",
    }


def validate_exact_lidar_archive(
    path: Path,
    root: Path,
    *,
    method_id: str,
    expected_scene_attempt_freeze_sha256: str,
    require_sources: bool,
) -> dict[str, Any]:
    path = path.resolve()
    root = root.resolve()
    if path != root / "archive_manifest.json" or not path.is_file() or path.is_symlink():
        raise RuntimeError("LiDAR archive manifest path mismatch")
    base_errors = validate_archive_manifest(
        path,
        root,
        expected_scene_attempt_freeze_sha256=expected_scene_attempt_freeze_sha256,
    )
    if base_errors:
        raise RuntimeError("LiDAR base archive validation failed: " + "; ".join(base_errors))
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("scene") != SCENE or payload.get("method_id") != method_id:
        raise RuntimeError("LiDAR archive scene/method mismatch")
    rows = payload.get("inventory", [])
    expected = expected_lidar_archive_relatives(method_id)
    observed = [str(row.get("relative_path", "")) for row in rows]
    if len(observed) != len(expected) or set(observed) != expected:
        raise RuntimeError("LiDAR archive exact file inventory mismatch")
    if len(observed) != len(set(observed)):
        raise RuntimeError("LiDAR archive contains duplicate inventory rows")

    bindings_path = root / "source_bindings.json"
    if not bindings_path.is_file() or bindings_path.is_symlink():
        raise RuntimeError("LiDAR archive source bindings are missing")
    bindings = json.loads(bindings_path.read_text(encoding="utf-8"))
    binding_rows = bindings.get("files", [])
    expected_bound = expected - {"source_bindings.json"}
    observed_bound = [str(row.get("relative_path", "")) for row in binding_rows]
    if (
        set(bindings)
        != {"schema", "scene", "method_id", "files", "canonical_sha256"}
        or bindings.get("schema")
        != "m3m_gcp_100k_lidar_lightweight_archive_source_bindings_v1"
        or bindings.get("scene") != SCENE
        or bindings.get("method_id") != method_id
        or bindings.get("canonical_sha256") != canonical_sha256(bindings)
        or len(observed_bound) != len(expected_bound)
        or set(observed_bound) != expected_bound
        or len(observed_bound) != len(set(observed_bound))
    ):
        raise RuntimeError("LiDAR archive source-binding inventory mismatch")
    manifest_rows = {str(row["relative_path"]): row for row in rows}
    for row in binding_rows:
        if set(row) != {"relative_path", "source_path", "bytes", "sha256"}:
            raise RuntimeError("LiDAR archive source-binding row fields mismatch")
        relative = str(row["relative_path"])
        rel = PurePosixPath(relative)
        if rel.is_absolute() or ".." in rel.parts:
            raise RuntimeError(f"unsafe LiDAR archive source binding: {relative}")
        archived = root.joinpath(*rel.parts)
        manifest_row = manifest_rows[relative]
        if (
            archived.stat().st_size != int(row["bytes"])
            or sha256_file(archived) != row["sha256"]
            or int(manifest_row["bytes"]) != int(row["bytes"])
            or manifest_row["sha256"] != row["sha256"]
        ):
            raise RuntimeError(f"LiDAR archive/source-binding byte mismatch: {relative}")
        if require_sources:
            source = Path(str(row["source_path"])).resolve()
            if (
                not source.is_file()
                or source.is_symlink()
                or source.stat().st_size != int(row["bytes"])
                or sha256_file(source) != row["sha256"]
            ):
                raise RuntimeError(f"LiDAR archive source changed: {source}")
    return payload
