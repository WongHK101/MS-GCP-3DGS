#!/usr/bin/env python3
"""Create the one immutable, hash-bound LiDAR attempt freeze for a scene."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from m3m_gcp_lidar_artifacts import (
    METHOD_IDS,
    PROTOCOL_ID,
    canonical_sha256,
    sha256_file,
    validate_failure_evidence_file,
)


def require_bound_file(path_value: object, sha_value: object, label: str) -> None:
    path = Path(str(path_value))
    if not path.is_absolute() or not path.is_file():
        raise RuntimeError(f"{label} is not an existing absolute file")
    if sha256_file(path) != sha_value:
        raise RuntimeError(f"{label} SHA mismatch")


def validate_methods(methods: dict, schema: dict, scene: str) -> None:
    if set(methods) != {"schema", "protocol_id", "scene", "methods", "canonical_sha256"}:
        raise RuntimeError("methods manifest top-level field inventory mismatch")
    if methods.get("schema") != "m3m_gcp_lidar_formal_methods_v1":
        raise RuntimeError("methods manifest schema mismatch")
    if methods.get("protocol_id") != PROTOCOL_ID or methods.get("scene") != scene:
        raise RuntimeError("methods manifest identity mismatch")
    if methods.get("canonical_sha256") != canonical_sha256(methods):
        raise RuntimeError("methods manifest canonical SHA mismatch")
    rows = methods.get("methods", [])
    if [row.get("method_id") for row in rows] != list(METHOD_IDS):
        raise RuntimeError("methods manifest is not the exact ordered ten-method pool")
    exact_fields = set(schema["formal_methods_manifest"]["method_fields_exact"])
    for row in rows:
        method_id = str(row.get("method_id"))
        if set(row) != exact_fields:
            raise RuntimeError(f"{method_id}: method field inventory mismatch")
        if not isinstance(row.get("method_name"), str) or not row["method_name"]:
            raise RuntimeError(f"{method_id}: method name missing")
        if not Path(str(row.get("run_root"))).is_absolute():
            raise RuntimeError(f"{method_id}: run root is not absolute")
        require_bound_file(row.get("recipe_path"), row.get("recipe_sha256"), f"{method_id} recipe")
        require_bound_file(
            row.get("renderer_adapter_path"),
            row.get("renderer_adapter_sha256"),
            f"{method_id} renderer adapter",
        )
        status = row.get("attempt_status")
        if status == "READY_FOR_EVALUATION":
            require_bound_file(
                row.get("model_checkpoint_path"),
                row.get("model_checkpoint_sha256"),
                f"{method_id} checkpoint",
            )
            if row.get("failure_evidence_path") is not None or row.get("failure_evidence_sha256") is not None:
                raise RuntimeError(f"{method_id}: ready attempt carries failure evidence")
        elif status in {"OOM_UNRANKED", "FAILED_UNRANKED"}:
            if row.get("model_checkpoint_path") is not None or row.get("model_checkpoint_sha256") is not None:
                raise RuntimeError(f"{method_id}: failed attempt carries a checkpoint")
            failure_path = Path(str(row.get("failure_evidence_path")))
            errors = validate_failure_evidence_file(
                failure_path,
                expected_sha256=str(row.get("failure_evidence_sha256")),
                expected_scene=scene,
                expected_method_id=method_id,
                expected_status=str(status),
            )
            if errors:
                raise RuntimeError(f"{method_id}: invalid failure evidence: {'; '.join(errors)}")
        else:
            raise RuntimeError(f"{method_id}: invalid attempt status")


def write_exclusive(path: Path, payload: dict) -> None:
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods-manifest", type=Path, required=True)
    parser.add_argument("--artifact-schema", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    methods_path = args.methods_manifest.resolve()
    schema_path = args.artifact_schema.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("scene attempt freeze already exists; replacement is forbidden")
    methods = json.loads(methods_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validate_methods(methods, schema, args.scene)
    payload = {
        "schema": "m3m_gcp_lidar_scene_attempt_freeze_v1",
        "protocol_id": PROTOCOL_ID,
        "scene": args.scene,
        "methods_manifest_path": str(methods_path),
        "methods_manifest_file_sha256": sha256_file(methods_path),
        "methods_manifest_canonical_sha256": methods["canonical_sha256"],
        "frozen_method_ids": list(METHOD_IDS),
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    payload["canonical_sha256"] = canonical_sha256(payload)
    write_exclusive(output, payload)
    print(json.dumps({"status": "PASS_SCENE_ATTEMPT_FREEZE_CREATED", "path": str(output), "sha256": sha256_file(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
