#!/usr/bin/env python3
"""Authorize one READY 100K method for frozen GCP evaluation on its active packets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from m3m_gcp_lidar_artifacts import canonical_sha256, command_sha256, sha256_file


SCENE = "gcp_100000_20260610"
GCP_PROTOCOL_ID = "m3m_gcp_native_quarter_geometry_v2"
EXPECTED_PROTOCOL_RELEASE_SHA = "21fbac75d66433169535ea7440c31393f7a5ecdb4ed94fcefd31d1780c28bea4"
EXPECTED_DATA_CONTRACT_SHA = "9141cf90e5bcdf342e5d47e58aa3a0aa48300bd461411ae495ce974993e5ed13"


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


def bound_component(candidate: dict[str, Any], key: str) -> tuple[Path, dict[str, Any]]:
    row = candidate[key]
    path = Path(str(row["path"])).resolve()
    payload = require_json(path, str(row["sha256"]))
    if row.get("canonical_sha256") is not None and (
        payload.get("canonical_sha256") != row["canonical_sha256"]
        or canonical_sha256(payload) != row["canonical_sha256"]
    ):
        raise RuntimeError(f"candidate {key} canonical binding mismatch")
    return path, payload


def packet_model_hashes(registry_method: dict[str, Any]) -> set[str]:
    candidates: list[Any] = [
        registry_method.get("formal_model_sha256"),
        registry_method.get("point_cloud_sha256"),
        registry_method.get("attempt_model_identity_sha256"),
    ]
    candidates.extend((registry_method.get("formal_model_aux_sha256") or {}).values())
    return {str(value) for value in candidates if isinstance(value, str) and len(value) == 64}


def packet_content_hashes(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value} if len(value) == 64 else set()
    if not isinstance(value, dict):
        return set()
    hashes = {
        str(value.get("sha256", "")),
        *[str(row.get("sha256", "")) for row in value.get("files", []) if isinstance(row, dict)],
    }
    return {item for item in hashes if len(item) == 64}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--method-id", required=True)
    parser.add_argument("--packet-manifest", type=Path, required=True)
    parser.add_argument("--gcp-data-root", type=Path, required=True)
    parser.add_argument("--gcp-protocol-release", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--verification-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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
    if candidate.get("canonical_sha256") != activation["candidate_manifest_canonical_sha256"]:
        raise RuntimeError("activation/candidate canonical binding mismatch")
    _registry_path, registry = bound_component(candidate, "rgb_registry")
    method_rows = {str(row["method_id"]): row for row in registry.get("methods", [])}
    if args.method_id not in method_rows or args.method_id not in registry.get("ready_method_ids", []):
        raise RuntimeError("method is not READY in the activated registry")
    method = method_rows[args.method_id]

    formal_input_path = Path(str(candidate["formal_input_manifest"]["path"])).resolve()
    formal_input = require_json(formal_input_path, str(candidate["formal_input_manifest"]["sha256"]))
    train_names = {str(row["image_name"]) for row in formal_input["images"] if row.get("role") == "train"}
    if len(train_names) != 2196:
        raise RuntimeError("frozen train allowlist count mismatch")

    packet_path = args.packet_manifest.resolve()
    packet = require_json(packet_path)
    depth_rows = packet.get("depth_index", [])
    packet_names = [str(row.get("image_name", "")) for row in depth_rows]
    if (
        packet.get("schema") != "ms_gcp_metric_depth_packet_manifest_v2"
        or packet.get("protocol_id") != GCP_PROTOCOL_ID
        or packet.get("scene") != SCENE
        or packet.get("rendered_view_count") != 2196
        or len(depth_rows) != 2196
        or len(packet.get("packet_index", [])) != 2196
        or len(set(packet_names)) != 2196
        or set(packet_names) != train_names
        or any(row.get("split") != "train" for row in depth_rows)
    ):
        raise RuntimeError("active packet manifest is not the exact 2,196-view GCP input")
    allowed_model_hashes = packet_model_hashes(method)
    observed_model_hashes = packet_content_hashes(packet.get("model_content_hash"))
    if not observed_model_hashes.intersection(allowed_model_hashes):
        raise RuntimeError(
            "packet model-content inventory is not bound by the activated model registry: "
            f"{sorted(observed_model_hashes)}"
        )

    data_root = args.gcp_data_root.resolve()
    data_contract = data_root / "DATA_CONTRACT_DRAFT.json"
    if not data_contract.is_file() or sha256_file(data_contract) != EXPECTED_DATA_CONTRACT_SHA:
        raise RuntimeError("GCP data contract identity mismatch")
    protocol_release = args.gcp_protocol_release.resolve()
    if not protocol_release.is_file() or sha256_file(protocol_release) != EXPECTED_PROTOCOL_RELEASE_SHA:
        raise RuntimeError("GCP protocol-release identity mismatch")
    python = args.python.resolve()
    if not python.is_file():
        raise FileNotFoundError(python)
    addendum_repo = Path(str(registry["shared"]["benchmark_repo_template"])).resolve()
    evaluator = addendum_repo / "code" / "gcp" / "evaluate_m3m_native_quarter_geometry.py"
    verifier = addendum_repo / "code" / "gcp" / "verify_m3m_native_quarter_geometry_outputs.py"
    for path in (evaluator, verifier):
        if not path.is_file():
            raise FileNotFoundError(path)

    output_root = args.output_root.resolve()
    verification_output = args.verification_output.resolve()
    authorization_output = args.output.resolve()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError("GCP output root must be fresh")
    if verification_output.exists() or verification_output.is_symlink():
        raise FileExistsError("GCP verification output must be fresh")
    if authorization_output.exists() or authorization_output.is_symlink():
        raise FileExistsError("GCP authorization output already exists")
    if output_root != Path(str(candidate["formal_results_root"])).resolve() / "gcp" / args.method_id:
        raise RuntimeError("GCP output root is outside the method's frozen formal namespace")

    evaluator_command = [
        str(python), "-B", str(evaluator),
        "--data_root", str(data_root),
        "--protocol_release", str(protocol_release),
        "--scene", SCENE,
        "--method_id", args.method_id,
        "--metric_packet_manifest", str(packet_path),
        "--out_dir", str(output_root),
    ]
    verifier_command = [
        str(python), "-B", str(verifier),
        "--eval_dir", str(output_root),
        "--out", str(verification_output),
        "--tolerance", "1e-9",
    ]
    packet_names_sha = hashlib.sha256(
        json.dumps(sorted(packet_names), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    payload: dict[str, Any] = {
        "schema": "m3m_gcp_100k_gcp_execution_authorization_v1",
        "status": "ACTIVE_FROZEN",
        "execution_authorized": True,
        "scene": SCENE,
        "method_id": args.method_id,
        "protocol_id": GCP_PROTOCOL_ID,
        "three_track_activation_path": str(activation_path),
        "three_track_activation_sha256": sha256_file(activation_path),
        "scene_attempt_freeze_sha256": candidate["scene_attempt_freeze"]["sha256"],
        "methods_manifest_sha256": candidate["methods_manifest"]["sha256"],
        "attempt_model_identity_path": method["attempt_model_identity_path"],
        "attempt_model_identity_sha256": method["attempt_model_identity_sha256"],
        "allowed_model_content_hashes": sorted(allowed_model_hashes),
        "observed_packet_model_content_hashes": sorted(observed_model_hashes),
        "packet_manifest_path": str(packet_path),
        "packet_manifest_sha256": sha256_file(packet_path),
        "packet_view_count": 2196,
        "packet_names_canonical_sha256": packet_names_sha,
        "gcp_data_contract_path": str(data_contract),
        "gcp_data_contract_sha256": sha256_file(data_contract),
        "gcp_protocol_release_path": str(protocol_release),
        "gcp_protocol_release_sha256": sha256_file(protocol_release),
        "evaluator_path": str(evaluator),
        "evaluator_sha256": sha256_file(evaluator),
        "verifier_path": str(verifier),
        "verifier_sha256": sha256_file(verifier),
        "authorized_output_root": str(output_root),
        "authorized_verification_output": str(verification_output),
        "evaluator_command": evaluator_command,
        "evaluator_command_sha256": command_sha256(evaluator_command),
        "verifier_command": verifier_command,
        "verifier_command_sha256": command_sha256(verifier_command),
    }
    payload["canonical_sha256"] = canonical_sha256(payload)
    write_exclusive(authorization_output, payload)
    print(
        json.dumps(
            {
                "status": "PASS_100K_GCP_EXECUTION_AUTHORIZED",
                "path": str(authorization_output),
                "sha256": sha256_file(authorization_output),
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
