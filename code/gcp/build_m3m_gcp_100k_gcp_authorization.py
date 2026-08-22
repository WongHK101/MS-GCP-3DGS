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
from m3m_gcp_100k_raw_packet_state import validate_active_raw_packet_state
from metric_depth_packet import directory_tree_hash
from m3m_gcp_100k_three_track_runtime import (
    absolute_without_symlink_resolution,
    validate_addendum_runtime,
    validate_frozen_gcp_evaluation_runtime,
)
from run_m3m_gcp_100k_guarded import validate_model_identity_bundle


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


def expected_packet_model_content(method: dict[str, Any]) -> Any:
    method_id = str(method["method_id"])
    if method_id == "citygs_x":
        model_path = (
            Path(str(method["model_root"])).resolve()
            / str(method["formal_model_relative_path"])
        ).resolve()
        return directory_tree_hash(model_path.parent)
    if method_id == "metrogs":
        checkpoint = Path(str(method["formal_checkpoint"])).resolve()
        if not checkpoint.is_file() or sha256_file(checkpoint) != method["formal_model_sha256"]:
            raise RuntimeError("MetroGS activated checkpoint identity mismatch")
        return method["formal_model_sha256"]
    raise RuntimeError(f"new GCP packet authorization is unsupported for {method_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--method-id", required=True)
    parser.add_argument("--packet-manifest", type=Path, required=True)
    parser.add_argument("--packet-state", type=Path, required=True)
    parser.add_argument("--global-packet-state", type=Path, required=True)
    parser.add_argument("--gcp-packet-phase-success", type=Path, required=True)
    parser.add_argument("--gcp-camera-root-manifest", type=Path, required=True)
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
    if args.method_id not in {"citygs_x", "metrogs"}:
        raise RuntimeError("3DGS GCP must use the activated legacy-adoption receipt")
    _addendum_config_path, addendum_config = bound_component(candidate, "addendum_config")
    addendum_repo, runtime_config = validate_addendum_runtime(
        activation=activation,
        candidate=candidate,
        registry=registry,
        executing_file=Path(__file__),
    )
    if runtime_config != addendum_config:
        raise RuntimeError("GCP authorization addendum config changed")
    gcp_config = addendum_config.get("tracks", {}).get("gcp", {})
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
        raise RuntimeError("activated GCP model identity changed after scene freeze")

    formal_input_path = Path(str(candidate["formal_input_manifest"]["path"])).resolve()
    require_json(formal_input_path, str(candidate["formal_input_manifest"]["sha256"]))
    camera_manifest_path = args.gcp_camera_root_manifest.resolve()
    camera_manifest = require_json(camera_manifest_path)
    bound_camera = candidate["gcp_camera_root_manifest"]
    observations = camera_manifest.get("protocol_observations", {})
    observation_names = {
        str(value) for value in camera_manifest.get("output", {}).get("image_names", [])
    }
    if (
        camera_manifest.get("schema")
        != "m3m_gcp_100k_gcp_evaluation_camera_root_v1"
        or camera_manifest.get("status")
        != "PASS_GCP_EVALUATION_CAMERA_ROOT_NO_RGB_PIXELS"
        or camera_manifest.get("scene") != SCENE
        or camera_manifest.get("canonical_sha256") != canonical_sha256(camera_manifest)
        or observations.get("observation_count") != 256
        or observations.get("unique_camera_count") != 211
        or observations.get("formal_role_counts") != {"train": 187, "test": 24}
        or len(observation_names) != 211
        or camera_manifest_path != Path(str(bound_camera["path"])).resolve()
        or sha256_file(camera_manifest_path) != bound_camera["sha256"]
        or camera_manifest["canonical_sha256"] != bound_camera["canonical_sha256"]
        or activation.get("gcp_camera_root_manifest_sha256") != bound_camera["sha256"]
    ):
        raise RuntimeError("frozen 211-camera GCP observation root mismatch")

    packet_path = args.packet_manifest.resolve()
    packet = require_json(packet_path)
    depth_rows = packet.get("depth_index", [])
    packet_names = [str(row.get("image_name", "")) for row in depth_rows]
    if (
        packet.get("schema") != "ms_gcp_metric_depth_packet_manifest_v2"
        or packet.get("protocol_id") != GCP_PROTOCOL_ID
        or packet.get("scene") != SCENE
        or packet.get("rendered_view_count") != 211
        or len(depth_rows) != 211
        or len(packet.get("packet_index", [])) != 211
        or len(set(packet_names)) != 211
        or set(packet_names) != observation_names
    ):
        raise RuntimeError("active GCP packet manifest is not the exact 211-camera input")
    expected_model_content = expected_packet_model_content(method)
    if packet.get("model_content_hash") != expected_model_content:
        raise RuntimeError("GCP packet model-content inventory differs from the full frozen model")

    phase_success_path = args.gcp_packet_phase_success.resolve()
    phase_success = require_json(phase_success_path)
    packet_state_path = args.packet_state.resolve()
    packet_state = require_json(packet_state_path)
    packet_root = packet_path.parent.resolve()
    global_state_path = args.global_packet_state.resolve()
    validate_active_raw_packet_state(
        global_state_path,
        activation_path=activation_path,
        candidate=candidate,
        method_id=args.method_id,
        track="gcp",
        recipe_sha256=sha256_file(recipe_path),
        attempt_model_identity_sha256=sha256_file(identity_path),
        packet_set_root=packet_root,
        track_packet_state_path=packet_state_path,
    )
    manifest_products = [
        row
        for row in phase_success.get("products", [])
        if Path(str(row.get("path", ""))).resolve() == packet_path
    ]
    if (
        phase_success.get("schema")
        != "m3m_gcp_100k_gcp_packet_phase_success_v1"
        or phase_success.get("status") != "PASS_GCP_PACKET_211"
        or phase_success.get("scene") != SCENE
        or phase_success.get("method_id") != args.method_id
        or Path(str(phase_success.get("three_track_activation_path", ""))).resolve()
        != activation_path
        or phase_success.get("three_track_activation_sha256")
        != sha256_file(activation_path)
        or phase_success.get("scene_attempt_freeze_sha256")
        != candidate["scene_attempt_freeze"]["sha256"]
        or phase_success.get("methods_manifest_sha256")
        != candidate["methods_manifest"]["sha256"]
        or phase_success.get("attempt_model_identity_sha256")
        != method["attempt_model_identity_sha256"]
        or phase_success.get("gcp_camera_root_manifest_sha256")
        != sha256_file(camera_manifest_path)
        or Path(str(phase_success.get("packet_set_root", ""))).resolve()
        != packet_root
        or Path(str(phase_success.get("packet_state_path", ""))).resolve()
        != packet_state_path
        or phase_success.get("packet_state_sha256") != sha256_file(packet_state_path)
        or phase_success.get("global_raw_packet_state_path")
        != str(global_state_path)
        or phase_success.get("global_raw_packet_state_sha256")
        != sha256_file(global_state_path)
        or phase_success.get("canonical_sha256") != canonical_sha256(phase_success)
        or len(manifest_products) != 1
        or manifest_products[0].get("sha256") != sha256_file(packet_path)
        or manifest_products[0].get("bytes") != packet_path.stat().st_size
        or packet_state.get("schema") != "m3m_gcp_100k_single_gcp_packet_state_v1"
        or packet_state.get("method_id") != args.method_id
        or Path(str(packet_state.get("packet_set_root", ""))).resolve()
        != packet_root
        or packet_state.get("three_track_activation_sha256")
        != sha256_file(activation_path)
        or packet_state.get("candidate_manifest_sha256")
        != sha256_file(candidate_path)
        or packet_state.get("scene_attempt_freeze_sha256")
        != candidate["scene_attempt_freeze"]["sha256"]
        or packet_state.get("methods_manifest_sha256")
        != candidate["methods_manifest"]["sha256"]
        or packet_state.get("recipe_sha256") != sha256_file(recipe_path)
        or packet_state.get("attempt_model_identity_sha256")
        != method["attempt_model_identity_sha256"]
        or packet_state.get("gcp_camera_root_manifest_sha256")
        != sha256_file(camera_manifest_path)
        or packet_state.get("protocol_observation_count") != 256
        or packet_state.get("packet_view_count") != 211
        or packet_state.get("formal_role_counts") != {"train": 187, "test": 24}
        or packet_state.get("canonical_sha256") != canonical_sha256(packet_state)
    ):
        raise RuntimeError("GCP packet phase-success/state/current activation binding mismatch")

    data_root = args.gcp_data_root.resolve()
    data_contract = data_root / "DATA_CONTRACT_DRAFT.json"
    if not data_contract.is_file() or sha256_file(data_contract) != EXPECTED_DATA_CONTRACT_SHA:
        raise RuntimeError("GCP data contract identity mismatch")
    protocol_release = args.gcp_protocol_release.resolve()
    if not protocol_release.is_file() or sha256_file(protocol_release) != EXPECTED_PROTOCOL_RELEASE_SHA:
        raise RuntimeError("GCP protocol-release identity mismatch")
    python = absolute_without_symlink_resolution(args.python)
    evaluation_runtime_identity, evaluation_environment = (
        validate_frozen_gcp_evaluation_runtime(
            gcp_config,
            requested_python=python,
        )
    )
    evaluator = (addendum_repo / str(gcp_config["evaluator_path"])).resolve()
    verifier = (addendum_repo / str(gcp_config["verifier_path"])).resolve()
    if (
        evaluator.parent.parent.parent != addendum_repo
        or verifier.parent.parent.parent != addendum_repo
        or not evaluator.is_file()
        or not verifier.is_file()
        or sha256_file(evaluator) != gcp_config.get("evaluator_sha256")
        or sha256_file(verifier) != gcp_config.get("verifier_sha256")
    ):
        raise RuntimeError("activated GCP evaluator/verifier identity mismatch")

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
        "attempt_model_identity_canonical_sha256": method[
            "attempt_model_identity_canonical_sha256"
        ],
        "recipe_path": str(recipe_path),
        "recipe_sha256": sha256_file(recipe_path),
        "expected_model_content": expected_model_content,
        "observed_packet_model_content": packet["model_content_hash"],
        "gcp_camera_root_manifest_path": str(camera_manifest_path),
        "gcp_camera_root_manifest_sha256": sha256_file(camera_manifest_path),
        "gcp_camera_root_manifest_canonical_sha256": camera_manifest["canonical_sha256"],
        "gcp_packet_phase_success_path": str(phase_success_path),
        "gcp_packet_phase_success_sha256": sha256_file(phase_success_path),
        "gcp_packet_phase_success_canonical_sha256": phase_success["canonical_sha256"],
        "packet_state_path": str(packet_state_path),
        "packet_state_sha256": sha256_file(packet_state_path),
        "packet_state_canonical_sha256": packet_state["canonical_sha256"],
        "global_raw_packet_state_path": str(global_state_path),
        "global_raw_packet_state_sha256": sha256_file(global_state_path),
        "packet_manifest_path": str(packet_path),
        "packet_manifest_sha256": sha256_file(packet_path),
        "protocol_observation_count": 256,
        "packet_view_count": 211,
        "formal_role_counts": {"train": 187, "test": 24},
        "packet_names_canonical_sha256": packet_names_sha,
        "gcp_data_contract_path": str(data_contract),
        "gcp_data_contract_sha256": sha256_file(data_contract),
        "gcp_protocol_release_path": str(protocol_release),
        "gcp_protocol_release_sha256": sha256_file(protocol_release),
        "evaluator_path": str(evaluator),
        "evaluator_sha256": sha256_file(evaluator),
        "verifier_path": str(verifier),
        "verifier_sha256": sha256_file(verifier),
        "gcp_evaluation_python_path": str(python),
        "gcp_evaluation_runtime_identity": evaluation_runtime_identity,
        "gcp_evaluation_runtime_identity_canonical_sha256": canonical_sha256(
            evaluation_runtime_identity
        ),
        "gcp_evaluation_subprocess_environment": evaluation_environment,
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
