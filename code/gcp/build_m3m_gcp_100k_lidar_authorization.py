#!/usr/bin/env python3
"""Build one exact post-packet scene authorization for formal 100K LiDAR evaluation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from m3m_gcp_100k_raw_packet_state import validate_active_raw_packet_state
from m3m_gcp_100k_three_track_runtime import validate_addendum_runtime
from m3m_gcp_lidar_artifacts import canonical_sha256, sha256_file


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument(
        "--method-id", choices=("3dgs_original", "citygs_x", "metrogs"), required=True
    )
    parser.add_argument("--lidar-dispatch-receipt", type=Path, required=True)
    parser.add_argument("--global-packet-state", type=Path, required=True)
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
        raise RuntimeError("LiDAR authorization method is not activated READY")
    method = methods[args.method_id]
    recipe_path = Path(str(method["recipe_path"])).resolve()
    require_json(recipe_path, str(method["recipe_sha256"]))
    runtime_root = Path(str(candidate["candidate_output_root"])).resolve().parent
    dispatch_path = args.lidar_dispatch_receipt.resolve()
    expected_dispatch = (
        runtime_root / "lidar-packet-dispatch" / args.method_id / "dispatch_receipt.json"
    )
    output_path = args.output.resolve()
    expected_output = (
        runtime_root / "lidar-authorizations" / args.method_id / "scene_authorization.json"
    )
    formal_output = Path(str(candidate["formal_results_root"])).resolve() / "lidar" / args.method_id
    if (
        dispatch_path != expected_dispatch
        or output_path != expected_output
        or output_path.exists()
        or output_path.is_symlink()
        or formal_output.exists()
        or formal_output.is_symlink()
    ):
        raise RuntimeError("LiDAR authorization path/freshness mismatch")
    dispatch = require_json(dispatch_path)
    packet_path = Path(str(dispatch["packet_manifest_path"])).resolve()
    packet_root = Path(str(dispatch["packet_set_root"])).resolve()
    packet_state = Path(str(dispatch["packet_state_path"])).resolve()
    global_state_path = args.global_packet_state.resolve()
    validate_active_raw_packet_state(
        global_state_path,
        activation_path=activation_path,
        candidate=candidate,
        method_id=args.method_id,
        track="lidar",
        recipe_sha256=sha256_file(recipe_path),
        attempt_model_identity_sha256=method["attempt_model_identity_sha256"],
        packet_set_root=packet_root,
        track_packet_state_path=packet_state,
    )
    packet = require_json(packet_path)
    if (
        dispatch.get("status") != "PASS_LIDAR_PACKET_2196_DISPATCHED"
        or dispatch.get("method_id") != args.method_id
        or dispatch.get("three_track_activation_sha256") != sha256_file(activation_path)
        or dispatch.get("packet_manifest_sha256") != sha256_file(packet_path)
        or dispatch.get("global_raw_packet_state_sha256")
        != sha256_file(global_state_path)
        or dispatch.get("canonical_sha256") != canonical_sha256(dispatch)
        or packet.get("scene") != SCENE
        or packet.get("rendered_view_count") != 2196
        or len(packet.get("depth_index", [])) != 2196
    ):
        raise RuntimeError("LiDAR dispatch/packet binding mismatch")

    base_repo = Path(str(candidate["base_checkout"]["path"])).resolve()
    base_activation_path = Path(str(candidate["base_activation"]["path"])).resolve()
    base_activation = require_json(
        base_activation_path, str(candidate["base_activation"]["sha256"])
    )
    contract_path = base_repo / "configs" / "m3m_gcp_lidar_formal_v1.json"
    schema_path = base_repo / "configs" / "m3m_gcp_lidar_formal_artifact_schema_v1.json"
    formal_input_path = Path(str(candidate["formal_input_manifest"]["path"])).resolve()
    formal_input = require_json(
        formal_input_path, str(candidate["formal_input_manifest"]["sha256"])
    )
    freeze_path = Path(str(candidate["scene_attempt_freeze"]["path"])).resolve()
    methods_path = Path(str(candidate["methods_manifest"]["path"])).resolve()
    payload: dict[str, Any] = {
        "schema": "m3m_gcp_lidar_scene_execution_authorization_v1",
        "protocol_id": "m3m_gcp_lidar_rendered_surface_v1",
        "scene": SCENE,
        "selected_method_id": args.method_id,
        "review_task_id": base_activation["execution_plan_review_task_id"],
        "review_verdict": base_activation["execution_plan_review_verdict"],
        "execution_authorized": True,
        "contract_file_sha256": sha256_file(contract_path),
        "activation_manifest_sha256": sha256_file(base_activation_path),
        "artifact_schema_sha256": sha256_file(schema_path),
        "execution_plan_sha256": base_activation["execution_plan_sha256"],
        "formal_input_manifest_file_sha256": sha256_file(formal_input_path),
        "formal_input_manifest_canonical_sha256": formal_input["manifest_sha256"],
        "scene_attempt_freeze_path": str(freeze_path),
        "scene_attempt_freeze_sha256": sha256_file(freeze_path),
        "methods_manifest_file_sha256": sha256_file(methods_path),
        "methods_manifest_canonical_sha256": candidate["methods_manifest"][
            "canonical_sha256"
        ],
        "packet_manifest_path": str(packet_path),
        "packet_manifest_sha256": sha256_file(packet_path),
        "benchmark_commit": base_activation["benchmark_commit"],
        "benchmark_tree": base_activation["benchmark_tree"],
        "authorized_output_root": str(formal_output),
    }
    payload["canonical_sha256"] = canonical_sha256(payload)
    schema = require_json(schema_path)
    if set(payload) != set(
        schema["scene_execution_authorization"]["required_fields_exact"]
    ):
        raise RuntimeError("LiDAR scene authorization fields differ from frozen schema")
    write_exclusive(output_path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
