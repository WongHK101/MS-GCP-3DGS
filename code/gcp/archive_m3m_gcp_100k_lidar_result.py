#!/usr/bin/env python3
"""Create the exact byte-verified lightweight archive for one 100K LiDAR result."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from m3m_gcp_100k_lidar_archive import validate_exact_lidar_archive
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
    parser.add_argument("--scene-authorization", type=Path, required=True)
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--global-packet-state", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
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
        raise RuntimeError("LiDAR archive method is not activated READY")
    method = methods[args.method_id]
    recipe_path = Path(str(method["recipe_path"])).resolve()
    require_json(recipe_path, str(method["recipe_sha256"]))

    runtime_root = Path(str(candidate["candidate_output_root"])).resolve().parent
    formal_root = Path(str(candidate["formal_results_root"])).resolve()
    dispatch_path = args.lidar_dispatch_receipt.resolve()
    authorization_path = args.scene_authorization.resolve()
    execution_root = args.execution_root.resolve()
    evaluation_root = args.evaluation_root.resolve()
    verification_path = args.verification.resolve()
    global_state_path = args.global_packet_state.resolve()
    archive_root = args.archive_root.resolve()
    expected = {
        "dispatch": runtime_root
        / "lidar-packet-dispatch"
        / args.method_id
        / "dispatch_receipt.json",
        "authorization": runtime_root
        / "lidar-authorizations"
        / args.method_id
        / "scene_authorization.json",
        "execution": runtime_root / "lidar-execution" / args.method_id,
        "evaluation": formal_root / "lidar" / args.method_id,
        "verification": formal_root
        / "lidar"
        / args.method_id
        / "independent_verification.json",
        "archive": formal_root / "lidar-lightweight-archives" / args.method_id,
    }
    if (
        dispatch_path != expected["dispatch"]
        or authorization_path != expected["authorization"]
        or execution_root != expected["execution"]
        or evaluation_root != expected["evaluation"]
        or verification_path != expected["verification"]
        or archive_root != expected["archive"]
    ):
        raise RuntimeError("LiDAR archive path differs from activated namespace")
    if archive_root.exists() or archive_root.is_symlink():
        raise FileExistsError(archive_root)

    dispatch = require_json(dispatch_path)
    authorization = require_json(authorization_path)
    packet_path = Path(str(dispatch["packet_manifest_path"])).resolve()
    packet_root = Path(str(dispatch["packet_set_root"])).resolve()
    track_state_path = Path(str(dispatch["packet_state_path"])).resolve()
    validate_active_raw_packet_state(
        global_state_path,
        activation_path=activation_path,
        candidate=candidate,
        method_id=args.method_id,
        track="lidar",
        recipe_sha256=sha256_file(recipe_path),
        attempt_model_identity_sha256=method["attempt_model_identity_sha256"],
        packet_set_root=packet_root,
        track_packet_state_path=track_state_path,
    )
    if (
        dispatch.get("status") != "PASS_LIDAR_PACKET_2196_DISPATCHED"
        or dispatch.get("method_id") != args.method_id
        or dispatch.get("three_track_activation_sha256") != sha256_file(activation_path)
        or dispatch.get("packet_manifest_sha256") != sha256_file(packet_path)
        or dispatch.get("global_raw_packet_state_sha256")
        != sha256_file(global_state_path)
        or dispatch.get("canonical_sha256") != canonical_sha256(dispatch)
        or authorization.get("schema")
        != "m3m_gcp_lidar_scene_execution_authorization_v1"
        or authorization.get("selected_method_id") != args.method_id
        or Path(str(authorization.get("authorized_output_root", ""))).resolve()
        != evaluation_root
        or authorization.get("packet_manifest_sha256") != sha256_file(packet_path)
        or authorization.get("canonical_sha256") != canonical_sha256(authorization)
    ):
        raise RuntimeError("LiDAR dispatch/authorization binding mismatch")

    result_path = evaluation_root / "methods" / args.method_id / "metrics.json"
    distance_path = (
        evaluation_root
        / "methods"
        / args.method_id
        / "nearest_neighbor_distances.npz"
    )
    protocol_path = evaluation_root / "protocol_manifest.json"
    batch_path = evaluation_root / "batch_result.json"
    result = require_json(result_path)
    verification = require_json(verification_path)
    execution_receipt_path = execution_root / "execution_receipt.json"
    execution_receipt = require_json(execution_receipt_path)
    if (
        result.get("schema") != "m3m_gcp_lidar_method_result_v1"
        or result.get("scene") != SCENE
        or result.get("method_id") != args.method_id
        or result.get("train_view_count") != 2196
        or result.get("summary_row", {}).get("status") != "COMPLETE_RANKED"
        or result.get("packet_manifest_sha256") != sha256_file(packet_path)
        or result.get("scene_attempt_freeze_sha256")
        != candidate["scene_attempt_freeze"]["sha256"]
        or result.get("canonical_sha256") != canonical_sha256(result)
        or verification.get("status") != "PASS_VERIFIED_FORMAL_V1"
        or verification.get("method_id") != args.method_id
        or verification.get("method_result_sha256") != sha256_file(result_path)
        or verification.get("canonical_sha256") != canonical_sha256(verification)
        or execution_receipt.get("status")
        != "PASS_LIDAR_EVALUATOR_AND_INDEPENDENT_VERIFIER"
        or execution_receipt.get("method_id") != args.method_id
        or execution_receipt.get("result_sha256") != sha256_file(result_path)
        or execution_receipt.get("verification_sha256")
        != sha256_file(verification_path)
        or execution_receipt.get("canonical_sha256")
        != canonical_sha256(execution_receipt)
    ):
        raise RuntimeError("LiDAR evaluator/verifier/execution receipt mismatch")

    base_repo = Path(str(candidate["base_checkout"]["path"])).resolve()
    sources = {
        "contract/m3m_gcp_lidar_formal_v1.json": base_repo
        / "configs"
        / "m3m_gcp_lidar_formal_v1.json",
        "activation/activation_v3.json": Path(str(candidate["base_activation"]["path"])),
        "scene/scene_attempt_freeze_v3.json": Path(
            str(candidate["scene_attempt_freeze"]["path"])
        ),
        "scene/formal_methods_manifest.json": Path(str(candidate["methods_manifest"]["path"])),
        "scene/scene_authorization.json": authorization_path,
        "packet/depth_export_manifest.json": packet_path,
        "packet/dispatch_receipt.json": dispatch_path,
        "packet/active_raw_packet_state.json": global_state_path,
        "evaluation/protocol_manifest.json": protocol_path,
        f"evaluation/methods/{args.method_id}/metrics.json": result_path,
        f"evaluation/methods/{args.method_id}/nearest_neighbor_distances.npz": distance_path,
        "evaluation/batch_result.json": batch_path,
        "evaluation/independent_verification.json": verification_path,
        "protocol/train_view_allowlist.csv": base_repo
        / "configs"
        / "m3m_gcp_lidar_train_view_allowlists_v1"
        / f"{SCENE}.csv",
        "implementation/evaluate_m3m_gcp_lidar_formal_v1.py": base_repo
        / "code"
        / "gcp"
        / "evaluate_m3m_gcp_lidar_formal_v1.py",
        "implementation/verify_m3m_gcp_lidar_formal_v1.py": base_repo
        / "code"
        / "gcp"
        / "verify_m3m_gcp_lidar_formal_v1.py",
        "implementation/m3m_gcp_lidar_formal_artifact_schema_v1.json": base_repo
        / "configs"
        / "m3m_gcp_lidar_formal_artifact_schema_v1.json",
        "execution/evaluator.stdout.log": execution_root / "evaluator.stdout.log",
        "execution/evaluator.stderr.log": execution_root / "evaluator.stderr.log",
        "execution/verifier.stdout.log": execution_root / "verifier.stdout.log",
        "execution/verifier.stderr.log": execution_root / "verifier.stderr.log",
        "execution/execution_receipt.json": execution_receipt_path,
    }
    archive_root.mkdir(parents=True, exist_ok=False)
    binding_rows: list[dict[str, Any]] = []
    for relative, source in sorted(sources.items()):
        source = source.resolve()
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(source)
        destination = archive_root.joinpath(*relative.split("/"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as input_handle, destination.open("xb") as output_handle:
            shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
        os.chmod(destination, 0o444)
        if (
            destination.stat().st_size != source.stat().st_size
            or sha256_file(destination) != sha256_file(source)
        ):
            raise RuntimeError(f"LiDAR archive byte verification failed: {source}")
        binding_rows.append(
            {
                "relative_path": relative,
                "source_path": str(source),
                "bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
            }
        )
    bindings: dict[str, Any] = {
        "schema": "m3m_gcp_100k_lidar_lightweight_archive_source_bindings_v1",
        "scene": SCENE,
        "method_id": args.method_id,
        "files": binding_rows,
    }
    bindings["canonical_sha256"] = canonical_sha256(bindings)
    bindings_path = archive_root / "source_bindings.json"
    write_exclusive(bindings_path, bindings)
    inventory = [
        {
            "relative_path": path.relative_to(archive_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(archive_root.rglob("*"))
        if path.is_file()
    ]
    manifest_payload: dict[str, Any] = {
        "schema": "m3m_gcp_lidar_lightweight_archive_manifest_v1",
        "protocol_id": "m3m_gcp_lidar_rendered_surface_v1",
        "scene": SCENE,
        "method_id": args.method_id,
        "scene_attempt_freeze_sha256": candidate["scene_attempt_freeze"]["sha256"],
        "inventory": inventory,
    }
    manifest_payload["canonical_sha256"] = canonical_sha256(manifest_payload)
    manifest_path = archive_root / "archive_manifest.json"
    write_exclusive(manifest_path, manifest_payload)
    validate_exact_lidar_archive(
        manifest_path,
        archive_root,
        method_id=args.method_id,
        expected_scene_attempt_freeze_sha256=candidate["scene_attempt_freeze"]["sha256"],
        require_sources=True,
    )
    print(json.dumps(manifest_payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
