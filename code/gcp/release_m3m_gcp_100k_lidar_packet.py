#!/usr/bin/env python3
"""Release one 2,196-view LiDAR packet only through the reviewed addendum dispatcher."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from m3m_gcp_lidar_artifacts import canonical_sha256, command_sha256, sha256_file
from m3m_gcp_100k_raw_packet_state import (
    active_raw_packet_state_path,
    validate_active_raw_packet_state,
)
from m3m_gcp_100k_lidar_archive import validate_exact_lidar_archive
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


def current_model_content(method: dict[str, Any]) -> Any:
    method_id = str(method["method_id"])
    if method_id == "3dgs_original":
        return directory_tree_hash(Path(str(method["model_root"])).resolve())
    if method_id == "citygs_x":
        return directory_tree_hash(
            (
                Path(str(method["model_root"])).resolve()
                / str(method["formal_model_relative_path"])
            ).resolve().parent
        )
    if method_id == "metrogs":
        checkpoint = Path(str(method["formal_checkpoint"])).resolve()
        if sha256_file(checkpoint) != method["formal_model_sha256"]:
            raise RuntimeError("MetroGS LiDAR release checkpoint identity mismatch")
        return method["formal_model_sha256"]
    raise RuntimeError(f"unsupported READY method: {method_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--method-id", choices=("3dgs_original", "citygs_x", "metrogs"), required=True)
    parser.add_argument("--lidar-dispatch-receipt", type=Path, required=True)
    parser.add_argument("--packet-manifest", type=Path, required=True)
    parser.add_argument("--packet-set-root", type=Path, required=True)
    parser.add_argument("--packet-state", type=Path, required=True)
    parser.add_argument("--global-packet-state", type=Path, required=True)
    parser.add_argument("--lidar-method-result", type=Path, required=True)
    parser.add_argument("--lidar-verification", type=Path, required=True)
    parser.add_argument("--lidar-archive-manifest", type=Path, required=True)
    parser.add_argument("--lidar-archive-root", type=Path, required=True)
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
    if (
        candidate.get("canonical_sha256") != activation["candidate_manifest_canonical_sha256"]
        or canonical_sha256(candidate) != activation["candidate_manifest_canonical_sha256"]
    ):
        raise RuntimeError("activation/candidate binding mismatch")
    registry_row = candidate["rgb_registry"]
    registry = require_json(Path(str(registry_row["path"])), str(registry_row["sha256"]))
    methods = {str(row["method_id"]): row for row in registry.get("methods", [])}
    if args.method_id not in registry.get("ready_method_ids", []) or args.method_id not in methods:
        raise RuntimeError("LiDAR release method is not activated READY")
    method = methods[args.method_id]
    validate_addendum_runtime(
        activation=activation,
        candidate=candidate,
        registry=registry,
        executing_file=Path(__file__),
    )

    base_repo = Path(str(candidate["base_checkout"]["path"])).resolve()
    base = candidate["base_checkout"]
    if (
        subprocess.check_output(["git", "-C", str(base_repo), "rev-parse", "HEAD"], text=True).strip()
        != base["commit"]
        or subprocess.check_output(
            ["git", "-C", str(base_repo), "show", "-s", "--format=%T", "HEAD"], text=True
        ).strip()
        != base["tree"]
        or subprocess.check_output(["git", "-C", str(base_repo), "status", "--porcelain"], text=True).strip()
    ):
        raise RuntimeError("frozen base checkout identity mismatch")
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
        raise RuntimeError("LiDAR release frozen model identity mismatch")

    runtime_root = Path(str(candidate["candidate_output_root"])).resolve().parent
    dispatch_path = args.lidar_dispatch_receipt.resolve()
    expected_dispatch = runtime_root / "lidar-packet-dispatch" / args.method_id / "dispatch_receipt.json"
    dispatch = require_json(dispatch_path)
    packet_path = args.packet_manifest.resolve()
    packet_root = args.packet_set_root.resolve()
    state_path = args.packet_state.resolve()
    global_state_path = args.global_packet_state.resolve()
    result_path = args.lidar_method_result.resolve()
    verification_path = args.lidar_verification.resolve()
    archive_path = args.lidar_archive_manifest.resolve()
    archive_root = args.lidar_archive_root.resolve()
    intent_path = args.release_intent.resolve()
    receipt_path = args.deletion_receipt.resolve()
    expected_release_root = runtime_root / "lidar-packet-release" / args.method_id
    formal_root = Path(str(candidate["formal_results_root"])).resolve()
    expected_evaluation_root = formal_root / "lidar" / args.method_id
    expected_result = (
        expected_evaluation_root / "methods" / args.method_id / "metrics.json"
    )
    expected_verification = expected_evaluation_root / "independent_verification.json"
    expected_archive_root = formal_root / "lidar-lightweight-archives" / args.method_id
    if (
        dispatch_path != expected_dispatch
        or packet_root != Path(str(recipe["authorized_packet_set_root"])).resolve()
        or state_path != Path(str(recipe["authorized_packet_state"])).resolve()
        or global_state_path != active_raw_packet_state_path(candidate)
        or packet_path != packet_root / "depth_export_manifest.json"
        or intent_path != expected_release_root / "release_intent.json"
        or receipt_path != expected_release_root / "deletion_receipt.json"
        or result_path != expected_result
        or verification_path != expected_verification
        or archive_root != expected_archive_root
        or archive_path != expected_archive_root / "archive_manifest.json"
    ):
        raise RuntimeError("LiDAR release path differs from activated/frozen namespace")
    if receipt_path.exists() or receipt_path.is_symlink():
        raise FileExistsError(receipt_path)

    base_activation = Path(str(candidate["base_activation"]["path"])).resolve()
    plan = base_repo / "configs" / "m3m_gcp_native_quarter_100k_ten_method_execution_plan_v3.json"
    recipe_manifest = base_repo / "configs" / "m3m_gcp_native_quarter_100k_recipe_manifest_v3.json"
    roots = recipe["phase_roots"]["packet"]
    source = recipe["source_bindings"]["packet"]
    failure = Path(str(recipe["authorized_evidence_root"])).resolve() / "packet" / "failure.json"
    guard_python = str(recipe["phase_commands"]["packet"][0])
    guard = base_repo / "code" / "gcp" / "run_m3m_gcp_100k_guarded.py"
    command = [
        guard_python,
        "-B",
        str(guard),
        "--repo",
        str(base_repo),
        "--activation",
        str(base_activation),
        "--plan",
        str(plan),
        "--recipe-manifest",
        str(recipe_manifest),
        "--recipe",
        str(recipe_path),
        "--method-id",
        args.method_id,
        "--phase",
        "packet-release",
        "--run-root",
        str(Path(str(method["run_root"])).resolve()),
        "--dataset-root",
        str(Path(str(roots["dataset_root"])).resolve()),
        "--source-root",
        str(Path(str(source["root"])).resolve()),
        "--prior-root",
        str(Path(str(roots["prior_root"])).resolve()),
        "--failure-evidence",
        str(failure),
        "--packet-state",
        str(state_path),
        "--packet-set-root",
        str(packet_root),
        "--verification-report",
        str(verification_path),
        "--archive-manifest",
        str(archive_path),
    ]

    if intent_path.is_file():
        intent = require_json(intent_path)
        if (
            intent.get("schema") != "m3m_gcp_100k_lidar_packet_release_intent_v1"
            or intent.get("status") != "AUTHORIZED_TO_INVOKE_BASE_PACKET_RELEASE"
            or intent.get("method_id") != args.method_id
            or intent.get("three_track_activation_sha256") != sha256_file(activation_path)
            or intent.get("candidate_manifest_sha256") != sha256_file(candidate_path)
            or intent.get("scene_attempt_freeze_sha256")
            != candidate["scene_attempt_freeze"]["sha256"]
            or intent.get("methods_manifest_sha256")
            != candidate["methods_manifest"]["sha256"]
            or intent.get("recipe_sha256") != sha256_file(recipe_path)
            or intent.get("attempt_model_identity_sha256") != sha256_file(identity_path)
            or intent.get("lidar_dispatch_receipt_path") != str(dispatch_path)
            or intent.get("lidar_dispatch_receipt_sha256") != sha256_file(dispatch_path)
            or intent.get("packet_set_root") != str(packet_root)
            or intent.get("packet_state_path") != str(state_path)
            or intent.get("global_raw_packet_state_path") != str(global_state_path)
            or intent.get("authorized_targets_exact")
            != [str(packet_root), str(state_path), str(global_state_path)]
            or intent.get("lidar_method_result_path") != str(result_path)
            or intent.get("lidar_method_result_sha256") != sha256_file(result_path)
            or intent.get("lidar_verification_path") != str(verification_path)
            or intent.get("lidar_verification_sha256") != sha256_file(verification_path)
            or intent.get("lidar_archive_manifest_path") != str(archive_path)
            or intent.get("lidar_archive_manifest_sha256") != sha256_file(archive_path)
            or intent.get("base_release_command") != command
            or intent.get("base_release_command_sha256") != command_sha256(command)
            or intent.get("canonical_sha256") != canonical_sha256(intent)
        ):
            raise RuntimeError("existing LiDAR release intent mismatch")
        if not global_state_path.exists() and (
            packet_root.exists()
            or packet_root.is_symlink()
            or state_path.exists()
            or state_path.is_symlink()
        ):
            raise RuntimeError("LiDAR release continuation lost the global mutex early")
        if global_state_path.exists() or global_state_path.is_symlink():
            validate_active_raw_packet_state(
                global_state_path,
                activation_path=activation_path,
                candidate=candidate,
                method_id=args.method_id,
                track="lidar",
                recipe_sha256=sha256_file(recipe_path),
                attempt_model_identity_sha256=sha256_file(identity_path),
                packet_set_root=packet_root,
                track_packet_state_path=state_path,
            )
            if sha256_file(global_state_path) != intent.get(
                "global_raw_packet_state_sha256"
            ):
                raise RuntimeError("LiDAR release continuation global mutex SHA mismatch")
        validate_exact_lidar_archive(
            archive_path,
            archive_root,
            method_id=args.method_id,
            expected_scene_attempt_freeze_sha256=candidate["scene_attempt_freeze"][
                "sha256"
            ],
            require_sources=False,
        )
        recovery_result = require_json(result_path)
        recovery_verification = require_json(verification_path)
        if (
            recovery_result.get("schema") != "m3m_gcp_lidar_method_result_v1"
            or recovery_result.get("scene") != SCENE
            or recovery_result.get("method_id") != args.method_id
            or recovery_result.get("train_view_count") != 2196
            or recovery_result.get("summary_row", {}).get("status")
            != "COMPLETE_RANKED"
            or recovery_result.get("canonical_sha256")
            != canonical_sha256(recovery_result)
            or recovery_verification.get("status") != "PASS_VERIFIED_FORMAL_V1"
            or recovery_verification.get("scene") != SCENE
            or recovery_verification.get("method_id") != args.method_id
            or recovery_verification.get("method_result_sha256")
            != sha256_file(result_path)
            or recovery_verification.get("canonical_sha256")
            != canonical_sha256(recovery_verification)
        ):
            raise RuntimeError("existing LiDAR release intent result/archive gate mismatch")
        if packet_root.exists() or packet_root.is_symlink():
            if state_path.exists() and not state_path.is_symlink():
                if packet_root.is_symlink() or any(
                    path.is_symlink() for path in packet_root.rglob("*")
                ):
                    raise RuntimeError("LiDAR release continuation refuses packet symlinks")
                original_rows = {
                    str(row["path"]): row
                    for row in intent.get("packet_tree_before_delete", {}).get(
                        "files", []
                    )
                }
                current_rows = directory_tree_hash(packet_root).get("files", [])
                if any(
                    str(row["path"]) not in original_rows
                    or row.get("sha256")
                    != original_rows[str(row["path"])].get("sha256")
                    or row.get("bytes")
                    != original_rows[str(row["path"])].get("bytes")
                    for row in current_rows
                ):
                    raise RuntimeError("LiDAR release continuation packet subset mismatch")
                recovery_stdout = intent_path.parent / "base_release.recovery.stdout.log"
                recovery_stderr = intent_path.parent / "base_release.recovery.stderr.log"
                with recovery_stdout.open("xb") as stdout_handle, recovery_stderr.open(
                    "xb"
                ) as stderr_handle:
                    process = subprocess.run(
                        command, stdout=stdout_handle, stderr=stderr_handle, check=False
                    )
                if process.returncode != 0:
                    raise RuntimeError(
                        "unchanged base packet-release guard failed during authorized continuation"
                    )
            else:
                raise RuntimeError("inconsistent LiDAR release continuation targets")
        if state_path.exists() or state_path.is_symlink():
            if (
                state_path.is_symlink()
                or not state_path.is_file()
                or sha256_file(state_path) != intent.get("packet_state_sha256")
                or packet_root.exists()
                or packet_root.is_symlink()
            ):
                raise RuntimeError("LiDAR release continuation state identity mismatch")
            state_path.unlink()
        if global_state_path.exists() or global_state_path.is_symlink():
            if global_state_path.is_symlink():
                raise RuntimeError("LiDAR release continuation refuses symlinked global mutex")
            global_state_path.unlink()
        if (
            packet_root.exists()
            or packet_root.is_symlink()
            or state_path.exists()
            or state_path.is_symlink()
            or global_state_path.exists()
            or global_state_path.is_symlink()
        ):
            raise RuntimeError("LiDAR release continuation deletion postcondition failed")
        receipt: dict[str, Any] = {
            **intent,
            "schema": "m3m_gcp_100k_lidar_packet_deletion_receipt_v1",
            "status": "PASS_LIDAR_PACKET_DELETED_BY_BASE_GUARD",
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

    packet = require_json(packet_path)
    state = require_json(state_path)
    validate_active_raw_packet_state(
        global_state_path,
        activation_path=activation_path,
        candidate=candidate,
        method_id=args.method_id,
        track="lidar",
        recipe_sha256=sha256_file(recipe_path),
        attempt_model_identity_sha256=sha256_file(identity_path),
        packet_set_root=packet_root,
        track_packet_state_path=state_path,
    )
    result = require_json(result_path)
    verification = require_json(verification_path)
    if (
        dispatch.get("schema") != "m3m_gcp_100k_lidar_packet_dispatch_receipt_v1"
        or dispatch.get("status") != "PASS_LIDAR_PACKET_2196_DISPATCHED"
        or dispatch.get("method_id") != args.method_id
        or dispatch.get("three_track_activation_sha256") != sha256_file(activation_path)
        or dispatch.get("scene_attempt_freeze_sha256")
        != candidate["scene_attempt_freeze"]["sha256"]
        or dispatch.get("methods_manifest_sha256") != candidate["methods_manifest"]["sha256"]
        or dispatch.get("attempt_model_identity_sha256") != sha256_file(identity_path)
        or dispatch.get("packet_state_sha256") != sha256_file(state_path)
        or dispatch.get("packet_manifest_sha256") != sha256_file(packet_path)
        or dispatch.get("global_raw_packet_state_path") != str(global_state_path)
        or dispatch.get("global_raw_packet_state_sha256")
        != sha256_file(global_state_path)
        or dispatch.get("canonical_sha256") != canonical_sha256(dispatch)
        or state.get("method_id") != args.method_id
        or Path(str(state.get("packet_set_root", ""))).resolve() != packet_root
        or packet.get("scene") != SCENE
        or packet.get("rendered_view_count") != 2196
        or len(packet.get("depth_index", [])) != 2196
        or len(packet.get("packet_index", [])) != 2196
        or packet.get("model_content_hash") != current_model_content(method)
    ):
        raise RuntimeError("LiDAR packet dispatch/current model binding mismatch")
    if (
        result.get("schema") != "m3m_gcp_lidar_method_result_v1"
        or result.get("scene") != SCENE
        or result.get("method_id") != args.method_id
        or result.get("train_view_count") != 2196
        or result.get("summary_row", {}).get("status") != "COMPLETE_RANKED"
        or result.get("canonical_sha256") != canonical_sha256(result)
        or result.get("packet_manifest_sha256") != sha256_file(packet_path)
        or result.get("scene_attempt_freeze_sha256")
        != candidate["scene_attempt_freeze"]["sha256"]
        or result.get("formal_methods_manifest_sha256")
        != candidate["methods_manifest"]["sha256"]
        or result.get("model_checkpoint_sha256")
        != method["attempt_model_identity_sha256"]
        or result.get("recipe_sha256") != sha256_file(recipe_path)
        or verification.get("schema") != "m3m_gcp_lidar_formal_verification_v1"
        or verification.get("status") != "PASS_VERIFIED_FORMAL_V1"
        or verification.get("scene") != SCENE
        or verification.get("method_id") != args.method_id
        or verification.get("method_result_sha256") != sha256_file(result_path)
        or verification.get("scene_attempt_freeze_sha256")
        != candidate["scene_attempt_freeze"]["sha256"]
        or verification.get("canonical_sha256") != canonical_sha256(verification)
    ):
        raise RuntimeError("LiDAR result/independent-verifier binding mismatch")
    archive = validate_exact_lidar_archive(
        archive_path,
        archive_root,
        method_id=args.method_id,
        expected_scene_attempt_freeze_sha256=candidate["scene_attempt_freeze"]["sha256"],
        require_sources=True,
    )
    if (
        archive.get("scene") != SCENE
        or archive.get("method_id") != args.method_id
    ):
        raise RuntimeError("LiDAR lightweight archive gate mismatch")

    intent: dict[str, Any] = {
        "schema": "m3m_gcp_100k_lidar_packet_release_intent_v1",
        "status": "AUTHORIZED_TO_INVOKE_BASE_PACKET_RELEASE",
        "scene": SCENE,
        "method_id": args.method_id,
        "three_track_activation_path": str(activation_path),
        "three_track_activation_sha256": sha256_file(activation_path),
        "candidate_manifest_sha256": sha256_file(candidate_path),
        "scene_attempt_freeze_sha256": candidate["scene_attempt_freeze"]["sha256"],
        "methods_manifest_sha256": candidate["methods_manifest"]["sha256"],
        "recipe_sha256": sha256_file(recipe_path),
        "attempt_model_identity_sha256": sha256_file(identity_path),
        "lidar_dispatch_receipt_path": str(dispatch_path),
        "lidar_dispatch_receipt_sha256": sha256_file(dispatch_path),
        "packet_set_root": str(packet_root),
        "packet_state_path": str(state_path),
        "packet_state_sha256": sha256_file(state_path),
        "global_raw_packet_state_path": str(global_state_path),
        "global_raw_packet_state_sha256": sha256_file(global_state_path),
        "packet_manifest_sha256": sha256_file(packet_path),
        "packet_tree_before_delete": directory_tree_hash(packet_root),
        "lidar_method_result_sha256": sha256_file(result_path),
        "lidar_method_result_path": str(result_path),
        "lidar_verification_sha256": sha256_file(verification_path),
        "lidar_verification_path": str(verification_path),
        "lidar_archive_manifest_sha256": sha256_file(archive_path),
        "lidar_archive_manifest_path": str(archive_path),
        "base_release_command": command,
        "base_release_command_sha256": command_sha256(command),
        "authorized_targets_exact": [
            str(packet_root),
            str(state_path),
            str(global_state_path),
        ],
    }
    intent["canonical_sha256"] = canonical_sha256(intent)
    write_exclusive(intent_path, intent)
    stdout_path = intent_path.parent / "base_release.stdout.log"
    stderr_path = intent_path.parent / "base_release.stderr.log"
    with stdout_path.open("xb") as stdout_handle, stderr_path.open("xb") as stderr_handle:
        process = subprocess.run(command, stdout=stdout_handle, stderr=stderr_handle, check=False)
    if process.returncode != 0:
        raise RuntimeError(
            f"unchanged base packet-release guard failed with exit code {process.returncode}; "
            f"stdout={stdout_path}; stderr={stderr_path}"
        )
    global_state_path.unlink()
    if (
        packet_root.exists()
        or packet_root.is_symlink()
        or state_path.exists()
        or state_path.is_symlink()
        or global_state_path.exists()
        or global_state_path.is_symlink()
    ):
        raise RuntimeError("base packet-release deletion postcondition failed")
    receipt: dict[str, Any] = {
        **intent,
        "schema": "m3m_gcp_100k_lidar_packet_deletion_receipt_v1",
        "status": "PASS_LIDAR_PACKET_DELETED_BY_BASE_GUARD",
        "release_intent_path": str(intent_path),
        "release_intent_sha256": sha256_file(intent_path),
        "base_release_stdout_sha256": sha256_file(stdout_path),
        "base_release_stderr_sha256": sha256_file(stderr_path),
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
