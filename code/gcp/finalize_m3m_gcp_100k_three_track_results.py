#!/usr/bin/env python3
"""Seal the 100K three-track result only after every READY method has both deletion receipts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from m3m_gcp_lidar_artifacts import canonical_sha256, sha256_file
from m3m_gcp_100k_three_track_runtime import validate_addendum_runtime
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


def parse_bindings(values: list[str], label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        method, separator, raw_path = value.partition("=")
        if not separator or not method or not raw_path or method in result:
            raise RuntimeError(f"invalid or duplicate {label} binding: {value}")
        result[method] = Path(raw_path).resolve()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--rgb-execution-plan", type=Path, required=True)
    parser.add_argument("--gcp-deletion-receipt", action="append", default=[])
    parser.add_argument("--lidar-deletion-receipt", action="append", default=[])
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
    if (
        candidate.get("canonical_sha256") != activation["candidate_manifest_canonical_sha256"]
        or canonical_sha256(candidate) != activation["candidate_manifest_canonical_sha256"]
    ):
        raise RuntimeError("activation/candidate binding mismatch")
    registry_row = candidate["rgb_registry"]
    registry_path = Path(str(registry_row["path"])).resolve()
    registry = require_json(registry_path, str(registry_row["sha256"]))
    ready = [str(value) for value in registry.get("ready_method_ids", [])]
    methods = {str(row["method_id"]): row for row in registry.get("methods", [])}
    if not ready or set(ready) != set(methods):
        raise RuntimeError("activated READY registry mismatch")
    validate_addendum_runtime(
        activation=activation,
        candidate=candidate,
        registry=registry,
        executing_file=Path(__file__),
    )
    base_repo = Path(str(candidate["base_checkout"]["path"])).resolve()
    if (
        subprocess.check_output(["git", "-C", str(base_repo), "rev-parse", "HEAD"], text=True).strip()
        != candidate["base_checkout"]["commit"]
        or subprocess.check_output(
            ["git", "-C", str(base_repo), "show", "-s", "--format=%T", "HEAD"], text=True
        ).strip()
        != candidate["base_checkout"]["tree"]
        or subprocess.check_output(["git", "-C", str(base_repo), "status", "--porcelain"], text=True).strip()
    ):
        raise RuntimeError("finalization base checkout identity mismatch")

    rgb_plan_path = args.rgb_execution_plan.resolve()
    rgb_plan = require_json(rgb_plan_path)
    if (
        rgb_plan.get("schema") != "m3m_gcp_native_quarter_rgb_quality_100k_execution_plan_v1"
        or rgb_plan.get("status") != "ACTIVE_FROZEN"
        or rgb_plan.get("formal_execution_authorized") is not True
        or rgb_plan.get("scene") != SCENE
        or rgb_plan.get("three_track_activation_sha256") != sha256_file(activation_path)
        or rgb_plan.get("candidate_manifest_sha256") != sha256_file(candidate_path)
        or rgb_plan.get("scene_attempt_freeze_sha256")
        != candidate["scene_attempt_freeze"]["sha256"]
        or rgb_plan.get("registry_sha256") != sha256_file(registry_path)
        or rgb_plan.get("method_order") != ready
        or rgb_plan.get("canonical_sha256") != canonical_sha256(rgb_plan)
    ):
        raise RuntimeError("RGB execution plan/current activation binding mismatch")
    jobs = {str(row["method_id"]): row for row in rgb_plan.get("jobs", [])}
    if set(jobs) != set(ready):
        raise RuntimeError("RGB execution plan READY coverage mismatch")

    runtime_root = Path(str(candidate["candidate_output_root"])).resolve().parent
    gcp_receipts = parse_bindings(args.gcp_deletion_receipt, "GCP receipt")
    lidar_receipts = parse_bindings(args.lidar_deletion_receipt, "LiDAR receipt")
    if set(gcp_receipts) != set(ready) - {"3dgs_original"}:
        raise RuntimeError("GCP deletion-receipt coverage must equal new READY methods")
    if set(lidar_receipts) != set(ready):
        raise RuntimeError("LiDAR deletion-receipt coverage must equal all READY methods")

    rows: list[dict[str, Any]] = []
    for method_id in ready:
        method = methods[method_id]
        recipe_path = Path(str(method["recipe_path"])).resolve()
        recipe = require_json(recipe_path, str(method["recipe_sha256"]))
        identity_path = Path(str(method["attempt_model_identity_path"])).resolve()
        bound_recipe = dict(recipe)
        bound_recipe["_recipe_path"] = str(recipe_path)
        identity = validate_model_identity_bundle(
            manifest_path=identity_path,
            method_id=method_id,
            run_root=Path(str(method["run_root"])).resolve(),
            recipe=bound_recipe,
            repo=base_repo,
        )
        if (
            sha256_file(identity_path) != method["attempt_model_identity_sha256"]
            or identity["canonical_sha256"]
            != method["attempt_model_identity_canonical_sha256"]
        ):
            raise RuntimeError(f"{method_id}: final frozen model identity mismatch")
        job = jobs[method_id]
        artifact_root = Path(str(job["artifact_root"])).resolve()
        if artifact_root != Path(str(method["formal_output_root"])).resolve():
            raise RuntimeError(f"{method_id}: RGB artifact root mismatch")
        summary_path = artifact_root / "metrics" / "rgb_quality_summary.json"
        manifest_path = artifact_root / "metrics" / "evaluator_manifest.json"
        render_manifest_path = artifact_root / "rgb_render_manifest.json"
        summary = require_json(summary_path)
        manifest = require_json(manifest_path)
        render_manifest = require_json(render_manifest_path)
        if (
            summary.get("schema") != "m3m_gcp_native_quarter_rgb_quality_summary_v1"
            or summary.get("status") != "COMPLETE_RANKED"
            or summary.get("scene") != SCENE
            or summary.get("method_id") != method_id
            or summary.get("expected_test_view_count") != 314
            or summary.get("evaluated_test_view_count") != 314
            or summary.get("complete_test_coverage") is not True
            or summary.get("ranking_eligible") is not True
            or summary.get("formal_execution") is not True
            or render_manifest.get("schema") != "m3m_gcp_native_quarter_rgb_render_manifest_v1"
            or render_manifest.get("contract_status") != "ACTIVE_FROZEN"
            or render_manifest.get("scene") != SCENE
            or render_manifest.get("method_id") != method_id
            or render_manifest.get("required_test_view_count") != 314
            or render_manifest.get("rendered_test_view_count") != 314
            or render_manifest.get("complete_test_coverage") is not True
            or manifest.get("schema") != "m3m_gcp_native_quarter_rgb_evaluator_manifest_v1"
            or manifest.get("status") != "PASS_FORMAL"
            or manifest.get("method_id") != method_id
            or manifest.get("registry_sha256") != sha256_file(registry_path)
            or manifest.get("render_manifest_sha256") != sha256_file(render_manifest_path)
            or manifest.get("outputs_sha256", {}).get("rgb_quality_summary.json")
            != sha256_file(summary_path)
        ):
            raise RuntimeError(f"{method_id}: formal RGB result binding mismatch")

        if method_id == "3dgs_original":
            legacy_row = candidate["legacy_3dgs_gcp_adoption"]
            gcp_path = Path(str(legacy_row["path"])).resolve()
            gcp = require_json(gcp_path, str(legacy_row["sha256"]))
            if (
                gcp.get("status") != "PASS_LEGACY_GCP_ADOPTION_CANDIDATE"
                or gcp.get("method_id") != method_id
                or gcp.get("scene_attempt_freeze_sha256")
                != candidate["scene_attempt_freeze"]["sha256"]
                or gcp.get("canonical_sha256") != legacy_row["canonical_sha256"]
            ):
                raise RuntimeError("3DGS legacy GCP final binding mismatch")
            gcp_track = {
                "kind": "LEGACY_GCP_ADOPTION",
                "path": str(gcp_path),
                "sha256": sha256_file(gcp_path),
                "status": gcp["adopted_result"]["status"],
            }
        else:
            gcp_path = gcp_receipts[method_id]
            expected_gcp = runtime_root / "gcp-packet-release" / method_id / "deletion_receipt.json"
            gcp = require_json(gcp_path)
            if (
                gcp_path != expected_gcp
                or gcp.get("schema") != "m3m_gcp_100k_gcp_packet_deletion_receipt_v1"
                or gcp.get("status") != "PASS_GCP_PACKET_DELETED"
                or gcp.get("method_id") != method_id
                or gcp.get("three_track_activation_sha256") != sha256_file(activation_path)
                or gcp.get("attempt_model_identity_sha256")
                != method["attempt_model_identity_sha256"]
                or gcp.get("canonical_sha256") != canonical_sha256(gcp)
                or not Path(str(gcp["gcp_evaluation_summary_path"])).is_file()
                or sha256_file(Path(str(gcp["gcp_evaluation_summary_path"])))
                != gcp["gcp_evaluation_summary_sha256"]
            ):
                raise RuntimeError(f"{method_id}: GCP deletion/final-result receipt mismatch")
            gcp_track = {
                "kind": "NEW_GCP_RESULT_AND_DELETION_RECEIPT",
                "path": str(gcp_path),
                "sha256": sha256_file(gcp_path),
                "summary_path": gcp["gcp_evaluation_summary_path"],
                "summary_sha256": gcp["gcp_evaluation_summary_sha256"],
            }

        lidar_path = lidar_receipts[method_id]
        expected_lidar = runtime_root / "lidar-packet-release" / method_id / "deletion_receipt.json"
        lidar = require_json(lidar_path)
        result_path = Path(str(lidar.get("lidar_method_result_path", ""))).resolve()
        if (
            lidar_path != expected_lidar
            or lidar.get("schema") != "m3m_gcp_100k_lidar_packet_deletion_receipt_v1"
            or lidar.get("status") != "PASS_LIDAR_PACKET_DELETED_BY_BASE_GUARD"
            or lidar.get("method_id") != method_id
            or lidar.get("three_track_activation_sha256") != sha256_file(activation_path)
            or lidar.get("attempt_model_identity_sha256")
            != method["attempt_model_identity_sha256"]
            or lidar.get("canonical_sha256") != canonical_sha256(lidar)
            or not result_path.is_file()
            or sha256_file(result_path) != lidar.get("lidar_method_result_sha256")
        ):
            raise RuntimeError(f"{method_id}: LiDAR deletion/final-result receipt mismatch")
        rows.append(
            {
                "method_id": method_id,
                "attempt_model_identity_sha256": method["attempt_model_identity_sha256"],
                "rgb": {
                    "status": summary["status"],
                    "summary_path": str(summary_path),
                    "summary_sha256": sha256_file(summary_path),
                    "evaluator_manifest_path": str(manifest_path),
                    "evaluator_manifest_sha256": sha256_file(manifest_path),
                },
                "gcp": gcp_track,
                "lidar": {
                    "status": "PASS_VERIFIED_FORMAL_V1",
                    "deletion_receipt_path": str(lidar_path),
                    "deletion_receipt_sha256": sha256_file(lidar_path),
                    "result_path": str(result_path),
                    "result_sha256": sha256_file(result_path),
                },
            }
        )

    active_paths = [
        runtime_root / "gcp-packet-scratch" / method_id for method_id in ready if method_id != "3dgs_original"
    ]
    active_paths.append(runtime_root / "gcp-packet-scratch" / "ACTIVE_GCP_PACKET_STATE.json")
    for method_id in ready:
        recipe = require_json(
            Path(str(methods[method_id]["recipe_path"])),
            str(methods[method_id]["recipe_sha256"]),
        )
        active_paths.extend(
            (
                Path(str(recipe["authorized_packet_set_root"])).resolve(),
                Path(str(recipe["authorized_packet_state"])).resolve(),
            )
        )
    if any(path.exists() or path.is_symlink() for path in active_paths):
        raise RuntimeError("finalization refuses remaining active raw packet sets")

    output = args.output.resolve()
    expected_output = Path(str(candidate["formal_results_root"])).resolve() / "three_track_final_manifest_v1.json"
    if output != expected_output:
        raise RuntimeError("final result manifest path differs from activated namespace")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    payload: dict[str, Any] = {
        "schema": "m3m_gcp_100k_three_track_final_manifest_v1",
        "status": "PASS_ALL_READY_METHODS_THREE_TRACKS_FROZEN",
        "scene": SCENE,
        "three_track_activation_path": str(activation_path),
        "three_track_activation_sha256": sha256_file(activation_path),
        "candidate_manifest_sha256": sha256_file(candidate_path),
        "scene_attempt_freeze_sha256": candidate["scene_attempt_freeze"]["sha256"],
        "methods_manifest_sha256": candidate["methods_manifest"]["sha256"],
        "rgb_execution_plan_path": str(rgb_plan_path),
        "rgb_execution_plan_sha256": sha256_file(rgb_plan_path),
        "ready_method_ids": ready,
        "method_count": len(rows),
        "methods": rows,
        "all_new_gcp_packet_deletion_receipts_present": True,
        "all_lidar_packet_deletion_receipts_present": True,
        "all_active_raw_packet_sets_absent": True,
    }
    payload["canonical_sha256"] = canonical_sha256(payload)
    write_exclusive(output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
