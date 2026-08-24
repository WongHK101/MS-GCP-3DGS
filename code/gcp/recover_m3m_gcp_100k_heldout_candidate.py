#!/usr/bin/env python3
"""Recover 100K held-out LiDAR evaluation from complete retained packets.

This entrypoint never invokes a renderer.  It reuses exact packet sets retained
after an evaluator-only failure, updates only the benchmark evaluator checkout
identity, runs the common LiDAR evaluator, and performs the normal rolling
packet cleanup only after a successful COMPLETE_RANKED result.
"""

from __future__ import annotations

import argparse
import copy
import json
import traceback
from pathlib import Path
from typing import Any

from m3m_gcp_lidar_artifacts import canonical_sha256, command_sha256, sha256_file
from rgb_quality_contract import validate_benchmark_checkout
from run_m3m_gcp_100k_success_geometry_plan import (
    cleanup_packet_arrays,
    now,
    run_phase,
    terminal_result,
    write_json,
)


EXPECTED_PLAN_SCHEMA = "m3m_gcp_100k_success_geometry_execution_plan_v1"
EXPECTED_TRACK = "heldout_candidate"
EVALUATOR_NAME = "evaluate_m3m_gcp_lidar_success_v1.py"


def identity(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def replace_unique_value(argv: list[str], option: str, value: str) -> None:
    indices = [index for index, token in enumerate(argv) if token == option]
    if len(indices) != 1 or indices[0] + 1 >= len(argv):
        raise ValueError(f"expected exactly one value-bearing {option}")
    argv[indices[0] + 1] = value


def recovery_evaluate_phase(
    source: dict[str, Any],
    *,
    repo: Path,
    benchmark_commit: str,
    benchmark_tree: str,
    log_root: Path,
) -> dict[str, Any]:
    phase = copy.deepcopy(source)
    argv = [str(value) for value in phase["argv"]]
    evaluator_indices = [
        index for index, value in enumerate(argv) if Path(value).name == EVALUATOR_NAME
    ]
    if len(evaluator_indices) != 1:
        raise ValueError("source phase does not bind one exact LiDAR evaluator")
    argv[evaluator_indices[0]] = str((repo / "code/gcp" / EVALUATOR_NAME).resolve())
    replace_unique_value(argv, "--repo", str(repo.resolve()))
    replace_unique_value(argv, "--benchmark-commit", benchmark_commit)
    replace_unique_value(argv, "--benchmark-tree", benchmark_tree)
    phase["argv"] = argv
    phase["argv_sha256"] = command_sha256(argv)
    phase["working_directory"] = str(repo.resolve())
    phase["stdout"] = str((log_root / "stdout.log").resolve())
    phase["stderr"] = str((log_root / "stderr.log").resolve())
    return phase


def validate_packet_set(packet_root: Path) -> dict[str, Any]:
    packet_root = packet_root.resolve()
    manifest_path = packet_root / "depth_export_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("depth_index", [])
    if int(manifest.get("rendered_view_count", -1)) != 314 or len(rows) != 314:
        raise ValueError("recovery requires an exact complete 314-view packet manifest")
    packet_paths = [Path(str(row["packet_path"])).resolve() for row in rows]
    if len(set(packet_paths)) != 314:
        raise ValueError("recovery packet paths are not unique")
    for path in packet_paths:
        if path.parent != packet_root or not path.is_file() or path.is_symlink():
            raise ValueError(f"missing or unsafe retained packet: {path}")
    return {
        "manifest": identity(manifest_path),
        "camera_sets": manifest.get("camera_sets"),
        "packet_count": len(packet_paths),
        "packet_bytes": sum(path.stat().st_size for path in packet_paths),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--benchmark-commit", required=True)
    parser.add_argument("--benchmark-tree", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.repo = args.repo.expanduser().resolve()
    args.plan = args.plan.expanduser().resolve()
    args.receipt = args.receipt.expanduser().resolve()
    if args.receipt.exists() or args.receipt.is_symlink():
        raise FileExistsError(args.receipt)
    benchmark = validate_benchmark_checkout(
        benchmark_repo=args.repo,
        expected_commit=args.benchmark_commit,
        expected_tree=args.benchmark_tree,
        entrypoint=Path(__file__).resolve(),
    )
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if (
        plan.get("schema") != EXPECTED_PLAN_SCHEMA
        or plan.get("status") != "READY"
        or plan.get("surface_sampling_track") != EXPECTED_TRACK
        or plan.get("canonical_sha256") != canonical_sha256(plan)
    ):
        raise ValueError("held-out candidate source plan identity mismatch")
    if len(args.methods) != len(set(args.methods)):
        raise ValueError("duplicate recovery methods")
    jobs_by_id = {str(row["method_id"]): row for row in plan["jobs"]}
    unknown = set(args.methods) - set(jobs_by_id)
    if unknown:
        raise ValueError(f"unknown recovery methods: {sorted(unknown)}")

    receipt: dict[str, Any] = {
        "schema": "m3m_gcp_100k_heldout_candidate_evaluation_recovery_receipt_v1",
        "status": "RUNNING",
        "started_at": now(),
        "benchmark_repository": benchmark,
        "source_plan": identity(args.plan),
        "recovery_policy": {
            "packet_export_forbidden": True,
            "existing_complete_packets_required": True,
            "cleanup_only_after_returncode_zero_and_complete_ranked": True,
            "models_changed": False,
        },
        "selected_methods": list(args.methods),
        "jobs": [],
    }
    write_json(args.receipt, receipt)
    for method_id in args.methods:
        source_job = jobs_by_id[method_id]
        track_job = source_job["lidar"]
        run_root = Path(str(source_job["run_root"])).resolve()
        packet_root = Path(str(track_job["packet_root"])).resolve()
        output_root = Path(str(track_job["output_root"])).resolve()
        row: dict[str, Any] = {
            "method_id": method_id,
            "started_at": now(),
            "packet_root": str(packet_root),
            "output_root": str(output_root),
            "models_changed": False,
        }
        try:
            complete, result_status = terminal_result("lidar", output_root)
            if complete:
                row["status"] = "SKIPPED_ALREADY_COMPLETE_RANKED"
                row["result_status"] = result_status
                if any(packet_root.glob("*.npz")):
                    row["packet_cleanup"] = cleanup_packet_arrays(
                        packet_root, run_root, "recovery_found_terminal_result"
                    )
            else:
                if output_root.exists() or output_root.is_symlink():
                    raise FileExistsError(
                        f"refusing nonterminal pre-existing recovery output: {output_root}"
                    )
                row["retained_packet_set"] = validate_packet_set(packet_root)
                phase = recovery_evaluate_phase(
                    track_job["evaluate"],
                    repo=args.repo,
                    benchmark_commit=args.benchmark_commit,
                    benchmark_tree=args.benchmark_tree,
                    log_root=args.receipt.parent
                    / "geometry_logs_heldout_candidate_recovery_v1"
                    / method_id,
                )
                row["evaluate_command"] = {
                    "argv": phase["argv"],
                    "argv_sha256": phase["argv_sha256"],
                    "working_directory": phase["working_directory"],
                }
                evaluation = run_phase(phase, "lidar_evaluate_recovery")
                row["evaluation"] = evaluation
                complete, result_status = terminal_result("lidar", output_root)
                row["result_status"] = result_status
                if evaluation["returncode"] == 0 and complete:
                    row["status"] = "COMPLETE_RANKED_RECOVERED"
                    row["metrics"] = identity(output_root / "metrics.json")
                    row["packet_cleanup"] = cleanup_packet_arrays(
                        packet_root,
                        run_root,
                        "evaluation_only_recovery_successful_terminal",
                    )
                else:
                    row["status"] = "EVALUATION_RECOVERY_FAILED_PACKETS_RETAINED"
                    row["packet_cleanup"] = {
                        "status": "PACKET_ARRAYS_RETAINED_FOR_RECOVERY",
                        "packet_root": str(packet_root),
                    }
        except Exception as error:
            row.update(
                {
                    "status": "EVALUATION_RECOVERY_FAILED_PACKETS_RETAINED",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
            )
        row["finished_at"] = now()
        receipt["jobs"].append(row)
        write_json(args.receipt, receipt)

    accepted = {"COMPLETE_RANKED_RECOVERED", "SKIPPED_ALREADY_COMPLETE_RANKED"}
    receipt["status"] = (
        "COMPLETE_ALL_SELECTED_TERMINAL"
        if all(row["status"] in accepted for row in receipt["jobs"])
        else "COMPLETE_WITH_RECOVERY_FAILURES"
    )
    receipt["finished_at"] = now()
    receipt["canonical_sha256"] = canonical_sha256(receipt)
    write_json(args.receipt, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "COMPLETE_ALL_SELECTED_TERMINAL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
