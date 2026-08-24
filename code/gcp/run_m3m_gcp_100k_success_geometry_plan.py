#!/usr/bin/env python3
"""Run promoted 100K GCP/LiDAR jobs sequentially with rolling packet cleanup."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from m3m_gcp_lidar_artifacts import canonical_sha256, sha256_file


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run_phase(spec: dict[str, Any], phase_name: str) -> dict[str, Any]:
    stdout_path = Path(str(spec["stdout"])).resolve()
    stderr_path = Path(str(spec["stderr"])).resolve()
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.update(
        {str(key): str(value) for key, value in spec.get("environment", {}).items()}
    )
    started = now()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        completed = subprocess.run(
            [str(value) for value in spec["argv"]],
            cwd=str(spec["working_directory"]),
            env=environment,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    return {
        "phase": phase_name,
        "started_at": started,
        "finished_at": now(),
        "returncode": int(completed.returncode),
        "stdout": {"path": str(stdout_path), "sha256": sha256_file(stdout_path)},
        "stderr": {"path": str(stderr_path), "sha256": sha256_file(stderr_path)},
    }


def terminal_result(track: str, output_root: Path) -> tuple[bool, str]:
    result_path = (
        output_root / "evaluation_summary.json"
        if track == "gcp"
        else output_root / "metrics.json"
    )
    if not result_path.is_file():
        return False, "MISSING_RESULT"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    status = str(payload.get("status", ""))
    allowed = (
        {"COMPLETE_RANKED", "INCOMPLETE_UNRANKED"}
        if track == "gcp"
        else {"COMPLETE_RANKED"}
    )
    return status in allowed, status or "UNKNOWN_RESULT_STATUS"


def cleanup_packet_arrays(packet_root: Path, run_root: Path, reason: str) -> dict[str, Any]:
    packet_root = packet_root.resolve()
    run_root = run_root.resolve()
    formal_root = run_root / "formal_evaluation"
    if packet_root.parent != formal_root or packet_root.name not in {
        "gcp_packets_100k_success_v1",
        "gcp_packets_100k_success_v2",
        "gcp_packets_100k_success_v3",
        "lidar_packets_100k_success_v1",
    }:
        raise ValueError(f"refusing packet cleanup outside exact formal roots: {packet_root}")
    if packet_root.is_symlink():
        raise ValueError(f"refusing symlink packet cleanup: {packet_root}")
    manifest_path = packet_root / "depth_export_manifest.json"
    manifest_identity = None
    manifest_rows: dict[str, dict[str, Any]] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_identity = {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        }
        manifest_rows = {
            str(Path(str(row.get("packet_path", ""))).resolve()): row
            for row in manifest.get("depth_index", [])
        }
    removed: list[dict[str, Any]] = []
    for path in sorted(packet_root.glob("*.npz")):
        resolved = path.resolve()
        if resolved.parent != packet_root or path.is_symlink() or not path.is_file():
            raise ValueError(f"unsafe packet member: {path}")
        row = manifest_rows.get(str(resolved), {})
        removed.append(
            {
                "path": str(resolved),
                "bytes": path.stat().st_size,
                "manifest_sha256": row.get("packet_sha256"),
            }
        )
        path.unlink()
    receipt = {
        "schema": "m3m_gcp_100k_packet_array_cleanup_receipt_v1",
        "status": "PACKET_ARRAYS_REMOVED_LIGHTWEIGHT_EVIDENCE_RETAINED",
        "created_at": now(),
        "reason": reason,
        "packet_root": str(packet_root),
        "manifest": manifest_identity,
        "removed_file_count": len(removed),
        "removed_bytes": sum(int(row["bytes"]) for row in removed),
        "removed": removed,
        "models_changed": False,
    }
    receipt["canonical_sha256"] = canonical_sha256(receipt)
    write_json(packet_root / "PACKET_ARRAY_CLEANUP_RECEIPT.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--methods", nargs="*")
    parser.add_argument("--tracks", nargs="+", choices=("gcp", "lidar"), required=True)
    args = parser.parse_args()
    plan_path = args.plan.expanduser().resolve()
    receipt_path = args.receipt.expanduser().resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (
        plan.get("schema") != "m3m_gcp_100k_success_geometry_execution_plan_v1"
        or plan.get("status") != "READY"
        or plan.get("canonical_sha256") != canonical_sha256(plan)
    ):
        raise ValueError("100K success geometry plan identity mismatch")
    selected_methods = set(args.methods or plan["method_order"])
    unknown = selected_methods - set(plan["method_order"])
    if unknown:
        raise ValueError(f"unknown method IDs: {sorted(unknown)}")
    selected_tracks = [track for track in plan["track_order"] if track in args.tracks]
    receipt: dict[str, Any] = {
        "schema": "m3m_gcp_100k_success_geometry_execution_receipt_v1",
        "status": "RUNNING",
        "started_at": now(),
        "plan_path": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "selected_methods": [
            method for method in plan["method_order"] if method in selected_methods
        ],
        "selected_tracks": selected_tracks,
        "jobs": [],
    }
    write_json(receipt_path, receipt)
    for job in plan["jobs"]:
        method_id = str(job["method_id"])
        if method_id not in selected_methods:
            continue
        for track in selected_tracks:
            track_job = job[track]
            run_root = Path(str(job["run_root"])).resolve()
            packet_root = Path(str(track_job["packet_root"])).resolve()
            output_root = Path(str(track_job["output_root"])).resolve()
            row: dict[str, Any] = {
                "method_id": method_id,
                "track": track,
                "started_at": now(),
                "packet_root": str(packet_root),
                "output_root": str(output_root),
                "phases": [],
            }
            complete, result_status = terminal_result(track, output_root)
            if complete:
                row["status"] = "SKIPPED_ALREADY_TERMINAL"
                row["result_status"] = result_status
                if packet_root.is_dir() and any(packet_root.glob("*.npz")):
                    row["packet_cleanup"] = cleanup_packet_arrays(
                        packet_root, run_root, "resume_found_terminal_result"
                    )
            elif packet_root.exists() or packet_root.is_symlink():
                row["status"] = "INCOMPLETE_PREEXISTING_PACKET_UNRANKED"
            elif output_root.exists() or output_root.is_symlink():
                row["status"] = "INCOMPLETE_PREEXISTING_OUTPUT_UNRANKED"
            else:
                packet_phase = run_phase(track_job["packet"], f"{track}_packet")
                row["phases"].append(packet_phase)
                if packet_phase["returncode"] == 0:
                    evaluate_phase = run_phase(
                        track_job["evaluate"], f"{track}_evaluate"
                    )
                    row["phases"].append(evaluate_phase)
                    complete, result_status = terminal_result(track, output_root)
                    row["result_status"] = result_status
                    if evaluate_phase["returncode"] == 0 and complete:
                        row["status"] = (
                            "COMPLETE_RANKED"
                            if result_status == "COMPLETE_RANKED"
                            else "COMPLETE_UNRANKED"
                        )
                    else:
                        row["status"] = "EVALUATION_FAILED_UNRANKED"
                    row["packet_cleanup"] = cleanup_packet_arrays(
                        packet_root,
                        run_root,
                        "evaluator_process_terminal",
                    )
                else:
                    row["status"] = "PACKET_EXPORT_FAILED_UNRANKED"
                    if packet_root.is_dir():
                        row["packet_cleanup"] = cleanup_packet_arrays(
                            packet_root,
                            run_root,
                            "packet_export_process_terminal_failure",
                        )
            row["finished_at"] = now()
            receipt["jobs"].append(row)
            write_json(receipt_path, receipt)
    statuses = [row["status"] for row in receipt["jobs"]]
    accepted = {"COMPLETE_RANKED", "COMPLETE_UNRANKED", "SKIPPED_ALREADY_TERMINAL"}
    receipt["status"] = (
        "COMPLETE_ALL_SELECTED_TERMINAL"
        if statuses and all(status in accepted for status in statuses)
        else "COMPLETE_WITH_EXECUTION_FAILURES"
    )
    receipt["finished_at"] = now()
    receipt["canonical_sha256"] = canonical_sha256(receipt)
    write_json(receipt_path, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
