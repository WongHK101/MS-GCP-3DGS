#!/usr/bin/env python3
"""Run the promoted 100K RGB jobs sequentially with resumable receipts."""

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


def run_phase(job: dict[str, Any], phase: str) -> dict[str, Any]:
    spec = job[phase]
    stdout_path = Path(str(spec["stdout"])).resolve()
    stderr_path = Path(str(spec["stderr"])).resolve()
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.update({str(key): str(value) for key, value in spec.get("environment", {}).items()})
    environment.setdefault("CUDA_VISIBLE_DEVICES", "0")
    environment.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    environment.setdefault("PYTHONHASHSEED", "0")
    environment.setdefault("PYTHONUNBUFFERED", "1")
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
        "phase": phase,
        "started_at": started,
        "finished_at": now(),
        "returncode": int(completed.returncode),
        "stdout": {"path": str(stdout_path), "sha256": sha256_file(stdout_path)},
        "stderr": {"path": str(stderr_path), "sha256": sha256_file(stderr_path)},
    }


def completed_summary(artifact_root: Path) -> bool:
    summary = artifact_root / "metrics/rgb_quality_summary.json"
    if not summary.is_file():
        return False
    payload = json.loads(summary.read_text(encoding="utf-8"))
    return payload.get("status") == "COMPLETE_RANKED" and payload.get(
        "complete_test_coverage"
    ) is True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--methods", nargs="*")
    args = parser.parse_args()
    plan_path = args.plan.expanduser().resolve()
    receipt_path = args.receipt.expanduser().resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (
        plan.get("schema")
        != "m3m_gcp_native_quarter_rgb_quality_100k_success_execution_plan_v1"
        or plan.get("status") != "READY"
        or plan.get("canonical_sha256") != canonical_sha256(plan)
    ):
        raise RuntimeError("100K success RGB plan identity mismatch")
    selected = set(args.methods or plan["method_order"])
    unknown = selected - set(plan["method_order"])
    if unknown:
        raise ValueError(f"unknown method IDs: {sorted(unknown)}")

    receipt: dict[str, Any] = {
        "schema": "m3m_gcp_100k_success_rgb_execution_receipt_v1",
        "status": "RUNNING",
        "started_at": now(),
        "plan_path": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "selected_methods": [name for name in plan["method_order"] if name in selected],
        "jobs": [],
    }
    write_json(receipt_path, receipt)
    for job in plan["jobs"]:
        method_id = str(job["method_id"])
        if method_id not in selected:
            continue
        artifact_root = Path(str(job["artifact_root"])).resolve()
        job_receipt: dict[str, Any] = {
            "method_id": method_id,
            "artifact_root": str(artifact_root),
            "started_at": now(),
            "phases": [],
        }
        if completed_summary(artifact_root):
            job_receipt.update({"status": "SKIPPED_ALREADY_COMPLETE", "finished_at": now()})
        elif artifact_root.exists() or artifact_root.is_symlink():
            job_receipt.update(
                {
                    "status": "INCOMPLETE_PREEXISTING_OUTPUT_UNRANKED",
                    "finished_at": now(),
                }
            )
        else:
            artifact_root.mkdir(parents=True)
            render = run_phase(job, "render")
            job_receipt["phases"].append(render)
            if render["returncode"] == 0:
                metric = run_phase(job, "metric")
                job_receipt["phases"].append(metric)
                job_receipt["status"] = (
                    "COMPLETE_RANKED"
                    if metric["returncode"] == 0 and completed_summary(artifact_root)
                    else "METRIC_FAILED_UNRANKED"
                )
            else:
                job_receipt["status"] = "RENDER_FAILED_UNRANKED"
            job_receipt["finished_at"] = now()
            write_json(artifact_root / "rgb_job_receipt.json", job_receipt)
        receipt["jobs"].append(job_receipt)
        write_json(receipt_path, receipt)
    statuses = [row["status"] for row in receipt["jobs"]]
    receipt["status"] = (
        "COMPLETE_ALL_SELECTED"
        if statuses
        and all(status in {"COMPLETE_RANKED", "SKIPPED_ALREADY_COMPLETE"} for status in statuses)
        else "COMPLETE_WITH_UNRANKED_FAILURES"
    )
    receipt["finished_at"] = now()
    receipt["canonical_sha256"] = canonical_sha256(receipt)
    write_json(receipt_path, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
