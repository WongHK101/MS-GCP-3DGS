#!/usr/bin/env python3
"""Run an unchanged argv child with external GPU/process-tree/cgroup sampling."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

try:
    import resource
except ImportError:  # Windows supports helper tests but not formal execution.
    resource = None

from run_with_resource_probe import (
    parse_gnu_time_output,
    query_compute_apps,
    query_nvidia_smi,
    summarize_gpu_samples,
    utc_now,
    write_gpu_samples,
)


GIB = 1024**3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def _read_int(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return None if text == "max" else int(text)


def _memory_events(root: Path = Path("/sys/fs/cgroup")) -> dict[str, int]:
    path = root / "memory.events"
    if not path.is_file():
        return {}
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, value = line.split()
        result[key] = int(value)
    return result


def _proc_children(pid: int) -> list[int]:
    path = Path(f"/proc/{pid}/task/{pid}/children")
    try:
        return [int(value) for value in path.read_text(encoding="ascii").split()]
    except (OSError, ValueError):
        return []


def _process_tree(root_pid: int) -> list[int]:
    pending = [root_pid]
    seen = set()
    while pending:
        pid = pending.pop()
        if pid in seen or not Path(f"/proc/{pid}").exists():
            continue
        seen.add(pid)
        pending.extend(_proc_children(pid))
    return sorted(seen)


def _status_value_kib(pid: int, key: str) -> int:
    try:
        lines = Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines()
    except OSError:
        return 0
    prefix = key + ":"
    for line in lines:
        if line.startswith(prefix):
            parts = line.split()
            return int(parts[1]) if len(parts) > 1 else 0
    return 0


def _fd_snapshot(pids: list[int]) -> tuple[int, int]:
    total = 0
    jpeg = 0
    for pid in pids:
        root = Path(f"/proc/{pid}/fd")
        try:
            entries = list(root.iterdir())
        except OSError:
            continue
        total += len(entries)
        for item in entries:
            try:
                target = os.readlink(item)
            except OSError:
                continue
            if target.lower().endswith((".jpg", ".jpeg")):
                jpeg += 1
    return total, jpeg


def sample_process_tree(root_pid: int, origin: float) -> dict[str, Any]:
    pids = _process_tree(root_pid)
    fd_count, jpeg_fd_count = _fd_snapshot(pids)
    return {
        "monotonic_seconds": time.monotonic() - origin,
        "process_count": len(pids),
        "rss_kib": sum(_status_value_kib(pid, "VmRSS") for pid in pids),
        "hwm_kib_sum": sum(_status_value_kib(pid, "VmHWM") for pid in pids),
        "fd_count": fd_count,
        "jpeg_fd_count": jpeg_fd_count,
        "cgroup_memory_current_bytes": _read_int(Path("/sys/fs/cgroup/memory.current")),
    }


def _write_process_samples(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "monotonic_seconds", "process_count", "rss_kib", "hwm_kib_sum",
        "fd_count", "jpeg_fd_count", "cgroup_memory_current_bytes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _set_nofile(limit: int) -> None:
    if resource is None:
        raise RuntimeError("formal resource probe v2 requires a POSIX resource module")
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if limit > hard:
        raise RuntimeError(f"requested RLIMIT_NOFILE {limit} exceeds hard limit {hard}")
    resource.setrlimit(resource.RLIMIT_NOFILE, (limit, hard))


def validate_contract(contract: dict[str, Any]) -> None:
    if contract.get("schema") != "gs_gcp_resource_probe_contract_v2":
        raise ValueError("unknown resource contract")
    if float(contract.get("sampling_interval_seconds", 0)) != 1.0:
        raise ValueError("sampling interval must be exactly 1 Hz")
    if int(contract.get("rlimit_nofile_soft", 0)) != 65536:
        raise ValueError("RLIMIT_NOFILE soft limit must be 65536")
    if any(contract.get("non_invasive", {}).values()):
        raise ValueError("resource probe contract is invasive")


def gpu_idle_violations(rows: list[dict[str, Any]], idle: dict[str, Any]) -> list[str]:
    violations = []
    for row in rows:
        utilization = row.get("utilization_gpu_percent")
        memory = row.get("memory_used_mib")
        if utilization is None or float(utilization) > float(idle["max_utilization_percent"]):
            violations.append("GPU utilization is not idle")
        if memory is None or float(memory) > float(idle["max_memory_used_mib"]):
            violations.append("GPU memory is not idle")
    return violations


def run(args: argparse.Namespace) -> int:
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    validate_contract(contract)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("child argv is empty")
    time_binary = Path(args.time_binary).resolve()
    if sha256_file(time_binary) != contract["host_tool"]["binary_sha256"]:
        raise ValueError("GNU time binary SHA mismatch")
    gpu_indices = [int(value) for value in args.gpu_indices.split(",") if value.strip()]
    origin = time.monotonic()
    idle_rows = []
    idle = contract["gpu_idle_preflight"]
    for index in range(int(idle["samples"])):
        idle_rows.extend(query_nvidia_smi(args.nvidia_smi_binary, gpu_indices, origin, "idle_preflight"))
        if index + 1 < int(idle["samples"]):
            time.sleep(float(idle["interval_seconds"]))
    uuids = {str(row["gpu_uuid"]) for row in idle_rows}
    apps = query_compute_apps(args.nvidia_smi_binary, uuids)
    violations = gpu_idle_violations(idle_rows, idle)
    if apps and not bool(idle["foreign_compute_processes_allowed"]):
        violations.append("foreign GPU compute process detected")
    baseline_gpu = {}
    for gpu in gpu_indices:
        values = [float(row["memory_used_mib"]) for row in idle_rows if int(row["gpu_index"]) == gpu]
        baseline_gpu[gpu] = math.fsum(values) / len(values)
    write_json(output / "gpu_idle_preflight.json", {"passed": not violations, "samples": idle_rows, "apps": apps, "violations": violations})
    if violations:
        raise RuntimeError("; ".join(violations))
    events_before = _memory_events()
    cgroup_limit = _read_int(Path("/sys/fs/cgroup/memory.max"))
    cgroup_baseline = _read_int(Path("/sys/fs/cgroup/memory.current"))
    command_record = {
        "schema": "gs_gcp_resource_probe_command_v2",
        "argv": command,
        "working_directory": str(args.working_directory.resolve()),
        "contract_sha256": sha256_file(args.contract),
        "probe_source_sha256": sha256_file(Path(__file__).resolve()),
        "gpu_indices": gpu_indices,
        "rlimit_nofile_soft": contract["rlimit_nofile_soft"],
        "started_utc": utc_now(),
    }
    write_json(output / "command.json", command_record)
    wrapped = [str(time_binary), "-v", "-o", str(output / "gnu_time.txt"), "--", *command]
    gpu_rows: list[dict[str, Any]] = []
    process_rows: list[dict[str, Any]] = []
    sampler_errors: list[str] = []
    stop = threading.Event()
    with (output / "stdout.log").open("w", encoding="utf-8") as stdout, (output / "stderr.log").open("w", encoding="utf-8") as stderr:
        child = subprocess.Popen(
            wrapped,
            cwd=args.working_directory.resolve(),
            stdout=stdout,
            stderr=stderr,
            preexec_fn=lambda: _set_nofile(int(contract["rlimit_nofile_soft"])),
        )

        def sampler() -> None:
            next_sample = time.monotonic()
            while not stop.is_set():
                try:
                    process_rows.append(sample_process_tree(child.pid, origin))
                    gpu_rows.extend(query_nvidia_smi(args.nvidia_smi_binary, gpu_indices, origin, "runtime"))
                except Exception as exc:  # noqa: BLE001
                    sampler_errors.append(f"{type(exc).__name__}: {exc}")
                    return
                next_sample += float(contract["sampling_interval_seconds"])
                stop.wait(max(0.0, next_sample - time.monotonic()))

        thread = threading.Thread(target=sampler, daemon=True)
        thread.start()
        exit_code = child.wait()
        stop.set()
        thread.join(timeout=5.0)
    wall_seconds = time.monotonic() - origin
    write_gpu_samples(output / "gpu_samples.csv", [*idle_rows, *gpu_rows])
    _write_process_samples(output / "process_tree_samples.csv", process_rows)
    time_data = parse_gnu_time_output((output / "gnu_time.txt").read_text(encoding="utf-8", errors="replace"))
    gpu_summary = summarize_gpu_samples(gpu_rows, baseline_gpu)
    gpu_totals = sorted({
        float(row["memory_total_mib"])
        for row in gpu_rows
        if row.get("memory_total_mib") is not None
    })
    if len(gpu_totals) != len(gpu_indices):
        sampler_errors.append("GPU total-memory identity is missing or non-unique")
    events_after = _memory_events()
    event_delta = {key: events_after.get(key, 0) - events_before.get(key, 0) for key in sorted(set(events_before) | set(events_after))}
    fd_values = [int(row["fd_count"]) for row in process_rows]
    last_ten = fd_values[-10:]
    summary = {
        "schema": "gs_gcp_resource_probe_summary_v2",
        "status": "PASS" if exit_code == 0 and not sampler_errors else "METHOD_FAILURE",
        "child_exit_code": exit_code,
        "wall_seconds": wall_seconds,
        "probe_complete": not sampler_errors and bool(process_rows) and bool(gpu_rows),
        "sampler_errors": sampler_errors,
        "allocated_gpu_count": len(gpu_indices),
        "gpu_hours": wall_seconds * len(gpu_indices) / 3600.0,
        "gpu_memory_total_mib_per_device": gpu_totals,
        **gpu_summary,
        "process_maximum_rss_kib": int(time_data.get("Maximum resident set size (kbytes)", "0") or 0),
        "process_tree_sampled_peak_rss_kib": max((int(row["rss_kib"]) for row in process_rows), default=None),
        "cgroup_memory_baseline_bytes": cgroup_baseline,
        "cgroup_memory_limit_bytes": cgroup_limit,
        "cgroup_observed_peak_bytes": max((int(row["cgroup_memory_current_bytes"]) for row in process_rows if row["cgroup_memory_current_bytes"] is not None), default=None),
        "memory_events_before": events_before,
        "memory_events_after": events_after,
        "memory_events_delta": event_delta,
        "fd_peak": max(fd_values, default=None),
        "fd_last_ten_min": min(last_ten, default=None),
        "fd_last_ten_max": max(last_ten, default=None),
        "jpeg_fd_peak": max((int(row["jpeg_fd_count"]) for row in process_rows), default=None),
        "rlimit_nofile_soft": contract["rlimit_nofile_soft"],
        "ended_utc": utc_now(),
    }
    write_json(output / "resource_summary.json", summary)
    return exit_code if exit_code != 0 else (0 if summary["probe_complete"] else 70)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--working_directory", type=Path, required=True)
    parser.add_argument("--gpu_indices", required=True)
    parser.add_argument("--time_binary", required=True)
    parser.add_argument("--nvidia_smi_binary", default="nvidia-smi")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    try:
        return run(parser.parse_args())
    except Exception as exc:  # noqa: BLE001
        print(f"resource probe v2 failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
