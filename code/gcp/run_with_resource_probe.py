#!/usr/bin/env python3
"""Run an unmodified child process with external host/GPU resource probes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


GPU_FIELDS = [
    "timestamp_utc",
    "sample_phase",
    "monotonic_seconds",
    "gpu_index",
    "gpu_uuid",
    "gpu_name",
    "driver_version",
    "memory_used_mib",
    "memory_total_mib",
    "utilization_gpu_percent",
    "power_draw_watts",
]
NVIDIA_QUERY = (
    "index,uuid,name,driver_version,memory.used,memory.total,"
    "utilization.gpu,power.draw"
)
SAFE_ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "CUDA_HOME",
    "LD_LIBRARY_PATH",
    "PATH",
    "PYTHONHASHSEED",
    "PYTHONNOUSERSITE",
    "PYTHONDONTWRITEBYTECODE",
    "TORCH_CUDA_ARCH_LIST",
    "TORCH_EXTENSIONS_DIR",
    "TMPDIR",
)
CGROUP_MEMORY_FILES = (
    "memory.current",
    "memory.peak",
    "memory.events",
    "memory.events.local",
    "memory.max",
    "memory.swap.max",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_gpu_indices(value: str) -> list[int]:
    normalized = value.strip().lower()
    if normalized in {"", "none", "cpu"}:
        return []
    values = [int(item.strip()) for item in value.split(",")]
    if any(item < 0 for item in values):
        raise ValueError("GPU indices must be non-negative")
    if len(values) != len(set(values)):
        raise ValueError("GPU indices must be unique")
    return values


def _optional_float(value: str) -> float | None:
    stripped = value.strip()
    if not stripped or stripped.lower() in {"n/a", "[not supported]", "not supported"}:
        return None
    number = float(stripped)
    return number if math.isfinite(number) else None


def query_nvidia_smi(
    binary: str,
    gpu_indices: list[int],
    monotonic_origin: float,
    sample_phase: str,
) -> list[dict[str, Any]]:
    command = [
        binary,
        f"--query-gpu={NVIDIA_QUERY}",
        "--format=csv,noheader,nounits",
        "-i",
        ",".join(str(index) for index in gpu_indices),
    ]
    output = subprocess.check_output(command, text=True, encoding="utf-8", errors="strict")
    timestamp = utc_now()
    elapsed = time.monotonic() - monotonic_origin
    samples: list[dict[str, Any]] = []
    for row in csv.reader(output.splitlines(), skipinitialspace=True):
        if len(row) != 8:
            raise RuntimeError(f"unexpected nvidia-smi column count: {len(row)}")
        samples.append(
            {
                "timestamp_utc": timestamp,
                "sample_phase": sample_phase,
                "monotonic_seconds": elapsed,
                "gpu_index": int(row[0].strip()),
                "gpu_uuid": row[1].strip(),
                "gpu_name": row[2].strip(),
                "driver_version": row[3].strip(),
                "memory_used_mib": _optional_float(row[4]),
                "memory_total_mib": _optional_float(row[5]),
                "utilization_gpu_percent": _optional_float(row[6]),
                "power_draw_watts": _optional_float(row[7]),
            }
        )
    if {sample["gpu_index"] for sample in samples} != set(gpu_indices):
        raise RuntimeError("nvidia-smi did not return every requested GPU")
    return samples


def query_compute_apps(binary: str, selected_gpu_uuids: set[str]) -> list[dict[str, Any]]:
    command = [
        binary,
        "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(command, text=True, encoding="utf-8", errors="strict")
    rows: list[dict[str, Any]] = []
    for row in csv.reader(output.splitlines(), skipinitialspace=True):
        if not row:
            continue
        if len(row) != 4:
            raise RuntimeError(f"unexpected compute-app column count: {len(row)}")
        if row[0].strip() not in selected_gpu_uuids:
            continue
        rows.append(
            {
                "gpu_uuid": row[0].strip(),
                "pid": int(row[1].strip()),
                "process_name": row[2].strip(),
                "used_gpu_memory_mib": _optional_float(row[3]),
            }
        )
    return rows


def parse_gnu_time_output(text: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.strip().split(":", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def capture_cgroup_memory(root: Path = Path("/sys/fs/cgroup")) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for name in CGROUP_MEMORY_FILES:
        path = root / name
        if path.is_file():
            try:
                files[name] = {"available": True, "value": path.read_text(encoding="utf-8").strip()}
            except OSError as exc:
                files[name] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
        else:
            files[name] = {"available": False}
    return {
        "schema": "gs_gcp_cgroup_memory_snapshot_v1",
        "root": str(root),
        "files": files,
    }


def summarize_gpu_samples(
    samples: Iterable[dict[str, Any]],
    baseline_memory_by_gpu: dict[int, float] | None = None,
) -> dict[str, Any]:
    materialized = [row for row in samples if row.get("sample_phase", "runtime") == "runtime"]
    baseline_memory_by_gpu = baseline_memory_by_gpu or {}
    memory = [row["memory_used_mib"] for row in materialized if row.get("memory_used_mib") is not None]
    utilization = [
        row["utilization_gpu_percent"]
        for row in materialized
        if row.get("utilization_gpu_percent") is not None
    ]
    energy_wh = 0.0
    by_gpu: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in materialized:
        by_gpu[int(row["gpu_index"])].append(row)
    for rows in by_gpu.values():
        rows.sort(key=lambda row: float(row["monotonic_seconds"]))
        for current, following in zip(rows, rows[1:]):
            power = current.get("power_draw_watts")
            if power is None:
                continue
            delta_seconds = max(
                0.0,
                float(following["monotonic_seconds"]) - float(current["monotonic_seconds"]),
            )
            energy_wh += float(power) * delta_seconds / 3600.0
    return {
        "gpu_sample_count": len(materialized),
        "peak_gpu_memory_mib": max(
            (
                max(0.0, float(row["memory_used_mib"]) - baseline_memory_by_gpu[int(row["gpu_index"])])
                for row in materialized
                if row.get("memory_used_mib") is not None and int(row["gpu_index"]) in baseline_memory_by_gpu
            ),
            default=None,
        ),
        "peak_device_memory_used_mib": max(memory) if memory else None,
        "baseline_device_memory_used_mib_by_gpu": {
            str(index): value for index, value in sorted(baseline_memory_by_gpu.items())
        },
        "mean_gpu_utilization_percent": sum(utilization) / len(utilization) if utilization else None,
        "max_gpu_utilization_percent": max(utilization) if utilization else None,
        "estimated_gpu_energy_wh": energy_wh if materialized else None,
    }


def validate_contract(contract: dict[str, Any]) -> None:
    if contract.get("schema") != "gs_gcp_resource_probe_contract_v1":
        raise ValueError("unknown resource probe contract")
    interval = float(contract.get("sampling", {}).get("gpu_interval_seconds", -1))
    if interval != 1.0:
        raise ValueError("formal resource probe interval must be exactly 1.0 second")
    sampling = contract.get("sampling", {})
    if sampling.get("idle_preflight_samples") != 3:
        raise ValueError("formal GPU idle preflight must use exactly three samples")
    if float(sampling.get("idle_preflight_interval_seconds", -1)) != 1.0:
        raise ValueError("formal GPU idle preflight interval must be 1.0 second")
    host_tool = contract.get("host_tool", {})
    if len(str(host_tool.get("binary_sha256", ""))) != 64:
        raise ValueError("resource probe host tool SHA-256 is missing")
    if host_tool.get("tool_manifest_schema") != "gs_gcp_isolated_gnu_time_tool_v2":
        raise ValueError("resource probe tool manifest must use deterministic v2 schema")
    if len(str(host_tool.get("tool_manifest_sha256", ""))) != 64:
        raise ValueError("resource probe tool manifest SHA-256 is missing")
    if (
        host_tool.get("dynamic_dependency_normalization")
        != "strip_whitespace_and_runtime_load_addresses_v1"
    ):
        raise ValueError("resource probe dependency normalization is not frozen")
    if not str(host_tool.get("expected_path", "")).startswith("/"):
        raise ValueError("resource probe host tool path must be absolute")
    non_invasive = contract.get("non_invasive", {})
    if any(non_invasive.get(key) is not False for key in (
        "training_source_changes_allowed",
        "autograd_instrumentation_allowed",
        "loss_instrumentation_allowed",
        "child_command_shell_allowed",
    )):
        raise ValueError("resource probe contract is not non-invasive")


def write_gpu_samples(path: Path, samples: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=GPU_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(samples)


def _sampler_loop(
    stop: threading.Event,
    samples: list[dict[str, Any]],
    errors: list[str],
    binary: str,
    indices: list[int],
    origin: float,
    interval: float,
) -> None:
    next_sample = time.monotonic() + interval
    while not stop.wait(max(0.0, next_sample - time.monotonic())):
        try:
            samples.extend(query_nvidia_smi(binary, indices, origin, "runtime"))
        except Exception as exc:  # sampler failure is recorded and handled by the parent
            errors.append(f"{type(exc).__name__}: {exc}")
            return
        next_sample += interval


def run(args: argparse.Namespace) -> int:
    contract_path = args.contract.resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    validate_contract(contract)
    phase_types = set(contract["phase_types"])
    if args.phase not in phase_types:
        raise ValueError(f"unknown phase {args.phase!r}")

    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("child command is required after --")
    gpu_indices = parse_gpu_indices(args.gpu_indices)
    gpu_required = args.phase in set(contract["gpu_required_phases"])
    if gpu_required and not gpu_indices:
        raise ValueError(f"phase {args.phase} requires explicit GPU indices")

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"resource probe output already exists: {output_dir}")
    working_directory = args.working_directory.resolve()
    if not working_directory.is_dir():
        raise FileNotFoundError(f"working directory not found: {working_directory}")
    time_binary = shutil.which(args.time_binary) or args.time_binary
    if not Path(time_binary).is_file():
        raise FileNotFoundError(f"GNU time binary not found: {time_binary}")
    expected_time_sha = contract.get("host_tool", {}).get("binary_sha256")
    actual_time_sha = sha256_file(Path(time_binary))
    if actual_time_sha != expected_time_sha:
        raise RuntimeError(
            f"GNU time binary SHA mismatch: expected {expected_time_sha}, got {actual_time_sha}"
        )
    nvidia_binary = shutil.which(args.nvidia_smi_binary) or args.nvidia_smi_binary

    output_dir.mkdir(parents=True)
    files = contract["output_files"]
    command_path = output_dir / files["command"]
    gpu_path = output_dir / files["gpu_samples"]
    idle_path = output_dir / files["gpu_idle_preflight"]
    time_path = output_dir / files["gnu_time"]
    stdout_path = output_dir / files["stdout"]
    stderr_path = output_dir / files["stderr"]
    summary_path = output_dir / files["summary"]

    command_record = {
        "schema": "gs_gcp_resource_probe_command_v1",
        "phase": args.phase,
        "argv": command,
        "working_directory": str(working_directory),
        "gpu_indices": gpu_indices,
        "contract_path": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "probe_source_sha256": sha256_file(Path(__file__).resolve()),
        "time_binary_path": str(Path(time_binary).resolve()),
        "time_binary_sha256": actual_time_sha,
        "safe_environment": {key: os.environ[key] for key in SAFE_ENV_KEYS if key in os.environ},
    }
    command_path.write_text(json.dumps(command_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    preflight_origin = time.monotonic()
    preflight_samples: list[dict[str, Any]] = []
    preflight_apps: list[dict[str, Any]] = []
    baseline_memory_by_gpu: dict[int, float] = {}
    if gpu_indices:
        sample_count = int(contract["sampling"]["idle_preflight_samples"])
        for sample_index in range(sample_count):
            rows = query_nvidia_smi(nvidia_binary, gpu_indices, preflight_origin, "idle_preflight")
            preflight_samples.extend(rows)
            if sample_index + 1 < sample_count:
                time.sleep(float(contract["sampling"]["idle_preflight_interval_seconds"]))
        selected_uuids = {str(row["gpu_uuid"]) for row in preflight_samples}
        preflight_apps = query_compute_apps(nvidia_binary, selected_uuids)
        utilization_limit = float(contract["sampling"]["idle_max_utilization_percent"])
        memory_limit = float(contract["sampling"]["idle_max_memory_used_mib"])
        violations: list[str] = []
        for row in preflight_samples:
            if row["utilization_gpu_percent"] is None or row["utilization_gpu_percent"] > utilization_limit:
                violations.append(
                    f"gpu {row['gpu_index']} utilization {row['utilization_gpu_percent']} exceeds {utilization_limit}"
                )
            if row["memory_used_mib"] is None or row["memory_used_mib"] > memory_limit:
                violations.append(
                    f"gpu {row['gpu_index']} memory {row['memory_used_mib']} exceeds {memory_limit} MiB"
                )
        if preflight_apps and not contract["sampling"]["visible_foreign_compute_processes_allowed"]:
            violations.append(f"visible compute processes found before launch: {preflight_apps}")
        for index in gpu_indices:
            values = [
                float(row["memory_used_mib"])
                for row in preflight_samples
                if row["gpu_index"] == index and row["memory_used_mib"] is not None
            ]
            if len(values) != sample_count:
                violations.append(f"incomplete baseline memory samples for GPU {index}")
            else:
                baseline_memory_by_gpu[index] = sum(values) / len(values)
        idle_record = {
            "schema": "gs_gcp_gpu_idle_preflight_v1",
            "passed": not violations,
            "samples": preflight_samples,
            "visible_compute_apps": preflight_apps,
            "baseline_memory_used_mib_by_gpu": {
                str(index): value for index, value in sorted(baseline_memory_by_gpu.items())
            },
            "violations": violations,
        }
        idle_path.write_text(json.dumps(idle_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if violations:
            write_gpu_samples(gpu_path, preflight_samples)
            raise RuntimeError("GPU idle preflight failed: " + "; ".join(violations))
    else:
        idle_path.write_text(
            json.dumps({"schema": "gs_gcp_gpu_idle_preflight_v1", "passed": True, "not_applicable": True}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    origin = time.monotonic()
    started_utc = utc_now()
    cgroup_before = capture_cgroup_memory()
    samples: list[dict[str, Any]] = []
    sampler_errors: list[str] = []
    stop = threading.Event()
    sampler = None

    wrapped_command = [time_binary, "-v", "-o", str(time_path), "--", *command]
    child_exit_code = 127
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr_handle:
            child = subprocess.Popen(
                wrapped_command,
                cwd=working_directory,
                stdout=stdout_handle,
                stderr=stderr_handle,
            )
            if gpu_indices:
                samples.extend(query_nvidia_smi(nvidia_binary, gpu_indices, origin, "runtime"))
                sampler = threading.Thread(
                    target=_sampler_loop,
                    args=(
                        stop,
                        samples,
                        sampler_errors,
                        nvidia_binary,
                        gpu_indices,
                        origin,
                        float(contract["sampling"]["gpu_interval_seconds"]),
                    ),
                    daemon=True,
                )
                sampler.start()
            child_exit_code = child.wait()
    finally:
        stop.set()
        if sampler is not None:
            sampler.join(timeout=5.0)
        if gpu_indices and not sampler_errors:
            try:
                samples.extend(query_nvidia_smi(nvidia_binary, gpu_indices, origin, "runtime"))
            except Exception as exc:
                sampler_errors.append(f"{type(exc).__name__}: {exc}")

    wall_seconds = time.monotonic() - origin
    cgroup_after = capture_cgroup_memory()
    (output_dir / "cgroup_memory.json").write_text(
        json.dumps(
            {
                "schema": "gs_gcp_cgroup_memory_probe_v1",
                "before": cgroup_before,
                "after": cgroup_after,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_gpu_samples(gpu_path, [*preflight_samples, *samples])
    time_data = parse_gnu_time_output(time_path.read_text(encoding="utf-8", errors="replace") if time_path.exists() else "")
    gpu_summary = summarize_gpu_samples(samples, baseline_memory_by_gpu)
    probe_complete = not sampler_errors and (not gpu_required or bool(samples))
    summary = {
        "schema": "gs_gcp_resource_probe_summary_v1",
        "phase": args.phase,
        "started_utc": started_utc,
        "ended_utc": utc_now(),
        "wall_seconds": wall_seconds,
        "allocated_gpu_count": len(gpu_indices),
        "gpu_idle_preflight_passed": True,
        "gpu_idle_preflight_sample_count": len(preflight_samples),
        "gpu_hours": wall_seconds * len(gpu_indices) / 3600.0,
        **gpu_summary,
        "maximum_resident_set_size_kib": int(time_data["Maximum resident set size (kbytes)"])
        if time_data.get("Maximum resident set size (kbytes)", "").isdigit()
        else None,
        "child_exit_code": child_exit_code,
        "probe_complete": probe_complete,
        "sampler_errors": sampler_errors,
        "cgroup_memory_probe": "cgroup_memory.json",
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if child_exit_code != 0:
        return child_exit_code
    return 0 if probe_complete else 70


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--working_directory", required=True, type=Path)
    parser.add_argument("--gpu_indices", required=True)
    parser.add_argument("--time_binary", required=True)
    parser.add_argument("--nvidia_smi_binary", default="nvidia-smi")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def main() -> int:
    try:
        return run(parse_args())
    except Exception as exc:
        print(f"resource probe preflight failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
