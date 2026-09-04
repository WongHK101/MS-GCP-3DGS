#!/usr/bin/env python3
"""Resume-safe Core-3 LiDAR reevaluation for the frozen rectangular ROI.

This runner reuses the previously successful model-specific depth exporters.
It changes only packet/output paths and evaluates those packets with the
explicit image-defined rectangular ROI. Training products are read-only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EVAL_PYTHON = Path("/root/autodl-tmp/envs/m3m-gcp-lidar-eval/bin/python")
GCP_CSV = Path(
    "/root/autodl-tmp/datasets/M3M-GCP-native-quarter-preflight-data-v1/benchmark/"
    "source_release_v1_3_0/gcp_points_cgcs2000_cm108_v1_3_0.csv"
)
VERTICAL_SANITY = Path(
    "/root/autodl-tmp/staging/m3m-gcp-native-quarter/"
    "lidar-per-scene-v2-prep-20260828/vertical_datum_sanity.json"
)
LIDAR_ROOT = Path("/root/autodl-tmp/datasets/M3M-GCP-LiDAR-per-scene-v2")
PROTOCOL_ROOT = Path(
    "/root/autodl-tmp/datasets/M3M-GCP-native-quarter-benchmark-protocol-v2/scenes"
)
THREE_K_BATCH = Path(
    "/root/autodl-tmp/runs/m3m-gcp-native-quarter/lidar-rendered-surface/"
    "gcp_3000_20260602/heldout-candidate-v1-20260824T1922Z"
)
THREE_K_PLAN = THREE_K_BATCH / "plan.json"
THREE_K_ENV = Path(
    "/root/autodl-tmp/worktrees/m3m-gcp-native-quarter/benchmark/"
    "1879d42d634ed047164e669724a8509636dfe768/repo/configs/"
    "m3m_gcp_3k_heldout_candidate_environment_overrides_901_v1.json"
)
TWENTY_K_COMMANDS = Path(
    "/root/autodl-tmp/staging/m3m-gcp-native-quarter/"
    "lidar-per-scene-v2-prep-20260828/commands"
)
HUNDRED_K_HELDOUT_PLAN = Path(
    "/root/autodl-tmp/runs/m3m-gcp-native-quarter/qualification-100k-v1/"
    "gcp_100000_20260610/evaluation-runtime-success-v1/"
    "geometry_heldout_candidate_gsprior_domainfix_ee39d6e.json"
)
HUNDRED_K_FULL_PLAN = Path(
    "/root/autodl-tmp/runs/m3m-gcp-native-quarter/qualification-100k-v1/"
    "gcp_100000_20260610/evaluation-runtime-success-v1/geometry_execution_plan_v3.json"
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def canonical_sha256(payload: dict[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("canonical_sha256", None)
    encoded = json.dumps(
        clean,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def flag_value(argv: list[str], flag: str) -> str:
    try:
        return argv[argv.index(flag) + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"required flag missing or incomplete: {flag}") from exc


def replace_flag(argv: list[str], flag: str, value: Path | str) -> list[str]:
    updated = list(argv)
    try:
        index = updated.index(flag)
    except ValueError as exc:
        raise ValueError(f"cannot replace missing flag {flag}") from exc
    updated[index + 1] = str(value)
    return updated


def csv_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    names = [str(row["image_name"]) for row in rows]
    if not names or len(names) != len(set(names)):
        raise ValueError(f"invalid allowlist: {path}")
    return len(names)


def command_digest(argv: list[str], environment: dict[str, str], cwd: Path) -> str:
    payload = {"argv": argv, "environment": environment, "working_directory": str(cwd)}
    return canonical_sha256(payload)


def direct_job(
    *,
    source: Path,
    scene: str,
    track: str,
    method_id: str,
    run_root: Path,
    colmap_model: Path,
    packet_root: Path,
    environment_fallback: dict[str, str] | None = None,
) -> dict[str, Any]:
    command = read_json(source)
    argv = list(command["argv"])
    allowlist = Path(flag_value(argv, "--image_list_csv"))
    argv = replace_flag(argv, "--depth_output_dir", packet_root)
    argv = replace_flag(argv, "--manifest_path", packet_root / "depth_export_manifest.json")
    argv = replace_flag(argv, "--mapping_csv", packet_root / "depth_map_index.csv")
    environment = dict(command.get("environment") or environment_fallback or {})
    cwd = Path(command.get("working_directory") or command.get("cwd"))
    return {
        "scene": scene,
        "track": track,
        "method_id": method_id,
        "run_root": str(run_root),
        "colmap_model": str(colmap_model),
        "allowlist": str(allowlist),
        "expected_views": csv_count(allowlist),
        "packet_root": str(packet_root),
        "packet_manifest": str(packet_root / "depth_export_manifest.json"),
        "export_argv": argv,
        "export_environment": environment,
        "export_working_directory": str(cwd),
        "source_command": identity(source),
        "export_command_sha256": command_digest(argv, environment, cwd),
    }


def wrapper_job(
    *,
    source: dict[str, Any],
    scene: str,
    track: str,
    method_id: str,
    run_root: Path,
    colmap_model: Path,
    packet_root: Path,
) -> dict[str, Any]:
    packet = source["lidar"]["packet"]
    argv = replace_flag(list(packet["argv"]), "--packet-set-root", packet_root)
    allowlist = Path(flag_value(argv, "--train-allowlist"))
    environment = dict(packet.get("environment") or {})
    cwd = Path(packet["working_directory"])
    return {
        "scene": scene,
        "track": track,
        "method_id": method_id,
        "run_root": str(run_root),
        "colmap_model": str(colmap_model),
        "allowlist": str(allowlist),
        "expected_views": csv_count(allowlist),
        "packet_root": str(packet_root),
        "packet_manifest": str(packet_root / "depth_export_manifest.json"),
        "export_argv": argv,
        "export_environment": environment,
        "export_working_directory": str(cwd),
        "source_command": {
            "plan_method_id": method_id,
            "recorded_argv_sha256": packet.get("argv_sha256"),
        },
        "export_command_sha256": command_digest(argv, environment, cwd),
    }


def validate_hundred_k_gsprior_camera_domain(job: dict[str, Any]) -> None:
    """Keep the previously qualified 314-camera normalization binding."""
    expected_root = Path(
        "/root/autodl-tmp/datasets/M3M-GCP-gsprior-normalized-v1/"
        "gcp_100000_20260610/rgb_evaluation"
    )
    argv = job["export_argv"]
    actual_dataset = Path(flag_value(argv, "--dataset-root"))
    actual_prior = Path(flag_value(argv, "--prior-root"))
    allowlist = Path(flag_value(argv, "--train-allowlist"))
    if actual_dataset != expected_root or actual_prior != expected_root:
        raise ValueError(
            "100K GSPrior must reuse the qualified rgb_evaluation camera domain: "
            f"dataset={actual_dataset}, prior={actual_prior}"
        )
    if "gsprior_domainfix_ee39d6e" not in allowlist.name or job["expected_views"] != 314:
        raise ValueError(
            "100K GSPrior must reuse the qualified 314-view domain-fix allowlist: "
            f"allowlist={allowlist}, views={job['expected_views']}"
        )


def rebind_hundred_k_full_geometry_loader(job: dict[str, Any]) -> dict[str, Any]:
    """Use the current, RGB-free 100K wrapper for the 2,196-view export.

    The historical successful plan predates ``--geometry_camera_only``.  Reusing
    that wrapper makes the upstream Scene decode and retain all 2,196 RGB images
    even though the LiDAR packet renderer needs camera calibration only.  The
    current wrapper preserves the frozen cameras and renderer while installing
    the already-qualified RGB-free loader for 3DGS/RaDe-GS LiDAR exports.
    """

    if job["scene"] != "gcp_100000_20260610" or job["track"] != "train2196_sensitivity":
        raise ValueError("geometry-only wrapper rebind is restricted to 100K train2196")
    if job["method_id"] != "3dgs_original" or job["expected_views"] != 2196:
        raise ValueError("geometry-only wrapper rebind requires 3DGS and exactly 2,196 views")

    repository = Path(__file__).resolve().parents[2]
    wrapper = repository / "code/gcp/run_m3m_gcp_100k_packet_export.py"
    if not wrapper.is_file():
        raise FileNotFoundError(wrapper)

    rebound = dict(job)
    argv = list(job["export_argv"])
    wrapper_indices = [
        index
        for index, token in enumerate(argv)
        if Path(token).name == "run_m3m_gcp_100k_packet_export.py"
    ]
    if len(wrapper_indices) != 1:
        raise ValueError(f"expected exactly one 100K wrapper in export argv: {wrapper_indices}")
    if flag_value(argv, "--method-id") != "3dgs_original":
        raise ValueError("100K train2196 wrapper must retain method-id=3dgs_original")
    if flag_value(argv, "--camera-profile") != "lidar":
        raise ValueError("100K train2196 wrapper must retain camera-profile=lidar")

    argv[wrapper_indices[0]] = str(wrapper)
    argv = replace_flag(argv, "--benchmark-repo", repository)
    environment = dict(job["export_environment"])
    evaluation_repository = Path(flag_value(argv, "--evaluation-repo"))
    working_directory = Path(job["export_working_directory"])
    if working_directory != evaluation_repository:
        raise ValueError(
            "100K train2196 export working directory must remain the evaluation adapter: "
            f"cwd={working_directory}, evaluation_repo={evaluation_repository}"
        )
    rasterizer_python = evaluation_repository / "submodules/diff-gaussian-rasterization"
    environment["PYTHONPATH"] = str(rasterizer_python)
    rebound["export_argv"] = argv
    rebound["export_environment"] = environment
    rebound["export_working_directory"] = str(working_directory)
    rebound["source_command"] = {
        **dict(job["source_command"]),
        "recovery_wrapper": identity(wrapper),
        "camera_loader_policy": "geometry_camera_only",
        "rasterizer_python_binding": str(rasterizer_python),
        "reason": "avoid decoding RGB pixels for calibration-only 2,196-view LiDAR export",
    }
    rebound["export_command_sha256"] = command_digest(
        argv, environment, working_directory
    )
    return rebound


def build_jobs(batch_root: Path) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []

    three_plan = read_json(THREE_K_PLAN)
    environments = read_json(THREE_K_ENV)["environments"]
    three_colmap = Path(three_plan["inputs"]["colmap_cameras"]["path"]).parent
    for row in three_plan["methods"]:
        method = row["method_id"]
        jobs.append(
            direct_job(
                source=THREE_K_BATCH / "export_commands" / f"{method}.json",
                scene="gcp_3000_20260602",
                track="heldout_main",
                method_id=method,
                run_root=Path(row["run_root"]),
                colmap_model=three_colmap,
                packet_root=batch_root / "packets/gcp_3000_20260602/heldout_main" / method,
                environment_fallback=environments[method],
            )
        )

    twenty_methods = ["3dgs_original", "2dgs", "pgsr", "rade_gs", "sof", "citygs_x", "metrogs"]
    twenty_colmap = Path(
        "/root/autodl-tmp/datasets/M3M-GCP-20K-evaluation-camera-roots-v1/"
        "gcp_20000_20260602/rgb_heldout38/sparse/0"
    )
    for method in twenty_methods:
        source = TWENTY_K_COMMANDS / f"20k_{method}_heldout38.json"
        command = read_json(source)
        jobs.append(
            direct_job(
                source=source,
                scene="gcp_20000_20260602",
                track="heldout_main",
                method_id=method,
                run_root=Path(command["run_root"]),
                colmap_model=twenty_colmap,
                packet_root=batch_root / "packets/gcp_20000_20260602/heldout_main" / method,
            )
        )

    hundred_heldout = read_json(HUNDRED_K_HELDOUT_PLAN)
    for row in hundred_heldout["jobs"]:
        method = row["method_id"]
        colmap = Path(flag_value(row["lidar"]["evaluate"]["argv"], "--colmap-model"))
        job = wrapper_job(
            source=row,
            scene="gcp_100000_20260610",
            track="heldout_main",
            method_id=method,
            run_root=Path(row["run_root"]),
            colmap_model=colmap,
            packet_root=batch_root / "packets/gcp_100000_20260610/heldout_main" / method,
        )
        if method == "gsprior":
            validate_hundred_k_gsprior_camera_domain(job)
        jobs.append(job)

    # Validation tracks are retained because they established view-subset behavior.
    for row in three_plan["methods"]:
        method = row["method_id"]
        jobs.append(
            direct_job(
                source=Path(row["source_export_command"]["path"]),
                scene="gcp_3000_20260602",
                track="train66_sensitivity",
                method_id=method,
                run_root=Path(row["run_root"]),
                colmap_model=three_colmap,
                packet_root=batch_root / "packets/gcp_3000_20260602/train66_sensitivity" / method,
                environment_fallback=environments[method],
            )
        )

    hundred_full = read_json(HUNDRED_K_FULL_PLAN)
    row = next(item for item in hundred_full["jobs"] if item["method_id"] == "3dgs_original")
    colmap = Path(flag_value(row["lidar"]["evaluate"]["argv"], "--colmap-model"))
    jobs.append(
        rebind_hundred_k_full_geometry_loader(wrapper_job(
            source=row,
            scene="gcp_100000_20260610",
            track="train2196_sensitivity",
            method_id="3dgs_original",
            run_root=Path(row["run_root"]),
            colmap_model=colmap,
            packet_root=batch_root / "packets/gcp_100000_20260610/train2196_sensitivity/3dgs_original",
        ))
    )
    return jobs


def evaluator_argv(args: argparse.Namespace, job: dict[str, Any]) -> list[str]:
    scene = job["scene"]
    return [
        str(EVAL_PYTHON),
        "-B",
        str(args.evaluator),
        "--benchmark-repo",
        str(args.benchmark_repo),
        "--scene",
        scene,
        "--method-id",
        job["method_id"],
        "--lidar-root",
        str(LIDAR_ROOT / scene),
        "--gcp-csv",
        str(GCP_CSV),
        "--sim3-json",
        str(PROTOCOL_ROOT / scene / "common_sim3.json"),
        "--vertical-sanity",
        str(VERTICAL_SANITY),
        "--roi-config",
        str(args.roi_config),
        "--reference-cache-root",
        str(args.batch_root / "reference_cache" / scene),
        "--output-root",
        str(args.batch_root / "results" / scene / job["track"] / job["method_id"]),
        "--packet-manifest",
        job["packet_manifest"],
        "--run-root",
        job["run_root"],
        "--colmap-model",
        job["colmap_model"],
        "--allowlist-csv",
        job["allowlist"],
        "--query-workers",
        "-1",
    ]


def reference_argv(args: argparse.Namespace, scene: str) -> list[str]:
    return [
        str(EVAL_PYTHON),
        "-B",
        str(args.evaluator),
        "--benchmark-repo",
        str(args.benchmark_repo),
        "--scene",
        scene,
        "--lidar-root",
        str(LIDAR_ROOT / scene),
        "--gcp-csv",
        str(GCP_CSV),
        "--sim3-json",
        str(PROTOCOL_ROOT / scene / "common_sim3.json"),
        "--vertical-sanity",
        str(VERTICAL_SANITY),
        "--roi-config",
        str(args.roi_config),
        "--reference-cache-root",
        str(args.batch_root / "reference_cache" / scene),
        "--build-reference-only",
    ]


def available_memory_gib() -> float:
    fields: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        fields[key] = int(value.strip().split()[0])
    return fields["MemAvailable"] / (1024 * 1024)


def gpu_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        fields = subprocess.check_output(command, text=True, timeout=10).strip().split(",")
        return {
            "memory_used_mib": float(fields[0]),
            "utilization_percent": float(fields[1]),
            "temperature_c": float(fields[2]),
            "power_w": float(fields[3]),
        }
    except Exception as exc:  # pragma: no cover - hardware telemetry fallback
        return {"error": repr(exc)}


def packet_count(path: Path) -> int:
    return sum(1 for _ in path.rglob("*.npz")) if path.exists() else 0


def tail(path: Path, max_bytes: int = 4000) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - max_bytes))
        return handle.read().decode("utf-8", errors="replace")[-max_bytes:]


def run_process(
    *,
    argv: list[str],
    environment: dict[str, str],
    cwd: Path,
    log: Path,
    status_path: Path,
    status: dict[str, Any],
    stage: str,
    packet_root: Path | None = None,
) -> dict[str, Any]:
    log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(environment)
    started = time.monotonic()
    peak_gpu_mib = 0.0
    minimum_available_memory_gib = available_memory_gib()
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{now()}] START {stage}\n")
        handle.write(json.dumps(argv, ensure_ascii=False) + "\n")
        handle.flush()
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        while process.poll() is None:
            snapshot = gpu_snapshot()
            if "memory_used_mib" in snapshot:
                peak_gpu_mib = max(peak_gpu_mib, float(snapshot["memory_used_mib"]))
            minimum_available_memory_gib = min(minimum_available_memory_gib, available_memory_gib())
            status.update(
                {
                    "updated_at": now(),
                    "stage": stage,
                    "child_pid": process.pid,
                    "stage_elapsed_seconds": time.monotonic() - started,
                    "gpu": snapshot,
                    "peak_gpu_memory_mib": peak_gpu_mib,
                    "available_memory_gib": available_memory_gib(),
                    "minimum_available_memory_gib": minimum_available_memory_gib,
                    "disk_free_gib": shutil.disk_usage(status["batch_root"]).free / (1024**3),
                    "packet_npz_count": packet_count(packet_root) if packet_root else None,
                    "log_tail": tail(log),
                }
            )
            write_json(status_path, status)
            time.sleep(15)
        returncode = int(process.wait())
        handle.write(f"[{now()}] END {stage} returncode={returncode}\n")
    return {
        "returncode": returncode,
        "wall_seconds": time.monotonic() - started,
        "peak_gpu_memory_mib": peak_gpu_mib,
        "minimum_available_memory_gib": minimum_available_memory_gib,
        "log": identity(log),
    }


def complete_metrics(path: Path) -> bool:
    try:
        payload = read_json(path)
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return payload.get("status") == "COMPLETE_RANKED"


def archive_incomplete(path: Path, root: Path, label: str) -> Path | None:
    if not path.exists() and not path.is_symlink():
        return None
    destination = root / f"{label}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    os.rename(path, destination)
    return destination


def cleanup_packets(packet_root: Path, batch_root: Path) -> dict[str, Any]:
    resolved = packet_root.resolve()
    packet_parent = (batch_root / "packets").resolve()
    if packet_parent not in resolved.parents:
        raise ValueError(f"refusing cleanup outside batch packet root: {resolved}")
    files = list(resolved.rglob("*.npz"))
    total_bytes = sum(path.stat().st_size for path in files)
    for path in files:
        path.unlink()
    return {
        "status": "PACKET_ARRAYS_CLEANED_AFTER_COMPLETE_RANKED",
        "npz_removed": len(files),
        "bytes_removed": total_bytes,
        "packet_root": str(resolved),
    }


def preflight(args: argparse.Namespace, jobs: list[dict[str, Any]]) -> dict[str, Any]:
    required_files = [args.evaluator, args.roi_config, GCP_CSV, VERTICAL_SANITY]
    for scene in {job["scene"] for job in jobs}:
        required_files.append(PROTOCOL_ROOT / scene / "common_sim3.json")
        required_files.append(LIDAR_ROOT / scene / "lidars/terra_laz_1_4/cloud0.laz")
    for path in required_files:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    for job in jobs:
        for key in ("run_root", "colmap_model", "allowlist", "export_working_directory"):
            if not Path(job[key]).exists():
                raise FileNotFoundError(f"{key}: {job[key]}")
        if not Path(job["export_argv"][0]).is_file():
            raise FileNotFoundError(job["export_argv"][0])
        if csv_count(Path(job["allowlist"])) != job["expected_views"]:
            raise ValueError(f"allowlist changed: {job['scene']} {job['method_id']}")
    expected = {
        ("gcp_3000_20260602", "heldout_main"): (10, 12),
        ("gcp_3000_20260602", "train66_sensitivity"): (10, 66),
        ("gcp_20000_20260602", "heldout_main"): (7, 38),
        ("gcp_100000_20260610", "heldout_main"): (6, 314),
        ("gcp_100000_20260610", "train2196_sensitivity"): (1, 2196),
    }
    actual: dict[tuple[str, str], list[int]] = {}
    for job in jobs:
        actual.setdefault((job["scene"], job["track"]), []).append(job["expected_views"])
    for key, (method_count, view_count) in expected.items():
        if len(actual.get(key, [])) != method_count or set(actual[key]) != {view_count}:
            raise ValueError(f"job matrix mismatch for {key}: {actual.get(key)}")
    return {
        "status": "PASS",
        "job_count": len(jobs),
        "rendered_view_count": sum(job["expected_views"] for job in jobs),
        "matrix": {f"{scene}/{track}": values for (scene, track), values in actual.items()},
        "evaluator": identity(args.evaluator),
        "roi_config": identity(args.roi_config),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-repo", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--roi-config", type=Path, required=True)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.benchmark_repo = args.benchmark_repo.expanduser().resolve()
    args.evaluator = args.evaluator.expanduser().resolve()
    args.roi_config = args.roi_config.expanduser().resolve()
    args.batch_root = args.batch_root.expanduser().resolve()
    args.batch_root.mkdir(parents=True, exist_ok=True)
    status_path = args.batch_root / "batch_status.json"
    receipt_path = args.batch_root / "batch_receipt.json"
    jobs = build_jobs(args.batch_root)
    preflight_result = preflight(args, jobs)
    plan = {
        "schema": "uavgs_lidar_rectangular_roi_execution_plan_v1",
        "status": "FROZEN_READY",
        "created_at": now(),
        "batch_root": str(args.batch_root),
        "preflight": preflight_result,
        "jobs": jobs,
    }
    plan["canonical_sha256"] = canonical_sha256(plan)
    plan_path = args.batch_root / "execution_plan.json"
    if plan_path.exists():
        previous = read_json(plan_path)
        if previous.get("canonical_sha256") != plan["canonical_sha256"]:
            # created_at is not scientific identity; compare the frozen jobs and preflight identities.
            if previous.get("jobs") != plan["jobs"] or previous.get("preflight") != plan["preflight"]:
                raise ValueError("existing execution plan differs from current frozen plan")
            plan = previous
        else:
            plan = previous
    else:
        write_json(plan_path, plan)
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    status: dict[str, Any] = {
        "schema": "uavgs_lidar_rectangular_roi_batch_status_v1",
        "status": "RUNNING",
        "started_at": now(),
        "updated_at": now(),
        "batch_root": str(args.batch_root),
        "job_count": len(jobs),
        "completed_jobs": 0,
        "failed_jobs": 0,
        "current_job": None,
    }
    write_json(status_path, status)
    records: list[dict[str, Any]] = []

    self_test_log = args.batch_root / "logs/evaluator_self_test.log"
    self_test = run_process(
        argv=[str(EVAL_PYTHON), "-B", str(args.evaluator), "--benchmark-repo", str(args.benchmark_repo), "--self-test"],
        environment={"PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0", "PYTHONUNBUFFERED": "1"},
        cwd=args.benchmark_repo,
        log=self_test_log,
        status_path=status_path,
        status=status,
        stage="evaluator_self_test",
    )
    if self_test["returncode"] != 0:
        raise RuntimeError("evaluator self-test failed")

    for scene in ("gcp_3000_20260602", "gcp_20000_20260602", "gcp_100000_20260610"):
        manifest = args.batch_root / "reference_cache" / scene / "reference_cache_manifest.json"
        if manifest.is_file():
            continue
        result = run_process(
            argv=reference_argv(args, scene),
            environment={"PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0", "PYTHONUNBUFFERED": "1"},
            cwd=args.benchmark_repo,
            log=args.batch_root / "logs/reference" / f"{scene}.log",
            status_path=status_path,
            status=status,
            stage=f"reference:{scene}",
        )
        if result["returncode"] != 0:
            raise RuntimeError(f"reference build failed: {scene}")

    for index, job in enumerate(jobs, start=1):
        key = f"{job['scene']}/{job['track']}/{job['method_id']}"
        output_root = args.batch_root / "results" / job["scene"] / job["track"] / job["method_id"]
        metrics_path = output_root / "metrics.json"
        packet_root = Path(job["packet_root"])
        status.update({"current_job": key, "job_index": index, "updated_at": now()})
        write_json(status_path, status)
        if complete_metrics(metrics_path):
            cleanup = cleanup_packets(packet_root, args.batch_root) if packet_count(packet_root) else None
            records.append({"job": key, "status": "SKIPPED_ALREADY_COMPLETE", "cleanup": cleanup})
            status["completed_jobs"] += 1
            continue

        if output_root.exists():
            archive_incomplete(output_root, args.batch_root / "incomplete_outputs", key.replace("/", "__"))
        packet_manifest = Path(job["packet_manifest"])
        export_result = None
        if not packet_manifest.is_file():
            if packet_root.exists():
                archive_incomplete(packet_root, args.batch_root / "incomplete_packets", key.replace("/", "__"))
            # Model-specific exporters own creation of the exact packet directory.
            # Several validated large-scene exporters intentionally reject an
            # already-existing output directory, so create only its parent.
            packet_root.parent.mkdir(parents=True, exist_ok=True)
            export_result = run_process(
                argv=job["export_argv"],
                environment=job["export_environment"],
                cwd=Path(job["export_working_directory"]),
                log=args.batch_root / "logs/export" / job["scene"] / job["track"] / f"{job['method_id']}.log",
                status_path=status_path,
                status=status,
                stage=f"export:{key}",
                packet_root=packet_root,
            )
        current_packets = packet_count(packet_root)
        if (export_result and export_result["returncode"] != 0) or not packet_manifest.is_file() or current_packets != job["expected_views"]:
            record = {
                "job": key,
                "status": "FAILED_EXPORT_PACKET_RETAINED",
                "expected_views": job["expected_views"],
                "packet_npz_count": current_packets,
                "export": export_result,
            }
            records.append(record)
            status["failed_jobs"] += 1
            write_json(args.batch_root / "job_receipts" / f"{key.replace('/', '__')}.json", record)
            continue

        eval_result = run_process(
            argv=evaluator_argv(args, job),
            environment={"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0", "PYTHONUNBUFFERED": "1"},
            cwd=args.benchmark_repo,
            log=args.batch_root / "logs/evaluate" / job["scene"] / job["track"] / f"{job['method_id']}.log",
            status_path=status_path,
            status=status,
            stage=f"evaluate:{key}",
            packet_root=packet_root,
        )
        if eval_result["returncode"] == 0 and complete_metrics(metrics_path):
            cleanup = cleanup_packets(packet_root, args.batch_root)
            record = {
                "job": key,
                "status": "COMPLETE_RANKED_PACKETS_CLEANED",
                "expected_views": job["expected_views"],
                "export": export_result,
                "evaluate": eval_result,
                "metrics": identity(metrics_path),
                "cleanup": cleanup,
            }
            status["completed_jobs"] += 1
        else:
            record = {
                "job": key,
                "status": "FAILED_EVALUATION_PACKET_RETAINED",
                "expected_views": job["expected_views"],
                "packet_npz_count": packet_count(packet_root),
                "export": export_result,
                "evaluate": eval_result,
            }
            status["failed_jobs"] += 1
        records.append(record)
        write_json(args.batch_root / "job_receipts" / f"{key.replace('/', '__')}.json", record)

    success = status["completed_jobs"] == len(jobs) and status["failed_jobs"] == 0
    status.update(
        {
            "status": "COMPLETE" if success else "COMPLETE_WITH_FAILURES",
            "finished_at": now(),
            "updated_at": now(),
            "current_job": None,
            "child_pid": None,
        }
    )
    write_json(status_path, status)
    receipt = {
        "schema": "uavgs_lidar_rectangular_roi_batch_receipt_v1",
        "status": "COMPLETE_ALL_RANKED" if success else "COMPLETE_WITH_FAILURES",
        "created_at": now(),
        "execution_plan": identity(plan_path),
        "batch_status": identity(status_path),
        "records": records,
    }
    receipt["canonical_sha256"] = canonical_sha256(receipt)
    write_json(receipt_path, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
