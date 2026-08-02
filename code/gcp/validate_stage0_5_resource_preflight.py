#!/usr/bin/env python3
"""Apply the frozen Stage 0.5 camera-load resource feasibility gates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


GIB = 1024**3
MIB = 1024**2


def _require_number(value: Any, name: str) -> float:
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"missing or non-finite field: {name}")
    return float(value)


def validate_preflight(
    contract: dict[str, Any], resource: dict[str, Any], camera: dict[str, Any]
) -> dict[str, Any]:
    if contract.get("schema") != "gs_gcp_resource_probe_contract_v2":
        raise ValueError("resource contract schema mismatch")
    gates = contract["resource_gates"]
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, evidence: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "evidence": evidence})

    check("resource_probe_complete", resource.get("status") == "PASS" and resource.get("probe_complete") is True, {
        "status": resource.get("status"), "probe_complete": resource.get("probe_complete"),
    })
    check("camera_loader_pass", camera.get("status") == "PASS", camera.get("status"))
    check("camera_schema", camera.get("schema") == "gs_gcp_original_3dgs_camera_load_preflight_v2", camera.get("schema"))
    check("camera_resolution", int(camera.get("resolution", -1)) == 4, camera.get("resolution"))
    check("camera_data_device", camera.get("data_device") in {"cuda", "cpu"}, camera.get("data_device"))
    check(
        "host_allocator_policy",
        camera.get("host_allocator_policy") == "glibc_malloc_trim_threshold_zero_v1",
        camera.get("host_allocator_policy"),
    )
    check(
        "malloc_trim_threshold_env",
        camera.get("malloc_trim_threshold_env") == "0",
        camera.get("malloc_trim_threshold_env"),
    )
    check("point_tracks_not_read", camera.get("points3d_tracks_read") is False, camera.get("points3d_tracks_read"))
    check("camera_materialization_complete", (
        int(camera.get("camera_count", -1)) > 0
        and int(camera.get("camera_records_read_count", -2)) == int(camera.get("camera_count", -1))
        and int(camera.get("camera_tensors_materialized_count", -3)) == int(camera.get("camera_count", -1))
    ), {
        "camera_count": camera.get("camera_count"),
        "records_read": camera.get("camera_records_read_count"),
        "tensors_materialized": camera.get("camera_tensors_materialized_count"),
    })
    check("source_image_backing_closed", int(camera.get("currently_open_source_image_count", -1)) == 0, camera.get("currently_open_source_image_count"))

    cgroup_limit = _require_number(resource.get("cgroup_memory_limit_bytes"), "cgroup_memory_limit_bytes")
    cgroup_peak = _require_number(resource.get("cgroup_observed_peak_bytes"), "cgroup_observed_peak_bytes")
    cgroup_baseline = _require_number(resource.get("cgroup_memory_baseline_bytes"), "cgroup_memory_baseline_bytes")
    host_limit = min(
        float(gates["host_peak_fraction_of_cgroup_limit"]) * cgroup_limit,
        cgroup_limit - float(gates["host_minimum_headroom_gib"]) * GIB,
    )
    check("host_cgroup_peak_gate", cgroup_peak <= host_limit, {
        "observed_peak_bytes": cgroup_peak, "allowed_peak_bytes": host_limit,
    })

    totals = resource.get("gpu_memory_total_mib_per_device", [])
    if len(totals) != 1:
        raise ValueError("camera preflight requires exactly one GPU total-memory record")
    gpu_total_mib = _require_number(totals[0], "gpu_memory_total_mib_per_device[0]")
    gpu_peak_mib = _require_number(resource.get("peak_device_memory_used_mib"), "peak_device_memory_used_mib")
    gpu_limit_mib = min(
        float(gates["gpu_peak_fraction_of_total"]) * gpu_total_mib,
        gpu_total_mib - float(gates["gpu_minimum_headroom_gib"]) * 1024.0,
    )
    check("gpu_peak_gate", gpu_peak_mib <= gpu_limit_mib, {
        "observed_peak_mib": gpu_peak_mib, "allowed_peak_mib": gpu_limit_mib,
    })

    event_delta = resource.get("memory_events_delta", {})
    event_failures = {key: int(event_delta.get(key, 0)) for key in ("oom", "oom_kill", "max")}
    check("cgroup_memory_events", not any(event_failures.values()), event_failures)

    fd_peak = _require_number(resource.get("fd_peak"), "fd_peak")
    fd_min = _require_number(resource.get("fd_last_ten_min"), "fd_last_ten_min")
    fd_max = _require_number(resource.get("fd_last_ten_max"), "fd_last_ten_max")
    fd_before = _require_number(camera.get("fd_before"), "camera.fd_before")
    fd_after = _require_number(camera.get("fd_after"), "camera.fd_after")
    check("fd_peak_gate", fd_peak <= float(gates["fd_peak_max"]), fd_peak)
    check("fd_stable_range_gate", fd_max - fd_min <= float(gates["fd_last_ten_range_max"]), {
        "min": fd_min, "max": fd_max,
    })
    check("fd_camera_stable_gate", (
        fd_after <= fd_before + float(gates["fd_stable_baseline_delta_max"])
        and fd_after <= float(gates["fd_stable_absolute_max"])
    ), {"before": fd_before, "after": fd_after})
    check("jpeg_fd_closed", not camera.get("jpeg_fds_after_stabilization"), camera.get("jpeg_fds_after_stabilization"))

    theoretical = int(camera.get("theoretical_camera_tensor_bytes", -1))
    actual = int(camera.get("actual_camera_tensor_bytes", -2))
    check("camera_tensor_byte_identity", theoretical > 0 and theoretical == actual, {
        "theoretical": theoretical, "actual": actual,
    })
    allocated_delta = max(0, int(camera.get("torch_cuda_allocated_after", 0)) - int(camera.get("torch_cuda_allocated_before", 0)))
    reserved_delta = max(0, int(camera.get("torch_cuda_reserved_after", 0)) - int(camera.get("torch_cuda_reserved_before", 0)))
    gpu_delta_bytes = _require_number(resource.get("peak_gpu_memory_mib"), "peak_gpu_memory_mib") * MIB
    gpu_covers = gpu_delta_bytes + MIB >= actual if camera.get("data_device") == "cuda" else allocated_delta < actual
    check("gpu_observed_covers_camera_tensor", gpu_covers, {
        "gpu_observed_delta_bytes": gpu_delta_bytes,
        "camera_tensor_bytes": actual,
        "torch_allocated_delta_bytes": allocated_delta,
        "torch_reserved_delta_bytes": reserved_delta,
        "one_mib_sampling_slack_bytes": MIB,
        "data_device": camera.get("data_device"),
    })

    process_peak = _require_number(resource.get("process_tree_sampled_peak_rss_kib"), "process_tree_sampled_peak_rss_kib") * 1024.0
    process_delta = max(0.0, process_peak)
    cgroup_delta = max(0.0, cgroup_peak - cgroup_baseline)
    discrepancy = abs(process_delta - cgroup_delta)
    tolerance = max(
        float(gates["process_tree_cgroup_delta_absolute_tolerance_gib"]) * GIB,
        float(gates["process_tree_cgroup_delta_relative_tolerance"]) * max(process_delta, cgroup_delta),
    )
    check("process_tree_cgroup_consistency", discrepancy <= tolerance, {
        "process_tree_peak_bytes": process_peak,
        "process_tree_delta_bytes": process_delta,
        "cgroup_delta_bytes": cgroup_delta,
        "absolute_difference_bytes": discrepancy,
        "allowed_difference_bytes": tolerance,
    })
    check("ray_equivalence", float(camera.get("max_normalized_ray_coordinate_error", math.inf)) <= 1e-12, camera.get("max_normalized_ray_coordinate_error"))
    passed = all(row["passed"] for row in checks)
    return {
        "schema": "gs_gcp_stage0_5_resource_preflight_validation_v1",
        "status": "PASS" if passed else "BLOCKER",
        "feasibility_scope": "camera_load_only_not_training_success_guarantee",
        "checks": checks,
        "passed_count": sum(1 for row in checks if row["passed"]),
        "failed_count": sum(1 for row in checks if not row["passed"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--resource_summary", type=Path, required=True)
    parser.add_argument("--camera_report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate_preflight(
        json.loads(args.contract.read_text(encoding="utf-8")),
        json.loads(args.resource_summary.read_text(encoding="utf-8")),
        json.loads(args.camera_report.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
