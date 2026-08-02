#!/usr/bin/env python3
"""Tests for process-tree/cgroup resource probe helpers."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from run_with_resource_probe_v2 import (
    _memory_events,
    frozen_gate_violation,
    gpu_idle_violations,
    sample_process_tree,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[2]


def test_contract() -> None:
    contract = json.loads((ROOT / "configs" / "gs_gcp_resource_probe_contract_v2.json").read_text(encoding="utf-8"))
    validate_contract(contract)
    assert contract["resource_gates"]["fd_peak_max"] == 4096


def test_invasive_contract_rejected() -> None:
    contract = json.loads((ROOT / "configs" / "gs_gcp_resource_probe_contract_v2.json").read_text(encoding="utf-8"))
    contract["non_invasive"]["training_source_changes_allowed"] = True
    try:
        validate_contract(contract)
    except ValueError:
        pass
    else:
        raise AssertionError("invasive contract accepted")


def test_memory_events_parser() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        (root / "memory.events").write_text("low 1\noom 2\n", encoding="utf-8")
        assert _memory_events(root) == {"low": 1, "oom": 2}


def test_process_snapshot_current_process() -> None:
    import os, time
    row = sample_process_tree(os.getpid(), time.monotonic())
    if os.name == "posix":
        assert row["process_count"] >= 1
        assert row["rss_kib"] > 0
        assert row["fd_count"] > 0
    else:
        assert set(row) >= {"process_count", "rss_kib", "fd_count"}


def test_exact_zero_gpu_state_is_idle() -> None:
    rows = [{"utilization_gpu_percent": 0.0, "memory_used_mib": 0.0} for _ in range(3)]
    idle = {"max_utilization_percent": 5.0, "max_memory_used_mib": 1024.0}
    assert gpu_idle_violations(rows, idle) == []


def _contract() -> dict:
    return json.loads((ROOT / "configs" / "gs_gcp_resource_probe_contract_v2.json").read_text(encoding="utf-8"))


def test_host_gate_classification() -> None:
    contract = _contract()
    limit = 110 * 1024**3
    row = {"cgroup_memory_current_bytes": 103 * 1024**3, "fd_count": 10}
    result = frozen_gate_violation(contract, row, [], limit, {}, {})
    assert result["status"] == "HOST_RAM_BLOCKED"
    assert result["failure_reason"] == "host_cgroup_peak_exceeded_frozen_gate"


def test_gpu_gate_classification() -> None:
    contract = _contract()
    row = {"cgroup_memory_current_bytes": 1, "fd_count": 10}
    gpu = [{"gpu_index": 0, "memory_used_mib": 80000, "memory_total_mib": 97887}]
    result = frozen_gate_violation(contract, row, gpu, 110 * 1024**3, {}, {})
    assert result["status"] == "GPU_MEMORY_BLOCKED"


def test_fd_gate_classification() -> None:
    contract = _contract()
    row = {"cgroup_memory_current_bytes": 1, "fd_count": 4097}
    result = frozen_gate_violation(contract, row, [], 110 * 1024**3, {}, {})
    assert result["status"] == "FD_BLOCKED"


def test_cgroup_event_classification() -> None:
    contract = _contract()
    row = {"cgroup_memory_current_bytes": 1, "fd_count": 10}
    result = frozen_gate_violation(contract, row, [], 110 * 1024**3, {"oom": 1}, {"oom": 2})
    assert result["status"] == "HOST_RAM_BLOCKED"
    assert result["failure_reason"] == "cgroup_memory_event_oom"


def test_synthetic_child_accepts_existing_parent() -> None:
    with tempfile.TemporaryDirectory() as temp:
        parent = Path(temp) / "existing"
        parent.mkdir()
        output = parent / "synthetic.bin"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "code" / "gcp" / "stage0_5_probe_synthetic_child.py"),
                "--output",
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        assert output.is_file()


TESTS = [
    test_contract,
    test_invasive_contract_rejected,
    test_memory_events_parser,
    test_process_snapshot_current_process,
    test_exact_zero_gpu_state_is_idle,
    test_host_gate_classification,
    test_gpu_gate_classification,
    test_fd_gate_classification,
    test_cgroup_event_classification,
    test_synthetic_child_accepts_existing_parent,
]


def main() -> int:
    for test in TESTS:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS {len(TESTS)}/{len(TESTS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
