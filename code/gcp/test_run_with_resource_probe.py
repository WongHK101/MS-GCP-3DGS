#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from run_with_resource_probe import (
    capture_cgroup_memory,
    parse_gnu_time_output,
    parse_gpu_indices,
    summarize_gpu_samples,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[2]


class ResourceProbeTests(unittest.TestCase):
    def test_contract_is_non_invasive_and_one_hz(self) -> None:
        contract = json.loads((ROOT / "configs" / "gs_gcp_resource_probe_contract_v1.json").read_text(encoding="utf-8"))
        validate_contract(contract)

    def test_gpu_index_parser(self) -> None:
        self.assertEqual(parse_gpu_indices("0,2"), [0, 2])
        self.assertEqual(parse_gpu_indices("none"), [])
        with self.assertRaises(ValueError):
            parse_gpu_indices("1,1")

    def test_gnu_time_parser(self) -> None:
        parsed = parse_gnu_time_output(
            "\tUser time (seconds): 1.25\n\tMaximum resident set size (kbytes): 4096\n"
        )
        self.assertEqual(parsed["User time (seconds)"], "1.25")
        self.assertEqual(parsed["Maximum resident set size (kbytes)"], "4096")

    def test_gpu_summary_and_energy(self) -> None:
        rows = [
            {
                "gpu_index": 0,
                "sample_phase": "runtime",
                "monotonic_seconds": 0.0,
                "memory_used_mib": 100.0,
                "utilization_gpu_percent": 10.0,
                "power_draw_watts": 200.0,
            },
            {
                "gpu_index": 0,
                "sample_phase": "runtime",
                "monotonic_seconds": 2.0,
                "memory_used_mib": 300.0,
                "utilization_gpu_percent": 50.0,
                "power_draw_watts": 220.0,
            },
        ]
        summary = summarize_gpu_samples(rows, {0: 50.0})
        self.assertEqual(summary["peak_gpu_memory_mib"], 250.0)
        self.assertEqual(summary["peak_device_memory_used_mib"], 300.0)
        self.assertEqual(summary["mean_gpu_utilization_percent"], 30.0)
        self.assertTrue(math.isclose(summary["estimated_gpu_energy_wh"], 400.0 / 3600.0))

    def test_invasive_contract_rejected(self) -> None:
        contract = json.loads((ROOT / "configs" / "gs_gcp_resource_probe_contract_v1.json").read_text(encoding="utf-8"))
        contract["non_invasive"]["loss_instrumentation_allowed"] = True
        with self.assertRaises(ValueError):
            validate_contract(contract)

    def test_short_idle_preflight_rejected(self) -> None:
        contract = json.loads((ROOT / "configs" / "gs_gcp_resource_probe_contract_v1.json").read_text(encoding="utf-8"))
        contract["sampling"]["idle_preflight_samples"] = 1
        with self.assertRaises(ValueError):
            validate_contract(contract)

    def test_nondeterministic_tool_manifest_rejected(self) -> None:
        contract = json.loads((ROOT / "configs" / "gs_gcp_resource_probe_contract_v1.json").read_text(encoding="utf-8"))
        contract["host_tool"]["tool_manifest_schema"] = "gs_gcp_isolated_gnu_time_tool_v1"
        with self.assertRaises(ValueError):
            validate_contract(contract)

    def test_cgroup_memory_snapshot_records_available_and_missing_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "memory.current").write_text("1234\n", encoding="utf-8")
            (root / "memory.events").write_text("oom 0\noom_kill 0\n", encoding="utf-8")
            snapshot = capture_cgroup_memory(root)
        self.assertEqual(snapshot["files"]["memory.current"]["value"], "1234")
        self.assertEqual(snapshot["files"]["memory.events"]["value"], "oom 0\noom_kill 0")
        self.assertFalse(snapshot["files"]["memory.peak"]["available"])


if __name__ == "__main__":
    unittest.main()
