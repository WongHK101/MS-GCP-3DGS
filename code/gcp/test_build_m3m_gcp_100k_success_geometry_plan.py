#!/usr/bin/env python3
"""Regression tests for promoted 100K geometry runtime bindings."""

from __future__ import annotations

import unittest
from pathlib import Path

from build_m3m_gcp_100k_success_geometry_plan import (
    GEOMETRY_ADAPTER_PYTHONPATHS,
    environment,
    phase,
)


class GeometryRuntimeBindingTests(unittest.TestCase):
    def test_external_adapter_packages_precede_registry_compat_paths(self) -> None:
        for method_id in ("3dgs_original", "pgsr", "rade_gs", "metrogs"):
            with self.subTest(method_id=method_id):
                env = environment(
                    {
                        "method_id": method_id,
                        "pythonpath": ["/registry/compat"],
                    }
                )
                entries = env["PYTHONPATH"].split(":")
                self.assertEqual(
                    entries[: len(GEOMETRY_ADAPTER_PYTHONPATHS[method_id])],
                    [str(path) for path in GEOMETRY_ADAPTER_PYTHONPATHS[method_id]],
                )
                self.assertEqual(entries[-1], "/registry/compat")

    def test_installed_adapter_methods_keep_registry_paths_only(self) -> None:
        env = environment(
            {"method_id": "citygs_x", "pythonpath": ["/registry/compat"]}
        )
        self.assertEqual(env["PYTHONPATH"], "/registry/compat")

    def test_phase_records_requested_nofile_limit(self) -> None:
        spec = phase(
            ["python", "export.py"],
            working_directory=Path("."),
            env={},
            log_root=Path("logs"),
            nofile_soft_limit=65535,
        )
        self.assertEqual(spec["resource_limits"], {"nofile_soft": 65535})


if __name__ == "__main__":
    unittest.main()
