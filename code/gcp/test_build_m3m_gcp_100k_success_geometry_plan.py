#!/usr/bin/env python3
"""Regression tests for promoted 100K geometry runtime bindings."""

from __future__ import annotations

import unittest
from pathlib import Path

from build_m3m_gcp_100k_success_geometry_plan import (
    GEOMETRY_ADAPTER_PYTHONPATHS,
    GSPRIOR_ROOT,
    LIDAR_PAYLOAD_SHA256_INVENTORY,
    environment,
    packet_command,
    phase,
)
from m3m_gcp_100k_geometry_paths import (
    LIDAR_FULL_TRAIN_PACKET_ROOT_NAME,
    LIDAR_HELDOUT_CANDIDATE_PACKET_ROOT_NAME,
    formal_input_manifest_canonical_sha256,
    lidar_full_train_packet_manifest,
    lidar_full_train_packet_root,
    lidar_heldout_candidate_packet_manifest,
    lidar_heldout_candidate_packet_root,
)


class GeometryRuntimeBindingTests(unittest.TestCase):
    def test_formal_manifest_uses_declared_manifest_sha256_field(self) -> None:
        digest = "a" * 64
        self.assertEqual(
            formal_input_manifest_canonical_sha256(
                {"manifest_sha256": digest, "canonical_sha256": "b" * 64}
            ),
            digest,
        )
        with self.assertRaisesRegex(ValueError, "manifest_sha256"):
            formal_input_manifest_canonical_sha256({})

    def test_lidar_inventory_is_the_frozen_sha256_evidence_file(self) -> None:
        self.assertEqual(
            LIDAR_PAYLOAD_SHA256_INVENTORY.as_posix(),
            "/root/autodl-tmp/datasets/M3M-GCP-LiDAR-reference-v1/"
            "evaluation/evidence/source_payload_sha256_901.csv",
        )

    def test_full_train_lidar_packet_binding_is_shared_v3_namespace(self) -> None:
        run_root = Path("/tmp/promoted-run")
        packet_root = lidar_full_train_packet_root(run_root)
        self.assertEqual(packet_root.name, LIDAR_FULL_TRAIN_PACKET_ROOT_NAME)
        self.assertEqual(
            lidar_full_train_packet_manifest(run_root),
            packet_root / "depth_export_manifest.json",
        )

    def test_heldout_candidate_uses_a_separate_fixed_namespace(self) -> None:
        run_root = Path("/tmp/promoted-run")
        packet_root = lidar_heldout_candidate_packet_root(run_root)
        self.assertEqual(
            packet_root.name, LIDAR_HELDOUT_CANDIDATE_PACKET_ROOT_NAME
        )
        self.assertNotEqual(packet_root, lidar_full_train_packet_root(run_root))
        self.assertEqual(
            lidar_heldout_candidate_packet_manifest(run_root),
            packet_root / "depth_export_manifest.json",
        )

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

    def test_gsprior_heldout_uses_normalized_rgb_evaluation_cameras(self) -> None:
        command = packet_command(
            repo=Path("/benchmark"),
            method={"method_id": "gsprior", "run_root": "/run"},
            profile="lidar_heldout_candidate",
            camera_root=Path("/camera-root"),
            allowlist=Path("/heldout.csv"),
            packet_root=Path("/packets"),
            evaluation_root=Path("/adapter"),
            packet_python=Path("/env/python"),
        )
        dataset_index = command.index("--dataset-root") + 1
        prior_index = command.index("--prior-root") + 1
        expected = str(GSPRIOR_ROOT / "rgb_evaluation")
        self.assertEqual(command[dataset_index], expected)
        self.assertEqual(command[prior_index], expected)


if __name__ == "__main__":
    unittest.main()
