from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from run_m3m_gcp_100k_success_geometry_plan import (
    cleanup_packet_arrays,
    evaluation_cleanup_allowed,
    run_phase,
)


class GeometrySuccessRunnerTest(unittest.TestCase):
    def test_evaluation_failure_never_allows_packet_cleanup(self) -> None:
        self.assertFalse(evaluation_cleanup_allowed(1, True))
        self.assertFalse(evaluation_cleanup_allowed(0, False))
        self.assertTrue(evaluation_cleanup_allowed(0, True))

    def test_cleanup_removes_only_exact_packet_npz_members(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory) / "run"
            packet_root = (
                run_root / "formal_evaluation/gcp_packets_100k_success_v3"
            )
            packet_root.mkdir(parents=True)
            packet = packet_root / "view.npz"
            packet.write_bytes(b"packet")
            keep = packet_root / "depth_map_index.csv"
            keep.write_text("image_name\n", encoding="utf-8")
            manifest = {
                "depth_index": [
                    {
                        "packet_path": str(packet.resolve()),
                        "packet_sha256": "frozen-packet-sha",
                    }
                ]
            }
            (packet_root / "depth_export_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            receipt = cleanup_packet_arrays(packet_root, run_root, "unit_test")

            self.assertFalse(packet.exists())
            self.assertTrue(keep.is_file())
            self.assertEqual(receipt["removed_file_count"], 1)
            self.assertEqual(receipt["removed_bytes"], 6)
            self.assertEqual(
                receipt["removed"][0]["manifest_sha256"], "frozen-packet-sha"
            )
            self.assertTrue(
                (packet_root / "PACKET_ARRAY_CLEANUP_RECEIPT.json").is_file()
            )

    def test_cleanup_refuses_nonfixed_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory) / "run"
            unsafe = run_root / "formal_evaluation/not_a_packet_root"
            unsafe.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "exact formal roots"):
                cleanup_packet_arrays(unsafe, run_root, "unit_test")

    def test_cleanup_accepts_fresh_lidar_v3_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory) / "run"
            packet_root = (
                run_root / "formal_evaluation/lidar_packets_100k_success_v3"
            )
            packet_root.mkdir(parents=True)
            packet = packet_root / "view.npz"
            packet.write_bytes(b"packet")
            receipt = cleanup_packet_arrays(packet_root, run_root, "unit_test")
            self.assertFalse(packet.exists())
            self.assertEqual(receipt["removed_file_count"], 1)

    def test_cleanup_accepts_heldout_candidate_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory) / "run"
            packet_root = (
                run_root
                / "formal_evaluation/lidar_packets_100k_heldout_candidate_v1"
            )
            packet_root.mkdir(parents=True)
            packet = packet_root / "view.npz"
            packet.write_bytes(b"packet")
            receipt = cleanup_packet_arrays(packet_root, run_root, "unit_test")
            self.assertFalse(packet.exists())
            self.assertEqual(receipt["removed_file_count"], 1)

    @unittest.skipUnless(os.name == "posix", "RLIMIT_NOFILE is POSIX-only")
    def test_run_phase_applies_requested_nofile_soft_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = {
                "argv": [
                    sys.executable,
                    "-c",
                    (
                        "import resource; "
                        "print(resource.getrlimit(resource.RLIMIT_NOFILE)[0])"
                    ),
                ],
                "working_directory": str(root),
                "environment": {},
                "resource_limits": {"nofile_soft": 65535},
                "stdout": str(root / "stdout.log"),
                "stderr": str(root / "stderr.log"),
            }
            result = run_phase(spec, "unit_test")
            self.assertEqual(result["returncode"], 0)
            self.assertEqual(
                (root / "stdout.log").read_text(encoding="utf-8").strip(),
                "65535",
            )
            self.assertEqual(
                result["resource_limits"]["nofile_applied_soft"], 65535
            )


if __name__ == "__main__":
    unittest.main()
