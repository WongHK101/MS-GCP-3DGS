#!/usr/bin/env python3
"""CPU-only tests for the approved 100K held-out LiDAR candidate plan."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from build_m3m_gcp_100k_heldout_lidar_candidate_plan import (
    EXPECTED_HELDOUT_VIEWS,
    heldout_names,
    write_allowlist_exclusive,
)


class HeldoutLidarCandidatePlanTest(unittest.TestCase):
    def test_allowlist_is_exact_and_lf_only(self) -> None:
        names = [f"heldout_{index:04d}.JPG" for index in range(314)]
        split = {
            "scenes": [
                {
                    "scene": "gcp_100000_20260610",
                    "test_image_names": names,
                }
            ]
        }
        self.assertEqual(heldout_names(split), names)
        self.assertEqual(EXPECTED_HELDOUT_VIEWS, 314)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "heldout.csv"
            write_allowlist_exclusive(path, names)
            payload = path.read_bytes()
            self.assertNotIn(b"\r\n", payload)
            self.assertEqual(payload.count(b"\n"), 315)
            with self.assertRaises(FileExistsError):
                write_allowlist_exclusive(path, names)


if __name__ == "__main__":
    unittest.main()
