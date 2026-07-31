#!/usr/bin/env python3

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_gcp_point_quality_table import (  # noqa: E402
    compact_quality_fields,
    missing_base_join_fields,
    quality_join_fields,
    summarize_quality,
)


class GcpPointQualityTableTests(unittest.TestCase):
    def test_quality_summary_and_join(self) -> None:
        group = []
        for index in range(1, 28):
            group.append(
                {
                    "observation_name": f"G01_{index}",
                    "observation_index": index,
                    "east_m": 100.0 + index * 0.001,
                    "north_m": 200.0 - index * 0.001,
                    "height_m": 10.0 + index * 0.002,
                    "timestamp": np.datetime64("2026-01-01T00:00:00") + np.timedelta64(index, "s"),
                    "vrms_m": 0.004,
                    "hrms_m": 0.003,
                    "pdop": 1.0,
                    "solution_status": "固定解",
                    "satellite_count": 35,
                }
            )
        final = np.asarray([100.014, 199.986, 10.028], dtype=np.float64)
        quality = summarize_quality("G01", group, final)
        self.assertEqual(quality["observation_count"], 27)
        self.assertEqual(quality["fixed_solution_count"], 27)
        self.assertEqual(quality["nonfixed_solution_count"], 0)
        self.assertEqual(quality["fixed_solution_rate"], 1.0)
        self.assertEqual(quality["hrms_median_m"], 0.003)
        self.assertEqual(quality["vrms_median_m"], 0.004)
        record = compact_quality_fields(quality, epoch_sha256="a" * 64, coordinate_sha256="b" * 64)
        joined = quality_join_fields(record)
        self.assertEqual(joined["rtk_quality_source_status"], "direct_corrected_epoch_quality_available")
        self.assertEqual(joined["rtk_absolute_accuracy_status"], "not_independently_verified")
        self.assertEqual(len(joined["rtk_quality_record_sha256_v2"]), 64)

    def test_missing_base_is_not_filled_with_proxy_precision(self) -> None:
        joined = missing_base_join_fields()
        self.assertEqual(joined["rtk_quality_source_status"], "known_base_point_no_epoch_quality_record")
        self.assertEqual(joined["rtk_receiver_hrms_median_m"], "")
        self.assertEqual(
            joined["rtk_absolute_accuracy_status"],
            "not_available_in_frozen_authoritative_artifacts",
        )


if __name__ == "__main__":
    unittest.main()
