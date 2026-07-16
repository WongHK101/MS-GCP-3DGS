#!/usr/bin/env python3
"""Focused tests for corrected RTK observation auditing."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("audit_rtk_corrected_observation_quality.py")
SPEC = importlib.util.spec_from_file_location("audit_rtk_corrected_observation_quality", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class CorrectedRtkAuditTest(unittest.TestCase):
    def test_point_name_strips_only_numeric_observation_suffix(self) -> None:
        self.assertEqual(MODULE.point_name("G39_27"), "G39")
        self.assertEqual(MODULE.point_name("wy3_1_27"), "wy3_1")

    def test_pairwise_range(self) -> None:
        points = np.asarray([[0.0, 0.0], [3.0, 4.0], [1.0, 1.0]])
        self.assertAlmostEqual(MODULE.max_pairwise(points), 5.0)

    def test_date_formats_normalize(self) -> None:
        first = MODULE.parse_datetime("2026-05-05", "11:37:49")
        second = MODULE.parse_datetime("2026/5/5", "11:37:49")
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
