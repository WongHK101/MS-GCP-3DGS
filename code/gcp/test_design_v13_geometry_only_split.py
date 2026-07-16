#!/usr/bin/env python3
"""Focused tests for the residual-blind v1.3 split candidate design."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).with_name("design_v13_geometry_only_split.py")
SPEC = importlib.util.spec_from_file_location("design_v13_geometry_only_split", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class GeometryOnlySplitTest(unittest.TestCase):
    def test_vertical_extent_is_selected_when_feasible(self) -> None:
        frame = pd.DataFrame(
            [
                {"point_name": "A", "x_m": 0.0, "y_m": 0.0, "z_m": 0.0, "control_eligible": True},
                {"point_name": "B", "x_m": 10.0, "y_m": 0.0, "z_m": 0.0, "control_eligible": True},
                {"point_name": "C", "x_m": 10.0, "y_m": 10.0, "z_m": 0.0, "control_eligible": True},
                {"point_name": "D", "x_m": 0.0, "y_m": 10.0, "z_m": 0.0, "control_eligible": True},
                {"point_name": "ROOF", "x_m": 5.0, "y_m": 5.0, "z_m": 10.0, "control_eligible": True},
                {"point_name": "E", "x_m": 2.0, "y_m": 5.0, "z_m": 0.0, "control_eligible": True},
            ]
        )
        controls, metrics = MODULE.choose_controls(frame, 4)
        selected = set(frame.loc[sorted(controls), "point_name"])
        self.assertIn("ROOF", selected)
        self.assertGreaterEqual(metrics.height_range_ratio, MODULE.CONTROL_HEIGHT_RANGE_TARGET)

    def test_selection_is_deterministic(self) -> None:
        frame = pd.DataFrame(
            [
                {"point_name": name, "x_m": x, "y_m": y, "z_m": z, "control_eligible": True}
                for name, x, y, z in [
                    ("A", 0.0, 0.0, 0.0),
                    ("B", 2.0, 0.0, 0.5),
                    ("C", 2.0, 2.0, 1.0),
                    ("D", 0.0, 2.0, 0.2),
                    ("E", 1.0, 1.0, 0.8),
                ]
            ]
        )
        first, _ = MODULE.choose_controls(frame, 3)
        second, _ = MODULE.choose_controls(frame, 3)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
