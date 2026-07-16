#!/usr/bin/env python3
"""Focused tests for the residual-blind v1.3 split candidate design."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).with_name("design_v13_geometry_only_split.py")
SPEC = importlib.util.spec_from_file_location("design_v13_geometry_only_split", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class GeometryOnlySplitTest(unittest.TestCase):
    def test_user_accepted_full_rtk_points_have_normal_role_eligibility(self) -> None:
        self.assertTrue(MODULE.coordinate_is_formal_usable(MODULE.USER_ACCEPTED_FULL_RTK_STATUS))
        self.assertEqual(MODULE.SCENE_RULES["gcp_5000_20260602"]["forced_checkpoints"], {})
        self.assertEqual(MODULE.SCENE_RULES["gcp_50000_20260610"]["forced_checkpoints"], {})

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

    def test_image_exclusion_is_applied_before_good_view_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            annotations = root / "annotations.csv"
            exclusions = root / "exclusions.csv"
            pd.DataFrame(
                [
                    {"point_name": "G39", "image_name": "keep.JPG", "quality": "Good"},
                    {"point_name": "G39", "image_name": "drop.JPG", "quality": "Good"},
                ]
            ).to_csv(annotations, index=False)
            pd.DataFrame(
                [
                    {
                        "scene": "scene",
                        "image_name": "drop.JPG",
                        "formal_v1_3_include": "false",
                    }
                ]
            ).to_csv(exclusions, index=False)
            excluded = MODULE.load_image_exclusions(exclusions)
            summary = MODULE.annotation_summary(annotations, "scene", excluded)
            self.assertEqual(excluded, {("scene", "drop.JPG")})
            self.assertEqual(int(summary.iloc[0]["good_view_count"]), 1)

    def test_final_no_image_exclusion_report_is_not_stale(self) -> None:
        policy = MODULE.image_exclusion_policy_payload(set())
        lines = MODULE.image_exclusion_report_lines(set())
        self.assertEqual(policy["excluded_scene_image_count"], 0)
        self.assertEqual(policy["excluded_scene_images"], [])
        self.assertIn("Good-only", policy["selection_basis"])
        self.assertIn("No image-level exclusion", lines[0])
        self.assertNotIn("0002", lines[0])


if __name__ == "__main__":
    unittest.main()
