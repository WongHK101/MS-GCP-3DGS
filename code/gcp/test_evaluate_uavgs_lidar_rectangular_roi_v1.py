#!/usr/bin/env python3
"""Focused tests for the frozen image-defined rectangular LiDAR ROI."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CONFIG = REPO / "configs/uavgs_image_defined_rectangular_roi_v1.json"
MODULE_PATH = HERE / "evaluate_uavgs_lidar_rectangular_roi_v1.py"
SPEC = importlib.util.spec_from_file_location("uavgs_rectangular_roi", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class RectangularRoiTests(unittest.TestCase):
    def test_frozen_configuration_hash_and_all_scene_rectangles(self) -> None:
        payload = MODULE.read_json(CONFIG)
        self.assertEqual(payload["canonical_sha256"], MODULE.canonical_sha256(payload))
        self.assertEqual(len(payload["scenes"]), 6)
        for scene, definition in payload["scenes"].items():
            roi, loaded = MODULE.load_rectangular_roi(CONFIG, scene)
            self.assertEqual(list(roi.bounds), definition["rectangle_bounds_utm49n_m"])
            self.assertAlmostEqual(roi.area, definition["area_m2"], places=6)
            self.assertEqual(loaded["selection_basis"], "nadir_image_overlap")
            self.assertTrue(loaded["lidar_support_verified"])

    def test_core_three_expected_areas(self) -> None:
        expected = {
            "gcp_3000_20260602": 3708.8616486945734,
            "gcp_20000_20260602": 21736.243851444448,
            "gcp_100000_20260610": 126843.8667487227,
        }
        for scene, area in expected.items():
            roi, _ = MODULE.load_rectangular_roi(CONFIG, scene)
            self.assertAlmostEqual(roi.area, area, places=6)

    def test_non_rectangular_or_open_ring_is_rejected(self) -> None:
        payload = MODULE.read_json(CONFIG)
        payload.pop("canonical_sha256")
        payload["scenes"]["gcp_3000_20260602"]["rectangle_ring_utm49n_m"][-1][0] += 0.1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact closed axis-aligned rectangle"):
                MODULE.load_rectangular_roi(path, "gcp_3000_20260602")


if __name__ == "__main__":
    unittest.main()
