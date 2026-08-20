#!/usr/bin/env python3
"""Numeric, NPZ-schema and exact ranking tests for LiDAR formal v1."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from verify_m3m_gcp_lidar_formal_v1 import (
    SCENE_RANK_KEYS,
    competition_rank_rows,
    recompute_metrics,
    validate_distance_npz,
    validate_point_npz,
)


class FormalVerifierTest(unittest.TestCase):
    def test_metric_threshold_epsilon_and_float64(self) -> None:
        accuracy = np.asarray([0.1 + 5e-10, 0.2], dtype=np.float64)
        completeness = np.asarray([0.1, 0.3], dtype=np.float64)
        metrics = recompute_metrics(accuracy, completeness)
        self.assertEqual(metrics["precision_10cm"], 0.5)
        self.assertEqual(metrics["recall_10cm"], 0.5)
        self.assertEqual(metrics["fscore_10cm"], 0.5)

    def test_competition_rank_tolerance_and_method_id_display(self) -> None:
        rows = [
            {"method_id": "z", "fscore_10cm": 0.9, "chamfer_l1_mean_m": 0.1, "precision_10cm": 0.8},
            {"method_id": "a", "fscore_10cm": 0.9 + 5e-10, "chamfer_l1_mean_m": 0.1, "precision_10cm": 0.8},
            {"method_id": "m", "fscore_10cm": 0.8, "chamfer_l1_mean_m": 0.1, "precision_10cm": 0.8},
        ]
        ranked = competition_rank_rows(rows, SCENE_RANK_KEYS)
        self.assertEqual([(row["method_id"], row["rank"]) for row in ranked], [("a", 1), ("z", 1), ("m", 3)])

    def test_npz_keys_dtype_shape_origin_and_distance_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            origin = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
            reference = root / "reference.npz"
            surface = root / "surface.npz"
            distance = root / "distance.npz"
            np.savez_compressed(reference, points_local_m=np.ones((4, 3), dtype=np.float64), local_origin_utm49n_normal_height_m=origin)
            np.savez_compressed(surface, points_local_m=np.ones((2, 3), dtype=np.float64), local_origin_utm49n_normal_height_m=origin)
            np.savez_compressed(distance, reconstruction_to_lidar_m=np.ones(2, dtype=np.float64), lidar_to_reconstruction_m=np.ones(4, dtype=np.float64))
            ref_points, ref_origin = validate_point_npz(reference)
            surf_points, _ = validate_point_npz(surface, expected_origin=ref_origin)
            accuracy, completeness = validate_distance_npz(distance, reconstruction_count=len(surf_points), reference_count=len(ref_points))
            self.assertEqual((len(accuracy), len(completeness)), (2, 4))

    def test_npz_wrong_dtype_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.npz"
            np.savez_compressed(path, points_local_m=np.ones((2, 3), dtype=np.float32), local_origin_utm49n_normal_height_m=np.zeros(3, dtype=np.float64))
            with self.assertRaises(ValueError):
                validate_point_npz(path)


if __name__ == "__main__":
    unittest.main()
