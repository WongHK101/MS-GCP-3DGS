#!/usr/bin/env python3
"""Reference-cache compatibility tests for the 100K success evaluator."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


if "laspy" not in sys.modules and importlib.util.find_spec("laspy") is None:
    sys.modules["laspy"] = types.ModuleType("laspy")
if "pyproj" not in sys.modules and importlib.util.find_spec("pyproj") is None:
    pyproj = types.ModuleType("pyproj")
    pyproj.Transformer = object
    sys.modules["pyproj"] = pyproj
if "shapely" not in sys.modules and importlib.util.find_spec("shapely") is None:
    shapely = types.ModuleType("shapely")
    shapely.contains_xy = None
    geometry = types.ModuleType("shapely.geometry")
    geometry.MultiPoint = geometry.box = geometry.mapping = None
    sys.modules["shapely"] = shapely
    sys.modules["shapely.geometry"] = geometry

from evaluate_m3m_gcp_lidar_success_v1 import (
    canonical_sha256,
    expected_packet_names,
    reference_cache_binding_mode,
)


class ReferenceCacheBindingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.binding = {
            "scene": "gcp_100000_20260610",
            "contract": {"path": "/old/contract.json", "bytes": 10, "sha256": "old"},
            "geometry_release_manifest": {"sha256": "geometry"},
            "formal_input_manifest": {"sha256": "input"},
            "lidar_inventory": {"sha256": "lidar"},
            "gcp_csv": {"sha256": "gcp"},
            "sim3": {"sha256": "sim3"},
            "roi_buffer_m": 8.0,
            "normal_minus_ellipsoid_m": 23.980600991639484,
            "reference_voxel_m": 0.05,
            "local_origin_utm49n_normal_height_m": [1.0, 2.0, 0.0],
            "laz_files": {"cloud0.laz": {"bytes": 1, "sha256": "cloud"}},
        }
        self.binding["canonical_sha256"] = canonical_sha256(self.binding)

    def test_exact_binding(self) -> None:
        self.assertEqual(
            reference_cache_binding_mode(self.binding, copy.deepcopy(self.binding)),
            "EXACT_BINDING",
        )

    def test_contract_file_identity_only_is_compatible(self) -> None:
        expected = copy.deepcopy(self.binding)
        expected["contract"] = {
            "path": "/new/contract.json",
            "bytes": 11,
            "sha256": "new",
        }
        expected["canonical_sha256"] = canonical_sha256(expected)
        self.assertEqual(
            reference_cache_binding_mode(self.binding, expected),
            "SCIENTIFIC_BINDING_EQUAL_CONTRACT_FILE_IDENTITY_CHANGED",
        )

    def test_scientific_binding_change_is_rejected(self) -> None:
        for key, value in (
            ("reference_voxel_m", 0.10),
            ("roi_buffer_m", 9.0),
            ("normal_minus_ellipsoid_m", 0.0),
        ):
            with self.subTest(key=key):
                expected = copy.deepcopy(self.binding)
                expected[key] = value
                expected["canonical_sha256"] = canonical_sha256(expected)
                self.assertIsNone(reference_cache_binding_mode(self.binding, expected))

    def test_missing_contract_identity_is_rejected(self) -> None:
        expected = copy.deepcopy(self.binding)
        expected["contract"] = None
        expected["canonical_sha256"] = canonical_sha256(expected)
        self.assertIsNone(reference_cache_binding_mode(self.binding, expected))

    def test_invalid_binding_self_hash_is_rejected(self) -> None:
        expected = copy.deepcopy(self.binding)
        expected["contract"]["sha256"] = "new"
        self.assertIsNone(reference_cache_binding_mode(self.binding, expected))

    def test_surface_tracks_bind_exact_train_and_heldout_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            split_path = Path(directory) / "split.json"
            split_path.write_text(
                json.dumps(
                    {
                        "scenes": [
                            {
                                "scene": "gcp_100000_20260610",
                                "train_image_names": [
                                    f"train_{index:04d}.JPG"
                                    for index in range(2196)
                                ],
                                "test_image_names": [
                                    f"test_{index:04d}.JPG"
                                    for index in range(314)
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            train = expected_packet_names(split_path, "full_train")
            heldout = expected_packet_names(split_path, "heldout_candidate")
            self.assertEqual(len(train), 2196)
            self.assertEqual(len(heldout), 314)
            self.assertTrue(all(name.startswith("test_") for name in heldout))


if __name__ == "__main__":
    unittest.main()
