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
    materialize_heldout_candidate_manifest_alias,
    reference_cache_binding_mode,
)
import evaluate_m3m_gcp_lidar_formal_v1 as core


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

    def test_heldout_camera_set_alias_does_not_relax_full_train_validator(self) -> None:
        name = "heldout.JPG"
        manifest = {
            "schema": core.REQUIRED_PACKET_SCHEMA,
            "protocol_id": core.SOURCE_PROTOCOL_ID,
            "scene": "gcp_100000_20260610",
            "primary_depth_tensor": core.PRIMARY_DEPTH,
            "primary_depth_semantics": "camera_z",
            "image_domain": core.EXPECTED_IMAGE_DOMAIN,
            "pixel_coordinate_convention": core.EXPECTED_PIXEL_CONVENTION,
            "camera_z_unit_contract": "frozen_colmap_model_camera_z_units",
            "adapter_conformance_status": "PASS",
            "rendered_view_count": 1,
            "camera_sets": "frozen_evaluation_allowlist",
            "depth_index": [{"image_name": name}],
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "depth_export_manifest.json"
            source.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "formal v1 requires exact training-view packets"
            ):
                materialize_heldout_candidate_manifest_alias(
                    source,
                    manifest,
                    surface_sampling_track="full_train",
                    expected_image_names=(name,),
                )
            effective_path, effective, receipt = (
                materialize_heldout_candidate_manifest_alias(
                    source,
                    manifest,
                    surface_sampling_track="heldout_candidate",
                    expected_image_names=(name,),
                )
            )
            self.assertNotEqual(source, effective_path)
            self.assertEqual(effective["camera_sets"], "train")
            self.assertEqual(json.loads(source.read_text())["camera_sets"], manifest["camera_sets"])
            self.assertIsNotNone(receipt)
            core.validate_packet_manifest(
                effective,
                scene="gcp_100000_20260610",
                expected_image_names=(name,),
            )


if __name__ == "__main__":
    unittest.main()
