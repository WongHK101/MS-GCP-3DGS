from __future__ import annotations

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

from run_m3m_gcp_3k_heldout_candidate_validation import (
    CANDIDATE_CAMERA_SETS,
    candidate_summary_row,
    candidate_export_command,
    materialize_core_compatible_candidate_manifest,
    rank_positions,
)
import evaluate_m3m_gcp_lidar_formal_v1 as evaluator


class CandidateExportCommandTests(unittest.TestCase):
    def test_reused_legacy_candidate_row_derives_surface_voxel_count(self) -> None:
        payload = {
            "summary_row": {"method_id": "3dgs_original", "fscore_10cm": 0.5},
            "surface_audit": {"voxelized_points": 1234},
        }
        row = candidate_summary_row(payload)
        self.assertEqual(row["reconstruction_points"], 1234)
        self.assertNotIn("reconstruction_points", payload["summary_row"])

    def test_candidate_camera_set_alias_is_explicit_and_formal_default_stays_strict(self) -> None:
        image_name = "heldout.JPG"
        manifest = {
            "schema": evaluator.REQUIRED_PACKET_SCHEMA,
            "protocol_id": evaluator.SOURCE_PROTOCOL_ID,
            "scene": "gcp_3000_20260602",
            "primary_depth_tensor": evaluator.PRIMARY_DEPTH,
            "primary_depth_semantics": "camera_z",
            "image_domain": evaluator.EXPECTED_IMAGE_DOMAIN,
            "pixel_coordinate_convention": evaluator.EXPECTED_PIXEL_CONVENTION,
            "camera_z_unit_contract": "frozen_colmap_model_camera_z_units",
            "adapter_conformance_status": "PASS",
            "rendered_view_count": 1,
            "camera_sets": "frozen_evaluation_allowlist",
            "depth_index": [{"image_name": image_name}],
        }
        with self.assertRaisesRegex(ValueError, "formal v1 requires exact training-view packets"):
            evaluator.validate_packet_manifest(
                manifest,
                scene="gcp_3000_20260602",
                expected_image_names=(image_name,),
            )
        self.assertIn(manifest["camera_sets"], CANDIDATE_CAMERA_SETS)
        with tempfile.TemporaryDirectory() as temp:
            source_path = Path(temp) / "depth_export_manifest.json"
            source_path.write_text(json.dumps(manifest), encoding="utf-8")
            effective_path, effective, alias_receipt = (
                materialize_core_compatible_candidate_manifest(
                    source_path, names=(image_name,)
                )
            )
            self.assertNotEqual(effective_path, source_path)
            self.assertEqual(effective["camera_sets"], "train")
            self.assertEqual(json.loads(source_path.read_text())["camera_sets"], manifest["camera_sets"])
            self.assertIsNotNone(alias_receipt)
            evaluator.validate_packet_manifest(
                effective,
                scene="gcp_3000_20260602",
                expected_image_names=(image_name,),
            )

    def test_only_candidate_output_options_are_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cwd = root / "cwd"
            cwd.mkdir()
            original = {
                "argv": [
                    "/env/python",
                    "export.py",
                    "--camera_sets",
                    "train",
                    "--image_list_csv",
                    "/old/allowlist.csv",
                    "--depth_output_dir",
                    "/old/packets",
                    "--manifest_path",
                    "/old/packets/depth_export_manifest.json",
                    "--mapping_csv",
                    "/old/packets/depth_map_index.csv",
                    "--quiet",
                ],
                "working_directory": str(cwd),
                "runtime_environment": {},
            }
            allowlist = root / "allowlist.csv"
            packet_root = root / "candidate"
            argv, actual_cwd, _ = candidate_export_command(
                original,
                allowlist=allowlist,
                packet_root=packet_root,
                environment_override={"PYTHONPATH": "/proven/eval-site"},
            )
            self.assertEqual(actual_cwd, cwd.resolve())
            self.assertEqual(argv[argv.index("--camera_sets") + 1], "train")
            self.assertEqual(argv[argv.index("--image_list_csv") + 1], str(allowlist.resolve()))
            self.assertEqual(argv[argv.index("--depth_output_dir") + 1], str(packet_root.resolve()))
            self.assertEqual(
                argv[argv.index("--manifest_path") + 1],
                str((packet_root / "depth_export_manifest.json").resolve()),
            )
            self.assertEqual(
                argv[argv.index("--mapping_csv") + 1],
                str((packet_root / "depth_map_index.csv").resolve()),
            )

    def test_environment_override_is_applied(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cwd = root / "cwd"
            cwd.mkdir()
            original = {
                "argv": [
                    "/env/python",
                    "export.py",
                    "--image_list_csv",
                    "/old/list.csv",
                    "--depth_output_dir",
                    "/old/packets",
                    "--manifest_path",
                    "/old/manifest.json",
                    "--mapping_csv",
                    "/old/map.csv",
                ],
                "working_directory": str(cwd),
                "runtime_environment": {"OLD": "kept"},
            }
            _, _, environment = candidate_export_command(
                original,
                allowlist=root / "list.csv",
                packet_root=root / "packets",
                environment_override={"PYTHONPATH": "/proven/eval-site"},
            )
            self.assertEqual(environment["OLD"], "kept")
            self.assertEqual(environment["PYTHONPATH"], "/proven/eval-site")

    def test_rank_uses_fscore_then_chamfer_then_precision(self) -> None:
        rows = [
            {
                "method_id": "b",
                "fscore_10cm": 0.5,
                "chamfer_l1_mean_m": 0.2,
                "precision_10cm": 0.4,
            },
            {
                "method_id": "a",
                "fscore_10cm": 0.5,
                "chamfer_l1_mean_m": 0.1,
                "precision_10cm": 0.3,
            },
            {
                "method_id": "c",
                "fscore_10cm": 0.4,
                "chamfer_l1_mean_m": 0.01,
                "precision_10cm": 0.9,
            },
        ]
        self.assertEqual(rank_positions(rows), {"a": 1, "b": 2, "c": 3})


if __name__ == "__main__":
    unittest.main()
