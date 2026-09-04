import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("run_uavgs_lidar_rectangular_roi_repair.py")
SPEC = importlib.util.spec_from_file_location("uavgs_rect_roi_runner", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class FullTrainGeometryLoaderRebindTests(unittest.TestCase):
    def test_rebinds_only_wrapper_and_benchmark_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            packet_root = Path(directory) / "packets"
            original = {
                "scene": "gcp_100000_20260610",
                "track": "train2196_sensitivity",
                "method_id": "3dgs_original",
                "expected_views": 2196,
                "packet_root": str(packet_root),
                "export_argv": [
                    "/python",
                    "-B",
                    "/old/code/gcp/run_m3m_gcp_100k_packet_export.py",
                    "--method-id",
                    "3dgs_original",
                    "--camera-profile",
                    "lidar",
                    "--benchmark-repo",
                    "/old",
                    "--packet-set-root",
                    str(packet_root),
                ],
                "export_environment": {"PYTHONHASHSEED": "0"},
                "export_working_directory": "/old",
                "source_command": {"recorded_argv_sha256": "frozen"},
            }
            rebound = MODULE.rebind_hundred_k_full_geometry_loader(original)
            repository = MODULE_PATH.resolve().parents[2]
            self.assertEqual(
                Path(rebound["export_argv"][2]),
                repository / "code/gcp/run_m3m_gcp_100k_packet_export.py",
            )
            self.assertEqual(
                Path(MODULE.flag_value(rebound["export_argv"], "--benchmark-repo")),
                repository,
            )
            self.assertEqual(rebound["source_command"]["camera_loader_policy"], "geometry_camera_only")
            self.assertEqual(original["export_argv"][2], "/old/code/gcp/run_m3m_gcp_100k_packet_export.py")

    def test_rejects_non_full_train_jobs(self) -> None:
        job = {
            "scene": "gcp_100000_20260610",
            "track": "heldout_main",
            "method_id": "3dgs_original",
            "expected_views": 314,
        }
        with self.assertRaisesRegex(ValueError, "restricted"):
            MODULE.rebind_hundred_k_full_geometry_loader(job)


if __name__ == "__main__":
    unittest.main()
