from __future__ import annotations

import importlib.util
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

from m3m_gcp_lidar_artifacts import command_sha256
from recover_m3m_gcp_100k_heldout_candidate import recovery_evaluate_phase


class HeldoutCandidateRecoveryTest(unittest.TestCase):
    def test_recovery_changes_only_evaluator_checkout_and_logs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            (repo / "code/gcp").mkdir(parents=True)
            source = {
                "argv": [
                    "/env/python",
                    "-B",
                    "/old/repo/code/gcp/evaluate_m3m_gcp_lidar_success_v1.py",
                    "--repo",
                    "/old/repo",
                    "--benchmark-commit",
                    "old-commit",
                    "--benchmark-tree",
                    "old-tree",
                    "--packet-manifest",
                    "/packets/depth_export_manifest.json",
                    "--method-id",
                    "citygs_x",
                ],
                "working_directory": "/old/repo",
                "environment": {"CUDA_VISIBLE_DEVICES": "0"},
                "stdout": "/old/stdout.log",
                "stderr": "/old/stderr.log",
            }
            recovered = recovery_evaluate_phase(
                source,
                repo=repo,
                benchmark_commit="new-commit",
                benchmark_tree="new-tree",
                log_root=root / "logs",
            )
            argv = recovered["argv"]
            self.assertEqual(argv[argv.index("--repo") + 1], str(repo.resolve()))
            self.assertEqual(argv[argv.index("--benchmark-commit") + 1], "new-commit")
            self.assertEqual(argv[argv.index("--benchmark-tree") + 1], "new-tree")
            self.assertEqual(
                argv[argv.index("--packet-manifest") + 1],
                "/packets/depth_export_manifest.json",
            )
            self.assertEqual(argv[argv.index("--method-id") + 1], "citygs_x")
            self.assertEqual(recovered["argv_sha256"], command_sha256(argv))
            self.assertEqual(recovered["working_directory"], str(repo.resolve()))
            self.assertEqual(recovered["stdout"], str((root / "logs/stdout.log").resolve()))
            self.assertEqual(recovered["stderr"], str((root / "logs/stderr.log").resolve()))
            self.assertEqual(source["argv"][3], "--repo")
            self.assertEqual(source["argv"][4], "/old/repo")

    def test_recovery_can_append_exact_parallel_query_workers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            (repo / "code/gcp").mkdir(parents=True)
            source = {
                "argv": [
                    "/env/python",
                    "-B",
                    "/old/repo/code/gcp/evaluate_m3m_gcp_lidar_success_v1.py",
                    "--repo",
                    "/old/repo",
                    "--benchmark-commit",
                    "old-commit",
                    "--benchmark-tree",
                    "old-tree",
                ],
                "working_directory": "/old/repo",
                "environment": {},
                "stdout": "/old/stdout.log",
                "stderr": "/old/stderr.log",
            }
            recovered = recovery_evaluate_phase(
                source,
                repo=repo,
                benchmark_commit="new-commit",
                benchmark_tree="new-tree",
                log_root=root / "logs",
                query_workers=-1,
            )
            argv = recovered["argv"]
            self.assertEqual(argv.count("--query-workers"), 1)
            self.assertEqual(argv[argv.index("--query-workers") + 1], "-1")
            self.assertNotIn("--query-workers", source["argv"])
            self.assertEqual(recovered["argv_sha256"], command_sha256(argv))


if __name__ == "__main__":
    unittest.main()
