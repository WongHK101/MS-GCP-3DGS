#!/usr/bin/env python3
"""CPU-only invariants for the frozen 100K packet dispatcher."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path

from run_m3m_gcp_100k_packet_export import (
    CITYGS_X_PYTORCH3D_COMPAT_RELATIVE,
    SCENE,
    verify_allowlist,
    verify_checkpoint,
)


class PacketExport100KTest(unittest.TestCase):
    def test_citygs_compatibility_package_is_repository_bound(self) -> None:
        compat = Path(__file__).resolve().parents[2] / CITYGS_X_PYTORCH3D_COMPAT_RELATIVE
        self.assertTrue((compat / "pytorch3d/__init__.py").is_file())
        self.assertTrue((compat / "pytorch3d/transforms/__init__.py").is_file())
        self.assertNotIn("staging", str(CITYGS_X_PYTORCH3D_COMPAT_RELATIVE))

    def test_allowlist_requires_exact_unique_2196(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "allowlist.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["image_name"])
                writer.writeheader()
                writer.writerows(
                    {"image_name": f"image_{index:04d}.JPG"} for index in range(2196)
                )
            verify_allowlist(path)
            with path.open("a", encoding="utf-8", newline="") as handle:
                handle.write("image_0000.JPG\n")
            with self.assertRaisesRegex(RuntimeError, "2196 unique"):
                verify_allowlist(path)

    def test_gsprior_scale_is_read_only_from_frozen_prior_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_root = root / "run"
            point_cloud = run_root / "model" / "point_cloud" / "iteration_40000" / "point_cloud.ply"
            point_cloud.parent.mkdir(parents=True)
            point_cloud.write_bytes(b"ply\n")
            prior_root = root / "prior"
            prior_root.mkdir()
            (prior_root / "normalization_manifest.json").write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "scene_id": SCENE,
                        "transform": {
                            "original_colmap_units_per_normalized_unit": 12.5
                        },
                    }
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                method_id="gsprior", training_run_root=run_root, prior_root=prior_root
            )
            model_path, scale = verify_checkpoint(args, {"iteration": 40000})
            self.assertEqual(model_path, (run_root / "model").resolve())
            self.assertEqual(scale, 12.5)


if __name__ == "__main__":
    unittest.main()
