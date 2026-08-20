#!/usr/bin/env python3
"""Cross-platform byte tests for frozen LiDAR train-view allowlists."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from build_m3m_gcp_lidar_train_allowlists import SCENES


class BuildLidarAllowlistsTest(unittest.TestCase):
    def test_outputs_are_lf_only_and_manifest_hashes_exact_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split_path = root / "split.json"
            split = {
                "manifest_sha256": "a" * 64,
                "scenes": [
                    {
                        "scene": scene,
                        "assignments": [
                            {
                                "image_name": f"{scene}_{index:05d}.JPG",
                                "split_role": "train",
                            }
                            for index in range(count)
                        ],
                    }
                    for scene, count in SCENES.items()
                ],
            }
            split_path.write_text(json.dumps(split), encoding="utf-8")
            output_dir = root / "allowlists"
            manifest_path = root / "manifest.json"
            subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(Path(__file__).with_name("build_m3m_gcp_lidar_train_allowlists.py")),
                    "--split",
                    str(split_path),
                    "--output-dir",
                    str(output_dir),
                    "--manifest",
                    str(manifest_path),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            manifest_bytes = manifest_path.read_bytes()
            self.assertNotIn(b"\r\n", manifest_bytes)
            manifest = json.loads(manifest_bytes)
            for row in manifest["rows"]:
                csv_path = Path(row["path"])
                payload = csv_path.read_bytes()
                self.assertNotIn(b"\r\n", payload)
                self.assertEqual(hashlib.sha256(payload).hexdigest(), row["sha256"])


if __name__ == "__main__":
    unittest.main()
