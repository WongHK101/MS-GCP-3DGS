#!/usr/bin/env python3
"""Exact transient-checkpoint cleanup tests for CityGaussianV2 100K."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from run_citygaussian_v2_100k_pipeline import (
    cleanup_transient_checkpoints,
    sha256,
)


class CityGaussianV2100KPipelineTest(unittest.TestCase):
    def test_only_inventoried_coarse_and_block_checkpoints_are_removed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            coarse = root / "coarse/checkpoints/step=30000.ckpt"
            coarse.parent.mkdir(parents=True)
            coarse.write_bytes(b"coarse")
            blocks = []
            for block_id in range(16):
                path = root / f"fine/blocks/block_{block_id}/checkpoints/step=60000.ckpt"
                path.parent.mkdir(parents=True)
                path.write_bytes(f"block-{block_id}".encode("ascii"))
                blocks.append(path)
            merged = root / "fine/checkpoints/merged.ckpt"
            merged.parent.mkdir(parents=True)
            merged.write_bytes(b"merged")
            keep = root / "runtime/fine.yaml"
            keep.parent.mkdir(parents=True)
            keep.write_text("config", encoding="utf-8")
            inventory = root / "runtime/transient.json"

            result = cleanup_transient_checkpoints(root, merged, inventory)
            self.assertEqual(result["file_count"], 17)
            self.assertTrue(result["all_inventoried_files_removed"])
            self.assertFalse(coarse.exists())
            self.assertTrue(all(not path.exists() for path in blocks))
            self.assertEqual(sha256(merged), sha256(root / "fine/checkpoints/merged.ckpt"))
            self.assertTrue(keep.is_file())
            payload = json.loads(inventory.read_text(encoding="utf-8"))
            self.assertEqual(payload["file_count"], 17)
            self.assertEqual(payload["merged_checkpoint"]["sha256"], sha256(merged))


if __name__ == "__main__":
    unittest.main()
