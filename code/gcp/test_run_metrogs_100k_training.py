#!/usr/bin/env python3
"""Transient rank-checkpoint cleanup test for MetroGS 100K."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from run_metrogs_100k_training import cleanup_rank_checkpoint, sha256


class MetroGS100KTrainingTest(unittest.TestCase):
    def test_rank_checkpoint_is_inventoried_and_merged_is_retained(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory).resolve() / "model"
            checkpoints = model / "checkpoints"
            checkpoints.mkdir(parents=True)
            rank = checkpoints / "run-step=150000-rank=0.ckpt"
            merged = checkpoints / "run-step=150000.ckpt"
            rank.write_bytes(b"rank")
            merged.write_bytes(b"merged")
            merged_sha = sha256(merged)

            result = cleanup_rank_checkpoint(
                model_path=model,
                rank_checkpoint=rank,
                merged_checkpoint=merged,
            )
            self.assertFalse(rank.exists())
            self.assertTrue(merged.is_file())
            self.assertEqual(sha256(merged), merged_sha)
            self.assertFalse(result["rank_checkpoint"]["retained_after_merge"])
            self.assertTrue(result["merged_checkpoint"]["retained_after_merge"])
            inventory = json.loads(
                Path(result["inventory_path"]).read_text(encoding="utf-8")
            )
            self.assertEqual(
                inventory["rank_checkpoint"]["sha256"],
                result["rank_checkpoint"]["sha256"],
            )


if __name__ == "__main__":
    unittest.main()
