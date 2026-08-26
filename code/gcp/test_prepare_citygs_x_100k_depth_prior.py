#!/usr/bin/env python3
"""Focused checks for CityGS-X's official zero-neighbour mask semantics."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from prepare_citygs_x_100k_depth_prior import require_official_mask_inventory


class CityGSXMaskInventoryTest(unittest.TestCase):
    def test_accepts_only_upstream_zero_neighbour_omissions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.png").write_bytes(b"a")
            (root / "c.png").write_bytes(b"c")
            rows = [
                {"ref_name": "a", "nearest_name": ["c"]},
                {"ref_name": "b", "nearest_name": []},
                {"ref_name": "c", "nearest_name": ["a"]},
            ]
            (root / "multi_view.json").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )

            files, missing, index = require_official_mask_inventory(
                root, {"a", "b", "c"}
            )
            self.assertEqual([path.stem for path in files], ["a", "c"])
            self.assertEqual(missing, ["b"])
            self.assertEqual(index, root / "multi_view.json")

    def test_rejects_missing_mask_for_camera_with_neighbours(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.png").write_bytes(b"a")
            rows = [
                {"ref_name": "a", "nearest_name": ["b"]},
                {"ref_name": "b", "nearest_name": ["a"]},
            ]
            (root / "multi_view.json").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )

            with self.assertRaisesRegex(RuntimeError, "zero-neighbour rule"):
                require_official_mask_inventory(root, {"a", "b"})


if __name__ == "__main__":
    unittest.main()
