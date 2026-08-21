#!/usr/bin/env python3
"""Storage-preserving preparation tests for the MetroGS 100K prior route."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from prepare_metrogs_100k_training_priors import (
    image_files,
    replace_segment_rgb_copies_with_hardlinks,
    sha256,
)


class MetroGS100KPriorTest(unittest.TestCase):
    def test_segment_copies_become_exact_frozen_rgb_hardlinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "images"
            source.mkdir()
            source_files = []
            for index in range(4):
                path = source / f"image_{index}.JPG"
                path.write_bytes(f"rgb-{index}".encode("ascii"))
                source_files.append(path)
            segments = root / "segments"
            for block_idx in range(2):
                block = segments / f"block_{block_idx}" / "images"
                block.mkdir(parents=True)
                for source_path in source_files[block_idx * 2 : (block_idx + 1) * 2]:
                    shutil.copy2(source_path, block / f"renamed_{source_path.name}")

            summary = replace_segment_rgb_copies_with_hardlinks(segments, source, 2)
            self.assertEqual(summary["segment_link_count"], 4)
            self.assertEqual(summary["additional_physical_rgb_bytes"], 0)
            source_by_hash = {sha256(path): path for path in source_files}
            observed = Counter()
            for image in image_files(segments):
                frozen = source_by_hash[sha256(image)]
                self.assertEqual(
                    (image.stat().st_dev, image.stat().st_ino),
                    (frozen.stat().st_dev, frozen.stat().st_ino),
                )
                observed[sha256(image)] += 1
            self.assertEqual(observed, Counter(sha256(path) for path in source_files))


if __name__ == "__main__":
    unittest.main()
