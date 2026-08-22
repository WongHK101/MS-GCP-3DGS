#!/usr/bin/env python3
"""CPU-only source-patching tests for the bounded RaDe-GS rescue."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from materialize_rade_gs_ply_from_checkpoint import ply_vertex_count
from run_rade_gs_checkpoint_first_rescue import (
    NONFINAL_CHECKPOINT_CONDITION,
    ORIGINAL_CHECKPOINT_CONDITION,
    ORIGINAL_SAVE_BLOCK,
    patch_training_source,
)


class RadeSerializationRescueTest(unittest.TestCase):
    def test_patch_is_single_and_checkpoint_precedes_ply(self) -> None:
        source = ORIGINAL_SAVE_BLOCK + "\n" + ORIGINAL_CHECKPOINT_CONDITION + "    pass\n"
        patched = patch_training_source(source)
        self.assertEqual(
            patched.count("_m3m_save_final_serialization_state("), 1
        )
        self.assertEqual(patched.count(NONFINAL_CHECKPOINT_CONDITION), 1)
        self.assertLess(
            patched.index("_m3m_save_final_serialization_state("),
            patched.index("scene.save(iteration)"),
        )

    def test_ply_vertex_header_reader(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.ply"
            path.write_bytes(
                b"ply\nformat binary_little_endian 1.0\nelement vertex 17\nend_header\n"
            )
            self.assertEqual(ply_vertex_count(path), 17)


if __name__ == "__main__":
    unittest.main()
