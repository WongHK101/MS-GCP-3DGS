#!/usr/bin/env python3
"""Unit tests for six-scene preparation evidence normalization."""

from __future__ import annotations

import unittest

from build_m3m_six_scene_preparation_manifest import materialization_provenance


class MaterializationProvenanceTest(unittest.TestCase):
    def test_legacy_manifest_is_explicit_and_byte_normative(self) -> None:
        self.assertEqual(
            materialization_provenance({}),
            {
                "mode": "legacy_manifest_field_absent",
                "semantic_identity": "file bytes and manifest hashes are normative",
            },
        )

    def test_current_manifest_is_preserved(self) -> None:
        current = {
            "mode": "hardlink",
            "semantic_identity": "file bytes are normative",
        }
        self.assertIs(materialization_provenance({"file_materialization": current}), current)

    def test_malformed_field_fails_closed(self) -> None:
        with self.assertRaises(TypeError):
            materialization_provenance({"file_materialization": "hardlink"})


if __name__ == "__main__":
    unittest.main()
