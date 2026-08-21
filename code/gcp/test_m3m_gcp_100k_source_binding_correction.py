#!/usr/bin/env python3
"""Regression tests for the isolated 3DGS Linux identity correction."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from m3m_gcp_lidar_artifacts import canonical_sha256, sha256_file
from m3m_gcp_100k_source_binding_correction import (
    LINUX_HEADER_SHA,
    WINDOWS_HEADER_SHA,
    validate_source_binding_correction,
)


ROOT = Path(__file__).resolve().parents[2]
RECEIPT_REL = Path(
    "docs/protocol_evidence/"
    "m3m_gcp_100k_3dgs_linux_source_binding_correction_v1.json"
)


class SourceBindingCorrectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name).resolve()
        source_receipt = json.loads((ROOT / RECEIPT_REL).read_text(encoding="utf-8"))
        for row in source_receipt["repository_artifacts"]:
            source = ROOT / row["path"]
            destination = self.repo / row["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        self.receipt_path = self.repo / RECEIPT_REL
        self.receipt_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / RECEIPT_REL, self.receipt_path)
        self.plan = self._plan()

    def _receipt(self) -> dict:
        return json.loads(self.receipt_path.read_text(encoding="utf-8"))

    def _write_json(self, path: Path, payload: dict) -> None:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    def _plan(self) -> dict:
        receipt = self._receipt()
        manifest_row = next(
            row for row in receipt["repository_artifacts"]
            if row["role"] == "recipe_manifest_v3"
        )
        return {
            "source_binding_correction": {
                "receipt": {
                    "path": RECEIPT_REL.as_posix(),
                    "sha256": sha256_file(self.receipt_path),
                },
                "status_required": "SEALED_LINUX_IDENTITY_METADATA_CORRECTION",
                "type": "LINUX_IDENTITY_METADATA_CORRECTION_ONLY",
                "source_modified": False,
                "child_started": False,
                "attempt_consumed": False,
                "dual_hash_tolerance": False,
                "recipe_manifest": {
                    "path": manifest_row["path"],
                    "sha256": manifest_row["sha256"],
                    "canonical_sha256": manifest_row["canonical_sha256"],
                },
            }
        }

    def _rewrite_new_recipe_chain(self, header_value: object) -> None:
        receipt = self._receipt()
        rows = {row["role"]: row for row in receipt["repository_artifacts"]}
        recipe_path = self.repo / rows["3dgs_recipe_v3"]["path"]
        recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
        recipe["source_bindings"]["packet"]["required_files_sha256"][
            "submodules/diff-gaussian-rasterization/rasterize_points.h"
        ] = header_value
        recipe["canonical_sha256"] = canonical_sha256(recipe)
        self._write_json(recipe_path, recipe)
        rows["3dgs_recipe_v3"].update(
            bytes=recipe_path.stat().st_size,
            sha256=sha256_file(recipe_path),
            canonical_sha256=recipe["canonical_sha256"],
        )

        manifest_path = self.repo / rows["recipe_manifest_v3"]["path"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["recipes"][0].update(
            sha256=sha256_file(recipe_path),
            canonical_sha256=recipe["canonical_sha256"],
        )
        manifest["canonical_sha256"] = canonical_sha256(manifest)
        self._write_json(manifest_path, manifest)
        rows["recipe_manifest_v3"].update(
            bytes=manifest_path.stat().st_size,
            sha256=sha256_file(manifest_path),
            canonical_sha256=manifest["canonical_sha256"],
        )

        receipt["canonical_sha256"] = canonical_sha256(receipt)
        self._write_json(self.receipt_path, receipt)
        self.plan = self._plan()

    def test_exact_linux_correction_passes(self) -> None:
        receipt = validate_source_binding_correction(repo=self.repo, plan=self.plan)
        self.assertEqual(
            receipt["hash_correction"]["formal_linux_sha256"],
            LINUX_HEADER_SHA,
        )

    def test_windows_checkout_hash_is_rejected_even_if_chain_is_resealed(self) -> None:
        self._rewrite_new_recipe_chain(WINDOWS_HEADER_SHA)
        with self.assertRaisesRegex(RuntimeError, "single exact metadata correction"):
            validate_source_binding_correction(repo=self.repo, plan=self.plan)

    def test_missing_linux_identity_proof_is_rejected(self) -> None:
        receipt = self._receipt()
        row = next(
            item for item in receipt["repository_artifacts"]
            if item["role"] == "linux_identity_proof"
        )
        (self.repo / row["path"]).unlink()
        with self.assertRaises(RuntimeError):
            validate_source_binding_correction(repo=self.repo, plan=self.plan)

    def test_tampered_linux_identity_proof_is_rejected(self) -> None:
        receipt = self._receipt()
        row = next(
            item for item in receipt["repository_artifacts"]
            if item["role"] == "linux_identity_proof"
        )
        proof_path = self.repo / row["path"]
        proof_path.write_bytes(proof_path.read_bytes() + b"\n")
        with self.assertRaises(RuntimeError):
            validate_source_binding_correction(repo=self.repo, plan=self.plan)

    def test_dual_hash_tolerance_is_rejected_even_if_chain_is_resealed(self) -> None:
        self._rewrite_new_recipe_chain([LINUX_HEADER_SHA, WINDOWS_HEADER_SHA])
        with self.assertRaisesRegex(RuntimeError, "single exact metadata correction"):
            validate_source_binding_correction(repo=self.repo, plan=self.plan)


if __name__ == "__main__":
    unittest.main()
