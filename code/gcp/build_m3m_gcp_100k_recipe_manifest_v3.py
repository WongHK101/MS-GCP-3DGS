#!/usr/bin/env python3
"""Build manifest v3 with one Linux-identity-only 3DGS recipe correction."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OLD_MANIFEST_REL = "configs/m3m_gcp_native_quarter_100k_recipe_manifest_v2.json"
OLD_RECIPE_REL = "configs/m3m_gcp_native_quarter_100k_recipes_v2/3dgs_original.json"
NEW_MANIFEST_REL = "configs/m3m_gcp_native_quarter_100k_recipe_manifest_v3.json"
NEW_RECIPE_REL = "configs/m3m_gcp_native_quarter_100k_recipes_v3/3dgs_original.json"
LINUX_PROOF_REL = (
    "docs/protocol_evidence/"
    "3dgs_native_quarter_adapter_linux_identity_proof_v1.json"
)
RENDERER_PATCH_REL = (
    "patches/3dgs_original/native_quarter_raw_moment_renderer_2eee0e26_v1.patch"
)
RASTERIZER_PATCH_REL = (
    "patches/3dgs_original/native_quarter_raw_moment_rasterizer_59f5f77_v1.patch"
)
WINDOWS_HEADER_SHA = "7fdf17df5880f2819551e70a162937abff526b7e6b0337ccb8d6fe184f18c3f2"
LINUX_HEADER_SHA = "c4f5f2df458e75290bdaff7510b87395d5b8ca47ef07b51f47c9b5cb7e580629"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: dict[str, Any]) -> str:
    clean = {key: value for key, value in payload.items() if key != "canonical_sha256"}
    raw = json.dumps(clean, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> int:
    old_manifest_path = ROOT / OLD_MANIFEST_REL
    old_recipe_path = ROOT / OLD_RECIPE_REL
    proof_path = ROOT / LINUX_PROOF_REL
    renderer_patch = ROOT / RENDERER_PATCH_REL
    rasterizer_patch = ROOT / RASTERIZER_PATCH_REL
    old_manifest = json.loads(old_manifest_path.read_text(encoding="utf-8"))
    old_recipe = json.loads(old_recipe_path.read_text(encoding="utf-8"))
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    if (
        old_manifest.get("schema")
        != "m3m_gcp_native_quarter_100k_recipe_manifest_v2"
        or old_manifest.get("canonical_sha256") != canonical_sha256(old_manifest)
        or old_recipe.get("schema")
        != "m3m_gcp_native_quarter_100k_execution_recipe_v2"
        or old_recipe.get("method_id") != "3dgs_original"
        or old_recipe.get("canonical_sha256") != canonical_sha256(old_recipe)
    ):
        raise RuntimeError("frozen v2 recipe source identity mismatch")
    if (
        proof.get("schema") != "m3m_3dgs_eval_adapter_linux_identity_proof_v1"
        or proof.get("status") != "PASS"
        or proof.get("passed") is not True
        or proof.get("rasterizer_files", {}).get("rasterize_points.h")
        != LINUX_HEADER_SHA
        or proof.get("renderer_patch_sha256") != sha256_file(renderer_patch)
        or proof.get("rasterizer_patch_sha256") != sha256_file(rasterizer_patch)
    ):
        raise RuntimeError("formal Linux 3DGS identity proof mismatch")

    linux_files = {
        **{
            relative: digest
            for relative, digest in proof.get("renderer_files", {}).items()
        },
        **{
            f"submodules/diff-gaussian-rasterization/{relative}": digest
            for relative, digest in proof.get("rasterizer_files", {}).items()
        },
    }
    if len(linux_files) != 8 or set(linux_files) != set(
        old_recipe["source_bindings"]["packet"]["required_files_sha256"]
    ):
        raise RuntimeError("Linux proof does not cover the exact eight patched files")
    old_header = old_recipe["source_bindings"]["packet"][
        "required_files_sha256"
    ].get("submodules/diff-gaussian-rasterization/rasterize_points.h")
    if old_header != WINDOWS_HEADER_SHA:
        raise RuntimeError("v2 recipe no longer contains the isolated Windows hash")

    recipe = copy.deepcopy(old_recipe)
    recipe["schema"] = "m3m_gcp_native_quarter_100k_execution_recipe_v3"
    recipe["source_bindings"]["packet"]["required_files_sha256"] = linux_files
    recipe["benchmark_required_files_sha256"][LINUX_PROOF_REL] = sha256_file(
        proof_path
    )
    recipe["source_identity_correction"] = {
        "type": "LINUX_IDENTITY_METADATA_CORRECTION_ONLY",
        "phase": "packet",
        "source_modified": False,
        "child_started": False,
        "attempt_consumed": False,
        "dual_hash_tolerance": False,
        "superseded_windows_header_sha256": WINDOWS_HEADER_SHA,
        "formal_linux_header_sha256": LINUX_HEADER_SHA,
        "linux_identity_proof": {
            "path": LINUX_PROOF_REL,
            "sha256": sha256_file(proof_path),
            "status_required": "PASS",
            "patched_file_count": 8,
        },
        "frozen_patches": [
            {"path": RENDERER_PATCH_REL, "sha256": sha256_file(renderer_patch)},
            {"path": RASTERIZER_PATCH_REL, "sha256": sha256_file(rasterizer_patch)},
        ],
    }
    recipe["canonical_sha256"] = canonical_sha256(recipe)
    new_recipe_path = ROOT / NEW_RECIPE_REL
    write_json(new_recipe_path, recipe)

    rows = copy.deepcopy(old_manifest["recipes"])
    if [row.get("method_id") for row in rows] != old_manifest.get("method_order"):
        raise RuntimeError("v2 manifest method order mismatch")
    rows[0] = {
        "method_id": "3dgs_original",
        "path": NEW_RECIPE_REL,
        "sha256": sha256_file(new_recipe_path),
        "canonical_sha256": recipe["canonical_sha256"],
    }
    manifest = {
        "schema": "m3m_gcp_native_quarter_100k_recipe_manifest_v3",
        "scene": old_manifest["scene"],
        "seed": old_manifest["seed"],
        "status": "REVIEW_CANDIDATE_NOT_EXECUTION_AUTHORIZED",
        "method_order": old_manifest["method_order"],
        "recipes": rows,
        "previous_manifest": {
            "path": OLD_MANIFEST_REL,
            "sha256": sha256_file(old_manifest_path),
            "canonical_sha256": old_manifest["canonical_sha256"],
        },
        "correction_scope": {
            "type": "LINUX_IDENTITY_METADATA_CORRECTION_ONLY",
            "changed_method_ids": ["3dgs_original"],
            "unchanged_v2_recipe_rows": 9,
            "source_modified": False,
            "method_order_budget_command_or_path_changed": False,
        },
    }
    manifest["canonical_sha256"] = canonical_sha256(manifest)
    write_json(ROOT / NEW_MANIFEST_REL, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
