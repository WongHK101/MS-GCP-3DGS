#!/usr/bin/env python3
"""Focused tests for the frozen v1.3 original-3DGS recipe."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from validate_gcp_v13_original_3dgs_recipe import validate_recipe


ROOT = Path(__file__).resolve().parents[2]
RECIPE = json.loads((ROOT / "configs" / "gcp_v13_original_3dgs_recipe_v1.json").read_text(encoding="utf-8"))
SIX_SCENE_RECIPE = json.loads(
    (ROOT / "configs" / "gcp_v13_original_3dgs_six_scene_recipe_v1.json").read_text(encoding="utf-8")
)


def errors(recipe: dict) -> list[str]:
    return validate_recipe(recipe, repo_root=ROOT)


def test_frozen_recipe_passes() -> None:
    assert errors(copy.deepcopy(RECIPE)) == []


def test_frozen_six_scene_recipe_passes() -> None:
    assert errors(copy.deepcopy(SIX_SCENE_RECIPE)) == []


def test_rejects_missing_six_scene() -> None:
    recipe = copy.deepcopy(SIX_SCENE_RECIPE)
    del recipe["scenes"]["gcp_5000_20260602"]
    assert any("exact frozen scene set" in item for item in errors(recipe))


def test_rejects_six_scene_source_hash_change() -> None:
    recipe = copy.deepcopy(SIX_SCENE_RECIPE)
    recipe["scenes"]["gcp_5000_20260602"]["images_bin_sha256"] = "0" * 64
    assert any("gcp_5000_20260602.images_bin_sha256" in item for item in errors(recipe))


def test_rejects_training_parameter_change() -> None:
    recipe = copy.deepcopy(RECIPE)
    recipe["training"]["densify_until_iter"] = 16000
    assert any("densify_until_iter" in item for item in errors(recipe))


def test_rejects_gcp_training_leakage() -> None:
    recipe = copy.deepcopy(RECIPE)
    recipe["training"]["gcp_split_visible_to_training"] = True
    assert any("gcp_split_visible_to_training" in item for item in errors(recipe))


def test_rejects_mutated_official_commit() -> None:
    recipe = copy.deepcopy(RECIPE)
    recipe["source_provenance"]["repository_commit"] = "0" * 40
    assert any("official 3DGS commit" in item for item in errors(recipe))


def test_rejects_training_source_patch() -> None:
    recipe = copy.deepcopy(RECIPE)
    recipe["build_compatibility"]["training_source_modified"] = True
    assert any("training source must remain unmodified" in item for item in errors(recipe))


def test_rejects_patch_hash_mismatch() -> None:
    recipe = copy.deepcopy(RECIPE)
    recipe["build_compatibility"]["simple_knn_build_copy_patch"]["patch_sha256"] = "0" * 64
    assert any("patch SHA mismatch" in item for item in errors(recipe))


def test_rejects_shared_output_roots() -> None:
    recipe = copy.deepcopy(RECIPE)
    recipe["server_roots"]["run_root_template"] = "/root/autodl-tmp/runs/ms-gcp-v13/shared/<run_id>"
    assert any("run root is not method/scene-isolated" in item for item in errors(recipe))


def test_rejects_mutable_isolation_policy() -> None:
    recipe = copy.deepcopy(RECIPE)
    recipe["isolation"]["overwrite_policy"] = "overwrite"
    assert any("overwrite_policy" in item for item in errors(recipe))


def main() -> int:
    tests = [
        test_frozen_recipe_passes,
        test_frozen_six_scene_recipe_passes,
        test_rejects_missing_six_scene,
        test_rejects_six_scene_source_hash_change,
        test_rejects_training_parameter_change,
        test_rejects_gcp_training_leakage,
        test_rejects_mutated_official_commit,
        test_rejects_training_source_patch,
        test_rejects_patch_hash_mismatch,
        test_rejects_shared_output_roots,
        test_rejects_mutable_isolation_policy,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS {len(tests)}/{len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
