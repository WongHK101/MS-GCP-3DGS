#!/usr/bin/env python3
"""Focused tests for the pre-registered GS-GCP original-3DGS recipe."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from validate_gs_gcp_v13_original_3dgs_recipe import validate_recipe


ROOT = Path(__file__).resolve().parents[2]
RECIPE = json.loads((ROOT / "configs" / "gs_gcp_v13_original_3dgs_recipe_v2.json").read_text(encoding="utf-8"))


def errors(recipe: dict) -> list[str]:
    return validate_recipe(recipe, repo_root=ROOT)


def test_recipe_passes() -> None:
    assert errors(copy.deepcopy(RECIPE)) == []


def test_rejects_r8_resolution() -> None:
    recipe = copy.deepcopy(RECIPE)
    recipe["scene"]["resolution_argument"] = 8
    assert any("resolution_argument" in item for item in errors(recipe))


def test_rejects_missing_resolution_probe() -> None:
    recipe = copy.deepcopy(RECIPE)
    recipe["scene"]["loaded_tensor_hash_probe_required"] = False
    assert any("loaded_tensor_hash_probe_required" in item for item in errors(recipe))


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


def test_rejects_old_run_namespace() -> None:
    recipe = copy.deepcopy(RECIPE)
    recipe["server_roots"]["run_root_template"] = "/root/autodl-tmp/runs/legacy/3dgs/<run_id>"
    assert any("gs-gcp-v13 namespace" in item for item in errors(recipe))


def test_rejects_mutable_isolation_policy() -> None:
    recipe = copy.deepcopy(RECIPE)
    recipe["isolation"]["overwrite_policy"] = "overwrite"
    assert any("overwrite_policy" in item for item in errors(recipe))


def main() -> int:
    tests = [
        test_recipe_passes,
        test_rejects_r8_resolution,
        test_rejects_missing_resolution_probe,
        test_rejects_training_parameter_change,
        test_rejects_gcp_training_leakage,
        test_rejects_mutated_official_commit,
        test_rejects_training_source_patch,
        test_rejects_old_run_namespace,
        test_rejects_mutable_isolation_policy,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS {len(tests)}/{len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
