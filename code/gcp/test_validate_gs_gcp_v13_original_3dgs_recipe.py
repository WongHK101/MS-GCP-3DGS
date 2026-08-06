#!/usr/bin/env python3
"""Focused tests for the clean GS-GCP original-3DGS R4 recipe."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from validate_gs_gcp_v13_original_3dgs_recipe import validate_recipe


ROOT = Path(__file__).resolve().parents[2]
RECIPE = json.loads((ROOT / "configs" / "gs_gcp_v13_original_3dgs_recipe_v3.json").read_text(encoding="utf-8"))


def errors(recipe: dict) -> list[str]:
    return validate_recipe(recipe, repo_root=ROOT)


def test_recipe_passes() -> None:
    assert errors(copy.deepcopy(RECIPE)) == []


def test_rejects_non_materialized_resolution() -> None:
    recipe = copy.deepcopy(RECIPE)
    recipe["qualification_scene"]["official_resolution_argument"] = 4
    assert any("official_resolution_argument" in item for item in errors(recipe))


def test_rejects_missing_materialization_probe() -> None:
    recipe = copy.deepcopy(RECIPE)
    recipe["execution"]["materialized_input_verification_required"] = False
    assert any("materialized_input_verification_required" in item for item in errors(recipe))


def test_rejects_training_parameter_change() -> None:
    recipe = copy.deepcopy(RECIPE)
    recipe["training"]["densify_until_iter"] = 16000
    assert any("densify_until_iter" in item for item in errors(recipe))


def test_rejects_gcp_training_leakage() -> None:
    recipe = copy.deepcopy(RECIPE)
    recipe["training"]["split_role_labels_visible_to_optimizer"] = True
    assert any("split_role_labels_visible_to_optimizer" in item for item in errors(recipe))


def test_rejects_mutated_official_commit() -> None:
    recipe = copy.deepcopy(RECIPE)
    recipe["source_provenance"]["repository_commit"] = "0" * 40
    assert any("repository_commit" in item for item in errors(recipe))


def test_rejects_training_source_patch() -> None:
    recipe = copy.deepcopy(RECIPE)
    recipe["build_compatibility"]["training_source_modified"] = True
    assert any("training_source_modified" in item for item in errors(recipe))


def test_rejects_serializer_patch() -> None:
    recipe = copy.deepcopy(RECIPE)
    recipe["build_compatibility"]["serializer_patch_allowed"] = True
    assert any("serializer_patch_allowed" in item for item in errors(recipe))


def test_rejects_old_run_namespace() -> None:
    recipe = copy.deepcopy(RECIPE)
    recipe["server_roots"]["run_root_template"] = "/root/autodl-tmp/runs/legacy/3dgs/<run_id>"
    assert any("clean R4 route" in item for item in errors(recipe))


def test_rejects_mutable_isolation_policy() -> None:
    recipe = copy.deepcopy(RECIPE)
    recipe["isolation"]["overwrite_policy"] = "overwrite"
    assert any("overwrite_policy" in item for item in errors(recipe))


TESTS = [
    test_recipe_passes,
    test_rejects_non_materialized_resolution,
    test_rejects_missing_materialization_probe,
    test_rejects_training_parameter_change,
    test_rejects_gcp_training_leakage,
    test_rejects_mutated_official_commit,
    test_rejects_training_source_patch,
    test_rejects_serializer_patch,
    test_rejects_old_run_namespace,
    test_rejects_mutable_isolation_policy,
]


def main() -> int:
    for test in TESTS:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS {len(TESTS)}/{len(TESTS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
