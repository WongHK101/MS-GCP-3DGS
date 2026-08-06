#!/usr/bin/env python3
"""Validate the locked clean-R4 original-3DGS full-matrix plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from validate_gs_gcp_method_registry import validate_registry
from validate_gs_gcp_v13_original_3dgs_recipe import validate_recipe


SCHEMA = "gs_gcp_v13_original_3dgs_full_matrix_plan_v2"
EXPECTED_ORDER = [
    "gcp_100000_20260610",
    "gcp_50000_20260610",
    "gcp_20000_20260602",
    "gcp_10000_20260610",
    "gcp_5000_20260602",
]
EXPECTED_SCENES = {
    "gcp_3000_20260602": (94, 82, 12, 1414, 1024),
    "gcp_5000_20260602": (101, 88, 13, 1414, 1025),
    "gcp_10000_20260610": (976, 854, 122, 1414, 1025),
    "gcp_20000_20260602": (298, 260, 38, 1414, 1024),
    "gcp_50000_20260610": (2208, 1932, 276, 1414, 1025),
    "gcp_100000_20260610": (2510, 2196, 314, 1414, 1025),
}


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def validate_plan(
    plan: dict[str, Any],
    registry: dict[str, Any],
    repo_root: Path,
    *,
    scene: str | None = None,
    scene_root: Path | None = None,
    release_root: Path | None = None,
) -> dict[str, Any]:
    del scene_root, release_root
    errors: list[str] = []
    _require(plan.get("schema") == SCHEMA, "unknown full-matrix plan schema", errors)
    _require(plan.get("status") == "blocked_pending_clean_r4_3k_qualification", "full matrix must remain locked", errors)
    _require(plan.get("method_id") == "3dgs_original", "method must be original 3DGS", errors)
    _require(
        plan.get("training_recipe") == "configs/gs_gcp_v13_original_3dgs_recipe_v3.json",
        "active recipe must be clean R4 v3",
        errors,
    )
    _require(
        plan.get("input_materialization_contract") == "configs/gs_gcp_r4_input_materialization_v1.json",
        "active R4 input contract mismatch",
        errors,
    )

    registry_result = validate_registry(registry, repo_root)
    _require(registry_result["passed"], "method registry validation failed", errors)
    _require(registry_result["full_scene_matrix_eligible"] == [], "no method may enter the full matrix yet", errors)
    _require(registry_result["qualification_allowed"] == ["3dgs_original"], "3DGS must remain admitted for 3K only", errors)

    recipe_path = repo_root / str(plan.get("training_recipe", ""))
    if not recipe_path.is_file():
        errors.append("active recipe is missing")
    else:
        recipe_errors = validate_recipe(json.loads(recipe_path.read_text(encoding="utf-8")), repo_root=repo_root)
        errors.extend(f"recipe: {item}" for item in recipe_errors)

    qualification = plan.get("qualification", {})
    _require(qualification.get("scene") == "gcp_3000_20260602", "qualification scene mismatch", errors)
    _require(qualification.get("status") == "NOT_RUN", "clean R4 qualification must not inherit a result", errors)
    _require(qualification.get("full_matrix_authorized") is False, "full matrix must not be authorized", errors)
    _require(
        qualification.get("external_review_status") == "CLEAN_R4_CONTRACT_PASS",
        "clean R4 contract review must pass before qualification",
        errors,
    )
    review_evidence = (repo_root / str(qualification.get("contract_review_evidence", ""))).resolve()
    _require(review_evidence.is_relative_to(repo_root.resolve()), "clean R4 contract review evidence escapes repository", errors)
    _require(review_evidence.is_file(), "clean R4 contract review evidence is missing", errors)
    if review_evidence.is_file():
        import hashlib

        actual_review_sha = hashlib.sha256(review_evidence.read_bytes()).hexdigest()
        _require(
            actual_review_sha == qualification.get("contract_review_evidence_sha256"),
            "clean R4 contract review evidence SHA mismatch",
            errors,
        )
    legacy = plan.get("legacy_route", {})
    _require(legacy.get("formal_reuse_allowed") is False, "legacy full-matrix evidence must not be reused", errors)

    identity = plan.get("frozen_method_identity", {})
    expected_identity = {
        "training_source_commit": "2eee0e26d2d5fd00ec462df47752223952f6bf4e",
        "training_source_tree": "5eee127dfc0942bf83d9fdd72328e03ec0cbf6c4",
        "runtime_training_patch": None,
        "training_iterations": 30000,
        "seed": 0,
        "formal_model": "point_cloud/iteration_30000/point_cloud.ply",
        "resolution_rule_id": "graphdeco_quarter_resolution_v1",
        "official_resolution_argument_on_materialized_inputs": 1,
        "formal_tensor": "alpha_normalized_expected_camera_z",
        "formal_formula": "M1/A",
        "formal_semantics": "camera_z",
    }
    for key, expected in expected_identity.items():
        _require(identity.get(key) == expected, f"frozen identity mismatch: {key}", errors)

    _require(plan.get("execution_order_after_qualification") == EXPECTED_ORDER, "execution order mismatch", errors)
    execution = plan.get("execution", {})
    _require(execution.get("old_checkpoint_or_result_reuse") is False, "old results must not be reused", errors)
    _require(execution.get("overwrite_policy") == "fail_if_exists", "run roots must not be overwritten", errors)
    _require(execution.get("fresh_hardware_manifest_required") is True, "fresh hardware binding is required", errors)

    scenes = plan.get("scenes", [])
    scene_ids = [row.get("scene") for row in scenes if isinstance(row, dict)]
    _require(len(scene_ids) == len(set(scene_ids)), "duplicate scene rows", errors)
    _require(set(scene_ids) == set(EXPECTED_SCENES), "scene set mismatch", errors)
    by_scene = {row["scene"]: row for row in scenes if isinstance(row, dict) and row.get("scene")}
    for scene_id, expected in EXPECTED_SCENES.items():
        row = by_scene.get(scene_id, {})
        values = tuple(
            row.get(field)
            for field in ("full_view_count", "train_view_count", "test_view_count", "loaded_width", "loaded_height")
        )
        _require(values == expected, f"{scene_id}: counts or R4 dimensions mismatch", errors)
        expected_status = (
            "pending_clean_r4_qualification"
            if scene_id == "gcp_3000_20260602"
            else "blocked_pending_clean_r4_3k_pass"
        )
        _require(row.get("status") == expected_status, f"{scene_id}: status mismatch", errors)

    if scene is not None:
        errors.append("full-matrix runtime launch is locked until clean R4 3K qualification and external acceptance")

    return {
        "schema": "gs_gcp_v13_original_3dgs_full_matrix_validation_v2",
        "passed": not errors,
        "launch_authorized": False,
        "method_id": plan.get("method_id"),
        "scene_count": len(scene_ids),
        "execution_order_after_qualification": plan.get("execution_order_after_qualification", []),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--repo_root", type=Path, default=root)
    parser.add_argument("--plan", type=Path, default=root / "configs/gs_gcp_v13_original_3dgs_full_matrix_v2.json")
    parser.add_argument("--registry", type=Path, default=root / "configs/gs_gcp_method_registry_v1.json")
    parser.add_argument("--scene")
    parser.add_argument("--scene_root", type=Path)
    parser.add_argument("--release_root", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = validate_plan(
        json.loads(args.plan.read_text(encoding="utf-8")),
        json.loads(args.registry.read_text(encoding="utf-8")),
        args.repo_root.resolve(),
        scene=args.scene,
        scene_root=args.scene_root.resolve() if args.scene_root else None,
        release_root=args.release_root.resolve() if args.release_root else None,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
