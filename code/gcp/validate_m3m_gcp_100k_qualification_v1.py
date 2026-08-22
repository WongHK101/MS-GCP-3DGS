#!/usr/bin/env python3
"""Validate the generated 100K qualification plan and corrected route contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from build_m3m_gcp_100k_qualification_recipes import (
    COMMON_ENV,
    GNU_TIME,
    METHOD_INPUT_EVIDENCE_SHA,
    METHOD_ORDER,
    ROOT,
    canonical,
    sha256,
)


PLAN = ROOT / "configs" / "m3m_gcp_native_quarter_100k_qualification_v1.json"


def option_values(command: list[str], option: str) -> list[str]:
    index = command.index(option) + 1
    values = []
    while index < len(command) and not command[index].startswith("-"):
        values.append(command[index])
        index += 1
    return values


def load_recipe(row: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / row["path"]
    if sha256(path) != row["file_sha256"]:
        raise RuntimeError(f"recipe file identity mismatch: {path}")
    recipe = json.loads(path.read_text(encoding="utf-8"))
    if canonical(recipe) != recipe["canonical_sha256"]:
        raise RuntimeError(f"recipe canonical identity mismatch: {path}")
    if recipe["canonical_sha256"] != row["canonical_sha256"]:
        raise RuntimeError(f"plan-to-recipe identity mismatch: {path}")
    return recipe


def main() -> int:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    if canonical(plan) != plan["canonical_sha256"]:
        raise RuntimeError("qualification plan canonical identity mismatch")
    if plan["method_order"] != METHOD_ORDER:
        raise RuntimeError("qualification method order mismatch")
    if len(plan["recipes"]) != len(METHOD_ORDER):
        raise RuntimeError("qualification recipe count mismatch")
    if plan["per_method_audit"] != "NOT_REQUIRED_UNLESS_SCIENTIFIC_RED_LINE":
        raise RuntimeError("per-method audit was reintroduced")

    recipes = {row["method_id"]: load_recipe(row) for row in plan["recipes"]}
    if set(recipes) != set(METHOD_ORDER):
        raise RuntimeError("qualification method inventory mismatch")
    for method, recipe in recipes.items():
        if recipe["status"] != "REVIEW_CANDIDATE_NOT_EXECUTION_AUTHORIZED":
            raise RuntimeError(f"{method}: unexpected pre-review status")
        contract = recipe["scientific_contract"]
        if contract["truth_access"] != {"gcp_training": False, "lidar_training": False}:
            raise RuntimeError(f"{method}: truth access is not denied")
        if contract["result_driven_tuning"] != "FORBIDDEN":
            raise RuntimeError(f"{method}: result-driven tuning was enabled")
        if recipe["retry_policy"]["metric_based_attempt_selection"] != "FORBIDDEN":
            raise RuntimeError(f"{method}: metric-based retry selection was enabled")
        promotion = recipe["promotion_policy"]
        if promotion["requires_retraining"] is not False:
            raise RuntimeError(f"{method}: qualification promotion would wastefully retrain")
        if promotion["metric_based_promotion"] != "FORBIDDEN":
            raise RuntimeError(f"{method}: metric-based promotion was enabled")
        if recipe["historical_attempts"]["delete"] is not False:
            raise RuntimeError(f"{method}: old diagnostic deletion was authorized")
        if method == "3dgs_original":
            if recipe["training"] is not None or recipe["reuse_model"]["retrain_allowed"]:
                raise RuntimeError("3DGS reuse route unexpectedly retrains")
            continue
        training = recipe["training"]
        if not training["attempt_root_template"].startswith(
            "/root/autodl-tmp/runs/m3m-gcp-native-quarter/qualification-100k-v1/"
        ):
            raise RuntimeError(f"{method}: output escaped the qualification namespace")
        for key, value in COMMON_ENV.items():
            if training["environment"].get(key) != value:
                raise RuntimeError(f"{method}: missing common environment {key}")
        if training["resource_probe"]["enforce_contract_gates"] is not False:
            raise RuntimeError(f"{method}: warning telemetry became a rigid resource gate")
        if training["resource_probe"]["time_binary"] != GNU_TIME:
            raise RuntimeError(f"{method}: frozen GNU time path mismatch")
        binding = recipe.get("prepared_method_input_binding", {})
        if binding.get("evidence_sha256") != METHOD_INPUT_EVIDENCE_SHA:
            raise RuntimeError(f"{method}: per-method input evidence binding missing")
        if binding.get("all_image_sfm_precedes_train_test_split") is not True:
            raise RuntimeError(f"{method}: all-image SfM lineage is not bound")
        for relative, expected in recipe.get(
            "benchmark_required_files_sha256", {}
        ).items():
            path = ROOT / relative
            if not path.is_file() or sha256(path) != expected:
                raise RuntimeError(f"{method}: input-validator dependency mismatch")

    if recipes["pgsr"]["training"]["environment"]["PYTHONPATH"] != (
        "{repo}/compat/pgsr/pytorch3d_transforms_minimal_v1"
    ):
        raise RuntimeError("PGSR compatibility import path mismatch")
    if recipes["gsprior"]["training"]["environment"]["PYTHONPATH"] != (
        "{repo}/compat/gsprior/pytorch3d_transforms_minimal_v1:{repo}/code/gcp"
    ):
        raise RuntimeError("GSPrior compatibility import path mismatch")
    for method in ("2dgs", "rade_gs"):
        command = recipes[method]["training"]["command"]
        if option_values(command, "--test_iterations") != ["30001"]:
            raise RuntimeError(f"{method}: training-time evaluation was not deferred")
        if option_values(command, "--save_iterations") != ["7000", "30000"]:
            raise RuntimeError(f"{method}: save schedule changed")
    city = recipes["citygaussian_v2"]["training"]["command"]
    if (
        "--sequential_blocks" not in city
        or "--resume_from" not in city
        or "--resume_manifest" not in city
    ):
        raise RuntimeError("CityGaussianV2 sequential resume route missing")
    if "--defer_evaluation" not in recipes["citygs_x"]["training"]["command"]:
        raise RuntimeError("CityGS-X save-before-evaluate route missing")
    metro = recipes["metrogs"]["training"]["command"]
    if "--formal_input_manifest" not in metro:
        raise RuntimeError("MetroGS authoritative manifest binding missing")
    if not recipes["qgs"]["training"]["materializations"]:
        raise RuntimeError("QGS resolved training configuration is not materialized")
    expected_profiles = {
        "citygaussian_v2": "city_train_records_with_full_all_image_sfm_points",
        "citygs_x": "city_train_records_with_full_all_image_sfm_points",
        "metrogs": "metrogs_reciprocal_train_track_closure_after_all_image_sfm",
    }
    for method, expected in expected_profiles.items():
        if recipes[method]["prepared_method_input_binding"]["input_profile"] != expected:
            raise RuntimeError(f"{method}: prepared input profile mismatch")
    if recipes["gsprior"]["prepared_method_input_binding"]["input_profile"] != (
        "exact_formal_train_view_from_shared_all_image_sfm"
    ):
        raise RuntimeError("GSPrior normalization source profile mismatch")
    if "prior" not in recipes["gsprior"]["phase_commands"]:
        raise RuntimeError("GSPrior normalization lineage command missing")

    result = {
        "status": "PASS",
        "schema": "m3m_gcp_100k_qualification_validation_v1",
        "plan": str(PLAN),
        "plan_sha256": sha256(PLAN),
        "method_count": len(recipes),
        "per_method_audit": False,
        "scientific_red_line_gate": True,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
