from __future__ import annotations

import copy
import json
from pathlib import Path

from validate_m3m_native_quarter_method_registry import EXPECTED_METHODS, validate_registry


REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY = REPO_ROOT / "configs" / "m3m_gcp_native_quarter_method_registry_v2.json"


def test_native_quarter_registry_passes() -> None:
    result = validate_registry(json.loads(REGISTRY.read_text(encoding="utf-8")), REPO_ROOT)
    assert result["passed"], result["errors"]
    assert set(result["method_ids"]) == EXPECTED_METHODS
    assert result["training_allowed_methods"] == []


def test_qgs_is_public_and_external_priors_are_explicit() -> None:
    value = json.loads(REGISTRY.read_text(encoding="utf-8"))
    by_id = {method["method_id"]: method for method in value["methods"]}
    assert by_id["qgs"]["source"]["official_repository"] == "https://github.com/will-zzy/QGS"
    assert by_id["qgs"]["source"]["commit"]
    assert by_id["citygs_x"]["external_priors"]
    assert by_id["metrogs"]["external_priors"]


def test_preliminary_scope_is_closed_to_three_representatives() -> None:
    value = json.loads(REGISTRY.read_text(encoding="utf-8"))
    scope = value["preliminary_evidence_scope"]
    assert scope["status"] == "CLOSED_THREE_FAMILIES_ONE_REPRESENTATIVE_EACH"
    assert scope["selected_method_ids"] == ["3dgs_original", "2dgs", "gof"]
    assert scope["deferred_candidate_method_ids"] == [
        "pgsr", "rade_gs", "qgs", "citygaussian_v2", "citygs_x", "metrogs"
    ]
    assert scope["candidate_expansion_status"] == "LOCKED_UNLESS_EXPLICITLY_REOPENED"
    assert scope["six_scene_matrix_status"] == "LOCKED"
    assert scope["multi_seed_status"] == "NOT_AUTHORIZED"
    assert value["per_method_training_allowed_methods"] == []
    assert value["global_training_allowed"] is False


def test_preliminary_scope_cannot_be_reopened_silently() -> None:
    value = json.loads(REGISTRY.read_text(encoding="utf-8"))
    mutated = copy.deepcopy(value)
    mutated["preliminary_evidence_scope"]["candidate_expansion_status"] = "OPEN"
    result = validate_registry(mutated, REPO_ROOT)
    assert not result["passed"]
    assert any("candidate expansion is not locked" in error for error in result["errors"])


def test_3dgs_formal_3k_result_is_complete_without_unlocking_the_matrix() -> None:
    value = json.loads(REGISTRY.read_text(encoding="utf-8"))
    three_dgs = next(method for method in value["methods"] if method["method_id"] == "3dgs_original")
    assert three_dgs["recipe_status"] == "FROZEN_3K_TRAINING_AUTHORIZED"
    assert three_dgs["common_adapter"]["status"].endswith("PREFLIGHT_PASS")
    assert three_dgs["three_k_qualification_status"] == "FORMAL_3K_COMPLETE_RANKED"
    assert three_dgs["formal_3k_result"]["status"] == "COMPLETE_RANKED"
    assert three_dgs["formal_3k_result"]["rerun_allowed"] is False
    assert three_dgs["three_k_training_allowed"] is False
    assert three_dgs["full_scene_matrix_eligible"] is False
    assert value["global_training_allowed"] is False
    assert value["per_method_training_allowed_methods"] == []
    assert value["coverage_and_ranking_contract"]["minimum_oblique_azimuth_bin_circular_separation"] == 2


def test_completed_3dgs_seed_zero_run_cannot_be_reopened_silently() -> None:
    value = json.loads(REGISTRY.read_text(encoding="utf-8"))
    mutated = copy.deepcopy(value)
    three_dgs = next(method for method in mutated["methods"] if method["method_id"] == "3dgs_original")
    three_dgs["three_k_training_allowed"] = True
    mutated["per_method_training_allowed_methods"] = ["3dgs_original"]
    result = validate_registry(mutated, REPO_ROOT)
    assert not result["passed"]
    assert any("must not remain launchable" in error for error in result["errors"])


def test_2dgs_formal_3k_result_is_complete_and_relocked() -> None:
    value = json.loads(REGISTRY.read_text(encoding="utf-8"))
    two_dgs = next(method for method in value["methods"] if method["method_id"] == "2dgs")
    assert two_dgs["recipe_status"] == "FROZEN_3K_FORMAL_COMPLETE_RELOCKED"
    assert two_dgs["common_adapter"]["status"] == "GPU_BUILD_SYNTHETIC_AND_REAL_3K_PACKET_EVALUATOR_PREFLIGHT_PASS"
    assert two_dgs["three_k_qualification_status"] == "FORMAL_3K_COMPLETE_RANKED"
    assert two_dgs["formal_3k_result"]["status"] == "COMPLETE_RANKED"
    assert two_dgs["formal_3k_result"]["rerun_allowed"] is False
    assert two_dgs["three_k_training_allowed"] is False
    assert value["per_method_training_allowed_methods"] == []
    assert value["global_training_allowed"] is False
    assert two_dgs["full_scene_matrix_eligible"] is False


def test_gof_formal_3k_result_is_complete_and_relocked() -> None:
    value = json.loads(REGISTRY.read_text(encoding="utf-8"))
    gof = next(method for method in value["methods"] if method["method_id"] == "gof")
    assert gof["recipe_status"] == "FROZEN_3K_FORMAL_COMPLETE_RELOCKED"
    assert gof["common_adapter"]["status"] == "GPU_BUILD_SYNTHETIC_AND_REAL_3K_PACKET_EVALUATOR_PREFLIGHT_PASS"
    assert gof["three_k_qualification_status"] == "FORMAL_3K_COMPLETE_RANKED"
    assert gof["formal_3k_result"]["status"] == "COMPLETE_RANKED"
    assert gof["formal_3k_result"]["rerun_allowed"] is False
    assert gof["three_k_training_allowed"] is False
    assert gof["full_scene_matrix_eligible"] is False
    assert value["per_method_training_allowed_methods"] == []
    assert value["global_training_allowed"] is False


def test_completed_gof_seed_zero_run_cannot_be_reopened_silently() -> None:
    value = json.loads(REGISTRY.read_text(encoding="utf-8"))
    mutated = copy.deepcopy(value)
    gof = next(method for method in mutated["methods"] if method["method_id"] == "gof")
    gof["three_k_training_allowed"] = True
    mutated["per_method_training_allowed_methods"] = ["gof"]
    result = validate_registry(mutated, REPO_ROOT)
    assert not result["passed"]
    assert any("must not remain launchable" in error for error in result["errors"])


def test_gof_formal_checkpoint_metric_cannot_drift_from_report() -> None:
    value = json.loads(REGISTRY.read_text(encoding="utf-8"))
    mutated = copy.deepcopy(value)
    gof = next(method for method in mutated["methods"] if method["method_id"] == "gof")
    gof["formal_3k_result"]["checkpoint_rmse_3d_m"] += 0.001
    result = validate_registry(mutated, REPO_ROOT)
    assert not result["passed"]
    assert any("GOF checkpoint RMSE-3D mismatch" in error for error in result["errors"])
