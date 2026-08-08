from __future__ import annotations

import copy
import json
from pathlib import Path

from validate_m3m_native_quarter_3dgs_recipe import validate_recipe


REPO_ROOT = Path(__file__).resolve().parents[2]
RECIPE = REPO_ROOT / "configs" / "m3m_gcp_native_quarter_3dgs_3k_recipe_v1.json"


def load_recipe() -> dict:
    return json.loads(RECIPE.read_text(encoding="utf-8"))


def test_native_quarter_recipe_passes_and_remains_locked() -> None:
    result = validate_recipe(load_recipe(), REPO_ROOT)
    assert result["passed"], result["errors"]
    assert result["training_allowed"] is False
    assert result["gpu_renderer_build_preflight_passed"] is False
    assert result["frozen_3k_real_packet_camera_preflight_passed"] is False


def test_legacy_clean_r4_command_is_rejected() -> None:
    value = copy.deepcopy(load_recipe())
    value["execution"]["command_template"] = value["execution"]["command_template"].replace(
        "<native_quarter_release>/formal_inputs/gcp_3000_20260602/train",
        "/root/r4_clean_v1/gcp_3000_20260602/train",
    )
    result = validate_recipe(value, REPO_ROOT)
    assert not result["passed"]
    assert any("clean-R4" in error for error in result["errors"])


def test_premature_gpu_unlock_is_rejected() -> None:
    value = copy.deepcopy(load_recipe())
    value["execution"]["training_authorized"] = True
    value["evaluation_adapter"]["gpu_build_and_real_packet_camera_preflight_passed"] = True
    value["qualification"]["three_k_training_allowed"] = True
    result = validate_recipe(value, REPO_ROOT)
    assert not result["passed"]
    assert any("training must remain locked" in error for error in result["errors"])
    assert any("prematurely" in error for error in result["errors"])


def test_training_hyperparameter_drift_is_rejected() -> None:
    value = copy.deepcopy(load_recipe())
    value["training"]["densify_until_iter"] = 14999
    result = validate_recipe(value, REPO_ROOT)
    assert not result["passed"]
    assert "training densify_until_iter mismatch" in result["errors"]
