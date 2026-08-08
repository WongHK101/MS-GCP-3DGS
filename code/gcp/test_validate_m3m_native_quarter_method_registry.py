from __future__ import annotations

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


def test_3dgs_is_frozen_but_not_unlocked() -> None:
    value = json.loads(REGISTRY.read_text(encoding="utf-8"))
    three_dgs = next(method for method in value["methods"] if method["method_id"] == "3dgs_original")
    assert three_dgs["recipe_status"] == "FROZEN_TRAINING_LOCKED_PENDING_GPU_ADAPTER_PREFLIGHT"
    assert three_dgs["common_adapter"]["status"].endswith("GPU_BUILD_AND_REAL_3K_PACKET_PENDING")
    assert three_dgs["three_k_training_allowed"] is False
    assert value["coverage_and_ranking_contract"]["minimum_oblique_azimuth_bin_circular_separation"] == 2
