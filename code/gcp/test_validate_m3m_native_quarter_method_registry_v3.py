from __future__ import annotations

import json
from pathlib import Path

from validate_m3m_native_quarter_method_registry_v3 import validate_registry


REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY = REPO_ROOT / "configs" / "m3m_gcp_native_quarter_method_registry_v3.json"


def load_registry() -> dict:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def validate_mutation(tmp_path: Path, value: dict) -> dict:
    path = tmp_path / "mutated_registry.json"
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return validate_registry(REPO_ROOT, path)


def test_current_registry_passes_with_only_the_qgs_one_use_gate() -> None:
    result = validate_registry(REPO_ROOT, REGISTRY)
    assert result["passed"] is True
    assert result["status"] == "PASS"
    assert result["method_count"] == 11
    assert result["active_method_count"] == 10
    assert result["candidate_method_count"] == 8
    assert result["training_allowed_methods"] == ["qgs"]


def test_global_training_unlock_fails_closed(tmp_path: Path) -> None:
    value = load_registry()
    value["global_training_allowed"] = True
    result = validate_mutation(tmp_path, value)
    assert result["passed"] is False
    assert "global training lock missing" in result["errors"]


def test_consumed_gate_can_be_closed_while_no_method_is_launchable(tmp_path: Path) -> None:
    value = load_registry()
    value["current_one_use_launch_gate"] = None
    value["per_method_training_allowed_methods"] = []
    value["status"] = "EIGHT_METHOD_3K_BATCH_ACTIVE"
    method = next(item for item in value["methods"] if item["method_id"] == "qgs")
    method["three_k_training_allowed"] = False
    result = validate_mutation(tmp_path, value)
    assert result["passed"] is True
    assert result["training_allowed_methods"] == []


def test_training_flag_without_gate_fails_closed(tmp_path: Path) -> None:
    value = load_registry()
    value["current_one_use_launch_gate"] = None
    value["per_method_training_allowed_methods"] = []
    value["status"] = "EIGHT_METHOD_3K_BATCH_ACTIVE"
    result = validate_mutation(tmp_path, value)
    assert result["passed"] is False
    assert "method training flags disagree with the allowlist" in result["errors"]


def test_one_use_gate_hash_is_bound(tmp_path: Path) -> None:
    value = load_registry()
    value["current_one_use_launch_gate"]["sha256"] = "0" * 64
    result = validate_mutation(tmp_path, value)
    assert result["passed"] is False
    assert "one-use gate file SHA mismatch" in result["errors"]


def test_execution_plan_hash_is_bound(tmp_path: Path) -> None:
    value = load_registry()
    value["execution_plan"]["sha256"] = "0" * 64
    result = validate_mutation(tmp_path, value)
    assert result["passed"] is False
    assert "execution plan SHA mismatch" in result["errors"]


def test_retired_gof_cannot_reenter_active_pool(tmp_path: Path) -> None:
    value = load_registry()
    value["active_benchmark_method_ids"][-1] = "gof"
    result = validate_mutation(tmp_path, value)
    assert result["passed"] is False
    assert "retired GOF leaked into active pool" in result["errors"]


def test_missing_license_methods_remain_internal_only(tmp_path: Path) -> None:
    value = load_registry()
    method = next(item for item in value["methods"] if item["method_id"] == "gsprior")
    method["source"]["license_status"] = "present"
    result = validate_mutation(tmp_path, value)
    assert result["passed"] is False
    assert "gsprior: missing-license boundary absent" in result["errors"]


def test_citygaussian_cannot_be_misreported_as_rgb_only(tmp_path: Path) -> None:
    value = load_registry()
    method = next(item for item in value["methods"] if item["method_id"] == "citygaussian_v2")
    method["input_class"] = "rgb_colmap_only"
    result = validate_mutation(tmp_path, value)
    assert result["passed"] is False
    assert "CityGaussianV2 input stratum mismatch" in result["errors"]


def test_six_scene_matrix_cannot_open_during_3k_batch(tmp_path: Path) -> None:
    value = load_registry()
    value["batch_execution_scope"]["six_scene_matrix_status"] = "OPEN"
    result = validate_mutation(tmp_path, value)
    assert result["passed"] is False
    assert "six-scene matrix unlocked" in result["errors"]


def test_qualified_candidate_report_hash_is_bound(tmp_path: Path) -> None:
    value = load_registry()
    method = next(item for item in value["methods"] if item["method_id"] == "qgs")
    method["qualification_report_sha256"] = "0" * 64
    result = validate_mutation(tmp_path, value)
    assert result["passed"] is False
    assert "qgs: qualification report file SHA mismatch" in result["errors"]
