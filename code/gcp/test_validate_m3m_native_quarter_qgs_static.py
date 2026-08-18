from __future__ import annotations

import json
from pathlib import Path

from validate_m3m_native_quarter_qgs_static import validate


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_frozen_qgs_recipe_adapter_qualification_and_gate_pass() -> None:
    result = validate(REPO_ROOT)
    assert result["passed"] is True
    assert result["status"] == "PASS"
    assert result["method_id"] == "qgs"
    assert result["errors"] == []


def test_qgs_gate_is_single_seed_single_budget_and_no_resume() -> None:
    gate = json.loads(
        (REPO_ROOT / "configs/launch_gates/m3m_gcp_native_quarter_qgs_3k_seed0_30k_gate_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert gate["seed"] == 0
    assert gate["official_budget"] == {"type": "iterations", "value": 30000}
    assert gate["single_fresh_run_allowed"] is True
    assert gate["resume_allowed"] is False
    assert gate["overwrite_allowed"] is False
    assert gate["result_driven_retry_allowed"] is False
