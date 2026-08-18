from __future__ import annotations

import json
from pathlib import Path

from check_m3m_native_quarter_batch_launch_v3 import check_launch


REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY = REPO_ROOT / "configs" / "m3m_gcp_native_quarter_method_registry_v3.json"


def test_exact_rade_gs_one_use_gate_authorizes_only_the_frozen_run() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    gate = json.loads(
        (REPO_ROOT / registry["current_one_use_launch_gate"]["path"]).read_text(encoding="utf-8")
    )
    result = check_launch(
        registry,
        REPO_ROOT,
        method_id="rade_gs",
        scene="gcp_3000_20260602",
        seed=0,
        budget_value=30000,
        run_root=gate["run_root"],
        run_root_exists=False,
    )
    assert result["passed"] is True
    assert result["status"] == "AUTHORIZED"
    assert result["errors"] == []


def test_wrong_run_root_seed_and_budget_fail_closed() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    result = check_launch(
        registry,
        REPO_ROOT,
        method_id="rade_gs",
        scene="gcp_3000_20260602",
        seed=1,
        budget_value=29999,
        run_root="/root/autodl-tmp/runs/m3m-gcp-native-quarter/rade_gs/gcp_3000_20260602/not-authorized",
        run_root_exists=False,
    )
    assert result["passed"] is False
    assert "gate seed mismatch" in result["errors"]
    assert "gate budget mismatch" in result["errors"]
    assert "gate run root mismatch" in result["errors"]


def test_non_allowlisted_method_is_denied() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    result = check_launch(
        registry,
        REPO_ROOT,
        method_id="qgs",
        scene="gcp_3000_20260602",
        seed=0,
        budget_value=30000,
        run_root="/root/autodl-tmp/runs/m3m-gcp-native-quarter/qgs/gcp_3000_20260602/not-created",
        run_root_exists=False,
    )
    assert result["passed"] is False
    assert "method allowlist is not exact" in result["errors"]
    assert "method is not technically qualified" in result["errors"]


def test_existing_run_root_is_always_denied() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    gate = json.loads(
        (REPO_ROOT / registry["current_one_use_launch_gate"]["path"]).read_text(encoding="utf-8")
    )
    result = check_launch(
        registry,
        REPO_ROOT,
        method_id="rade_gs",
        scene="gcp_3000_20260602",
        seed=0,
        budget_value=30000,
        run_root=gate["run_root"],
        run_root_exists=True,
    )
    assert result["passed"] is False
    assert "run root already exists; overwrite and resume are forbidden" in result["errors"]
