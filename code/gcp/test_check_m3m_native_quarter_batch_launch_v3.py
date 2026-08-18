from __future__ import annotations

import json
from pathlib import Path

from check_m3m_native_quarter_batch_launch_v3 import check_launch


REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY = REPO_ROOT / "configs" / "m3m_gcp_native_quarter_method_registry_v3.json"


QGS_GATE_PATH = "configs/launch_gates/m3m_gcp_native_quarter_qgs_3k_seed0_30k_gate_v1.json"
QGS_GATE_SHA256 = "e882d98d5203128e081cf1849499ef5bef49d1213b84a635815c2d9ffc0d08b5"


def load_qgs_gate_fixture() -> tuple[dict, dict]:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    method = next(item for item in registry["methods"] if item["method_id"] == "qgs")
    method["lifecycle_role"] = "ACTIVE_3K_CANDIDATE"
    method["technical_qualification_status"] = "TECHNICALLY_QUALIFIED"
    method["formal_3k_result"] = {"status": "NOT_ATTEMPTED"}
    method["three_k_training_allowed"] = True
    registry["per_method_training_allowed_methods"] = ["qgs"]
    registry["current_one_use_launch_gate"] = {
        "method_id": "qgs",
        "path": QGS_GATE_PATH,
        "sha256": QGS_GATE_SHA256,
    }
    gate = json.loads((REPO_ROOT / QGS_GATE_PATH).read_text(encoding="utf-8"))
    return registry, gate


def test_current_sof_one_use_gate_authorizes_exact_frozen_run() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    gate_ref = registry["current_one_use_launch_gate"]
    assert gate_ref["method_id"] == "sof"
    gate = json.loads((REPO_ROOT / gate_ref["path"]).read_text(encoding="utf-8"))
    result = check_launch(
        registry,
        REPO_ROOT,
        method_id="sof",
        scene="gcp_3000_20260602",
        seed=0,
        budget_value=30000,
        run_root=gate["run_root"],
        run_root_exists=False,
    )
    assert result["passed"] is True
    assert result["status"] == "AUTHORIZED"
    assert result["errors"] == []


def test_exact_qgs_one_use_gate_authorizes_only_the_frozen_run() -> None:
    registry, gate = load_qgs_gate_fixture()
    result = check_launch(
        registry,
        REPO_ROOT,
        method_id="qgs",
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
    registry, _ = load_qgs_gate_fixture()
    result = check_launch(
        registry,
        REPO_ROOT,
        method_id="qgs",
        scene="gcp_3000_20260602",
        seed=1,
        budget_value=29999,
        run_root="/root/autodl-tmp/runs/m3m-gcp-native-quarter/qgs/gcp_3000_20260602/not-authorized",
        run_root_exists=False,
    )
    assert result["passed"] is False
    assert "gate seed mismatch" in result["errors"]
    assert "gate budget mismatch" in result["errors"]
    assert "gate run root mismatch" in result["errors"]


def test_non_allowlisted_method_is_denied() -> None:
    registry, _ = load_qgs_gate_fixture()
    result = check_launch(
        registry,
        REPO_ROOT,
        method_id="sof",
        scene="gcp_3000_20260602",
        seed=0,
        budget_value=30000,
        run_root="/root/autodl-tmp/runs/m3m-gcp-native-quarter/sof/gcp_3000_20260602/not-created",
        run_root_exists=False,
    )
    assert result["passed"] is False
    assert "method allowlist is not exact" in result["errors"]
    assert "gate reference method mismatch" in result["errors"]


def test_existing_run_root_is_always_denied() -> None:
    registry, gate = load_qgs_gate_fixture()
    result = check_launch(
        registry,
        REPO_ROOT,
        method_id="qgs",
        scene="gcp_3000_20260602",
        seed=0,
        budget_value=30000,
        run_root=gate["run_root"],
        run_root_exists=True,
    )
    assert result["passed"] is False
    assert "run root already exists; overwrite and resume are forbidden" in result["errors"]


def test_gate_cannot_substitute_adapter_or_qualification_evidence() -> None:
    registry, gate = load_qgs_gate_fixture()
    method = next(item for item in registry["methods"] if item["method_id"] == "qgs")
    method["adapter_config_sha256"] = "0" * 64
    method["qualification_report_sha256"] = "1" * 64
    result = check_launch(
        registry,
        REPO_ROOT,
        method_id="qgs",
        scene="gcp_3000_20260602",
        seed=0,
        budget_value=30000,
        run_root=gate["run_root"],
        run_root_exists=False,
    )
    assert result["passed"] is False
    assert "frozen adapter SHA mismatch" in result["errors"]
    assert "method qualification report SHA mismatch" in result["errors"]
