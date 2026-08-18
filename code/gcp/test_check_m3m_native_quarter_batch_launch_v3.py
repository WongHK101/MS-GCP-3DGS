from __future__ import annotations

import json
from pathlib import Path

from check_m3m_native_quarter_batch_launch_v3 import check_launch


REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY = REPO_ROOT / "configs" / "m3m_gcp_native_quarter_method_registry_v3.json"


def test_batch_registry_fails_closed_before_method_gate() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    result = check_launch(
        registry,
        REPO_ROOT,
        method_id="pgsr",
        scene="gcp_3000_20260602",
        seed=0,
        budget_value=30000,
        run_root="/root/autodl-tmp/runs/m3m-gcp-native-quarter/pgsr/gcp_3000_20260602/not-created",
        run_root_exists=False,
    )
    assert result["passed"] is False
    assert result["status"] == "DENIED"
    assert "method allowlist is not exact" in result["errors"]
    assert "current one-use launch gate is absent" in result["errors"]


def test_existing_run_root_is_always_denied() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    result = check_launch(
        registry,
        REPO_ROOT,
        method_id="pgsr",
        scene="gcp_3000_20260602",
        seed=0,
        budget_value=30000,
        run_root="/root/autodl-tmp/runs/m3m-gcp-native-quarter/pgsr/gcp_3000_20260602/existing",
        run_root_exists=True,
    )
    assert result["passed"] is False
    assert "run root already exists; overwrite and resume are forbidden" in result["errors"]
