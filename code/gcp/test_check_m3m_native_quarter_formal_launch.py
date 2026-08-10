from __future__ import annotations

import json
from pathlib import Path

from check_m3m_native_quarter_formal_launch import check_launch


REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY = REPO_ROOT / "configs" / "m3m_gcp_native_quarter_method_registry_v2.json"


def current_registry() -> dict:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def check(method_id: str) -> dict:
    return check_launch(
        current_registry(),
        REPO_ROOT,
        method_id=method_id,
        scene="gcp_3000_20260602",
        seed=0,
        iterations=30000,
        run_root=f"/root/autodl-tmp/runs/m3m-gcp-native-quarter/{method_id}/gcp_3000_20260602/test-run",
        run_root_exists=False,
    )


def test_completed_3dgs_is_denied() -> None:
    result = check("3dgs_original")
    assert result["passed"] is False
    assert any("forbids rerun" in error for error in result["errors"])


def test_completed_2dgs_exact_run_is_denied() -> None:
    result = check("2dgs")
    assert result["passed"] is False
    assert any("forbids rerun" in error for error in result["errors"])


def test_completed_gof_exact_run_is_denied() -> None:
    result = check("gof")
    assert result["passed"] is False
    assert any("forbids rerun" in error for error in result["errors"])


def test_completed_gof_existing_run_root_is_denied() -> None:
    result = check_launch(
        current_registry(),
        REPO_ROOT,
        method_id="gof",
        scene="gcp_3000_20260602",
        seed=0,
        iterations=30000,
        run_root="/root/autodl-tmp/runs/m3m-gcp-native-quarter/gof/gcp_3000_20260602/existing-run",
        run_root_exists=True,
    )
    assert result["passed"] is False
    assert any("forbids rerun" in error for error in result["errors"])
    assert any("already exists" in error for error in result["errors"])


def test_qualified_2dgs_wrong_seed_is_denied() -> None:
    result = check_launch(
        current_registry(),
        REPO_ROOT,
        method_id="2dgs",
        scene="gcp_3000_20260602",
        seed=1,
        iterations=30000,
        run_root="/root/autodl-tmp/runs/m3m-gcp-native-quarter/2dgs/gcp_3000_20260602/wrong-seed",
        run_root_exists=False,
    )
    assert result["passed"] is False
    assert any("seed differs" in error for error in result["errors"])


def test_unknown_method_is_denied() -> None:
    result = check("unknown")
    assert result["passed"] is False
    assert any("unknown method" in error for error in result["errors"])
