#!/usr/bin/env python3
"""Static regressions for the single-scope activation-v4 guard successor."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

from m3m_gcp_lidar_artifacts import canonical_sha256, sha256_file


ROOT = Path(__file__).resolve().parents[2]
PLAN_V3 = ROOT / "configs/m3m_gcp_native_quarter_100k_ten_method_execution_plan_v3.json"
PLAN_V4 = ROOT / "configs/m3m_gcp_native_quarter_100k_ten_method_execution_plan_v4.json"
RECIPES = ROOT / "configs/m3m_gcp_native_quarter_100k_recipe_manifest_v3.json"
RECEIPT = ROOT / "docs/protocol_evidence/m3m_gcp_100k_activation_v3_to_v4_continuity.json"
POINTER = ROOT / "configs/m3m_gcp_native_quarter_current.json"
PRIOR_METHODS = {"gsprior", "citygaussian_v2", "citygs_x", "metrogs"}
FAILED_OUTCOMES = {
    "2dgs": "FAILED_UNRANKED",
    "pgsr": "FAILED_UNRANKED",
    "rade_gs": "FAILED_UNRANKED",
    "qgs": "OOM_UNRANKED",
    "gsprior": "FAILED_UNRANKED",
    "sof": "FAILED_UNRANKED",
    "citygaussian_v2": "FAILED_UNRANKED",
    "citygs_x": "FAILED_UNRANKED",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_v4_plan_is_a_single_scope_successor() -> None:
    old = load(PLAN_V3)
    new = load(PLAN_V4)
    assert new["schema"] == "m3m_gcp_native_quarter_100k_ten_method_execution_plan_v4"
    assert new["canonical_sha256"] == canonical_sha256(new)
    assert new["method_order"] == old["method_order"]
    assert new["budgets"] == old["budgets"]
    assert new["recipe_manifest"] == old["recipe_manifest"]
    assert new["truth_deny"] == old["truth_deny"]
    assert new["storage"] == old["storage"]
    assert new["successor_scope"] == {
        "previous_activation": (
            "/root/autodl-tmp/runs/m3m-gcp-native-quarter/formal-100k-v2/activation_v3.json"
        ),
        "previous_execution_plan": (
            "configs/m3m_gcp_native_quarter_100k_ten_method_execution_plan_v3.json"
        ),
        "recipe_manifest_changed": False,
        "method_order_budget_dataset_prior_or_training_command_changed": False,
        "guard_fix": (
            "training rehashes prior command with source_root and phase roots from the "
            "frozen prior bindings instead of the active training source root"
        ),
        "pre_child_rejection_consumed_attempt": False,
        "prior_regeneration_allowed": False,
    }
    assert new["activation_manifest_path"].endswith("/activation_v4.json")
    assert new["attempt_freeze"]["attempt_manifest_path"].endswith(
        "/scene_attempts_v4.json"
    )
    assert new["attempt_freeze"]["scene_attempt_freeze_path"].endswith(
        "/scene_attempt_freeze_v4.json"
    )


def test_v4_closure_hashes_every_changed_executable() -> None:
    plan = load(PLAN_V4)
    closure = plan["execution_closure"]
    expected = {
        "activation_builder": "code/gcp/build_m3m_gcp_lidar_100k_activation_v4.py",
        "activation_continuity_validator": (
            "code/gcp/m3m_gcp_100k_activation_v4_continuity.py"
        ),
        "activation_continuity_builder": (
            "code/gcp/build_m3m_gcp_100k_activation_v4_continuity.py"
        ),
        "execution_plan_v4_builder": "code/gcp/build_m3m_gcp_100k_execution_plan_v4.py",
        "guarded_runner": "code/gcp/run_m3m_gcp_100k_guarded.py",
        "attempt_manifest_builder": "code/gcp/build_m3m_gcp_100k_attempt_manifest.py",
        "attempt_freezer": "code/gcp/freeze_m3m_gcp_lidar_scene_attempts.py",
    }
    for label, relative in expected.items():
        row = closure[label]
        path = ROOT / relative
        assert row["path"] == relative
        assert row["sha256"] == sha256_file(path)
    assert closure["prior_phase_context_reconstructed_from_frozen_phase_bindings"] is True


def test_continuity_receipt_inherits_every_finished_state_once() -> None:
    receipt = load(RECEIPT)
    assert receipt["schema"] == "m3m_gcp_100k_activation_continuity_v2"
    assert receipt["status"] == "SEALED_V3_TO_V4_GUARD_CONTINUITY"
    assert receipt["canonical_sha256"] == canonical_sha256(receipt)
    assert receipt["inherited_ready_model"]["method_id"] == "3dgs_original"
    assert receipt["inherited_ready_model"]["retrain_allowed"] is False
    outcomes = {
        row["method_id"]: row["formal_status"]
        for row in receipt["inherited_terminal_outcomes"]
    }
    assert outcomes == FAILED_OUTCOMES
    assert all(
        row["formal_attempt_consumed"] is True and row["retry_allowed"] is False
        for row in receipt["inherited_terminal_outcomes"]
    )
    assert receipt["metrogs_prior"]["status"] == "PASS"
    assert receipt["metrogs_prior"]["rerun_allowed"] is False
    rejection = receipt["metrogs_training_prechild_rejection"]
    assert rejection["child_started"] is False
    assert rejection["run_root_created"] is False
    assert rejection["formal_attempt_consumed"] is False
    assert rejection["retry_allowed_only_after_guard_fix_and_successor_review"] is True


def test_all_prior_methods_have_one_exact_reconstructible_context() -> None:
    manifest = load(RECIPES)
    seen = set()
    for row in manifest["recipes"]:
        recipe = load(ROOT / row["path"])
        method_id = recipe["method_id"]
        if "prior" not in recipe.get("phase_commands", {}):
            continue
        seen.add(method_id)
        prior_source = recipe["source_bindings"]["prior"]
        prior_roots = recipe["phase_roots"]["prior"]
        training_roots = recipe["phase_roots"]["training"]
        assert PurePosixPath(prior_source["root"]).is_absolute()
        assert PurePosixPath(prior_roots["dataset_root"]).is_absolute()
        assert PurePosixPath(prior_roots["prior_root"]).is_absolute()
        assert PurePosixPath(training_roots["dataset_root"]).is_absolute()
        assert PurePosixPath(training_roots["prior_root"]).is_absolute()
        replacements = {
            "repo": str(ROOT),
            "source_root": prior_source["root"],
            "dataset_root": prior_roots["dataset_root"],
            "prior_root": prior_roots["prior_root"],
            "run_root": recipe["authorized_run_root"],
            "packet_set_root": recipe["authorized_packet_set_root"],
        }
        rebuilt = [value.format(**replacements) for value in recipe["phase_commands"]["prior"]]
        assert rebuilt
        assert all("{" not in value and "}" not in value for value in rebuilt)
    assert seen == PRIOR_METHODS
    metro = next(
        load(ROOT / row["path"])
        for row in manifest["recipes"]
        if row["method_id"] == "metrogs"
    )
    assert (
        metro["source_bindings"]["prior"]["root"]
        != metro["source_bindings"]["training"]["root"]
    )


def test_current_pointer_selects_only_v4_as_active_100k_plan() -> None:
    pointer = load(POINTER)
    lidar = pointer["lidar_track"]
    assert lidar["one_hundred_k_execution_plan"].endswith(
        "m3m_gcp_native_quarter_100k_ten_method_execution_plan_v4.json"
    )
    continuity = lidar["one_hundred_k_activation_continuity"]
    assert continuity["previous_activation"].endswith("/activation_v3.json")
    assert continuity["next_activation"].endswith("/activation_v4.json")
    assert continuity["receipt"].endswith(
        "m3m_gcp_100k_activation_v3_to_v4_continuity.json"
    )
