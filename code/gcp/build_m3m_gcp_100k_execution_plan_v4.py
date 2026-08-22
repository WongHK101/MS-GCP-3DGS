#!/usr/bin/env python3
"""Build the activation-v4 100K successor plan without changing method recipes."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from m3m_gcp_lidar_artifacts import canonical_sha256, sha256_file


ROOT = Path(__file__).resolve().parents[2]
PREVIOUS_PLAN = (
    ROOT / "configs/m3m_gcp_native_quarter_100k_ten_method_execution_plan_v3.json"
)
RECIPE_MANIFEST = (
    ROOT / "configs/m3m_gcp_native_quarter_100k_recipe_manifest_v3.json"
)
CONTINUITY_RECEIPT = (
    ROOT
    / "docs/protocol_evidence/m3m_gcp_100k_activation_v3_to_v4_continuity.json"
)
OUTPUT = (
    ROOT / "configs/m3m_gcp_native_quarter_100k_ten_method_execution_plan_v4.json"
)


def file_binding(relative_path: str) -> dict[str, Any]:
    path = ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": relative_path, "sha256": sha256_file(path)}


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError("refusing to overwrite execution plan v4")
    previous = json.loads(PREVIOUS_PLAN.read_text(encoding="utf-8"))
    recipes = json.loads(RECIPE_MANIFEST.read_text(encoding="utf-8"))
    receipt = json.loads(CONTINUITY_RECEIPT.read_text(encoding="utf-8"))
    if (
        previous.get("schema")
        != "m3m_gcp_native_quarter_100k_ten_method_execution_plan_v3"
        or previous.get("canonical_sha256") != canonical_sha256(previous)
        or recipes.get("schema")
        != "m3m_gcp_native_quarter_100k_recipe_manifest_v3"
        or recipes.get("canonical_sha256") != canonical_sha256(recipes)
        or receipt.get("schema") != "m3m_gcp_100k_activation_continuity_v2"
        or receipt.get("status") != "SEALED_V3_TO_V4_GUARD_CONTINUITY"
        or receipt.get("canonical_sha256") != canonical_sha256(receipt)
    ):
        raise RuntimeError("v4 predecessor artifacts are not exact")

    plan = copy.deepcopy(previous)
    plan["schema"] = "m3m_gcp_native_quarter_100k_ten_method_execution_plan_v4"
    plan["plan_id"] = "m3m-gcp-native-quarter-100k-ten-method-seed0-v4-continuation"
    plan["activation_manifest_path"] = (
        "/root/autodl-tmp/runs/m3m-gcp-native-quarter/formal-100k-v2/activation_v4.json"
    )
    plan["activation_continuity"] = {
        "continued_run_namespace": (
            "/root/autodl-tmp/runs/m3m-gcp-native-quarter/formal-100k-v2"
        ),
        "final_attempt_freeze_authorization": "activation_v4_only",
        "inherited_failed_methods_forbidden_to_launch": sorted(
            row["method_id"] for row in receipt["inherited_terminal_outcomes"]
        ),
        "3dgs_retraining_forbidden": True,
        "metrogs_prior_rerun_forbidden": True,
        "metrogs_training_is_only_unfinished_attempt": True,
        "previous_execution_plan": {
            "path": PREVIOUS_PLAN.relative_to(ROOT).as_posix(),
            "bytes": PREVIOUS_PLAN.stat().st_size,
            "sha256": sha256_file(PREVIOUS_PLAN),
            "canonical_sha256": previous["canonical_sha256"],
        },
        "previous_recipe_manifest_bytes_unchanged": True,
        "remote_artifacts_must_remain_byte_identical": True,
        "receipt": {
            "path": CONTINUITY_RECEIPT.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(CONTINUITY_RECEIPT),
        },
        "status_required": "SEALED_V3_TO_V4_GUARD_CONTINUITY",
    }

    freeze = plan["attempt_freeze"]
    freeze.update(
        {
            "attempt_manifest_path": (
                "/root/autodl-tmp/runs/m3m-gcp-native-quarter/formal-100k-v2/scene_attempts_v4.json"
            ),
            "execution_plan_path": (
                "configs/m3m_gcp_native_quarter_100k_ten_method_execution_plan_v4.json"
            ),
            "model_identity_root": (
                "/root/autodl-tmp/runs/m3m-gcp-native-quarter/formal-100k-v2/model-identities-v4"
            ),
            "scene_attempt_freeze_path": (
                "/root/autodl-tmp/runs/m3m-gcp-native-quarter/formal-100k-v2/scene_attempt_freeze_v4.json"
            ),
        }
    )

    closure = plan["execution_closure"]
    closure["activation_builder"] = file_binding(
        "code/gcp/build_m3m_gcp_lidar_100k_activation_v4.py"
    )
    closure["activation_continuity_validator"] = file_binding(
        "code/gcp/m3m_gcp_100k_activation_v4_continuity.py"
    )
    closure["activation_continuity_builder"] = file_binding(
        "code/gcp/build_m3m_gcp_100k_activation_v4_continuity.py"
    )
    closure["execution_plan_v4_builder"] = file_binding(
        "code/gcp/build_m3m_gcp_100k_execution_plan_v4.py"
    )
    closure["guarded_runner"] = file_binding(
        "code/gcp/run_m3m_gcp_100k_guarded.py"
    )
    closure["attempt_manifest_builder"] = file_binding(
        "code/gcp/build_m3m_gcp_100k_attempt_manifest.py"
    )
    closure["attempt_freezer"] = file_binding(
        "code/gcp/freeze_m3m_gcp_lidar_scene_attempts.py"
    )
    closure[
        "prior_phase_context_reconstructed_from_frozen_phase_bindings"
    ] = True

    plan["retry_policy"]["activation_v3_to_v4_continuity"] = (
        "3DGS READY and eight terminal failures are inherited; MetroGS prior PASS is "
        "inherited without rerun; the pre-child training rejection consumed no attempt; "
        "activation_v4 authorizes only the unfinished MetroGS training attempt before freeze"
    )
    plan["retry_policy"].pop("activation_v2_to_v3_continuity", None)
    plan["successor_scope"] = {
        "previous_activation": (
            "/root/autodl-tmp/runs/m3m-gcp-native-quarter/formal-100k-v2/activation_v3.json"
        ),
        "previous_execution_plan": PREVIOUS_PLAN.relative_to(ROOT).as_posix(),
        "recipe_manifest_changed": False,
        "method_order_budget_dataset_prior_or_training_command_changed": False,
        "guard_fix": (
            "training rehashes prior command with source_root and phase roots from the "
            "frozen prior bindings instead of the active training source root"
        ),
        "pre_child_rejection_consumed_attempt": False,
        "prior_regeneration_allowed": False,
    }
    plan["canonical_sha256"] = canonical_sha256(plan)
    OUTPUT.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
