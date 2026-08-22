#!/usr/bin/env python3
"""Seal activation-v3 terminal outcomes and the MetroGS prior for activation-v4."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from m3m_gcp_lidar_artifacts import canonical_sha256, sha256_file


SCENE = "gcp_100000_20260610"
PREVIOUS_COMMIT = "e33368db9333f826a3e808ff00c437c1a6c63b82"
PREVIOUS_TREE = "4620a434bd081af9274fdfc37dbb0d673636edfc"
PREVIOUS_PLAN = "configs/m3m_gcp_native_quarter_100k_ten_method_execution_plan_v3.json"
PREVIOUS_RECIPES = "configs/m3m_gcp_native_quarter_100k_recipe_manifest_v3.json"
PREVIOUS_NOTE = "docs/M3M_GCP_100K_TEN_METHOD_TIME_SPACE_EXECUTION_PLAN_V3.md"
RUN_NAMESPACE = Path(
    "/root/autodl-tmp/runs/m3m-gcp-native-quarter/formal-100k-v2"
)
SCENE_ROOT = RUN_NAMESPACE / SCENE
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


def git_value(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, encoding="utf-8"
    ).strip()


def file_row(role: str, path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"continuity artifact is not a regular file: {path}")
    value = path.relative_to(relative_to).as_posix() if relative_to else str(path)
    row: dict[str, Any] = {
        "role": role,
        "path": value,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        canonical = payload.get("canonical_sha256")
        if canonical is not None:
            if canonical != canonical_sha256(payload):
                raise RuntimeError(f"canonical JSON changed: {path}")
            row["canonical_sha256"] = canonical
    return row


def require_bound_file(path: Path, expected_sha: object, label: str) -> None:
    if (
        not path.is_absolute()
        or not path.is_file()
        or path.is_symlink()
        or sha256_file(path) != expected_sha
    ):
        raise RuntimeError(f"{label} binding mismatch: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite a continuity receipt")
    try:
        output.relative_to(repo)
    except ValueError:
        pass
    else:
        raise RuntimeError("continuity receipt staging output must be outside the checkout")
    if git_value(repo, "status", "--porcelain"):
        raise RuntimeError("previous activation checkout is dirty")
    if (
        git_value(repo, "rev-parse", "HEAD") != PREVIOUS_COMMIT
        or git_value(repo, "rev-parse", "HEAD^{tree}") != PREVIOUS_TREE
    ):
        raise RuntimeError("continuity must be sealed from the exact activation-v3 checkout")

    plan_path = repo / PREVIOUS_PLAN
    recipes_path = repo / PREVIOUS_RECIPES
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    recipes = json.loads(recipes_path.read_text(encoding="utf-8"))
    if (
        plan.get("schema")
        != "m3m_gcp_native_quarter_100k_ten_method_execution_plan_v3"
        or plan.get("canonical_sha256") != canonical_sha256(plan)
        or recipes.get("schema")
        != "m3m_gcp_native_quarter_100k_recipe_manifest_v3"
        or recipes.get("canonical_sha256") != canonical_sha256(recipes)
    ):
        raise RuntimeError("activation-v3 repository artifacts changed")

    activation_path = RUN_NAMESPACE / "activation_v3.json"
    activation = json.loads(activation_path.read_text(encoding="utf-8"))
    if (
        activation.get("schema") != "m3m_gcp_lidar_formal_activation_v1"
        or activation.get("benchmark_commit") != PREVIOUS_COMMIT
        or activation.get("benchmark_tree") != PREVIOUS_TREE
        or activation.get("execution_plan_path") != PREVIOUS_PLAN
        or activation.get("execution_plan_sha256") != sha256_file(plan_path)
        or activation.get("recipe_manifest_path") != PREVIOUS_RECIPES
        or activation.get("recipe_manifest_sha256") != sha256_file(recipes_path)
        or activation.get("canonical_sha256") != canonical_sha256(activation)
    ):
        raise RuntimeError("activation-v3 identity mismatch")

    recipe_by_method = {
        str(row["method_id"]): json.loads(
            (repo / str(row["path"])).read_text(encoding="utf-8")
        )
        for row in recipes["recipes"]
    }
    reuse = recipe_by_method["3dgs_original"]["reuse_model_binding"]
    reused_model = (
        Path(str(reuse["run_root"])) / str(reuse["point_cloud_relative_path"])
    ).resolve()
    if (
        reused_model.stat().st_size != int(reuse["point_cloud_bytes"])
        or sha256_file(reused_model) != reuse["point_cloud_sha256"]
        or reuse.get("retrain_allowed") is not False
    ):
        raise RuntimeError("reused 3DGS model changed")

    remote_artifacts = [
        file_row("activation_v3", activation_path),
        file_row("3dgs_reused_model", reused_model),
    ]
    inherited_outcomes = []
    for method_id, expected_status in FAILED_OUTCOMES.items():
        failure_path = SCENE_ROOT / method_id / "evidence/training/failure.json"
        failure = json.loads(failure_path.read_text(encoding="utf-8"))
        if (
            failure.get("schema") != "m3m_gcp_lidar_failure_evidence_v1"
            or failure.get("scene") != SCENE
            or failure.get("method_id") != method_id
            or failure.get("status") != expected_status
            or failure.get("failure_stage") != "training"
            or failure.get("canonical_sha256") != canonical_sha256(failure)
        ):
            raise RuntimeError(f"{method_id} failure is not the frozen terminal outcome")
        if expected_status == "OOM_UNRANKED":
            if failure.get("oom_signal") != "CUDA_OUT_OF_MEMORY":
                raise RuntimeError("QGS frozen OOM signal mismatch")
        elif failure.get("oom_signal") is not None:
            raise RuntimeError(f"unexpected OOM signal for {method_id}")
        for path_key, sha_key in (
            ("environment_manifest_path", "environment_manifest_sha256"),
            ("stdout_path", "stdout_sha256"),
            ("stderr_path", "stderr_sha256"),
        ):
            require_bound_file(
                Path(str(failure[path_key])), failure[sha_key], f"{method_id} {path_key}"
            )
        row = file_row(f"{method_id}_failure", failure_path)
        remote_artifacts.append(row)
        inherited_outcomes.append(
            {
                "method_id": method_id,
                "formal_status": expected_status,
                "formal_attempt_consumed": True,
                "retry_allowed": False,
                "failure_sha256": row["sha256"],
            }
        )

    metro_evidence = SCENE_ROOT / "metrogs/evidence"
    prior_root = Path(
        "/root/autodl-tmp/datasets/M3M-GCP-metrogs-prior-v2/gcp_100000_20260610"
    )
    prior_paths = {
        "metrogs_prior_phase_success": metro_evidence / "prior/phase_success.json",
        "metrogs_prior_environment": metro_evidence / "prior/environment.json",
        "metrogs_training_priors": prior_root / "training_priors.json",
        "metrogs_prior_pass_marker": prior_root / "TRAINING_PRIORS_PASS",
        "metrogs_prior_merged_ply": prior_root / "additional_points/metrogs_pi3_merged.ply",
    }
    prior_rows = {role: file_row(role, path) for role, path in prior_paths.items()}
    prior_phase = json.loads(prior_paths["metrogs_prior_phase_success"].read_text(encoding="utf-8"))
    if (
        prior_phase.get("schema") != "m3m_gcp_100k_phase_success_v2"
        or prior_phase.get("status") != "PASS"
        or prior_phase.get("scene") != SCENE
        or prior_phase.get("method_id") != "metrogs"
        or prior_phase.get("phase") != "prior"
        or prior_phase.get("canonical_sha256") != canonical_sha256(prior_phase)
        or prior_phase.get("environment_manifest_sha256")
        != prior_rows["metrogs_prior_environment"]["sha256"]
    ):
        raise RuntimeError("MetroGS prior phase success mismatch")
    prior_products = {
        str(Path(str(row.get("path", ""))).resolve()): row
        for row in prior_phase.get("products", [])
        if isinstance(row, dict)
    }
    for role in (
        "metrogs_training_priors",
        "metrogs_prior_pass_marker",
        "metrogs_prior_merged_ply",
    ):
        row = prior_rows[role]
        product = prior_products.get(str(Path(str(row["path"])).resolve()))
        if (
            not isinstance(product, dict)
            or product.get("bytes") != row["bytes"]
            or product.get("sha256") != row["sha256"]
        ):
            raise RuntimeError(f"MetroGS prior phase omits exact product: {role}")
    remote_artifacts.extend(prior_rows.values())

    guard_console = metro_evidence / "training/guard-console.log"
    console_text = guard_console.read_text(encoding="utf-8", errors="replace")
    if "training requires the exact successful prior phase" not in console_text:
        raise RuntimeError("MetroGS pre-child guard rejection text mismatch")
    run_root = SCENE_ROOT / "metrogs/seed0-v2"
    failure_path = metro_evidence / "training/failure.json"
    if run_root.exists() or failure_path.exists():
        raise RuntimeError("MetroGS pre-child rejection consumed or polluted the attempt")
    console_row = file_row("metrogs_training_prechild_guard_console", guard_console)
    remote_artifacts.append(console_row)

    payload = {
        "schema": "m3m_gcp_100k_activation_continuity_v2",
        "status": "SEALED_V3_TO_V4_GUARD_CONTINUITY",
        "scene": SCENE,
        "sealed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "previous_reviewed_checkout": {
            "commit": PREVIOUS_COMMIT,
            "tree": PREVIOUS_TREE,
            "review_task_id": "019ff12c-cb29-7cb2-8fb6-1d82c5f8c54b",
            "review_verdict": "PASS_100K_TIME_SPACE_EXECUTION_PLAN_V1",
        },
        "repository_artifacts": [
            file_row("execution_plan_v3", plan_path, relative_to=repo),
            file_row("recipe_manifest_v3", recipes_path, relative_to=repo),
            file_row("execution_note_v3", repo / PREVIOUS_NOTE, relative_to=repo),
        ],
        "remote_artifacts": remote_artifacts,
        "inherited_ready_model": {
            "method_id": "3dgs_original",
            "formal_status": "READY_FOR_EVALUATION",
            "retrain_allowed": False,
            "model_bytes": reused_model.stat().st_size,
            "model_sha256": sha256_file(reused_model),
        },
        "inherited_terminal_outcomes": inherited_outcomes,
        "metrogs_prior": {
            "status": "PASS",
            "rerun_allowed": False,
            "phase_success_sha256": prior_rows["metrogs_prior_phase_success"]["sha256"],
            "training_priors_sha256": prior_rows["metrogs_training_priors"]["sha256"],
            "pass_marker_sha256": prior_rows["metrogs_prior_pass_marker"]["sha256"],
            "merged_ply_sha256": prior_rows["metrogs_prior_merged_ply"]["sha256"],
        },
        "metrogs_training_prechild_rejection": {
            "cause": "TRAINING_CONTEXT_REUSED_TRAINING_SOURCE_ROOT_WHILE_REHASHING_PRIOR_COMMAND",
            "child_started": False,
            "run_root_created": False,
            "formal_attempt_consumed": False,
            "retry_allowed_only_after_guard_fix_and_successor_review": True,
            "run_root": str(run_root),
            "failure_path": str(failure_path),
            "console_sha256": console_row["sha256"],
        },
        "transition_policy": {
            "activation_v3_immutable": True,
            "activation_v4_path": str(RUN_NAMESPACE / "activation_v4.json"),
            "continued_run_namespace": str(RUN_NAMESPACE),
            "inherited_failed_methods_forbidden_to_launch": sorted(FAILED_OUTCOMES),
            "3dgs_retraining_forbidden": True,
            "metrogs_prior_rerun_forbidden": True,
            "metrogs_training_is_only_unfinished_attempt": True,
            "final_attempt_freeze_authorization": "activation_v4_only",
            "remote_artifacts_must_remain_byte_identical": True,
            "manual_guard_bypass_forbidden": True,
        },
        "reviewer_ruling": {
            "thread_id": "019ff12c-cb29-7cb2-8fb6-1d82c5f8c54b",
            "prechild_attempt_consumed": False,
            "new_activation_required": "activation_v4",
            "full_guard_100k_lidar_three_track_regression_required": True,
            "exact_commit_tree_review_required": True,
        },
    }
    payload["canonical_sha256"] = canonical_sha256(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

