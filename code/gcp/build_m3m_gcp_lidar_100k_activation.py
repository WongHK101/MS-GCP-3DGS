#!/usr/bin/env python3
"""Build the dual-review activation for the exact reviewed 100K checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from m3m_gcp_lidar_artifacts import canonical_sha256, sha256_file


PROTOCOL_ID = "m3m_gcp_lidar_rendered_surface_v1"
REVIEW_TASK_ID = "019ff12c-cb29-7cb2-8fb6-1d82c5f8c54b"
PROTOCOL_VERDICT = "PASS_LIDAR_V1_AND_SIX_SCENE_PREPARATION_V2"
PLAN_VERDICT = "PASS_100K_TIME_SPACE_EXECUTION_PLAN_V1"
CONTRACT = "configs/m3m_gcp_lidar_formal_v1.json"
SCHEMA = "configs/m3m_gcp_lidar_formal_artifact_schema_v1.json"
LOCAL_PREPARATION = "docs/protocol_evidence/m3m_gcp_six_scene_common_preparation_local_v2.json"
REMOTE_PREPARATION = "docs/protocol_evidence/m3m_gcp_six_scene_common_preparation_remote_v2.json"
PLAN = "configs/m3m_gcp_native_quarter_100k_ten_method_execution_plan_v2.json"
RECIPES = "configs/m3m_gcp_native_quarter_100k_recipe_manifest_v2.json"


def git_value(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, encoding="utf-8"
    ).strip()


def git_blob_sha256(repo: Path, commit: str, relative_path: str) -> str:
    payload = subprocess.check_output(
        ["git", "-C", str(repo), "show", f"{commit}:{relative_path}"]
    )
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--protocol-reviewed-commit", required=True)
    parser.add_argument("--execution-plan-reviewed-commit", required=True)
    parser.add_argument("--execution-plan-review-verdict", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite an activation manifest")
    try:
        output.relative_to(repo)
    except ValueError:
        pass
    else:
        raise RuntimeError("activation output must be outside the clean benchmark checkout")
    if args.execution_plan_review_verdict != PLAN_VERDICT:
        raise RuntimeError("exact 100K execution-plan review verdict is absent")
    if git_value(repo, "status", "--porcelain"):
        raise RuntimeError("reviewed benchmark checkout is dirty")
    head = git_value(repo, "rev-parse", "HEAD")
    tree = git_value(repo, "show", "-s", "--format=%T", "HEAD")
    if head != args.execution_plan_reviewed_commit:
        raise RuntimeError("checkout differs from execution-plan reviewed commit")
    plan_tree = git_value(
        repo, "show", "-s", "--format=%T", args.execution_plan_reviewed_commit
    )
    protocol_tree = git_value(
        repo, "show", "-s", "--format=%T", args.protocol_reviewed_commit
    )
    paths = {
        "contract": repo / CONTRACT,
        "schema": repo / SCHEMA,
        "local": repo / LOCAL_PREPARATION,
        "remote": repo / REMOTE_PREPARATION,
        "plan": repo / PLAN,
        "recipes": repo / RECIPES,
    }
    if any(not path.is_file() for path in paths.values()):
        raise RuntimeError("one or more activation-bound files are missing")
    for key, relative in (
        ("contract", CONTRACT), ("schema", SCHEMA),
        ("local", LOCAL_PREPARATION), ("remote", REMOTE_PREPARATION),
    ):
        if git_blob_sha256(repo, args.protocol_reviewed_commit, relative) != sha256_file(paths[key]):
            raise RuntimeError(f"Phase-1 reviewed artifact changed: {relative}")
    if sha256_file(paths["local"]) != sha256_file(paths["remote"]):
        raise RuntimeError("local/remote common-preparation evidence bytes differ")
    contract = json.loads(paths["contract"].read_text(encoding="utf-8"))
    plan = json.loads(paths["plan"].read_text(encoding="utf-8"))
    recipes = json.loads(paths["recipes"].read_text(encoding="utf-8"))
    if (
        contract.get("protocol_id") != PROTOCOL_ID
        or contract.get("execution_authorized") is not False
        or contract.get("review", {}).get("protocol_review_task_id") != REVIEW_TASK_ID
    ):
        raise RuntimeError("formal LiDAR contract candidate identity changed")
    phase1 = plan.get("formal_lidar_protocol", {}).get("phase1_review", {})
    if (
        phase1.get("task_id") != REVIEW_TASK_ID
        or phase1.get("verdict") != PROTOCOL_VERDICT
        or phase1.get("reviewed_commit") != args.protocol_reviewed_commit
        or phase1.get("reviewed_tree") != protocol_tree
        or phase1.get("protocol_pass_alone_authorizes_execution") is not False
    ):
        raise RuntimeError("execution plan does not bind the exact Phase-1 PASS")
    if (
        plan.get("schema")
        != "m3m_gcp_native_quarter_100k_ten_method_execution_plan_v2"
        or plan.get("status") != "REVIEW_CANDIDATE_NOT_EXECUTION_AUTHORIZED"
        or plan.get("execution_authorized") is not False
        or plan.get("review", {}).get("task_id") != REVIEW_TASK_ID
        or plan.get("review", {}).get("required_pass_verdict") != PLAN_VERDICT
        or plan.get("canonical_sha256") != canonical_sha256(plan)
    ):
        raise RuntimeError("100K execution-plan candidate identity changed")
    if (
        recipes.get("schema") != "m3m_gcp_native_quarter_100k_recipe_manifest_v2"
        or recipes.get("status") != "REVIEW_CANDIDATE_NOT_EXECUTION_AUTHORIZED"
        or recipes.get("canonical_sha256") != canonical_sha256(recipes)
        or len(recipes.get("recipes", [])) != 10
    ):
        raise RuntimeError("100K recipe-manifest candidate identity changed")
    payload = {
        "schema": "m3m_gcp_lidar_formal_activation_v1",
        "protocol_id": PROTOCOL_ID,
        "protocol_review_task_id": REVIEW_TASK_ID,
        "protocol_review_verdict": PROTOCOL_VERDICT,
        "protocol_reviewed_commit": args.protocol_reviewed_commit,
        "protocol_reviewed_tree": protocol_tree,
        "execution_plan_review_task_id": REVIEW_TASK_ID,
        "execution_plan_review_verdict": PLAN_VERDICT,
        "execution_plan_reviewed_commit": args.execution_plan_reviewed_commit,
        "execution_plan_reviewed_tree": plan_tree,
        "execution_authorized": True,
        "contract_file_sha256": sha256_file(paths["contract"]),
        "artifact_schema_sha256": sha256_file(paths["schema"]),
        "common_preparation_local_path": LOCAL_PREPARATION,
        "common_preparation_local_sha256": sha256_file(paths["local"]),
        "common_preparation_remote_path": REMOTE_PREPARATION,
        "common_preparation_remote_sha256": sha256_file(paths["remote"]),
        "execution_plan_path": PLAN,
        "execution_plan_sha256": sha256_file(paths["plan"]),
        "recipe_manifest_path": RECIPES,
        "recipe_manifest_sha256": sha256_file(paths["recipes"]),
        "benchmark_commit": head,
        "benchmark_tree": tree,
    }
    payload["canonical_sha256"] = canonical_sha256(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
