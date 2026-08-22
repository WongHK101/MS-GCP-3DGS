#!/usr/bin/env python3
"""Build unranked RGB adapter smoke commands from a post-freeze review candidate."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

from build_m3m_native_quarter_rgb_quality_3k_commands import _environment, _render_argv
from m3m_gcp_lidar_artifacts import canonical_sha256, command_sha256, sha256_file


SCENE = "gcp_100000_20260610"


def require_json(path: Path, expected_sha: str | None = None) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if expected_sha is not None and sha256_file(path) != expected_sha:
        raise RuntimeError(f"file SHA mismatch: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def git_value(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, encoding="utf-8"
    ).strip()


def write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(
            descriptor,
            (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
    finally:
        os.close(descriptor)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--addendum-repo", type=Path, required=True)
    parser.add_argument("--technical-smoke-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidate_path = args.candidate_manifest.resolve()
    candidate = require_json(candidate_path)
    if (
        candidate.get("schema") != "m3m_gcp_100k_three_track_candidate_manifest_v1"
        or candidate.get("status") != "REVIEW_CANDIDATE_NOT_EXECUTION_AUTHORIZED"
        or candidate.get("execution_authorized") is not False
        or candidate.get("scene") != SCENE
        or candidate.get("canonical_sha256") != canonical_sha256(candidate)
    ):
        raise RuntimeError("three-track review candidate mismatch")
    repo = args.addendum_repo.resolve()
    if git_value(repo, "status", "--porcelain"):
        raise RuntimeError("addendum checkout is dirty")
    commit = git_value(repo, "rev-parse", "HEAD")
    tree = git_value(repo, "show", "-s", "--format=%T", "HEAD")
    if candidate.get("addendum_checkout") != {"commit": commit, "tree": tree}:
        raise RuntimeError("addendum checkout differs from candidate")

    registry_path = Path(str(candidate["rgb_registry"]["path"])).resolve()
    registry = require_json(registry_path, str(candidate["rgb_registry"]["sha256"]))
    contract_path = Path(str(candidate["rgb_contract"]["path"])).resolve()
    require_json(contract_path, str(candidate["rgb_contract"]["sha256"]))
    if (
        registry.get("status") != "ACTIVE_FROZEN"
        or registry.get("scene") != SCENE
        or registry.get("canonical_sha256") != candidate["rgb_registry"]["canonical_sha256"]
        or canonical_sha256(registry) != candidate["rgb_registry"]["canonical_sha256"]
    ):
        raise RuntimeError("candidate RGB registry mismatch")

    smoke_root = args.technical_smoke_root.resolve()
    if smoke_root.exists() or smoke_root.is_symlink():
        raise FileExistsError("technical-smoke root must be fresh")
    formal_root = Path(str(candidate["formal_results_root"])).resolve()
    try:
        smoke_root.relative_to(formal_root)
    except ValueError:
        pass
    else:
        raise RuntimeError("technical-smoke root must be outside formal results")

    shared = registry["shared"]
    jobs: list[dict[str, Any]] = []
    for method in registry["methods"]:
        method_id = str(method["method_id"])
        artifact_root = smoke_root / method_id
        argv = _render_argv(
            method=method,
            shared=shared,
            benchmark_repo=str(repo),
            benchmark_commit=commit,
            benchmark_tree=tree,
            contract_path=str(contract_path),
            artifact_root=str(artifact_root),
        )
        argv.extend(
            [
                "--allow_review_candidate",
                "--technical_smoke_root",
                str(artifact_root),
            ]
        )
        jobs.append(
            {
                "method_id": method_id,
                "artifact_root": str(artifact_root),
                "working_directory": str(method["source_root"]),
                "environment": _environment(method),
                "argv": argv,
                "argv_sha256": command_sha256(argv),
                "shell_preview": shlex.join(argv),
                "stdout": str(artifact_root / "render.stdout.log"),
                "stderr": str(artifact_root / "render.stderr.log"),
            }
        )
    payload: dict[str, Any] = {
        "schema": "m3m_gcp_native_quarter_rgb_quality_100k_smoke_plan_v1",
        "status": "TECHNICAL_SMOKE_ONLY_UNRANKED",
        "formal_execution_authorized": False,
        "scene": SCENE,
        "candidate_manifest_path": str(candidate_path),
        "candidate_manifest_sha256": sha256_file(candidate_path),
        "registry_path": str(registry_path),
        "registry_sha256": sha256_file(registry_path),
        "contract_path": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "benchmark_repo": str(repo),
        "benchmark_commit": commit,
        "benchmark_tree": tree,
        "technical_smoke_root": str(smoke_root),
        "job_count": len(jobs),
        "method_order": [row["method_id"] for row in jobs],
        "semantics": {
            "renders_all_314_heldout_cameras_to_verify_real_loader_and_renderer_paths": True,
            "shared_metrics_not_executed": True,
            "outputs_are_unranked_and_must_never_be_promoted": True,
            "formal_output_roots_untouched": True,
        },
        "jobs": jobs,
    }
    payload["canonical_sha256"] = canonical_sha256(payload)
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    write_exclusive(output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
