#!/usr/bin/env python3
"""Build the activation-bound formal 100K RGB render and metric commands."""

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


def git_value(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, encoding="utf-8"
    ).strip()


def require_json(path: Path, expected_sha: str | None = None) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if expected_sha is not None and sha256_file(path) != expected_sha:
        raise RuntimeError(f"file SHA mismatch: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


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
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    activation_path = args.activation.resolve()
    activation = require_json(activation_path)
    if (
        activation.get("schema") != "m3m_gcp_100k_three_track_activation_v1"
        or activation.get("status") != "ACTIVE_FROZEN"
        or activation.get("execution_authorized") is not True
        or activation.get("scene") != SCENE
        or activation.get("canonical_sha256") != canonical_sha256(activation)
    ):
        raise RuntimeError("three-track activation mismatch")
    candidate_path = Path(str(activation["candidate_manifest_path"])).resolve()
    candidate = require_json(candidate_path, str(activation["candidate_manifest_sha256"]))
    if candidate.get("canonical_sha256") != activation["candidate_manifest_canonical_sha256"]:
        raise RuntimeError("activation/candidate binding mismatch")
    registry_path = Path(str(activation["rgb_registry_path"])).resolve()
    registry = require_json(registry_path, str(activation["rgb_registry_sha256"]))
    contract_path = Path(str(candidate["rgb_contract"]["path"])).resolve()
    if not contract_path.is_file() or sha256_file(contract_path) != activation["rgb_contract_sha256"]:
        raise RuntimeError("activation/RGB-contract binding mismatch")
    if (
        registry.get("schema") != "m3m_gcp_native_quarter_rgb_quality_100k_registry_v1"
        or registry.get("status") != "ACTIVE_FROZEN"
        or registry.get("scene") != SCENE
        or registry.get("canonical_sha256") != canonical_sha256(registry)
        or registry.get("active_method_count") != len(registry.get("methods", []))
        or registry.get("ready_method_ids") != [row["method_id"] for row in registry.get("methods", [])]
    ):
        raise RuntimeError("activated 100K RGB registry mismatch")

    repo = Path(str(registry["shared"]["benchmark_repo_template"])).resolve()
    if git_value(repo, "status", "--porcelain"):
        raise RuntimeError("addendum checkout is dirty")
    commit = git_value(repo, "rev-parse", "HEAD")
    tree = git_value(repo, "show", "-s", "--format=%T", "HEAD")
    if commit != activation["reviewed_addendum_commit"] or tree != activation["reviewed_addendum_tree"]:
        raise RuntimeError("addendum checkout differs from activated review identity")
    evaluator = repo / "code" / "gcp" / "evaluate_m3m_native_quarter_rgb_quality.py"
    if not evaluator.is_file():
        raise FileNotFoundError(evaluator)

    shared = registry["shared"]
    jobs: list[dict[str, Any]] = []
    for method in registry["methods"]:
        artifact_root = Path(str(method["formal_output_root"])).resolve()
        if artifact_root.exists() or artifact_root.is_symlink():
            raise FileExistsError(f"RGB formal output root must be fresh: {artifact_root}")
        render_argv = _render_argv(
            method=method,
            shared=shared,
            benchmark_repo=str(repo),
            benchmark_commit=commit,
            benchmark_tree=tree,
            contract_path=str(contract_path),
            artifact_root=str(artifact_root),
        )
        metric_argv = [
            f"{shared['metric_environment']}/bin/python",
            "-B",
            str(evaluator),
            "--rgb_contract",
            str(contract_path),
            "--registry",
            str(registry_path),
            "--benchmark_repo",
            str(repo),
            "--benchmark_commit",
            commit,
            "--benchmark_tree",
            tree,
            "--input_manifest",
            str(shared["input_manifest"]),
            "--input_root",
            str(shared["input_root"]),
            "--render_manifest",
            str(artifact_root / "rgb_render_manifest.json"),
            "--scene",
            SCENE,
            "--method_id",
            str(method["method_id"]),
            "--metric_reference_root",
            str(shared["metric_reference_root"]),
            "--vgg16_weights",
            str(shared["vgg16_weights"]),
            "--lpips_vgg_weights",
            str(shared["lpips_vgg_weights"]),
            "--device",
            str(shared["metric_device"]),
            "--output_dir",
            str(artifact_root / "metrics"),
        ]
        jobs.append(
            {
                "method_id": method["method_id"],
                "artifact_root": str(artifact_root),
                "precondition_absent_paths": [
                    str(artifact_root),
                    str(artifact_root / "renders"),
                    str(artifact_root / "rgb_render_manifest.json"),
                    str(artifact_root / "metrics"),
                ],
                "render": {
                    "working_directory": str(method["source_root"]),
                    "environment": _environment(method),
                    "argv": render_argv,
                    "argv_sha256": command_sha256(render_argv),
                    "shell_preview": shlex.join(render_argv),
                    "stdout": str(artifact_root / "render.stdout.log"),
                    "stderr": str(artifact_root / "render.stderr.log"),
                },
                "metric": {
                    "working_directory": str(shared["metric_reference_root"]),
                    "environment": {},
                    "argv": metric_argv,
                    "argv_sha256": command_sha256(metric_argv),
                    "shell_preview": shlex.join(metric_argv),
                    "stdout": str(artifact_root / "metric.stdout.log"),
                    "stderr": str(artifact_root / "metric.stderr.log"),
                },
            }
        )
    payload: dict[str, Any] = {
        "schema": "m3m_gcp_native_quarter_rgb_quality_100k_execution_plan_v1",
        "suite_id": registry["suite_id"],
        "status": "ACTIVE_FROZEN",
        "formal_execution_authorized": True,
        "scene": SCENE,
        "three_track_activation_path": str(activation_path),
        "three_track_activation_sha256": sha256_file(activation_path),
        "candidate_manifest_sha256": sha256_file(candidate_path),
        "scene_attempt_freeze_sha256": candidate["scene_attempt_freeze"]["sha256"],
        "registry_path": str(registry_path),
        "registry_sha256": sha256_file(registry_path),
        "contract_path": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "benchmark_repo": str(repo),
        "benchmark_commit": commit,
        "benchmark_tree": tree,
        "job_count": len(jobs),
        "method_order": [job["method_id"] for job in jobs],
        "execution_semantics": {
            "order": "render adapter, then the one shared evaluator, per READY method",
            "one_method_failure_does_not_abort_later_methods": True,
            "no_retry_or_selection_from_rgb": True,
            "heldout_view_count_required": 314,
        },
        "jobs": jobs,
    }
    payload["canonical_sha256"] = canonical_sha256(payload)
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    write_exclusive(output, payload)
    print(
        json.dumps(
            {
                "status": "PASS_100K_RGB_COMMANDS_FROZEN",
                "path": str(output),
                "sha256": sha256_file(output),
                "canonical_sha256": payload["canonical_sha256"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
