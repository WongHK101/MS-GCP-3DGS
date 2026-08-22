#!/usr/bin/env python3
"""Launch one reviewed 100K qualification recipe with explicit phase environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_m3m_gcp_100k_guarded import (
    validate_frozen_training_images,
    validate_prepared_method_input,
)


ATTEMPT_PATTERN = re.compile(r"^[0-9]{8}T[0-9]{6}Z(?:-[a-z0-9-]+)?$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(payload: dict[str, Any]) -> str:
    clean = {key: value for key, value in payload.items() if key != "canonical_sha256"}
    raw = json.dumps(clean, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True
    ).rstrip("\r\n")


def render(value: str, bindings: dict[str, str]) -> str:
    return value.format_map(bindings)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def validate_bound_training_input(
    recipe: dict[str, Any], dataset_root: Path
) -> None:
    """Reuse only the old guard's pure data checks, not its lifecycle state machine."""
    validate_frozen_training_images(recipe, dataset_root)
    validate_prepared_method_input(recipe, dataset_root)


def verify_plan_binding(
    *, recipe: dict[str, Any], recipe_path: Path, benchmark_repo: Path
) -> dict[str, Any]:
    plan_path = (
        benchmark_repo
        / "configs"
        / "m3m_gcp_native_quarter_100k_qualification_v1.json"
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if canonical(plan) != plan.get("canonical_sha256"):
        raise RuntimeError("qualification plan canonical identity mismatch")
    try:
        relative = recipe_path.relative_to(benchmark_repo).as_posix()
    except ValueError as exc:
        raise RuntimeError("recipe must be inside the reviewed benchmark worktree") from exc
    rows = [row for row in plan["recipes"] if row["path"] == relative]
    if len(rows) != 1:
        raise RuntimeError("recipe is not uniquely owned by the qualification plan")
    row = rows[0]
    if row["method_id"] != recipe["method_id"]:
        raise RuntimeError("plan-to-recipe method identity mismatch")
    if row["file_sha256"] != sha256(recipe_path):
        raise RuntimeError("plan-to-recipe file identity mismatch")
    if row["canonical_sha256"] != recipe["canonical_sha256"]:
        raise RuntimeError("plan-to-recipe canonical identity mismatch")
    if plan["plan_id"] != recipe["plan_id"]:
        raise RuntimeError("plan-to-recipe qualification identity mismatch")
    return plan


def verify_scientific_inputs(
    recipe: dict[str, Any], benchmark_repo: Path, attempt_id: str
) -> tuple[dict[str, str], Path, list[str], dict[str, str]]:
    if canonical(recipe) != recipe.get("canonical_sha256"):
        raise RuntimeError("recipe canonical identity mismatch")
    if recipe.get("status") != "REVIEW_CANDIDATE_NOT_EXECUTION_AUTHORIZED":
        raise RuntimeError("unexpected qualification recipe status")
    if recipe.get("training") is None:
        raise RuntimeError("this recipe reuses a model and has no training phase")
    if not ATTEMPT_PATTERN.fullmatch(attempt_id):
        raise ValueError("attempt ID must be UTC YYYYMMDDTHHMMSSZ with an optional suffix")

    training = recipe["training"]
    run_root = Path(
        training["attempt_root_template"].format(attempt_id=attempt_id)
    ).resolve()
    if run_root.exists():
        raise FileExistsError(f"fresh attempt root already exists: {run_root}")
    source_root = Path(training["source_root"]).resolve()
    dataset_root = Path(training["dataset_root"]).resolve()
    prior_root = Path(training["prior_root"]).resolve()
    interpreter = Path(training["interpreter"])
    for required in (benchmark_repo / ".git", source_root / ".git", dataset_root, prior_root, interpreter):
        if not required.exists():
            raise FileNotFoundError(required)

    scientific = recipe["scientific_contract"]
    manifest_spec = scientific["formal_input_manifest"]
    manifest_path = Path(manifest_spec["path"]).resolve()
    if sha256(manifest_path) != manifest_spec["file_sha256"]:
        raise RuntimeError("formal input manifest file identity mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("manifest_sha256") != manifest_spec["canonical_sha256"]:
        raise RuntimeError("formal input manifest canonical identity mismatch")
    if manifest.get("scene") != recipe["scene"]:
        raise RuntimeError("formal input scene mismatch")
    if int(manifest.get("train_view_count", -1)) != manifest_spec["train_views"]:
        raise RuntimeError("formal input train-view count mismatch")
    if int(manifest.get("test_view_count", -1)) != manifest_spec["heldout_views"]:
        raise RuntimeError("formal input heldout-view count mismatch")

    source = training["source_binding"]
    if git_output(source_root, "rev-parse", "HEAD") != source["commit"]:
        raise RuntimeError("method source commit mismatch")
    if git_output(source_root, "rev-parse", "HEAD^{tree}") != source["tree"]:
        raise RuntimeError("method source tree mismatch")
    if git_output(source_root, "status", "--porcelain=v1") != source["required_status"]:
        raise RuntimeError("method source compatibility status mismatch")
    for relative, expected in source["required_files_sha256"].items():
        path = source_root / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"method source-file identity mismatch: {relative}")

    validate_bound_training_input(recipe, dataset_root)

    bindings = {
        "repo": str(benchmark_repo),
        "source_root": str(source_root),
        "dataset_root": str(dataset_root),
        "prior_root": str(prior_root),
        "run_root": str(run_root),
        "attempt_id": attempt_id,
    }
    command = [render(str(value), bindings) for value in training["command"]]
    environment = {
        key: render(str(value), bindings)
        for key, value in training["environment"].items()
    }
    if Path(command[0]) != interpreter:
        raise RuntimeError("training command interpreter differs from recipe binding")
    return bindings, run_root, command, environment


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--attempt_id", required=True)
    parser.add_argument("--benchmark_repo", type=Path, required=True)
    parser.add_argument(
        "--reviewed_commit",
        help="Exact benchmark commit approved by the one prelaunch batch review.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Without this flag, perform only read-only preflight and print the launch.",
    )
    args = parser.parse_args()

    recipe_path = args.recipe.resolve()
    benchmark_repo = args.benchmark_repo.resolve()
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    plan = verify_plan_binding(
        recipe=recipe, recipe_path=recipe_path, benchmark_repo=benchmark_repo
    )
    bindings, run_root, command, environment = verify_scientific_inputs(
        recipe, benchmark_repo, args.attempt_id
    )
    head = git_output(benchmark_repo, "rev-parse", "HEAD")
    if git_output(benchmark_repo, "status", "--porcelain=v1"):
        raise RuntimeError("benchmark qualification worktree must be clean")
    launch = {
        "schema": "m3m_gcp_100k_qualification_launch_v1",
        "status": "PREFLIGHT_PASS" if not args.execute else "LAUNCHING",
        "method_id": recipe["method_id"],
        "scene": recipe["scene"],
        "attempt_id": args.attempt_id,
        "recipe_path": str(recipe_path),
        "recipe_sha256": sha256(recipe_path),
        "recipe_canonical_sha256": recipe["canonical_sha256"],
        "plan_canonical_sha256": plan["canonical_sha256"],
        "benchmark_commit": head,
        "run_root": str(run_root),
        "command": command,
        "environment": environment,
        "started_utc": utc_now(),
    }
    if not args.execute:
        print(json.dumps(launch, indent=2, sort_keys=True))
        return 0
    if not args.reviewed_commit or args.reviewed_commit != head:
        raise RuntimeError("--execute requires the exact batch-reviewed benchmark commit")

    run_root.mkdir(parents=True)
    for item in recipe["training"].get("materializations", []):
        destination = run_root / item["relative_path"]
        if destination.exists():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(render(item["content"], bindings), encoding="utf-8")
    write_json(run_root / "qualification_launch_receipt.json", launch)

    env = dict(os.environ)
    for name in (
        "PYTHONPATH",
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD",
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        env.pop(name, None)
    env.update(environment)
    interpreter = Path(recipe["training"]["interpreter"])
    env["PATH"] = str(interpreter.parent) + os.pathsep + env.get("PATH", "")

    probe = recipe["training"]["resource_probe"]
    probe_command = [
        sys.executable,
        render(probe["script"], bindings),
        "--contract",
        render(probe["contract"], bindings),
        "--output_dir",
        str(run_root / "telemetry"),
        "--working_directory",
        render(recipe["training"]["working_directory"], bindings),
        "--gpu_indices",
        probe["gpu_indices"],
        "--time_binary",
        probe["time_binary"],
        "--failure_stage",
        "training",
    ]
    if probe.get("timeout_seconds") is not None:
        probe_command.extend(
            ["--timeout_seconds", str(probe["timeout_seconds"])]
        )
    probe_command.extend(["--", *command])
    completed = subprocess.run(probe_command, env=env, check=False)
    terminal = {
        "schema": "m3m_gcp_100k_qualification_terminal_v1",
        "method_id": recipe["method_id"],
        "scene": recipe["scene"],
        "attempt_id": args.attempt_id,
        "status": "CHILD_EXIT_ZERO" if completed.returncode == 0 else "CHILD_FAILED",
        "returncode": completed.returncode,
        "finished_utc": utc_now(),
        "telemetry_root": str(run_root / "telemetry"),
        "note": "saved-model and evaluator validation remain separate qualification phases",
    }
    write_json(run_root / "qualification_terminal_receipt.json", terminal)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
