#!/usr/bin/env python3
"""Fail closed unless one exact native-quarter method run is formally authorized."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any


PROTOCOL_ID = "m3m_gcp_native_quarter_geometry_v2"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_launch(
    registry: dict[str, Any],
    repo_root: Path,
    *,
    method_id: str,
    scene: str,
    seed: int,
    iterations: int,
    run_root: str,
    run_root_exists: bool,
) -> dict[str, Any]:
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(registry.get("protocol_id") == PROTOCOL_ID, "registry protocol mismatch")
    require(registry.get("global_training_allowed") is False, "global training must remain locked")
    methods = {str(item.get("method_id")): item for item in registry.get("methods", [])}
    method = methods.get(method_id)
    require(method is not None, f"unknown method: {method_id}")
    allowlist = registry.get("per_method_training_allowed_methods", [])
    computed_allowlist = sorted(
        str(item.get("method_id"))
        for item in registry.get("methods", [])
        if item.get("three_k_training_allowed") is True
    )
    require(sorted(allowlist) == computed_allowlist, "registry allowlist and method flags disagree")
    require(method_id in allowlist, f"method is not formally allowlisted: {method_id}")

    recipe: dict[str, Any] = {}
    recipe_path: Path | None = None
    if method is not None:
        formal = method.get("formal_3k_result", {})
        require(formal.get("rerun_allowed") is not False, "completed formal result forbids rerun")
        require(not str(formal.get("status", "")).startswith("COMPLETE"), "completed formal result already exists")
        require(method.get("three_k_training_allowed") is True, "method training flag is locked")
        require(method.get("recipe_status") == "FROZEN_3K_TRAINING_AUTHORIZED", "recipe is not training-authorized")
        require(method.get("three_k_qualification_status") == "QUALIFIED_3K_TRAINING_AUTHORIZED", "method qualification is incomplete")
        relative = method.get("recipe")
        if isinstance(relative, str) and relative:
            recipe_path = (repo_root / relative).resolve()
            require(recipe_path.is_relative_to(repo_root.resolve()), "recipe escapes repository")
            require(recipe_path.is_file(), "recipe file is missing")
            if recipe_path.is_file():
                require(file_sha256(recipe_path) == method.get("recipe_sha256"), "recipe SHA mismatch")
                recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
        else:
            errors.append("recipe path is missing")

    require(recipe.get("protocol_id") == PROTOCOL_ID, "recipe protocol mismatch")
    require(recipe.get("method", {}).get("method_id") == method_id, "recipe method mismatch")
    require(recipe.get("status") == "FROZEN_3K_TRAINING_AUTHORIZED", "recipe status remains locked")
    require(recipe.get("execution", {}).get("training_authorized") is True, "recipe execution authorization missing")
    require(recipe.get("qualification", {}).get("three_k_training_allowed") is True, "recipe qualification authorization missing")
    require(recipe.get("qualification_scene", {}).get("scene_id") == scene, "scene differs from frozen qualification scene")
    require(recipe.get("training", {}).get("seed") == seed, "seed differs from frozen recipe")
    require(recipe.get("training", {}).get("iterations") == iterations, "iteration count differs from frozen recipe")

    pure_run = PurePosixPath(run_root)
    expected_prefix = PurePosixPath("/root/autodl-tmp/runs/m3m-gcp-native-quarter") / method_id / scene
    require(pure_run.is_absolute(), "run root must be absolute")
    require(pure_run != expected_prefix, "run root must include a unique run id")
    require(expected_prefix in pure_run.parents, "run root is outside the frozen method/scene namespace")
    require(not run_root_exists, "run root already exists; overwrite and resume are forbidden")

    return {
        "schema": "m3m_gcp_native_quarter_formal_launch_gate_v1",
        "protocol_id": PROTOCOL_ID,
        "method_id": method_id,
        "scene": scene,
        "seed": seed,
        "iterations": iterations,
        "run_root": run_root,
        "recipe": str(recipe_path) if recipe_path else None,
        "passed": not errors,
        "status": "AUTHORIZED" if not errors else "DENIED",
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--registry", type=Path, default=default_root / "configs" / "m3m_gcp_native_quarter_method_registry_v2.json")
    parser.add_argument("--repo_root", type=Path, default=default_root)
    parser.add_argument("--method_id", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--run_root", required=True)
    args = parser.parse_args()
    result = check_launch(
        json.loads(args.registry.read_text(encoding="utf-8")),
        args.repo_root.resolve(),
        method_id=args.method_id,
        scene=args.scene,
        seed=args.seed,
        iterations=args.iterations,
        run_root=args.run_root,
        run_root_exists=Path(args.run_root).exists(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
