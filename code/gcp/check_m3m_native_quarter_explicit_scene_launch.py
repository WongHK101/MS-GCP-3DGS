#!/usr/bin/env python3
"""Fail closed unless the exact one-time native-quarter scene run is authorized."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any


PROTOCOL_ID = "m3m_gcp_native_quarter_geometry_v2"
AUTHORIZED_STATUS = "AUTHORIZED_NOT_STARTED"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: dict[str, Any]) -> str:
    clean = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    encoded = json.dumps(clean, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def check_launch(
    registry: dict[str, Any],
    repo_root: Path,
    input_release_root: Path,
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
    require(registry.get("per_method_training_allowed_methods") == [], "3K method allowlist must remain empty")
    scope = registry.get("preliminary_evidence_scope", {})
    require(scope.get("six_scene_matrix_status") == "LOCKED", "six-scene matrix must remain locked")
    require(scope.get("multi_seed_status") == "NOT_AUTHORIZED", "multi-seed execution must remain locked")

    methods = {str(item.get("method_id")): item for item in registry.get("methods", [])}
    method = methods.get(method_id)
    require(method is not None, f"unknown method: {method_id}")
    if method is not None:
        require(method.get("three_k_training_allowed") is False, "3K training flag must remain locked")
        require(method.get("full_scene_matrix_eligible") is False, "full scene matrix must remain locked")

    entry = registry.get("explicit_scene_run_authorization", {})
    require(entry.get("status") == AUTHORIZED_STATUS, "explicit scene authorization is not active")
    require(entry.get("single_fresh_run_allowed") is True, "single fresh run is not allowed")
    require(entry.get("resume_allowed") is False, "resume must be forbidden")
    require(entry.get("rerun_after_attempt_allowed") is False, "rerun-after-attempt must be forbidden")
    require(entry.get("other_methods_scenes_seeds_allowed") is False, "authorization scope is not isolated")
    require(entry.get("method_id") == method_id, "method differs from explicit authorization")
    require(entry.get("scene") == scene, "scene differs from explicit authorization")
    require(entry.get("seed") == seed, "seed differs from explicit authorization")
    require(entry.get("iterations") == iterations, "iterations differ from explicit authorization")
    require(entry.get("run_root") == run_root, "run root differs from explicit authorization")

    authorization: dict[str, Any] = {}
    authorization_path: Path | None = None
    relative = entry.get("authorization")
    if isinstance(relative, str) and relative:
        authorization_path = (repo_root / relative).resolve()
        require(authorization_path.is_relative_to(repo_root.resolve()), "authorization escapes repository")
        require(authorization_path.is_file(), "authorization file is missing")
        if authorization_path.is_file():
            require(file_sha256(authorization_path) == entry.get("authorization_sha256"), "authorization SHA mismatch")
            authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    else:
        errors.append("authorization path is missing")

    require(authorization.get("protocol_id") == PROTOCOL_ID, "authorization protocol mismatch")
    require(authorization.get("authorization_id") == entry.get("authorization_id"), "authorization ID mismatch")
    require(authorization.get("status") == AUTHORIZED_STATUS, "authorization file is not launchable")
    auth_method = authorization.get("method", {})
    require(auth_method.get("method_id") == method_id, "authorization method mismatch")
    if method is not None:
        require(auth_method.get("repository_commit") == method.get("source", {}).get("commit"), "source commit mismatch")
        require(auth_method.get("repository_tree") == method.get("source", {}).get("tree"), "source tree mismatch")
    auth_input = authorization.get("input", {})
    require(auth_input.get("scene") == scene, "authorization input scene mismatch")
    auth_training = authorization.get("training", {})
    require(auth_training.get("seed") == seed, "authorization training seed mismatch")
    require(auth_training.get("iterations") == iterations, "authorization training iterations mismatch")
    auth_execution = authorization.get("execution", {})
    require(auth_execution.get("run_root") == run_root, "authorization execution run root mismatch")
    require(auth_execution.get("single_fresh_run_allowed") is True, "authorization does not allow one fresh run")
    require(auth_execution.get("resume_allowed") is False, "authorization permits resume")
    require(auth_execution.get("rerun_after_attempt_allowed") is False, "authorization permits rerun")
    require(auth_execution.get("overwrite_allowed") is False, "authorization permits overwrite")
    locks = authorization.get("scope_locks", {})
    for key in (
        "protocol_changed",
        "three_k_rerun_authorized",
        "other_scene_authorized",
        "other_method_authorized",
        "other_seed_authorized",
        "six_scene_matrix_authorized",
        "candidate_pool_reopened",
        "tgs_gcp_in_scope",
    ):
        require(locks.get(key) is False, f"scope lock mismatch: {key}")

    manifest_path: Path | None = None
    manifest: dict[str, Any] = {}
    manifest_relative = auth_input.get("formal_input_manifest")
    if isinstance(manifest_relative, str) and manifest_relative:
        manifest_path = (input_release_root / manifest_relative).resolve()
        require(manifest_path.is_relative_to(input_release_root.resolve()), "input manifest escapes release root")
        require(manifest_path.is_file(), "input manifest is missing")
        if manifest_path.is_file():
            require(file_sha256(manifest_path) == auth_input.get("formal_input_manifest_file_sha256"), "input manifest file SHA mismatch")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        errors.append("input manifest path is missing")
    require(manifest.get("manifest_sha256") == auth_input.get("formal_input_manifest_canonical_sha256"), "input manifest canonical identity mismatch")
    require(canonical_sha256(manifest) == manifest.get("manifest_sha256"), "input manifest canonical SHA is invalid")
    require(manifest.get("scene") == scene, "input manifest scene mismatch")
    for field in ("full_view_count", "train_view_count", "test_view_count"):
        require(manifest.get(field) == auth_input.get(field), f"input {field} mismatch")
    require(manifest.get("pixel_domain") == auth_input.get("pixel_domain"), "input pixel domain mismatch")

    pure_run = PurePosixPath(run_root)
    namespace = str(auth_method.get("run_namespace", ""))
    expected_parent = PurePosixPath("/root/autodl-tmp/runs/m3m-gcp-native-quarter") / namespace / scene
    require(bool(namespace), "run namespace is missing")
    require(pure_run.is_absolute(), "run root must be absolute")
    require(pure_run.parent == expected_parent, "run root is outside the exact method/scene namespace")
    require(not run_root_exists, "run root already exists; overwrite, resume, and rerun are forbidden")

    return {
        "schema": "m3m_gcp_native_quarter_explicit_scene_launch_gate_v1",
        "protocol_id": PROTOCOL_ID,
        "authorization_id": entry.get("authorization_id"),
        "method_id": method_id,
        "scene": scene,
        "seed": seed,
        "iterations": iterations,
        "run_root": run_root,
        "authorization": str(authorization_path) if authorization_path else None,
        "input_manifest": str(manifest_path) if manifest_path else None,
        "passed": not errors,
        "status": "AUTHORIZED" if not errors else "DENIED",
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--registry", type=Path, default=default_root / "configs" / "m3m_gcp_native_quarter_method_registry_v2.json")
    parser.add_argument("--repo_root", type=Path, default=default_root)
    parser.add_argument("--input_release_root", required=True, type=Path)
    parser.add_argument("--method_id", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--run_root", required=True)
    args = parser.parse_args()
    result = check_launch(
        json.loads(args.registry.read_text(encoding="utf-8")),
        args.repo_root.resolve(),
        args.input_release_root.resolve(),
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
