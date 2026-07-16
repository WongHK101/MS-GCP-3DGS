#!/usr/bin/env python3
"""Validate that an MS-GCP method run cannot write into shared source roots."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


CONTRACT_SCHEMA = "ms_gcp_method_workspace_isolation_contract_v1"
RUN_SCHEMA = "ms_gcp_method_run_layout_v1"
TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _flavor(value: str):
    if re.match(r"^[A-Za-z]:[\\/]", value):
        return PureWindowsPath
    if value.startswith("/"):
        return PurePosixPath
    raise ValueError(f"path must be absolute: {value!r}")


def _pure(value: str):
    flavor = _flavor(value)
    return flavor(value)


def _parts(value: str) -> tuple[str, ...]:
    path = _pure(value)
    parts = tuple(path.parts)
    if isinstance(path, PureWindowsPath):
        return tuple(part.casefold() for part in parts)
    return parts


def is_within(path: str, root: str) -> bool:
    path_parts = _parts(path)
    root_parts = _parts(root)
    return len(path_parts) >= len(root_parts) and path_parts[: len(root_parts)] == root_parts


def _require_sha256(value: Any, name: str, errors: list[str]) -> None:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        errors.append(f"{name} must be lowercase SHA-256")


def validate_run_layout(
    manifest: dict[str, Any],
    contract: dict[str, Any],
    *,
    check_run_root_absent: bool = False,
) -> list[str]:
    errors: list[str] = []
    if contract.get("schema") != CONTRACT_SCHEMA:
        errors.append("unknown isolation contract schema")
        return errors
    if manifest.get("schema") != RUN_SCHEMA:
        errors.append("unknown run layout schema")

    for field in contract.get("required_run_fields", []):
        if field not in manifest or manifest[field] in (None, ""):
            errors.append(f"missing required run field: {field}")

    method_id = str(manifest.get("method_id", ""))
    scene = str(manifest.get("scene", ""))
    run_id = str(manifest.get("run_id", ""))
    for name, value in [("method_id", method_id), ("scene", scene), ("run_id", run_id)]:
        if not TOKEN_RE.fullmatch(value):
            errors.append(f"unsafe {name}: {value!r}")

    path_fields = [
        "code_root",
        "environment_root",
        "dataset_root",
        "release_root",
        "build_root",
        "run_root",
        "torch_extensions_dir",
        "temp_root",
    ]
    paths: dict[str, str] = {}
    for field in path_fields:
        value = manifest.get(field)
        if not isinstance(value, str) or not value:
            continue
        try:
            _pure(value)
        except ValueError as exc:
            errors.append(f"{field}: {exc}")
        else:
            paths[field] = value

    immutable_roots = [paths[name] for name in ("dataset_root", "release_root", "code_root") if name in paths]
    mutable_fields = ["environment_root", "build_root", "run_root", "torch_extensions_dir", "temp_root"]
    for field in mutable_fields:
        value = paths.get(field)
        if not value:
            continue
        for root in immutable_roots:
            try:
                if is_within(value, root):
                    errors.append(f"mutable {field} must not be inside immutable root: {root}")
            except ValueError:
                errors.append(f"path flavor mismatch between {field} and immutable root")

    run_root = paths.get("run_root")
    if run_root:
        run_parts = _parts(run_root)
        expected_tokens = {method_id.casefold(), scene.casefold(), run_id.casefold()}
        if not expected_tokens.issubset(set(part.casefold() for part in run_parts)):
            errors.append("run_root must contain method_id, scene, and run_id as path components")
        if check_run_root_absent and Path(run_root).exists():
            errors.append("run_root already exists; overwrite is forbidden")

    for field in ("build_root", "torch_extensions_dir", "temp_root"):
        value = paths.get(field)
        if not value:
            continue
        components = {part.casefold() for part in _parts(value)}
        if method_id.casefold() not in components or run_id.casefold() not in components:
            errors.append(f"{field} must be method- and run-specific")

    if paths.get("torch_extensions_dir") and paths.get("build_root"):
        if not is_within(paths["torch_extensions_dir"], paths["build_root"]):
            errors.append("torch_extensions_dir must be inside the method/run build_root")
    if paths.get("temp_root") and run_root and not is_within(paths["temp_root"], run_root):
        errors.append("temp_root must be inside run_root")

    output_subdirs = manifest.get("output_subdirs")
    required_outputs = contract.get("required_output_subdirs", {})
    if not isinstance(output_subdirs, dict):
        errors.append("output_subdirs must be an object")
    elif run_root:
        for role, leaf in required_outputs.items():
            value = output_subdirs.get(role)
            if not isinstance(value, str):
                errors.append(f"missing output_subdirs.{role}")
                continue
            try:
                if not is_within(value, run_root) or _pure(value).name != leaf:
                    errors.append(f"output_subdirs.{role} must be run_root/{leaf}")
            except ValueError:
                errors.append(f"output_subdirs.{role} has incompatible path flavor")

    policies = manifest.get("policies")
    if not isinstance(policies, dict):
        errors.append("policies must be an object")
    else:
        for key, expected in contract.get("required_policy_values", {}).items():
            if policies.get(key) != expected:
                errors.append(f"policies.{key} must equal {expected!r}")

    env_vars = manifest.get("env_vars")
    if not isinstance(env_vars, dict):
        errors.append("env_vars must be an object")
    else:
        for key, expected in contract.get("required_env_values", {}).items():
            if env_vars.get(key) != expected:
                errors.append(f"env_vars.{key} must equal {expected!r}")
        if paths.get("torch_extensions_dir") and env_vars.get("TORCH_EXTENSIONS_DIR") != paths["torch_extensions_dir"]:
            errors.append("TORCH_EXTENSIONS_DIR must equal the isolated torch_extensions_dir")
        if paths.get("temp_root") and env_vars.get("TMPDIR") != paths["temp_root"]:
            errors.append("TMPDIR must equal the isolated temp_root")

    _require_sha256(manifest.get("environment_lock_sha256"), "environment_lock_sha256", errors)
    _require_sha256(manifest.get("release_root_digest"), "release_root_digest", errors)
    code_commit = manifest.get("code_commit")
    if not isinstance(code_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", code_commit):
        errors.append("code_commit must be a full lowercase Git commit")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "configs" / "gcp_v13_workspace_isolation_v1.json",
    )
    parser.add_argument("--require_nonexistent_run_root", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    errors = validate_run_layout(
        manifest,
        contract,
        check_run_root_absent=args.require_nonexistent_run_root,
    )
    report = {
        "schema": "ms_gcp_method_workspace_isolation_validation_v1",
        "status": "pass" if not errors else "fail",
        "error_count": len(errors),
        "errors": errors,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
