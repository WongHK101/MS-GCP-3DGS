#!/usr/bin/env python3
"""Shared immutable-checkout gate for the activated 100K three-track addendum."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from m3m_gcp_lidar_artifacts import canonical_sha256, sha256_file


GCP_EVALUATION_RUNTIME_PROBE = r"""
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import numpy._core._multiarray_umath as multiarray


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


print(json.dumps({
    "numpy_core_path": str(Path(multiarray.__file__).resolve()),
    "numpy_core_sha256": digest(multiarray.__file__),
    "numpy_init_path": str(Path(np.__file__).resolve()),
    "numpy_init_sha256": digest(np.__file__),
    "numpy_version": np.__version__,
    "python_implementation": platform.python_implementation(),
    "python_version": platform.python_version(),
    "sys_base_prefix": sys.base_prefix,
    "sys_executable": sys.executable,
    "sys_prefix": sys.prefix,
}, sort_keys=True))
"""


def absolute_without_symlink_resolution(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def probe_gcp_evaluation_runtime(
    python: Path, *, subprocess_environment: dict[str, str]
) -> dict[str, Any]:
    python = absolute_without_symlink_resolution(python)
    if not python.is_file():
        raise FileNotFoundError(python)
    completed = subprocess.run(
        [str(python), "-I", "-B", "-c", GCP_EVALUATION_RUNTIME_PROBE],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        env={str(key): str(value) for key, value in subprocess_environment.items()},
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "frozen GCP evaluation runtime probe failed: "
            f"exit={completed.returncode}; stderr={completed.stderr.strip()}"
        )
    try:
        probed = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("frozen GCP evaluation runtime probe emitted invalid JSON") from exc
    if not isinstance(probed, dict):
        raise RuntimeError("frozen GCP evaluation runtime probe did not emit an object")
    return {
        "python_path": str(python),
        "python_resolved_path": str(python.resolve()),
        "python_binary_sha256": sha256_file(python),
        **probed,
    }


def validate_frozen_gcp_evaluation_runtime(
    gcp_config: dict[str, Any], *, requested_python: Path
) -> tuple[dict[str, Any], dict[str, str]]:
    runtime = gcp_config.get("evaluation_runtime", {})
    expected_python = absolute_without_symlink_resolution(
        Path(str(runtime.get("python_path", "")))
    )
    requested_python = absolute_without_symlink_resolution(requested_python)
    environment = runtime.get("subprocess_environment")
    expected_identity = runtime.get("identity")
    if (
        requested_python != expected_python
        or not isinstance(environment, dict)
        or not environment
        or any(not isinstance(key, str) or not isinstance(value, str) for key, value in environment.items())
        or not isinstance(expected_identity, dict)
        or not expected_identity
    ):
        raise RuntimeError("requested GCP evaluation Python differs from the activated frozen runtime")
    observed = probe_gcp_evaluation_runtime(
        requested_python,
        subprocess_environment={str(key): str(value) for key, value in environment.items()},
    )
    if observed != expected_identity:
        raise RuntimeError("frozen GCP evaluation environment identity mismatch")
    return observed, {str(key): str(value) for key, value in environment.items()}


def git_value(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, encoding="utf-8"
    ).strip()


def validate_addendum_runtime(
    *,
    activation: dict[str, Any],
    candidate: dict[str, Any],
    registry: dict[str, Any],
    executing_file: Path,
) -> tuple[Path, dict[str, Any]]:
    repo = Path(str(registry.get("shared", {}).get("benchmark_repo_template", ""))).resolve()
    executing_file = executing_file.resolve()
    try:
        executing_file.relative_to(repo)
    except ValueError as exc:
        raise RuntimeError("executing addendum file is outside the activated checkout") from exc
    if (
        git_value(repo, "rev-parse", "HEAD") != activation.get("reviewed_addendum_commit")
        or git_value(repo, "show", "-s", "--format=%T", "HEAD")
        != activation.get("reviewed_addendum_tree")
        or git_value(repo, "status", "--porcelain")
    ):
        raise RuntimeError("addendum checkout differs from the activated reviewed identity")
    row = candidate.get("addendum_config", {})
    config_path = Path(str(row.get("path", ""))).resolve()
    if (
        not config_path.is_file()
        or config_path.is_symlink()
        or sha256_file(config_path) != row.get("sha256")
    ):
        raise RuntimeError("activated addendum config file binding mismatch")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        config.get("canonical_sha256") != row.get("canonical_sha256")
        or canonical_sha256(config) != row.get("canonical_sha256")
    ):
        raise RuntimeError("activated addendum config canonical binding mismatch")
    for relative, expected_sha in config.get("bound_addendum_files", {}).items():
        path = (repo / str(relative)).resolve()
        try:
            path.relative_to(repo)
        except ValueError as exc:
            raise RuntimeError("bound addendum file escapes reviewed checkout") from exc
        if not path.is_file() or path.is_symlink() or sha256_file(path) != expected_sha:
            raise RuntimeError(f"bound addendum runtime file changed: {relative}")
    return repo, config
