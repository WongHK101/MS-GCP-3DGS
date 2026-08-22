#!/usr/bin/env python3
"""Shared immutable-checkout gate for the activated 100K three-track addendum."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from m3m_gcp_lidar_artifacts import canonical_sha256, sha256_file


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
