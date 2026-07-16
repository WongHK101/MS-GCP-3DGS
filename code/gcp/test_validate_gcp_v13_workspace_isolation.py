#!/usr/bin/env python3
"""Focused tests for the v1.3 method workspace isolation contract."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from validate_gcp_v13_workspace_isolation import validate_run_layout


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = json.loads((ROOT / "configs" / "gcp_v13_workspace_isolation_v1.json").read_text(encoding="utf-8"))
HEX40 = "1" * 40
HEX64 = "2" * 64


def valid_manifest(prefix: str = "/root/autodl-tmp") -> dict:
    run = f"{prefix}/runs/ms-gcp-v13/3dgs/gcp_3000_20260602/run_001"
    build = f"{prefix}/build/ms-gcp-v13/3dgs/{HEX40}/run_001"
    return {
        "schema": "ms_gcp_method_run_layout_v1",
        "method_id": "3dgs",
        "scene": "gcp_3000_20260602",
        "run_id": "run_001",
        "code_root": f"{prefix}/worktrees/ms-gcp-v13/3dgs/{HEX40}",
        "code_commit": HEX40,
        "environment_root": f"{prefix}/envs/ms-gcp-v13/3dgs/{HEX64}",
        "environment_lock_sha256": HEX64,
        "dataset_root": f"{prefix}/datasets/ms-gcp-v13/{HEX64}",
        "release_root": f"{prefix}/datasets/ms-gcp-v13/{HEX64}/release",
        "release_root_digest": HEX64,
        "build_root": build,
        "run_root": run,
        "torch_extensions_dir": f"{build}/torch_extensions",
        "temp_root": f"{run}/tmp",
        "output_subdirs": {
            role: f"{run}/{leaf}" for role, leaf in CONTRACT["required_output_subdirs"].items()
        },
        "policies": copy.deepcopy(CONTRACT["required_policy_values"]),
        "env_vars": {
            "PYTHONNOUSERSITE": "1",
            "TORCH_EXTENSIONS_DIR": f"{build}/torch_extensions",
            "TMPDIR": f"{run}/tmp",
        },
    }


def test_valid_posix_layout() -> None:
    assert validate_run_layout(valid_manifest(), CONTRACT) == []


def test_valid_windows_layout() -> None:
    manifest = valid_manifest("E:")
    assert validate_run_layout(manifest, CONTRACT) == []


def test_rejects_dataset_pollution() -> None:
    manifest = valid_manifest()
    manifest["run_root"] = f"{manifest['dataset_root']}/3dgs/gcp_3000_20260602/run_001"
    for role, leaf in CONTRACT["required_output_subdirs"].items():
        manifest["output_subdirs"][role] = f"{manifest['run_root']}/{leaf}"
    manifest["temp_root"] = f"{manifest['run_root']}/tmp"
    manifest["env_vars"]["TMPDIR"] = manifest["temp_root"]
    errors = validate_run_layout(manifest, CONTRACT)
    assert any("inside immutable root" in error for error in errors)


def test_rejects_shared_cuda_cache() -> None:
    manifest = valid_manifest()
    manifest["torch_extensions_dir"] = "/root/.cache/torch_extensions"
    manifest["env_vars"]["TORCH_EXTENSIONS_DIR"] = manifest["torch_extensions_dir"]
    errors = validate_run_layout(manifest, CONTRACT)
    assert any("method- and run-specific" in error for error in errors)
    assert any("inside the method/run build_root" in error for error in errors)


def test_rejects_cross_method_run_root() -> None:
    manifest = valid_manifest()
    manifest["run_root"] = "/root/autodl-tmp/runs/ms-gcp-v13/2dgs/gcp_3000_20260602/run_001"
    for role, leaf in CONTRACT["required_output_subdirs"].items():
        manifest["output_subdirs"][role] = f"{manifest['run_root']}/{leaf}"
    manifest["temp_root"] = f"{manifest['run_root']}/tmp"
    manifest["env_vars"]["TMPDIR"] = manifest["temp_root"]
    errors = validate_run_layout(manifest, CONTRACT)
    assert any("run_root must contain method_id" in error for error in errors)


def test_rejects_output_outside_run_root() -> None:
    manifest = valid_manifest()
    manifest["output_subdirs"]["packets"] = "/tmp/shared_packets/03_packets"
    errors = validate_run_layout(manifest, CONTRACT)
    assert any("output_subdirs.packets" in error for error in errors)


def test_rejects_overwrite_and_global_install_policy() -> None:
    manifest = valid_manifest()
    manifest["policies"]["overwrite_policy"] = "overwrite"
    manifest["policies"]["global_python_install_allowed"] = True
    errors = validate_run_layout(manifest, CONTRACT)
    assert any("overwrite_policy" in error for error in errors)
    assert any("global_python_install_allowed" in error for error in errors)


def main() -> int:
    tests = [
        test_valid_posix_layout,
        test_valid_windows_layout,
        test_rejects_dataset_pollution,
        test_rejects_shared_cuda_cache,
        test_rejects_cross_method_run_root,
        test_rejects_output_outside_run_root,
        test_rejects_overwrite_and_global_install_policy,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS {len(tests)}/{len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
