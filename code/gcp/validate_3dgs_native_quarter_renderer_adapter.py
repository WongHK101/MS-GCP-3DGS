#!/usr/bin/env python3
"""Validate the evaluation-only raw-moment patch on frozen official 3DGS."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from m3m_native_quarter_protocol import PROTOCOL_ID, sha256_file


def git(path: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), *args],
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def validate(repo_root: Path, adapter_path: Path, patched_source: Path) -> dict[str, Any]:
    adapter = json.loads(adapter_path.read_text(encoding="utf-8"))
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(
        adapter.get("schema") == "m3m_gcp_native_quarter_3dgs_renderer_adapter_v1",
        "adapter schema mismatch",
    )
    require(adapter.get("protocol_id") == PROTOCOL_ID, "adapter protocol id mismatch")
    source = adapter.get("source", {})
    submodule = patched_source / "submodules" / "diff-gaussian-rasterization"
    try:
        require(git(patched_source, "rev-parse", "HEAD") == source.get("repository_commit"), "source commit mismatch")
        require(git(patched_source, "show", "-s", "--format=%T", "HEAD") == source.get("repository_tree"), "source tree mismatch")
        require(git(submodule, "rev-parse", "HEAD") == source.get("rasterizer_commit"), "rasterizer commit mismatch")
    except Exception as exc:
        errors.append(f"git identity check failed: {exc}")

    patch_evidence = []
    for entry in adapter.get("patches", []):
        patch_path = repo_root / str(entry.get("path", ""))
        require(patch_path.is_file(), f"adapter patch missing: {patch_path}")
        if patch_path.is_file():
            actual_sha = sha256_file(patch_path)
            require(actual_sha == entry.get("sha256"), f"adapter patch SHA mismatch: {patch_path}")
            apply_root = patched_source if entry.get("apply_root") == "official_repository_root" else submodule
            reverse_check = subprocess.run(
                ["git", "-C", str(apply_root), "apply", "--check", "--reverse", str(patch_path)],
                text=True,
                capture_output=True,
                check=False,
            )
            require(reverse_check.returncode == 0, f"patch is not exactly applied: {patch_path}")
            patch_evidence.append(
                {
                    "path": entry.get("path"),
                    "sha256": actual_sha,
                    "reverse_apply_check_passed": reverse_check.returncode == 0,
                }
            )

    file_evidence = []
    for relative, expected_sha in adapter.get("patched_file_sha256", {}).items():
        path = patched_source / relative
        require(path.is_file(), f"patched source file missing: {relative}")
        actual_sha = sha256_file(path) if path.is_file() else ""
        require(actual_sha == expected_sha, f"patched source SHA mismatch: {relative}")
        file_evidence.append({"path": relative, "sha256": actual_sha})

    renderer_text = (patched_source / "gaussian_renderer" / "__init__.py").read_text(encoding="utf-8")
    rasterizer_text = (submodule / "cuda_rasterizer" / "forward.cu").read_text(encoding="utf-8")
    required_renderer_tokens = (
        "return_raw_metric_depth_accumulators",
        'output["raw_metric_depth_accumulators"]',
    )
    required_rasterizer_tokens = (
        "accumulated_alpha += weight",
        "weighted_camera_z_sum += weight * camera_z",
        "weighted_camera_z_second_moment += weight * camera_z * camera_z",
        "weighted_inverse_camera_z_sum += weight / camera_z",
    )
    for token in required_renderer_tokens:
        require(token in renderer_text, f"renderer API token missing: {token}")
    for token in required_rasterizer_tokens:
        require(token in rasterizer_text, f"rasterizer accumulator token missing: {token}")

    return {
        "schema": "m3m_gcp_native_quarter_3dgs_renderer_adapter_static_validation_v1",
        "protocol_id": PROTOCOL_ID,
        "adapter_id": adapter.get("adapter_id"),
        "passed": not errors,
        "status": "STATIC_PATCH_PREFLIGHT_PASS_GPU_RENDER_PREFLIGHT_PENDING" if not errors else "FAIL",
        "source_commit": source.get("repository_commit"),
        "rasterizer_commit": source.get("rasterizer_commit"),
        "patches": patch_evidence,
        "patched_files": file_evidence,
        "training_source_modified": False,
        "evaluation_copy_only": True,
        "gpu_render_preflight_passed": False,
        "remaining_gate": adapter.get("remaining_gate"),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--adapter",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "configs"
        / "m3m_gcp_native_quarter_3dgs_renderer_adapter_v1.json",
    )
    parser.add_argument("--patched_source", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = validate(args.repo_root.resolve(), args.adapter.resolve(), args.patched_source.resolve())
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
