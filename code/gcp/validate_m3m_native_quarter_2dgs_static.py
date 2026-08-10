#!/usr/bin/env python3
"""Validate the frozen 2DGS recipe and evaluation-only raw-moment patch set."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


PROTOCOL_ID = "m3m_gcp_native_quarter_geometry_v2"
MAIN_COMMIT = "335ad612f2e783a4e57b9cbc4d1e167bd599fc98"
MAIN_TREE = "ad1da88f43447bde046712835db70e271816282e"
RASTERIZER_COMMIT = "e0ed0207b3e0669960cfad70852200a4a5847f61"
RASTERIZER_TREE = "0be3f326aaa8bf794913a96e1eba6b5b66a9b764"
SIMPLE_KNN_COMMIT = "f155ec04131cb579f53443a06879d37115f4612f"
SIMPLE_KNN_TREE = "6c55d0d2c0941f1e6121bfa13b8c7ff690374d9c"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(path: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), *args],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def validate(repo_root: Path, recipe_path: Path, adapter_path: Path, patched_source: Path) -> dict[str, Any]:
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    adapter = json.loads(adapter_path.read_text(encoding="utf-8"))
    require(recipe.get("schema") == "m3m_gcp_native_quarter_2dgs_recipe_v1", "recipe schema mismatch")
    require(adapter.get("schema") == "m3m_gcp_native_quarter_2dgs_renderer_adapter_v1", "adapter schema mismatch")
    require(recipe.get("protocol_id") == PROTOCOL_ID, "recipe protocol mismatch")
    require(adapter.get("protocol_id") == PROTOCOL_ID, "adapter protocol mismatch")
    require(recipe.get("method", {}).get("method_id") == "2dgs", "method mismatch")
    require(recipe.get("status") == "FROZEN_3K_FORMAL_COMPLETE_RELOCKED", "recipe status mismatch")
    require(recipe.get("execution", {}).get("training_authorized") is False, "completed formal run must be re-locked")
    qualification = recipe.get("qualification", {})
    require(qualification.get("three_k_training_allowed") is False, "completed formal run remains launchable")
    require(qualification.get("formal_3k_completed") is True, "formal completion state missing")
    require(qualification.get("formal_3k_result", {}).get("rerun_allowed") is False, "formal rerun lock missing")
    for key in (
        "gpu_official_training_extension_build_passed",
        "gpu_evaluation_adapter_build_passed",
        "synthetic_raw_moment_conformance_passed",
        "frozen_3k_real_packet_camera_preflight_passed",
    ):
        require(qualification.get(key) is True, f"qualification.{key} did not pass")
    require(qualification.get("full_scene_matrix_allowed") is False, "full matrix must remain locked")
    require(qualification.get("global_training_allowed") is False, "global training must remain locked")

    source = recipe.get("source_provenance", {})
    require(source.get("repository_commit") == MAIN_COMMIT, "main commit mismatch")
    require(source.get("repository_tree") == MAIN_TREE, "main tree mismatch")
    require(source.get("submodules", {}).get("diff-surfel-rasterization", {}).get("commit") == RASTERIZER_COMMIT, "rasterizer commit mismatch")
    require(source.get("submodules", {}).get("diff-surfel-rasterization", {}).get("tree") == RASTERIZER_TREE, "rasterizer tree mismatch")
    require(source.get("submodules", {}).get("simple-knn", {}).get("commit") == SIMPLE_KNN_COMMIT, "simple-knn commit mismatch")
    require(source.get("submodules", {}).get("simple-knn", {}).get("tree") == SIMPLE_KNN_TREE, "simple-knn tree mismatch")

    training = recipe.get("training", {})
    expected_training = {
        "iterations": 30000,
        "seed": 0,
        "resolution": 1,
        "depth_ratio": 0.0,
        "lambda_dist": 0.0,
        "lambda_normal": 0.05,
        "eval_holdout": False,
    }
    for key, expected in expected_training.items():
        require(training.get(key) == expected, f"training.{key} mismatch")
    require(recipe.get("qualification_scene", {}).get("train_view_count") == 82, "train camera count mismatch")
    require(recipe.get("qualification_scene", {}).get("test_view_count") == 12, "held-out camera count mismatch")
    require(recipe.get("qualification_scene", {}).get("loader_preflight", {}).get("status") == "PASS", "loader preflight missing")

    patch_evidence: list[dict[str, Any]] = []
    for spec in adapter.get("patches", []):
        relative = str(spec.get("path", ""))
        path = (repo_root / relative).resolve()
        require(path.is_relative_to(repo_root), f"patch escapes repo: {relative}")
        require(path.is_file(), f"patch missing: {relative}")
        actual = file_sha256(path) if path.is_file() else None
        require(actual == spec.get("sha256"), f"patch SHA mismatch: {relative}")
        patch_evidence.append({"path": relative, "sha256": actual})

    rasterizer = patched_source / "submodules" / "diff-surfel-rasterization"
    simple_knn = patched_source / "submodules" / "simple-knn"
    try:
        require(git(patched_source, "rev-parse", "HEAD") == MAIN_COMMIT, "patched source main commit mismatch")
        require(git(patched_source, "show", "-s", "--format=%T", "HEAD") == MAIN_TREE, "patched source main tree mismatch")
        require(git(rasterizer, "rev-parse", "HEAD") == RASTERIZER_COMMIT, "patched rasterizer commit mismatch")
        require(git(rasterizer, "show", "-s", "--format=%T", "HEAD") == RASTERIZER_TREE, "patched rasterizer tree mismatch")
        require(git(simple_knn, "rev-parse", "HEAD") == SIMPLE_KNN_COMMIT, "patched simple-knn commit mismatch")
        require(git(simple_knn, "show", "-s", "--format=%T", "HEAD") == SIMPLE_KNN_TREE, "patched simple-knn tree mismatch")

        main_modified = set(git(patched_source, "diff", "--name-only").splitlines())
        rasterizer_modified = set(git(rasterizer, "diff", "--name-only").splitlines())
        simple_knn_modified = set(git(simple_knn, "diff", "--name-only").splitlines())
        require(
            main_modified
            == {
                "gaussian_renderer/__init__.py",
                "submodules/diff-surfel-rasterization",
                "submodules/simple-knn",
            },
            f"unexpected main modifications: {sorted(main_modified)}",
        )
        require(
            rasterizer_modified == {
                "cuda_rasterizer/auxiliary.h",
                "cuda_rasterizer/forward.cu",
                "rasterize_points.cu",
            },
            f"unexpected rasterizer modifications: {sorted(rasterizer_modified)}",
        )
        require(simple_knn_modified == {"simple_knn.cu"}, f"unexpected simple-knn modifications: {sorted(simple_knn_modified)}")

        for spec in adapter.get("patches", []):
            apply_root = patched_source if spec.get("apply_root") == "official_repository_root" else rasterizer
            subprocess.check_output(
                ["git", "-C", str(apply_root), "apply", "--check", "--reverse", str((repo_root / spec["path"]).resolve())],
                text=True,
                stderr=subprocess.STDOUT,
            )
        simple_patch = repo_root / recipe["build_compatibility"]["simple_knn_build_copy_patch"]["path"]
        subprocess.check_output(
            ["git", "-C", str(simple_knn), "apply", "--check", "--reverse", str(simple_patch.resolve())],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        errors.append(f"git patch identity check failed: {exc}")

    renderer_text = (patched_source / "gaussian_renderer" / "__init__.py").read_text(encoding="utf-8")
    forward_text = (rasterizer / "cuda_rasterizer" / "forward.cu").read_text(encoding="utf-8")
    rasterize_text = (rasterizer / "rasterize_points.cu").read_text(encoding="utf-8")
    required_tokens = {
        "renderer parameter": "return_raw_metric_depth_accumulators = False",
        "renderer payload": "rets['raw_metric_depth_accumulators'] = torch.cat(",
        "native alpha": "out_others[pix_id + ALPHA_OFFSET * H * W] = 1 - T",
        "native first moment": "D  += depth * w",
        "second moment": "D2 += depth * depth * w",
        "inverse moment": "Hinv += w / depth",
        "nine output planes": "torch::full({3+3+1+2, H, W}",
    }
    bodies = {
        "renderer parameter": renderer_text,
        "renderer payload": renderer_text,
        "native alpha": forward_text,
        "native first moment": forward_text,
        "second moment": forward_text,
        "inverse moment": forward_text,
        "nine output planes": rasterize_text,
    }
    for label, token in required_tokens.items():
        require(token in bodies[label], f"missing {label} token")

    return {
        "schema": "m3m_gcp_native_quarter_2dgs_static_validation_v1",
        "protocol_id": PROTOCOL_ID,
        "method_id": "2dgs",
        "adapter_id": adapter.get("adapter_id"),
        "status": "PASS" if not errors else "FAIL",
        "passed": not errors,
        "formal_training_authorized": False,
        "training_source_modified": False,
        "evaluation_copy_only": True,
        "source_identity": {
            "repository_commit": MAIN_COMMIT,
            "repository_tree": MAIN_TREE,
            "rasterizer_commit": RASTERIZER_COMMIT,
            "rasterizer_tree": RASTERIZER_TREE,
            "simple_knn_commit": SIMPLE_KNN_COMMIT,
            "simple_knn_tree": SIMPLE_KNN_TREE,
        },
        "patches": patch_evidence,
        "primary_common_planes_are_native_2dgs_outputs": True,
        "diagnostic_m2_h_are_evaluation_only": True,
        "remaining_gates": [],
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    repo_default = Path(__file__).resolve().parents[2]
    parser.add_argument("--repo_root", type=Path, default=repo_default)
    parser.add_argument("--recipe", type=Path, default=repo_default / "configs" / "m3m_gcp_native_quarter_2dgs_3k_recipe_v1.json")
    parser.add_argument("--adapter", type=Path, default=repo_default / "configs" / "m3m_gcp_native_quarter_2dgs_renderer_adapter_v1.json")
    parser.add_argument("--patched_source", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = validate(args.repo_root.resolve(), args.recipe.resolve(), args.adapter.resolve(), args.patched_source.resolve())
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
