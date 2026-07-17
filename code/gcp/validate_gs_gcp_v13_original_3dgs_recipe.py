#!/usr/bin/env python3
"""Validate the pre-registered original-3DGS 3K GS-GCP recipe."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from gs_gcp_resolution import RULE_ID, validate_contract


SCHEMA = "gs_gcp_v1_3_method_recipe_v2"
METHOD_COMMIT = "2eee0e26d2d5fd00ec462df47752223952f6bf4e"
METHOD_TREE = "5eee127dfc0942bf83d9fdd72328e03ec0cbf6c4"
RASTERIZER_COMMIT = "59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d"
KNN_COMMIT = "44f764299fa305faf6ec5ebd99939e0508331503"
RELEASE_DIGEST = "513f8999fe4b110f15bcbecad7932895781cee755ee9ccd7a14ff10298546d75"
ENV_LOCK = "29f8997ba141357bbeddca9014757ab5a97acb9dd5ac312beda9e5f94acce0ed"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(path: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={path.as_posix()}", "-C", str(path), *args],
        text=True,
    ).strip()


def validate_recipe(recipe: dict[str, Any], *, repo_root: Path, official_source: Path | None = None) -> list[str]:
    errors: list[str] = []
    if recipe.get("schema") != SCHEMA:
        errors.append("unknown recipe schema")
    if recipe.get("status") != "pre_registered_for_3k_qualification":
        errors.append("recipe must be pre-registered for 3K qualification")

    source = recipe.get("source_provenance", {})
    if source.get("repository_commit") != METHOD_COMMIT:
        errors.append("official 3DGS commit mismatch")
    if source.get("repository_tree") != METHOD_TREE:
        errors.append("official 3DGS tree mismatch")
    submodules = source.get("submodules", {})
    if submodules.get("diff-gaussian-rasterization") != RASTERIZER_COMMIT:
        errors.append("rasterizer commit mismatch")
    if submodules.get("simple-knn") != KNN_COMMIT:
        errors.append("simple-knn commit mismatch")

    patch = recipe.get("build_compatibility", {}).get("simple_knn_build_copy_patch", {})
    patch_path = repo_root / str(patch.get("patch_path", ""))
    if not patch_path.is_file():
        errors.append("declared simple-knn compatibility patch is missing")
    elif sha256_file(patch_path) != patch.get("patch_sha256"):
        errors.append("simple-knn compatibility patch SHA mismatch")
    if recipe.get("build_compatibility", {}).get("training_source_modified") is not False:
        errors.append("training source must remain unmodified")

    release = recipe.get("release", {})
    if release.get("payload_root_digest_sha256") != RELEASE_DIGEST:
        errors.append("release root digest mismatch")
    if release.get("training_inputs_used_for_selection_or_loss") is not False:
        errors.append("release GCP data must not influence training")

    scene = recipe.get("scene", {})
    expected_scene = {
        "scene_id": "gcp_3000_20260602",
        "image_directory": "images",
        "image_count": 94,
        "resolution_argument": -1,
        "resolution_contract": "configs/gs_gcp_training_resolution_v1.json",
        "resolution_rule_id": RULE_ID,
        "loaded_dimension_manifest_required": True,
        "loaded_tensor_hash_probe_required": True,
        "camera_model": "PINHOLE",
        "initial_point_count": 61302,
        "data_role": "read_only_training_source_mirror",
    }
    for key, expected in expected_scene.items():
        if scene.get(key) != expected:
            errors.append(f"scene.{key} must equal {expected!r}")
    for name in ("source_manifest_sha256", "cameras_bin_sha256", "images_bin_sha256", "points3d_bin_sha256"):
        if not isinstance(scene.get(name), str) or not re.fullmatch(r"[0-9a-f]{64}", scene[name]):
            errors.append(f"scene.{name} must be lowercase SHA-256")

    resolution_contract_path = repo_root / str(scene.get("resolution_contract", ""))
    if not resolution_contract_path.is_file():
        errors.append("resolution contract is missing")
    else:
        resolution_contract = json.loads(resolution_contract_path.read_text(encoding="utf-8"))
        errors.extend(f"resolution contract: {item}" for item in validate_contract(resolution_contract))

    training = recipe.get("training", {})
    expected_training = {
        "iterations": 30000,
        "seed": 0,
        "sh_degree": 3,
        "white_background": False,
        "data_device": "cuda",
        "eval_holdout": False,
        "convert_shs_python": False,
        "compute_cov3d_python": False,
        "position_lr_init": 0.00016,
        "position_lr_final": 0.0000016,
        "position_lr_delay_mult": 0.01,
        "position_lr_max_steps": 30000,
        "feature_lr": 0.0025,
        "opacity_lr": 0.05,
        "scaling_lr": 0.005,
        "rotation_lr": 0.001,
        "percent_dense": 0.01,
        "lambda_dssim": 0.2,
        "densification_interval": 100,
        "opacity_reset_interval": 3000,
        "densify_from_iter": 500,
        "densify_until_iter": 15000,
        "densify_grad_threshold": 0.0002,
        "random_background": False,
        "test_iterations": [7000, 30000],
        "save_iterations": [7000, 30000],
        "checkpoint_iterations": [],
        "start_checkpoint": None,
    }
    for key, expected in expected_training.items():
        if training.get(key) != expected:
            errors.append(f"training.{key} must equal {expected!r}")
    for field in (
        "gcp_annotations_visible_to_training",
        "gcp_split_visible_to_training",
        "survey_coordinates_visible_to_training",
    ):
        if training.get(field) is not False:
            errors.append(f"training.{field} must be false")

    if recipe.get("runtime", {}).get("environment_lock_sha256") != ENV_LOCK:
        errors.append("runtime environment lock mismatch")
    isolation = recipe.get("isolation", {})
    for key, expected in {
        "dataset_access": "read_only",
        "release_access": "read_only",
        "code_runtime_writes": "forbidden",
        "overwrite_policy": "fail_if_exists",
        "method_specific_environment": True,
        "method_and_run_specific_build_cache": True,
        "method_and_scene_specific_run_root": True,
        "source_tree_digest_before_after_required": True,
    }.items():
        if isolation.get(key) != expected:
            errors.append(f"isolation.{key} must equal {expected!r}")

    roots = recipe.get("server_roots", {})
    for key in ("code_root", "dataset_root", "release_root", "build_root_template", "run_root_template"):
        value = str(roots.get(key, ""))
        if not value.startswith("/root/autodl-tmp/"):
            errors.append(f"server_roots.{key} is not an isolated absolute path")
        if "/gs-gcp-v13/" not in value:
            errors.append(f"server_roots.{key} must use the gs-gcp-v13 namespace")
    if "/build/gs-gcp-v13/3dgs-original/" not in str(roots.get("build_root_template", "")):
        errors.append("build root is not method-isolated")
    if "/runs/gs-gcp-v13/3dgs-original/gcp_3000_20260602/" not in str(roots.get("run_root_template", "")):
        errors.append("run root is not method/scene-isolated")

    if official_source is not None:
        try:
            if _git(official_source, "rev-parse", "HEAD") != METHOD_COMMIT:
                errors.append("official source HEAD does not match recipe")
            if _git(official_source, "rev-parse", "HEAD^{tree}") != METHOD_TREE:
                errors.append("official source tree does not match recipe")
            if _git(official_source, "status", "--porcelain"):
                errors.append("official source worktree is dirty")
        except (OSError, subprocess.CalledProcessError) as exc:
            errors.append(f"official source verification failed: {exc}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recipe",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "configs" / "gs_gcp_v13_original_3dgs_recipe_v2.json",
    )
    parser.add_argument("--official_source", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    repo_root = Path(__file__).resolve().parents[2]
    errors = validate_recipe(recipe, repo_root=repo_root, official_source=args.official_source)
    report = {
        "schema": "gs_gcp_v1_3_method_recipe_validation_v2",
        "recipe": str(args.recipe),
        "recipe_sha256": sha256_file(args.recipe),
        "status": "pass" if not errors else "fail",
        "error_count": len(errors),
        "errors": errors,
    }
    text = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
