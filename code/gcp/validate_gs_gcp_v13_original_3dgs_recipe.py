#!/usr/bin/env python3
"""Validate the clean official-3DGS R4 qualification recipe."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from materialize_gs_gcp_r4_inputs import RULE_ID, canonical_sha256, validate_contract


SCHEMA = "gs_gcp_v1_3_method_recipe_v3"
STATUS = "pre_registered_for_clean_r4_3k_qualification"
METHOD_COMMIT = "2eee0e26d2d5fd00ec462df47752223952f6bf4e"
METHOD_TREE = "5eee127dfc0942bf83d9fdd72328e03ec0cbf6c4"
RASTERIZER_COMMIT = "59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d"
KNN_COMMIT = "44f764299fa305faf6ec5ebd99939e0508331503"
RELEASE_DIGEST = "513f8999fe4b110f15bcbecad7932895781cee755ee9ccd7a14ff10298546d75"
ENV_LOCK = "29f8997ba141357bbeddca9014757ab5a97acb9dd5ac312beda9e5f94acce0ed"
SPLIT_FILE_SHA = "4535ce1b72dd36a0ba9a46fcf80843bba86b3af1f486ab11fa6d2ca636d1c37e"
SPLIT_CANONICAL_SHA = "823c992c400b625afa126d8f0d9f5e50af129b43f38199662316017e753302b2"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


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


def _expect(mapping: dict[str, Any], expected: dict[str, Any], prefix: str, errors: list[str]) -> None:
    for key, value in expected.items():
        if mapping.get(key) != value:
            errors.append(f"{prefix}{key} must equal {value!r}")


def validate_recipe(recipe: dict[str, Any], *, repo_root: Path, official_source: Path | None = None) -> list[str]:
    errors: list[str] = []
    if recipe.get("schema") != SCHEMA:
        errors.append("unknown recipe schema")
    if recipe.get("status") != STATUS:
        errors.append("recipe must be pre-registered for clean R4 3K qualification")
    if recipe.get("legacy_route_policy") != (
        "old 1600-width, path-backed, serializer-modified recipes, checkpoints, results, and "
        "qualification decisions are historical evidence only and cannot be inherited"
    ):
        errors.append("legacy route revocation is not explicit")

    source = recipe.get("source_provenance", {})
    _expect(
        source,
        {"repository_commit": METHOD_COMMIT, "repository_tree": METHOD_TREE},
        "source_provenance.",
        errors,
    )
    submodules = source.get("submodules", {})
    _expect(
        submodules,
        {"diff-gaussian-rasterization": RASTERIZER_COMMIT, "simple-knn": KNN_COMMIT},
        "source_provenance.submodules.",
        errors,
    )

    build = recipe.get("build_compatibility", {})
    _expect(
        build,
        {"training_source_modified": False, "runtime_training_patch": None, "serializer_patch_allowed": False},
        "build_compatibility.",
        errors,
    )
    patch = build.get("simple_knn_build_copy_patch", {})
    patch_path = repo_root / str(patch.get("patch_path", ""))
    if not patch_path.is_file():
        errors.append("declared simple-knn compatibility patch is missing")
    elif sha256_file(patch_path) != patch.get("patch_sha256"):
        errors.append("simple-knn compatibility patch SHA mismatch")

    release = recipe.get("release", {})
    _expect(
        release,
        {
            "payload_root_digest_sha256": RELEASE_DIGEST,
            "gcp_annotations_or_survey_coordinates_visible_to_training": False,
        },
        "release.",
        errors,
    )

    protocol = recipe.get("protocol", {})
    _expect(
        protocol,
        {
            "resolution_contract": "configs/gs_gcp_quarter_resolution_v1.json",
            "resolution_rule_id": RULE_ID,
            "input_materialization_contract": "configs/gs_gcp_r4_input_materialization_v1.json",
            "split_manifest": "configs/gs_gcp_rgb_holdout_split_manifest_v1.json",
            "split_manifest_file_sha256": SPLIT_FILE_SHA,
            "split_manifest_canonical_sha256": SPLIT_CANONICAL_SHA,
            "holdout_semantics": "image_loss_holdout_under_shared_all_image_sfm_v1",
        },
        "protocol.",
        errors,
    )
    for field in ("input_materialization_contract_sha256", "split_manifest_file_sha256", "split_manifest_canonical_sha256"):
        if not SHA256_RE.fullmatch(str(protocol.get(field, ""))):
            errors.append(f"protocol.{field} must be lowercase SHA-256")

    resolution_path = repo_root / str(protocol.get("resolution_contract", ""))
    if not resolution_path.is_file():
        errors.append("quarter-resolution contract is missing")
    else:
        resolution = json.loads(resolution_path.read_text(encoding="utf-8"))
        _expect(
            resolution,
            {"rule_id": RULE_ID, "reference_method_argument": 4},
            "resolution contract.",
            errors,
        )
    materialization_path = repo_root / str(protocol.get("input_materialization_contract", ""))
    if not materialization_path.is_file():
        errors.append("R4 materialization contract is missing")
    else:
        if sha256_file(materialization_path) != protocol.get("input_materialization_contract_sha256"):
            errors.append("R4 materialization contract SHA mismatch")
        contract = json.loads(materialization_path.read_text(encoding="utf-8"))
        errors.extend(f"R4 materialization contract: {item}" for item in validate_contract(contract, repo_root))
    split_path = repo_root / str(protocol.get("split_manifest", ""))
    if not split_path.is_file():
        errors.append("split manifest is missing")
    else:
        if sha256_file(split_path) != SPLIT_FILE_SHA:
            errors.append("split manifest file SHA mismatch")
        split = json.loads(split_path.read_text(encoding="utf-8"))
        if canonical_sha256(split) != SPLIT_CANONICAL_SHA:
            errors.append("split manifest canonical SHA mismatch")

    scene = recipe.get("qualification_scene", {})
    _expect(
        scene,
        {
            "scene_id": "gcp_3000_20260602",
            "full_view_count": 94,
            "train_view_count": 82,
            "test_view_count": 12,
            "training_input_root_role": "train",
            "heldout_input_root_role": "test",
            "image_directory": "images",
            "image_format": "lossless_rgb_png",
            "loaded_width": 1414,
            "loaded_height": 1024,
            "official_resolution_argument": 1,
            "official_eval_argument": False,
            "camera_model": "PINHOLE",
            "points2d_tracks_present": False,
            "materialized_input_manifest_sha256": "88e354a7cc387975f6686020cf15a3584bfe28769c46360400dcfc027d82921c",
            "materialized_input_file_count": 101,
            "materialized_input_file_bytes": 250576506,
            "source_cameras_bin_sha256": "49cfa412254ff4bfb68473cc1b2262a95362e9542020424c166304a3831962f7",
            "source_images_bin_sha256": "4929f95d7a68d19820999efc734ff8d892c51508191d0380d00e3ae00b82a5d7",
            "shared_initial_ply_sha256": "fc9a8f52de12062bd363b5aa837230833ded1f239feb62accd7d1128f51ae81d",
        },
        "qualification_scene.",
        errors,
    )

    training = recipe.get("training", {})
    _expect(
        training,
        {
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
            "formal_checkpoint": "point_cloud/iteration_30000/point_cloud.ply",
            "gcp_annotations_visible_to_training": False,
            "split_role_labels_visible_to_optimizer": False,
            "survey_coordinates_visible_to_training": False,
        },
        "training.",
        errors,
    )

    execution = recipe.get("execution", {})
    _expect(
        execution,
        {
            "resume_allowed": False,
            "preexisting_checkpoint_allowed": False,
            "source_tree_clean_before_and_after_required": True,
            "materialized_input_verification_required": True,
            "fresh_hardware_manifest_required": True,
        },
        "execution.",
        errors,
    )
    command = str(execution.get("command_template", ""))
    if "--resolution 1" not in command or "/train" not in command:
        errors.append("execution command must bind the train-only root at resolution 1")
    if "--start_checkpoint" in command:
        errors.append("execution command must not resume a checkpoint")

    runtime = recipe.get("runtime", {})
    _expect(runtime, {"pillow": "11.1.0", "environment_lock_sha256": ENV_LOCK}, "runtime.", errors)
    roots = recipe.get("server_roots", {})
    for key in (
        "official_code_root",
        "environment_root",
        "frozen_source_root_template",
        "materialized_input_root_template",
        "build_root_template",
        "run_root_template",
    ):
        value = str(roots.get(key, ""))
        if not value.startswith("/root/autodl-tmp/"):
            errors.append(f"server_roots.{key} is not an isolated absolute path")
    if "/3dgs-original-clean-r4/" not in str(roots.get("run_root_template", "")):
        errors.append("run root does not identify the clean R4 route")

    isolation = recipe.get("isolation", {})
    _expect(
        isolation,
        {
            "source_dataset_access": "read_only",
            "materialized_input_access_during_training": "read_only",
            "code_runtime_writes": "forbidden",
            "overwrite_policy": "fail_if_exists",
            "method_specific_environment": True,
            "method_and_run_specific_build_cache": True,
            "method_and_scene_specific_run_root": True,
        },
        "isolation.",
        errors,
    )
    qualification = recipe.get("qualification", {})
    _expect(
        qualification,
        {
            "external_review_required_before_formal_training": True,
            "full_matrix_authorized_before_pass": False,
            "legacy_qualification_inheritance": False,
        },
        "qualification.",
        errors,
    )

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
        default=Path(__file__).resolve().parents[2] / "configs" / "gs_gcp_v13_original_3dgs_recipe_v3.json",
    )
    parser.add_argument("--official_source", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    repo_root = Path(__file__).resolve().parents[2]
    errors = validate_recipe(recipe, repo_root=repo_root, official_source=args.official_source)
    report = {
        "schema": "gs_gcp_v1_3_method_recipe_validation_v3",
        "recipe": str(args.recipe),
        "recipe_sha256": sha256_file(args.recipe),
        "status": "pass" if not errors else "fail",
        "error_count": len(errors),
        "errors": errors,
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
