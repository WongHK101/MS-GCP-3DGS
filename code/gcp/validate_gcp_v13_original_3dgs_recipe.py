#!/usr/bin/env python3
"""Validate the frozen original-3DGS recipe before a formal v1.3 run."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


SCHEMA = "ms_gcp_v1_3_method_recipe_v1"
METHOD_COMMIT = "2eee0e26d2d5fd00ec462df47752223952f6bf4e"
METHOD_TREE = "5eee127dfc0942bf83d9fdd72328e03ec0cbf6c4"
RASTERIZER_COMMIT = "59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d"
KNN_COMMIT = "44f764299fa305faf6ec5ebd99939e0508331503"
RELEASE_DIGEST = "513f8999fe4b110f15bcbecad7932895781cee755ee9ccd7a14ff10298546d75"
ENV_LOCK = "29f8997ba141357bbeddca9014757ab5a97acb9dd5ac312beda9e5f94acce0ed"
SIX_SCENE_SPECS = {
    "gcp_3000_20260602": (94, 875035671, "49cfa412254ff4bfb68473cc1b2262a95362e9542020424c166304a3831962f7", "4929f95d7a68d19820999efc734ff8d892c51508191d0380d00e3ae00b82a5d7", "44f88eabb7e536416ff8bcf211b7c22f1bb6d2ca6eff2731099e771c97ca689f"),
    "gcp_5000_20260602": (101, 859832717, "67aa45289b4d08ec926874efa9babd40187bb64c1ab3e1bced494bad95809905", "eedccc6f18ca59ecdda34f1b89d8d85c446a10f30c936a071f52bf4c0fab65f2", "6f3eaf1f210f17cb93f3846d5fe76f1ff017589bcae36fdc0eb5756b96e05e88"),
    "gcp_10000_20260610": (976, 8810224189, "19afdf4485a35d35a1aeb4e04aa8746807a17510a929e40b30f55e4a16d82a85", "1c26685db1b29f92278d78fbe21ed2b548076c12cdfc1c5e48fe2035048c37c4", "9cf0b7dc47234d8d9cbc8e85fbc9ae41ae376f2147d8c8e8402005a6e9940d22"),
    "gcp_20000_20260602": (298, 3479416169, "a9491e66e8b7315782350ced4a92f70bedbc60898bdd5d4c168d501fe135832c", "61ece0edc96802b8194aaebe8367bad9e7f01e4cf6bb69187797ce95d8e60d5c", "2bdf219c457871211970a2c543a90d2530f7c888656e75550a03518dbec27435"),
    "gcp_50000_20260610": (2208, 22949781257, "57d9ee85b27179425b78515d7cc6211793bf25e5a2b49610b73dcd1daad55420", "50af7d00017b49fc5532ea0c389306be5f0a3be742690849e659bc919d6c4d02", "0d98402f86696b475ff85cb4b4c0cc4c7b30d1dfb417a3d9d976f597dd4628ae"),
    "gcp_100000_20260610": (2510, 25605641533, "b7441b8024bc37d8307ab932cab2a072320ad4a22f3f3eae66c138bc6510d7a4", "2311871ffc339b9a1acce5e833e5b5207b2dd74de48e4e88804bfd0e19a0b7f3", "09fc811f32558a11a47bada7393bf7bce2585cbe68eb4872ffce72025b0fc9aa"),
}


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
    status = recipe.get("status")
    if status not in {"frozen_for_3k_reference_smoke", "frozen_for_six_scene_reference"}:
        errors.append("recipe status is not frozen")

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

    scene_hash_values: list[Any] = []
    if status == "frozen_for_3k_reference_smoke":
        scene = recipe.get("scene", {})
        expected_scene = {
            "scene_id": "gcp_3000_20260602",
            "image_directory": "images",
            "image_count": 94,
            "resolution_argument": 8,
            "camera_model": "PINHOLE",
            "initial_point_count": 61302,
            "data_role": "read_only_training_source_mirror",
        }
        for key, expected in expected_scene.items():
            if scene.get(key) != expected:
                errors.append(f"scene.{key} must equal {expected!r}")
        scene_hash_values.extend([
            scene.get("source_manifest_sha256"),
            scene.get("cameras_bin_sha256"),
            scene.get("images_bin_sha256"),
            scene.get("points3d_bin_sha256"),
        ])
    elif status == "frozen_for_six_scene_reference":
        scenes = recipe.get("scenes")
        if not isinstance(scenes, dict) or set(scenes) != set(SIX_SCENE_SPECS):
            errors.append("six-scene recipe must contain the exact frozen scene set")
            scenes = scenes if isinstance(scenes, dict) else {}
        for scene_id, expected in SIX_SCENE_SPECS.items():
            record = scenes.get(scene_id, {})
            image_count, image_bytes, cameras_sha, images_sha, points_sha = expected
            required = {
                "image_directory": "images",
                "image_count": image_count,
                "image_bytes": image_bytes,
                "resolution_argument": 8,
                "camera_model": "PINHOLE",
                "cameras_bin_sha256": cameras_sha,
                "images_bin_sha256": images_sha,
                "points3d_bin_sha256": points_sha,
            }
            for key, value in required.items():
                if record.get(key) != value:
                    errors.append(f"scenes.{scene_id}.{key} must equal {value!r}")
            scene_hash_values.extend([cameras_sha, images_sha, points_sha])

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
    for forbidden in (
        "gcp_annotations_visible_to_training",
        "gcp_split_visible_to_training",
        "survey_coordinates_visible_to_training",
    ):
        if training.get(forbidden) is not False:
            errors.append(f"training.{forbidden} must be false")

    runtime = recipe.get("runtime", {})
    if runtime.get("environment_lock_sha256") != ENV_LOCK:
        errors.append("runtime environment lock mismatch")
    isolation = recipe.get("isolation", {})
    expected_isolation = {
        "dataset_access": "read_only",
        "release_access": "read_only",
        "code_runtime_writes": "forbidden",
        "overwrite_policy": "fail_if_exists",
        "method_specific_environment": True,
        "method_and_run_specific_build_cache": True,
        "method_and_scene_specific_run_root": True,
        "source_tree_digest_before_after_required": True,
    }
    for key, expected in expected_isolation.items():
        if isolation.get(key) != expected:
            errors.append(f"isolation.{key} must equal {expected!r}")

    roots = recipe.get("server_roots", {})
    mutable = " ".join(str(roots.get(k, "")) for k in ("build_root_template", "run_root_template"))
    dataset_key = "dataset_root_template" if status == "frozen_for_six_scene_reference" else "dataset_root"
    for immutable_key in ("code_root", dataset_key, "release_root"):
        value = str(roots.get(immutable_key, ""))
        if not value.startswith("/root/autodl-tmp/"):
            errors.append(f"server_roots.{immutable_key} is not an isolated absolute path")
        if value and value in mutable:
            errors.append(f"mutable output template overlaps {immutable_key}")
    if "/build/ms-gcp-v13/3dgs-original/" not in str(roots.get("build_root_template", "")):
        errors.append("build root is not method-isolated")
    expected_run_fragment = "/runs/ms-gcp-v13/3dgs-original/<scene>/" if status == "frozen_for_six_scene_reference" else "/runs/ms-gcp-v13/3dgs-original/gcp_3000_20260602/"
    if expected_run_fragment not in str(roots.get("run_root_template", "")):
        errors.append("run root is not method/scene-isolated")
    if status == "frozen_for_six_scene_reference" and "<scene>" not in str(roots.get("build_root_template", "")):
        errors.append("build root is not scene-isolated")

    for value in scene_hash_values:
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            errors.append("scene source hashes must be lowercase SHA-256")

    if official_source is not None:
        try:
            if _git(official_source, "rev-parse", "HEAD") != METHOD_COMMIT:
                errors.append("official source HEAD does not match recipe")
            if _git(official_source, "rev-parse", "HEAD^{tree}") != METHOD_TREE:
                errors.append("official source tree does not match recipe")
            if _git(official_source, "status", "--porcelain"):
                errors.append("official source worktree is dirty")
            rasterizer = official_source / "submodules" / "diff-gaussian-rasterization"
            simple_knn = official_source / "submodules" / "simple-knn"
            if _git(rasterizer, "rev-parse", "HEAD") != RASTERIZER_COMMIT:
                errors.append("official rasterizer checkout mismatch")
            if _git(simple_knn, "rev-parse", "HEAD") != KNN_COMMIT:
                errors.append("official simple-knn checkout mismatch")
            if _git(rasterizer, "status", "--porcelain") or _git(simple_knn, "status", "--porcelain"):
                errors.append("official submodule checkout is dirty")
        except (OSError, subprocess.CalledProcessError) as exc:
            errors.append(f"official source verification failed: {exc}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=Path(__file__).resolve().parents[2] / "configs" / "gcp_v13_original_3dgs_recipe_v1.json")
    parser.add_argument("--official_source", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    repo_root = Path(__file__).resolve().parents[2]
    errors = validate_recipe(recipe, repo_root=repo_root, official_source=args.official_source)
    report = {
        "schema": "ms_gcp_v1_3_method_recipe_validation_v1",
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
