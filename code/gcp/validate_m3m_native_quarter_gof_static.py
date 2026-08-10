#!/usr/bin/env python3
"""Validate the frozen GOF recipe and evaluation-only raw-moment patch set."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


PROTOCOL_ID = "m3m_gcp_native_quarter_geometry_v2"
MAIN_COMMIT = "5245b20e5d11acd6d1ff5af4b890dc2bedd99693"
MAIN_TREE = "b398209f5721e3944d4f95477c962a78e2106198"
LICENSE_BLOB = "c869e695fa63bfde6f887d63a24a2a71f03480ac"
VENDORED_TREES = {
    "submodules/diff-gaussian-rasterization": "7b75ce386fb5c2f874c9605c90610158126f02a3",
    "submodules/simple-knn": "3468dcff56d59c35d7f0776248bef37f2e138aba",
    "submodules/tetra-triangulation": "06ce0acb28d07f36d6faacb5310b979b840dec8d",
}
CANONICAL_BLOBS = {
    "gaussian_renderer/__init__.py": "d1595831c598e1f54cd2bd06df6ca7073cf3a09d",
    "submodules/diff-gaussian-rasterization/cuda_rasterizer/auxiliary.h": "e8184a0e2a06ec2ba2dbed1ca00d591366512be5",
    "submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu": "50e639019affb67eb7a1b44ea8c6336b23f2815b",
    "submodules/simple-knn/simple_knn.cu": "e72e4c96ea9d161514835fc2fcee62b94954f2d9",
    "train.py": "03c7630bca25db1715d99469ec845e931f95e564",
    "arguments/__init__.py": "65e715b29f71c65b7587579f3d9a77a684b7809e",
}


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
        stderr=subprocess.STDOUT,
    ).strip()


def validate(repo_root: Path, recipe_path: Path, adapter_path: Path, source: Path) -> dict[str, Any]:
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    adapter = json.loads(adapter_path.read_text(encoding="utf-8"))

    require(recipe.get("schema") == "m3m_gcp_native_quarter_gof_recipe_v1", "recipe schema mismatch")
    require(adapter.get("schema") == "m3m_gcp_native_quarter_gof_renderer_adapter_v1", "adapter schema mismatch")
    require(recipe.get("protocol_id") == PROTOCOL_ID, "recipe protocol mismatch")
    require(adapter.get("protocol_id") == PROTOCOL_ID, "adapter protocol mismatch")
    require(recipe.get("method", {}).get("method_id") == "gof", "method mismatch")
    require(recipe.get("status") == "FROZEN_3K_FORMAL_COMPLETE_RELOCKED", "recipe status mismatch")
    require(
        adapter.get("status") == "GPU_BUILD_SYNTHETIC_AND_REAL_3K_PACKET_EVALUATOR_PREFLIGHT_PASS",
        "adapter status mismatch",
    )

    source_pin = recipe.get("source_provenance", {})
    require(source_pin.get("repository_commit") == MAIN_COMMIT, "main commit pin mismatch")
    require(source_pin.get("repository_tree") == MAIN_TREE, "main tree pin mismatch")
    require(source_pin.get("license_git_blob") == LICENSE_BLOB, "license blob pin mismatch")
    for path, expected in VENDORED_TREES.items():
        key = path.rsplit("/", 1)[-1]
        require(source_pin.get("vendored_trees", {}).get(key) == expected, f"recipe vendored tree mismatch: {path}")
        require(adapter.get("source", {}).get("vendored_trees", {}).get(path) == expected, f"adapter vendored tree mismatch: {path}")
    require(adapter.get("source", {}).get("canonical_git_blobs") == CANONICAL_BLOBS, "adapter canonical blob map mismatch")

    input_release = recipe.get("input_release", {})
    require(input_release.get("directory_name") == "M3M-GCP-colmap-native-quarter-v1", "input release mismatch")
    require(
        input_release.get("release_root_digest_sha256")
        == "513f8999fe4b110f15bcbecad7932895781cee755ee9ccd7a14ff10298546d75",
        "input release digest mismatch",
    )
    scene = recipe.get("qualification_scene", {})
    require(scene.get("scene_id") == "gcp_3000_20260602", "qualification scene mismatch")
    require(scene.get("train_view_count") == 82, "train camera count mismatch")
    require(scene.get("test_view_count") == 12, "held-out camera count mismatch")
    require(scene.get("official_resolution_argument") == 1, "resolution argument mismatch")
    require(scene.get("official_eval_argument") is False, "official eval must remain false")

    training = recipe.get("training", {})
    expected_training = {
        "iterations": 30000,
        "seed": 0,
        "resolution": 1,
        "kernel_size": 0.0,
        "eval_holdout": False,
        "ray_jitter": False,
        "resample_gt_image": False,
        "load_allres": False,
        "sample_more_highres": False,
        "use_decoupled_appearance": False,
        "lambda_distortion": 100.0,
        "lambda_depth_normal": 0.05,
        "distortion_from_iter": 15000,
        "depth_normal_from_iter": 15000,
        "densify_until_iter": 15000,
    }
    for key, expected in expected_training.items():
        require(training.get(key) == expected, f"training.{key} mismatch")
    require(training.get("external_geometry_prior") is None, "external prior must be absent")
    for key in ("gcp_annotations_visible_to_training", "split_role_labels_visible_to_optimizer", "survey_coordinates_visible_to_training"):
        require(training.get(key) is False, f"training isolation flag not false: {key}")
    require(training.get("test_iterations") == [7000, 30000], "test iterations mismatch")
    require(training.get("save_iterations") == [7000, 30000], "save iterations mismatch")
    require(training.get("checkpoint_iterations") == [], "checkpoint iterations must be empty")
    require(training.get("start_checkpoint") is None, "start checkpoint must be null")

    execution = recipe.get("execution", {})
    require(execution.get("training_authorized") is False, "completed formal run must be re-locked")
    require(execution.get("resume_allowed") is False, "resume must remain forbidden")
    require(execution.get("preexisting_checkpoint_allowed") is False, "pre-existing checkpoint must remain forbidden")
    command = execution.get("command_template", "")
    for token in (
        "formal_inputs/gcp_3000_20260602/train",
        "--resolution 1",
        "--iterations 30000",
        "--kernel_size 0",
        "--test_iterations 7000 30000",
        "--save_iterations 7000 30000",
    ):
        require(token in command, f"command template missing {token}")
    for forbidden in ("--eval", "--use_decoupled_appearance", "--ray_jitter", "--resample_gt_image", "--sample_more_highres"):
        require(forbidden not in command, f"command template unexpectedly enables {forbidden}")

    qualification = recipe.get("qualification", {})
    require(qualification.get("source_identity_passed") is True, "source identity status missing")
    require(qualification.get("license_review_recorded") is True, "license review status missing")
    require(qualification.get("recipe_static_freeze_passed") is True, "static recipe freeze status missing")
    require(qualification.get("local_patch_replay_passed") is True, "patch replay status missing")
    for key in (
        "real_input_loader_preflight_passed",
        "gpu_official_training_extension_build_passed",
        "gpu_evaluation_adapter_build_passed",
        "synthetic_raw_moment_conformance_passed",
        "frozen_3k_real_packet_camera_preflight_passed",
        "one_iteration_technical_smoke_completed",
    ):
        require(qualification.get(key) is True, f"qualification gate did not pass: {key}")
    require(qualification.get("three_k_training_allowed") is False, "completed formal run remains launchable")
    require(qualification.get("formal_3k_completed") is True, "formal completion state missing")
    require(qualification.get("formal_3k_result", {}).get("rerun_allowed") is False, "formal rerun lock missing")
    require(qualification.get("full_scene_matrix_allowed") is False, "full matrix must remain locked")
    require(qualification.get("global_training_allowed") is False, "global training must remain locked")

    raw_output = adapter.get("raw_output", {})
    require(raw_output.get("rendered_image_plane_indices") == [7, 9, 10, 11], "raw plane mapping mismatch")
    require(raw_output.get("physical_surface_claim") is False, "physical surface claim must be false")
    require("must never substitute" in raw_output.get("native_depth_channel_policy", ""), "native depth exclusion is not frozen")
    training_identity = adapter.get("training_identity", {})
    require(training_identity.get("training_patch_allowed") is False, "training patch unexpectedly allowed")
    require(training_identity.get("training_rasterizer_patch_allowed") is False, "training rasterizer patch unexpectedly allowed")
    require(training_identity.get("checkpoint_mutation_allowed") is False, "checkpoint mutation unexpectedly allowed")
    require(training_identity.get("evaluation_copy_required") is True, "evaluation copy requirement missing")

    patch_evidence: list[dict[str, Any]] = []
    all_patch_specs = list(adapter.get("patches", [])) + [recipe.get("build_compatibility", {}).get("simple_knn_build_copy_patch", {})]
    for spec in all_patch_specs:
        relative = str(spec.get("path", ""))
        patch = (repo_root / relative).resolve()
        require(patch.is_relative_to(repo_root), f"patch escapes repository: {relative}")
        require(patch.is_file(), f"patch missing: {relative}")
        actual = file_sha256(patch) if patch.is_file() else None
        require(actual == spec.get("sha256"), f"patch SHA mismatch: {relative}")
        patch_evidence.append({"path": relative, "sha256": actual})

    try:
        require(git(source, "rev-parse", "HEAD") == MAIN_COMMIT, "source HEAD mismatch")
        require(git(source, "show", "-s", "--format=%T", "HEAD") == MAIN_TREE, "source tree mismatch")
        require(git(source, "status", "--porcelain") == "", "official source is not clean")
        require(git(source, "rev-parse", "HEAD:LICENSE.md") == LICENSE_BLOB, "source license blob mismatch")
        for path, expected in VENDORED_TREES.items():
            require(git(source, "rev-parse", f"HEAD:{path}") == expected, f"source vendored tree mismatch: {path}")
        for path, expected in CANONICAL_BLOBS.items():
            require(git(source, "rev-parse", f"HEAD:{path}") == expected, f"source blob mismatch: {path}")
        for spec in all_patch_specs:
            patch = (repo_root / spec["path"]).resolve()
            subprocess.check_output(
                ["git", "-C", str(source), "apply", "--check", str(patch)],
                text=True,
                stderr=subprocess.STDOUT,
            )
    except (OSError, subprocess.CalledProcessError) as exc:
        errors.append(f"source/patch identity check failed: {exc}")

    train_text = (source / "train.py").read_text(encoding="utf-8")
    args_text = (source / "arguments" / "__init__.py").read_text(encoding="utf-8")
    forward_text = (source / "submodules" / "diff-gaussian-rasterization" / "cuda_rasterizer" / "forward.cu").read_text(encoding="utf-8")
    seed_tokens = ("random.seed(0)", "np.random.seed(0)", "torch.manual_seed(0)")
    for token in seed_tokens:
        require(token in train_text, f"missing frozen seed token: {token}")
    default_tokens = (
        "self.eval = False",
        "self._kernel_size = 0.0",
        "self.ray_jitter = False",
        "self.resample_gt_image = False",
        "self.sample_more_highres = False",
        "self.use_decoupled_appearance = False",
        "self.iterations = 30_000",
        "self.lambda_distortion = 100",
        "self.lambda_depth_normal = 0.05",
        "self.densify_until_iter = 15_000",
    )
    for token in default_tokens:
        require(token in args_text, f"missing frozen upstream default: {token}")
    for token in (
        "float3 ray_point = { ray.x , ray.y, 1.0 }",
        "float t = -BB/(2*AA)",
        "if (alpha < 1.0f / 255.0f)",
        "if (test_T < 0.0001f)",
        "C[CHANNELS * 2 + 1] += alpha * T",
    ):
        require(token in forward_text, f"missing frozen camera-z/compositing token: {token}")
    require("gcp" not in (train_text + args_text).lower(), "GCP-specific token found in official training surface")

    renderer_patch = (repo_root / "patches/gof/native_quarter_raw_moments_renderer_5245b20_v1.patch").read_text(encoding="utf-8")
    rasterizer_patch = (repo_root / "patches/gof/native_quarter_raw_moments_rasterizer_5245b20_v1.patch").read_text(encoding="utf-8")
    simple_patch = (repo_root / "patches/gof/simple_knn_vendored_cuda12_cfloat_v1.patch").read_text(encoding="utf-8")
    patch_tokens = {
        "renderer flag": "return_raw_metric_depth_accumulators=False",
        "renderer payload": "rendered_image[[7, 9, 10, 11], ...]",
        "twelve output planes": "#define OUTPUT_CHANNELS 12",
        "first moment": "weighted_camera_z_sum += metric_weight * t",
        "second moment": "weighted_camera_z_second_moment += metric_weight * t * t",
        "inverse moment": "weighted_inverse_camera_z_sum += metric_weight / t",
        "same official weight": "const float metric_weight = alpha * T",
        "modern CUDA cfloat": "#include <cfloat>",
    }
    patch_bodies = {
        "renderer flag": renderer_patch,
        "renderer payload": renderer_patch,
        "twelve output planes": rasterizer_patch,
        "first moment": rasterizer_patch,
        "second moment": rasterizer_patch,
        "inverse moment": rasterizer_patch,
        "same official weight": rasterizer_patch,
        "modern CUDA cfloat": simple_patch,
    }
    for label, token in patch_tokens.items():
        require(token in patch_bodies[label], f"missing patch token: {label}")

    return {
        "schema": "m3m_gcp_native_quarter_gof_static_validation_v1",
        "protocol_id": PROTOCOL_ID,
        "method_id": "gof",
        "adapter_id": adapter.get("adapter_id"),
        "status": "PASS" if not errors else "FAIL",
        "passed": not errors,
        "formal_training_authorized": False,
        "training_source_modified": False,
        "evaluation_copy_only": True,
        "source_identity": {
            "repository_commit": MAIN_COMMIT,
            "repository_tree": MAIN_TREE,
            "license_git_blob": LICENSE_BLOB,
            "vendored_trees": VENDORED_TREES,
            "canonical_git_blobs": CANONICAL_BLOBS,
        },
        "patches": patch_evidence,
        "common_primary": "A/M1 expected camera-z under official GOF compositing weights",
        "native_depth_channel_used_as_common_primary": False,
        "native_opacity_level_set_mesh_role": "diagnostic_only",
        "physical_surface_claim": False,
        "remaining_gates": [],
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    repo_default = Path(__file__).resolve().parents[2]
    parser.add_argument("--repo_root", type=Path, default=repo_default)
    parser.add_argument("--recipe", type=Path, default=repo_default / "configs" / "m3m_gcp_native_quarter_gof_3k_recipe_v1.json")
    parser.add_argument("--adapter", type=Path, default=repo_default / "configs" / "m3m_gcp_native_quarter_gof_renderer_adapter_v1.json")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = validate(args.repo_root.resolve(), args.recipe.resolve(), args.adapter.resolve(), args.source.resolve())
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
