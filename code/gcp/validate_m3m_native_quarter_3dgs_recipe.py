#!/usr/bin/env python3
"""Validate the frozen native-quarter Original 3DGS 3K recipe."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


SHA256 = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_PROTOCOL = "m3m_gcp_native_quarter_geometry_v2"
EXPECTED_SOURCE = {
    "repository_commit": "2eee0e26d2d5fd00ec462df47752223952f6bf4e",
    "repository_tree": "5eee127dfc0942bf83d9fdd72328e03ec0cbf6c4",
}
EXPECTED_TRAINING = {
    "iterations": 30000,
    "seed": 0,
    "sh_degree": 3,
    "white_background": False,
    "data_device": "cuda",
    "eval_holdout": False,
    "convert_shs_python": False,
    "compute_cov3d_python": False,
    "detect_anomaly": False,
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
}
EXPECTED_RAW_ACCUMULATORS = [
    "accumulated_alpha",
    "weighted_camera_z_sum",
    "weighted_camera_z_second_moment",
    "weighted_inverse_camera_z_sum",
]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_recipe(
    value: dict[str, Any], repo_root: Path, data_root: Path | None = None
) -> dict[str, Any]:
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(value.get("schema") == "m3m_gcp_native_quarter_3dgs_recipe_v1", "unknown schema")
    require(value.get("protocol_id") == EXPECTED_PROTOCOL, "protocol mismatch")
    require(
        value.get("status") == "FROZEN_TRAINING_LOCKED_PENDING_GPU_ADAPTER_PREFLIGHT",
        "recipe status mismatch",
    )
    require(value.get("method", {}).get("method_id") == "3dgs_original", "method mismatch")

    source = value.get("source_provenance", {})
    for key, expected in EXPECTED_SOURCE.items():
        require(source.get(key) == expected, f"source {key} mismatch")
    require(
        source.get("submodules", {}).get("diff-gaussian-rasterization")
        == "59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d",
        "rasterizer submodule mismatch",
    )
    require(
        source.get("submodules", {}).get("simple-knn")
        == "44f764299fa305faf6ec5ebd99939e0508331503",
        "simple-knn submodule mismatch",
    )
    build = value.get("build_compatibility", {})
    require(build.get("training_source_modified") is False, "training source must remain unmodified")
    require(build.get("runtime_training_patch") is None, "runtime training patch is forbidden")
    require(build.get("serializer_patch_allowed") is False, "serializer patch must remain forbidden")

    release = value.get("input_release", {})
    require(release.get("directory_name") == "M3M-GCP-colmap-native-quarter-v1", "input release mismatch")
    require(
        release.get("release_root_digest_sha256")
        == "513f8999fe4b110f15bcbecad7932895781cee755ee9ccd7a14ff10298546d75",
        "input release digest mismatch",
    )
    require(
        release.get("pixel_domain") == "colmap_4_0_4_image_undistorter_pinhole_max_1414",
        "pixel domain mismatch",
    )
    require(bool(SHA256.fullmatch(str(release.get("data_contract_file_sha256", "")))), "data contract SHA invalid")
    require(
        bool(SHA256.fullmatch(str(release.get("formal_input_manifest_file_sha256", "")))),
        "formal input manifest SHA invalid",
    )

    scene = value.get("qualification_scene", {})
    expected_scene = {
        "scene_id": "gcp_3000_20260602",
        "full_view_count": 94,
        "train_view_count": 82,
        "test_view_count": 12,
        "training_input_root_role": "train",
        "heldout_input_root_role": "test",
        "image_directory": "images",
        "image_format": "byte_preserved_colmap_undistorter_jpeg",
        "loaded_width": 1414,
        "loaded_height": 1025,
        "official_resolution_argument": 1,
        "official_eval_argument": False,
        "camera_model": "PINHOLE",
        "points2d_tracks_present": False,
        "points3d_bin_present": False,
        "shared_initial_ply_relative_path": "sparse/0/points3D.ply",
        "shared_initial_ply_sha256": "fc9a8f52de12062bd363b5aa837230833ded1f239feb62accd7d1128f51ae81d",
    }
    for key, expected in expected_scene.items():
        require(scene.get(key) == expected, f"qualification scene {key} mismatch")
    require("camera-view counts" in str(scene.get("count_semantics", "")), "view-count semantics missing")

    training = value.get("training", {})
    for key, expected in EXPECTED_TRAINING.items():
        require(training.get(key) == expected, f"training {key} mismatch")

    execution = value.get("execution", {})
    command = str(execution.get("command_template", ""))
    require("/formal_inputs/gcp_3000_20260602/train" in command, "command does not bind the frozen train root")
    require("--resolution 1" in command, "command does not use native resolution argument 1")
    require("--iterations 30000" in command, "command iteration count mismatch")
    require("--eval" not in command, "official eval split must not be enabled")
    require("/test" not in command, "held-out test root leaked into training command")
    require("clean_r4" not in command.lower() and "r4_clean" not in command.lower(), "legacy clean-R4 input leaked into command")
    require(execution.get("resume_allowed") is False, "resume must remain forbidden")
    require(execution.get("preexisting_checkpoint_allowed") is False, "preexisting checkpoint must remain forbidden")
    require(execution.get("training_authorized") is False, "training must remain locked")

    adapter = value.get("evaluation_adapter", {})
    require(adapter.get("raw_float32_accumulators") == EXPECTED_RAW_ACCUMULATORS, "raw accumulator contract mismatch")
    require(adapter.get("gpu_build_and_real_packet_camera_preflight_passed") is False, "GPU gate was prematurely marked passed")
    evidence_specs = [
        ("config", "config_sha256", "adapter config"),
        ("renderer_patch", "renderer_patch_sha256", "renderer patch"),
        ("rasterizer_patch", "rasterizer_patch_sha256", "rasterizer patch"),
        ("static_patch_validation", "static_patch_validation_sha256", "static patch report"),
        ("cpu_operator_preflight", "cpu_operator_preflight_sha256", "CPU preflight report"),
    ]
    evidence_values: dict[str, dict[str, Any]] = {}
    resolved_repo = repo_root.resolve()
    for path_key, sha_key, label in evidence_specs:
        relative = adapter.get(path_key)
        expected_sha = str(adapter.get(sha_key, ""))
        require(isinstance(relative, str) and bool(relative), f"{label} path missing")
        require(bool(SHA256.fullmatch(expected_sha)), f"{label} SHA invalid")
        if isinstance(relative, str) and relative:
            path = (resolved_repo / relative).resolve()
            require(path.is_relative_to(resolved_repo), f"{label} escapes repository")
            require(path.is_file(), f"{label} missing")
            if path.is_file():
                require(file_sha256(path) == expected_sha, f"{label} SHA mismatch")
                if path.suffix == ".json":
                    evidence_values[path_key] = json.loads(path.read_text(encoding="utf-8"))

    adapter_config = evidence_values.get("config", {})
    require(adapter_config.get("protocol_id") == EXPECTED_PROTOCOL, "adapter config protocol mismatch")
    require(
        adapter_config.get("status") == "STATIC_PATCH_PREFLIGHT_PASS_GPU_RENDER_PREFLIGHT_PENDING",
        "adapter config status mismatch",
    )
    require(
        adapter_config.get("raw_output", {}).get("planes") == EXPECTED_RAW_ACCUMULATORS,
        "adapter config raw planes mismatch",
    )
    static_report = evidence_values.get("static_patch_validation", {})
    require(static_report.get("passed") is True, "static patch validation did not pass")
    require(static_report.get("gpu_render_preflight_passed") is False, "static report claims GPU render pass")
    cpu_report = evidence_values.get("cpu_operator_preflight", {})
    require(cpu_report.get("protocol_id") == EXPECTED_PROTOCOL, "CPU report protocol mismatch")
    require(cpu_report.get("status") == "PASS" and cpu_report.get("passed") is True, "CPU preflight did not pass")

    qualification = value.get("qualification", {})
    require(qualification.get("local_static_patch_preflight_passed") is True, "static preflight state mismatch")
    require(qualification.get("cpu_operator_preflight_passed") is True, "CPU preflight state mismatch")
    require(qualification.get("gpu_renderer_build_preflight_required") is True, "GPU build gate missing")
    require(qualification.get("frozen_3k_real_packet_camera_preflight_required") is True, "real 3K packet gate missing")
    require(qualification.get("three_k_training_allowed") is False, "3K training must remain locked")
    require(qualification.get("full_matrix_authorized_before_pass") is False, "full matrix must remain locked")

    data_verified = data_root is not None
    if data_root is not None:
        resolved_data = data_root.resolve()
        require(resolved_data.name == release.get("directory_name"), "data root directory name mismatch")
        data_specs = [
            (release.get("data_contract"), release.get("data_contract_file_sha256"), "data contract"),
            (release.get("formal_input_manifest"), release.get("formal_input_manifest_file_sha256"), "formal input manifest"),
            (
                "formal_inputs/gcp_3000_20260602/train/sparse/0/cameras.bin",
                scene.get("train_cameras_bin_sha256"),
                "train cameras",
            ),
            (
                "formal_inputs/gcp_3000_20260602/train/sparse/0/images.bin",
                scene.get("train_images_bin_sha256"),
                "train images",
            ),
            (
                "formal_inputs/gcp_3000_20260602/test/sparse/0/images.bin",
                scene.get("test_images_bin_sha256"),
                "test images",
            ),
            (
                "formal_inputs/gcp_3000_20260602/train/sparse/0/points3D.ply",
                scene.get("shared_initial_ply_sha256"),
                "initial PLY",
            ),
        ]
        for relative, expected_sha, label in data_specs:
            path = resolved_data / str(relative)
            require(path.is_file(), f"{label} missing from data root")
            if path.is_file():
                require(file_sha256(path) == expected_sha, f"{label} SHA mismatch in data root")
        manifest_path = resolved_data / str(release.get("formal_input_manifest"))
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            require(
                manifest.get("manifest_sha256") == release.get("formal_input_manifest_canonical_sha256"),
                "formal input manifest canonical SHA mismatch",
            )
            require(
                [manifest.get("full_view_count"), manifest.get("train_view_count"), manifest.get("test_view_count")]
                == [94, 82, 12],
                "formal input manifest view counts mismatch",
            )
            require(
                manifest.get("official_3dgs_binding")
                == {
                    "eval": False,
                    "heldout_root": "test",
                    "official_training_source_modified": False,
                    "resolution_argument": 1,
                    "training_root": "train",
                },
                "formal input manifest 3DGS binding mismatch",
            )

    return {
        "schema": "m3m_gcp_native_quarter_3dgs_recipe_validation_v1",
        "protocol_id": EXPECTED_PROTOCOL,
        "recipe_id": value.get("recipe_id"),
        "passed": not errors,
        "training_allowed": False,
        "data_root_verified": data_verified,
        "static_patch_preflight_passed": static_report.get("passed") is True,
        "cpu_operator_preflight_passed": cpu_report.get("passed") is True,
        "gpu_renderer_build_preflight_passed": False,
        "frozen_3k_real_packet_camera_preflight_passed": False,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--repo_root", type=Path)
    parser.add_argument("--data_root", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    repo_root = (args.repo_root or args.recipe.resolve().parents[1]).resolve()
    result = validate_recipe(
        json.loads(args.recipe.read_text(encoding="utf-8")),
        repo_root,
        args.data_root,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
