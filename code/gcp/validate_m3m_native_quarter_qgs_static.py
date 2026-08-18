#!/usr/bin/env python3
"""Fail-closed static validation for the frozen QGS 3K recipe and one-use gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROTOCOL_ID = "m3m_gcp_native_quarter_geometry_v2"
COMMIT = "74d05c945e99fcaef7afe5a8831903be71ad9b55"
TREE = "c20af6da770b9ecc9c4e1730b40671ea63ec1419"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(repo_root: Path) -> dict[str, Any]:
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    relative_paths = {
        "recipe": "configs/m3m_gcp_native_quarter_qgs_3k_recipe_v1.json",
        "training_yaml": "configs/m3m_gcp_native_quarter_qgs_3k_training_v1.yaml",
        "training_entrypoint": "code/gcp/run_qgs_training.py",
        "exporter": "code/gcp/export_qgs_depth_maps.py",
        "adapter": "configs/m3m_gcp_native_quarter_qgs_renderer_adapter_v1.json",
        "qualification": "docs/protocol_evidence/qgs_native_quarter_gpu_real_3k_qualification_v1.json",
        "truth_deny": "docs/protocol_evidence/qgs_native_quarter_truth_deny_v1.json",
        "gate": "configs/launch_gates/m3m_gcp_native_quarter_qgs_3k_seed0_30k_gate_v1.json",
        "renderer_patch": "patches/qgs/native_quarter_raw_moments_renderer_74d05c9_v1.patch",
        "rasterizer_patch": "patches/qgs/native_quarter_raw_moments_rasterizer_74d05c9_v1.patch",
    }
    paths = {name: repo_root / relative for name, relative in relative_paths.items()}
    for name, path in paths.items():
        require(path.is_file(), f"{name} file missing")
    if errors:
        return {
            "schema": "m3m_gcp_native_quarter_qgs_static_validation_v1",
            "status": "FAIL",
            "passed": False,
            "errors": errors,
        }

    recipe = json.loads(paths["recipe"].read_text(encoding="utf-8"))
    adapter = json.loads(paths["adapter"].read_text(encoding="utf-8"))
    qualification = json.loads(paths["qualification"].read_text(encoding="utf-8"))
    truth = json.loads(paths["truth_deny"].read_text(encoding="utf-8"))
    gate = json.loads(paths["gate"].read_text(encoding="utf-8"))
    yaml_text = paths["training_yaml"].read_text(encoding="utf-8")

    require(recipe.get("schema") == "m3m_gcp_native_quarter_qgs_recipe_v1", "recipe schema mismatch")
    require(recipe.get("protocol_id") == PROTOCOL_ID, "recipe protocol mismatch")
    require(recipe.get("status") == "FROZEN_3K_TRAINING_AUTHORIZED_NOT_STARTED", "recipe status mismatch")
    source = recipe.get("source_provenance", {})
    require(source.get("repository_commit") == COMMIT, "recipe source commit mismatch")
    require(source.get("repository_tree") == TREE, "recipe source tree mismatch")
    require(
        source.get("frozen_linux_license_sha256")
        == "cd5c95b3cfff3acc1bd412420c770f88809331c3db6872df11a970147aa8e81f",
        "frozen Linux license identity mismatch",
    )

    compatibility = recipe.get("build_compatibility", {})
    require(compatibility.get("official_source_freeze_modified") is False, "official source was modified")
    require(compatibility.get("official_training_renderer_modified") is False, "training renderer was modified")
    require(compatibility.get("official_training_rasterizer_modified") is False, "training rasterizer was modified")
    require(compatibility.get("training_source_patch_required") is False, "training source patch was introduced")
    require(compatibility.get("external_training_entrypoint_required") is True, "QGS entrypoint boundary missing")
    require(
        compatibility.get("external_training_entrypoint_sha256") == file_sha256(paths["training_entrypoint"]),
        "training entrypoint SHA mismatch",
    )
    require(
        compatibility.get("formal_training_yaml_sha256") == file_sha256(paths["training_yaml"]),
        "training YAML SHA mismatch",
    )

    input_release = recipe.get("input_release", {})
    require(
        input_release.get("release_root_digest_sha256")
        == "513f8999fe4b110f15bcbecad7932895781cee755ee9ccd7a14ff10298546d75",
        "input release mismatch",
    )
    require(input_release.get("qgs_internal_downsample_or_reencode_allowed") is False, "QGS re-encode path enabled")
    scene = recipe.get("qualification_scene", {})
    require(scene.get("train_view_count") == 82 and scene.get("test_view_count") == 12, "view counts mismatch")
    require(scene.get("loaded_width") == 1414 and scene.get("loaded_height") == 1025, "pixel dimensions mismatch")
    require(scene.get("loader_train_camera_count") == 82 and scene.get("loader_test_camera_count") == 0, "loader split mismatch")

    training = recipe.get("training", {})
    require(training.get("official_budget") == {"type": "iterations", "value": 30000}, "official budget mismatch")
    require(training.get("seed") == 0 and training.get("downsample") == 1.0, "seed or downsample mismatch")
    require(training.get("eval_holdout") is False, "held-out RGB enabled")
    require(training.get("multi_view_num") == 8, "QGS neighbor count mismatch")
    require(training.get("multi_view_max_dis") == 1_000_000_000.0, "scale-dependent neighbor cap restored")
    require(training.get("regularization_from_iter") == 7000, "regularization schedule changed")
    require(training.get("save_iterations") == [30000], "formal checkpoint schedule mismatch")
    require(training.get("checkpoint_iterations") == [] and training.get("start_checkpoint") is None, "resume checkpoint enabled")
    require(training.get("gcp_annotations_visible_to_training") is False, "GCP truth enabled")
    require(training.get("lidar_visible_to_training") is False, "LiDAR enabled")
    command = str(recipe.get("execution", {}).get("command_template", ""))
    for token in ("run_qgs_training.py", "m3m_gcp_native_quarter_qgs_3k_training_v1.yaml", "--save_iterations 30000"):
        require(token in command, f"formal command missing: {token}")
    require("/test" not in command and "release_v1_3_0" not in command, "forbidden input leaked into command")

    for token in (
        "downsample: 1.0",
        "ncc_scale: 1.0",
        "eval: false",
        "iterations: 30000",
        "normal_from_iter: 7000",
        "multi_view_weight_from_iter: 7000",
        "multi_view_max_dis: 1000000000.0",
    ):
        require(token in yaml_text, f"formal YAML missing: {token}")

    expected_recipe_refs = (
        (recipe["evaluation_adapter"]["config"], recipe["evaluation_adapter"]["config_sha256"]),
        (recipe["evaluation_adapter"]["exporter"], recipe["evaluation_adapter"]["exporter_sha256"]),
        (recipe["evaluation_adapter"]["renderer_patch"], recipe["evaluation_adapter"]["renderer_patch_sha256"]),
        (recipe["evaluation_adapter"]["rasterizer_patch"], recipe["evaluation_adapter"]["rasterizer_patch_sha256"]),
        (recipe["qualification"]["qualification_evidence"], recipe["qualification"]["qualification_evidence_sha256"]),
        (recipe["qualification"]["truth_deny_evidence"], recipe["qualification"]["truth_deny_evidence_sha256"]),
    )
    for relative, expected_sha in expected_recipe_refs:
        path = repo_root / relative
        require(path.is_file(), f"referenced file missing: {relative}")
        if path.is_file():
            require(file_sha256(path) == expected_sha, f"referenced file SHA mismatch: {relative}")

    require(adapter.get("protocol_id") == PROTOCOL_ID, "adapter protocol mismatch")
    require(
        adapter.get("status") == "GPU_BUILD_SYNTHETIC_AND_REAL_3K_PACKET_EVALUATOR_PREFLIGHT_PASS",
        "adapter not qualified",
    )
    raw = adapter.get("raw_output", {})
    require(
        raw.get("planes")
        == [
            "accumulated_alpha",
            "weighted_camera_z_sum",
            "weighted_camera_z_second_moment",
            "weighted_inverse_camera_z_sum",
        ],
        "adapter raw planes mismatch",
    )
    require(raw.get("native_channel_mapping") == {
        "accumulated_alpha": 7,
        "weighted_camera_z_sum": 6,
        "weighted_camera_z_second_moment": 8,
        "weighted_inverse_camera_z_sum": 9,
    }, "native channel mapping mismatch")
    require(raw.get("native_quadric_intersection_depth_primary") is False, "native QGS depth promoted to common primary")
    require(raw.get("physical_surface_claim") is False, "adapter makes a physical-surface claim")
    identity = adapter.get("training_identity", {})
    require(identity.get("official_training_rasterizer_channels") == 13, "training channel count changed")
    require(identity.get("evaluation_adapter_rasterizer_channels") == 13, "evaluation channel count changed")

    require(qualification.get("passed") is True and qualification.get("status") == "PASS", "qualification failed")
    require(qualification.get("boundary", {}).get("formal_qgs_training_started") is False, "qualification started formal training")
    require(qualification.get("frozen_input", {}).get("shared_initial_ply_unchanged_after_smoke") is True, "input PLY changed")
    require(qualification.get("frozen_input", {}).get("qgs_cv2_reencode_path_exercised") is False, "qualification re-encoded input")
    require(qualification.get("recipe_basis", {}).get("result_or_truth_used") is False, "recipe used truth/result")
    require(qualification.get("raw_moment_conformance", {}).get("status") == "PASS", "synthetic conformance failed")
    require(qualification.get("raw_moment_conformance", {}).get("pixel_resorting_path") is True, "native resorting path untested")
    require(qualification.get("geometry_branch_smoke", {}).get("camera_count") == 82, "geometry camera count mismatch")
    require(qualification.get("geometry_branch_smoke", {}).get("zero_neighbor_count") == 0, "zero-neighbor qualification")
    require(qualification.get("packet_preflight", {}).get("formal_packet_camera_count") == 66, "real-camera packet count mismatch")
    require(qualification.get("packet_preflight", {}).get("variance_validation_failing_pixel_total") == 0, "variance validation failed")
    require(qualification.get("evaluator", {}).get("status") == "COMPLETE_RANKED", "evaluator preflight failed")
    require(qualification.get("evaluator", {}).get("qualification_only_not_benchmark_result") is True, "qualification score promoted")

    require(truth.get("passed") is True and truth.get("status") == "PASS", "truth deny failed")
    require(truth.get("gcp_annotations_visible_to_training") is False, "truth deny enables GCP")
    require(truth.get("lidar_visible_to_training") is False, "truth deny enables LiDAR")
    require(truth.get("heldout_rgb_visible_to_training") is False, "truth deny enables held-out RGB")
    require(truth.get("result_driven_recipe_selection") is False, "truth deny permits result tuning")

    require(gate.get("status") == "AUTHORIZED_NOT_STARTED", "one-use gate not launchable")
    require(gate.get("method_id") == "qgs" and gate.get("scene") == "gcp_3000_20260602", "gate identity mismatch")
    require(gate.get("seed") == 0 and gate.get("official_budget", {}).get("value") == 30000, "gate seed/budget mismatch")
    for key, path_key in (
        ("recipe_sha256", "recipe"),
        ("training_yaml_sha256", "training_yaml"),
        ("training_entrypoint_sha256", "training_entrypoint"),
        ("adapter_config_sha256", "adapter"),
        ("qualification_report_sha256", "qualification"),
        ("truth_deny_report_sha256", "truth_deny"),
    ):
        require(gate.get(key) == file_sha256(paths[path_key]), f"gate {key} mismatch")
    require(gate.get("single_fresh_run_allowed") is True, "gate does not allow one fresh run")
    require(gate.get("resume_allowed") is False and gate.get("overwrite_allowed") is False, "gate permits resume/overwrite")
    require(gate.get("six_scene_matrix_allowed") is False and gate.get("global_training_allowed") is False, "gate unlocks broader scope")

    return {
        "schema": "m3m_gcp_native_quarter_qgs_static_validation_v1",
        "status": "PASS" if not errors else "FAIL",
        "passed": not errors,
        "protocol_id": PROTOCOL_ID,
        "method_id": "qgs",
        "recipe_sha256": file_sha256(paths["recipe"]),
        "training_yaml_sha256": file_sha256(paths["training_yaml"]),
        "training_entrypoint_sha256": file_sha256(paths["training_entrypoint"]),
        "exporter_sha256": file_sha256(paths["exporter"]),
        "adapter_sha256": file_sha256(paths["adapter"]),
        "qualification_sha256": file_sha256(paths["qualification"]),
        "truth_deny_sha256": file_sha256(paths["truth_deny"]),
        "gate_sha256": file_sha256(paths["gate"]),
        "errors": errors,
    }


def main() -> int:
    default_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=default_root)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(args.repo_root.resolve())
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
