#!/usr/bin/env python3
"""Fail-closed static validation for the frozen RaDe-GS 3K recipe and one-use gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROTOCOL_ID = "m3m_gcp_native_quarter_geometry_v2"
COMMIT = "d72f20792005ae1d6555a82aa2d15345f247604e"
TREE = "e37a9f1bfec5b593371402d19fb5259cbcb6efa1"


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

    recipe_path = repo_root / "configs/m3m_gcp_native_quarter_rade_gs_3k_recipe_v1.json"
    adapter_path = repo_root / "configs/m3m_gcp_native_quarter_rade_gs_renderer_adapter_v1.json"
    qualification_path = repo_root / "docs/protocol_evidence/rade_gs_native_quarter_gpu_real_3k_qualification_v1.json"
    truth_path = repo_root / "docs/protocol_evidence/rade_gs_native_quarter_truth_deny_v1.json"
    gate_path = repo_root / "configs/launch_gates/m3m_gcp_native_quarter_rade_gs_3k_seed0_30k_gate_v1.json"
    for path, label in (
        (recipe_path, "recipe"),
        (adapter_path, "adapter"),
        (qualification_path, "qualification"),
        (truth_path, "truth deny"),
        (gate_path, "one-use gate"),
    ):
        require(path.is_file(), f"{label} file missing")
    if errors:
        return {"schema": "m3m_gcp_native_quarter_rade_gs_static_validation_v1", "status": "FAIL", "passed": False, "errors": errors}

    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    adapter = json.loads(adapter_path.read_text(encoding="utf-8"))
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    gate = json.loads(gate_path.read_text(encoding="utf-8"))

    require(recipe.get("schema") == "m3m_gcp_native_quarter_rade_gs_recipe_v1", "recipe schema mismatch")
    require(recipe.get("protocol_id") == PROTOCOL_ID, "recipe protocol mismatch")
    require(recipe.get("status") == "FROZEN_3K_TRAINING_AUTHORIZED_NOT_STARTED", "recipe status mismatch")
    source = recipe.get("source_provenance", {})
    require(source.get("repository_commit") == COMMIT, "recipe source commit mismatch")
    require(source.get("repository_tree") == TREE, "recipe source tree mismatch")
    require(
        source.get("frozen_linux_license_sha256") == "cd5c95b3cfff3acc1bd412420c770f88809331c3db6872df11a970147aa8e81f",
        "frozen Linux license identity mismatch",
    )
    compatibility = recipe.get("build_compatibility", {})
    require(compatibility.get("official_source_freeze_modified") is False, "official source was modified")
    require(compatibility.get("official_training_renderer_modified") is False, "training renderer was modified")
    require(compatibility.get("official_training_rasterizer_modified") is False, "training rasterizer was modified")
    require(compatibility.get("training_source_patch_required") is False, "training source patch was introduced")
    input_release = recipe.get("input_release", {})
    require(
        input_release.get("release_root_digest_sha256")
        == "513f8999fe4b110f15bcbecad7932895781cee755ee9ccd7a14ff10298546d75",
        "input release mismatch",
    )
    scene = recipe.get("qualification_scene", {})
    require(scene.get("train_view_count") == 82 and scene.get("test_view_count") == 12, "view counts mismatch")
    require(scene.get("loaded_width") == 1414 and scene.get("loaded_height") == 1025, "pixel dimensions mismatch")
    require(scene.get("loader_train_camera_count") == 82 and scene.get("loader_test_camera_count") == 0, "loader split mismatch")
    training = recipe.get("training", {})
    require(training.get("official_budget") == {"type": "iterations", "value": 30000}, "official budget mismatch")
    require(training.get("seed") == 0 and training.get("resolution") == 1, "seed or resolution mismatch")
    require(training.get("eval_holdout") is False, "held-out RGB enabled")
    require(training.get("use_decoupled_appearance") == 3, "official outdoor appearance mode changed")
    require(training.get("kernel_size") == 0.0 and training.get("depth_ratio") == 0.0, "native depth recipe changed")
    require(training.get("multi_view_num") == 8, "RaDe-GS neighbor count mismatch")
    require(training.get("multi_view_max_dis") == 1_000_000_000.0, "scale-dependent neighbor cap restored")
    require(training.get("neighbor_validation", {}).get("zero_neighbor_count") == 0, "zero-neighbor recipe")
    require(training.get("regularization_from_iter") == 7000, "regularization schedule changed")
    require(training.get("gcp_annotations_visible_to_training") is False, "GCP truth enabled")
    require(training.get("lidar_visible_to_training") is False, "LiDAR enabled")
    command = str(recipe.get("execution", {}).get("command_template", ""))
    for token in ("--resolution 1", "--iterations 30000", "--use_decoupled_appearance 3", "--multi_view_max_dis 1000000000"):
        require(token in command, f"formal command missing: {token}")
    require("/test" not in command and "release_v1_3_0" not in command, "forbidden input leaked into command")

    expected_refs = (
        (recipe["evaluation_adapter"]["config"], recipe["evaluation_adapter"]["config_sha256"]),
        (recipe["qualification"]["qualification_evidence"], recipe["qualification"]["qualification_evidence_sha256"]),
    )
    for relative, expected_sha in expected_refs:
        path = repo_root / relative
        require(path.is_file(), f"referenced file missing: {relative}")
        if path.is_file():
            require(file_sha256(path) == expected_sha, f"referenced file SHA mismatch: {relative}")
    for key in ("renderer_patch", "rasterizer_patch"):
        relative = recipe.get("evaluation_adapter", {}).get(key, "")
        expected_sha = recipe.get("evaluation_adapter", {}).get(f"{key}_sha256")
        path = repo_root / relative
        require(path.is_file(), f"adapter patch missing: {relative}")
        if path.is_file():
            require(file_sha256(path) == expected_sha, f"adapter patch SHA mismatch: {relative}")

    require(adapter.get("protocol_id") == PROTOCOL_ID, "adapter protocol mismatch")
    require(adapter.get("status") == "GPU_BUILD_SYNTHETIC_AND_REAL_3K_PACKET_EVALUATOR_PREFLIGHT_PASS", "adapter not qualified")
    require(adapter.get("raw_output", {}).get("planes") == [
        "accumulated_alpha",
        "weighted_camera_z_sum",
        "weighted_camera_z_second_moment",
        "weighted_inverse_camera_z_sum",
    ], "adapter raw planes mismatch")
    require(adapter.get("raw_output", {}).get("native_plane_depth_primary") is False, "native RaDe-GS depth promoted to common primary")
    require(adapter.get("raw_output", {}).get("physical_surface_claim") is False, "adapter makes a physical-surface claim")
    require(adapter.get("training_identity", {}).get("official_training_rasterizer_channels") == 3, "training rasterizer changed")
    require(adapter.get("training_identity", {}).get("evaluation_adapter_rasterizer_channels") == 6, "evaluation channel count mismatch")

    require(qualification.get("passed") is True and qualification.get("status") == "PASS", "qualification failed")
    require(qualification.get("boundary", {}).get("formal_rade_gs_training_started") is False, "qualification started formal training")
    require(qualification.get("frozen_input", {}).get("shared_initial_ply_unchanged_after_smoke") is True, "input PLY changed")
    require(qualification.get("recipe_basis", {}).get("result_or_truth_used") is False, "recipe used truth/result")
    require(qualification.get("raw_moment_conformance", {}).get("status") == "PASS", "synthetic conformance failed")
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
    require(gate.get("method_id") == "rade_gs" and gate.get("scene") == "gcp_3000_20260602", "gate identity mismatch")
    require(gate.get("seed") == 0 and gate.get("official_budget", {}).get("value") == 30000, "gate seed/budget mismatch")
    require(gate.get("recipe_sha256") == file_sha256(recipe_path), "gate recipe SHA mismatch")
    require(gate.get("adapter_config_sha256") == file_sha256(adapter_path), "gate adapter SHA mismatch")
    require(gate.get("qualification_report_sha256") == file_sha256(qualification_path), "gate qualification SHA mismatch")
    require(gate.get("truth_deny_report_sha256") == file_sha256(truth_path), "gate truth-deny SHA mismatch")
    require(gate.get("single_fresh_run_allowed") is True, "gate does not allow one fresh run")
    require(gate.get("resume_allowed") is False and gate.get("overwrite_allowed") is False, "gate permits resume/overwrite")
    require(gate.get("six_scene_matrix_allowed") is False and gate.get("global_training_allowed") is False, "gate unlocks broader scope")

    return {
        "schema": "m3m_gcp_native_quarter_rade_gs_static_validation_v1",
        "status": "PASS" if not errors else "FAIL",
        "passed": not errors,
        "protocol_id": PROTOCOL_ID,
        "method_id": "rade_gs",
        "recipe_sha256": file_sha256(recipe_path),
        "adapter_sha256": file_sha256(adapter_path),
        "qualification_sha256": file_sha256(qualification_path),
        "truth_deny_sha256": file_sha256(truth_path),
        "gate_sha256": file_sha256(gate_path),
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
