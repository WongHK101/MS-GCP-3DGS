#!/usr/bin/env python3
"""Build the exact post-attempt 100K RGB/GCP/LiDAR review candidate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from m3m_gcp_lidar_artifacts import (
    canonical_sha256,
    sha256_file,
    validate_scene_attempt_freeze,
)
from metric_depth_packet import directory_tree_hash
from run_m3m_gcp_100k_guarded import validate_model_identity_bundle


SCENE = "gcp_100000_20260610"
BASE_COMMIT = "04f453dbf0d438addaa087b1402f7b1acdfc987d"
BASE_TREE = "f84727f89620e8679049863d1bdbf6d8aaf2c491"
BASE_ACTIVATION_SHA = "72e2715011f6b4e170bed2e7a40d4f5507bebb6f5c0a68894aecfd42a83c3d0e"
BASE_ACTIVATION_CANONICAL_SHA = "8c03e0131c2ef6ee4dba17906ad4d38232711e93222a574321c485e2228ac140"
BASE_PLAN = "configs/m3m_gcp_native_quarter_100k_ten_method_execution_plan_v4.json"
BASE_PLAN_SHA = "c8b15b9ec12e798dcae11fb8636d5944b1a5fda99c43ef73ad9d3e2454b72ba1"
BASE_RECIPE_MANIFEST = "configs/m3m_gcp_native_quarter_100k_recipe_manifest_v3.json"
BASE_RECIPE_MANIFEST_SHA = "0789f8d8f5a145ab8c531c0a5b34d211bc7e7c2c5018552c48ff5687c37dc4d2"
ADDENDUM_CONFIG = "configs/m3m_gcp_native_quarter_100k_three_track_evaluation_addendum_v1.json"
FORMAL_INPUT_SHA = "c2cf9e951d95fee12a28d942e95c5c420df55bc364738b3f8737fed1c78bef3d"
FORMAL_INPUT_CANONICAL_SHA = "5b4fe34743310bd2225feb2dd236200606be933002fec19d2c9ecb9f3ba6769d"
GCP_PROTOCOL_ID = "m3m_gcp_native_quarter_geometry_v2"
LIDAR_PROTOCOL_ID = "m3m_gcp_lidar_rendered_surface_v1"
RGB_SUITE_ID = "m3m_gcp_native_quarter_rgb_quality_v1"
COMMON_SIM3_SHA = "f2bdfe649891f666371db64d9b504aee49bb1312fde33801408d72bea6def000"
GCP_PROTOCOL_RELEASE_SHA = "21fbac75d66433169535ea7440c31393f7a5ecdb4ed94fcefd31d1780c28bea4"
GCP_DATA_CONTRACT_SHA = "9141cf90e5bcdf342e5d47e58aa3a0aa48300bd461411ae495ce974993e5ed13"
OBSERVATION_SEMANTICS_SHA = "92c28d8c64ff9b9659bfc9ea3b62f8b80d96641cd0c0d1c671eaeb611adcf945"
REVIEW_TASK_ID = "019ff12c-cb29-7cb2-8fb6-1d82c5f8c54b"
REQUIRED_REVIEW_VERDICT = "PASS_100K_THREE_TRACK_EVALUATION_ADDENDUM_V1"
READY_METHODS_SUPPORTED = {"3dgs_original", "citygs_x", "metrogs"}

LEGACY_FILES = {
    "prelaunch": "evaluation_prelaunch.json",
    "allowlist": "allowlist.csv",
    "evaluation_subset_manifest": "evaluation_subset_manifest_pre_colmap_compat.json",
    "packet_manifest": "packets/depth_export_manifest.json",
    "evaluation_summary": "evaluator/evaluation_summary.json",
    "evaluator_manifest": "evaluator/evaluator_manifest.json",
    "point_results": "evaluator/point_results.csv",
    "observation_samples": "evaluator/observation_samples.csv",
    "independent_verification": "evaluator_output_verification.json",
}


def git_value(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, encoding="utf-8"
    ).strip()


def write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(
            descriptor,
            (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
    finally:
        os.close(descriptor)


def require_file(path: Path, expected_sha: str | None = None) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if expected_sha is not None and sha256_file(path) != expected_sha:
        raise RuntimeError(f"file SHA mismatch: {path}")
    return path


def require_clean_checkout(repo: Path, *, expected_commit: str | None = None) -> tuple[str, str]:
    repo = repo.resolve()
    if git_value(repo, "status", "--porcelain"):
        raise RuntimeError(f"checkout is dirty: {repo}")
    commit = git_value(repo, "rev-parse", "HEAD")
    tree = git_value(repo, "show", "-s", "--format=%T", "HEAD")
    if expected_commit is not None and commit != expected_commit:
        raise RuntimeError(f"checkout commit mismatch: {repo}: {commit}")
    return commit, tree


def validate_base_activation(path: Path) -> dict[str, Any]:
    require_file(path, BASE_ACTIVATION_SHA)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != "m3m_gcp_lidar_formal_activation_v1"
        or payload.get("execution_authorized") is not True
        or payload.get("benchmark_commit") != BASE_COMMIT
        or payload.get("benchmark_tree") != BASE_TREE
        or payload.get("canonical_sha256") != BASE_ACTIVATION_CANONICAL_SHA
        or canonical_sha256(payload) != BASE_ACTIVATION_CANONICAL_SHA
    ):
        raise RuntimeError("base activation identity mismatch")
    return payload


def validate_formal_input(path: Path) -> dict[str, Any]:
    require_file(path, FORMAL_INPUT_SHA)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("scene") != SCENE
        or payload.get("manifest_sha256") != FORMAL_INPUT_CANONICAL_SHA
        or payload.get("full_view_count") != 2510
        or payload.get("train_view_count") != 2196
        or payload.get("test_view_count") != 314
    ):
        raise RuntimeError("formal input identity mismatch")
    rows = payload.get("images", [])
    if len(rows) != 2510:
        raise RuntimeError("formal input image inventory mismatch")
    if sum(row.get("role") == "train" for row in rows) != 2196:
        raise RuntimeError("formal train-role inventory mismatch")
    if sum(row.get("role") == "test" for row in rows) != 314:
        raise RuntimeError("formal test-role inventory mismatch")
    return payload


def validate_rgb_camera_root(path: Path, *, formal_input: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    require_file(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != "m3m_gcp_100k_rgb_evaluation_camera_root_v1"
        or payload.get("status") != "PASS_RGB_EVALUATION_CAMERA_ROOT"
        or payload.get("scene") != SCENE
        or payload.get("canonical_sha256") != canonical_sha256(payload)
        or payload.get("formal_manifest", {}).get("sha256") != FORMAL_INPUT_SHA
        or payload.get("formal_manifest", {}).get("canonical_sha256") != FORMAL_INPUT_CANONICAL_SHA
        or payload.get("output", {}).get("view_count") != 314
        or payload.get("truth_boundary", {}).get("training_or_prior_use_forbidden") is not True
    ):
        raise RuntimeError("RGB camera-root evidence mismatch")
    root = Path(str(payload.get("output", {}).get("root", ""))).resolve()
    if path.resolve() != root / "RGB_EVALUATION_CAMERA_ROOT_MANIFEST.json":
        raise RuntimeError("RGB camera-root manifest path mismatch")
    if Path(str(payload.get("output", {}).get("images_symlink_target", ""))).resolve() != (
        Path(str(payload.get("formal_scene_root", ""))).resolve() / "test" / "images"
    ):
        raise RuntimeError("RGB camera-root image target mismatch")
    files = payload.get("output", {}).get("files", {})
    for name in ("cameras.bin", "images.bin", "points3D.bin", "points3D.ply"):
        row = files.get(name, {})
        file_path = root / "sparse" / "0" / name
        require_file(file_path, str(row.get("sha256", "")))
        if row.get("path") != str(file_path) or row.get("bytes") != file_path.stat().st_size:
            raise RuntimeError(f"RGB camera-root file inventory mismatch: {name}")
    test_names = {str(row["image_name"]) for row in formal_input["images"] if row.get("role") == "test"}
    actual_names = {item.name for item in (root / "images").iterdir() if item.is_file()}
    if actual_names != test_names:
        raise RuntimeError("RGB camera-root heldout names mismatch")
    return payload, root


def validate_gcp_camera_root(path: Path, *, formal_input: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    require_file(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    output = payload.get("output", {})
    observations = payload.get("protocol_observations", {})
    if (
        payload.get("schema") != "m3m_gcp_100k_gcp_evaluation_camera_root_v1"
        or payload.get("status")
        != "PASS_GCP_EVALUATION_CAMERA_ROOT_NO_RGB_PIXELS"
        or payload.get("scene") != SCENE
        or payload.get("protocol_id") != GCP_PROTOCOL_ID
        or payload.get("canonical_sha256") != canonical_sha256(payload)
        or payload.get("formal_input_manifest", {}).get("sha256") != FORMAL_INPUT_SHA
        or payload.get("formal_input_manifest", {}).get("canonical_sha256")
        != FORMAL_INPUT_CANONICAL_SHA
        or observations.get("observation_count") != 256
        or observations.get("unique_camera_count") != 211
        or observations.get("formal_role_counts") != {"train": 187, "test": 24}
        or output.get("camera_view_count") != 211
        or payload.get("rgb_truth_boundary", {}).get("real_rgb_pixels_present") is not False
    ):
        raise RuntimeError("GCP camera-root evidence mismatch")
    root = Path(str(output.get("root", ""))).resolve()
    if path.resolve() != root / "GCP_EVALUATION_CAMERA_ROOT_MANIFEST.json":
        raise RuntimeError("GCP camera-root manifest path mismatch")
    image_names = [str(value) for value in output.get("image_names", [])]
    if len(image_names) != 211 or len(set(image_names)) != 211:
        raise RuntimeError("GCP camera-root name inventory mismatch")
    role_by_name = {
        str(row["image_name"]): str(row["role"]) for row in formal_input["images"]
    }
    role_counts = {
        role: sum(role_by_name.get(name) == role for name in image_names)
        for role in ("train", "test")
    }
    if role_counts != {"train": 187, "test": 24}:
        raise RuntimeError("GCP camera-root frozen role inventory mismatch")
    placeholder_sha = str(output.get("placeholder", {}).get("sha256", ""))
    for name in image_names:
        image = root / "images" / name
        require_file(image, placeholder_sha)
    for name, row in output.get("sparse_files", {}).items():
        sparse = root / "sparse" / "0" / str(name)
        require_file(sparse, str(row.get("sha256", "")))
        if row.get("path") != str(sparse) or row.get("bytes") != sparse.stat().st_size:
            raise RuntimeError(f"GCP camera-root sparse inventory mismatch: {name}")
    return payload, root


def load_attempt_freeze(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    require_file(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    errors, methods = validate_scene_attempt_freeze(
        payload, freeze_path=path.resolve(), expected_scene=SCENE
    )
    if errors or methods is None:
        raise RuntimeError("invalid scene-attempt freeze: " + "; ".join(errors))
    return payload, methods


def raw_model_identity(
    method_id: str,
    row: dict[str, Any],
    *,
    base_repo: Path,
) -> dict[str, Any]:
    """Return renderer fields only after the base frozen identity fully revalidates."""
    run_root = Path(str(row["run_root"])).resolve()
    recipe_path = require_file(Path(str(row["recipe_path"])), str(row["recipe_sha256"]))
    try:
        recipe_path.relative_to(base_repo.resolve())
    except ValueError as exc:
        raise RuntimeError(f"{method_id}: recipe is outside the frozen base checkout") from exc
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    if (
        recipe.get("method_id") != method_id
        or recipe.get("scene") != SCENE
        or Path(str(recipe.get("authorized_run_root", ""))).resolve() != run_root
    ):
        raise RuntimeError(f"{method_id}: recipe/run-root identity mismatch")
    identity_path = require_file(
        Path(str(row["model_checkpoint_path"])),
        str(row["model_checkpoint_sha256"]),
    )
    plan = json.loads(require_file(base_repo / BASE_PLAN, BASE_PLAN_SHA).read_text(encoding="utf-8"))
    expected_identity_root = Path(
        str(plan.get("attempt_freeze", {}).get("model_identity_root", ""))
    ).resolve()
    if identity_path != expected_identity_root / f"{method_id}.json":
        raise RuntimeError(f"{method_id}: model identity is outside the frozen identity root")
    bound_recipe = dict(recipe)
    bound_recipe["_recipe_path"] = str(recipe_path)
    identity = validate_model_identity_bundle(
        manifest_path=identity_path,
        method_id=method_id,
        run_root=run_root,
        recipe=bound_recipe,
        repo=base_repo,
    )
    inventory = {
        str(Path(str(item["path"])).resolve()): item
        for item in identity["inventory"]
    }

    def frozen_file(path: Path) -> Path:
        resolved = require_file(path)
        try:
            resolved.relative_to(run_root)
        except ValueError as exc:
            raise RuntimeError(f"{method_id}: renderer model path escapes frozen run root") from exc
        item = inventory.get(str(resolved))
        if (
            item is None
            or item.get("bytes") != resolved.stat().st_size
            or item.get("sha256") != sha256_file(resolved)
        ):
            raise RuntimeError(f"{method_id}: renderer model file is absent from base identity: {resolved}")
        return resolved

    def evaluation_dependency(path: Path) -> tuple[Path, dict[str, Any]]:
        resolved = require_file(path)
        try:
            resolved.relative_to(run_root)
        except ValueError as exc:
            raise RuntimeError(f"{method_id}: evaluation dependency escapes frozen run root") from exc
        return resolved, {
            "path": str(resolved),
            "bytes": resolved.stat().st_size,
            "sha256": sha256_file(resolved),
            "role": "renderer_evaluation_dependency_not_model_checkpoint",
        }

    common = {
        "recipe_path": str(recipe_path),
        "recipe_sha256": sha256_file(recipe_path),
        "attempt_model_identity_path": str(identity_path),
        "attempt_model_identity_sha256": sha256_file(identity_path),
        "attempt_model_identity_canonical_sha256": identity["canonical_sha256"],
    }
    if method_id == "3dgs_original":
        ply = frozen_file(
            run_root / str(recipe["reuse_model_binding"]["point_cloud_relative_path"])
        )
        if sha256_file(ply) != recipe["reuse_model_binding"]["point_cloud_sha256"]:
            raise RuntimeError("3DGS reused model differs from the frozen recipe")
        cfg_args = frozen_file(run_root / "model" / "cfg_args")
        return {
            **common,
            "model_root": str(run_root / "model"),
            "formal_model_path": str(ply),
            "formal_model_sha256": sha256_file(ply),
            "cfg_args_path": str(cfg_args),
            "cfg_args_sha256": sha256_file(cfg_args),
            "iteration": 30000,
        }
    summary_path = frozen_file(run_root / "model" / "training_wrapper_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "TRAINING_PASS" or summary.get("method_id") != method_id:
        raise RuntimeError(f"{method_id}: training wrapper summary mismatch")
    checkpoint = summary["checkpoint"]
    if method_id == "citygs_x":
        checkpoint_root = Path(str(checkpoint["path"])).resolve()
        ply = frozen_file(checkpoint_root / str(checkpoint["point_cloud_file"]))
        additional = frozen_file(checkpoint_root / "additional_attributes.npz")
        checkpoints = frozen_file(checkpoint_root / "checkpoints.pth")
        cfg_args, cfg_dependency = evaluation_dependency(run_root / "model" / "cfg_args")
        return {
            **common,
            "model_root": str(run_root / "model"),
            "formal_model_path": str(ply),
            "formal_model_sha256": sha256_file(ply),
            "formal_model_aux_sha256": {
                "additional_attributes.npz": sha256_file(additional),
                "checkpoints.pth": sha256_file(checkpoints),
            },
            "cfg_args_path": str(cfg_args),
            "cfg_args_sha256": sha256_file(cfg_args),
            "evaluation_dependencies": [cfg_dependency],
            "training_summary_path": str(summary_path),
            "training_summary_sha256": sha256_file(summary_path),
            "iteration": 100000,
        }
    if method_id == "metrogs":
        checkpoint_path = frozen_file(Path(str(checkpoint["merged_path"])))
        point_cloud = frozen_file(Path(str(checkpoint["point_cloud_path"])))
        cameras, cameras_dependency = evaluation_dependency(run_root / "model" / "cameras.json")
        return {
            **common,
            "model_root": str(run_root / "model"),
            "formal_checkpoint": str(checkpoint_path),
            "formal_model_path": str(checkpoint_path),
            "formal_model_sha256": sha256_file(checkpoint_path),
            "point_cloud_path": str(point_cloud),
            "point_cloud_sha256": sha256_file(point_cloud),
            "training_cameras_json": str(cameras),
            "training_cameras_json_sha256": sha256_file(cameras),
            "evaluation_dependencies": [cameras_dependency],
            "training_summary_path": str(summary_path),
            "training_summary_sha256": sha256_file(summary_path),
            "iteration": 150000,
        }
    raise RuntimeError(f"unsupported READY method for RGB registry: {method_id}")


def build_rgb_registry(
    *,
    addendum_repo: Path,
    base_repo: Path,
    methods: dict[str, Any],
    formal_input_path: Path,
    rgb_camera_root: Path,
    rgb_camera_manifest: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    template_path = addendum_repo / "configs" / "m3m_gcp_native_quarter_rgb_quality_3k_registry_v1.json"
    template = json.loads(require_file(template_path).read_text(encoding="utf-8"))
    template_rows = {str(row["method_id"]): row for row in template.get("methods", [])}
    ready_rows = [row for row in methods["methods"] if row.get("attempt_status") == "READY_FOR_EVALUATION"]
    ready_ids = [str(row["method_id"]) for row in ready_rows]
    if not ready_ids or not set(ready_ids).issubset(READY_METHODS_SUPPORTED):
        raise RuntimeError(f"unsupported READY method set: {ready_ids}")
    camera_sha = {
        name: str(rgb_camera_manifest["output"]["files"][name]["sha256"])
        for name in ("cameras.bin", "images.bin", "points3D.bin")
    }
    method_rows: list[dict[str, Any]] = []
    for attempt_row in ready_rows:
        method_id = str(attempt_row["method_id"])
        row = dict(template_rows[method_id])
        model = raw_model_identity(method_id, attempt_row, base_repo=base_repo)
        row["scene"] = SCENE
        row["run_root"] = str(Path(str(attempt_row["run_root"])).resolve())
        row["model_root"] = model["model_root"]
        row["iteration"] = model["iteration"]
        row["camera_root"] = str(rgb_camera_root)
        row["camera_sparse_sha256"] = camera_sha
        row["attempt_model_identity_path"] = model["attempt_model_identity_path"]
        row["attempt_model_identity_sha256"] = model["attempt_model_identity_sha256"]
        row["attempt_model_identity_canonical_sha256"] = model[
            "attempt_model_identity_canonical_sha256"
        ]
        row["recipe_path"] = model["recipe_path"]
        row["recipe_sha256"] = model["recipe_sha256"]
        row["evaluation_dependencies"] = model.get("evaluation_dependencies", [])
        row["formal_output_root"] = str(output_root / "rgb" / method_id)
        for stale in (
            "formal_model_relative_path",
            "formal_model_sha256",
            "formal_model_aux_sha256",
            "cfg_args_sha256",
            "formal_checkpoint",
            "training_cameras_json",
            "training_cameras_json_sha256",
        ):
            row.pop(stale, None)
        if method_id in {"3dgs_original", "citygs_x"}:
            row["formal_model_relative_path"] = str(
                Path(model["formal_model_path"]).relative_to(Path(model["model_root"]))
            )
            row["formal_model_sha256"] = model["formal_model_sha256"]
            row["cfg_args_sha256"] = model["cfg_args_sha256"]
            if method_id == "citygs_x":
                row["formal_model_aux_sha256"] = model["formal_model_aux_sha256"]
        else:
            row["formal_checkpoint"] = model["formal_checkpoint"]
            row["formal_model_sha256"] = model["formal_model_sha256"]
            row["training_cameras_json"] = model["training_cameras_json"]
            row["training_cameras_json_sha256"] = model["training_cameras_json_sha256"]
        method_rows.append(row)

    shared = dict(template["shared"])
    shared.update(
        {
            "benchmark_repo_template": str(addendum_repo),
            "contract_relative_path": "configs/m3m_gcp_native_quarter_rgb_quality_100k_v1.json",
            "registry_relative_path": None,
            "input_manifest": str(formal_input_path),
            "input_root": str(formal_input_path.parent),
            "default_camera_root": str(rgb_camera_root),
            "default_camera_sparse_sha256": camera_sha,
            "graphdeco_camera_root": str(rgb_camera_root),
            "graphdeco_camera_sparse_sha256": camera_sha,
            "graphdeco_camera_root_policy": "immutable 314-heldout-view loader root; exact test sparse records plus deterministic empty points3D.bin compatibility member",
            "output_relative_path": None,
            "formal_output_root": str(output_root / "rgb"),
        }
    )
    payload: dict[str, Any] = {
        "schema": "m3m_gcp_native_quarter_rgb_quality_100k_registry_v1",
        "suite_id": RGB_SUITE_ID,
        "status": "ACTIVE_FROZEN",
        "scene": SCENE,
        "server": "AutoDL-901",
        "active_method_count": len(method_rows),
        "ready_method_ids": ready_ids,
        "failed_or_oom_methods_excluded": [
            str(row["method_id"]) for row in methods["methods"] if row.get("attempt_status") != "READY_FOR_EVALUATION"
        ],
        "shared": shared,
        "methods": method_rows,
        "execution_policy": {
            "scene_attempt_freeze_required_before_rgb": True,
            "three_track_addendum_activation_required": True,
            "formal_contract_status_required": "ACTIVE_FROZEN",
            "one_method_failure_does_not_abort_other_methods": True,
            "failure_is_recorded_not_repaired_with_test_truth": True,
            "render_then_shared_metric": True,
            "model_files_remain_on_901": True,
            "only_lightweight_metrics_and_manifests_are_pulled_local": True,
        },
    }
    payload["canonical_sha256"] = canonical_sha256(payload)
    return payload


def validate_legacy_result_metadata(
    *,
    prelaunch: dict[str, Any],
    summary: dict[str, Any],
    evaluator_manifest: dict[str, Any],
    verifier: dict[str, Any],
    formal_model_sha256: str,
) -> None:
    if (
        prelaunch.get("status") != "PASS"
        or prelaunch.get("scene") != SCENE
        or prelaunch.get("method_id") != "3dgs_original"
        or prelaunch.get("formal_model_ply_sha256") != formal_model_sha256
        or summary.get("status") != "COMPLETE_RANKED"
        or summary.get("ranking_eligible") is not True
        or summary.get("scene") != SCENE
        or summary.get("method_id") != "3dgs_original"
        or summary.get("protocol_id") != GCP_PROTOCOL_ID
        or summary.get("common_sim3_sha256") != COMMON_SIM3_SHA
        or summary.get("method_specific_sim3_fitted") is not False
        or evaluator_manifest.get("schema")
        != "m3m_gcp_native_quarter_evaluator_run_manifest_v2"
        or evaluator_manifest.get("protocol_release_manifest_sha256")
        != GCP_PROTOCOL_RELEASE_SHA
        or evaluator_manifest.get("source_data_contract_sha256")
        != GCP_DATA_CONTRACT_SHA
        or evaluator_manifest.get("sim3_policy")
        != "frozen_common_transform_no_method_refit"
        or verifier.get("status") != "PASS"
        or verifier.get("passed") is not True
        or verifier.get("scene") != SCENE
        or verifier.get("method_id") != "3dgs_original"
        or verifier.get("ranking_status") != summary.get("status")
        or verifier.get("recomputed_residual_statistics")
        != summary.get("residual_statistics")
        or verifier.get("method_specific_sim3_fitted") is not False
        or verifier.get("common_sim3_recomputation_passed") is not True
        or verifier.get("dependency_hashes_passed") is not True
        or verifier.get("output_hashes_passed") is not True
    ):
        raise RuntimeError("legacy 3DGS GCP result or verifier mismatch")


def validate_legacy_protocol_dependencies(
    *, release: dict[str, Any], payload_rows: dict[str, Any], scene_observations_sha256: str
) -> None:
    if (
        release.get("protocol_id") != GCP_PROTOCOL_ID
        or release.get("method_result_sim3_refit_allowed") is not False
        or release.get("source_data", {}).get("data_contract_sha256")
        != GCP_DATA_CONTRACT_SHA
        or payload_rows.get("observation_semantics.csv", {}).get("sha256")
        != OBSERVATION_SEMANTICS_SHA
        or payload_rows.get(f"scenes/{SCENE}/common_sim3.json", {}).get("sha256")
        != COMMON_SIM3_SHA
        or payload_rows.get(
            f"scenes/{SCENE}/triangulation_observation_residuals.csv", {}
        ).get("sha256")
        != scene_observations_sha256
    ):
        raise RuntimeError("legacy protocol release/common Sim(3) dependency mismatch")


def build_legacy_adoption(
    *,
    base_repo: Path,
    addendum_config: dict[str, Any],
    methods: dict[str, Any],
    freeze_path: Path,
    freeze: dict[str, Any],
    formal_input: dict[str, Any],
    legacy_root: Path,
) -> dict[str, Any]:
    legacy_root = legacy_root.resolve()
    legacy_config = addendum_config.get("legacy_3dgs_gcp", {})
    if legacy_root != Path(str(legacy_config.get("root", ""))).resolve():
        raise RuntimeError("legacy GCP root differs from the reviewed addendum config")
    method_row = next(row for row in methods["methods"] if row["method_id"] == "3dgs_original")
    if method_row.get("attempt_status") != "READY_FOR_EVALUATION":
        raise RuntimeError("3DGS is not READY for legacy GCP adoption")
    model = raw_model_identity("3dgs_original", method_row, base_repo=base_repo)
    legacy_paths = {name: require_file(legacy_root / relative) for name, relative in LEGACY_FILES.items()}
    prelaunch = json.loads(legacy_paths["prelaunch"].read_text(encoding="utf-8"))
    packet = json.loads(legacy_paths["packet_manifest"].read_text(encoding="utf-8"))
    summary = json.loads(legacy_paths["evaluation_summary"].read_text(encoding="utf-8"))
    evaluator_manifest = json.loads(
        legacy_paths["evaluator_manifest"].read_text(encoding="utf-8")
    )
    verifier = json.loads(legacy_paths["independent_verification"].read_text(encoding="utf-8"))
    subset = json.loads(
        legacy_paths["evaluation_subset_manifest"].read_text(encoding="utf-8")
    )
    validate_legacy_result_metadata(
        prelaunch=prelaunch,
        summary=summary,
        evaluator_manifest=evaluator_manifest,
        verifier=verifier,
        formal_model_sha256=model["formal_model_sha256"],
    )
    common_sim3 = require_file(
        Path(str(summary.get("common_sim3_path", ""))), COMMON_SIM3_SHA
    )
    protocol_release = require_file(
        Path(str(evaluator_manifest.get("protocol_release_manifest", ""))),
        GCP_PROTOCOL_RELEASE_SHA,
    )
    data_contract = require_file(
        Path(str(evaluator_manifest.get("source_data_contract", ""))),
        GCP_DATA_CONTRACT_SHA,
    )
    release = json.loads(protocol_release.read_text(encoding="utf-8"))
    payload_rows = {
        str(row.get("path")): row for row in release.get("payload_files", [])
    }
    observation_semantics = require_file(
        protocol_release.parent / "observation_semantics.csv", OBSERVATION_SEMANTICS_SHA
    )
    scene_observations = require_file(
        protocol_release.parent
        / "scenes"
        / SCENE
        / "triangulation_observation_residuals.csv",
        "4332c503b35a51b36d0dc679b5d318c936219df3abb7dc9ac8115593e3a5ae52",
    )
    validate_legacy_protocol_dependencies(
        release=release,
        payload_rows=payload_rows,
        scene_observations_sha256=sha256_file(scene_observations),
    )
    for name, expected_sha in evaluator_manifest.get("outputs", {}).items():
        output_path = require_file(legacy_root / "evaluator" / str(name), str(expected_sha))
        if output_path != legacy_paths.get(
            {
                "evaluation_summary.json": "evaluation_summary",
                "point_results.csv": "point_results",
                "observation_samples.csv": "observation_samples",
            }.get(str(name), ""),
            output_path,
        ):
            raise RuntimeError(f"legacy evaluator output path mismatch: {name}")
    packet_sha = sha256_file(legacy_paths["packet_manifest"])
    packet_model_content = packet.get("model_content_hash")
    expected_packet_model_content = directory_tree_hash(
        Path(str(model["model_root"])).resolve()
    )
    if (
        summary.get("packet_manifest_sha256") != packet_sha
        or packet.get("rendered_view_count") != 211
        or len(packet.get("depth_index", [])) != 211
        or len(packet.get("packet_index", [])) != 211
        or packet.get("scene") != SCENE
        or packet.get("protocol_id") != GCP_PROTOCOL_ID
        or Path(str(packet.get("model_path", ""))).resolve()
        != Path(str(model["model_root"])).resolve()
        or packet_model_content != expected_packet_model_content
        or packet_sha != legacy_config.get("packet_manifest_sha256")
        or model["formal_model_sha256"] != legacy_config.get("formal_model_sha256")
    ):
        raise RuntimeError("legacy 3DGS GCP packet binding mismatch")
    packet_names = [str(row["image_name"]) for row in packet["depth_index"]]
    with observation_semantics.open("r", encoding="utf-8-sig", newline="") as handle:
        protocol_rows = [
            row for row in csv.DictReader(handle) if row.get("scene") == SCENE
        ]
    protocol_names = {str(row.get("image_name", "")) for row in protocol_rows}
    role_by_name = {
        str(row["image_name"]): str(row["role"]) for row in formal_input["images"]
    }
    role_counts = {
        role: sum(1 for name in protocol_names if role_by_name.get(name) == role)
        for role in ("train", "test")
    }
    with legacy_paths["allowlist"].open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        allowlist_names = {
            str(row.get("image_name", "")) for row in csv.DictReader(handle)
        }
    subset_names = {str(row.get("image_name", "")) for row in subset.get("images", [])}
    if (
        len(packet_names) != 211
        or len(set(packet_names)) != 211
        or len(protocol_rows) != 256
        or len(protocol_names) != 211
        or set(packet_names) != protocol_names
        or allowlist_names != protocol_names
        or subset_names != protocol_names
        or role_counts != {"train": 187, "test": 24}
        or prelaunch.get("allowlist_count") != 211
        or prelaunch.get("allowlist_sha256") != sha256_file(legacy_paths["allowlist"])
        or prelaunch.get("evaluation_subset_camera_view_count") != 211
        or prelaunch.get("evaluation_subset_observation_count") != 256
        or prelaunch.get("evaluation_subset_manifest_file_sha256")
        != sha256_file(legacy_paths["evaluation_subset_manifest"])
        or prelaunch.get("evaluation_subset_manifest_canonical_sha256")
        != subset.get("manifest_sha256")
        or subset.get("protocol_observations_file_sha256")
        != sha256_file(scene_observations)
    ):
        raise RuntimeError(
            "legacy GCP packet is not the exact frozen 211-camera/256-observation set"
        )
    names_sha = hashlib.sha256(
        json.dumps(sorted(packet_names), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    bound_files = {
        name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for name, path in legacy_paths.items()
    }
    payload: dict[str, Any] = {
        "schema": "m3m_gcp_100k_legacy_gcp_adoption_receipt_v1",
        "status": "PASS_LEGACY_GCP_ADOPTION_CANDIDATE",
        "scene": SCENE,
        "method_id": "3dgs_original",
        "protocol_id": GCP_PROTOCOL_ID,
        "scene_attempt_freeze_path": str(freeze_path),
        "scene_attempt_freeze_sha256": sha256_file(freeze_path),
        "scene_attempt_freeze_canonical_sha256": freeze["canonical_sha256"],
        "methods_manifest_path": freeze["methods_manifest_path"],
        "methods_manifest_file_sha256": freeze["methods_manifest_file_sha256"],
        "methods_manifest_canonical_sha256": freeze["methods_manifest_canonical_sha256"],
        "attempt_model_identity_path": method_row["model_checkpoint_path"],
        "attempt_model_identity_sha256": method_row["model_checkpoint_sha256"],
        "formal_model_path": model["formal_model_path"],
        "formal_model_sha256": model["formal_model_sha256"],
        "packet_model_content": expected_packet_model_content,
        "legacy_files": bound_files,
        "legacy_packet_subset_proof": {
            "packet_view_count": 211,
            "protocol_observation_count": 256,
            "formal_role_counts": role_counts,
            "frozen_train_view_count": 2196,
            "frozen_test_view_count": 314,
            "unique_packet_names": True,
            "packet_names_equal_protocol_observation_camera_names": True,
            "sorted_packet_names_canonical_sha256": names_sha,
        },
        "frozen_gcp_dependencies": {
            "common_sim3_path": str(common_sim3),
            "common_sim3_sha256": sha256_file(common_sim3),
            "protocol_release_path": str(protocol_release),
            "protocol_release_sha256": sha256_file(protocol_release),
            "data_contract_path": str(data_contract),
            "data_contract_sha256": sha256_file(data_contract),
            "observation_semantics_path": str(observation_semantics),
            "observation_semantics_sha256": sha256_file(observation_semantics),
            "scene_observations_path": str(scene_observations),
            "scene_observations_sha256": sha256_file(scene_observations),
            "evaluator_outputs": evaluator_manifest["outputs"],
            "method_specific_sim3_fitted": False,
        },
        "adopted_result": {
            "status": summary["status"],
            "ranking_eligible": summary["ranking_eligible"],
            "point_counts": summary["point_counts"],
            "checkpoint_coverage_rate": summary["checkpoint_coverage_rate"],
            "residual_statistics": summary["residual_statistics"],
        },
        "metrics_recomputed": False,
        "result_bytes_modified": False,
        "method_specific_sim3_fitted": False,
        "activation_required_before_unified_result_use": True,
    }
    payload["canonical_sha256"] = canonical_sha256(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--addendum-repo", type=Path, required=True)
    parser.add_argument("--base-repo", type=Path, required=True)
    parser.add_argument("--base-activation", type=Path, required=True)
    parser.add_argument("--scene-attempt-freeze", type=Path, required=True)
    parser.add_argument("--formal-input-manifest", type=Path, required=True)
    parser.add_argument("--rgb-camera-root-manifest", type=Path, required=True)
    parser.add_argument("--gcp-camera-root-manifest", type=Path, required=True)
    parser.add_argument("--legacy-3dgs-gcp-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    addendum_repo = args.addendum_repo.resolve()
    base_repo = args.base_repo.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(output_root)
    for repo in (addendum_repo, base_repo):
        try:
            output_root.relative_to(repo)
        except ValueError:
            pass
        else:
            raise RuntimeError("candidate output must be outside clean checkouts")

    addendum_commit, addendum_tree = require_clean_checkout(addendum_repo)
    base_commit, base_tree = require_clean_checkout(base_repo, expected_commit=BASE_COMMIT)
    if base_tree != BASE_TREE:
        raise RuntimeError("base checkout tree mismatch")
    require_file(base_repo / BASE_PLAN, BASE_PLAN_SHA)
    require_file(base_repo / BASE_RECIPE_MANIFEST, BASE_RECIPE_MANIFEST_SHA)
    addendum_config_path = require_file(addendum_repo / ADDENDUM_CONFIG)
    addendum_config = json.loads(addendum_config_path.read_text(encoding="utf-8"))
    if (
        addendum_config.get("schema")
        != "m3m_gcp_native_quarter_100k_three_track_evaluation_addendum_v1"
        or addendum_config.get("status") != "REVIEW_CANDIDATE_NOT_EXECUTION_AUTHORIZED"
        or addendum_config.get("execution_authorized") is not False
        or addendum_config.get("scene") != SCENE
        or addendum_config.get("canonical_sha256") != canonical_sha256(addendum_config)
        or addendum_config.get("review", {}).get("task_id") != REVIEW_TASK_ID
        or addendum_config.get("review", {}).get("required_verdict") != REQUIRED_REVIEW_VERDICT
    ):
        raise RuntimeError("three-track addendum config identity mismatch")
    for relative, expected_sha in addendum_config.get("bound_addendum_files", {}).items():
        require_file(addendum_repo / str(relative), str(expected_sha))
    runtime = addendum_config["runtime_artifacts"]
    if output_root != Path(str(runtime["candidate_root"])).resolve():
        raise RuntimeError("candidate output root differs from reviewed addendum config")
    base_activation = validate_base_activation(args.base_activation.resolve())
    freeze_path = args.scene_attempt_freeze.resolve()
    freeze, methods = load_attempt_freeze(freeze_path)
    formal_input_path = args.formal_input_manifest.resolve()
    formal_input = validate_formal_input(formal_input_path)
    rgb_camera_manifest, rgb_camera_root = validate_rgb_camera_root(
        args.rgb_camera_root_manifest.resolve(), formal_input=formal_input
    )
    gcp_camera_manifest, _gcp_camera_root = validate_gcp_camera_root(
        args.gcp_camera_root_manifest.resolve(), formal_input=formal_input
    )

    rgb_contract_path = addendum_repo / "configs" / "m3m_gcp_native_quarter_rgb_quality_100k_v1.json"
    rgb_contract = json.loads(require_file(rgb_contract_path).read_text(encoding="utf-8"))
    binding = rgb_contract.get("input_binding", {}).get("scene_bindings", {}).get(SCENE, {})
    if (
        rgb_contract.get("schema") != "m3m_gcp_native_quarter_rgb_quality_contract_v1"
        or rgb_contract.get("suite_id") != RGB_SUITE_ID
        or rgb_contract.get("status") != "ACTIVE_FROZEN"
        or binding.get("formal_input_manifest_file_sha256") != FORMAL_INPUT_SHA
        or binding.get("formal_input_manifest_canonical_sha256") != FORMAL_INPUT_CANONICAL_SHA
        or [binding.get("full_view_count"), binding.get("train_view_count"), binding.get("test_view_count")]
        != [2510, 2196, 314]
    ):
        raise RuntimeError("100K RGB contract mismatch")

    registry_root = output_root
    rgb_registry = build_rgb_registry(
        addendum_repo=addendum_repo,
        base_repo=base_repo,
        methods=methods,
        formal_input_path=formal_input_path,
        rgb_camera_root=rgb_camera_root,
        rgb_camera_manifest=rgb_camera_manifest,
        output_root=Path(str(runtime["formal_results_root"])).resolve(),
    )
    legacy = build_legacy_adoption(
        base_repo=base_repo,
        addendum_config=addendum_config,
        methods=methods,
        freeze_path=freeze_path,
        freeze=freeze,
        formal_input=formal_input,
        legacy_root=args.legacy_3dgs_gcp_root.resolve(),
    )

    registry_path = registry_root / "rgb_quality_100k_registry_v1.json"
    legacy_path = registry_root / "legacy_3dgs_gcp_adoption_receipt_v1.json"
    manifest_path = registry_root / "three_track_candidate_manifest_v1.json"
    write_exclusive(registry_path, rgb_registry)
    write_exclusive(legacy_path, legacy)
    candidate: dict[str, Any] = {
        "schema": "m3m_gcp_100k_three_track_candidate_manifest_v1",
        "status": "REVIEW_CANDIDATE_NOT_EXECUTION_AUTHORIZED",
        "execution_authorized": False,
        "scene": SCENE,
        "review": {
            "task_id": REVIEW_TASK_ID,
            "required_verdict": REQUIRED_REVIEW_VERDICT,
        },
        "addendum_checkout": {"commit": addendum_commit, "tree": addendum_tree},
        "base_checkout": {
            "path": str(base_repo),
            "commit": base_commit,
            "tree": base_tree,
        },
        "addendum_config": {
            "path": str(addendum_config_path),
            "sha256": sha256_file(addendum_config_path),
            "canonical_sha256": addendum_config["canonical_sha256"],
        },
        "base_activation": {
            "path": str(args.base_activation.resolve()),
            "sha256": sha256_file(args.base_activation.resolve()),
            "canonical_sha256": base_activation["canonical_sha256"],
        },
        "scene_attempt_freeze": {
            "path": str(freeze_path),
            "sha256": sha256_file(freeze_path),
            "canonical_sha256": freeze["canonical_sha256"],
        },
        "methods_manifest": {
            "path": freeze["methods_manifest_path"],
            "sha256": freeze["methods_manifest_file_sha256"],
            "canonical_sha256": freeze["methods_manifest_canonical_sha256"],
        },
        "formal_input_manifest": {
            "path": str(formal_input_path),
            "sha256": FORMAL_INPUT_SHA,
            "canonical_sha256": FORMAL_INPUT_CANONICAL_SHA,
        },
        "rgb_camera_root_manifest": {
            "path": str(args.rgb_camera_root_manifest.resolve()),
            "sha256": sha256_file(args.rgb_camera_root_manifest.resolve()),
            "canonical_sha256": rgb_camera_manifest["canonical_sha256"],
        },
        "gcp_camera_root_manifest": {
            "path": str(args.gcp_camera_root_manifest.resolve()),
            "sha256": sha256_file(args.gcp_camera_root_manifest.resolve()),
            "canonical_sha256": gcp_camera_manifest["canonical_sha256"],
        },
        "rgb_contract": {"path": str(rgb_contract_path), "sha256": sha256_file(rgb_contract_path)},
        "rgb_registry": {
            "path": str(registry_path),
            "sha256": sha256_file(registry_path),
            "canonical_sha256": rgb_registry["canonical_sha256"],
        },
        "legacy_3dgs_gcp_adoption": {
            "path": str(legacy_path),
            "sha256": sha256_file(legacy_path),
            "canonical_sha256": legacy["canonical_sha256"],
        },
        "tracks": {
            "rgb": {"suite_id": RGB_SUITE_ID, "heldout_view_count": 314, "depends_on_raw_metric_depth_packets": False},
            "gcp": {
                "protocol_id": GCP_PROTOCOL_ID,
                "protocol_observation_count": 256,
                "camera_view_count": 211,
                "formal_role_counts": {"train": 187, "test": 24},
                "real_rgb_pixels_present": False,
                "separate_packet_lifecycle": True,
            },
            "lidar": {"protocol_id": LIDAR_PROTOCOL_ID, "existing_scientific_contract_unchanged": True},
        },
        "packet_release_gate": {
            "gcp_packet": {
                "required_gates": ["GCP_VERIFIER_PASS", "GCP_LIGHTWEIGHT_ARCHIVE_PASS"],
                "required_deletion_receipt": True,
                "applies_to": ["citygs_x", "metrogs"],
            },
            "lidar_packet": {
                "precondition": "GCP_DELETION_RECEIPT_OR_LEGACY_3DGS_GCP_ADOPTION_PASS",
                "required_gates": ["LIDAR_VERIFIER_PASS", "LIDAR_LIGHTWEIGHT_ARCHIVE_PASS"],
                "required_deletion_receipt": True,
            },
            "rgb_blocks_packet_release": False,
            "raw_gcp_and_lidar_packets_may_coexist": False,
        },
        "formal_results_root": str(Path(str(runtime["formal_results_root"])).resolve()),
        "candidate_output_root": str(output_root),
    }
    candidate["canonical_sha256"] = canonical_sha256(candidate)
    write_exclusive(manifest_path, candidate)
    print(
        json.dumps(
            {
                "status": "PASS_100K_THREE_TRACK_REVIEW_CANDIDATE_CREATED",
                "candidate_manifest": str(manifest_path),
                "candidate_manifest_sha256": sha256_file(manifest_path),
                "ready_method_ids": rgb_registry["ready_method_ids"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
