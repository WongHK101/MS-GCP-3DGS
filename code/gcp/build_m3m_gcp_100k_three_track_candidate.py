#!/usr/bin/env python3
"""Build the exact post-attempt 100K RGB/GCP/LiDAR review candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from m3m_gcp_lidar_artifacts import (
    METHOD_IDS,
    canonical_sha256,
    sha256_file,
    validate_scene_attempt_freeze,
)


SCENE = "gcp_100000_20260610"
BASE_COMMIT = "e33368db9333f826a3e808ff00c437c1a6c63b82"
BASE_TREE = "4620a434bd081af9274fdfc37dbb0d673636edfc"
BASE_ACTIVATION_SHA = "2645864d7680833809a712c7e80967193cad1bbe3f4398e3d3fe5dbffefe72b2"
BASE_ACTIVATION_CANONICAL_SHA = "4ee6c89d26adcb19ea7571ddcead1f7355f3c43e9c5e8875c5980f6c4b25cb9d"
BASE_PLAN = "configs/m3m_gcp_native_quarter_100k_ten_method_execution_plan_v3.json"
BASE_PLAN_SHA = "11930b292b3212485377af59aa822b6b3ddb30d332f6148b937c0f8ac809de09"
BASE_RECIPE_MANIFEST = "configs/m3m_gcp_native_quarter_100k_recipe_manifest_v3.json"
BASE_RECIPE_MANIFEST_SHA = "0789f8d8f5a145ab8c531c0a5b34d211bc7e7c2c5018552c48ff5687c37dc4d2"
ADDENDUM_CONFIG = "configs/m3m_gcp_native_quarter_100k_three_track_evaluation_addendum_v1.json"
FORMAL_INPUT_SHA = "c2cf9e951d95fee12a28d942e95c5c420df55bc364738b3f8737fed1c78bef3d"
FORMAL_INPUT_CANONICAL_SHA = "5b4fe34743310bd2225feb2dd236200606be933002fec19d2c9ecb9f3ba6769d"
GCP_PROTOCOL_ID = "m3m_gcp_native_quarter_geometry_v2"
LIDAR_PROTOCOL_ID = "m3m_gcp_lidar_rendered_surface_v1"
RGB_SUITE_ID = "m3m_gcp_native_quarter_rgb_quality_v1"
REVIEW_TASK_ID = "019ff12c-cb29-7cb2-8fb6-1d82c5f8c54b"
REQUIRED_REVIEW_VERDICT = "PASS_100K_THREE_TRACK_EVALUATION_ADDENDUM_V1"
READY_METHODS_SUPPORTED = {"3dgs_original", "citygs_x", "metrogs"}

LEGACY_FILES = {
    "prelaunch": "evaluation_prelaunch.json",
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


def load_attempt_freeze(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    require_file(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    errors, methods = validate_scene_attempt_freeze(
        payload, freeze_path=path.resolve(), expected_scene=SCENE
    )
    if errors or methods is None:
        raise RuntimeError("invalid scene-attempt freeze: " + "; ".join(errors))
    return payload, methods


def raw_model_identity(method_id: str, row: dict[str, Any]) -> dict[str, Any]:
    run_root = Path(str(row["run_root"])).resolve()
    recipe_path = require_file(Path(str(row["recipe_path"])), str(row["recipe_sha256"]))
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    if recipe.get("method_id") != method_id or recipe.get("scene") != SCENE:
        raise RuntimeError(f"{method_id}: recipe identity mismatch")
    if method_id == "3dgs_original":
        ply = run_root / str(recipe["reuse_model_binding"]["point_cloud_relative_path"])
        require_file(ply, str(recipe["reuse_model_binding"]["point_cloud_sha256"]))
        cfg_args = require_file(run_root / "model" / "cfg_args")
        return {
            "model_root": str(run_root / "model"),
            "formal_model_path": str(ply),
            "formal_model_sha256": sha256_file(ply),
            "cfg_args_path": str(cfg_args),
            "cfg_args_sha256": sha256_file(cfg_args),
            "iteration": 30000,
        }
    summary_path = require_file(run_root / "model" / "training_wrapper_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "TRAINING_PASS" or summary.get("method_id") != method_id:
        raise RuntimeError(f"{method_id}: training wrapper summary mismatch")
    checkpoint = summary["checkpoint"]
    if method_id == "citygs_x":
        checkpoint_root = Path(str(checkpoint["path"])).resolve()
        ply = require_file(checkpoint_root / str(checkpoint["point_cloud_file"]))
        additional = require_file(checkpoint_root / "additional_attributes.npz")
        checkpoints = require_file(checkpoint_root / "checkpoints.pth")
        cfg_args = require_file(run_root / "model" / "cfg_args")
        return {
            "model_root": str(run_root / "model"),
            "formal_model_path": str(ply),
            "formal_model_sha256": sha256_file(ply),
            "formal_model_aux_sha256": {
                "additional_attributes.npz": sha256_file(additional),
                "checkpoints.pth": sha256_file(checkpoints),
            },
            "cfg_args_path": str(cfg_args),
            "cfg_args_sha256": sha256_file(cfg_args),
            "training_summary_path": str(summary_path),
            "training_summary_sha256": sha256_file(summary_path),
            "iteration": 100000,
        }
    if method_id == "metrogs":
        checkpoint_path = require_file(Path(str(checkpoint["merged_path"])))
        point_cloud = require_file(Path(str(checkpoint["point_cloud_path"])))
        cameras = require_file(run_root / "model" / "cameras.json")
        return {
            "model_root": str(run_root / "model"),
            "formal_checkpoint": str(checkpoint_path),
            "formal_model_path": str(checkpoint_path),
            "formal_model_sha256": sha256_file(checkpoint_path),
            "point_cloud_path": str(point_cloud),
            "point_cloud_sha256": sha256_file(point_cloud),
            "training_cameras_json": str(cameras),
            "training_cameras_json_sha256": sha256_file(cameras),
            "training_summary_path": str(summary_path),
            "training_summary_sha256": sha256_file(summary_path),
            "iteration": 150000,
        }
    raise RuntimeError(f"unsupported READY method for RGB registry: {method_id}")


def build_rgb_registry(
    *,
    addendum_repo: Path,
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
        model = raw_model_identity(method_id, attempt_row)
        row["scene"] = SCENE
        row["run_root"] = str(Path(str(attempt_row["run_root"])).resolve())
        row["model_root"] = model["model_root"]
        row["iteration"] = model["iteration"]
        row["camera_root"] = str(rgb_camera_root)
        row["camera_sparse_sha256"] = camera_sha
        row["attempt_model_identity_path"] = attempt_row["model_checkpoint_path"]
        row["attempt_model_identity_sha256"] = attempt_row["model_checkpoint_sha256"]
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


def build_legacy_adoption(
    *,
    methods: dict[str, Any],
    freeze_path: Path,
    freeze: dict[str, Any],
    formal_input: dict[str, Any],
    legacy_root: Path,
) -> dict[str, Any]:
    method_row = next(row for row in methods["methods"] if row["method_id"] == "3dgs_original")
    if method_row.get("attempt_status") != "READY_FOR_EVALUATION":
        raise RuntimeError("3DGS is not READY for legacy GCP adoption")
    model = raw_model_identity("3dgs_original", method_row)
    legacy_paths = {name: require_file(legacy_root / relative) for name, relative in LEGACY_FILES.items()}
    prelaunch = json.loads(legacy_paths["prelaunch"].read_text(encoding="utf-8"))
    packet = json.loads(legacy_paths["packet_manifest"].read_text(encoding="utf-8"))
    summary = json.loads(legacy_paths["evaluation_summary"].read_text(encoding="utf-8"))
    verifier = json.loads(legacy_paths["independent_verification"].read_text(encoding="utf-8"))
    if (
        prelaunch.get("status") != "PASS"
        or prelaunch.get("scene") != SCENE
        or prelaunch.get("method_id") != "3dgs_original"
        or prelaunch.get("formal_model_ply_sha256") != model["formal_model_sha256"]
        or summary.get("status") != "COMPLETE_RANKED"
        or summary.get("ranking_eligible") is not True
        or summary.get("scene") != SCENE
        or summary.get("method_id") != "3dgs_original"
        or summary.get("protocol_id") != GCP_PROTOCOL_ID
        or verifier.get("status") != "PASS"
        or verifier.get("passed") is not True
        or verifier.get("scene") != SCENE
        or verifier.get("method_id") != "3dgs_original"
        or verifier.get("ranking_status") != summary.get("status")
        or verifier.get("recomputed_residual_statistics") != summary.get("residual_statistics")
    ):
        raise RuntimeError("legacy 3DGS GCP result or verifier mismatch")
    packet_sha = sha256_file(legacy_paths["packet_manifest"])
    packet_model_content = packet.get("model_content_hash")
    packet_model_file_hashes = {
        str(row.get("sha256"))
        for row in (packet_model_content.get("files", []) if isinstance(packet_model_content, dict) else [])
    }
    if (
        summary.get("packet_manifest_sha256") != packet_sha
        or packet.get("rendered_view_count") != 211
        or len(packet.get("depth_index", [])) != 211
        or len(packet.get("packet_index", [])) != 211
        or packet.get("scene") != SCENE
        or packet.get("protocol_id") != GCP_PROTOCOL_ID
        or model["formal_model_sha256"] not in packet_model_file_hashes
    ):
        raise RuntimeError("legacy 3DGS GCP packet binding mismatch")
    packet_names = [str(row["image_name"]) for row in packet["depth_index"]]
    train_names = {str(row["image_name"]) for row in formal_input["images"] if row.get("role") == "train"}
    if len(set(packet_names)) != 211 or not set(packet_names).issubset(train_names):
        raise RuntimeError("legacy GCP packet names are not a unique 211-view train subset")
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
        "legacy_files": bound_files,
        "legacy_packet_subset_proof": {
            "packet_view_count": 211,
            "frozen_train_view_count": 2196,
            "unique_packet_names": True,
            "all_packet_names_are_in_frozen_train_allowlist": True,
            "sorted_packet_names_canonical_sha256": names_sha,
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
        methods=methods,
        formal_input_path=formal_input_path,
        rgb_camera_root=rgb_camera_root,
        rgb_camera_manifest=rgb_camera_manifest,
        output_root=Path(str(runtime["formal_results_root"])).resolve(),
    )
    legacy = build_legacy_adoption(
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
        "base_checkout": {"commit": base_commit, "tree": base_tree},
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
            "gcp": {"protocol_id": GCP_PROTOCOL_ID, "new_ready_methods_use_current_packet_set": True},
            "lidar": {"protocol_id": LIDAR_PROTOCOL_ID, "existing_scientific_contract_unchanged": True},
        },
        "packet_release_gate": {
            "required_gcp_gate": "NEW_GCP_VERIFIER_PASS_OR_LEGACY_GCP_ADOPTION_PASS",
            "required_lidar_gates": ["LIDAR_VERIFIER_PASS", "LIDAR_LIGHTWEIGHT_ARCHIVE_PASS"],
            "rgb_blocks_packet_release": False,
            "raw_packet_delete_before_all_required_gates": False,
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
