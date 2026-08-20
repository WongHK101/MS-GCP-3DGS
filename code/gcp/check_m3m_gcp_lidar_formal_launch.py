#!/usr/bin/env python3
"""Fail-closed launch gate for formal M3M-GCP LiDAR evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from m3m_gcp_lidar_artifacts import (
    validate_failure_evidence_file,
    validate_scene_attempt_freeze,
)


PROTOCOL_ID = "m3m_gcp_lidar_rendered_surface_v1"
ACTIVE_METHOD_CLASSES = {
    "3dgs_original": "rgb_colmap_only",
    "2dgs": "rgb_colmap_only",
    "pgsr": "rgb_colmap_only",
    "rade_gs": "rgb_colmap_only",
    "qgs": "rgb_colmap_only",
    "gsprior": "rgb_colmap_only",
    "sof": "rgb_colmap_only",
    "citygaussian_v2": "rgb_colmap_external_geometry_prior",
    "citygs_x": "rgb_colmap_external_geometry_prior",
    "metrogs": "rgb_colmap_external_geometry_prior",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: Any, *, self_field: str = "canonical_sha256") -> str:
    clean = dict(payload)
    clean.pop(self_field, None)
    raw = json.dumps(clean, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def git_value(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, encoding="utf-8"
    ).strip()


def safe_payload_path(root: Path, relative_path: str) -> Path:
    rel = PurePosixPath(relative_path)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"unsafe payload path: {relative_path}")
    return root.joinpath(*rel.parts)


def validate_packet_files(
    *,
    manifest_path: Path,
    expected_image_names: tuple[str, ...],
    packet_schema: dict[str, Any],
) -> list[str]:
    """Verify every selected-method packet byte and NPZ contract before output."""
    errors: list[str] = []
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"packet manifest unreadable: {exc}"]
    required = {
        "schema": "ms_gcp_metric_depth_packet_manifest_v2",
        "protocol_id": "m3m_gcp_native_quarter_geometry_v2",
        "primary_depth_tensor": "alpha_normalized_expected_camera_z",
        "primary_depth_semantics": "camera_z",
        "image_domain": "colmap_4_0_4_image_undistorter_pinhole_max_1414",
        "pixel_coordinate_convention": "zero_based_pixel_centers",
        "camera_z_unit_contract": "frozen_colmap_model_camera_z_units",
        "adapter_conformance_status": "PASS",
        "camera_sets": "train",
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            errors.append(f"packet manifest {key} mismatch")
    entries = payload.get("depth_index", [])
    names = tuple(sorted(str(row.get("image_name", "")) for row in entries))
    if names != expected_image_names or int(payload.get("rendered_view_count", -1)) != len(expected_image_names):
        errors.append("packet image inventory differs from exact train-view allowlist")
    if len(names) != len(set(names)):
        errors.append("packet image names are not unique")
    keys_exact = set(packet_schema.get("keys_exact", []))
    required_fields = set(packet_schema.get("depth_index_fields_required", []))
    manifest_dir = manifest_path.parent.resolve()
    for entry in entries:
        image_name = str(entry.get("image_name", "<missing>"))
        missing = sorted(required_fields - set(entry))
        if missing:
            errors.append(f"{image_name}: depth-index fields missing: {missing}")
            continue
        if entry.get("split") != "train":
            errors.append(f"{image_name}: packet split is not train")
        if entry.get("dtype") != "float32":
            errors.append(f"{image_name}: depth-index dtype is not float32")
        try:
            width, height = int(entry["width"]), int(entry["height"])
            packet_path = Path(str(entry["packet_path"])).resolve()
            if packet_path.parent != manifest_dir:
                errors.append(f"{image_name}: packet path escapes manifest directory")
                continue
            if not packet_path.is_file():
                errors.append(f"{image_name}: packet missing")
                continue
            if packet_path.stat().st_size != int(entry["packet_bytes"]):
                errors.append(f"{image_name}: packet byte count mismatch")
                continue
            if sha256_file(packet_path) != entry["packet_sha256"]:
                errors.append(f"{image_name}: packet SHA mismatch")
                continue
            with np.load(packet_path, allow_pickle=False) as packet:
                if set(packet.files) != keys_exact:
                    errors.append(f"{image_name}: packet NPZ keys mismatch")
                    continue
                declared_keys = set(str(entry["tensor_names"]).split("|"))
                if declared_keys != keys_exact:
                    errors.append(f"{image_name}: tensor_names does not match NPZ keys")
                for key in packet.files:
                    expected_dtype = np.dtype("bool" if key == "metric_depth_valid_mask" else "float32")
                    if packet[key].dtype != expected_dtype:
                        errors.append(f"{image_name}: {key} dtype mismatch")
                    if packet[key].shape != (height, width):
                        errors.append(f"{image_name}: {key} shape mismatch")
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f"{image_name}: packet validation error: {exc}")
    return errors


def validate_launch(
    *,
    repo: Path,
    contract_path: Path,
    activation_path: Path,
    schema_path: Path,
    split_path: Path,
    registry_path: Path,
    geometry_release_root: Path,
    formal_input_root: Path,
    colmap_model: Path,
    lidar_inventory_path: Path,
    lidar_root: Path,
    gcp_path: Path,
    sim3_path: Path,
    methods_path: Path,
    scene_attempt_freeze_path: Path,
    scene_authorization_path: Path,
    scene: str,
    selected_method_id: str,
    output_root: Path,
) -> list[str]:
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    activation = json.loads(activation_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    methods = json.loads(methods_path.read_text(encoding="utf-8"))
    scene_authorization = json.loads(scene_authorization_path.read_text(encoding="utf-8"))
    sim3 = json.loads(sim3_path.read_text(encoding="utf-8"))

    require(contract.get("protocol_id") == PROTOCOL_ID, "contract protocol mismatch")
    require(contract.get("status") == "REVIEW_CANDIDATE_NOT_EXECUTION_AUTHORIZED", "reviewed contract candidate identity changed")
    require(contract.get("execution_authorized") is False, "reviewed contract authorization bit changed")
    activation_schema = schema.get("activation_manifest", {})
    require(set(activation) == set(activation_schema.get("required_fields_exact", [])), "activation fields mismatch")
    require(activation.get("schema") == "m3m_gcp_lidar_formal_activation_v1", "activation schema mismatch")
    require(activation.get("protocol_id") == PROTOCOL_ID, "activation protocol mismatch")
    required_verdict = "PASS_100K_TIME_SPACE_EXECUTION_PLAN_V1"
    require(activation.get("review_verdict") == required_verdict, "activation review verdict mismatch")
    require(activation.get("review_task_id") == contract.get("review", {}).get("review_task_id"), "activation review task mismatch")
    require(activation.get("execution_authorized") is True, "activation does not authorize execution")
    require(activation.get("contract_file_sha256") == sha256_file(contract_path), "activation contract SHA mismatch")
    require(activation.get("artifact_schema_sha256") == sha256_file(schema_path), "activation artifact-schema SHA mismatch")
    require(activation.get("canonical_sha256") == canonical_sha256(activation), "activation canonical SHA mismatch")
    for path_field, sha_field, label in (
        ("execution_plan_path", "execution_plan_sha256", "execution plan"),
        ("recipe_manifest_path", "recipe_manifest_sha256", "recipe manifest"),
    ):
        raw_path = Path(str(activation.get(path_field, "")))
        bound_path = raw_path if raw_path.is_absolute() else repo / raw_path
        require(bound_path.is_file(), f"activation {label} file missing")
        if bound_path.is_file():
            require(sha256_file(bound_path) == activation.get(sha_field), f"activation {label} SHA mismatch")
    require(activation.get("benchmark_commit") == activation.get("reviewed_commit"), "activation reviewed commit mismatch")
    require(activation.get("benchmark_tree") == activation.get("reviewed_tree"), "activation reviewed tree mismatch")
    require(schema.get("schema") == "m3m_gcp_lidar_formal_artifact_schema_v1", "artifact schema mismatch")
    authorization_fields = set(
        schema.get("scene_execution_authorization", {}).get("required_fields_exact", [])
    )
    require(set(scene_authorization) == authorization_fields, "scene authorization fields mismatch")
    require(scene_authorization.get("schema") == "m3m_gcp_lidar_scene_execution_authorization_v1", "scene authorization schema mismatch")
    require(scene_authorization.get("protocol_id") == PROTOCOL_ID, "scene authorization protocol mismatch")
    require(scene_authorization.get("scene") == scene, "scene authorization scene mismatch")
    require(scene_authorization.get("selected_method_id") == selected_method_id, "scene authorization selected method mismatch")
    require(scene_authorization.get("review_task_id") == contract.get("review", {}).get("review_task_id"), "scene authorization review task mismatch")
    require(scene_authorization.get("review_verdict") == required_verdict, "scene authorization review verdict mismatch")
    require(scene_authorization.get("execution_authorized") is True, "scene authorization does not authorize execution")
    require(scene_authorization.get("contract_file_sha256") == sha256_file(contract_path), "scene authorization contract SHA mismatch")
    require(scene_authorization.get("activation_manifest_sha256") == sha256_file(activation_path), "scene authorization activation SHA mismatch")
    require(scene_authorization.get("artifact_schema_sha256") == sha256_file(schema_path), "scene authorization artifact-schema SHA mismatch")
    require(scene_authorization.get("execution_plan_sha256") == activation.get("execution_plan_sha256"), "scene authorization execution-plan SHA mismatch")
    require(scene_authorization.get("canonical_sha256") == canonical_sha256(scene_authorization), "scene authorization canonical SHA mismatch")

    implementation = contract.get("implementation", {})
    repo_files = {
        "evaluator": repo / implementation.get("evaluator_path", "__missing__"),
        "verifier": repo / implementation.get("verifier_path", "__missing__"),
        "artifact_schema": repo / implementation.get("artifact_schema_path", "__missing__"),
        "ranker": repo / implementation.get("ranker_path", "__missing__"),
        "launch_gate": repo / implementation.get("launch_gate_path", "__missing__"),
        "artifact_helpers": repo / implementation.get("artifact_helpers_path", "__missing__"),
    }
    for key, path in repo_files.items():
        require(path.is_file(), f"{key} file missing")
        if path.is_file():
            require(sha256_file(path) == implementation.get(f"{key}_sha256"), f"{key} SHA mismatch")

    require(sha256_file(split_path) == contract["source_data_release"]["split_manifest_file_sha256"], "split SHA mismatch")
    geometry_binding = contract.get("source_geometry_binding", {})
    release_pin = repo / str(geometry_binding.get("release_pin_path", "__missing__"))
    release_manifest = geometry_release_root / str(
        geometry_binding.get("release_manifest_relative_path", "__missing__")
    )
    require(release_pin.is_file(), "geometry release pin missing")
    if release_pin.is_file():
        require(sha256_file(release_pin) == geometry_binding.get("release_pin_sha256"), "geometry release pin SHA mismatch")
    require(release_manifest.is_file(), "geometry release manifest missing")
    if release_manifest.is_file():
        require(sha256_file(release_manifest) == geometry_binding.get("release_manifest_sha256"), "geometry release manifest SHA mismatch")
    require(gcp_path.is_file(), "frozen GCP coordinate file missing")
    if gcp_path.is_file():
        require(sha256_file(gcp_path) == geometry_binding.get("gcp_points_sha256"), "frozen GCP coordinate SHA mismatch")
    lidar = contract.get("lidar_source", {})
    require(lidar_inventory_path.is_file(), "LiDAR inventory file missing")
    if lidar_inventory_path.is_file():
        require(sha256_file(lidar_inventory_path) == lidar.get("payload_sha256_inventory_file_sha256"), "LiDAR inventory SHA mismatch")
        with lidar_inventory_path.open("r", encoding="utf-8-sig", newline="") as handle:
            inventory_rows = list(csv.DictReader(handle))
        inventory_laz = {
            row.get("relative_path_utf8_nfc", ""): {
                "bytes": int(row.get("bytes", -1)),
                "sha256": row.get("sha256"),
            }
            for row in inventory_rows
            if row.get("relative_path_utf8_nfc", "").endswith(".laz")
        }
        expected_laz = lidar.get("laz_files_exact", {})
        require(inventory_laz == expected_laz, "LiDAR LAZ rows differ from contract")
        actual_laz = {
            path.relative_to(lidar_root).as_posix()
            for path in lidar_root.joinpath("lidars", "terra_laz_1_4").glob("*.laz")
        }
        require(actual_laz == set(expected_laz), "LiDAR directory is not the exact frozen nine-LAZ set")
        for relative, identity in expected_laz.items():
            try:
                path = safe_payload_path(lidar_root, relative)
            except ValueError as exc:
                errors.append(str(exc))
                continue
            require(path.is_file(), f"LiDAR file missing: {relative}")
            if path.is_file():
                require(path.stat().st_size == int(identity["bytes"]), f"LiDAR byte count mismatch: {relative}")
                require(sha256_file(path) == identity["sha256"], f"LiDAR SHA mismatch: {relative}")
    method_binding = contract.get("method_registry_binding", {})
    require(sha256_file(registry_path) == method_binding.get("file_sha256"), "method registry SHA mismatch")
    require(registry.get("active_benchmark_method_ids") == list(ACTIVE_METHOD_CLASSES), "active method order mismatch")
    registry_classes = {
        row["method_id"]: row.get("input_class")
        for row in registry.get("methods", [])
        if row.get("method_id") in ACTIVE_METHOD_CLASSES
    }
    require(registry_classes == ACTIVE_METHOD_CLASSES, "registry input-class mapping mismatch")
    require(method_binding.get("active_method_input_classes") == ACTIVE_METHOD_CLASSES, "contract input-class mapping mismatch")

    scene_rows = {row["scene"]: row for row in contract.get("scenes", [])}
    require(scene in scene_rows, "scene absent from contract")
    sim3_binding = contract.get("source_geometry_binding", {}).get("scene_common_sim3_sha256", {})
    require(sha256_file(sim3_path) == sim3_binding.get(scene), "scene Sim3 SHA mismatch")
    require(sim3.get("protocol_id") == contract.get("source_geometry_protocol_id"), "scene Sim3 protocol mismatch")
    require(sim3.get("scene") == scene, "scene Sim3 scene mismatch")
    require(sim3.get("method_result_refit_forbidden") is True, "scene Sim3 permits method refit")

    input_manifest_path = formal_input_root / "NATIVE_QUARTER_INPUT_MANIFEST.json"
    require(input_manifest_path.is_file(), "formal input manifest missing")
    input_manifest: dict[str, Any] = {}
    if input_manifest_path.is_file():
        input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
        frozen_input = contract.get("formal_input_binding", {}).get("scene_manifests", {}).get(scene, {})
        require(sha256_file(input_manifest_path) == frozen_input.get("file_sha256"), "formal input manifest file SHA mismatch")
        require(input_manifest.get("scene") == scene, "formal input scene mismatch")
        require(input_manifest.get("release_root_digest_sha256") == contract["source_data_release"]["release_root_digest_sha256"], "formal input release mismatch")
        require(input_manifest.get("manifest_sha256") == canonical_sha256(input_manifest, self_field="manifest_sha256"), "formal input canonical SHA mismatch")
        require(input_manifest.get("manifest_sha256") == frozen_input.get("canonical_sha256"), "formal input manifest frozen canonical SHA mismatch")
        require(input_manifest.get("train_view_count") == scene_rows.get(scene, {}).get("train_views"), "formal input train-view count mismatch")
        expected_model_files = set(contract.get("formal_input_binding", {}).get("source_model_files_exact", []))
        require(set(input_manifest.get("source_model_sha256", {})) == expected_model_files, "formal input source-model inventory mismatch")
        expected_colmap_model = formal_input_root / "train" / "sparse" / "0"
        require(colmap_model.resolve() == expected_colmap_model.resolve(), "COLMAP model is not the exact formal train model")
        train_roles = [row for row in input_manifest.get("roles", []) if row.get("role") == "train"]
        require(len(train_roles) == 1, "formal input must contain exactly one train role")
        train_role = train_roles[0] if len(train_roles) == 1 else {}
        train_model_hashes = {
            "cameras.bin": train_role.get("cameras_bin_sha256"),
            "images.bin": train_role.get("images_bin_sha256"),
            "points3D.ply": train_role.get("points3d_ply_sha256"),
        }
        for filename, expected_sha in train_model_hashes.items():
            path = expected_colmap_model / filename
            require(path.is_file(), f"formal train model file missing: {filename}")
            if path.is_file():
                require(sha256_file(path) == expected_sha, f"formal train model SHA mismatch: {filename}")
        train_images = [row for row in input_manifest.get("images", []) if row.get("role") == "train"]
        require(len(train_images) == scene_rows.get(scene, {}).get("train_views"), "formal train JPEG inventory count mismatch")
        for row in train_images:
            try:
                path = safe_payload_path(formal_input_root, str(row.get("relative_path", "")))
            except ValueError as exc:
                errors.append(str(exc))
                continue
            image_name = str(row.get("image_name", "<missing>"))
            require(path.is_file(), f"formal train JPEG missing: {image_name}")
            if path.is_file():
                require(path.stat().st_size == int(row.get("jpeg_bytes", -1)), f"formal train JPEG byte count mismatch: {image_name}")
                require(sha256_file(path) == row.get("jpeg_sha256"), f"formal train JPEG SHA mismatch: {image_name}")

    split_rows = {row["scene"]: row for row in split.get("scenes", [])}
    train_names = sorted(split_rows.get(scene, {}).get("train_image_names", []))
    require(len(train_names) == scene_rows.get(scene, {}).get("train_views"), "frozen train-view count mismatch")
    manifest_train_names = sorted(
        str(row.get("image_name", ""))
        for row in input_manifest.get("images", [])
        if row.get("role") == "train"
    )
    require(manifest_train_names == train_names, "formal train JPEG names differ from frozen split")
    surface_binding = contract.get("reconstruction_surface", {})
    allowlist_manifest_path = repo / str(
        surface_binding.get("view_allowlist_manifest_path", "__missing__")
    )
    require(allowlist_manifest_path.is_file(), "frozen view-allowlist manifest missing")
    if allowlist_manifest_path.is_file():
        require(
            sha256_file(allowlist_manifest_path)
            == surface_binding.get("view_allowlist_manifest_file_sha256"),
            "frozen view-allowlist manifest file SHA mismatch",
        )
        allowlist_manifest = json.loads(
            allowlist_manifest_path.read_text(encoding="utf-8")
        )
        require(
            allowlist_manifest.get("canonical_sha256")
            == surface_binding.get("view_allowlist_manifest_canonical_sha256"),
            "frozen view-allowlist manifest canonical SHA mismatch",
        )
        allowlist_row = next(
            (row for row in allowlist_manifest.get("rows", []) if row.get("scene") == scene),
            {},
        )
        allowlist_path = repo / str(allowlist_row.get("path", "__missing__"))
        require(allowlist_path.is_file(), "scene view-allowlist CSV missing")
        if allowlist_path.is_file():
            require(
                sha256_file(allowlist_path) == allowlist_row.get("sha256"),
                "scene view-allowlist CSV SHA mismatch",
            )
            with allowlist_path.open("r", encoding="utf-8-sig", newline="") as handle:
                allowlist_names = sorted(
                    str(row.get("image_name", "")) for row in csv.DictReader(handle)
                )
            require(allowlist_names == train_names, "scene view-allowlist differs from frozen split")
    methods_rows = methods.get("methods", [])
    require(methods.get("schema") == "m3m_gcp_lidar_formal_methods_v1", "methods schema mismatch")
    require(methods.get("protocol_id") == PROTOCOL_ID, "methods protocol mismatch")
    require(methods.get("scene") == scene, "methods scene mismatch")
    require(methods.get("canonical_sha256") == canonical_sha256(methods), "methods canonical SHA mismatch")
    require([row.get("method_id") for row in methods_rows] == list(ACTIVE_METHOD_CLASSES), "methods manifest is not the exact ordered ten-method pool")
    require(selected_method_id in ACTIVE_METHOD_CLASSES, "selected method is not in the frozen pool")
    require(scene_authorization.get("methods_manifest_file_sha256") == sha256_file(methods_path), "scene authorization methods file SHA mismatch")
    require(scene_authorization.get("methods_manifest_canonical_sha256") == methods.get("canonical_sha256"), "scene authorization methods canonical SHA mismatch")
    require(scene_authorization.get("formal_input_manifest_file_sha256") == sha256_file(input_manifest_path), "scene authorization formal-input file SHA mismatch")
    require(scene_authorization.get("formal_input_manifest_canonical_sha256") == contract.get("formal_input_binding", {}).get("scene_manifests", {}).get(scene, {}).get("canonical_sha256"), "scene authorization formal-input canonical SHA mismatch")
    require(scene_authorization.get("scene_attempt_freeze_path") == str(scene_attempt_freeze_path), "scene authorization freeze path mismatch")
    require(scene_attempt_freeze_path.is_file(), "scene attempt freeze file missing")
    if scene_attempt_freeze_path.is_file():
        require(scene_authorization.get("scene_attempt_freeze_sha256") == sha256_file(scene_attempt_freeze_path), "scene authorization freeze SHA mismatch")
        freeze_payload = json.loads(scene_attempt_freeze_path.read_text(encoding="utf-8"))
        freeze_errors, frozen_methods = validate_scene_attempt_freeze(
            freeze_payload, freeze_path=scene_attempt_freeze_path, expected_scene=scene
        )
        errors.extend(f"scene attempt freeze: {error}" for error in freeze_errors)
        if frozen_methods is not None:
            require(Path(str(freeze_payload.get("methods_manifest_path"))).resolve() == methods_path.resolve(), "scene attempt freeze references a different methods manifest")
            require(freeze_payload.get("methods_manifest_file_sha256") == sha256_file(methods_path), "scene attempt freeze methods file SHA mismatch")
            require(freeze_payload.get("methods_manifest_canonical_sha256") == methods.get("canonical_sha256"), "scene attempt freeze methods canonical SHA mismatch")
    expected_method_fields = set(
        schema.get("formal_methods_manifest", {}).get("method_fields_exact", [])
    )
    seen: set[str] = set()
    for row in methods_rows:
        require(set(row) == expected_method_fields, f"method fields differ from artifact schema: {row.get('method_id')}")
        row_method_id = row.get("method_id")
        require(row_method_id in ACTIVE_METHOD_CLASSES, f"unknown method: {row_method_id}")
        require(row_method_id not in seen, f"duplicate method: {row_method_id}")
        seen.add(row_method_id)
        require(row.get("input_class") == ACTIVE_METHOD_CLASSES.get(row_method_id), f"{row_method_id}: input class mismatch")
        registry_row = next(
            (item for item in registry.get("methods", []) if item.get("method_id") == row_method_id),
            {},
        )
        require(row.get("method_name") == registry_row.get("display_name"), f"{row_method_id}: method name mismatch")
        for field in (
            "run_root",
            "recipe_path",
            "recipe_sha256",
            "renderer_adapter_path",
            "renderer_adapter_sha256",
        ):
            require(bool(row.get(field)), f"{row_method_id}: missing {field}")
        for path_field, sha_field in (
            ("recipe_path", "recipe_sha256"),
            ("renderer_adapter_path", "renderer_adapter_sha256"),
        ):
            path = Path(str(row.get(path_field, "")))
            require(path.is_file(), f"{row_method_id}: missing {path_field}")
            if path.is_file():
                require(sha256_file(path) == row.get(sha_field), f"{row_method_id}: {sha_field} mismatch")
        attempt_status = row.get("attempt_status")
        require(
            attempt_status in {"READY_FOR_EVALUATION", "OOM_UNRANKED", "FAILED_UNRANKED"},
            f"{row_method_id}: invalid attempt status",
        )
        if attempt_status == "READY_FOR_EVALUATION":
            require(bool(row.get("model_checkpoint_path")), f"{row_method_id}: missing model checkpoint path")
            require(bool(row.get("model_checkpoint_sha256")), f"{row_method_id}: missing model checkpoint SHA")
            require(row.get("failure_evidence_path") is None, f"{row_method_id}: ready method carries failure evidence path")
            require(row.get("failure_evidence_sha256") is None, f"{row_method_id}: ready method carries failure evidence SHA")
            model_path = Path(str(row.get("model_checkpoint_path", "")))
            require(model_path.is_file(), f"{row_method_id}: missing model checkpoint")
            if model_path.is_file():
                require(sha256_file(model_path) == row.get("model_checkpoint_sha256"), f"{row_method_id}: model checkpoint SHA mismatch")
        elif attempt_status in {"OOM_UNRANKED", "FAILED_UNRANKED"}:
            require(row.get("model_checkpoint_path") is None, f"{row_method_id}: failed method carries model checkpoint path")
            require(row.get("model_checkpoint_sha256") is None, f"{row_method_id}: failed method carries model checkpoint SHA")
            failure_path = Path(str(row.get("failure_evidence_path", "")))
            require(failure_path.is_file(), f"{row_method_id}: missing failure evidence")
            require(bool(row.get("failure_evidence_sha256")), f"{row_method_id}: missing failure evidence SHA")
            if failure_path.is_file():
                failure_errors = validate_failure_evidence_file(
                    failure_path,
                    expected_sha256=str(row.get("failure_evidence_sha256", "")),
                    expected_scene=scene,
                    expected_method_id=str(row_method_id),
                    expected_status=str(attempt_status),
                )
                errors.extend(f"{row_method_id}: {error}" for error in failure_errors)

    selected = next((row for row in methods_rows if row.get("method_id") == selected_method_id), None)
    if selected is not None:
        require(selected.get("attempt_status") == "READY_FOR_EVALUATION", f"{selected_method_id}: selected method is not ready for evaluation")
        packet_manifest_path = Path(str(scene_authorization.get("packet_manifest_path", "")))
        expected_packet_manifest_path = (
            Path(str(selected["run_root"]))
            / "formal_evaluation"
            / "packets"
            / "depth_export_manifest.json"
        )
        require(
            packet_manifest_path.resolve() == expected_packet_manifest_path.resolve(),
            f"{selected_method_id}: packet manifest is not under the frozen run root",
        )
        require(packet_manifest_path.is_file(), f"{selected_method_id}: missing packet manifest")
        if packet_manifest_path.is_file():
            require(
                sha256_file(packet_manifest_path)
                == scene_authorization.get("packet_manifest_sha256"),
                f"{selected_method_id}: packet manifest SHA mismatch",
            )
            errors.extend(
                f"{selected_method_id}: {error}"
                for error in validate_packet_files(
                    manifest_path=packet_manifest_path,
                    expected_image_names=tuple(train_names),
                    packet_schema=schema.get("depth_packet_manifest", {}).get("packet_npz", {}),
                )
            )

    expected_commit = activation.get("benchmark_commit")
    expected_tree = activation.get("benchmark_tree")
    require(git_value(repo, "rev-parse", "HEAD") == expected_commit, "benchmark commit mismatch")
    require(git_value(repo, "show", "-s", "--format=%T", "HEAD") == expected_tree, "benchmark tree mismatch")
    require(git_value(repo, "status", "--porcelain") == "", "benchmark checkout is dirty")
    require(scene_authorization.get("benchmark_commit") == expected_commit, "scene authorization benchmark commit mismatch")
    require(scene_authorization.get("benchmark_tree") == expected_tree, "scene authorization benchmark tree mismatch")
    require(output_root.is_absolute(), "output root must be absolute")
    require(
        Path(str(scene_authorization.get("authorized_output_root", ""))).resolve()
        == output_root.resolve(),
        "output root is not authorized for selected method",
    )
    require(not output_root.exists(), "formal output root already exists; overwrite/resume is forbidden")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--artifact-schema", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--geometry-release-root", type=Path, required=True)
    parser.add_argument("--formal-input-root", type=Path, required=True)
    parser.add_argument("--colmap-model", type=Path, required=True)
    parser.add_argument("--lidar-inventory", type=Path, required=True)
    parser.add_argument("--lidar-root", type=Path, required=True)
    parser.add_argument("--gcp-csv", type=Path, required=True)
    parser.add_argument("--sim3", type=Path, required=True)
    parser.add_argument("--methods", type=Path, required=True)
    parser.add_argument("--scene-attempt-freeze", type=Path, required=True)
    parser.add_argument("--scene-authorization", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--method-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    errors = validate_launch(
        repo=args.repo.resolve(),
        contract_path=args.contract.resolve(),
        activation_path=args.activation.resolve(),
        schema_path=args.artifact_schema.resolve(),
        split_path=args.split.resolve(),
        registry_path=args.registry.resolve(),
        geometry_release_root=args.geometry_release_root.resolve(),
        formal_input_root=args.formal_input_root.resolve(),
        colmap_model=args.colmap_model.resolve(),
        lidar_inventory_path=args.lidar_inventory.resolve(),
        lidar_root=args.lidar_root.resolve(),
        gcp_path=args.gcp_csv.resolve(),
        sim3_path=args.sim3.resolve(),
        methods_path=args.methods.resolve(),
        scene_attempt_freeze_path=args.scene_attempt_freeze.resolve(),
        scene_authorization_path=args.scene_authorization.resolve(),
        scene=args.scene,
        selected_method_id=args.method_id,
        output_root=args.output_root,
    )
    print(json.dumps({"status": "PASS_READY" if not errors else "FAIL", "errors": errors}, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
