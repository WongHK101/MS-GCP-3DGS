#!/usr/bin/env python3
"""Fail-closed launch gate for formal M3M-GCP LiDAR evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


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
    gcp_path: Path,
    sim3_path: Path,
    methods_path: Path,
    scene: str,
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
    sim3 = json.loads(sim3_path.read_text(encoding="utf-8"))

    require(contract.get("protocol_id") == PROTOCOL_ID, "contract protocol mismatch")
    require(contract.get("status") == "ACTIVE_FROZEN", "contract is not ACTIVE_FROZEN")
    require(contract.get("execution_authorized") is True, "contract execution is not authorized")
    require(activation.get("schema") == "m3m_gcp_lidar_formal_activation_v1", "activation schema mismatch")
    require(activation.get("protocol_id") == PROTOCOL_ID, "activation protocol mismatch")
    require(activation.get("review_verdict") == "PASS_LIDAR_V1_AND_SIX_SCENE_PREPARATION", "activation review verdict mismatch")
    require(activation.get("execution_authorized") is True, "activation does not authorize execution")
    require(activation.get("contract_file_sha256") == sha256_file(contract_path), "activation contract SHA mismatch")
    require(activation.get("artifact_schema_sha256") == sha256_file(schema_path), "activation artifact-schema SHA mismatch")
    require(activation.get("canonical_sha256") == canonical_sha256(activation), "activation canonical SHA mismatch")
    require(schema.get("schema") == "m3m_gcp_lidar_formal_artifact_schema_v1", "artifact schema mismatch")

    implementation = contract.get("implementation", {})
    repo_files = {
        "evaluator": repo / implementation.get("evaluator_path", "__missing__"),
        "verifier": repo / implementation.get("verifier_path", "__missing__"),
        "artifact_schema": repo / implementation.get("artifact_schema_path", "__missing__"),
        "ranker": repo / implementation.get("ranker_path", "__missing__"),
        "launch_gate": repo / implementation.get("launch_gate_path", "__missing__"),
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
    if input_manifest_path.is_file():
        input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
        require(input_manifest.get("scene") == scene, "formal input scene mismatch")
        require(input_manifest.get("release_root_digest_sha256") == contract["source_data_release"]["release_root_digest_sha256"], "formal input release mismatch")
        require(input_manifest.get("manifest_sha256") == canonical_sha256(input_manifest, self_field="manifest_sha256"), "formal input canonical SHA mismatch")
        require(input_manifest.get("train_view_count") == scene_rows.get(scene, {}).get("train_views"), "formal input train-view count mismatch")
        for filename, expected_sha in input_manifest.get("source_model_sha256", {}).items():
            path = colmap_model / filename
            require(path.is_file(), f"COLMAP model file missing: {filename}")
            if path.is_file():
                require(sha256_file(path) == expected_sha, f"COLMAP model SHA mismatch: {filename}")

    split_rows = {row["scene"]: row for row in split.get("scenes", [])}
    train_names = sorted(split_rows.get(scene, {}).get("train_image_names", []))
    require(len(train_names) == scene_rows.get(scene, {}).get("train_views"), "frozen train-view count mismatch")
    methods_rows = methods.get("methods", [])
    require(methods.get("schema") == "m3m_gcp_lidar_formal_methods_v1", "methods schema mismatch")
    require(methods.get("protocol_id") == PROTOCOL_ID, "methods protocol mismatch")
    require(methods.get("scene") == scene, "methods scene mismatch")
    require(methods.get("canonical_sha256") == canonical_sha256(methods), "methods canonical SHA mismatch")
    require(bool(methods_rows), "methods list is empty")
    expected_method_fields = set(
        schema.get("formal_methods_manifest", {}).get("method_fields_exact", [])
    )
    seen: set[str] = set()
    for row in methods_rows:
        require(set(row) == expected_method_fields, f"method fields differ from artifact schema: {row.get('method_id')}")
        method_id = row.get("method_id")
        require(method_id in ACTIVE_METHOD_CLASSES, f"unknown method: {method_id}")
        require(method_id not in seen, f"duplicate method: {method_id}")
        seen.add(method_id)
        require(row.get("input_class") == ACTIVE_METHOD_CLASSES.get(method_id), f"{method_id}: input class mismatch")
        for field in (
            "run_root",
            "model_checkpoint_path",
            "model_checkpoint_sha256",
            "recipe_path",
            "recipe_sha256",
            "renderer_adapter_path",
            "renderer_adapter_sha256",
            "packet_manifest_path",
            "packet_manifest_sha256",
        ):
            require(bool(row.get(field)), f"{method_id}: missing {field}")
        for path_field, sha_field in (
            ("model_checkpoint_path", "model_checkpoint_sha256"),
            ("recipe_path", "recipe_sha256"),
            ("renderer_adapter_path", "renderer_adapter_sha256"),
            ("packet_manifest_path", "packet_manifest_sha256"),
        ):
            path = Path(str(row.get(path_field, "")))
            require(path.is_file(), f"{method_id}: missing {path_field}")
            if path.is_file():
                require(sha256_file(path) == row.get(sha_field), f"{method_id}: {sha_field} mismatch")

    expected_commit = activation.get("benchmark_commit")
    expected_tree = activation.get("benchmark_tree")
    require(git_value(repo, "rev-parse", "HEAD") == expected_commit, "benchmark commit mismatch")
    require(git_value(repo, "show", "-s", "--format=%T", "HEAD") == expected_tree, "benchmark tree mismatch")
    require(git_value(repo, "status", "--porcelain") == "", "benchmark checkout is dirty")
    require(output_root.is_absolute(), "output root must be absolute")
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
    parser.add_argument("--gcp-csv", type=Path, required=True)
    parser.add_argument("--sim3", type=Path, required=True)
    parser.add_argument("--methods", type=Path, required=True)
    parser.add_argument("--scene", required=True)
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
        gcp_path=args.gcp_csv.resolve(),
        sim3_path=args.sim3.resolve(),
        methods_path=args.methods.resolve(),
        scene=args.scene,
        output_root=args.output_root,
    )
    print(json.dumps({"status": "PASS_READY" if not errors else "FAIL", "errors": errors}, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
