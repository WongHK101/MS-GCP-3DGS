#!/usr/bin/env python3
"""Fail-closed formal-launch gate tests using a clean temporary Git repo."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from check_m3m_gcp_lidar_formal_launch import (
    ACTIVE_METHOD_CLASSES,
    canonical_sha256,
    sha256_file,
    validate_launch,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


class LaunchGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        for rel in ("code/evaluator.py", "code/verifier.py", "code/ranker.py", "code/launch.py"):
            path = self.repo / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(rel, encoding="utf-8")
        schema_path = self.repo / "schema.json"
        method_fields = [
            "method_id", "method_name", "input_class", "run_root",
            "model_checkpoint_path", "model_checkpoint_sha256",
            "recipe_path", "recipe_sha256", "renderer_adapter_path",
            "renderer_adapter_sha256", "packet_manifest_path", "packet_manifest_sha256",
        ]
        write_json(schema_path, {
            "schema": "m3m_gcp_lidar_formal_artifact_schema_v1",
            "formal_methods_manifest": {"method_fields_exact": method_fields},
        })
        registry_path = self.repo / "registry.json"
        write_json(registry_path, {
            "active_benchmark_method_ids": list(ACTIVE_METHOD_CLASSES),
            "methods": [
                {"method_id": key, "input_class": value}
                for key, value in ACTIVE_METHOD_CLASSES.items()
            ],
        })
        split_path = self.repo / "split.json"
        write_json(split_path, {"scenes": [{"scene": "scene", "train_image_names": ["a"]}]})
        release_pin = self.repo / "release_pin.json"
        release_pin.write_text("pin", encoding="utf-8")
        geometry_root = self.root / "geometry"
        geometry_root.mkdir()
        release_manifest = geometry_root / "protocol_release_manifest.json"
        release_manifest.write_text("manifest", encoding="utf-8")
        gcp = self.root / "gcp.csv"
        gcp.write_text("point_name\nA\n", encoding="utf-8")
        lidar_inventory = self.root / "lidar.csv"
        lidar_inventory.write_text("path,sha256\n", encoding="utf-8")
        sim3 = self.root / "sim3.json"
        write_json(sim3, {
            "protocol_id": "m3m_gcp_native_quarter_geometry_v2",
            "scene": "scene",
            "method_result_refit_forbidden": True,
        })
        formal_input = self.root / "formal_input"
        colmap = formal_input / "train" / "sparse" / "0"
        colmap.mkdir(parents=True)
        (colmap / "cameras.bin").write_bytes(b"camera")
        input_manifest = {
            "scene": "scene",
            "release_root_digest_sha256": "release",
            "train_view_count": 1,
            "source_model_sha256": {"cameras.bin": sha256_file(colmap / "cameras.bin")},
        }
        input_manifest["manifest_sha256"] = canonical_sha256(input_manifest, self_field="manifest_sha256")
        write_json(formal_input / "NATIVE_QUARTER_INPUT_MANIFEST.json", input_manifest)
        model = self.repo / "model.bin"
        recipe = self.repo / "recipe.json"
        adapter = self.repo / "adapter.json"
        packet = self.repo / "packet.json"
        for path, content in ((model, "model"), (recipe, "recipe"), (adapter, "adapter"), (packet, "packet")):
            path.write_text(content, encoding="utf-8")
        methods_path = self.repo / "methods.json"
        method_row = {
            "method_id": "3dgs_original",
            "method_name": "3DGS",
            "input_class": "rgb_colmap_only",
            "run_root": str(self.repo),
            "model_checkpoint_path": str(model),
            "model_checkpoint_sha256": sha256_file(model),
            "recipe_path": str(recipe),
            "recipe_sha256": sha256_file(recipe),
            "renderer_adapter_path": str(adapter),
            "renderer_adapter_sha256": sha256_file(adapter),
            "packet_manifest_path": str(packet),
            "packet_manifest_sha256": sha256_file(packet),
        }
        methods = {
            "schema": "m3m_gcp_lidar_formal_methods_v1",
            "protocol_id": "m3m_gcp_lidar_rendered_surface_v1",
            "scene": "scene",
            "methods": [method_row],
        }
        methods["canonical_sha256"] = canonical_sha256(methods)
        write_json(methods_path, methods)
        contract_path = self.repo / "contract.json"
        contract = {
            "protocol_id": "m3m_gcp_lidar_rendered_surface_v1",
            "status": "ACTIVE_FROZEN",
            "execution_authorized": True,
            "source_geometry_protocol_id": "m3m_gcp_native_quarter_geometry_v2",
            "source_data_release": {
                "split_manifest_file_sha256": sha256_file(split_path),
                "release_root_digest_sha256": "release",
            },
            "source_geometry_binding": {
                "release_pin_path": "release_pin.json",
                "release_pin_sha256": sha256_file(release_pin),
                "release_manifest_relative_path": "protocol_release_manifest.json",
                "release_manifest_sha256": sha256_file(release_manifest),
                "gcp_points_sha256": sha256_file(gcp),
                "scene_common_sim3_sha256": {"scene": sha256_file(sim3)},
            },
            "lidar_source": {"payload_sha256_inventory_file_sha256": sha256_file(lidar_inventory)},
            "method_registry_binding": {
                "file_sha256": sha256_file(registry_path),
                "active_method_input_classes": ACTIVE_METHOD_CLASSES,
            },
            "scenes": [{"scene": "scene", "train_views": 1}],
            "implementation": {
                "evaluator_path": "code/evaluator.py", "evaluator_sha256": sha256_file(self.repo / "code/evaluator.py"),
                "verifier_path": "code/verifier.py", "verifier_sha256": sha256_file(self.repo / "code/verifier.py"),
                "artifact_schema_path": "schema.json", "artifact_schema_sha256": sha256_file(schema_path),
                "ranker_path": "code/ranker.py", "ranker_sha256": sha256_file(self.repo / "code/ranker.py"),
                "launch_gate_path": "code/launch.py", "launch_gate_sha256": sha256_file(self.repo / "code/launch.py"),
            },
        }
        write_json(contract_path, contract)
        subprocess.run(["git", "-C", str(self.repo), "add", "."], check=True)
        subprocess.run([
            "git", "-C", str(self.repo), "-c", "user.name=Test", "-c",
            "user.email=test@example.invalid", "commit", "-qm", "fixture",
        ], check=True)
        commit = subprocess.check_output(["git", "-C", str(self.repo), "rev-parse", "HEAD"], text=True).strip()
        tree = subprocess.check_output(["git", "-C", str(self.repo), "show", "-s", "--format=%T", "HEAD"], text=True).strip()
        activation_path = self.root / "activation.json"
        activation = {
            "schema": "m3m_gcp_lidar_formal_activation_v1",
            "protocol_id": "m3m_gcp_lidar_rendered_surface_v1",
            "review_verdict": "PASS_LIDAR_V1_AND_SIX_SCENE_PREPARATION",
            "execution_authorized": True,
            "contract_file_sha256": sha256_file(contract_path),
            "artifact_schema_sha256": sha256_file(schema_path),
            "benchmark_commit": commit,
            "benchmark_tree": tree,
        }
        activation["canonical_sha256"] = canonical_sha256(activation)
        write_json(activation_path, activation)
        self.kwargs = {
            "repo": self.repo, "contract_path": contract_path,
            "activation_path": activation_path, "schema_path": schema_path,
            "split_path": split_path, "registry_path": registry_path,
            "geometry_release_root": geometry_root, "formal_input_root": formal_input,
            "colmap_model": colmap, "lidar_inventory_path": lidar_inventory,
            "gcp_path": gcp, "sim3_path": sim3, "methods_path": methods_path,
            "scene": "scene", "output_root": self.root / "new-output",
        }

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_clean_active_launch_passes(self) -> None:
        self.assertEqual(validate_launch(**self.kwargs), [])

    def test_existing_output_is_rejected(self) -> None:
        self.kwargs["output_root"].mkdir()
        errors = validate_launch(**self.kwargs)
        self.assertTrue(any("already exists" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
