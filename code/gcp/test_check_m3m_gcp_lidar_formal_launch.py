#!/usr/bin/env python3
"""Fail-closed launch-gate tests with real LAZ/NPZ byte bindings."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from check_m3m_gcp_lidar_formal_launch import (
    ACTIVE_METHOD_CLASSES, canonical_sha256, sha256_file, validate_launch,
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

        self.schema_path = self.repo / "schema.json"
        method_fields = [
            "method_id", "method_name", "input_class", "run_root",
            "model_checkpoint_path", "model_checkpoint_sha256", "recipe_path", "recipe_sha256",
            "renderer_adapter_path", "renderer_adapter_sha256",
            "attempt_status", "failure_evidence_path", "failure_evidence_sha256",
        ]
        auth_fields = [
            "schema", "protocol_id", "scene", "selected_method_id", "review_task_id", "review_verdict",
            "execution_authorized", "contract_file_sha256", "activation_manifest_sha256",
            "artifact_schema_sha256", "formal_input_manifest_file_sha256",
            "formal_input_manifest_canonical_sha256", "methods_manifest_file_sha256",
            "methods_manifest_canonical_sha256", "packet_manifest_path", "packet_manifest_sha256",
            "benchmark_commit", "benchmark_tree", "authorized_output_root", "canonical_sha256",
        ]
        packet_keys = [
            "accumulated_alpha", "weighted_camera_z_sum", "weighted_camera_z_second_moment",
            "weighted_inverse_camera_z_sum", "alpha_normalized_expected_camera_z",
            "alpha_normalized_expected_inverse_camera_z", "harmonic_camera_z",
            "camera_z_variance", "metric_depth_valid_mask",
            "historical_invalid_unnormalized_inverse_depth",
        ]
        write_json(self.schema_path, {
            "schema": "m3m_gcp_lidar_formal_artifact_schema_v1",
            "formal_methods_manifest": {"method_fields_exact": method_fields},
            "scene_execution_authorization": {"required_fields_exact": auth_fields},
            "depth_packet_manifest": {"packet_npz": {
                "keys_exact": packet_keys,
                "depth_index_fields_required": [
                    "image_name", "split", "width", "height", "packet_path",
                    "packet_bytes", "packet_sha256", "dtype", "tensor_names",
                ],
            }},
        })
        self.registry_path = self.repo / "registry.json"
        write_json(self.registry_path, {
            "active_benchmark_method_ids": list(ACTIVE_METHOD_CLASSES),
            "methods": [
                {"method_id": key, "display_name": key, "input_class": value}
                for key, value in ACTIVE_METHOD_CLASSES.items()
            ],
        })
        self.split_path = self.repo / "split.json"
        write_json(self.split_path, {"scenes": [{"scene": "scene", "train_image_names": ["a.JPG"]}]})
        self.release_pin = self.repo / "release_pin.json"
        self.release_pin.write_text("pin", encoding="utf-8")
        self.geometry_root = self.root / "geometry"
        self.geometry_root.mkdir()
        self.release_manifest = self.geometry_root / "protocol_release_manifest.json"
        self.release_manifest.write_text("manifest", encoding="utf-8")
        self.gcp = self.root / "gcp.csv"
        self.gcp.write_text("point_name\nA\n", encoding="utf-8")
        self.sim3 = self.root / "sim3.json"
        write_json(self.sim3, {
            "protocol_id": "m3m_gcp_native_quarter_geometry_v2",
            "scene": "scene", "method_result_refit_forbidden": True,
        })

        self.formal_input = self.root / "formal_input"
        self.colmap = self.formal_input / "train" / "sparse" / "0"
        self.colmap.mkdir(parents=True)
        for name in ("cameras.bin", "images.bin", "points3D.bin", "points3D.ply"):
            (self.colmap / name).write_bytes(name.encode())
        train_image = self.formal_input / "train" / "images" / "a.JPG"
        train_image.parent.mkdir(parents=True)
        train_image.write_bytes(b"image")
        input_manifest = {
            "scene": "scene", "release_root_digest_sha256": "release", "train_view_count": 1,
            "source_model_sha256": {
                name: sha256_file(self.colmap / name)
                for name in ("cameras.bin", "images.bin", "points3D.bin", "points3D.ply")
            },
            "roles": [{"role": "train", "root": "train", "camera_count": 1, "image_count": 1, "points2d_tracks_present": False, "points3d_bin_present": False, "cameras_bin_sha256": sha256_file(self.colmap / "cameras.bin"), "images_bin_sha256": sha256_file(self.colmap / "images.bin"), "points3d_ply_sha256": sha256_file(self.colmap / "points3D.ply")}],
            "images": [{"role": "train", "image_name": "a.JPG", "relative_path": "train/images/a.JPG", "jpeg_bytes": train_image.stat().st_size, "jpeg_sha256": sha256_file(train_image)}],
        }
        input_manifest["manifest_sha256"] = canonical_sha256(input_manifest, self_field="manifest_sha256")
        self.input_manifest_path = self.formal_input / "NATIVE_QUARTER_INPUT_MANIFEST.json"
        write_json(self.input_manifest_path, input_manifest)

        self.lidar_root = self.root / "lidar"
        laz = self.lidar_root / "lidars" / "terra_laz_1_4" / "cloud0.laz"
        laz.parent.mkdir(parents=True)
        laz.write_bytes(b"frozen-laz")
        self.lidar_inventory = self.lidar_root / "inventory.csv"
        self.lidar_inventory.write_text(
            "relative_path_utf8_nfc,bytes,sha256\n"
            f"lidars/terra_laz_1_4/cloud0.laz,{laz.stat().st_size},{sha256_file(laz)}\n",
            encoding="utf-8",
        )

        packets_dir = self.root / "assets" / "3dgs_original" / "formal_evaluation" / "packets"
        packets_dir.mkdir(parents=True)
        self.packet_path = packets_dir / "a_metric_depth_packet.npz"
        arrays = {key: np.ones((2, 3), dtype=np.float32) for key in packet_keys}
        arrays["metric_depth_valid_mask"] = np.ones((2, 3), dtype=np.bool_)
        np.savez(self.packet_path, **arrays)
        packet_manifest = {
            "schema": "ms_gcp_metric_depth_packet_manifest_v2",
            "protocol_id": "m3m_gcp_native_quarter_geometry_v2", "scene": "scene",
            "primary_depth_tensor": "alpha_normalized_expected_camera_z",
            "primary_depth_semantics": "camera_z",
            "image_domain": "colmap_4_0_4_image_undistorter_pinhole_max_1414",
            "pixel_coordinate_convention": "zero_based_pixel_centers",
            "camera_z_unit_contract": "frozen_colmap_model_camera_z_units",
            "adapter_conformance_status": "PASS", "camera_sets": "train", "rendered_view_count": 1,
            "depth_index": [{
                "image_name": "a.JPG", "split": "train", "width": 3, "height": 2,
                "packet_path": str(self.packet_path), "packet_bytes": self.packet_path.stat().st_size,
                "packet_sha256": sha256_file(self.packet_path), "dtype": "float32",
                "tensor_names": "|".join(packet_keys),
            }],
        }
        self.packet_manifest_path = packets_dir / "depth_export_manifest.json"
        write_json(self.packet_manifest_path, packet_manifest)

        rows = []
        for method_id, input_class in ACTIVE_METHOD_CLASSES.items():
            asset_root = self.root / "assets" / method_id
            asset_root.mkdir(parents=True, exist_ok=True)
            model, recipe, adapter = (asset_root / name for name in ("model", "recipe", "adapter"))
            for path in (model, recipe, adapter):
                path.write_text(f"{method_id}-{path.name}", encoding="utf-8")
            rows.append({
                "method_id": method_id, "method_name": method_id, "input_class": input_class,
                "attempt_status": "READY_FOR_EVALUATION",
                "run_root": str(asset_root), "model_checkpoint_path": str(model),
                "model_checkpoint_sha256": sha256_file(model), "recipe_path": str(recipe),
                "recipe_sha256": sha256_file(recipe), "renderer_adapter_path": str(adapter),
                "renderer_adapter_sha256": sha256_file(adapter),
                "failure_evidence_path": None, "failure_evidence_sha256": None,
            })
        methods = {"schema": "m3m_gcp_lidar_formal_methods_v1", "protocol_id": "m3m_gcp_lidar_rendered_surface_v1", "scene": "scene", "methods": rows}
        methods["canonical_sha256"] = canonical_sha256(methods)
        self.methods_path = self.root / "methods.json"
        write_json(self.methods_path, methods)

        self.contract_path = self.repo / "contract.json"
        contract = {
            "protocol_id": "m3m_gcp_lidar_rendered_surface_v1", "status": "ACTIVE_FROZEN",
            "execution_authorized": True, "source_geometry_protocol_id": "m3m_gcp_native_quarter_geometry_v2",
            "review": {"review_task_id": "review"},
            "source_data_release": {"split_manifest_file_sha256": sha256_file(self.split_path), "release_root_digest_sha256": "release"},
            "formal_input_binding": {"manifest_filename": "NATIVE_QUARTER_INPUT_MANIFEST.json", "source_model_files_exact": ["cameras.bin", "images.bin", "points3D.bin", "points3D.ply"], "execution_input_bytes": "exact train role cameras.bin, images.bin, points3D.ply and every train JPEG are rehashed from the externally bound manifest", "scene_manifests": {"scene": {"file_sha256": sha256_file(self.input_manifest_path), "canonical_sha256": input_manifest["manifest_sha256"]}}},
            "source_geometry_binding": {"release_pin_path": "release_pin.json", "release_pin_sha256": sha256_file(self.release_pin), "release_manifest_relative_path": "protocol_release_manifest.json", "release_manifest_sha256": sha256_file(self.release_manifest), "gcp_points_sha256": sha256_file(self.gcp), "scene_common_sim3_sha256": {"scene": sha256_file(self.sim3)}},
            "lidar_source": {"payload_sha256_inventory_file_sha256": sha256_file(self.lidar_inventory), "laz_files_exact": {"lidars/terra_laz_1_4/cloud0.laz": {"bytes": laz.stat().st_size, "sha256": sha256_file(laz)}}},
            "method_registry_binding": {"file_sha256": sha256_file(self.registry_path), "active_method_input_classes": ACTIVE_METHOD_CLASSES},
            "scenes": [{"scene": "scene", "train_views": 1}],
            "implementation": {
                "evaluator_path": "code/evaluator.py", "evaluator_sha256": sha256_file(self.repo / "code/evaluator.py"),
                "verifier_path": "code/verifier.py", "verifier_sha256": sha256_file(self.repo / "code/verifier.py"),
                "artifact_schema_path": "schema.json", "artifact_schema_sha256": sha256_file(self.schema_path),
                "ranker_path": "code/ranker.py", "ranker_sha256": sha256_file(self.repo / "code/ranker.py"),
                "launch_gate_path": "code/launch.py", "launch_gate_sha256": sha256_file(self.repo / "code/launch.py"),
            },
        }
        write_json(self.contract_path, contract)
        subprocess.run(["git", "-C", str(self.repo), "add", "."], check=True)
        subprocess.run(["git", "-C", str(self.repo), "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture"], check=True)
        commit = subprocess.check_output(["git", "-C", str(self.repo), "rev-parse", "HEAD"], text=True).strip()
        tree = subprocess.check_output(["git", "-C", str(self.repo), "show", "-s", "--format=%T", "HEAD"], text=True).strip()
        self.activation_path = self.root / "activation.json"
        activation = {"schema": "m3m_gcp_lidar_formal_activation_v1", "protocol_id": "m3m_gcp_lidar_rendered_surface_v1", "review_verdict": "PASS_LIDAR_V1_AND_SIX_SCENE_PREPARATION", "execution_authorized": True, "contract_file_sha256": sha256_file(self.contract_path), "artifact_schema_sha256": sha256_file(self.schema_path), "benchmark_commit": commit, "benchmark_tree": tree}
        activation["canonical_sha256"] = canonical_sha256(activation)
        write_json(self.activation_path, activation)
        self.output_root = (self.root / "new-output").resolve()
        self.scene_authorization_path = self.root / "scene-authorization.json"
        authorization = {
            "schema": "m3m_gcp_lidar_scene_execution_authorization_v1", "protocol_id": "m3m_gcp_lidar_rendered_surface_v1", "scene": "scene", "selected_method_id": "3dgs_original", "review_task_id": "review", "review_verdict": "PASS_SCENE_EXECUTION_PLAN", "execution_authorized": True,
            "contract_file_sha256": sha256_file(self.contract_path), "activation_manifest_sha256": sha256_file(self.activation_path), "artifact_schema_sha256": sha256_file(self.schema_path),
            "formal_input_manifest_file_sha256": sha256_file(self.input_manifest_path), "formal_input_manifest_canonical_sha256": input_manifest["manifest_sha256"],
            "methods_manifest_file_sha256": sha256_file(self.methods_path), "methods_manifest_canonical_sha256": methods["canonical_sha256"],
            "packet_manifest_path": str(self.packet_manifest_path), "packet_manifest_sha256": sha256_file(self.packet_manifest_path),
            "benchmark_commit": commit, "benchmark_tree": tree, "authorized_output_root": str(self.output_root),
        }
        authorization["canonical_sha256"] = canonical_sha256(authorization)
        write_json(self.scene_authorization_path, authorization)
        self.kwargs = {
            "repo": self.repo, "contract_path": self.contract_path, "activation_path": self.activation_path,
            "schema_path": self.schema_path, "split_path": self.split_path, "registry_path": self.registry_path,
            "geometry_release_root": self.geometry_root, "formal_input_root": self.formal_input,
            "colmap_model": self.colmap, "lidar_inventory_path": self.lidar_inventory,
            "lidar_root": self.lidar_root, "gcp_path": self.gcp, "sim3_path": self.sim3,
            "methods_path": self.methods_path, "scene_authorization_path": self.scene_authorization_path,
            "scene": "scene", "selected_method_id": "3dgs_original", "output_root": self.output_root,
        }

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_clean_active_launch_passes(self) -> None:
        self.assertEqual(validate_launch(**self.kwargs), [])

    def test_existing_output_is_rejected(self) -> None:
        self.output_root.mkdir()
        self.assertTrue(any("already exists" in error for error in validate_launch(**self.kwargs)))

    def test_tampered_packet_bytes_are_rejected(self) -> None:
        self.packet_path.write_bytes(b"tampered")
        self.assertTrue(any("packet byte count mismatch" in error for error in validate_launch(**self.kwargs)))

    def test_packet_manifest_is_bound_per_selected_method(self) -> None:
        payload = json.loads(self.scene_authorization_path.read_text(encoding="utf-8"))
        payload["packet_manifest_sha256"] = "0" * 64
        payload["canonical_sha256"] = canonical_sha256(payload)
        write_json(self.scene_authorization_path, payload)
        self.assertTrue(any("packet manifest SHA mismatch" in error for error in validate_launch(**self.kwargs)))

    def test_selected_method_authorization_must_match_cli(self) -> None:
        payload = json.loads(self.scene_authorization_path.read_text(encoding="utf-8"))
        payload["selected_method_id"] = "2dgs"
        payload["canonical_sha256"] = canonical_sha256(payload)
        write_json(self.scene_authorization_path, payload)
        self.assertTrue(any("selected method mismatch" in error for error in validate_launch(**self.kwargs)))

    def test_tampered_laz_bytes_are_rejected(self) -> None:
        laz = self.lidar_root / "lidars" / "terra_laz_1_4" / "cloud0.laz"
        laz.write_bytes(b"tampered")
        self.assertTrue(any("LiDAR byte count mismatch" in error for error in validate_launch(**self.kwargs)))

    def test_methods_manifest_must_be_complete_ten(self) -> None:
        payload = json.loads(self.methods_path.read_text(encoding="utf-8"))
        payload["methods"].pop()
        payload["canonical_sha256"] = canonical_sha256(payload)
        write_json(self.methods_path, payload)
        self.assertTrue(any("exact ordered ten-method pool" in error for error in validate_launch(**self.kwargs)))

    def test_failed_peer_does_not_block_ready_selected_method(self) -> None:
        payload = json.loads(self.methods_path.read_text(encoding="utf-8"))
        peer = next(row for row in payload["methods"] if row["method_id"] == "2dgs")
        failure = self.root / "assets" / "2dgs" / "failure.json"
        failure.write_text("oom", encoding="utf-8")
        peer["attempt_status"] = "OOM_UNRANKED"
        peer["model_checkpoint_path"] = None
        peer["model_checkpoint_sha256"] = None
        peer["failure_evidence_path"] = str(failure)
        peer["failure_evidence_sha256"] = sha256_file(failure)
        payload["canonical_sha256"] = canonical_sha256(payload)
        write_json(self.methods_path, payload)
        self._rebind_methods_authorization(payload)
        self.assertEqual(validate_launch(**self.kwargs), [])

    def test_failed_method_cannot_be_selected_for_evaluation(self) -> None:
        payload = json.loads(self.methods_path.read_text(encoding="utf-8"))
        selected = next(row for row in payload["methods"] if row["method_id"] == "3dgs_original")
        failure = self.root / "assets" / "3dgs_original" / "failure.json"
        failure.write_text("oom", encoding="utf-8")
        selected["attempt_status"] = "OOM_UNRANKED"
        selected["model_checkpoint_path"] = None
        selected["model_checkpoint_sha256"] = None
        selected["failure_evidence_path"] = str(failure)
        selected["failure_evidence_sha256"] = sha256_file(failure)
        payload["canonical_sha256"] = canonical_sha256(payload)
        write_json(self.methods_path, payload)
        self._rebind_methods_authorization(payload)
        self.assertTrue(any("not ready for evaluation" in error for error in validate_launch(**self.kwargs)))

    def test_unreviewed_output_root_is_rejected(self) -> None:
        self.kwargs["output_root"] = (self.root / "unreviewed").resolve()
        self.assertTrue(any("not authorized" in error for error in validate_launch(**self.kwargs)))

    def _rebind_methods_authorization(self, methods: dict) -> None:
        authorization = json.loads(self.scene_authorization_path.read_text(encoding="utf-8"))
        authorization["methods_manifest_file_sha256"] = sha256_file(self.methods_path)
        authorization["methods_manifest_canonical_sha256"] = methods["canonical_sha256"]
        authorization["canonical_sha256"] = canonical_sha256(authorization)
        write_json(self.scene_authorization_path, authorization)


if __name__ == "__main__":
    unittest.main()
