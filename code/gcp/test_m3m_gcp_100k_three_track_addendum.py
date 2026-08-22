#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import build_m3m_gcp_100k_three_track_candidate as candidate_builder
from m3m_gcp_100k_three_track_runtime import validate_addendum_runtime
from m3m_gcp_lidar_artifacts import canonical_sha256, sha256_file
from metric_depth_packet import directory_tree_hash


REPO = Path(__file__).resolve().parents[2]
ADDENDUM = REPO / "configs" / "m3m_gcp_native_quarter_100k_three_track_evaluation_addendum_v1.json"


class ThreeTrackAddendumStaticTest(unittest.TestCase):
    def test_addendum_canonical_bound_files_and_separate_lifecycles(self) -> None:
        payload = json.loads(ADDENDUM.read_text(encoding="utf-8"))
        self.assertEqual(payload["canonical_sha256"], canonical_sha256(payload))
        for relative, expected_sha in payload["bound_addendum_files"].items():
            self.assertEqual(sha256_file(REPO / relative), expected_sha, relative)
        gcp = payload["tracks"]["gcp"]
        self.assertEqual(gcp["protocol_observation_count"], 256)
        self.assertEqual(gcp["packet_camera_count"], 211)
        self.assertEqual(gcp["formal_role_counts"], {"train": 187, "test": 24})
        self.assertFalse(gcp["real_rgb_pixels_present"])
        self.assertEqual(payload["tracks"]["lidar"]["packet_train_view_count"], 2196)
        self.assertFalse(payload["rolling_packet_lifecycle"]["gcp_and_lidar_raw_packets_may_coexist"])

    def test_100k_rgb_contract_changes_only_binding_derivation_and_gate(self) -> None:
        base = json.loads(
            (REPO / "configs" / "m3m_gcp_native_quarter_rgb_quality_v1.json").read_text(
                encoding="utf-8"
            )
        )
        extended = json.loads(
            (REPO / "configs" / "m3m_gcp_native_quarter_rgb_quality_100k_v1.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            extended["derivation"]["source_contract_sha256"],
            sha256_file(REPO / "configs" / "m3m_gcp_native_quarter_rgb_quality_v1.json"),
        )
        base_clean = copy.deepcopy(base)
        extended_clean = copy.deepcopy(extended)
        extended_clean.pop("derivation")
        base_clean["input_binding"]["scene_bindings"] = {}
        extended_clean["input_binding"]["scene_bindings"] = {}
        extended_clean["formal_gate"].pop("three_track_addendum_activation_required")
        extended_clean["formal_gate"]["activation_preflight"] = base_clean["formal_gate"][
            "activation_preflight"
        ]
        extended_clean["formal_gate"]["benchmark_checkout_identity"] = base_clean[
            "formal_gate"
        ]["benchmark_checkout_identity"]
        self.assertEqual(extended_clean, base_clean)

    def test_obsolete_authorization_only_release_path_is_removed(self) -> None:
        self.assertFalse(
            (REPO / "code" / "gcp" / "authorize_m3m_gcp_100k_three_track_packet_release.py").exists()
        )
        finalizer = (REPO / "code" / "gcp" / "finalize_m3m_gcp_100k_three_track_results.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("GCP deletion-receipt coverage must equal new READY methods", finalizer)
        self.assertIn("LiDAR deletion-receipt coverage must equal all READY methods", finalizer)

    def test_rgb_formal_commands_require_fresh_pass_ready_preflight(self) -> None:
        source = (
            REPO / "code" / "gcp" / "build_m3m_native_quarter_rgb_quality_100k_commands.py"
        ).read_text(encoding="utf-8")
        self.assertIn('parser.add_argument("--activation-preflight", type=Path, required=True)', source)
        self.assertIn('preflight.get("status") != "PASS_READY"', source)
        self.assertIn("fresh PASS_READY 100K RGB preflight is absent or stale", source)


class ThreeTrackAddendumNegativeTest(unittest.TestCase):
    def test_wrong_citygs_x_main_model_is_rejected_even_if_aux_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "iteration_100000"
            checkpoint.mkdir()
            main = checkpoint / "point_cloud.ply"
            aux = checkpoint / "additional_attributes.npz"
            weights = checkpoint / "checkpoints.pth"
            main.write_bytes(b"frozen-main")
            aux.write_bytes(b"same-aux")
            weights.write_bytes(b"same-weights")
            expected = directory_tree_hash(checkpoint)
            aux_sha = sha256_file(aux)
            main.write_bytes(b"wrong-main")
            observed = directory_tree_hash(checkpoint)
            self.assertEqual(sha256_file(aux), aux_sha)
            self.assertNotEqual(observed, expected)

    def _runtime_fixture(self, root: Path) -> tuple[dict, dict, dict, Path]:
        repo = root / "reviewed"
        executable = repo / "code" / "gcp" / "fake_runtime.py"
        config_path = repo / "configs" / "addendum.json"
        executable.parent.mkdir(parents=True)
        config_path.parent.mkdir(parents=True)
        executable.write_text("print('frozen')\n", encoding="utf-8")
        config = {
            "bound_addendum_files": {
                "code/gcp/fake_runtime.py": sha256_file(executable),
            }
        }
        config["canonical_sha256"] = canonical_sha256(config)
        config_path.write_text(json.dumps(config, sort_keys=True) + "\n", encoding="utf-8")
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.invalid"], check=True)
        subprocess.run(["git", "-C", str(repo), "config", "user.name", "Protocol Test"], check=True)
        subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
        subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "frozen"], check=True)
        commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        tree = subprocess.check_output(
            ["git", "-C", str(repo), "show", "-s", "--format=%T", "HEAD"], text=True
        ).strip()
        activation = {"reviewed_addendum_commit": commit, "reviewed_addendum_tree": tree}
        candidate = {
            "addendum_config": {
                "path": str(config_path.resolve()),
                "sha256": sha256_file(config_path),
                "canonical_sha256": config["canonical_sha256"],
            }
        }
        registry = {"shared": {"benchmark_repo_template": str(repo.resolve())}}
        return activation, candidate, registry, executable

    def test_runtime_gate_rejects_source_or_adapter_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            activation, candidate, registry, executable = self._runtime_fixture(Path(temporary))
            validate_addendum_runtime(
                activation=activation,
                candidate=candidate,
                registry=registry,
                executing_file=executable,
            )
            executable.write_text("print('tampered')\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                validate_addendum_runtime(
                    activation=activation,
                    candidate=candidate,
                    registry=registry,
                    executing_file=executable,
                )

    def test_runtime_gate_rejects_stale_other_activation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            activation, candidate, registry, executable = self._runtime_fixture(Path(temporary))
            activation["reviewed_addendum_commit"] = "0" * 40
            with self.assertRaises(RuntimeError):
                validate_addendum_runtime(
                    activation=activation,
                    candidate=candidate,
                    registry=registry,
                    executing_file=executable,
                )

    def test_gcp_camera_root_rejects_named_image_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "camera"
            images = root / "images"
            sparse = root / "sparse" / "0"
            images.mkdir(parents=True)
            sparse.mkdir(parents=True)
            placeholder = root / "BLACK_CAMERA_LOADER_PLACEHOLDER.png"
            placeholder.write_bytes(b"placeholder")
            names = [f"view_{index:03d}.png" for index in range(211)]
            for name in names:
                (images / name).write_bytes(b"placeholder")
            sparse_rows = {}
            for name in ("cameras.bin", "images.bin", "points3D.bin", "points3D.ply"):
                path = sparse / name
                path.write_bytes(name.encode("ascii"))
                sparse_rows[name] = {
                    "path": str(path.resolve()),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            payload = {
                "schema": "m3m_gcp_100k_gcp_evaluation_camera_root_v1",
                "status": "PASS_GCP_EVALUATION_CAMERA_ROOT_NO_RGB_PIXELS",
                "scene": candidate_builder.SCENE,
                "protocol_id": candidate_builder.GCP_PROTOCOL_ID,
                "formal_input_manifest": {
                    "sha256": candidate_builder.FORMAL_INPUT_SHA,
                    "canonical_sha256": candidate_builder.FORMAL_INPUT_CANONICAL_SHA,
                },
                "protocol_observations": {
                    "observation_count": 256,
                    "unique_camera_count": 211,
                    "formal_role_counts": {"train": 187, "test": 24},
                },
                "output": {
                    "root": str(root.resolve()),
                    "camera_view_count": 211,
                    "image_names": names,
                    "placeholder": {"sha256": sha256_file(placeholder)},
                    "sparse_files": sparse_rows,
                },
                "rgb_truth_boundary": {"real_rgb_pixels_present": False},
            }
            payload["canonical_sha256"] = canonical_sha256(payload)
            manifest = root / "GCP_EVALUATION_CAMERA_ROOT_MANIFEST.json"
            manifest.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
            formal = {
                "images": [
                    {"image_name": name, "role": "train" if index < 187 else "test"}
                    for index, name in enumerate(names)
                ]
            }
            candidate_builder.validate_gcp_camera_root(manifest, formal_input=formal)
            (images / names[0]).write_bytes(b"real-or-tampered-pixels")
            with self.assertRaises(RuntimeError):
                candidate_builder.validate_gcp_camera_root(manifest, formal_input=formal)

    def test_legacy_metadata_rejects_wrong_common_sim3_protocol_and_root(self) -> None:
        model_sha = "a" * 64
        prelaunch = {
            "status": "PASS",
            "scene": candidate_builder.SCENE,
            "method_id": "3dgs_original",
            "formal_model_ply_sha256": model_sha,
        }
        summary = {
            "status": "COMPLETE_RANKED",
            "ranking_eligible": True,
            "scene": candidate_builder.SCENE,
            "method_id": "3dgs_original",
            "protocol_id": candidate_builder.GCP_PROTOCOL_ID,
            "common_sim3_sha256": candidate_builder.COMMON_SIM3_SHA,
            "method_specific_sim3_fitted": False,
            "residual_statistics": {},
        }
        evaluator = {
            "schema": "m3m_gcp_native_quarter_evaluator_run_manifest_v2",
            "protocol_release_manifest_sha256": candidate_builder.GCP_PROTOCOL_RELEASE_SHA,
            "source_data_contract_sha256": candidate_builder.GCP_DATA_CONTRACT_SHA,
            "sim3_policy": "frozen_common_transform_no_method_refit",
        }
        verifier = {
            "status": "PASS",
            "passed": True,
            "scene": candidate_builder.SCENE,
            "method_id": "3dgs_original",
            "ranking_status": "COMPLETE_RANKED",
            "recomputed_residual_statistics": {},
            "method_specific_sim3_fitted": False,
            "common_sim3_recomputation_passed": True,
            "dependency_hashes_passed": True,
            "output_hashes_passed": True,
        }
        candidate_builder.validate_legacy_result_metadata(
            prelaunch=prelaunch,
            summary=summary,
            evaluator_manifest=evaluator,
            verifier=verifier,
            formal_model_sha256=model_sha,
        )
        bad_summary = dict(summary)
        bad_summary["common_sim3_sha256"] = "0" * 64
        with self.assertRaises(RuntimeError):
            candidate_builder.validate_legacy_result_metadata(
                prelaunch=prelaunch,
                summary=bad_summary,
                evaluator_manifest=evaluator,
                verifier=verifier,
                formal_model_sha256=model_sha,
            )
        release = {
            "protocol_id": candidate_builder.GCP_PROTOCOL_ID,
            "method_result_sim3_refit_allowed": False,
            "source_data": {"data_contract_sha256": candidate_builder.GCP_DATA_CONTRACT_SHA},
        }
        rows = {
            "observation_semantics.csv": {"sha256": candidate_builder.OBSERVATION_SEMANTICS_SHA},
            f"scenes/{candidate_builder.SCENE}/common_sim3.json": {
                "sha256": candidate_builder.COMMON_SIM3_SHA
            },
            f"scenes/{candidate_builder.SCENE}/triangulation_observation_residuals.csv": {
                "sha256": "b" * 64
            },
        }
        candidate_builder.validate_legacy_protocol_dependencies(
            release=release, payload_rows=rows, scene_observations_sha256="b" * 64
        )
        bad_release = dict(release)
        bad_release["protocol_id"] = "other"
        with self.assertRaises(RuntimeError):
            candidate_builder.validate_legacy_protocol_dependencies(
                release=bad_release, payload_rows=rows, scene_observations_sha256="b" * 64
            )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "legacy GCP root differs"):
                candidate_builder.build_legacy_adoption(
                    base_repo=root,
                    addendum_config={"legacy_3dgs_gcp": {"root": str(root / "expected")}},
                    methods={},
                    freeze_path=root / "freeze.json",
                    freeze={},
                    formal_input={},
                    legacy_root=root / "wrong",
                )


if __name__ == "__main__":
    unittest.main()
