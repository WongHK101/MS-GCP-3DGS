#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import build_m3m_gcp_100k_three_track_candidate as candidate_builder
from m3m_gcp_100k_three_track_runtime import (
    probe_gcp_evaluation_runtime,
    validate_addendum_runtime,
    validate_frozen_gcp_evaluation_runtime,
)
from m3m_gcp_lidar_artifacts import canonical_sha256, sha256_file
from metric_depth_packet import directory_tree_hash
from cleanup_m3m_gcp_100k_failed_packet import validate_failure_archive
from m3m_gcp_100k_lidar_archive import (
    expected_lidar_archive_relatives,
    validate_exact_lidar_archive,
)
from m3m_gcp_100k_raw_packet_state import (
    acquire_active_raw_packet_state,
    validate_active_raw_packet_state,
)
from release_m3m_gcp_100k_gcp_packet import validate_archive as validate_gcp_archive
from run_m3m_gcp_100k_gcp_packet_guarded import validate_frozen_packet_python


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
        self.assertTrue(
            payload["rolling_packet_lifecycle"][
                "shared_global_raw_packet_mutex_acquired_by_exclusive_create"
            ]
        )
        self.assertTrue(
            payload["failure_policy"][
                "packet_or_postprocessing_failure_requires_immutable_no_retry_cleanup_receipt"
            ]
        )

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

    def test_release_recovery_and_failed_cleanup_are_full_chain_bound(self) -> None:
        gcp_release = (
            REPO / "code" / "gcp" / "release_m3m_gcp_100k_gcp_packet.py"
        ).read_text(encoding="utf-8")
        lidar_release = (
            REPO / "code" / "gcp" / "release_m3m_gcp_100k_lidar_packet.py"
        ).read_text(encoding="utf-8")
        failed_cleanup = (
            REPO / "code" / "gcp" / "cleanup_m3m_gcp_100k_failed_packet.py"
        ).read_text(encoding="utf-8")
        self.assertIn("packet_tree_before_delete", gcp_release)
        self.assertIn("validate_archive(", gcp_release)
        self.assertIn("validate_active_raw_packet_state(", gcp_release)
        self.assertIn("validate_exact_lidar_archive(", lidar_release)
        self.assertIn("validate_active_raw_packet_state(", lidar_release)
        self.assertIn("existing failed-packet cleanup intent mismatch", failed_cleanup)
        self.assertIn("failed cleanup continuation packet subset mismatch", failed_cleanup)
        self.assertIn('"retry_forbidden": True', failed_cleanup)


class ThreeTrackAddendumNegativeTest(unittest.TestCase):
    def test_gcp_archive_requires_exact_inventory_and_retained_bytes(self) -> None:
        relatives = (
            "gcp_execution_authorization.json",
            "gcp_packet_phase_success.json",
            "gcp_packet_state.json",
            "active_raw_packet_state.json",
            "depth_export_manifest.json",
            "gcp_evaluation_execution_receipt.json",
            "evaluation/observation_samples.csv",
            "evaluation/point_results.csv",
            "evaluation/evaluation_summary.json",
            "evaluation/evaluator_manifest.json",
            "independent_verification.json",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_root = root / "source"
            archive_root = root / "archive"
            rows = []
            for index, relative in enumerate(relatives):
                source = source_root / relative
                archived = archive_root / relative
                source.parent.mkdir(parents=True, exist_ok=True)
                archived.parent.mkdir(parents=True, exist_ok=True)
                payload = f"frozen-{index}".encode("ascii")
                source.write_bytes(payload)
                archived.write_bytes(payload)
                rows.append(
                    {
                        "source_path": str(source.resolve()),
                        "archive_path": str(archived.resolve()),
                        "bytes": len(payload),
                        "sha256": sha256_file(archived),
                    }
                )
            manifest_payload = {
                "schema": "m3m_gcp_100k_gcp_lightweight_archive_v1",
                "status": "PASS_GCP_LIGHTWEIGHT_ARCHIVE_BYTE_VERIFIED",
                "archive_root": str(archive_root.resolve()),
                "files": rows,
                "source_and_archive_bytes_reverified": True,
                "raw_metric_depth_packet_files_archived": False,
            }
            manifest_payload["canonical_sha256"] = canonical_sha256(manifest_payload)
            manifest = archive_root / "archive_manifest.json"
            manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
            validate_gcp_archive(manifest, archive_root)
            shutil.rmtree(source_root)
            validate_gcp_archive(manifest, archive_root, require_sources=False)
            (archive_root / relatives[0]).write_bytes(b"tampered")
            with self.assertRaises(RuntimeError):
                validate_gcp_archive(manifest, archive_root, require_sources=False)

    def test_empty_lidar_and_failure_archives_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lidar_root = root / "lidar"
            lidar_root.mkdir()
            lidar = {
                "schema": "m3m_gcp_lidar_lightweight_archive_manifest_v1",
                "protocol_id": "m3m_gcp_lidar_rendered_surface_v1",
                "scene": candidate_builder.SCENE,
                "method_id": "citygs_x",
                "scene_attempt_freeze_sha256": "a" * 64,
                "inventory": [],
            }
            lidar["canonical_sha256"] = canonical_sha256(lidar)
            lidar_manifest = lidar_root / "archive_manifest.json"
            lidar_manifest.write_text(json.dumps(lidar), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                validate_exact_lidar_archive(
                    lidar_manifest,
                    lidar_root,
                    method_id="citygs_x",
                    expected_scene_attempt_freeze_sha256="a" * 64,
                    require_sources=False,
                )

            failure_root = root / "failure"
            failure_root.mkdir()
            failure = {
                "schema": "m3m_gcp_100k_failed_packet_lightweight_archive_v1",
                "status": "PASS_FAILURE_EVIDENCE_ARCHIVED",
                "files": [],
            }
            failure["canonical_sha256"] = canonical_sha256(failure)
            failure_manifest = failure_root / "archive_manifest.json"
            failure_manifest.write_text(json.dumps(failure), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                validate_failure_archive(
                    failure_manifest,
                    failure_root,
                    expected_scene=candidate_builder.SCENE,
                    expected_method_id="citygs_x",
                    expected_track="gcp",
                    expected_activation_sha256="a" * 64,
                    expected_failure_evidence_sha256="b" * 64,
                    expected_global_state_sha256="c" * 64,
                )

    def test_failure_archive_binds_semantics_primary_shas_and_every_log(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_root = root / "archive"
            log_path = archive_root / "logs" / "00_eval.stderr.log"
            log_path.parent.mkdir(parents=True)
            log_path.write_bytes(b"frozen stderr\n")
            activation_sha = "a" * 64
            global_state = {
                "schema": "m3m_gcp_100k_active_raw_packet_state_v1",
                "status": "ACTIVE_EXCLUSIVE_RAW_PACKET",
                "scene": candidate_builder.SCENE,
                "method_id": "citygs_x",
                "track": "gcp",
                "three_track_activation_sha256": activation_sha,
            }
            global_state["canonical_sha256"] = canonical_sha256(global_state)
            global_path = archive_root / "active_raw_packet_state.json"
            global_path.write_text(json.dumps(global_state), encoding="utf-8")
            failure = {
                "schema": "m3m_gcp_100k_gcp_evaluation_failure_v1",
                "status": "INCOMPLETE_UNRANKED",
                "scene": candidate_builder.SCENE,
                "method_id": "citygs_x",
                "three_track_activation_sha256": activation_sha,
                "global_raw_packet_state_sha256": sha256_file(global_path),
                "logs": [
                    {
                        "path": str((root / "source" / "eval.stderr.log").resolve()),
                        "bytes": log_path.stat().st_size,
                        "sha256": sha256_file(log_path),
                    }
                ],
                "retry_forbidden_after_child_start": True,
            }
            failure["canonical_sha256"] = canonical_sha256(failure)
            failure_path = archive_root / "failure_evidence.json"
            failure_path.write_text(json.dumps(failure), encoding="utf-8")
            failure_sha = sha256_file(failure_path)
            global_sha = sha256_file(global_path)
            declared_map = [
                {
                    "source_path": failure["logs"][0]["path"],
                    "relative_path": "logs/00_eval.stderr.log",
                    "bytes": log_path.stat().st_size,
                    "sha256": sha256_file(log_path),
                }
            ]
            manifest_payload = {
                "schema": "m3m_gcp_100k_failed_packet_lightweight_archive_v1",
                "status": "PASS_FAILURE_EVIDENCE_ARCHIVED",
                "scene": candidate_builder.SCENE,
                "method_id": "citygs_x",
                "track": "gcp",
                "three_track_activation_sha256": activation_sha,
                "failure_evidence_sha256": failure_sha,
                "global_raw_packet_state_sha256": global_sha,
                "declared_log_count": 1,
                "declared_log_archive_map": declared_map,
                "files": [
                    {
                        "relative_path": path.relative_to(archive_root).as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                    for path in sorted(archive_root.rglob("*"))
                    if path.is_file()
                ],
            }
            manifest_payload["canonical_sha256"] = canonical_sha256(manifest_payload)
            manifest = archive_root / "archive_manifest.json"

            def write_manifest(payload: dict) -> None:
                payload["canonical_sha256"] = canonical_sha256(payload)
                manifest.write_text(json.dumps(payload), encoding="utf-8")

            write_manifest(manifest_payload)
            validate_failure_archive(
                manifest,
                archive_root,
                expected_scene=candidate_builder.SCENE,
                expected_method_id="citygs_x",
                expected_track="gcp",
                expected_activation_sha256=activation_sha,
                expected_failure_evidence_sha256=failure_sha,
                expected_global_state_sha256=global_sha,
            )
            for key, wrong in (
                ("scene", "wrong_scene"),
                ("method_id", "metrogs"),
                ("track", "lidar"),
                ("three_track_activation_sha256", "0" * 64),
                ("failure_evidence_sha256", "1" * 64),
                ("global_raw_packet_state_sha256", "2" * 64),
            ):
                tampered = copy.deepcopy(manifest_payload)
                tampered[key] = wrong
                write_manifest(tampered)
                with self.assertRaises(RuntimeError):
                    validate_failure_archive(
                        manifest,
                        archive_root,
                        expected_scene=candidate_builder.SCENE,
                        expected_method_id="citygs_x",
                        expected_track="gcp",
                        expected_activation_sha256=activation_sha,
                        expected_failure_evidence_sha256=failure_sha,
                        expected_global_state_sha256=global_sha,
                    )
            write_manifest(manifest_payload)
            log_path.unlink()
            with self.assertRaises(RuntimeError):
                validate_failure_archive(
                    manifest,
                    archive_root,
                    expected_scene=candidate_builder.SCENE,
                    expected_method_id="citygs_x",
                    expected_track="gcp",
                    expected_activation_sha256=activation_sha,
                    expected_failure_evidence_sha256=failure_sha,
                    expected_global_state_sha256=global_sha,
                )

    def test_exact_lidar_archive_validates_sources_then_retained_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_root = root / "source"
            archive_root = root / "archive"
            method_id = "citygs_x"
            relatives = expected_lidar_archive_relatives(method_id)
            binding_rows = []
            for index, relative in enumerate(sorted(relatives - {"source_bindings.json"})):
                source = source_root / relative
                archived = archive_root / relative
                source.parent.mkdir(parents=True, exist_ok=True)
                archived.parent.mkdir(parents=True, exist_ok=True)
                content = f"lidar-{index}".encode("ascii")
                source.write_bytes(content)
                archived.write_bytes(content)
                binding_rows.append(
                    {
                        "relative_path": relative,
                        "source_path": str(source.resolve()),
                        "bytes": len(content),
                        "sha256": sha256_file(archived),
                    }
                )
            bindings = {
                "schema": "m3m_gcp_100k_lidar_lightweight_archive_source_bindings_v1",
                "scene": candidate_builder.SCENE,
                "method_id": method_id,
                "files": binding_rows,
            }
            bindings["canonical_sha256"] = canonical_sha256(bindings)
            bindings_path = archive_root / "source_bindings.json"
            bindings_path.write_text(json.dumps(bindings), encoding="utf-8")
            archive = {
                "schema": "m3m_gcp_lidar_lightweight_archive_manifest_v1",
                "protocol_id": "m3m_gcp_lidar_rendered_surface_v1",
                "scene": candidate_builder.SCENE,
                "method_id": method_id,
                "scene_attempt_freeze_sha256": "a" * 64,
                "inventory": [
                    {
                        "relative_path": path.relative_to(archive_root).as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                    for path in sorted(archive_root.rglob("*"))
                    if path.is_file()
                ],
            }
            archive["canonical_sha256"] = canonical_sha256(archive)
            manifest = archive_root / "archive_manifest.json"
            manifest.write_text(json.dumps(archive), encoding="utf-8")
            validate_exact_lidar_archive(
                manifest,
                archive_root,
                method_id=method_id,
                expected_scene_attempt_freeze_sha256="a" * 64,
                require_sources=True,
            )
            shutil.rmtree(source_root)
            validate_exact_lidar_archive(
                manifest,
                archive_root,
                method_id=method_id,
                expected_scene_attempt_freeze_sha256="a" * 64,
                require_sources=False,
            )
            (archive_root / sorted(relatives - {"source_bindings.json"})[0]).write_bytes(
                b"tampered"
            )
            with self.assertRaises(RuntimeError):
                validate_exact_lidar_archive(
                    manifest,
                    archive_root,
                    method_id=method_id,
                    expected_scene_attempt_freeze_sha256="a" * 64,
                    require_sources=False,
                )

    def test_wrong_packet_python_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frozen = root / "frozen-python"
            wrong = root / "wrong-python"
            frozen.write_bytes(b"frozen")
            wrong.write_bytes(b"wrong")
            recipe = {"phase_commands": {"packet": [str(frozen)]}}
            self.assertEqual(
                validate_frozen_packet_python(recipe, current_python=frozen),
                frozen.resolve(),
            )
            with self.assertRaises(RuntimeError):
                validate_frozen_packet_python(recipe, current_python=wrong)

    def test_gcp_evaluation_runtime_rejects_wrong_python_or_environment(self) -> None:
        python = Path(os.path.abspath(sys.executable))
        environment = {str(key): str(value) for key, value in os.environ.items()}
        observed = probe_gcp_evaluation_runtime(
            python,
            subprocess_environment=environment,
        )
        gcp_config = {
            "evaluation_runtime": {
                "python_path": str(python),
                "subprocess_environment": environment,
                "identity": observed,
            }
        }
        actual, actual_environment = validate_frozen_gcp_evaluation_runtime(
            gcp_config,
            requested_python=python,
        )
        self.assertEqual(actual, observed)
        self.assertEqual(actual_environment, environment)
        with tempfile.TemporaryDirectory() as temporary:
            wrong = Path(temporary) / "wrong-python"
            wrong.write_bytes(b"not the frozen interpreter")
            with self.assertRaises(RuntimeError):
                validate_frozen_gcp_evaluation_runtime(
                    gcp_config,
                    requested_python=wrong,
                )
        changed = copy.deepcopy(gcp_config)
        changed["evaluation_runtime"]["identity"]["numpy_core_sha256"] = "0" * 64
        with self.assertRaises(RuntimeError):
            validate_frozen_gcp_evaluation_runtime(
                changed,
                requested_python=python,
            )

    def test_global_raw_packet_mutex_rejects_cross_track_acquisition(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate_root = root / "runtime" / "candidate"
            candidate_root.mkdir(parents=True)
            candidate_manifest = candidate_root / "three_track_candidate_manifest_v1.json"
            candidate_manifest.write_text("{}\n", encoding="utf-8")
            recipe_path = root / "recipe.json"
            recipe = {
                "authorized_packet_set_root": str(root / "lidar-packet"),
                "authorized_packet_state": str(root / "lidar-state.json"),
            }
            recipe_path.write_text(json.dumps(recipe), encoding="utf-8")
            activation_path = root / "activation.json"
            activation_path.write_text("{}\n", encoding="utf-8")
            candidate = {
                "candidate_output_root": str(candidate_root),
                "scene_attempt_freeze": {"sha256": "a" * 64},
                "methods_manifest": {"sha256": "b" * 64},
            }
            registry = {
                "methods": [
                    {
                        "method_id": "citygs_x",
                        "recipe_path": str(recipe_path),
                        "recipe_sha256": sha256_file(recipe_path),
                    }
                ]
            }
            packet_root = root / "runtime" / "gcp-packet-scratch" / "citygs_x"
            track_state = (
                root
                / "runtime"
                / "gcp-packet-scratch"
                / "ACTIVE_GCP_PACKET_STATE.json"
            )
            state_path, state = acquire_active_raw_packet_state(
                activation_path=activation_path,
                candidate=candidate,
                registry=registry,
                method_id="citygs_x",
                track="gcp",
                recipe_sha256=sha256_file(recipe_path),
                attempt_model_identity_sha256="c" * 64,
                packet_set_root=packet_root,
                track_packet_state_path=track_state,
                owner_evidence_root=root / "evidence",
            )
            validate_active_raw_packet_state(
                state_path,
                activation_path=activation_path,
                candidate=candidate,
                method_id="citygs_x",
                track="gcp",
                recipe_sha256=sha256_file(recipe_path),
                attempt_model_identity_sha256="c" * 64,
                packet_set_root=packet_root,
                track_packet_state_path=track_state,
            )
            self.assertEqual(state["track"], "gcp")
            with self.assertRaises(FileExistsError):
                acquire_active_raw_packet_state(
                    activation_path=activation_path,
                    candidate=candidate,
                    registry=registry,
                    method_id="citygs_x",
                    track="lidar",
                    recipe_sha256=sha256_file(recipe_path),
                    attempt_model_identity_sha256="c" * 64,
                    packet_set_root=root / "lidar-packet",
                    track_packet_state_path=root / "lidar-state.json",
                    owner_evidence_root=root / "lidar-evidence",
                )
            state_path.chmod(0o666)
            state_path.unlink()

            def race(track: str) -> str:
                try:
                    acquire_active_raw_packet_state(
                        activation_path=activation_path,
                        candidate=candidate,
                        registry=registry,
                        method_id="citygs_x",
                        track=track,
                        recipe_sha256=sha256_file(recipe_path),
                        attempt_model_identity_sha256="c" * 64,
                        packet_set_root=(
                            packet_root if track == "gcp" else root / "lidar-packet"
                        ),
                        track_packet_state_path=(
                            track_state if track == "gcp" else root / "lidar-state.json"
                        ),
                        owner_evidence_root=root / f"{track}-race-evidence",
                    )
                    return "ACQUIRED"
                except FileExistsError:
                    return "BLOCKED"

            with ThreadPoolExecutor(max_workers=2) as executor:
                outcomes = list(executor.map(race, ("gcp", "lidar")))
            self.assertEqual(sorted(outcomes), ["ACQUIRED", "BLOCKED"])

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
