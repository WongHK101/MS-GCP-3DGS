#!/usr/bin/env python3
"""Executable positive/negative regressions for the 100K post-attempt closure."""

from __future__ import annotations

import copy
import json
import subprocess
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import m3m_gcp_100k_postattempt_closure as closure
from m3m_gcp_100k_activation_v4_continuity import (
    POSTATTEMPT_TERMINAL,
    PRELAUNCH_FRESH,
    validate_activation_v4_continuity,
    validate_prelaunch_fresh_state,
)
from m3m_gcp_lidar_artifacts import (
    PROTOCOL_ID,
    canonical_sha256,
    command_sha256,
    sha256_file,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def canonical_file(path: Path, payload: dict) -> dict:
    payload = copy.deepcopy(payload)
    payload["canonical_sha256"] = canonical_sha256(payload)
    write_json(path, payload)
    return payload


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, encoding="utf-8"
    ).strip()


def init_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    git(repo, "config", "user.name", "Postattempt Test")
    git(repo, "config", "user.email", "postattempt@example.invalid")
    git(repo, "config", "core.autocrlf", "false")


class PostattemptFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.repo = root / "closure-repo"
        self.formal_repo = root / "formal-repo"
        self.remote = root / "remote"
        self.formal_root = self.remote / "formal-100k-v2"
        self.evidence_root = (
            self.formal_root / closure.SCENE / closure.METHOD_ID / "evidence"
        )
        self.run_root = self.formal_root / closure.SCENE / closure.METHOD_ID / "seed0-v2"
        self.dataset = self.remote / "dataset"
        self.source_manifest = self.remote / "formal-input" / "NATIVE_QUARTER_INPUT_MANIFEST.json"
        self.prior_products: list[Path] = []
        init_repo(self.formal_repo)
        (self.formal_repo / "train.py").write_text("print('never run')\n", encoding="utf-8")
        git(self.formal_repo, "add", "train.py")
        git(self.formal_repo, "commit", "-qm", "formal activation checkout")
        self.formal_commit = git(self.formal_repo, "rev-parse", "HEAD")
        self.formal_tree = git(self.formal_repo, "rev-parse", "HEAD^{tree}")
        init_repo(self.repo)
        self._build_artifacts()
        git(self.repo, "add", ".")
        git(self.repo, "commit", "-qm", "closure inputs")

    def _build_artifacts(self) -> None:
        self.dataset.mkdir(parents=True)
        self.run_root.mkdir(parents=True)
        canonical_file(
            self.source_manifest,
            {"schema": "synthetic_formal_input", "scene": closure.SCENE},
        )
        plan_path = self.repo / closure.PLAN_RELATIVE
        recipe_path = self.repo / closure.METRO_RECIPE_RELATIVE
        manifest_path = self.repo / closure.RECIPE_MANIFEST_RELATIVE
        old_receipt_path = self.repo / closure.OLD_CONTINUITY_RELATIVE
        recipe = {
            "schema": "m3m_gcp_native_quarter_100k_method_recipe_v2",
            "method_id": closure.METHOD_ID,
            "scene": closure.SCENE,
            "authorized_run_root": str(self.run_root),
            "authorized_packet_set_root": str(self.remote / "packet"),
            "authorized_evidence_root": str(self.evidence_root),
            "renderer_adapter_sha256": "a" * 64,
            "formal_input_manifest": {
                "path": str(self.source_manifest),
                "file_sha256": sha256_file(self.source_manifest),
            },
            "source_bindings": {
                "training": {"root": str(self.remote / "source")},
            },
            "phase_roots": {
                "training": {
                    "dataset_root": str(self.dataset),
                    "prior_root": str(self.dataset),
                }
            },
            "phase_commands": {
                "training": [
                    "python",
                    "{repo}/train.py",
                    "--dataset",
                    "{dataset_root}",
                    "--source",
                    "{source_root}",
                    "--prior",
                    "{prior_root}",
                    "--run",
                    "{run_root}",
                ]
            },
            "budget": {
                "type": "effective_image_iterations",
                "value": 150000,
                "optimizer_steps": 37500,
            },
        }
        canonical_file(recipe_path, recipe)
        canonical_file(
            manifest_path,
            {
                "schema": "m3m_gcp_native_quarter_100k_recipe_manifest_v3",
                "method_order": [closure.METHOD_ID],
                "recipes": [
                    {
                        "method_id": closure.METHOD_ID,
                        "path": closure.METRO_RECIPE_RELATIVE.as_posix(),
                        "sha256": sha256_file(recipe_path),
                    }
                ],
            },
        )
        canonical_file(
            plan_path,
            {
                "schema": "m3m_gcp_native_quarter_100k_ten_method_execution_plan_v4",
                "scene": closure.SCENE,
                "execution_authorized": False,
            },
        )
        canonical_file(
            old_receipt_path,
            {
                "schema": "m3m_gcp_100k_activation_continuity_v2",
                "status": "SEALED_V3_TO_V4_GUARD_CONTINUITY",
                "scene": closure.SCENE,
            },
        )
        activation_path = self.formal_root / "activation_v4.json"
        canonical_file(
            activation_path,
            {
                "schema": "m3m_gcp_lidar_formal_activation_v1",
                "execution_authorized": True,
                "benchmark_commit": self.formal_commit,
                "benchmark_tree": self.formal_tree,
                "execution_plan_reviewed_commit": self.formal_commit,
                "execution_plan_reviewed_tree": self.formal_tree,
                "execution_plan_path": closure.PLAN_RELATIVE.as_posix(),
                "execution_plan_sha256": sha256_file(plan_path),
                "recipe_manifest_path": closure.RECIPE_MANIFEST_RELATIVE.as_posix(),
                "recipe_manifest_sha256": sha256_file(manifest_path),
            },
        )
        prior_environment = canonical_file(
            self.evidence_root / "prior" / "environment.json",
            {"schema": "synthetic_prior_environment", "scene": closure.SCENE},
        )
        for index, suffix in enumerate((".npy", ".npy", ".json")):
            product = self.dataset / f"prior-{index}{suffix}"
            product.write_bytes(f"product-{index}".encode("utf-8"))
            self.prior_products.append(product)
        prior_success = {
            "schema": "m3m_gcp_100k_phase_success_v2",
            "status": "PASS",
            "scene": closure.SCENE,
            "method_id": closure.METHOD_ID,
            "phase": "prior",
            "recipe_sha256": sha256_file(recipe_path),
            "environment_manifest_path": str(self.evidence_root / "prior" / "environment.json"),
            "environment_manifest_sha256": sha256_file(
                self.evidence_root / "prior" / "environment.json"
            ),
            "products": [
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "validation": {"kind": "hash_bound_file_v1"},
                }
                for path in self.prior_products
            ],
        }
        canonical_file(self.evidence_root / "prior" / "phase_success.json", prior_success)

        recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
        command = closure._expected_training_command(recipe, repo_path=self.formal_repo)
        training = self.evidence_root / "training"
        training.mkdir(parents=True)
        stdout = training / "command.stdout.log"
        stderr = training / "command.stderr.log"
        stdout.write_bytes(b"")
        missing = self.dataset / "NATIVE_QUARTER_INPUT_MANIFEST.json"
        stderr.write_text(f"Traceback\nFileNotFoundError: {missing}\n", encoding="utf-8")
        environment = canonical_file(
            training / "environment.json",
            {
                "schema": "m3m_gcp_100k_execution_environment_v2",
                "scene": closure.SCENE,
                "method_id": closure.METHOD_ID,
                "phase": "training",
                "argv": command,
                "started_at_utc": "2026-08-22T11:12:14Z",
                "resource_limits": {"child_actual": {"soft": 65536, "hard": 1048576}},
            },
        )
        failure = {
            "schema": "m3m_gcp_lidar_failure_evidence_v1",
            "protocol_id": PROTOCOL_ID,
            "scene": closure.SCENE,
            "method_id": closure.METHOD_ID,
            "input_class": "rgb_colmap_external_geometry_prior",
            "seed": 0,
            "status": "FAILED_UNRANKED",
            "failure_stage": "training",
            "run_root": str(self.run_root),
            "model_checkpoint_sha256": None,
            "scene_attempt_freeze_sha256": None,
            "command_argv": command,
            "command_sha256": command_sha256(command),
            "environment_manifest_path": str(training / "environment.json"),
            "environment_manifest_sha256": sha256_file(training / "environment.json"),
            "recipe_sha256": sha256_file(recipe_path),
            "renderer_adapter_sha256": "a" * 64,
            "started_at_utc": "2026-08-22T11:12:14Z",
            "ended_at_utc": "2026-08-22T11:12:24Z",
            "exit_code": 1,
            "last_valid_progress": {"unit": "optimizer_steps", "value": 0.0},
            "peak_gpu_memory_mib": 0.0,
            "process_maximum_rss_kib": 54000,
            "cgroup_memory_events_delta": {"oom": 0, "oom_kill": 0, "max": 0},
            "oom_signal": None,
            "stdout_path": str(stdout),
            "stdout_sha256": sha256_file(stdout),
            "stderr_path": str(stderr),
            "stderr_sha256": sha256_file(stderr),
            "errors": ["command exited with code 1"],
        }
        canonical_file(training / "failure.json", failure)
        (training / "guard-console-v4.log").write_text(
            json.dumps(
                {
                    "status": "FAILED_UNRANKED",
                    "failure_evidence": str(training / "failure.json"),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        self.plan = json.loads(plan_path.read_text(encoding="utf-8"))

    def patches(self) -> ExitStack:
        stack = ExitStack()
        activation = self.formal_root / "activation_v4.json"
        activation_v5 = self.formal_root / "activation_v5.json"
        evidence = self.evidence_root
        repository_paths = {
            "execution_plan_v4": closure.PLAN_RELATIVE,
            "recipe_manifest_v3": closure.RECIPE_MANIFEST_RELATIVE,
            "metrogs_recipe_v2": closure.METRO_RECIPE_RELATIVE,
            "activation_v3_to_v4_continuity": closure.OLD_CONTINUITY_RELATIVE,
        }
        remote_paths = {
            "activation_v4": activation,
            "metrogs_training_failure": evidence / "training" / "failure.json",
            "metrogs_training_environment": evidence / "training" / "environment.json",
            "metrogs_training_stdout": evidence / "training" / "command.stdout.log",
            "metrogs_training_stderr": evidence / "training" / "command.stderr.log",
            "metrogs_training_guard_console_v4": evidence / "training" / "guard-console-v4.log",
            "metrogs_prior_phase_success": evidence / "prior" / "phase_success.json",
            "metrogs_prior_environment": evidence / "prior" / "environment.json",
        }
        values = {
            "FORMAL_EXECUTION_REPO": self.formal_repo,
            "FORMAL_RUN_ROOT": self.formal_root,
            "ACTIVATION_V4_PATH": activation,
            "ACTIVATION_V5_PATH": activation_v5,
            "METRO_EVIDENCE_ROOT": evidence,
            "METRO_FAILURE_PATH": evidence / "training" / "failure.json",
            "METRO_GUARD_CONSOLE_PATH": evidence / "training" / "guard-console-v4.log",
            "METRO_PRIOR_SUCCESS_PATH": evidence / "prior" / "phase_success.json",
            "ACTIVATION_V4_COMMIT": self.formal_commit,
            "ACTIVATION_V4_TREE": self.formal_tree,
            "EXPECTED_PRIOR_PRODUCT_COUNT": 3,
            "EXPECTED_PRIOR_DEPTH_COUNT": 2,
            "REPOSITORY_ROLE_PATHS": repository_paths,
            "REMOTE_ROLE_PATHS": remote_paths,
        }
        for name, value in values.items():
            stack.enter_context(mock.patch.object(closure, name, value))
        return stack

    def build_and_track_receipt(self) -> tuple[Path, dict]:
        with self.patches():
            payload = closure.build_postattempt_closure_payload(
                repo=self.repo, plan=self.plan
            )
        path = self.repo / closure.POSTATTEMPT_RECEIPT_RELATIVE
        write_json(path, payload)
        git(self.repo, "add", path.relative_to(self.repo).as_posix())
        git(self.repo, "commit", "-qm", "seal post-attempt closure")
        return path, payload


class PostattemptClosureTest(unittest.TestCase):
    def test_prelaunch_fresh_is_explicit_and_rejects_existing_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            rejection = {
                "run_root": str(root / "run"),
                "failure_path": str(root / "evidence" / "failure.json"),
            }
            validate_prelaunch_fresh_state(rejection)
            Path(rejection["run_root"]).mkdir()
            with self.assertRaisesRegex(RuntimeError, "fresh attempt"):
                validate_prelaunch_fresh_state(rejection)
        with self.assertRaises(TypeError):
            validate_activation_v4_continuity(repo=Path.cwd(), plan={})
        self.assertNotEqual(PRELAUNCH_FRESH, POSTATTEMPT_TERMINAL)

    def test_exact_postattempt_receipt_passes_and_is_tracked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = PostattemptFixture(Path(directory).resolve())
            receipt_path, _ = fixture.build_and_track_receipt()
            with fixture.patches():
                result = closure.validate_postattempt_closure(
                    repo=fixture.repo,
                    plan=fixture.plan,
                    receipt_path=receipt_path,
                )
            self.assertEqual(result["status"], closure.RECEIPT_STATUS)

    def test_terminal_path_status_and_artifact_tamper_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = PostattemptFixture(Path(directory).resolve())
            _, payload = fixture.build_and_track_receipt()
            wrong_path = copy.deepcopy(payload)
            wrong_path["metrogs_terminal"]["failure_path"] += ".alternate"
            wrong_path["canonical_sha256"] = canonical_sha256(wrong_path)
            with fixture.patches(), self.assertRaisesRegex(RuntimeError, "terminal closure"):
                closure._validate_payload(
                    repo=fixture.repo, plan=fixture.plan, payload=wrong_path
                )

            failure_path = fixture.evidence_root / "training" / "failure.json"
            original = failure_path.read_bytes()
            failure = json.loads(original)
            failure["status"] = "OOM_UNRANKED"
            failure["canonical_sha256"] = canonical_sha256(failure)
            write_json(failure_path, failure)
            with fixture.patches(), self.assertRaisesRegex(RuntimeError, "changed"):
                closure._validate_payload(
                    repo=fixture.repo, plan=fixture.plan, payload=payload
                )
            failure_path.write_bytes(original)

            fixture.prior_products[0].write_bytes(b"tampered")
            with fixture.patches(), self.assertRaisesRegex(RuntimeError, "prior product changed"):
                closure._validate_payload(
                    repo=fixture.repo, plan=fixture.plan, payload=payload
                )

    def test_postattempt_phase_success_or_final_model_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = PostattemptFixture(Path(directory).resolve())
            _, payload = fixture.build_and_track_receipt()
            phase_success = fixture.evidence_root / "training" / "phase_success.json"
            phase_success.write_text("{}", encoding="utf-8")
            with fixture.patches(), self.assertRaisesRegex(RuntimeError, "success/model"):
                closure._validate_payload(
                    repo=fixture.repo, plan=fixture.plan, payload=payload
                )
            phase_success.unlink()
            (fixture.run_root / "model").mkdir()
            with fixture.patches(), self.assertRaisesRegex(RuntimeError, "empty attempt root"):
                closure._validate_payload(
                    repo=fixture.repo, plan=fixture.plan, payload=payload
                )

    def test_lifecycle_call_sites_cannot_confuse_closure_with_execution(self) -> None:
        root = Path(__file__).resolve().parents[2]
        guard = (root / "code/gcp/run_m3m_gcp_100k_guarded.py").read_text(encoding="utf-8")
        activation = (
            root / "code/gcp/build_m3m_gcp_lidar_100k_activation_v4.py"
        ).read_text(encoding="utf-8")
        attempts = (
            root / "code/gcp/build_m3m_gcp_100k_attempt_manifest.py"
        ).read_text(encoding="utf-8")
        freezer = (
            root / "code/gcp/freeze_m3m_gcp_lidar_scene_attempts.py"
        ).read_text(encoding="utf-8")
        self.assertIn("mode=PRELAUNCH_FRESH", guard)
        self.assertIn("mode=PRELAUNCH_FRESH", activation)
        self.assertNotIn("POSTATTEMPT_TERMINAL", guard)
        self.assertNotIn("POSTATTEMPT_TERMINAL", activation)
        self.assertIn("mode=POSTATTEMPT_TERMINAL", attempts)
        self.assertIn("mode=POSTATTEMPT_TERMINAL", freezer)
        self.assertIn("--postattempt-closure", attempts)
        self.assertIn("--postattempt-closure", freezer)


if __name__ == "__main__":
    unittest.main()
