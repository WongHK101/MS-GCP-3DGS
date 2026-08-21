#!/usr/bin/env python3
"""Dual-review 100K activation-builder tests."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from m3m_gcp_lidar_artifacts import canonical_sha256, sha256_file


TASK = "019ff12c-cb29-7cb2-8fb6-1d82c5f8c54b"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


class ActivationBuilderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.repo = self.root / "repo"
        self.repo.mkdir()
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        (self.repo / ".gitattributes").write_text("*.json text eol=lf\n", encoding="utf-8")
        source_root = Path(__file__).resolve().parent
        script_root = self.repo / "code" / "gcp"
        script_root.mkdir(parents=True)
        for name in (
            "build_m3m_gcp_lidar_100k_activation.py",
            "m3m_gcp_lidar_artifacts.py",
            "m3m_gcp_100k_continuity.py",
        ):
            shutil.copy2(source_root / name, script_root / name)
        (script_root / "m3m_gcp_100k_source_binding_correction.py").write_text(
            "def validate_source_binding_correction(**kwargs):\n    return {}\n",
            encoding="utf-8",
        )
        self.script = script_root / "build_m3m_gcp_lidar_100k_activation.py"
        contract = self.repo / "configs" / "m3m_gcp_lidar_formal_v1.json"
        schema = self.repo / "configs" / "m3m_gcp_lidar_formal_artifact_schema_v1.json"
        local = self.repo / "docs" / "protocol_evidence" / "m3m_gcp_six_scene_common_preparation_local_v2.json"
        remote = self.repo / "docs" / "protocol_evidence" / "m3m_gcp_six_scene_common_preparation_remote_v2.json"
        write_json(contract, {
            "protocol_id": "m3m_gcp_lidar_rendered_surface_v1",
            "execution_authorized": False,
            "review": {"protocol_review_task_id": TASK},
        })
        write_json(schema, {"schema": "m3m_gcp_lidar_formal_artifact_schema_v1"})
        preparation = {
            "status": "PASS_COMMON_SCENE_PREPARATION_NO_TRAINING",
            "scene_count": 6, "contract_file_sha256": sha256_file(contract),
            "training_started": False, "formal_evaluation": "NOT_STARTED",
        }
        write_json(local, preparation)
        write_json(remote, preparation)
        self._commit("phase1")
        self.phase1_commit = self._git("rev-parse", "HEAD")
        self.phase1_tree = self._git("show", "-s", "--format=%T", "HEAD")
        self.output = self.root / "activation.json"

        previous_plan_path = self.repo / "previous-plan.json"
        previous_plan = {
            "schema": "m3m_gcp_native_quarter_100k_ten_method_execution_plan_v2"
        }
        previous_plan["canonical_sha256"] = canonical_sha256(previous_plan)
        write_json(previous_plan_path, previous_plan)
        previous_recipes_path = self.repo / "previous-recipes.json"
        previous_recipes = {
            "schema": "m3m_gcp_native_quarter_100k_recipe_manifest_v2"
        }
        previous_recipes["canonical_sha256"] = canonical_sha256(previous_recipes)
        write_json(previous_recipes_path, previous_recipes)
        previous_note_path = self.repo / "previous-note.md"
        previous_note_path.write_text("previous\n", encoding="utf-8")

        continuity_root = self.root / "continuity"
        continuity_root.mkdir()
        environment = continuity_root / "environment.json"
        stdout = continuity_root / "stdout.log"
        stderr = continuity_root / "stderr.log"
        write_json(environment, {"sealed": True})
        stdout.write_text("stdout\n", encoding="utf-8")
        stderr.write_text("stderr\n", encoding="utf-8")
        failure_path = continuity_root / "failure.json"
        failure = {
            "schema": "m3m_gcp_lidar_failure_evidence_v1",
            "scene": "gcp_100000_20260610",
            "method_id": "2dgs",
            "status": "FAILED_UNRANKED",
            "oom_signal": None,
            "environment_manifest_sha256": sha256_file(environment),
            "stdout_sha256": sha256_file(stdout),
            "stderr_sha256": sha256_file(stderr),
        }
        failure["canonical_sha256"] = canonical_sha256(failure)
        write_json(failure_path, failure)
        supplement_path = continuity_root / "supplement.json"
        write_json(supplement_path, {
            "formal_status_unchanged": "FAILED_UNRANKED",
            "formal_attempt_consumed": True,
            "retry_allowed": False,
            "bound_evidence": {"failure_file_sha256": sha256_file(failure_path)},
        })
        console = continuity_root / "pgsr-console.log"
        console.write_text(
            "RuntimeError: method source runtime status mismatch\n", encoding="utf-8"
        )
        activation_v2_path = continuity_root / "activation-v2.json"
        activation_v2 = {
            "schema": "m3m_gcp_lidar_formal_activation_v1",
            "benchmark_commit": "a64752b5f7375d79b0e9d82ca1f0e782ac6f0f86",
            "benchmark_tree": "9cbc07527c87614bf74cc3239360fe4a53519ef8",
            "execution_plan_reviewed_commit": "a64752b5f7375d79b0e9d82ca1f0e782ac6f0f86",
            "execution_plan_reviewed_tree": "9cbc07527c87614bf74cc3239360fe4a53519ef8",
            "execution_plan_path": "previous-plan.json",
            "execution_plan_sha256": sha256_file(previous_plan_path),
            "recipe_manifest_path": "previous-recipes.json",
            "recipe_manifest_sha256": sha256_file(previous_recipes_path),
        }
        activation_v2["canonical_sha256"] = canonical_sha256(activation_v2)
        write_json(activation_v2_path, activation_v2)

        def row(role: str, path: Path) -> dict:
            return {
                "role": role, "path": str(path.resolve()),
                "bytes": path.stat().st_size, "sha256": sha256_file(path),
            }

        continuity_path = self.repo / "continuity.json"
        continuity = {
            "schema": "m3m_gcp_100k_activation_continuity_v1",
            "status": "SEALED_V2_TO_V3_CONTINUITY",
            "scene": "gcp_100000_20260610",
            "previous_reviewed_checkout": {
                "commit": "a64752b5f7375d79b0e9d82ca1f0e782ac6f0f86",
                "tree": "9cbc07527c87614bf74cc3239360fe4a53519ef8",
                "review_task_id": TASK,
                "review_verdict": "PASS_100K_TIME_SPACE_EXECUTION_PLAN_V1",
            },
            "repository_artifacts": [
                {**row("execution_plan_v2", previous_plan_path),
                 "path": "previous-plan.json",
                 "canonical_sha256": previous_plan["canonical_sha256"]},
                {**row("recipe_manifest_v2", previous_recipes_path),
                 "path": "previous-recipes.json",
                 "canonical_sha256": previous_recipes["canonical_sha256"]},
                {**row("execution_note_v2", previous_note_path),
                 "path": "previous-note.md", "canonical_sha256": None},
            ],
            "remote_artifacts": [
                row("activation_v2", activation_v2_path),
                row("2dgs_failure", failure_path),
                row("2dgs_classification_supplement", supplement_path),
                row("2dgs_environment", environment),
                row("2dgs_stdout", stdout),
                row("2dgs_stderr", stderr),
                row("pgsr_prechild_guard_console", console),
            ],
            "inherited_method_outcomes": [{
                "method_id": "2dgs", "formal_status": "FAILED_UNRANKED",
                "formal_oom_signal": None, "formal_attempt_consumed": True,
                "retry_allowed": False, "failure_sha256": sha256_file(failure_path),
                "classification_supplement_sha256": sha256_file(supplement_path),
            }],
            "pre_child_guard_rejections": [{
                "method_id": "pgsr", "child_started": False,
                "run_root_created": False, "evidence_root_created": False,
                "formal_attempt_consumed": False,
                "retry_allowed_only_after_exact_guard_fix_and_new_review": True,
                "run_root": str((self.root / "pgsr-run").resolve()),
                "evidence_root": str((self.root / "pgsr-evidence").resolve()),
                "console_sha256": sha256_file(console),
            }],
            "transition_policy": {
                "activation_v2_immutable": True,
                "activation_v3_path": str(self.output.resolve()),
                "continued_run_namespace": "/root/autodl-tmp/runs/m3m-gcp-native-quarter/formal-100k-v2",
                "recipe_manifest_v2_bytes_unchanged": True,
                "execution_plan_v2_bytes_unchanged": True,
                "inherited_final_methods_forbidden_to_launch": ["2dgs"],
                "final_attempt_freeze_authorization": "activation_v3_only",
                "remote_artifacts_must_remain_byte_identical": True,
                "manual_guard_bypass_forbidden": True,
            },
        }
        continuity["canonical_sha256"] = canonical_sha256(continuity)
        write_json(continuity_path, continuity)
        plan = {
            "schema": "m3m_gcp_native_quarter_100k_ten_method_execution_plan_v3",
            "status": "REVIEW_CANDIDATE_NOT_EXECUTION_AUTHORIZED",
            "execution_authorized": False,
            "activation_manifest_path": str(self.output.resolve()),
            "activation_continuity": {
                "receipt": {"path": "continuity.json", "sha256": sha256_file(continuity_path)},
                "status_required": "SEALED_V2_TO_V3_CONTINUITY",
                "remote_artifacts_must_remain_byte_identical": True,
                "recipe_manifest_v2_bytes_unchanged": True,
                "execution_plan_v2_bytes_unchanged": True,
                "inherited_final_methods_forbidden_to_launch": ["2dgs"],
            },
            "formal_lidar_protocol": {"phase1_review": {
                "task_id": TASK,
                "verdict": "PASS_LIDAR_V1_AND_SIX_SCENE_PREPARATION_V2",
                "reviewed_commit": self.phase1_commit,
                "reviewed_tree": self.phase1_tree,
                "protocol_pass_alone_authorizes_execution": False,
            }},
            "review": {
                "task_id": TASK,
                "required_pass_verdict": "PASS_100K_TIME_SPACE_EXECUTION_PLAN_V1",
            },
        }
        plan["canonical_sha256"] = canonical_sha256(plan)
        write_json(
            self.repo / "configs" / "m3m_gcp_native_quarter_100k_ten_method_execution_plan_v3.json",
            plan,
        )
        recipes = {
            "schema": "m3m_gcp_native_quarter_100k_recipe_manifest_v3",
            "status": "REVIEW_CANDIDATE_NOT_EXECUTION_AUTHORIZED",
            "recipes": [{"method_id": str(index)} for index in range(10)],
        }
        recipes["canonical_sha256"] = canonical_sha256(recipes)
        write_json(
            self.repo / "configs" / "m3m_gcp_native_quarter_100k_recipe_manifest_v3.json",
            recipes,
        )
        self._commit("phase2")
        self.phase2_commit = self._git("rev-parse", "HEAD")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _git(self, *args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(self.repo), *args], text=True
        ).strip()

    def _commit(self, message: str) -> None:
        subprocess.run(["git", "-C", str(self.repo), "add", "."], check=True)
        subprocess.run([
            "git", "-C", str(self.repo), "-c", "user.name=Test",
            "-c", "user.email=test@example.invalid", "commit", "-qm", message,
        ], check=True)

    def _command(self, verdict: str) -> list[str]:
        return [
            sys.executable, "-B", str(self.script), "--repo", str(self.repo),
            "--protocol-reviewed-commit", self.phase1_commit,
            "--execution-plan-reviewed-commit", self.phase2_commit,
            "--execution-plan-review-verdict", verdict,
            "--output", str(self.output),
        ]

    def test_exact_two_reviews_build_activation(self) -> None:
        subprocess.run(self._command("PASS_100K_TIME_SPACE_EXECUTION_PLAN_V1"), check=True, capture_output=True)
        payload = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(payload["protocol_reviewed_commit"], self.phase1_commit)
        self.assertEqual(payload["execution_plan_reviewed_commit"], self.phase2_commit)
        self.assertTrue(payload["execution_authorized"])
        self.assertEqual(payload["canonical_sha256"], canonical_sha256(payload))

    def test_missing_exact_phase2_verdict_is_rejected(self) -> None:
        result = subprocess.run(self._command("PENDING"), capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("exact 100K execution-plan review verdict is absent", result.stderr)


if __name__ == "__main__":
    unittest.main()
