#!/usr/bin/env python3
"""Dual-review 100K activation-builder tests."""

from __future__ import annotations

import json
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
        plan = {
            "schema": "m3m_gcp_native_quarter_100k_ten_method_execution_plan_v1",
            "status": "REVIEW_CANDIDATE_NOT_EXECUTION_AUTHORIZED",
            "execution_authorized": False,
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
            self.repo / "configs" / "m3m_gcp_native_quarter_100k_ten_method_execution_plan_v1.json",
            plan,
        )
        recipes = {
            "schema": "m3m_gcp_native_quarter_100k_recipe_manifest_v1",
            "status": "REVIEW_CANDIDATE_NOT_EXECUTION_AUTHORIZED",
            "recipes": [{"method_id": str(index)} for index in range(10)],
        }
        recipes["canonical_sha256"] = canonical_sha256(recipes)
        write_json(
            self.repo / "configs" / "m3m_gcp_native_quarter_100k_recipe_manifest_v1.json",
            recipes,
        )
        self._commit("phase2")
        self.phase2_commit = self._git("rev-parse", "HEAD")
        self.output = self.root / "activation.json"
        self.script = Path(__file__).resolve().parent / "build_m3m_gcp_lidar_100k_activation.py"

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
