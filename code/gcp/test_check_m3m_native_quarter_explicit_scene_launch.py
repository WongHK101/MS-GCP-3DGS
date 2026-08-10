import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from check_m3m_native_quarter_explicit_scene_launch import (  # noqa: E402
    PROTOCOL_ID,
    canonical_sha256,
    check_launch,
    file_sha256,
)


class ExplicitSceneLaunchGateTest(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[dict, Path, Path, str]:
        repo = root / "repo"
        release = root / "release"
        config = repo / "configs" / "authorization.json"
        manifest_path = release / "formal_inputs" / "scene" / "NATIVE_QUARTER_INPUT_MANIFEST.json"
        config.parent.mkdir(parents=True)
        manifest_path.parent.mkdir(parents=True)
        manifest = {
            "schema": "gs_gcp_colmap_native_quarter_materialized_input_manifest_v1",
            "scene": "scene",
            "pixel_domain": "native-quarter",
            "full_view_count": 8,
            "train_view_count": 7,
            "test_view_count": 1,
        }
        manifest["manifest_sha256"] = canonical_sha256(manifest)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        run_root = "/root/autodl-tmp/runs/m3m-gcp-native-quarter/method-ns/scene/run-1"
        locks = {
            "protocol_changed": False,
            "three_k_rerun_authorized": False,
            "other_scene_authorized": False,
            "other_method_authorized": False,
            "other_seed_authorized": False,
            "six_scene_matrix_authorized": False,
            "candidate_pool_reopened": False,
            "tgs_gcp_in_scope": False,
        }
        authorization = {
            "authorization_id": "auth-1",
            "protocol_id": PROTOCOL_ID,
            "status": "AUTHORIZED_NOT_STARTED",
            "method": {
                "method_id": "method",
                "run_namespace": "method-ns",
                "repository_commit": "commit",
                "repository_tree": "tree",
            },
            "input": {
                "scene": "scene",
                "formal_input_manifest": "formal_inputs/scene/NATIVE_QUARTER_INPUT_MANIFEST.json",
                "formal_input_manifest_file_sha256": file_sha256(manifest_path),
                "formal_input_manifest_canonical_sha256": manifest["manifest_sha256"],
                "pixel_domain": "native-quarter",
                "full_view_count": 8,
                "train_view_count": 7,
                "test_view_count": 1,
            },
            "training": {"seed": 0, "iterations": 30000},
            "execution": {
                "run_root": run_root,
                "single_fresh_run_allowed": True,
                "resume_allowed": False,
                "rerun_after_attempt_allowed": False,
                "overwrite_allowed": False,
            },
            "scope_locks": locks,
        }
        config.write_text(json.dumps(authorization), encoding="utf-8")
        registry = {
            "protocol_id": PROTOCOL_ID,
            "global_training_allowed": False,
            "per_method_training_allowed_methods": [],
            "preliminary_evidence_scope": {
                "six_scene_matrix_status": "LOCKED",
                "multi_seed_status": "NOT_AUTHORIZED",
            },
            "methods": [{
                "method_id": "method",
                "source": {"commit": "commit", "tree": "tree"},
                "three_k_training_allowed": False,
                "full_scene_matrix_eligible": False,
            }],
            "explicit_scene_run_authorization": {
                "authorization_id": "auth-1",
                "status": "AUTHORIZED_NOT_STARTED",
                "authorization": "configs/authorization.json",
                "authorization_sha256": file_sha256(config),
                "method_id": "method",
                "scene": "scene",
                "seed": 0,
                "iterations": 30000,
                "run_root": run_root,
                "single_fresh_run_allowed": True,
                "resume_allowed": False,
                "rerun_after_attempt_allowed": False,
                "other_methods_scenes_seeds_allowed": False,
            },
        }
        return registry, repo, release, run_root

    def test_exact_fresh_run_is_authorized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry, repo, release, run_root = self._fixture(Path(directory))
            result = check_launch(
                registry,
                repo,
                release,
                method_id="method",
                scene="scene",
                seed=0,
                iterations=30000,
                run_root=run_root,
                run_root_exists=False,
            )
            self.assertTrue(result["passed"], result["errors"])

    def test_relocked_or_existing_run_is_denied(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry, repo, release, run_root = self._fixture(Path(directory))
            relocked = copy.deepcopy(registry)
            relocked["explicit_scene_run_authorization"]["status"] = "COMPLETED_RELOCKED"
            result = check_launch(
                relocked,
                repo,
                release,
                method_id="method",
                scene="scene",
                seed=0,
                iterations=30000,
                run_root=run_root,
                run_root_exists=True,
            )
            self.assertFalse(result["passed"])
            self.assertIn("explicit scene authorization is not active", result["errors"])
            self.assertTrue(any("run root already exists" in error for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
