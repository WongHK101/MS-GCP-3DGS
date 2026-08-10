import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path, PurePosixPath
from unittest import mock


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import check_m3m_native_quarter_preiteration_compatibility_retry as gate  # noqa: E402


class PreiterationCompatibilityRetryGateTest(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[dict, Path, Path, Path]:
        repo = root / "repo"
        release = root / "release"
        run = root / "failed_run"
        (repo / "configs").mkdir(parents=True)
        manifest_path = release / "formal_inputs" / "gcp_100000_20260610" / "NATIVE_QUARTER_INPUT_MANIFEST.json"
        manifest_path.parent.mkdir(parents=True)
        manifest = {
            "scene": "gcp_100000_20260610",
            "train_view_count": 2196,
            "test_view_count": 314,
        }
        manifest["manifest_sha256"] = gate.canonical_sha256(manifest)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        policy_path = repo / "configs" / "policy.json"
        policy_path.write_text("{}\n", encoding="utf-8")
        summary_root = run / "resource_probe_retry1"
        summary_root.mkdir(parents=True)
        summary = {
            "outer_probe_exit_code": 137,
            "status": "METHOD_FAILURE",
            "max_gpu_utilization_percent": 1.0,
            "peak_gpu_memory_mib": 39523.0,
            "process_maximum_rss_kib": 110096564,
            "memory_events_delta": {"oom": 0, "oom_kill": 0},
        }
        summary_path = summary_root / "resource_summary.json"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        gnu_time = summary_root / "gnu_time.txt"
        gnu_time.write_text("Command terminated by signal 9\n", encoding="utf-8")
        (run / "state.txt").write_text("TRAINING_FAILED\n", encoding="utf-8")
        (run / "retry1_probe_exit_code.txt").write_text("137\n", encoding="utf-8")
        model = run / "model"
        model.mkdir()
        for name in ("cameras.json", "cfg_args", "input.ply"):
            (model / name).write_bytes(name.encode())

        authorization = {
            "protocol_id": gate.PROTOCOL_ID,
            "authorization_id": "retry-auth",
            "status": "AUTHORIZED_NOT_STARTED",
            "method": {
                "method_id": "3dgs_original",
                "repository_commit": "2eee0e26d2d5fd00ec462df47752223952f6bf4e",
                "repository_tree": "5eee127dfc0942bf83d9fdd72328e03ec0cbf6c4",
            },
            "input": {
                "formal_input_manifest": "formal_inputs/gcp_100000_20260610/NATIVE_QUARTER_INPUT_MANIFEST.json",
                "formal_input_manifest_file_sha256": gate.file_sha256(manifest_path),
                "formal_input_manifest_canonical_sha256": manifest["manifest_sha256"],
            },
            "initial_attempt": {
                "resource_summary": "resource_probe_retry1/resource_summary.json",
                "resource_summary_sha256": gate.file_sha256(summary_path),
                "gnu_time_sha256": gate.file_sha256(gnu_time),
                "model_files_to_preserve": ["cameras.json", "cfg_args", "input.ply"],
            },
            "compatibility_policy": {
                "policy_id": gate.POLICY_ID,
                "environment": {"MALLOC_TRIM_THRESHOLD_": "0"},
                "source_contract": "configs/policy.json",
                "source_contract_sha256": gate.file_sha256(policy_path),
                "training_tensor_semantics_changed": False,
                "rng_semantics_changed": False,
                "camera_order_changed": False,
                "image_pixels_changed": False,
            },
            "retry": {
                "single_retry_allowed": True,
                "resume_allowed": False,
                "failed_model_archive": "model_failed_preiteration_eager_no_trim",
                "resource_probe_root": "resource_probe_retry2",
            },
            "scope_locks": {"other_scene_authorized": False, "other_retry_authorized": False},
        }
        auth_path = repo / "configs" / "retry.json"
        auth_path.write_text(json.dumps(authorization), encoding="utf-8")
        registry = {
            "protocol_id": gate.PROTOCOL_ID,
            "global_training_allowed": False,
            "per_method_training_allowed_methods": [],
            "preliminary_evidence_scope": {
                "six_scene_matrix_status": "LOCKED",
                "multi_seed_status": "NOT_AUTHORIZED",
            },
            "explicit_scene_run_authorization": {
                "status": "ATTEMPTED_PREITERATION_HOST_MEMORY_KILL_RELOCKED",
                "single_fresh_run_allowed": False,
                "formal_iteration_reached": 0,
                "checkpoint_file_count": 0,
            },
            "explicit_scene_compatibility_retry_authorization": {
                "authorization_id": "retry-auth",
                "status": "AUTHORIZED_NOT_STARTED",
                "authorization": "configs/retry.json",
                "authorization_sha256": gate.file_sha256(auth_path),
                "method_id": "3dgs_original",
                "scene": "gcp_100000_20260610",
                "seed": 0,
                "iterations": 30000,
                "run_root": run.as_posix(),
                "compatibility_policy": gate.POLICY_ID,
                "environment": {"MALLOC_TRIM_THRESHOLD_": "0"},
                "single_preiteration_retry_allowed": True,
                "new_run_root_allowed": False,
                "resume_allowed": False,
                "other_methods_scenes_seeds_retries_allowed": False,
            },
        }
        return registry, repo, release, run

    def test_exact_preiteration_failure_is_authorized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry, repo, release, run = self._fixture(Path(directory))
            with mock.patch.object(gate, "FROZEN_RUN_ROOT", PurePosixPath(run.as_posix())):
                result = gate.check_retry(registry, repo, release, run)
            self.assertTrue(result["passed"], result["errors"])

    def test_checkpoint_or_second_probe_denies_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry, repo, release, run = self._fixture(Path(directory))
            point_cloud = run / "model" / "point_cloud" / "iteration_1"
            point_cloud.mkdir(parents=True)
            (point_cloud / "point_cloud.ply").write_bytes(b"ply")
            (run / "resource_probe_retry2").mkdir()
            with mock.patch.object(gate, "FROZEN_RUN_ROOT", PurePosixPath(run.as_posix())):
                result = gate.check_retry(registry, repo, release, run)
            self.assertFalse(result["passed"])
            self.assertTrue(any("unexpected files" in error for error in result["errors"]))
            self.assertTrue(any("resource probe target already exists" in error for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
