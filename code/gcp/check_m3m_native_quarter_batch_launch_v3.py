#!/usr/bin/env python3
"""Fail closed unless one exact v3 batch method has a valid single-use launch gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any


PROTOCOL_ID = "m3m_gcp_native_quarter_geometry_v2"
BATCH_ID = "m3m-gcp-3k-eight-method-seed0-20260818"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_launch(
    registry: dict[str, Any],
    repo_root: Path,
    *,
    method_id: str,
    scene: str,
    seed: int,
    budget_value: int,
    run_root: str,
    run_root_exists: bool,
) -> dict[str, Any]:
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(registry.get("schema") == "m3m_gcp_native_quarter_method_registry_v3", "registry schema mismatch")
    require(registry.get("protocol_id") == PROTOCOL_ID, "registry protocol mismatch")
    require(registry.get("batch_id") == BATCH_ID, "registry batch mismatch")
    require(registry.get("global_training_allowed") is False, "global training must remain locked")
    require(registry.get("per_method_training_allowed_methods") == [method_id], "method allowlist is not exact")
    require(registry.get("batch_controller", {}).get("maximum_concurrent_formal_trainings") == 1, "formal concurrency is not one")

    methods = {str(item.get("method_id")): item for item in registry.get("methods", [])}
    method = methods.get(method_id)
    require(method is not None, f"unknown method: {method_id}")
    if method is not None:
        require(method.get("lifecycle_role") == "ACTIVE_3K_CANDIDATE", "method is not an active 3K candidate")
        require(method.get("technical_qualification_status") == "TECHNICALLY_QUALIFIED", "method is not technically qualified")
        require(method.get("three_k_training_allowed") is True, "method training flag is locked")
        require(method.get("formal_3k_result", {}).get("status") == "NOT_ATTEMPTED", "method already has a formal attempt")
        require(method.get("six_scene_run_allowed") is False, "six-scene execution must remain locked")

    gate_ref = registry.get("current_one_use_launch_gate")
    require(isinstance(gate_ref, dict), "current one-use launch gate is absent")
    gate: dict[str, Any] = {}
    gate_path: Path | None = None
    if isinstance(gate_ref, dict):
        require(gate_ref.get("method_id") == method_id, "gate reference method mismatch")
        relative = gate_ref.get("path")
        if isinstance(relative, str) and relative:
            gate_path = (repo_root / relative).resolve()
            require(gate_path.is_relative_to(repo_root.resolve()), "gate file escapes repository")
            require(gate_path.is_file(), "gate file is missing")
            if gate_path.is_file():
                require(file_sha256(gate_path) == gate_ref.get("sha256"), "gate file SHA mismatch")
                gate = json.loads(gate_path.read_text(encoding="utf-8"))
        else:
            errors.append("gate file path is missing")

    require(gate.get("schema") == "m3m_gcp_native_quarter_one_use_method_launch_gate_v1", "gate schema mismatch")
    require(gate.get("status") == "AUTHORIZED_NOT_STARTED", "gate is not launchable")
    require(gate.get("protocol_id") == PROTOCOL_ID, "gate protocol mismatch")
    require(gate.get("batch_id") == BATCH_ID, "gate batch mismatch")
    require(gate.get("method_id") == method_id, "gate method mismatch")
    require(gate.get("scene") == scene, "gate scene mismatch")
    require(gate.get("seed") == seed == 0, "gate seed mismatch")
    require(gate.get("official_budget", {}).get("value") == budget_value, "gate budget mismatch")
    require(gate.get("run_root") == run_root, "gate run root mismatch")
    require(gate.get("single_fresh_run_allowed") is True, "gate does not allow one fresh run")
    require(gate.get("resume_allowed") is False, "gate permits resume")
    require(gate.get("overwrite_allowed") is False, "gate permits overwrite")
    require(gate.get("result_driven_retry_allowed") is False, "gate permits result-driven retry")
    if method_id == "metrogs":
        require(gate.get("maximum_training_wall_seconds") == 54000, "MetroGS wall-time limit mismatch")
        require(
            gate.get("wall_time_limit_enforced_by")
            == "/usr/bin/timeout --signal=TERM --kill-after=120s 54000s",
            "MetroGS wall-time enforcement mismatch",
        )
        require(
            gate.get("wall_time_limit_terminal_status") == "INCOMPLETE_UNRANKED",
            "MetroGS wall-time terminal status mismatch",
        )
        require(gate.get("wall_time_limit_retry_allowed") is False, "MetroGS wall-time limit permits retry")

    if method is not None:
        source = method.get("source", {})
        require(gate.get("source", {}).get("commit") == source.get("commit"), "gate source commit mismatch")
        require(gate.get("source", {}).get("tree") == source.get("tree"), "gate source tree mismatch")
        recipe_path = repo_root / str(method.get("recipe", ""))
        require(recipe_path.is_file(), "frozen method recipe is missing")
        if recipe_path.is_file():
            require(file_sha256(recipe_path) == method.get("recipe_sha256"), "frozen recipe SHA mismatch")
            require(gate.get("recipe_sha256") == method.get("recipe_sha256"), "gate recipe SHA mismatch")

        adapter_relative = method.get("adapter_config") or method.get("renderer_adapter")
        adapter_sha = method.get("adapter_config_sha256") or method.get("renderer_adapter_sha256")
        adapter_path = repo_root / str(adapter_relative or "")
        require(isinstance(adapter_relative, str) and bool(adapter_relative), "method adapter path is missing")
        require(isinstance(adapter_sha, str) and len(adapter_sha) == 64, "method adapter SHA is missing")
        require(adapter_path.is_file(), "frozen method adapter is missing")
        if adapter_path.is_file():
            require(file_sha256(adapter_path) == adapter_sha, "frozen adapter SHA mismatch")
            require(gate.get("adapter_config") == adapter_relative, "gate adapter path mismatch")
            require(gate.get("adapter_config_sha256") == adapter_sha, "gate adapter SHA mismatch")

        qualification_relative = method.get("qualification_report")
        qualification_sha = method.get("qualification_report_sha256")
        qualification_path = repo_root / str(qualification_relative or "")
        require(
            isinstance(qualification_relative, str) and bool(qualification_relative),
            "method qualification report path is missing",
        )
        require(
            isinstance(qualification_sha, str) and len(qualification_sha) == 64,
            "method qualification report SHA is missing",
        )
        require(qualification_path.is_file(), "method qualification report is missing")
        if qualification_path.is_file():
            require(file_sha256(qualification_path) == qualification_sha, "method qualification report SHA mismatch")
            require(gate.get("qualification_report") == qualification_relative, "gate qualification report path mismatch")
            require(
                gate.get("qualification_report_sha256") == qualification_sha,
                "gate qualification report SHA mismatch",
            )

        truth_relative = method.get("truth_deny_report")
        truth_sha = method.get("truth_deny_report_sha256")
        if truth_relative is not None or truth_sha is not None:
            truth_path = repo_root / str(truth_relative or "")
            require(isinstance(truth_relative, str) and bool(truth_relative), "method truth-deny path is missing")
            require(isinstance(truth_sha, str) and len(truth_sha) == 64, "method truth-deny SHA is missing")
            require(truth_path.is_file(), "method truth-deny report is missing")
            if truth_path.is_file():
                require(file_sha256(truth_path) == truth_sha, "method truth-deny report SHA mismatch")
                require(gate.get("truth_deny_report") == truth_relative, "gate truth-deny path mismatch")
                require(gate.get("truth_deny_report_sha256") == truth_sha, "gate truth-deny SHA mismatch")

    required_gates = (
        "source_frozen",
        "recipe_frozen",
        "environment_ready",
        "adapter_conformant",
        "synthetic_preflight_pass",
        "real_camera_preflight_pass",
        "evaluator_preflight_pass",
        "truth_deny_pass",
    )
    for key in required_gates:
        require(gate.get("qualification_gates", {}).get(key) is True, f"qualification gate did not pass: {key}")

    boundary = registry.get("training_data_boundary", {})
    require(
        gate.get("source_scene_root") == boundary.get("allowed_source_scene_root"),
        "source scene root mismatch",
    )
    require(
        gate.get("training_input_root") == boundary.get("formal_training_input_root"),
        "formal training input root mismatch",
    )
    require(gate.get("denied_truth_roots") == boundary.get("denied_truth_roots"), "truth deny roots mismatch")
    require(gate.get("gcp_truth_training_access") is False, "gate enables GCP truth access")
    require(gate.get("lidar_training_access") is False, "gate enables LiDAR access")
    require(gate.get("heldout_rgb_training_access") is False, "gate enables held-out RGB access")

    pure_run = PurePosixPath(run_root)
    expected_parent = PurePosixPath("/root/autodl-tmp/runs/m3m-gcp-native-quarter") / method_id / scene
    require(pure_run.is_absolute(), "run root must be absolute")
    require(pure_run.parent == expected_parent, "run root is outside the exact method/scene namespace")
    require(not run_root_exists, "run root already exists; overwrite and resume are forbidden")

    return {
        "schema": "m3m_gcp_native_quarter_batch_launch_check_v3",
        "protocol_id": PROTOCOL_ID,
        "batch_id": BATCH_ID,
        "method_id": method_id,
        "scene": scene,
        "seed": seed,
        "official_budget_value": budget_value,
        "run_root": run_root,
        "gate": str(gate_path) if gate_path else None,
        "passed": not errors,
        "status": "AUTHORIZED" if not errors else "DENIED",
        "errors": errors,
    }


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--registry",
        type=Path,
        default=repo_root / "configs" / "m3m_gcp_native_quarter_method_registry_v3.json",
    )
    parser.add_argument("--method-id", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--budget-value", required=True, type=int)
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    result = check_launch(
        registry,
        args.repo_root.resolve(),
        method_id=args.method_id,
        scene=args.scene,
        seed=args.seed,
        budget_value=args.budget_value,
        run_root=args.run_root,
        run_root_exists=Path(args.run_root).exists(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
