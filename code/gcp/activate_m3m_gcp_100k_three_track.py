#!/usr/bin/env python3
"""Activate one independently reviewed 100K RGB/GCP/LiDAR candidate."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from m3m_gcp_lidar_artifacts import canonical_sha256, sha256_file, validate_scene_attempt_freeze


SCENE = "gcp_100000_20260610"
REVIEW_TASK_ID = "019ff12c-cb29-7cb2-8fb6-1d82c5f8c54b"
REQUIRED_REVIEW_VERDICT = "PASS_100K_THREE_TRACK_EVALUATION_ADDENDUM_V1"


def git_value(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, encoding="utf-8"
    ).strip()


def require_bound_json(row: dict[str, Any], label: str) -> tuple[Path, dict[str, Any]]:
    path = Path(str(row.get("path", ""))).resolve()
    if not path.is_file() or sha256_file(path) != row.get("sha256"):
        raise RuntimeError(f"{label} file binding mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_canonical = row.get("canonical_sha256")
    if expected_canonical is not None and (
        payload.get("canonical_sha256") != expected_canonical
        or canonical_sha256(payload) != expected_canonical
    ):
        raise RuntimeError(f"{label} canonical binding mismatch")
    return path, payload


def write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(
            descriptor,
            (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
    finally:
        os.close(descriptor)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--addendum-repo", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--reviewed-commit", required=True)
    parser.add_argument("--review-verdict", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo = args.addendum_repo.resolve()
    candidate_path = args.candidate_manifest.resolve()
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    try:
        output.relative_to(repo)
    except ValueError:
        pass
    else:
        raise RuntimeError("activation output must be outside the clean addendum checkout")
    if args.review_verdict != REQUIRED_REVIEW_VERDICT:
        raise RuntimeError("exact addendum review verdict is absent")
    if git_value(repo, "status", "--porcelain"):
        raise RuntimeError("addendum checkout is dirty")
    commit = git_value(repo, "rev-parse", "HEAD")
    tree = git_value(repo, "show", "-s", "--format=%T", "HEAD")
    if commit != args.reviewed_commit:
        raise RuntimeError("addendum checkout differs from reviewed commit")

    if not candidate_path.is_file():
        raise FileNotFoundError(candidate_path)
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    if (
        candidate.get("schema") != "m3m_gcp_100k_three_track_candidate_manifest_v1"
        or candidate.get("status") != "REVIEW_CANDIDATE_NOT_EXECUTION_AUTHORIZED"
        or candidate.get("execution_authorized") is not False
        or candidate.get("scene") != SCENE
        or candidate.get("canonical_sha256") != canonical_sha256(candidate)
        or candidate.get("review", {}).get("task_id") != REVIEW_TASK_ID
        or candidate.get("review", {}).get("required_verdict") != REQUIRED_REVIEW_VERDICT
        or candidate.get("addendum_checkout") != {"commit": commit, "tree": tree}
    ):
        raise RuntimeError("three-track candidate identity mismatch")

    _addendum_config_path, addendum_config = require_bound_json(
        candidate["addendum_config"], "three-track addendum config"
    )
    if (
        addendum_config.get("schema")
        != "m3m_gcp_native_quarter_100k_three_track_evaluation_addendum_v1"
        or addendum_config.get("status") != "REVIEW_CANDIDATE_NOT_EXECUTION_AUTHORIZED"
        or addendum_config.get("execution_authorized") is not False
        or output != Path(str(addendum_config["runtime_artifacts"]["activation_path"])).resolve()
    ):
        raise RuntimeError("three-track addendum config/runtime binding mismatch")

    _base_activation_path, base_activation = require_bound_json(candidate["base_activation"], "base activation")
    freeze_path, freeze = require_bound_json(candidate["scene_attempt_freeze"], "scene-attempt freeze")
    errors, methods = validate_scene_attempt_freeze(
        freeze, freeze_path=freeze_path, expected_scene=SCENE
    )
    if errors or methods is None:
        raise RuntimeError("scene-attempt freeze failed activation validation: " + "; ".join(errors))
    methods_path, methods_payload = require_bound_json(candidate["methods_manifest"], "methods manifest")
    if methods_path != Path(str(freeze["methods_manifest_path"])).resolve() or methods_payload != methods:
        raise RuntimeError("candidate methods/freeze binding mismatch")
    formal_input_path = Path(str(candidate["formal_input_manifest"].get("path", ""))).resolve()
    if (
        not formal_input_path.is_file()
        or sha256_file(formal_input_path) != candidate["formal_input_manifest"].get("sha256")
    ):
        raise RuntimeError("formal input manifest file binding mismatch")
    formal_input = json.loads(formal_input_path.read_text(encoding="utf-8"))
    if formal_input.get("manifest_sha256") != candidate["formal_input_manifest"]["canonical_sha256"]:
        raise RuntimeError("formal input manifest canonical-field binding mismatch")
    _camera_path, camera = require_bound_json(candidate["rgb_camera_root_manifest"], "RGB camera root")
    _gcp_camera_path, gcp_camera = require_bound_json(
        candidate["gcp_camera_root_manifest"], "GCP camera root"
    )
    registry_path, registry = require_bound_json(candidate["rgb_registry"], "RGB registry")
    legacy_path, legacy = require_bound_json(candidate["legacy_3dgs_gcp_adoption"], "legacy GCP adoption")
    if (
        registry.get("status") != "ACTIVE_FROZEN"
        or registry.get("scene") != SCENE
        or registry.get("ready_method_ids")
        != [row["method_id"] for row in methods["methods"] if row["attempt_status"] == "READY_FOR_EVALUATION"]
        or legacy.get("status") != "PASS_LEGACY_GCP_ADOPTION_CANDIDATE"
        or legacy.get("scene_attempt_freeze_sha256") != sha256_file(freeze_path)
        or legacy.get("methods_manifest_file_sha256") != sha256_file(methods_path)
        or camera.get("status") != "PASS_RGB_EVALUATION_CAMERA_ROOT"
        or gcp_camera.get("status")
        != "PASS_GCP_EVALUATION_CAMERA_ROOT_NO_RGB_PIXELS"
        or gcp_camera.get("protocol_observations", {}).get("observation_count") != 256
        or gcp_camera.get("protocol_observations", {}).get("unique_camera_count") != 211
        or gcp_camera.get("protocol_observations", {}).get("formal_role_counts")
        != {"train": 187, "test": 24}
        or gcp_camera.get("rgb_truth_boundary", {}).get("real_rgb_pixels_present") is not False
        or base_activation.get("execution_authorized") is not True
    ):
        raise RuntimeError("candidate component semantic binding mismatch")

    formal_results_root = Path(str(candidate.get("formal_results_root", ""))).resolve()
    if formal_results_root.exists() or formal_results_root.is_symlink():
        raise FileExistsError("formal results root must be absent at activation")
    if output.parent != candidate_path.parent.parent:
        raise RuntimeError("activation output must be the sibling of the candidate directory")

    payload: dict[str, Any] = {
        "schema": "m3m_gcp_100k_three_track_activation_v1",
        "status": "ACTIVE_FROZEN",
        "execution_authorized": True,
        "scene": SCENE,
        "review_task_id": REVIEW_TASK_ID,
        "review_verdict": REQUIRED_REVIEW_VERDICT,
        "reviewed_addendum_commit": commit,
        "reviewed_addendum_tree": tree,
        "candidate_manifest_path": str(candidate_path),
        "candidate_manifest_sha256": sha256_file(candidate_path),
        "candidate_manifest_canonical_sha256": candidate["canonical_sha256"],
        "addendum_config_sha256": candidate["addendum_config"]["sha256"],
        "base_activation_sha256": candidate["base_activation"]["sha256"],
        "scene_attempt_freeze_sha256": candidate["scene_attempt_freeze"]["sha256"],
        "methods_manifest_sha256": candidate["methods_manifest"]["sha256"],
        "rgb_camera_root_manifest_sha256": candidate["rgb_camera_root_manifest"]["sha256"],
        "gcp_camera_root_manifest_sha256": candidate["gcp_camera_root_manifest"]["sha256"],
        "rgb_contract_sha256": candidate["rgb_contract"]["sha256"],
        "rgb_registry_path": str(registry_path),
        "rgb_registry_sha256": sha256_file(registry_path),
        "legacy_3dgs_gcp_adoption_path": str(legacy_path),
        "legacy_3dgs_gcp_adoption_sha256": sha256_file(legacy_path),
        "formal_results_root": str(formal_results_root),
        "packet_release_gate": candidate["packet_release_gate"],
    }
    payload["canonical_sha256"] = canonical_sha256(payload)
    write_exclusive(output, payload)
    print(
        json.dumps(
            {
                "status": "PASS_100K_THREE_TRACK_ACTIVATED",
                "path": str(output),
                "sha256": sha256_file(output),
                "canonical_sha256": payload["canonical_sha256"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
