#!/usr/bin/env python3
"""Guard one exact 211-camera post-freeze GCP packet export."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import run_m3m_gcp_100k_packet_export as packet_dispatch
from m3m_gcp_100k_raw_packet_state import acquire_active_raw_packet_state
from m3m_gcp_100k_three_track_runtime import validate_addendum_runtime
from m3m_gcp_lidar_artifacts import canonical_sha256, command_sha256, sha256_file
from run_m3m_gcp_100k_guarded import (
    PACKET_CAP_BYTES,
    REQUIRED_NOFILE_SOFT,
    cgroup_memory_events,
    configure_nofile_limit,
    directory_bytes,
    gpu_memory_for_pid,
    memory_event_delta,
    observe_child_nofile_limit,
    require_idle_gpu,
    utc_now,
    validate_capacity,
    validate_external_files,
    validate_model_identity_bundle,
    validate_source_binding,
)


SCENE = "gcp_100000_20260610"
ALLOWED_METHODS = {"citygs_x", "metrogs"}


def require_json(path: Path, expected_sha: str | None = None) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if expected_sha is not None and sha256_file(path) != expected_sha:
        raise RuntimeError(f"file SHA mismatch: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


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


def file_identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def validate_frozen_packet_python(
    recipe: dict[str, Any], *, current_python: Path
) -> Path:
    command = recipe.get("phase_commands", {}).get("packet", [])
    if not isinstance(command, list) or not command:
        raise RuntimeError("recipe lacks a frozen packet command")
    frozen = Path(str(command[0])).expanduser()
    if not frozen.is_absolute() or not frozen.is_file():
        raise RuntimeError(f"frozen packet Python is missing or not absolute: {frozen}")
    if current_python.resolve() != frozen.resolve():
        raise RuntimeError(
            f"GCP packet guard must run with recipe-frozen Python: {frozen}; "
            f"observed {current_python.resolve()}"
        )
    return frozen


def load_activation(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    activation = require_json(path)
    if (
        activation.get("schema") != "m3m_gcp_100k_three_track_activation_v1"
        or activation.get("status") != "ACTIVE_FROZEN"
        or activation.get("execution_authorized") is not True
        or activation.get("scene") != SCENE
        or activation.get("canonical_sha256") != canonical_sha256(activation)
    ):
        raise RuntimeError("three-track activation mismatch")
    candidate_path = Path(str(activation["candidate_manifest_path"])).resolve()
    candidate = require_json(candidate_path, str(activation["candidate_manifest_sha256"]))
    if (
        candidate.get("canonical_sha256") != activation["candidate_manifest_canonical_sha256"]
        or canonical_sha256(candidate) != activation["candidate_manifest_canonical_sha256"]
    ):
        raise RuntimeError("activation/candidate binding mismatch")
    registry_path = Path(str(candidate["rgb_registry"]["path"])).resolve()
    registry = require_json(registry_path, str(candidate["rgb_registry"]["sha256"]))
    if (
        registry.get("canonical_sha256") != candidate["rgb_registry"]["canonical_sha256"]
        or canonical_sha256(registry) != candidate["rgb_registry"]["canonical_sha256"]
    ):
        raise RuntimeError("candidate registry binding mismatch")
    return activation, candidate, registry


def validate_camera_manifest(path: Path) -> tuple[dict[str, Any], Path, list[str]]:
    payload = require_json(path)
    output = payload.get("output", {})
    observations = payload.get("protocol_observations", {})
    if (
        payload.get("schema") != "m3m_gcp_100k_gcp_evaluation_camera_root_v1"
        or payload.get("status")
        != "PASS_GCP_EVALUATION_CAMERA_ROOT_NO_RGB_PIXELS"
        or payload.get("scene") != SCENE
        or payload.get("canonical_sha256") != canonical_sha256(payload)
        or observations.get("observation_count") != 256
        or observations.get("unique_camera_count") != 211
        or observations.get("formal_role_counts") != {"train": 187, "test": 24}
        or payload.get("rgb_truth_boundary", {}).get("real_rgb_pixels_present") is not False
        or output.get("camera_view_count") != 211
    ):
        raise RuntimeError("GCP camera-root manifest mismatch")
    root = Path(str(output.get("root", ""))).resolve()
    if path.resolve() != root / "GCP_EVALUATION_CAMERA_ROOT_MANIFEST.json":
        raise RuntimeError("GCP camera-root manifest path mismatch")
    names = [str(value) for value in output.get("image_names", [])]
    if len(names) != 211 or len(set(names)) != 211:
        raise RuntimeError("GCP camera-root name inventory mismatch")
    placeholder_sha = output.get("placeholder", {}).get("sha256")
    for name in names:
        image = root / "images" / name
        if not image.is_file() or sha256_file(image) != placeholder_sha:
            raise RuntimeError(f"GCP camera root exposes non-placeholder RGB: {name}")
    for name, row in output.get("sparse_files", {}).items():
        sparse = root / "sparse" / "0" / str(name)
        if (
            not sparse.is_file()
            or sparse.stat().st_size != row.get("bytes")
            or sha256_file(sparse) != row.get("sha256")
        ):
            raise RuntimeError(f"GCP camera sparse identity mismatch: {name}")
    return payload, root, sorted(names)


def validate_packet_outputs(root: Path, expected_names: list[str]) -> list[dict[str, Any]]:
    manifest_path = root / "depth_export_manifest.json"
    mapping_path = root / "depth_map_index.csv"
    manifest = require_json(manifest_path)
    depth_rows = manifest.get("depth_index", [])
    packet_rows = manifest.get("packet_index", [])
    actual_names = [str(row.get("image_name", "")) for row in depth_rows]
    if (
        manifest.get("schema") != "ms_gcp_metric_depth_packet_manifest_v2"
        or manifest.get("protocol_id") != "m3m_gcp_native_quarter_geometry_v2"
        or manifest.get("scene") != SCENE
        or manifest.get("rendered_view_count") != 211
        or len(depth_rows) != 211
        or len(packet_rows) != 211
        or len(set(actual_names)) != 211
        or set(actual_names) != set(expected_names)
    ):
        raise RuntimeError("GCP packet manifest is not the exact 211-camera observation set")
    products = [file_identity(manifest_path)]
    root_resolved = root.resolve()
    packet_paths: list[Path] = []
    for row in depth_rows:
        packet = Path(str(row.get("packet_path", ""))).resolve()
        if (
            packet.parent != root_resolved
            or not packet.is_file()
            or packet.stat().st_size != int(row.get("packet_bytes", -1))
            or sha256_file(packet) != row.get("packet_sha256")
            or row.get("packet_recompute_passed") is not True
        ):
            raise RuntimeError(f"GCP packet identity mismatch: {packet}")
        packet_paths.append(packet)
    if len(set(packet_paths)) != 211:
        raise RuntimeError("GCP packet file paths are not unique")
    if not mapping_path.is_file():
        raise FileNotFoundError(mapping_path)
    with mapping_path.open("r", encoding="utf-8-sig", newline="") as handle:
        mapping = list(csv.DictReader(handle))
    if (
        len(mapping) != 211
        or [str(row.get("image_name", "")) for row in mapping] != actual_names
        or [Path(str(row.get("packet_path", ""))).resolve() for row in mapping]
        != packet_paths
    ):
        raise RuntimeError("GCP packet mapping differs from the manifest")
    products.append(file_identity(mapping_path))
    products.extend(file_identity(path) for path in sorted(packet_paths, key=str))
    return products


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--method-id", choices=tuple(sorted(ALLOWED_METHODS)), required=True)
    parser.add_argument("--camera-root-manifest", type=Path, required=True)
    parser.add_argument("--packet-set-root", type=Path, required=True)
    parser.add_argument("--packet-state", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--capacity-root", type=Path, default=Path("/root/autodl-tmp"))
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    args = parser.parse_args()

    activation_path = args.activation.resolve()
    activation, candidate, registry = load_activation(activation_path)
    validate_addendum_runtime(
        activation=activation,
        candidate=candidate,
        registry=registry,
        executing_file=Path(__file__),
    )
    methods = {
        str(row["method_id"]): row for row in registry.get("methods", [])
    }
    if args.method_id not in methods or args.method_id not in registry.get(
        "ready_method_ids", []
    ):
        raise RuntimeError("GCP packet method is not activated READY")
    method = methods[args.method_id]
    base_repo = Path(str(candidate["base_checkout"]["path"])).resolve()
    if (
        subprocess.check_output(["git", "-C", str(base_repo), "rev-parse", "HEAD"], text=True).strip()
        != candidate["base_checkout"]["commit"]
        or subprocess.check_output(
            ["git", "-C", str(base_repo), "show", "-s", "--format=%T", "HEAD"], text=True
        ).strip()
        != candidate["base_checkout"]["tree"]
        or subprocess.check_output(
            ["git", "-C", str(base_repo), "status", "--porcelain"], text=True
        ).strip()
    ):
        raise RuntimeError("frozen base checkout identity mismatch")
    recipe_path = Path(str(method["recipe_path"])).resolve()
    recipe = require_json(recipe_path, str(method["recipe_sha256"]))
    run_root = Path(str(method["run_root"])).resolve()
    source_root = Path(str(recipe["source_bindings"]["packet"]["root"])).resolve()
    if (
        recipe.get("method_id") != args.method_id
        or Path(str(recipe.get("authorized_run_root", ""))).resolve() != run_root
    ):
        raise RuntimeError("GCP packet recipe/run-root mismatch")
    frozen_packet_python = validate_frozen_packet_python(
        recipe, current_python=Path(sys.executable)
    )
    bound_recipe = dict(recipe)
    bound_recipe["_recipe_path"] = str(recipe_path)
    identity_path = Path(str(method["attempt_model_identity_path"])).resolve()
    identity = validate_model_identity_bundle(
        manifest_path=identity_path,
        method_id=args.method_id,
        run_root=run_root,
        recipe=bound_recipe,
        repo=base_repo,
    )
    if (
        sha256_file(identity_path) != method["attempt_model_identity_sha256"]
        or identity["canonical_sha256"]
        != method["attempt_model_identity_canonical_sha256"]
    ):
        raise RuntimeError("GCP packet model identity differs from activated registry")
    validate_source_binding(recipe, source_root, "packet")
    validate_external_files(recipe, "packet")
    camera_manifest, camera_root, expected_names = validate_camera_manifest(
        args.camera_root_manifest.resolve()
    )
    bound_camera = candidate["gcp_camera_root_manifest"]
    if (
        args.camera_root_manifest.resolve()
        != Path(str(bound_camera["path"])).resolve()
        or sha256_file(args.camera_root_manifest.resolve()) != bound_camera["sha256"]
        or camera_manifest["canonical_sha256"] != bound_camera["canonical_sha256"]
        or activation.get("gcp_camera_root_manifest_sha256") != bound_camera["sha256"]
    ):
        raise RuntimeError("GCP packet camera root differs from the activated candidate")

    runtime_root = Path(str(candidate["candidate_output_root"])).resolve().parent
    packet_root = args.packet_set_root.resolve()
    packet_state = args.packet_state.resolve()
    evidence_root = args.evidence_root.resolve()
    expected_packet_root = runtime_root / "gcp-packet-scratch" / args.method_id
    expected_state = runtime_root / "gcp-packet-scratch" / "ACTIVE_GCP_PACKET_STATE.json"
    expected_evidence = runtime_root / "gcp-packet-evidence" / args.method_id
    if (
        packet_root != expected_packet_root
        or packet_state != expected_state
        or evidence_root != expected_evidence
    ):
        raise RuntimeError("GCP packet scratch/evidence paths differ from frozen runtime namespace")
    for path in (packet_root, packet_state, evidence_root):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"GCP packet phase requires a fresh path: {path}")
    validate_capacity(args.capacity_root.resolve(), "packet")
    gpu_prelaunch = require_idle_gpu()
    nofile = configure_nofile_limit(REQUIRED_NOFILE_SOFT)

    allowlist = evidence_root / "gcp_observation_camera_allowlist.csv"
    roots = recipe["phase_roots"]["packet"]
    dispatch_args = SimpleNamespace(
        method_id=args.method_id,
        benchmark_repo=base_repo,
        evaluation_repo=source_root,
        training_run_root=run_root,
        dataset_root=Path(str(roots["dataset_root"])).resolve(),
        prior_root=Path(str(roots["prior_root"])).resolve(),
        camera_root=camera_root,
        train_allowlist=allowlist,
        packet_set_root=packet_root,
    )
    command = packet_dispatch.build_command(dispatch_args)
    if Path(command[0]).resolve() != frozen_packet_python.resolve():
        raise RuntimeError("GCP exporter command did not retain recipe-frozen Python")
    command[0] = str(frozen_packet_python)

    global_state_path, global_state = acquire_active_raw_packet_state(
        activation_path=activation_path,
        candidate=candidate,
        registry=registry,
        method_id=args.method_id,
        track="gcp",
        recipe_sha256=sha256_file(recipe_path),
        attempt_model_identity_sha256=sha256_file(identity_path),
        packet_set_root=packet_root,
        track_packet_state_path=packet_state,
        owner_evidence_root=evidence_root,
    )
    evidence_root.mkdir(parents=True)
    with allowlist.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_name"])
        writer.writeheader()
        writer.writerows({"image_name": name} for name in expected_names)
    stdout_path = evidence_root / "command.stdout.log"
    stderr_path = evidence_root / "command.stderr.log"
    environment_path = evidence_root / "environment.json"
    success_path = evidence_root / "phase_success.json"
    failure_path = evidence_root / "failure.json"
    state = {
        "schema": "m3m_gcp_100k_single_gcp_packet_state_v1",
        "scene": SCENE,
        "method_id": args.method_id,
        "packet_set_root": str(packet_root),
        "three_track_activation_sha256": sha256_file(activation_path),
        "candidate_manifest_sha256": sha256_file(
            Path(str(activation["candidate_manifest_path"])).resolve()
        ),
        "scene_attempt_freeze_sha256": candidate["scene_attempt_freeze"]["sha256"],
        "methods_manifest_sha256": candidate["methods_manifest"]["sha256"],
        "recipe_sha256": sha256_file(recipe_path),
        "attempt_model_identity_sha256": sha256_file(identity_path),
        "gcp_camera_root_manifest_sha256": sha256_file(
            args.camera_root_manifest.resolve()
        ),
        "protocol_observation_count": 256,
        "packet_view_count": 211,
        "formal_role_counts": {"train": 187, "test": 24},
        "global_raw_packet_state_path": str(global_state_path),
        "global_raw_packet_state_sha256": sha256_file(global_state_path),
        "created_at_utc": utc_now(),
    }
    state["canonical_sha256"] = canonical_sha256(state)
    write_exclusive(packet_state, state)

    started = utc_now()
    before = cgroup_memory_events()
    environment = {
        "schema": "m3m_gcp_100k_gcp_packet_environment_v1",
        "scene": SCENE,
        "method_id": args.method_id,
        "argv": command,
        "python": sys.version,
        "frozen_packet_python": file_identity(frozen_packet_python),
        "gpu_prelaunch": gpu_prelaunch,
        "resource_limits": nofile,
        "three_track_activation_sha256": sha256_file(activation_path),
        "scene_attempt_freeze_sha256": candidate["scene_attempt_freeze"]["sha256"],
        "global_raw_packet_state_path": str(global_state_path),
        "global_raw_packet_state_sha256": sha256_file(global_state_path),
        "global_raw_packet_state_canonical_sha256": global_state["canonical_sha256"],
        "started_at_utc": started,
    }
    with stdout_path.open("x", encoding="utf-8") as stdout_handle, stderr_path.open(
        "x", encoding="utf-8"
    ) as stderr_handle:
        process = subprocess.Popen(
            command,
            cwd=source_root,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
        )
        child_nofile = observe_child_nofile_limit(process)
        environment["resource_limits"]["child_actual"] = child_nofile
        environment["canonical_sha256"] = canonical_sha256(environment)
        write_exclusive(environment_path, environment)
        peak_gpu = 0.0
        cap_error: str | None = None
        while process.poll() is None:
            peak_gpu = max(peak_gpu, gpu_memory_for_pid(process.pid))
            if directory_bytes(packet_root) > PACKET_CAP_BYTES:
                cap_error = "GCP packet scratch exceeded 100 GiB"
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
            time.sleep(max(1.0, args.poll_seconds))
        exit_code = int(process.wait())
    after = cgroup_memory_events()
    delta = memory_event_delta(before, after)
    ended = utc_now()
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
    oom = (
        ("cuda" in stderr_text.lower() and "out of memory" in stderr_text.lower())
        or delta["oom"] > 0
        or delta["oom_kill"] > 0
    )
    try:
        if exit_code != 0:
            raise RuntimeError(f"GCP packet exporter exited with code {exit_code}")
        if cap_error:
            raise RuntimeError(cap_error)
        products = validate_packet_outputs(packet_root, expected_names)
        if directory_bytes(packet_root) > PACKET_CAP_BYTES:
            raise RuntimeError("GCP packet scratch exceeded 100 GiB after validation")
    except Exception as exc:
        failure: dict[str, Any] = {
            "schema": "m3m_gcp_100k_gcp_packet_failure_v1",
            "status": "OOM_UNRANKED" if oom else "FAILED_UNRANKED",
            "scene": SCENE,
            "method_id": args.method_id,
            "failure_stage": "gcp_packet_export",
            "three_track_activation_sha256": sha256_file(activation_path),
            "scene_attempt_freeze_sha256": candidate["scene_attempt_freeze"]["sha256"],
            "attempt_model_identity_sha256": sha256_file(identity_path),
            "command_sha256": command_sha256(command),
            "environment_sha256": sha256_file(environment_path),
            "packet_state_sha256": sha256_file(packet_state),
            "global_raw_packet_state_path": str(global_state_path),
            "global_raw_packet_state_sha256": sha256_file(global_state_path),
            "stdout_sha256": sha256_file(stdout_path),
            "stderr_sha256": sha256_file(stderr_path),
            "logs": [file_identity(stdout_path), file_identity(stderr_path)],
            "exit_code": exit_code,
            "peak_gpu_memory_mib": peak_gpu,
            "cgroup_memory_events_delta": delta,
            "error": f"{type(exc).__name__}: {exc}",
            "retry_forbidden_after_export_child_start": True,
            "started_at_utc": started,
            "ended_at_utc": ended,
        }
        failure["canonical_sha256"] = canonical_sha256(failure)
        write_exclusive(failure_path, failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2, sort_keys=True))
        return 1

    progress_values = [
        int(value)
        for value in re.findall(r"([0-9]+)/211", stdout_path.read_text(encoding="utf-8", errors="replace") + stderr_text)
    ]
    success: dict[str, Any] = {
        "schema": "m3m_gcp_100k_gcp_packet_phase_success_v1",
        "status": "PASS_GCP_PACKET_211",
        "scene": SCENE,
        "method_id": args.method_id,
        "three_track_activation_path": str(activation_path),
        "three_track_activation_sha256": sha256_file(activation_path),
        "candidate_manifest_sha256": sha256_file(
            Path(str(activation["candidate_manifest_path"])).resolve()
        ),
        "scene_attempt_freeze_sha256": candidate["scene_attempt_freeze"]["sha256"],
        "methods_manifest_sha256": candidate["methods_manifest"]["sha256"],
        "recipe_path": str(recipe_path),
        "recipe_sha256": sha256_file(recipe_path),
        "attempt_model_identity_path": str(identity_path),
        "attempt_model_identity_sha256": sha256_file(identity_path),
        "gcp_camera_root_manifest_path": str(args.camera_root_manifest.resolve()),
        "gcp_camera_root_manifest_sha256": sha256_file(args.camera_root_manifest.resolve()),
        "protocol_observation_count": 256,
        "packet_view_count": 211,
        "formal_role_counts": {"train": 187, "test": 24},
        "packet_set_root": str(packet_root),
        "packet_state_path": str(packet_state),
        "packet_state_sha256": sha256_file(packet_state),
        "global_raw_packet_state_path": str(global_state_path),
        "global_raw_packet_state_sha256": sha256_file(global_state_path),
        "allowlist_path": str(allowlist),
        "allowlist_sha256": sha256_file(allowlist),
        "command": command,
        "command_sha256": command_sha256(command),
        "environment_path": str(environment_path),
        "environment_sha256": sha256_file(environment_path),
        "stdout_path": str(stdout_path),
        "stdout_sha256": sha256_file(stdout_path),
        "stderr_path": str(stderr_path),
        "stderr_sha256": sha256_file(stderr_path),
        "last_observed_progress": max(progress_values, default=211),
        "peak_gpu_memory_mib": peak_gpu,
        "cgroup_memory_events_delta": delta,
        "products": products,
        "started_at_utc": started,
        "ended_at_utc": ended,
    }
    success["canonical_sha256"] = canonical_sha256(success)
    write_exclusive(success_path, success)
    print(json.dumps(success, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
