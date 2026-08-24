#!/usr/bin/env python3
"""Build the approved 314-view 100K held-out LiDAR candidate plan."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
from typing import Any

from build_m3m_gcp_100k_success_geometry_plan import (
    FORMAL_ROOT,
    GEOMETRY_EVALUATION_ROOTS,
    GEOMETRY_PACKET_PYTHONS,
    GSPRIOR_ROOT,
    LIDAR_ENV,
    LIDAR_PAYLOAD_SHA256_INVENTORY,
    LIDAR_ROOT,
    METHODS,
    PROTOCOL_DATA_ROOT,
    PROTOCOL_ROOT,
    SCENE,
    TRAIN_ROOT,
    environment,
    packet_command,
    phase,
    read_json,
    write_exclusive,
)
from m3m_gcp_100k_geometry_paths import lidar_heldout_candidate_packet_root
from m3m_gcp_lidar_artifacts import canonical_sha256, sha256_file
from rgb_quality_contract import validate_benchmark_checkout


RGB_CAMERA_ROOT = Path(
    f"/root/autodl-tmp/datasets/"
    f"M3M-GCP-100K-rgb-evaluation-camera-root-v1/{SCENE}"
)
EXPECTED_HELDOUT_VIEWS = 314


def write_allowlist_exclusive(path: Path, names: list[str]) -> None:
    if len(names) != EXPECTED_HELDOUT_VIEWS or len(set(names)) != len(names):
        raise ValueError("held-out allowlist must contain 314 unique names")
    stream = io.StringIO(newline="\n")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["image_name"])
    writer.writerows((name,) for name in names)
    payload = stream.getvalue().encode("utf-8")
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
        0o444,
    )
    try:
        os.write(descriptor, payload)
    finally:
        os.close(descriptor)


def heldout_names(split: dict[str, Any]) -> list[str]:
    rows = [row for row in split["scenes"] if row["scene"] == SCENE]
    if len(rows) != 1:
        raise ValueError("split lacks a unique 100K scene")
    names = [str(name) for name in rows[0]["test_image_names"]]
    if len(names) != EXPECTED_HELDOUT_VIEWS or len(set(names)) != len(names):
        raise ValueError("split lacks exactly 314 unique held-out views")
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-repo", type=Path, required=True)
    parser.add_argument("--benchmark-commit", required=True)
    parser.add_argument("--benchmark-tree", required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allowlist-output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.benchmark_repo.expanduser().resolve()
    runtime = args.runtime_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    allowlist = args.allowlist_output.expanduser().resolve()
    benchmark = validate_benchmark_checkout(
        benchmark_repo=repo,
        expected_commit=args.benchmark_commit,
        expected_tree=args.benchmark_tree,
        entrypoint=Path(__file__).resolve(),
    )

    registry_path = runtime / "rgb_success_registry.json"
    inventory_path = runtime / "qualification_outcome_inventory.json"
    promotion_path = runtime / "success_subset_promotion_receipt.json"
    registry = read_json(registry_path)
    inventory = read_json(inventory_path)
    promotion = read_json(promotion_path)
    if (
        registry.get("schema")
        != "m3m_gcp_native_quarter_rgb_quality_100k_success_registry_v1"
        or registry.get("status") != "ACTIVE_FROZEN"
        or registry.get("canonical_sha256") != canonical_sha256(registry)
        or registry.get("ready_method_ids") != METHODS
        or inventory.get("canonical_sha256") != canonical_sha256(inventory)
        or promotion.get("canonical_sha256") != canonical_sha256(promotion)
        or promotion.get("eligible_method_ids") != METHODS
    ):
        raise ValueError("success runtime identity mismatch")

    contract = repo / "configs/m3m_gcp_lidar_formal_v1.json"
    artifact_schema = repo / "configs/m3m_gcp_lidar_formal_artifact_schema_v1.json"
    split_path = repo / "configs/gs_gcp_rgb_holdout_split_manifest_v1.json"
    gcp_csv = (
        PROTOCOL_DATA_ROOT
        / "benchmark/source_release_v1_3_0/gcp_points_cgcs2000_cm108_v1_3_0.csv"
    )
    sim3 = PROTOCOL_ROOT / "scenes" / SCENE / "common_sim3.json"
    camera_manifest = RGB_CAMERA_ROOT / "RGB_EVALUATION_CAMERA_ROOT_MANIFEST.json"
    for path in (
        contract,
        artifact_schema,
        split_path,
        gcp_csv,
        sim3,
        LIDAR_PAYLOAD_SHA256_INVENTORY,
        camera_manifest,
        FORMAL_ROOT / "NATIVE_QUARTER_INPUT_MANIFEST.json",
        FORMAL_ROOT / "test/sparse/0/cameras.bin",
        FORMAL_ROOT / "test/sparse/0/images.bin",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    split = read_json(split_path)
    names = heldout_names(split)
    write_allowlist_exclusive(allowlist, names)

    common_env = {
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONUNBUFFERED": "1",
    }
    jobs: list[dict[str, Any]] = []
    for method in registry["methods"]:
        method_id = str(method["method_id"])
        run_root = Path(str(method["run_root"])).resolve()
        evaluation_root = GEOMETRY_EVALUATION_ROOTS[method_id].resolve()
        packet_python = GEOMETRY_PACKET_PYTHONS.get(
            method_id, Path(str(method["environment"])) / "bin/python"
        )
        if not (evaluation_root / ".git").exists() or not packet_python.is_file():
            raise FileNotFoundError(
                f"geometry evaluation runtime missing: {method_id}"
            )
        packet_root = lidar_heldout_candidate_packet_root(run_root)
        output_root = (
            run_root
            / "formal_evaluation/lidar_geometry_100k_heldout_candidate_v1"
        )
        log_root = runtime / "geometry_logs_heldout_candidate_v1" / method_id
        packet_argv = packet_command(
            repo=repo,
            method=method,
            profile="lidar_heldout_candidate",
            camera_root=RGB_CAMERA_ROOT,
            allowlist=allowlist,
            packet_root=packet_root,
            evaluation_root=evaluation_root,
            packet_python=packet_python,
        )
        evaluate_argv = [
            f"{LIDAR_ENV}/bin/python",
            "-B",
            str(repo / "code/gcp/evaluate_m3m_gcp_lidar_success_v1.py"),
            "--repo",
            str(repo),
            "--benchmark-commit",
            args.benchmark_commit,
            "--benchmark-tree",
            args.benchmark_tree,
            "--registry",
            str(registry_path),
            "--contract",
            str(contract),
            "--artifact-schema",
            str(artifact_schema),
            "--split",
            str(split_path),
            "--geometry-release-root",
            str(PROTOCOL_ROOT),
            "--formal-input-root",
            str(FORMAL_ROOT),
            "--lidar-inventory",
            str(LIDAR_PAYLOAD_SHA256_INVENTORY),
            "--lidar-root",
            str(LIDAR_ROOT),
            "--colmap-model",
            str(FORMAL_ROOT / "test/sparse/0"),
            "--gcp-csv",
            str(gcp_csv),
            "--sim3-json",
            str(sim3),
            "--packet-manifest",
            str(packet_root / "depth_export_manifest.json"),
            "--reference-cache-root",
            str(runtime / "lidar_reference_cache_v1"),
            "--output-root",
            str(output_root),
            "--method-id",
            method_id,
            "--surface-sampling-track",
            "heldout_candidate",
        ]
        jobs.append(
            {
                "method_id": method_id,
                "run_root": str(run_root),
                "lidar": {
                    "packet_root": str(packet_root),
                    "output_root": str(output_root),
                    "packet": phase(
                        packet_argv,
                        working_directory=evaluation_root,
                        env=environment(method),
                        log_root=log_root / "lidar_packet",
                        nofile_soft_limit=65535,
                    ),
                    "evaluate": phase(
                        evaluate_argv,
                        working_directory=repo,
                        env=common_env,
                        log_root=log_root / "lidar_evaluate",
                    ),
                },
            }
        )

    payload = {
        "schema": "m3m_gcp_100k_success_geometry_execution_plan_v1",
        "status": "READY",
        "scene": SCENE,
        "surface_sampling_track": "heldout_candidate",
        "candidate_protocol_id": (
            "m3m_gcp_lidar_heldout_visible_surface_candidate_v1"
        ),
        "benchmark_repository": benchmark,
        "registry": {
            "path": str(registry_path),
            "sha256": sha256_file(registry_path),
        },
        "heldout_allowlist": {
            "path": str(allowlist),
            "sha256": sha256_file(allowlist),
            "view_count": len(names),
            "source_split_sha256": sha256_file(split_path),
        },
        "heldout_camera_root_manifest": {
            "path": str(camera_manifest),
            "sha256": sha256_file(camera_manifest),
        },
        "method_order": METHODS,
        "track_order": ["lidar"],
        "job_count": len(jobs),
        "execution_semantics": {
            "candidate_validation_not_formal_global_track": True,
            "heldout_rgb_pixels_used_by_depth_renderer": False,
            "heldout_rgb_decode_avoided_when_geometry_camera_only_supported": True,
            "one_packet_set_at_a_time": True,
            "packet_npz_deleted_only_after_evaluator_terminal_receipt": True,
            "metric_based_retry_or_view_selection": False,
            "common_sim3_roi_voxel_thresholds_unchanged": True,
            "geometry_camera_only_loader_scope": [
                "3dgs_original",
                "rade_gs",
            ],
        },
        "jobs": jobs,
    }
    payload["canonical_sha256"] = canonical_sha256(payload)
    write_exclusive(output, payload)
    print(
        json.dumps(
            {
                "status": "PASS_100K_HELDOUT_LIDAR_CANDIDATE_PLAN",
                "path": str(output),
                "sha256": sha256_file(output),
                "allowlist_sha256": sha256_file(allowlist),
                "job_count": len(jobs),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
