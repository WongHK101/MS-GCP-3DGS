#!/usr/bin/env python3
"""Build sequential GCP and LiDAR commands for promoted 100K successes."""

from __future__ import annotations

import argparse
import json
import os
import shlex
from pathlib import Path
from typing import Any

from m3m_gcp_lidar_artifacts import canonical_sha256, command_sha256, sha256_file
from m3m_gcp_100k_geometry_paths import lidar_full_train_packet_root
from rgb_quality_contract import validate_benchmark_checkout


SCENE = "gcp_100000_20260610"
METHODS = [
    "3dgs_original",
    "pgsr",
    "rade_gs",
    "citygs_x",
    "metrogs",
    "gsprior",
]
DATA_ROOT = Path("/root/autodl-tmp/datasets/M3M-GCP-colmap-native-quarter-v1")
PROTOCOL_DATA_ROOT = Path(
    "/root/autodl-tmp/datasets/M3M-GCP-native-quarter-preflight-data-v1"
)
FORMAL_ROOT = DATA_ROOT / "formal_inputs" / SCENE
TRAIN_ROOT = FORMAL_ROOT / "train"
GCP_CAMERA_ROOT = Path(
    f"/root/autodl-tmp/datasets/M3M-GCP-100K-gcp-evaluation-camera-root-v1/{SCENE}"
)
LIDAR_CAMERA_ROOT = Path(
    f"/root/autodl-tmp/datasets/M3M-GCP-100K-evaluation-camera-root-v1/{SCENE}"
)
GSPRIOR_ROOT = Path(
    f"/root/autodl-tmp/datasets/M3M-GCP-gsprior-normalized-v1/{SCENE}"
)
PROTOCOL_ROOT = Path(
    "/root/autodl-tmp/datasets/M3M-GCP-native-quarter-benchmark-protocol-v2"
)
LIDAR_ROOT = Path("/root/autodl-tmp/datasets/M3M-GCP-LiDAR-reference-v1")
LIDAR_PAYLOAD_SHA256_INVENTORY = (
    LIDAR_ROOT / "evaluation/evidence/source_payload_sha256_901.csv"
)
LIDAR_ENV = Path("/root/autodl-tmp/envs/m3m-gcp-lidar-eval")
GEOMETRY_EVALUATION_ROOTS = {
    "3dgs_original": Path(
        "/root/autodl-tmp/worktrees/m3m-gcp-native-quarter/3dgs-original/"
        "2eee0e26d2d5fd00ec462df47752223952f6bf4e/eval-adapter-v1"
    ),
    "pgsr": Path(
        "/root/autodl-tmp/worktrees/m3m-gcp-native-quarter/pgsr/"
        "de24f1a38b350387e8d8fe381b2cd70c1ae946e7/eval-adapter-v1"
    ),
    "rade_gs": Path(
        "/root/autodl-tmp/worktrees/m3m-gcp-native-quarter/rade_gs/"
        "d72f20792005ae1d6555a82aa2d15345f247604e/eval-adapter-v1"
    ),
    "citygs_x": Path(
        "/root/autodl-tmp/worktrees/m3m-gcp-native-quarter/citygs_x/"
        "27617f2486505e3b6fe75345edf7c2b11161bc2a/eval-adapter-v1"
    ),
    "metrogs": Path(
        "/root/autodl-tmp/worktrees/m3m-gcp-native-quarter/metrogs/"
        "8cf9ac13c0c34b65c1a935d181c4634909e60f3f/eval-adapter-v1"
    ),
    "gsprior": Path(
        "/root/autodl-tmp/worktrees/m3m-gcp-native-quarter/gsprior/"
        "dcb7c89fb6b60f068b440de45d064ecc7fbcba55/eval-adapter-v1"
    ),
}
GEOMETRY_PACKET_PYTHONS = {
    "citygs_x": Path(
        "/root/autodl-tmp/envs/m3m-gcp-native-quarter/citygs_x/"
        "eval-py310-torch2.7.1-cu128-v1/bin/python"
    )
}

# Some geometry adapters deliberately keep their patched CUDA extension outside
# the training environment.  Bind those conformance-tested packages explicitly
# so packet export cannot silently fall back to the method's unpatched training
# rasterizer.
GEOMETRY_ADAPTER_PYTHONPATHS = {
    "3dgs_original": [
        GEOMETRY_EVALUATION_ROOTS["3dgs_original"]
        / "submodules/diff-gaussian-rasterization"
    ],
    "pgsr": [
        Path(
            "/root/autodl-tmp/build/m3m-gcp-native-quarter/pgsr/"
            "de24f1a38b350387e8d8fe381b2cd70c1ae946e7/qualification-v1/eval-site"
        )
    ],
    "rade_gs": [
        Path(
            "/root/autodl-tmp/build/m3m-gcp-native-quarter/rade_gs/"
            "d72f20792005ae1d6555a82aa2d15345f247604e/qualification-v1/eval-site"
        )
    ],
    "metrogs": [
        Path(
            "/root/autodl-tmp/build/m3m-gcp-native-quarter/metrogs/"
            "8cf9ac13c0c34b65c1a935d181c4634909e60f3f/"
            "eval-extension-v1/python"
        )
    ],
}


def read_json(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(
            descriptor,
            (
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n"
            ).encode("utf-8"),
        )
    finally:
        os.close(descriptor)


def environment(method: dict[str, Any]) -> dict[str, str]:
    output = {
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONUNBUFFERED": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    output.update(
        {
            str(key): str(value)
            for key, value in method.get("environment_variables", {}).items()
        }
    )
    pythonpath = [
        *GEOMETRY_ADAPTER_PYTHONPATHS.get(str(method["method_id"]), []),
        *method.get("pythonpath", []),
    ]
    if pythonpath:
        output["PYTHONPATH"] = ":".join(str(path) for path in pythonpath)
    return output


def phase(
    argv: list[str],
    *,
    working_directory: Path,
    env: dict[str, str],
    log_root: Path,
    nofile_soft_limit: int | None = None,
) -> dict[str, Any]:
    output = {
        "argv": argv,
        "argv_sha256": command_sha256(argv),
        "shell_preview": shlex.join(argv),
        "working_directory": str(working_directory.resolve()),
        "environment": env,
        "stdout": str((log_root / "stdout.log").resolve()),
        "stderr": str((log_root / "stderr.log").resolve()),
    }
    if nofile_soft_limit is not None:
        output["resource_limits"] = {"nofile_soft": int(nofile_soft_limit)}
    return output


def packet_command(
    *,
    repo: Path,
    method: dict[str, Any],
    profile: str,
    camera_root: Path,
    allowlist: Path,
    packet_root: Path,
    evaluation_root: Path,
    packet_python: Path,
) -> list[str]:
    method_id = str(method["method_id"])
    normalized_role = "gcp_evaluation" if profile == "gcp" else "lidar_evaluation"
    dataset_root = (
        GSPRIOR_ROOT / normalized_role if method_id == "gsprior" else TRAIN_ROOT
    )
    return [
        str(packet_python),
        "-B",
        str(repo / "code/gcp/run_m3m_gcp_100k_packet_export.py"),
        "--method-id",
        method_id,
        "--camera-profile",
        profile,
        "--benchmark-repo",
        str(repo),
        "--evaluation-repo",
        str(evaluation_root),
        "--training-run-root",
        str(method["run_root"]),
        "--dataset-root",
        str(dataset_root),
        "--prior-root",
        str(dataset_root),
        "--camera-root",
        str(camera_root),
        "--train-allowlist",
        str(allowlist),
        "--packet-set-root",
        str(packet_root),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-repo", type=Path, required=True)
    parser.add_argument("--benchmark-commit", required=True)
    parser.add_argument("--benchmark-tree", required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.benchmark_repo.expanduser().resolve()
    runtime = args.runtime_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
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

    gcp_allowlist = runtime / "gcp_211_view_allowlist.csv"
    lidar_allowlist = (
        repo
        / "configs/m3m_gcp_lidar_train_view_allowlists_v1"
        / f"{SCENE}.csv"
    )
    contract = repo / "configs/m3m_gcp_lidar_formal_v1.json"
    artifact_schema = repo / "configs/m3m_gcp_lidar_formal_artifact_schema_v1.json"
    split = repo / "configs/gs_gcp_rgb_holdout_split_manifest_v1.json"
    gcp_csv = (
        PROTOCOL_DATA_ROOT
        / "benchmark/source_release_v1_3_0/gcp_points_cgcs2000_cm108_v1_3_0.csv"
    )
    sim3 = PROTOCOL_ROOT / "scenes" / SCENE / "common_sim3.json"
    lidar_inventory = LIDAR_PAYLOAD_SHA256_INVENTORY
    for path in (
        gcp_allowlist,
        lidar_allowlist,
        contract,
        artifact_schema,
        split,
        gcp_csv,
        sim3,
        lidar_inventory,
        PROTOCOL_ROOT / "protocol_release_manifest.json",
        FORMAL_ROOT / "NATIVE_QUARTER_INPUT_MANIFEST.json",
        GSPRIOR_ROOT / "gcp_evaluation/normalization_manifest.json",
        GSPRIOR_ROOT / "lidar_evaluation/normalization_manifest.json",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

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
        method_env = environment(method)
        # v1 used the retired training-camera binding and v2 was consumed by
        # the failed adapter-runtime preflight.  Preserve both receipts and use
        # a fresh packet namespace for the corrected formal execution.
        gcp_packet_root = (
            run_root / "formal_evaluation/gcp_packets_100k_success_v3"
        )
        # v1 was consumed by the inherited 1024-NOFILE failure and v2 by the
        # classic loader's 110-GiB RGB-decode ceiling.  Preserve both receipts;
        # v3 uses the parity-proven geometry-camera-only loader.
        lidar_packet_root = lidar_full_train_packet_root(run_root)
        gcp_output = run_root / "formal_evaluation/gcp_geometry_100k_success_v1"
        lidar_output = run_root / "formal_evaluation/lidar_geometry_100k_success_v1"
        log_root = runtime / "geometry_logs_camera_only_v1" / method_id
        gcp_packet_argv = packet_command(
            repo=repo,
            method=method,
            profile="gcp",
            camera_root=GCP_CAMERA_ROOT,
            allowlist=gcp_allowlist,
            packet_root=gcp_packet_root,
            evaluation_root=evaluation_root,
            packet_python=packet_python,
        )
        lidar_packet_argv = packet_command(
            repo=repo,
            method=method,
            profile="lidar",
            camera_root=LIDAR_CAMERA_ROOT,
            allowlist=lidar_allowlist,
            packet_root=lidar_packet_root,
            evaluation_root=evaluation_root,
            packet_python=packet_python,
        )
        gcp_eval_argv = [
            f"{LIDAR_ENV}/bin/python",
            "-B",
            str(repo / "code/gcp/evaluate_m3m_native_quarter_geometry.py"),
            "--data_root",
            str(PROTOCOL_DATA_ROOT),
            "--protocol_release",
            str(PROTOCOL_ROOT),
            "--scene",
            SCENE,
            "--method_id",
            method_id,
            "--metric_packet_manifest",
            str(gcp_packet_root / "depth_export_manifest.json"),
            "--out_dir",
            str(gcp_output),
        ]
        lidar_eval_argv = [
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
            str(split),
            "--geometry-release-root",
            str(PROTOCOL_ROOT),
            "--formal-input-root",
            str(FORMAL_ROOT),
            "--lidar-inventory",
            str(lidar_inventory),
            "--lidar-root",
            str(LIDAR_ROOT),
            "--colmap-model",
            str(TRAIN_ROOT / "sparse/0"),
            "--gcp-csv",
            str(gcp_csv),
            "--sim3-json",
            str(sim3),
            "--packet-manifest",
            str(lidar_packet_root / "depth_export_manifest.json"),
            "--reference-cache-root",
            str(runtime / "lidar_reference_cache_v1"),
            "--output-root",
            str(lidar_output),
            "--method-id",
            method_id,
        ]
        jobs.append(
            {
                "method_id": method_id,
                "run_root": str(run_root),
                "gcp": {
                    "packet_root": str(gcp_packet_root),
                    "output_root": str(gcp_output),
                    "packet": phase(
                        gcp_packet_argv,
                        working_directory=evaluation_root,
                        env=method_env,
                        log_root=log_root / "gcp_packet",
                    ),
                    "evaluate": phase(
                        gcp_eval_argv,
                        working_directory=repo,
                        env=common_env,
                        log_root=log_root / "gcp_evaluate",
                    ),
                },
                "lidar": {
                    "packet_root": str(lidar_packet_root),
                    "output_root": str(lidar_output),
                    "packet": phase(
                        lidar_packet_argv,
                        working_directory=evaluation_root,
                        env=method_env,
                        log_root=log_root / "lidar_packet",
                        nofile_soft_limit=65535,
                    ),
                    "evaluate": phase(
                        lidar_eval_argv,
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
        "benchmark_repository": benchmark,
        "registry": {
            "path": str(registry_path),
            "sha256": sha256_file(registry_path),
        },
        "method_order": METHODS,
        "track_order": ["gcp", "lidar"],
        "job_count": len(jobs),
        "execution_semantics": {
            "one_packet_set_at_a_time": True,
            "packet_npz_deleted_only_after_evaluator_terminal_receipt": True,
            "continue_after_method_failure": True,
            "metric_based_retry_or_selection": False,
            "legacy_activation_v4_used": False,
            "lidar_numeric_core": "evaluate_m3m_gcp_lidar_formal_v1.py unchanged functions",
            "geometry_camera_only_loader_scope": ["3dgs_original", "rade_gs"],
            "geometry_camera_only_parity_required": True,
        },
        "jobs": jobs,
    }
    payload["canonical_sha256"] = canonical_sha256(payload)
    write_exclusive(output, payload)
    print(
        json.dumps(
            {
                "status": "PASS_100K_SUCCESS_GEOMETRY_PLAN",
                "path": str(output),
                "sha256": sha256_file(output),
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
