#!/usr/bin/env python3
"""Build common RGB commands for the promoted 100K success subset."""

from __future__ import annotations

import argparse
import json
import os
import shlex
from pathlib import Path
from typing import Any

from build_m3m_native_quarter_rgb_quality_3k_commands import _environment, _render_argv
from m3m_gcp_lidar_artifacts import canonical_sha256, command_sha256, sha256_file
from rgb_quality_contract import validate_benchmark_checkout


SCENE = "gcp_100000_20260610"
EXPECTED_METHODS = [
    "3dgs_original",
    "pgsr",
    "rade_gs",
    "citygs_x",
    "metrogs",
    "gsprior",
]


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
            (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
                "utf-8"
            ),
        )
    finally:
        os.close(descriptor)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-repo", type=Path, required=True)
    parser.add_argument("--benchmark-commit", required=True)
    parser.add_argument("--benchmark-tree", required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo = args.benchmark_repo.expanduser().resolve()
    runtime_root = args.runtime_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    benchmark_identity = validate_benchmark_checkout(
        benchmark_repo=repo,
        expected_commit=args.benchmark_commit,
        expected_tree=args.benchmark_tree,
        entrypoint=Path(__file__).resolve(),
    )
    registry_path = runtime_root / "rgb_success_registry.json"
    contract_path = runtime_root / "rgb_quality_100k_success_contract.json"
    inventory_path = runtime_root / "qualification_outcome_inventory.json"
    promotion_path = runtime_root / "success_subset_promotion_receipt.json"
    registry = read_json(registry_path)
    contract = read_json(contract_path)
    inventory = read_json(inventory_path)
    promotion = read_json(promotion_path)
    if (
        registry.get("schema")
        != "m3m_gcp_native_quarter_rgb_quality_100k_success_registry_v1"
        or registry.get("status") != "ACTIVE_FROZEN"
        or registry.get("scene") != SCENE
        or registry.get("canonical_sha256") != canonical_sha256(registry)
        or registry.get("ready_method_ids") != EXPECTED_METHODS
        or inventory.get("canonical_sha256") != canonical_sha256(inventory)
        or promotion.get("canonical_sha256") != canonical_sha256(promotion)
        or promotion.get("eligible_method_ids") != EXPECTED_METHODS
        or contract.get("status") != "ACTIVE_FROZEN"
        or contract.get("formal_gate", {}).get("legacy_activation_v4_required") is not False
    ):
        raise RuntimeError("100K success runtime identity mismatch")

    shared = registry["shared"]
    evaluator = repo / "code/gcp/evaluate_m3m_native_quarter_rgb_quality.py"
    jobs: list[dict[str, Any]] = []
    for method in registry["methods"]:
        artifact_root = Path(str(method["formal_output_root"])).resolve()
        render_argv = _render_argv(
            method=method,
            shared=shared,
            benchmark_repo=str(repo),
            benchmark_commit=args.benchmark_commit,
            benchmark_tree=args.benchmark_tree,
            contract_path=str(contract_path),
            artifact_root=str(artifact_root),
        )
        metric_argv = [
            f"{shared['metric_environment']}/bin/python",
            "-B",
            str(evaluator),
            "--rgb_contract",
            str(contract_path),
            "--registry",
            str(registry_path),
            "--benchmark_repo",
            str(repo),
            "--benchmark_commit",
            args.benchmark_commit,
            "--benchmark_tree",
            args.benchmark_tree,
            "--input_manifest",
            str(shared["input_manifest"]),
            "--input_root",
            str(shared["input_root"]),
            "--render_manifest",
            str(artifact_root / "rgb_render_manifest.json"),
            "--scene",
            SCENE,
            "--method_id",
            str(method["method_id"]),
            "--metric_reference_root",
            str(shared["metric_reference_root"]),
            "--vgg16_weights",
            str(shared["vgg16_weights"]),
            "--lpips_vgg_weights",
            str(shared["lpips_vgg_weights"]),
            "--device",
            str(shared["metric_device"]),
            "--output_dir",
            str(artifact_root / "metrics"),
        ]
        jobs.append(
            {
                "method_id": method["method_id"],
                "artifact_root": str(artifact_root),
                "render": {
                    "working_directory": str(method["source_root"]),
                    "environment": _environment(method),
                    "argv": render_argv,
                    "argv_sha256": command_sha256(render_argv),
                    "shell_preview": shlex.join(render_argv),
                    "stdout": str(artifact_root / "render.stdout.log"),
                    "stderr": str(artifact_root / "render.stderr.log"),
                },
                "metric": {
                    "working_directory": str(shared["metric_reference_root"]),
                    "environment": {},
                    "argv": metric_argv,
                    "argv_sha256": command_sha256(metric_argv),
                    "shell_preview": shlex.join(metric_argv),
                    "stdout": str(artifact_root / "metric.stdout.log"),
                    "stderr": str(artifact_root / "metric.stderr.log"),
                },
            }
        )
    payload: dict[str, Any] = {
        "schema": "m3m_gcp_native_quarter_rgb_quality_100k_success_execution_plan_v1",
        "status": "READY",
        "scene": SCENE,
        "benchmark_repository": benchmark_identity,
        "outcome_inventory_path": str(inventory_path),
        "outcome_inventory_sha256": sha256_file(inventory_path),
        "promotion_receipt_path": str(promotion_path),
        "promotion_receipt_sha256": sha256_file(promotion_path),
        "contract_path": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "registry_path": str(registry_path),
        "registry_sha256": sha256_file(registry_path),
        "method_order": EXPECTED_METHODS,
        "job_count": len(jobs),
        "execution_semantics": {
            "order": "render then one shared metric evaluator per method",
            "continue_after_method_failure": True,
            "no_metric_based_retry_or_selection": True,
            "legacy_activation_v4_used": False,
            "first_formal_job_is_the_integrated_real_end_to_end_preflight": True,
        },
        "jobs": jobs,
    }
    payload["canonical_sha256"] = canonical_sha256(payload)
    write_exclusive(output, payload)
    print(
        json.dumps(
            {
                "status": "PASS_100K_SUCCESS_RGB_COMMANDS",
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
