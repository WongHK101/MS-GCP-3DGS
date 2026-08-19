#!/usr/bin/env python3
"""Build, but do not execute, the exact 3K RGB render and shared-metric commands."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path, PurePosixPath
from typing import Any


PLAN_SCHEMA = "m3m_gcp_native_quarter_rgb_quality_3k_execution_plan_v1"


def _resolve_camera_root(method: dict[str, Any], shared: dict[str, Any]) -> str:
    value = str(method["camera_root"])
    shared_roots = {
        "shared.default_camera_root": "default_camera_root",
        "shared.graphdeco_camera_root": "graphdeco_camera_root",
    }
    if value in shared_roots:
        return str(shared[shared_roots[value]])
    if not PurePosixPath(value).is_absolute():
        raise ValueError(f"invalid camera root for {method['method_id']}: {value}")
    return value


def _environment(method: dict[str, Any]) -> dict[str, str]:
    environment = {
        str(key): str(value)
        for key, value in method.get("environment_variables", {}).items()
    }
    pythonpath = method.get("pythonpath", [])
    if pythonpath:
        environment["PYTHONPATH"] = ":".join(str(value) for value in pythonpath)
    return environment


def _render_argv(
    *,
    method: dict[str, Any],
    shared: dict[str, Any],
    benchmark_repo: str,
    contract_path: str,
    artifact_root: str,
) -> list[str]:
    method_id = str(method["method_id"])
    python = f"{method['environment']}/bin/python"
    adapter = f"{benchmark_repo}/code/gcp/{method['adapter']}"
    common = [
        "--rgb_contract",
        contract_path,
        "--input_manifest",
        str(shared["input_manifest"]),
        "--scene",
        str(method.get("scene", "gcp_3000_20260602")),
        "--output_dir",
        f"{artifact_root}/renders",
        "--manifest_path",
        f"{artifact_root}/rgb_render_manifest.json",
        "--appearance_policy",
        str(method["appearance_policy"]),
    ]
    camera_root = _resolve_camera_root(method, shared)
    if method["adapter"] == "export_gaussian_rgb.py":
        argv = [
            python,
            "-B",
            adapter,
            "--train_repo",
            str(method["source_root"]),
            "--iteration",
            str(method["iteration"]),
            "--camera_sets",
            "all",
            *common,
            "--method_id",
            method_id,
            "-s",
            camera_root,
            "-m",
            str(method["model_root"]),
            "--images",
            "images",
            "-r",
            "1",
        ]
    elif method["adapter"] == "export_qgs_rgb.py":
        argv = [
            python,
            "-B",
            adapter,
            "--train_repo",
            str(method["source_root"]),
            "--qgs_config_path",
            str(method["config_path"]),
            "--model_path",
            str(method["model_root"]),
            "--camera_source_path",
            camera_root,
            "--iteration",
            str(method["iteration"]),
            "--camera_sets",
            "all",
            *common,
            "--method_id",
            method_id,
        ]
    elif method["adapter"] == "export_lightning_gaussian_rgb.py":
        argv = [
            python,
            "-B",
            adapter,
            "--repo",
            str(method["source_root"]),
            "--checkpoint",
            str(method["formal_checkpoint"]),
            "--camera_root",
            camera_root,
            *common,
            "--method_id",
            method_id,
        ]
        if method_id == "metrogs":
            argv.extend(["--training_cameras_json", str(method["training_cameras_json"])])
    elif method["adapter"] == "export_citygs_x_rgb.py":
        argv = [
            python,
            "-B",
            adapter,
            "--repo",
            str(method["source_root"]),
            "--model_path",
            str(method["model_root"]),
            "--pytorch3d_compat",
            str(method["pytorch3d_compat"]),
            "--camera_root",
            camera_root,
            "--iteration",
            str(method["iteration"]),
            *common,
        ]
    else:
        raise ValueError(f"unsupported RGB adapter: {method['adapter']}")
    argv.extend(str(value) for value in method.get("extra_cli", []))
    if method.get("splatting_config_path"):
        argv.extend(["--splatting_config_path", str(method["splatting_config_path"])])
    return argv


def build_plan(
    registry: dict[str, Any],
    *,
    benchmark_repo: str,
    allow_review_candidate: bool = False,
) -> dict[str, Any]:
    status = registry.get("status")
    if status != "ACTIVE_FROZEN" and not (
        allow_review_candidate and status == "REVIEW_CANDIDATE_NOT_FORMAL"
    ):
        raise ValueError(f"registry is not executable: {status}")
    benchmark_path = PurePosixPath(benchmark_repo)
    if not benchmark_path.is_absolute():
        raise ValueError("benchmark repository path must be absolute")
    if "DEPLOYED_REVIEWED_COMMIT" in benchmark_repo:
        raise ValueError("benchmark repository path still contains the review placeholder")

    shared = registry["shared"]
    contract_path = str(benchmark_path / str(shared["contract_relative_path"]))
    evaluator = str(benchmark_path / "code/gcp/evaluate_m3m_native_quarter_rgb_quality.py")
    jobs: list[dict[str, Any]] = []
    for method in registry["methods"]:
        artifact_root = (
            f"{method['run_root']}/formal_evaluation/{shared['output_relative_path']}"
        )
        render_argv = _render_argv(
            method=method,
            shared=shared,
            benchmark_repo=str(benchmark_path),
            contract_path=contract_path,
            artifact_root=artifact_root,
        )
        metric_argv = [
            f"{shared['metric_environment']}/bin/python",
            "-B",
            evaluator,
            "--rgb_contract",
            contract_path,
            "--input_manifest",
            str(shared["input_manifest"]),
            "--input_root",
            str(shared["input_root"]),
            "--render_manifest",
            f"{artifact_root}/rgb_render_manifest.json",
            "--scene",
            str(registry["scene"]),
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
            f"{artifact_root}/metrics",
        ]
        jobs.append(
            {
                "method_id": method["method_id"],
                "artifact_root": artifact_root,
                "precondition_absent_paths": [
                    f"{artifact_root}/renders",
                    f"{artifact_root}/rgb_render_manifest.json",
                    f"{artifact_root}/metrics",
                ],
                "render": {
                    "working_directory": str(method["source_root"]),
                    "environment": _environment(method),
                    "argv": render_argv,
                    "shell_preview": shlex.join(render_argv),
                    "stdout": f"{artifact_root}/render.stdout.log",
                    "stderr": f"{artifact_root}/render.stderr.log",
                },
                "metric": {
                    "working_directory": str(shared["metric_reference_root"]),
                    "environment": {},
                    "argv": metric_argv,
                    "shell_preview": shlex.join(metric_argv),
                    "stdout": f"{artifact_root}/metric.stdout.log",
                    "stderr": f"{artifact_root}/metric.stderr.log",
                },
            }
        )
    return {
        "schema": PLAN_SCHEMA,
        "suite_id": registry["suite_id"],
        "registry_status": status,
        "formal_execution_authorized": status == "ACTIVE_FROZEN",
        "review_candidate_plan_only": status != "ACTIVE_FROZEN",
        "server": registry["server"],
        "scene": registry["scene"],
        "benchmark_repo": str(benchmark_path),
        "contract_path": contract_path,
        "job_count": len(jobs),
        "method_order": [job["method_id"] for job in jobs],
        "execution_semantics": {
            "order": "render adapter, then the one shared evaluator, per method",
            "concurrency": 1,
            "continue_after_method_failure": True,
            "no_overwrite": True,
            "formal_geometry_completion_required": True,
        },
        "jobs": jobs,
    }


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=repo_root / "configs" / "m3m_gcp_native_quarter_rgb_quality_3k_registry_v1.json",
    )
    parser.add_argument("--benchmark-repo", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-review-candidate",
        action="store_true",
        help="Build a non-executable audit preview while the suite is not ACTIVE_FROZEN.",
    )
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    plan = build_plan(
        registry,
        benchmark_repo=args.benchmark_repo,
        allow_review_candidate=args.allow_review_candidate,
    )
    rendered = json.dumps(plan, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
