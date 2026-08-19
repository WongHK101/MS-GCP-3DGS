#!/usr/bin/env python3
"""Read-only AutoDL-901 preflight for the native-quarter 3K RGB suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rgb_quality_contract import (  # noqa: E402
    directory_content_identity,
    sparse_model_sha256,
    validate_benchmark_checkout,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo: Path, *args: str, binary: bool = False) -> str | bytes:
    value = subprocess.check_output(
        ["git", "-C", str(repo), *args],
        text=not binary,
        stderr=subprocess.DEVNULL,
    )
    return value if binary else value.strip()


def _generated_cache_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    return (
        normalized.endswith(".pyc")
        or "/__pycache__/" in f"/{normalized}"
        or normalized.endswith("/__pycache__")
    )


def preflight(
    contract_path: Path,
    registry_path: Path,
    *,
    allow_metro_pending: bool,
    benchmark_repo: Path,
    benchmark_commit: str,
    benchmark_tree: str,
) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    pending: list[str] = []
    checks: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {}

    def record(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})
        if not passed:
            errors.append(f"{name}: {detail}")

    shared = registry["shared"]
    benchmark_identity = validate_benchmark_checkout(
        benchmark_repo=benchmark_repo,
        expected_commit=benchmark_commit,
        expected_tree=benchmark_tree,
        entrypoint=Path(__file__).resolve(),
    )
    input_manifest_path = Path(shared["input_manifest"])
    expected_binding = contract["input_binding"]["scene_bindings"][registry["scene"]]
    record("input_manifest_exists", input_manifest_path.is_file(), str(input_manifest_path))
    if input_manifest_path.is_file():
        actual = sha256_file(input_manifest_path)
        record(
            "input_manifest_file_sha256",
            actual == expected_binding["formal_input_manifest_file_sha256"],
            actual,
        )
        manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
        record(
            "input_manifest_canonical_sha256",
            manifest.get("manifest_sha256")
            == expected_binding["formal_input_manifest_canonical_sha256"],
            manifest.get("manifest_sha256"),
        )
        test_rows = [row for row in manifest.get("images", []) if row.get("role") == "test"]
        record("heldout_view_count", len(test_rows) == 12, len(test_rows))
        input_root = Path(shared["input_root"])
        bad_truth: list[str] = []
        for row in test_rows:
            path = input_root / str(row["relative_path"])
            if not path.is_file() or sha256_file(path) != row["jpeg_sha256"]:
                bad_truth.append(str(row["image_name"]))
        record("heldout_jpeg_identity", not bad_truth, bad_truth)

        camera_root = Path(shared["default_camera_root"])
        graphdeco_root = Path(shared["graphdeco_camera_root"])
        combined_names = sorted(path.name for path in (graphdeco_root / "images").iterdir() if path.is_file())
        expected_names = sorted(str(row["image_name"]) for row in manifest["images"])
        record("graphdeco_camera_image_inventory", combined_names == expected_names, len(combined_names))
        rows_by_name = {str(row["image_name"]): row for row in manifest["images"]}
        bad_combined_images = [
            name
            for name in combined_names
            if name not in rows_by_name
            or sha256_file(graphdeco_root / "images" / name) != rows_by_name[name]["jpeg_sha256"]
        ]
        record("graphdeco_camera_image_identity", not bad_combined_images, bad_combined_images)
        for root_name, root in (("camera_model", camera_root), ("graphdeco_camera", graphdeco_root)):
            expected_models = manifest.get("source_model_sha256", {})
            filenames = ("cameras.bin", "images.bin", "points3D.bin")
            if root_name == "graphdeco_camera":
                filenames = (*filenames, "points3D.ply")
            for filename in filenames:
                expected_sha = expected_models[filename]
                path = root / "sparse" / "0" / filename
                actual_sha = sha256_file(path) if path.is_file() else "MISSING"
                record(f"{root_name}_{filename}_sha256", actual_sha == expected_sha, actual_sha)

    reference_root = Path(shared["metric_reference_root"])
    for relative, expected_sha in contract["metric_reference"]["files_sha256"].items():
        path = reference_root / relative
        actual_sha = sha256_file(path) if path.is_file() else "MISSING"
        record(f"metric_source:{relative}", actual_sha == expected_sha, actual_sha)
    weights = {
        "vgg16-397923af.pth": Path(shared["vgg16_weights"]),
        "vgg.pth": Path(shared["lpips_vgg_weights"]),
    }
    for name, path in weights.items():
        actual_sha = sha256_file(path) if path.is_file() else "MISSING"
        record(
            f"metric_weight:{name}",
            actual_sha == contract["metric_reference"]["weights_sha256"][name],
            actual_sha,
        )
    metric_python = Path(shared["metric_environment"]) / "bin" / "python"
    if metric_python.is_file():
        runtime_script = (
            "import importlib.metadata,json,platform,torch,torchvision;"
            "print(json.dumps({'python':platform.python_version(),'torch':str(torch.__version__),"
            "'torchvision':str(torchvision.__version__),'Pillow':importlib.metadata.version('Pillow'),"
            "'numpy':importlib.metadata.version('numpy')}))"
        )
        actual_runtime = json.loads(
            subprocess.check_output([str(metric_python), "-c", runtime_script], text=True)
        )
        record(
            "metric_runtime",
            actual_runtime == contract["metric_reference"]["runtime"],
            actual_runtime,
        )
    else:
        record("metric_runtime", False, f"missing Python: {metric_python}")

    for method in registry["methods"]:
        method_id = str(method["method_id"])
        source = Path(method["source_root"])
        record(f"{method_id}:source_exists", source.is_dir(), str(source))
        if source.is_dir():
            commit = str(_git(source, "rev-parse", "HEAD"))
            record(f"{method_id}:source_commit", commit == method["source_commit"], commit)
            # Compare the complete tracked worktree against HEAD so an
            # accidentally staged patch cannot evade the frozen diff identity.
            diff = _git(source, "diff", "HEAD", "--binary", "--no-ext-diff", binary=True)
            assert isinstance(diff, bytes)
            diff_sha = hashlib.sha256(diff).hexdigest()
            expected_worktree = method.get("source_worktree")
            expected_diff = (
                expected_worktree["expected_tracked_diff_sha256"]
                if expected_worktree
                else hashlib.sha256(b"").hexdigest()
            )
            record(f"{method_id}:tracked_diff", diff_sha == expected_diff, diff_sha)
            if expected_worktree:
                for relative, expected_sha in expected_worktree[
                    "expected_tracked_files_sha256"
                ].items():
                    path = source / relative
                    actual_sha = sha256_file(path) if path.is_file() else "MISSING"
                    record(f"{method_id}:patched_file:{relative}", actual_sha == expected_sha, actual_sha)
            status = str(_git(source, "status", "--porcelain=v1", "--untracked-files=all"))
            unexpected_untracked = []
            for line in status.splitlines():
                if line.startswith("?? ") and not _generated_cache_path(line[3:]):
                    unexpected_untracked.append(line[3:])
            record(f"{method_id}:untracked_policy", not unexpected_untracked, unexpected_untracked)

        python = Path(method["environment"]) / "bin" / "python"
        record(f"{method_id}:environment_python", python.is_file(), str(python))
        run_root = Path(method["run_root"])
        geometry_summary = run_root / "formal_evaluation" / "evaluator" / "evaluation_summary.json"
        geometry_complete = False
        geometry_detail: Any = str(geometry_summary)
        if geometry_summary.is_file():
            geometry_payload = json.loads(geometry_summary.read_text(encoding="utf-8"))
            geometry_complete = geometry_payload.get("status") == "COMPLETE_RANKED"
            geometry_detail = {
                "path": str(geometry_summary),
                "status": geometry_payload.get("status"),
            }
        if not geometry_complete:
            if method_id == "metrogs" and allow_metro_pending:
                pending.append(f"{method_id}:geometry_formal_completion")
            else:
                record(f"{method_id}:geometry_formal_completion", False, geometry_detail)
        else:
            record(f"{method_id}:geometry_formal_completion", True, geometry_detail)

        if "formal_checkpoint" in method:
            model_path = Path(method["formal_checkpoint"])
        else:
            model_path = Path(method["model_root"]) / str(method["formal_model_relative_path"])
        if not model_path.is_file():
            if method_id == "metrogs" and allow_metro_pending:
                pending.append(f"{method_id}:formal_model")
            else:
                record(f"{method_id}:formal_model", False, str(model_path))
        else:
            actual_model_sha = sha256_file(model_path)
            model_detail = {
                "path": str(model_path),
                "bytes": model_path.stat().st_size,
                "sha256": actual_model_sha,
            }
            if (
                method_id == "metrogs"
                and allow_metro_pending
                and method.get("formal_model_sha256") == "PENDING_METRO_FORMAL_COMPLETION"
            ):
                record(f"{method_id}:formal_model_exists", True, model_detail)
                pending.append("metrogs:formal_model_sha256_activation")
            else:
                record(
                    f"{method_id}:formal_model",
                    actual_model_sha == method.get("formal_model_sha256"),
                    model_detail,
                )

        if "cfg_args_sha256" in method:
            cfg_args = Path(method["model_root"]) / "cfg_args"
            actual_sha = sha256_file(cfg_args) if cfg_args.is_file() else "MISSING"
            record(
                f"{method_id}:cfg_args_sha256",
                actual_sha == method["cfg_args_sha256"],
                actual_sha,
            )

        if "formal_model_aux_sha256" in method:
            checkpoint_dir = model_path.parent
            for name, expected_sha in method["formal_model_aux_sha256"].items():
                path = checkpoint_dir / name
                actual_sha = sha256_file(path) if path.is_file() else "MISSING"
                record(
                    f"{method_id}:formal_model_aux:{name}",
                    actual_sha == expected_sha,
                    actual_sha,
                )

        for optional_key in ("config_path", "training_cameras_json", "pytorch3d_compat", "splatting_config_path"):
            if optional_key not in method:
                continue
            path = Path(method[optional_key])
            exists = path.is_dir() if optional_key == "pytorch3d_compat" else path.is_file()
            record(f"{method_id}:{optional_key}", exists, str(path))
            if exists and optional_key == "splatting_config_path":
                actual_sha = sha256_file(path)
                record(
                    f"{method_id}:splatting_config_sha256",
                    actual_sha == method["splatting_config_sha256"],
                    actual_sha,
                )
            if exists and optional_key == "config_path" and "config_sha256" in method:
                actual_sha = sha256_file(path)
                record(
                    f"{method_id}:config_sha256",
                    actual_sha == method["config_sha256"],
                    actual_sha,
                )
            if exists and optional_key == "training_cameras_json":
                actual_sha = sha256_file(path)
                record(
                    f"{method_id}:training_cameras_json_sha256",
                    actual_sha == method.get("training_cameras_json_sha256"),
                    actual_sha,
                )
                camera_rows = json.loads(path.read_text(encoding="utf-8"))
                train_names = {
                    str(row["image_name"])
                    for row in manifest.get("images", [])
                    if row.get("role") == "train"
                }
                camera_names = [Path(str(row["img_name"])).name for row in camera_rows]
                # MetroGS renderer.setup() assigns camera.idx as appearance ID;
                # cameras.json `id` is the frozen dataset index.  The serialized
                # appearance_id field is the pre-setup COLMAP camera group.
                appearance_ids = [int(row["id"]) for row in camera_rows]
                normalized_ids = [float(row["normalized_appearance_id"]) for row in camera_rows]
                record(
                    f"{method_id}:training_camera_names",
                    len(camera_names) == 82 and set(camera_names) == train_names,
                    len(camera_names),
                )
                record(
                    f"{method_id}:appearance_ids",
                    sorted(appearance_ids) == list(range(82)),
                    {
                        "count": len(appearance_ids),
                        "min": min(appearance_ids) if appearance_ids else None,
                        "max": max(appearance_ids) if appearance_ids else None,
                    },
                )
                record(
                    f"{method_id}:normalized_appearance_ids",
                    all(0.0 <= value <= 1.0 for value in normalized_ids),
                    {"count": len(normalized_ids)},
                )

        camera_value = str(method["camera_root"])
        if camera_value == "shared.default_camera_root":
            camera_root = Path(shared["default_camera_root"])
            expected_camera_sha = shared["default_camera_sparse_sha256"]
        elif camera_value == "shared.graphdeco_camera_root":
            camera_root = Path(shared["graphdeco_camera_root"])
            expected_camera_sha = shared["graphdeco_camera_sparse_sha256"]
        else:
            camera_root = Path(camera_value)
            expected_camera_sha = method["camera_sparse_sha256"]
        actual_camera_sha = sparse_model_sha256(camera_root)
        record(
            f"{method_id}:camera_sparse_sha256",
            actual_camera_sha == expected_camera_sha,
            actual_camera_sha,
        )

        expected_pythonpath = method.get("pythonpath_content_identity", [])
        actual_pythonpath = [
            directory_content_identity(Path(str(row["path"])))
            for row in expected_pythonpath
        ]
        record(
            f"{method_id}:runtime_pythonpath_identity",
            actual_pythonpath == expected_pythonpath,
            actual_pythonpath,
        )

        artifact_root = run_root / "formal_evaluation" / shared["output_relative_path"]
        record(f"{method_id}:rgb_output_absent", not artifact_root.exists(), str(artifact_root))

    return {
        "schema": "m3m_gcp_native_quarter_rgb_quality_3k_preflight_v1",
        "status": "PASS_STATIC_METRO_PENDING" if not errors and pending else ("PASS_READY" if not errors else "FAIL"),
        "passed": not errors,
        "formal_launch_ready": not errors and not pending and contract.get("status") == "ACTIVE_FROZEN",
        "contract_status": contract.get("status"),
        "scene": registry.get("scene"),
        "method_count": len(registry.get("methods", [])),
        "inputs": {
            "preflight_path": str(Path(__file__).resolve()),
            "preflight_sha256": sha256_file(Path(__file__).resolve()),
            "contract_path": str(contract_path),
            "contract_sha256": sha256_file(contract_path),
            "registry_path": str(registry_path),
            "registry_sha256": sha256_file(registry_path),
            "benchmark_repo": str(benchmark_repo.expanduser().resolve()),
            "benchmark_commit": benchmark_identity["commit"],
            "benchmark_tree": benchmark_identity["tree"],
            "benchmark_clean": (
                benchmark_identity["tracked_diff_sha256"] == hashlib.sha256(b"").hexdigest()
                and benchmark_identity["tracked_modified_files_sha256"] == {}
                and benchmark_identity["unexpected_untracked_files"] == []
            ),
        },
        "pending": pending,
        "errors": errors,
        "checks": checks,
    }


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=repo_root / "configs" / "m3m_gcp_native_quarter_rgb_quality_v1.json",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=repo_root / "configs" / "m3m_gcp_native_quarter_rgb_quality_3k_registry_v1.json",
    )
    parser.add_argument("--allow-metro-pending", action="store_true")
    parser.add_argument("--benchmark-repo", type=Path, required=True)
    parser.add_argument("--benchmark-commit", required=True)
    parser.add_argument("--benchmark-tree", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = preflight(
        args.contract.resolve(),
        args.registry.resolve(),
        allow_metro_pending=args.allow_metro_pending,
        benchmark_repo=args.benchmark_repo.resolve(),
        benchmark_commit=args.benchmark_commit,
        benchmark_tree=args.benchmark_tree,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
