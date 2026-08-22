#!/usr/bin/env python3
"""Fail-closed runtime preflight for the activated 100K RGB track."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from evaluate_m3m_native_quarter_rgb_quality import validate_render_and_ground_truth
from m3m_gcp_lidar_artifacts import canonical_sha256, sha256_file
from m3m_gcp_100k_three_track_runtime import validate_addendum_runtime
from rgb_quality_contract import (
    directory_content_identity,
    git_identity,
    sparse_model_sha256,
    validate_benchmark_checkout,
)
from run_m3m_gcp_100k_guarded import validate_model_identity_bundle


SCENE = "gcp_100000_20260610"


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


def same_path(left: Any, right: Any) -> bool:
    try:
        return Path(str(left)).resolve() == Path(str(right)).resolve()
    except Exception:  # noqa: BLE001
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--benchmark-repo", type=Path, required=True)
    parser.add_argument("--technical-smoke-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    activation_path = args.activation.resolve()
    benchmark_repo = args.benchmark_repo.resolve()
    smoke_root = args.technical_smoke_root.resolve()
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)

    errors: list[str] = []
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})
        if not passed:
            errors.append(f"{name}: {detail}")

    activation = require_json(activation_path)
    record(
        "activation_identity",
        activation.get("schema") == "m3m_gcp_100k_three_track_activation_v1"
        and activation.get("status") == "ACTIVE_FROZEN"
        and activation.get("execution_authorized") is True
        and activation.get("scene") == SCENE
        and activation.get("canonical_sha256") == canonical_sha256(activation),
        str(activation_path),
    )
    candidate_path = Path(str(activation.get("candidate_manifest_path", ""))).resolve()
    candidate = require_json(candidate_path, str(activation.get("candidate_manifest_sha256", "")))
    record(
        "candidate_activation_binding",
        candidate.get("canonical_sha256") == activation.get("candidate_manifest_canonical_sha256")
        and canonical_sha256(candidate) == activation.get("candidate_manifest_canonical_sha256"),
        str(candidate_path),
    )
    registry_path = Path(str(candidate["rgb_registry"]["path"])).resolve()
    registry = require_json(registry_path, str(candidate["rgb_registry"]["sha256"]))
    validate_addendum_runtime(
        activation=activation,
        candidate=candidate,
        registry=registry,
        executing_file=Path(__file__),
    )
    contract_path = Path(str(candidate["rgb_contract"]["path"])).resolve()
    contract = require_json(contract_path, str(candidate["rgb_contract"]["sha256"]))
    record(
        "registry_identity",
        registry.get("schema") == "m3m_gcp_native_quarter_rgb_quality_100k_registry_v1"
        and registry.get("status") == "ACTIVE_FROZEN"
        and registry.get("scene") == SCENE
        and registry.get("canonical_sha256") == candidate["rgb_registry"].get("canonical_sha256")
        and canonical_sha256(registry) == candidate["rgb_registry"].get("canonical_sha256"),
        str(registry_path),
    )
    record(
        "ready_registry_cardinality",
        registry.get("active_method_count") == len(registry.get("methods", []))
        and registry.get("ready_method_ids")
        == [row.get("method_id") for row in registry.get("methods", [])],
        registry.get("ready_method_ids"),
    )

    benchmark_identity = validate_benchmark_checkout(
        benchmark_repo=benchmark_repo,
        expected_commit=str(activation["reviewed_addendum_commit"]),
        expected_tree=str(activation["reviewed_addendum_tree"]),
        entrypoint=Path(__file__).resolve(),
    )
    record(
        "benchmark_checkout_clean",
        benchmark_identity.get("tracked_modified_files_sha256") == {}
        and benchmark_identity.get("unexpected_untracked_files") == [],
        benchmark_identity,
    )
    record(
        "registry_benchmark_checkout",
        same_path(registry["shared"].get("benchmark_repo_template"), benchmark_repo),
        registry["shared"].get("benchmark_repo_template"),
    )

    shared = registry["shared"]
    input_manifest_path = Path(str(shared["input_manifest"])).resolve()
    input_manifest = require_json(
        input_manifest_path, str(candidate["formal_input_manifest"]["sha256"])
    )
    test_rows = [row for row in input_manifest.get("images", []) if row.get("role") == "test"]
    record("heldout_view_count", len(test_rows) == 314, len(test_rows))
    input_root = Path(str(shared["input_root"])).resolve()
    bad_truth: list[str] = []
    for row in test_rows:
        path = (input_root / str(row["relative_path"])).resolve()
        if (
            not path.is_file()
            or sha256_file(path) != row.get("jpeg_sha256")
            or path.stat().st_size != row.get("jpeg_bytes")
        ):
            bad_truth.append(str(row.get("image_name")))
    record("heldout_jpeg_identity", not bad_truth, bad_truth)

    camera_manifest_path = Path(str(candidate["rgb_camera_root_manifest"]["path"])).resolve()
    camera_manifest = require_json(
        camera_manifest_path, str(candidate["rgb_camera_root_manifest"]["sha256"])
    )
    camera_root = Path(str(camera_manifest["output"]["root"])).resolve()
    record(
        "camera_manifest_identity",
        camera_manifest.get("status") == "PASS_RGB_EVALUATION_CAMERA_ROOT"
        and camera_manifest.get("canonical_sha256")
        == candidate["rgb_camera_root_manifest"].get("canonical_sha256")
        and canonical_sha256(camera_manifest)
        == candidate["rgb_camera_root_manifest"].get("canonical_sha256"),
        str(camera_manifest_path),
    )
    record(
        "camera_root_registry_binding",
        same_path(camera_root, shared["default_camera_root"])
        and same_path(camera_root, shared["graphdeco_camera_root"]),
        str(camera_root),
    )
    expected_sparse = shared["default_camera_sparse_sha256"]
    record(
        "camera_sparse_identity",
        sparse_model_sha256(camera_root) == expected_sparse,
        sparse_model_sha256(camera_root),
    )
    expected_names = {str(row["image_name"]) for row in test_rows}
    actual_names = {
        path.name for path in (camera_root / "images").iterdir() if path.is_file()
    }
    record("camera_image_inventory", actual_names == expected_names, len(actual_names))

    metric_root = Path(str(shared["metric_reference_root"])).resolve()
    for relative, expected_sha in contract["metric_reference"]["files_sha256"].items():
        path = metric_root / relative
        actual_sha = sha256_file(path) if path.is_file() else "MISSING"
        record(f"metric_source:{relative}", actual_sha == expected_sha, actual_sha)
    weights = {
        "vgg16-397923af.pth": Path(str(shared["vgg16_weights"])).resolve(),
        "vgg.pth": Path(str(shared["lpips_vgg_weights"])).resolve(),
    }
    for name, path in weights.items():
        actual_sha = sha256_file(path) if path.is_file() else "MISSING"
        record(
            f"metric_weight:{name}",
            actual_sha == contract["metric_reference"]["weights_sha256"][name],
            actual_sha,
        )
    metric_python = Path(str(shared["metric_environment"])).resolve() / "bin" / "python"
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
        record("metric_runtime", False, str(metric_python))

    base_repo = Path(str(candidate["base_checkout"]["path"])).resolve()
    record(
        "base_checkout_identity",
        subprocess.check_output(["git", "-C", str(base_repo), "rev-parse", "HEAD"], text=True).strip()
        == candidate["base_checkout"]["commit"]
        and subprocess.check_output(
            ["git", "-C", str(base_repo), "show", "-s", "--format=%T", "HEAD"], text=True
        ).strip()
        == candidate["base_checkout"]["tree"]
        and not subprocess.check_output(
            ["git", "-C", str(base_repo), "status", "--porcelain"], text=True
        ).strip(),
        str(base_repo),
    )

    smoke_bindings: list[dict[str, Any]] = []
    for method in registry["methods"]:
        method_id = str(method["method_id"])
        recipe_path = Path(str(method["recipe_path"])).resolve()
        recipe = require_json(recipe_path, str(method["recipe_sha256"]))
        bound_recipe = dict(recipe)
        bound_recipe["_recipe_path"] = str(recipe_path)
        identity_path = Path(str(method["attempt_model_identity_path"])).resolve()
        identity = validate_model_identity_bundle(
            manifest_path=identity_path,
            method_id=method_id,
            run_root=Path(str(method["run_root"])).resolve(),
            recipe=bound_recipe,
            repo=base_repo,
        )
        record(
            f"{method_id}:model_identity",
            sha256_file(identity_path) == method["attempt_model_identity_sha256"]
            and identity["canonical_sha256"]
            == method["attempt_model_identity_canonical_sha256"],
            str(identity_path),
        )
        bad_dependencies: list[str] = []
        for dependency in method.get("evaluation_dependencies", []):
            path = Path(str(dependency["path"])).resolve()
            if (
                not path.is_file()
                or path.stat().st_size != dependency.get("bytes")
                or sha256_file(path) != dependency.get("sha256")
            ):
                bad_dependencies.append(str(path))
        record(f"{method_id}:evaluation_dependencies", not bad_dependencies, bad_dependencies)
        source_identity = git_identity(Path(str(method["source_root"])).resolve())
        expected_source = method.get("source_worktree") or {}
        record(
            f"{method_id}:source_identity",
            source_identity.get("commit") == method.get("source_commit")
            and source_identity.get("tracked_diff_sha256")
            == expected_source.get("expected_tracked_diff_sha256")
            and source_identity.get("tracked_modified_files_sha256")
            == expected_source.get("expected_tracked_files_sha256", {})
            and source_identity.get("unexpected_untracked_files") == [],
            source_identity,
        )
        record(
            f"{method_id}:environment_python",
            (Path(str(method["environment"])).resolve() / "bin" / "python").is_file(),
            method["environment"],
        )
        expected_pythonpath = method.get("pythonpath_content_identity", [])
        actual_pythonpath = [
            directory_content_identity(Path(str(row["path"])).resolve())
            for row in expected_pythonpath
        ]
        record(
            f"{method_id}:runtime_pythonpath_identity",
            actual_pythonpath == expected_pythonpath,
            actual_pythonpath,
        )
        formal_root = Path(str(method["formal_output_root"])).resolve()
        record(
            f"{method_id}:formal_output_absent",
            not formal_root.exists() and not formal_root.is_symlink(),
            str(formal_root),
        )
        smoke_method_root = (smoke_root / method_id).resolve()
        smoke_manifest_path = smoke_method_root / "rgb_render_manifest.json"
        if not smoke_manifest_path.is_file():
            record(f"{method_id}:technical_smoke", False, str(smoke_manifest_path))
            continue
        validation = validate_render_and_ground_truth(
            contract_path=contract_path,
            input_manifest_path=input_manifest_path,
            input_root=input_root,
            render_manifest_path=smoke_manifest_path,
            scene=SCENE,
            method_id=method_id,
            allow_review_candidate=True,
            registry_path=registry_path,
        )
        smoke_passed = not validation.get("errors") and validation.get(
            "provenance_validation", {}
        ).get("passed") is True
        record(f"{method_id}:technical_smoke", smoke_passed, validation.get("errors", []))
        smoke_bindings.append(
            {
                "method_id": method_id,
                "root": str(smoke_method_root),
                "render_manifest_path": str(smoke_manifest_path),
                "render_manifest_sha256": sha256_file(smoke_manifest_path),
                "rendered_view_count": validation.get("validated_count"),
            }
        )

    payload: dict[str, Any] = {
        "schema": "m3m_gcp_native_quarter_rgb_quality_100k_preflight_v1",
        "status": "PASS_READY" if not errors else "FAIL",
        "passed": not errors,
        "formal_launch_ready": not errors,
        "pending": [],
        "errors": errors,
        "scene": SCENE,
        "method_count": len(registry.get("methods", [])),
        "inputs": {
            "activation_path": str(activation_path),
            "activation_sha256": sha256_file(activation_path),
            "candidate_path": str(candidate_path),
            "candidate_sha256": sha256_file(candidate_path),
            "contract_path": str(contract_path),
            "contract_sha256": sha256_file(contract_path),
            "registry_path": str(registry_path),
            "registry_sha256": sha256_file(registry_path),
            "benchmark_repo": str(benchmark_repo),
            "benchmark_commit": benchmark_identity["commit"],
            "benchmark_tree": benchmark_identity["tree"],
            "benchmark_clean": (
                benchmark_identity.get("tracked_modified_files_sha256") == {}
                and benchmark_identity.get("unexpected_untracked_files") == []
            ),
            "technical_smoke_root": str(smoke_root),
        },
        "technical_smoke_bindings": smoke_bindings,
        "checks": checks,
    }
    payload["canonical_sha256"] = canonical_sha256(payload)
    write_exclusive(output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
