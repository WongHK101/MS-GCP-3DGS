from __future__ import annotations

import copy
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from evaluate_m3m_native_quarter_rgb_quality import (
    validate_registered_provenance,
    validate_render_and_ground_truth,
)
from build_m3m_native_quarter_rgb_quality_3k_commands import build_plan
from rgb_quality_contract import RgbRenderWriter, arithmetic_mean, git_identity, sha256_file
from validate_m3m_native_quarter_rgb_quality import validate


REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT = REPO_ROOT / "configs" / "m3m_gcp_native_quarter_rgb_quality_v1.json"
REGISTRY = REPO_ROOT / "configs" / "m3m_gcp_native_quarter_rgb_quality_3k_registry_v1.json"
POINTER = REPO_ROOT / "configs" / "m3m_gcp_native_quarter_current.json"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def _synthetic_contract_and_input(tmp_path: Path) -> tuple[Path, Path, Path]:
    input_root = tmp_path / "input"
    rows = []
    specs = [
        ("train.jpg", "train", np.full((4, 5, 3), 64, dtype=np.uint8)),
        ("test_a.jpg", "test", np.full((4, 5, 3), 128, dtype=np.uint8)),
        ("test_b.jpg", "test", np.full((4, 5, 3), 192, dtype=np.uint8)),
    ]
    for image_id, (name, role, array) in enumerate(specs, 1):
        relative = f"{role}/images/{name}"
        path = input_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(array, mode="RGB").save(path, format="JPEG", quality=95)
        rows.append(
            {
                "camera_id": 1,
                "height": 4,
                "image_id": image_id,
                "image_name": name,
                "jpeg_sha256": sha256_file(path),
                "relative_path": relative,
                "role": role,
                "width": 5,
            }
        )
    manifest = {
        "scene": "synthetic",
        "manifest_sha256": "1" * 64,
        "release_root_digest_sha256": "2" * 64,
        "pixel_domain": "synthetic_rgb",
        "holdout_semantics": "synthetic_holdout",
        "full_view_count": 3,
        "train_view_count": 1,
        "test_view_count": 2,
        "images": rows,
    }
    manifest_path = input_root / "NATIVE_QUARTER_INPUT_MANIFEST.json"
    _write_json(manifest_path, manifest)
    contract = {
        "schema": "m3m_gcp_native_quarter_rgb_quality_contract_v1",
        "suite_id": "m3m_gcp_native_quarter_rgb_quality_v1",
        "status": "REVIEW_CANDIDATE_NOT_FORMAL",
        "input_binding": {
            "release_root_digest_sha256": "2" * 64,
            "pixel_domain": "synthetic_rgb",
            "holdout_semantics": "synthetic_holdout",
            "scene_bindings": {
                "synthetic": {
                    "formal_input_manifest_file_sha256": sha256_file(manifest_path),
                    "formal_input_manifest_canonical_sha256": "1" * 64,
                    "full_view_count": 3,
                    "train_view_count": 1,
                    "test_view_count": 2,
                    "width": 5,
                    "height": 4,
                }
            },
        },
        "prediction_contract": {
            "file_format": "lossless RGB PNG",
            "quantization": "synthetic frozen quantization",
        },
    }
    contract_path = tmp_path / "contract.json"
    _write_json(contract_path, contract)
    return contract_path, manifest_path, input_root


def test_repository_contract_and_registry_pass() -> None:
    result = validate(REPO_ROOT, CONTRACT, REGISTRY, POINTER)
    assert result["passed"] is True, result["errors"]
    assert result["method_ids"][-1] == "metrogs"
    assert result["geometry_protocol_id"] == "m3m_gcp_native_quarter_geometry_v2"


def test_writer_and_common_validation_round_trip(tmp_path: Path) -> None:
    contract, input_manifest, input_root = _synthetic_contract_and_input(tmp_path)
    artifact_root = tmp_path / "artifact"
    writer = RgbRenderWriter(
        contract_path=contract,
        input_manifest_path=input_manifest,
        scene="synthetic",
        method_id="synthetic_method",
        output_dir=artifact_root / "renders",
        manifest_path=artifact_root / "rgb_render_manifest.json",
    )
    assert writer.expected_names == ["test_a.jpg", "test_b.jpg"]
    writer.save("test_a.jpg", np.full((3, 4, 5), 0.25, dtype=np.float32))
    writer.save("test_b.jpg", np.full((3, 4, 5), 0.75, dtype=np.float32))
    manifest = writer.finalize(provenance={"heldout_rgb_used_by_adapter": False})
    assert manifest["complete_test_coverage"] is True
    result = validate_render_and_ground_truth(
        contract_path=contract,
        input_manifest_path=input_manifest,
        input_root=input_root,
        render_manifest_path=artifact_root / "rgb_render_manifest.json",
        scene="synthetic",
        method_id="synthetic_method",
        allow_review_candidate=True,
    )
    assert result["passed"] is True, result["errors"]
    assert result["validated_count"] == 2


def test_incomplete_render_set_fails_closed(tmp_path: Path) -> None:
    contract, input_manifest, _input_root = _synthetic_contract_and_input(tmp_path)
    writer = RgbRenderWriter(
        contract_path=contract,
        input_manifest_path=input_manifest,
        scene="synthetic",
        method_id="synthetic_method",
        output_dir=tmp_path / "artifact" / "renders",
    )
    writer.save("test_a.jpg", np.zeros((3, 4, 5), dtype=np.float32))
    with pytest.raises(RuntimeError, match="incomplete RGB render set"):
        writer.finalize(provenance={})


def test_writer_rejects_path_components(tmp_path: Path) -> None:
    contract, input_manifest, _input_root = _synthetic_contract_and_input(tmp_path)
    writer = RgbRenderWriter(
        contract_path=contract,
        input_manifest_path=input_manifest,
        scene="synthetic",
        method_id="synthetic_method",
        output_dir=tmp_path / "artifact" / "renders",
    )
    with pytest.raises(ValueError, match="must be a basename"):
        writer.save("../test_a.jpg", np.zeros((3, 4, 5), dtype=np.float32))


def test_formal_evaluation_rejects_review_candidate(tmp_path: Path) -> None:
    contract, input_manifest, input_root = _synthetic_contract_and_input(tmp_path)
    artifact_root = tmp_path / "artifact"
    writer = RgbRenderWriter(
        contract_path=contract,
        input_manifest_path=input_manifest,
        scene="synthetic",
        method_id="synthetic_method",
        output_dir=artifact_root / "renders",
    )
    writer.save("test_a.jpg", np.zeros((3, 4, 5), dtype=np.float32))
    writer.save("test_b.jpg", np.zeros((3, 4, 5), dtype=np.float32))
    writer.finalize(provenance={})
    with pytest.raises(ValueError, match="contract status is not executable"):
        validate_render_and_ground_truth(
            contract_path=contract,
            input_manifest_path=input_manifest,
            input_root=input_root,
            render_manifest_path=artifact_root / "rgb_render_manifest.json",
            scene="synthetic",
            method_id="synthetic_method",
            allow_review_candidate=False,
        )


def test_formal_validation_requires_frozen_registry(tmp_path: Path) -> None:
    contract, input_manifest, input_root = _synthetic_contract_and_input(tmp_path)
    payload = json.loads(contract.read_text(encoding="utf-8"))
    payload["status"] = "ACTIVE_FROZEN"
    _write_json(contract, payload)
    artifact_root = tmp_path / "artifact"
    writer = RgbRenderWriter(
        contract_path=contract,
        input_manifest_path=input_manifest,
        scene="synthetic",
        method_id="synthetic_method",
        output_dir=artifact_root / "renders",
    )
    writer.save("test_a.jpg", np.zeros((3, 4, 5), dtype=np.float32))
    writer.save("test_b.jpg", np.zeros((3, 4, 5), dtype=np.float32))
    writer.finalize(provenance={})

    result = validate_render_and_ground_truth(
        contract_path=contract,
        input_manifest_path=input_manifest,
        input_root=input_root,
        render_manifest_path=artifact_root / "rgb_render_manifest.json",
        scene="synthetic",
        method_id="synthetic_method",
        allow_review_candidate=False,
    )

    assert result["passed"] is False
    assert "provenance: formal evaluation requires --registry" in result["errors"]


def test_registry_rejects_method_specific_evaluator(tmp_path: Path) -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    registry = copy.deepcopy(registry)
    registry["methods"][0]["evaluator"] = "method_native_metrics.py"
    mutated = tmp_path / "registry.json"
    _write_json(mutated, registry)
    result = validate(REPO_ROOT, CONTRACT, mutated, POINTER)
    assert result["passed"] is False
    assert "3dgs_original: method-specific evaluator registered" in result["errors"]


def test_arithmetic_mean_and_exact_match_policy() -> None:
    assert arithmetic_mean([1.0, 2.0, 3.0]) == 2.0
    assert math.isinf(arithmetic_mean([1.0, math.inf]))
    with pytest.raises(ValueError, match="non-finite"):
        arithmetic_mean([1.0, math.nan])


def test_git_identity_includes_staged_tracked_changes(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "RGB Test"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "rgb-test@example.invalid"],
        check=True,
    )
    tracked = repo / "tracked.txt"
    tracked.write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "base"], check=True)
    tracked.write_text("staged change\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "tracked.txt"], check=True)

    identity = git_identity(repo)

    assert identity["tracked_diff_sha256"] != hashlib.sha256(b"").hexdigest()
    assert identity["tracked_modified_files_sha256"] == {
        "tracked.txt": sha256_file(tracked)
    }


def test_registered_provenance_binds_and_detects_model_tamper(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "config", "user.name", "RGB Test"], check=True)
    subprocess.run(
        ["git", "-C", str(source), "config", "user.email", "rgb-test@example.invalid"],
        check=True,
    )
    renderer = source / "renderer.py"
    renderer.write_text("# frozen renderer\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(source), "add", "renderer.py"], check=True)
    subprocess.run(["git", "-C", str(source), "commit", "-q", "-m", "base"], check=True)
    source_identity = git_identity(source)
    camera_root = tmp_path / "cameras"
    camera_root.mkdir()
    model_root = tmp_path / "model"
    model_root.mkdir()
    model = model_root / "model.bin"
    model.write_bytes(b"frozen model")
    adapter = MODULE_DIR / "export_citygs_x_rgb.py"
    registry = {
        "suite_id": "m3m_gcp_native_quarter_rgb_quality_v1",
        "status": "ACTIVE_FROZEN",
        "scene": "synthetic",
        "shared": {"default_camera_root": str(camera_root)},
        "methods": [
            {
                "method_id": "synthetic_method",
                "adapter": adapter.name,
                "adapter_kind": "synthetic_rgb_v1",
                "source_root": str(source),
                "source_commit": source_identity["commit"],
                "camera_root": "shared.default_camera_root",
                "model_root": str(model_root),
                "formal_model_relative_path": model.name,
                "formal_model_sha256": sha256_file(model),
                "iteration": 1,
                "appearance_policy": "none",
            }
        ],
    }
    registry_path = tmp_path / "registry.json"
    _write_json(registry_path, registry)
    provenance = {
        "adapter_kind": "synthetic_rgb_v1",
        "adapter_path": str(adapter),
        "adapter_sha256": sha256_file(adapter),
        "renderer_repository": source_identity,
        "renderer_source_path": str(renderer),
        "renderer_source_sha256": sha256_file(renderer),
        "camera_source_root": str(camera_root),
        "model_root": str(model_root),
        "formal_model_path": str(model),
        "formal_model_sha256": sha256_file(model),
        "iteration": 1,
        "appearance_policy": "none",
        "white_background": False,
        "heldout_rgb_used_by_adapter": False,
        "heldout_rgb_consumed_by_renderer_or_policy": False,
        "test_time_optimization": False,
    }
    contract = {
        "suite_id": "m3m_gcp_native_quarter_rgb_quality_v1",
        "status": "ACTIVE_FROZEN",
    }

    passed = validate_registered_provenance(
        contract=contract,
        registry_path=registry_path,
        render_manifest={"provenance": provenance},
        scene="synthetic",
        method_id="synthetic_method",
        allow_review_candidate=False,
    )
    assert passed["passed"] is True, passed["errors"]

    model.write_bytes(b"tampered model")
    failed = validate_registered_provenance(
        contract=contract,
        registry_path=registry_path,
        render_manifest={"provenance": provenance},
        scene="synthetic",
        method_id="synthetic_method",
        allow_review_candidate=False,
    )
    assert failed["passed"] is False
    assert any("formal_model_sha256" in error for error in failed["errors"])


def test_review_candidate_command_plan_is_explicitly_non_executable() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    plan = build_plan(
        registry,
        benchmark_repo="/reviewed/benchmark/repo",
        allow_review_candidate=True,
    )
    assert plan["formal_execution_authorized"] is False
    assert plan["job_count"] == 10
    assert plan["method_order"][-1] == "metrogs"
    assert "--training_cameras_json" in plan["jobs"][-1]["render"]["argv"]
    assert all("--registry" in job["metric"]["argv"] for job in plan["jobs"])
    assert all(
        job["metric"]["argv"][2].endswith("evaluate_m3m_native_quarter_rgb_quality.py")
        for job in plan["jobs"]
    )
