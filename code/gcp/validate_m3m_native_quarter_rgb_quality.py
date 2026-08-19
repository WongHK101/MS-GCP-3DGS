#!/usr/bin/env python3
"""Validate the additive native-quarter RGB-quality contract and 3K registry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any


SUITE_ID = "m3m_gcp_native_quarter_rgb_quality_v1"
CONTRACT_SCHEMA = "m3m_gcp_native_quarter_rgb_quality_contract_v1"
REGISTRY_SCHEMA = "m3m_gcp_native_quarter_rgb_quality_3k_registry_v1"
ALLOWED_STATUSES = {"REVIEW_CANDIDATE_NOT_FORMAL", "ACTIVE_FROZEN"}
EXPECTED_METHODS = [
    "3dgs_original",
    "2dgs",
    "pgsr",
    "rade_gs",
    "qgs",
    "gsprior",
    "sof",
    "citygaussian_v2",
    "citygs_x",
    "metrogs",
]
EXPECTED_COMMITS = {
    "3dgs_original": "2eee0e26d2d5fd00ec462df47752223952f6bf4e",
    "2dgs": "335ad612f2e783a4e57b9cbc4d1e167bd599fc98",
    "pgsr": "de24f1a38b350387e8d8fe381b2cd70c1ae946e7",
    "rade_gs": "d72f20792005ae1d6555a82aa2d15345f247604e",
    "qgs": "74d05c945e99fcaef7afe5a8831903be71ad9b55",
    "gsprior": "dcb7c89fb6b60f068b440de45d064ecc7fbcba55",
    "sof": "b9eb4170c843014f5f96d54924976161bd675469",
    "citygaussian_v2": "e84c7c8774dd11d3f4189be3488e1220afa20a86",
    "citygs_x": "27617f2486505e3b6fe75345edf7c2b11161bc2a",
    "metrogs": "8cf9ac13c0c34b65c1a935d181c4634909e60f3f",
}
EXPECTED_ADAPTERS = {
    "3dgs_original": ("export_gaussian_rgb.py", "graphdeco_style_gaussian_rgb_v1"),
    "2dgs": ("export_gaussian_rgb.py", "graphdeco_style_gaussian_rgb_v1"),
    "pgsr": ("export_gaussian_rgb.py", "graphdeco_style_gaussian_rgb_v1"),
    "rade_gs": ("export_gaussian_rgb.py", "graphdeco_style_gaussian_rgb_v1"),
    "qgs": ("export_qgs_rgb.py", "qgs_rgb_v1"),
    "gsprior": ("export_gaussian_rgb.py", "graphdeco_style_gaussian_rgb_v1"),
    "sof": ("export_gaussian_rgb.py", "graphdeco_style_gaussian_rgb_v1"),
    "citygaussian_v2": ("export_lightning_gaussian_rgb.py", "lightning_gaussian_rgb_v1"),
    "citygs_x": ("export_citygs_x_rgb.py", "citygs_x_rgb_v1"),
    "metrogs": ("export_lightning_gaussian_rgb.py", "lightning_gaussian_rgb_v1"),
}
EXPECTED_METRIC_FILES = {
    "metrics.py": "bda39191dde1fad93abf56a994d6f799bc02209b8ac5000b038e5ecf3345d6d3",
    "utils/image_utils.py": "872a4507773b9378db9e0fefc90104dc474b27551af18ada73f000a7a00a4ba0",
    "utils/loss_utils.py": "53e4824dd41f847bd9c1ce146162886d1d63460acafbfed9db1f3d84dfe63f5f",
    "lpipsPyTorch/__init__.py": "a657ce7355782eb970f95fd0d2e26eec5b9023212d2b655814da6e15b70510b4",
    "lpipsPyTorch/modules/lpips.py": "3bece0b9cf9943af5b043026458819d259df9bfe7c2a1d3ffc6c905e7e5aa2b4",
    "lpipsPyTorch/modules/networks.py": "dfa6f152b0e3fbc23ac3bfaea52b46e9473e64ad539ea22ab030c57af51c7f14",
    "lpipsPyTorch/modules/utils.py": "1bd4a7d4e7b43215497675ed852936fe4f27e7ea3068afe0b0e1f7cfb6c48570",
}
EXPECTED_DIRTY_SOURCE_DIFFS = {
    "pgsr": "173addb7351cd3cfe1fbd56e7e2efed080adcd8a5b00a90756b872b4b9551bfd",
    "gsprior": "a2fa787c8e0de02826ee391bdb8a86e784038f60aa5b9a49b3ce4ba6bd883f27",
    "citygs_x": "bcf3545ed19ef93896d4d80a6bef330d69f4de4e3ab3fd3a3ba523a022364c6a",
}


def _absolute_posix(value: Any) -> bool:
    return isinstance(value, str) and PurePosixPath(value).is_absolute()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(
    repo_root: Path,
    contract_path: Path,
    registry_path: Path,
    current_pointer_path: Path,
) -> dict[str, Any]:
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    pointer = json.loads(current_pointer_path.read_text(encoding="utf-8"))

    require(contract.get("schema") == CONTRACT_SCHEMA, "contract schema mismatch")
    require(registry.get("schema") == REGISTRY_SCHEMA, "registry schema mismatch")
    require(contract.get("suite_id") == SUITE_ID, "contract suite ID mismatch")
    require(registry.get("suite_id") == SUITE_ID, "registry suite ID mismatch")
    require(contract.get("status") in ALLOWED_STATUSES, "contract status is invalid")
    require(registry.get("status") == contract.get("status"), "contract/registry status mismatch")

    relationship = contract.get("relationship_to_geometry_protocol", {})
    require(
        relationship.get("protocol_id") == "m3m_gcp_native_quarter_geometry_v2",
        "RGB suite is not bound to geometry protocol v2",
    )
    require(relationship.get("geometry_protocol_mutated") is False, "RGB suite mutates geometry")
    require(
        relationship.get("geometry_ranking_or_evidence_superseded") is False,
        "RGB suite supersedes geometry evidence",
    )
    require(pointer.get("status") == "ACTIVE", "current geometry pointer is not active")
    require(
        pointer.get("protocol_id") == "m3m_gcp_native_quarter_geometry_v2",
        "current geometry pointer changed",
    )

    binding = contract.get("input_binding", {})
    scene_binding = binding.get("scene_bindings", {}).get("gcp_3000_20260602", {})
    require(
        binding.get("release_root_digest_sha256")
        == "513f8999fe4b110f15bcbecad7932895781cee755ee9ccd7a14ff10298546d75",
        "source release digest mismatch",
    )
    require(
        binding.get("pixel_domain") == "colmap_4_0_4_image_undistorter_pinhole_max_1414",
        "pixel domain mismatch",
    )
    require(
        binding.get("holdout_semantics") == "image_loss_holdout_under_shared_all_image_sfm_v1",
        "holdout semantics mismatch",
    )
    require(
        scene_binding.get("formal_input_manifest_file_sha256")
        == "ae29817198f54f04e4133a7b5fd03df679dd6f259b2d1ef4125e825cbb8e422e",
        "3K input-manifest file SHA mismatch",
    )
    require(
        scene_binding.get("formal_input_manifest_canonical_sha256")
        == "4ae07aad9278e2eb5af2f04268f3301df56c6f6ada9ee51c6f125fdbb29e7ec8",
        "3K input-manifest canonical SHA mismatch",
    )
    require(
        [scene_binding.get(key) for key in ("full_view_count", "train_view_count", "test_view_count")]
        == [94, 82, 12],
        "3K view counts mismatch",
    )
    require(
        [scene_binding.get("width"), scene_binding.get("height")] == [1414, 1025],
        "3K image size mismatch",
    )

    prediction = contract.get("prediction_contract", {})
    forbidden = set(prediction.get("forbidden", []))
    require(
        {"resize", "crop", "pad", "test-RGB exposure fitting", "test-RGB appearance optimization", "method-specific metric implementation"}
        <= forbidden,
        "prediction forbidden-operation set is incomplete",
    )
    require(prediction.get("file_format") == "lossless RGB PNG", "prediction encoding mismatch")
    domain = contract.get("metric_domain", {})
    require(domain.get("mask") == "none; full frame", "metric mask is not full-frame")
    require(domain.get("crop") == "none", "metric crop is enabled")
    require(domain.get("color_space_conversion") == "none", "metric color conversion is enabled")

    metrics = contract.get("metrics", {})
    require(list(metrics) == ["PSNR", "SSIM", "LPIPS_VGG"], "metric set/order mismatch")
    require(metrics.get("PSNR", {}).get("direction") == "higher_is_better", "PSNR direction mismatch")
    require(metrics.get("SSIM", {}).get("direction") == "higher_is_better", "SSIM direction mismatch")
    require(metrics.get("LPIPS_VGG", {}).get("direction") == "lower_is_better", "LPIPS direction mismatch")
    reference = contract.get("metric_reference", {})
    require(
        reference.get("repository_commit") == EXPECTED_COMMITS["3dgs_original"],
        "metric reference commit mismatch",
    )
    require(reference.get("files_sha256") == EXPECTED_METRIC_FILES, "metric source hashes mismatch")
    require(
        reference.get("weights_sha256")
        == {
            "vgg16-397923af.pth": "397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0",
            "vgg.pth": "a78928a0af1e5f0fcb1f3b9e8f8c3a2a5a3de244d830ad5c1feddc79b8432868",
        },
        "metric weight hashes mismatch",
    )
    require(
        reference.get("runtime")
        == {
            "python": "3.10.12",
            "torch": "2.7.1+cu128",
            "torchvision": "0.22.1+cu128",
            "Pillow": "11.1.0",
            "numpy": "1.26.4",
        },
        "metric runtime identity mismatch",
    )

    aggregation = contract.get("coverage_and_aggregation", {})
    require(aggregation.get("complete_scene_required") is True, "complete-scene gate disabled")
    require(aggregation.get("incomplete_scene_status") == "INCOMPLETE_UNRANKED", "incomplete status mismatch")
    require(aggregation.get("per_scene") == "unweighted arithmetic mean over all frozen test views", "per-scene aggregation mismatch")
    require(aggregation.get("cross_scene") == "unweighted macro mean over complete scenes", "cross-scene aggregation mismatch")
    gate = contract.get("formal_gate", {})
    require(gate.get("review_required_before_first_formal_run") is True, "review gate disabled")
    require(gate.get("required_status_for_formal_run") == "ACTIVE_FROZEN", "formal status gate mismatch")
    require(gate.get("no_overwrite") is True, "no-overwrite gate disabled")

    methods = registry.get("methods", [])
    method_ids = [method.get("method_id") for method in methods if isinstance(method, dict)]
    require(method_ids == EXPECTED_METHODS, "active method order or membership mismatch")
    require(len(set(method_ids)) == len(method_ids), "duplicate method ID")
    require(registry.get("active_method_count") == 10, "active method count mismatch")
    require(registry.get("retired_methods_excluded") == ["gof"], "retired method exclusion mismatch")
    require("gof" not in method_ids, "retired GOF leaked into the active RGB suite")

    shared = registry.get("shared", {})
    for key in (
        "input_manifest",
        "input_root",
        "default_camera_root",
        "graphdeco_camera_root",
        "metric_environment",
        "metric_reference_root",
        "vgg16_weights",
        "lpips_vgg_weights",
    ):
        require(_absolute_posix(shared.get(key)), f"shared {key} must be an absolute POSIX path")
    require(shared.get("output_relative_path") == "rgb_quality_v1", "output directory identity mismatch")
    require(shared.get("metric_device") == "cuda:0", "metric device mismatch")
    require(
        shared.get("benchmark_repo_runtime_argument_required") is True,
        "benchmark repository must be supplied explicitly at command-plan generation",
    )
    benchmark_template = shared.get("benchmark_repo_template", "")
    require(_absolute_posix(benchmark_template), "benchmark repository template is not absolute")
    require(
        "{reviewed_active_commit}" in benchmark_template,
        "benchmark repository template does not expose the reviewed commit binding",
    )

    for method in methods:
        if not isinstance(method, dict):
            continue
        method_id = str(method.get("method_id"))
        expected_adapter = EXPECTED_ADAPTERS.get(method_id)
        require(expected_adapter is not None, f"{method_id}: no expected adapter")
        if expected_adapter:
            adapter_name, adapter_kind = expected_adapter
            require(method.get("adapter") == adapter_name, f"{method_id}: adapter mismatch")
            require(method.get("adapter_kind") == adapter_kind, f"{method_id}: adapter kind mismatch")
            require((repo_root / "code" / "gcp" / adapter_name).is_file(), f"{method_id}: adapter file missing")
        require(method.get("source_commit") == EXPECTED_COMMITS.get(method_id), f"{method_id}: source commit mismatch")
        worktree = method.get("source_worktree")
        if method_id in EXPECTED_DIRTY_SOURCE_DIFFS:
            require(isinstance(worktree, dict), f"{method_id}: patched source identity missing")
            if isinstance(worktree, dict):
                require(
                    worktree.get("expected_tracked_diff_sha256") == EXPECTED_DIRTY_SOURCE_DIFFS[method_id],
                    f"{method_id}: patched source diff identity mismatch",
                )
                files = worktree.get("expected_tracked_files_sha256")
                require(isinstance(files, dict) and bool(files), f"{method_id}: patched source file hashes missing")
                require(worktree.get("untracked_policy") == "generated_pycache_only", f"{method_id}: untracked-file policy mismatch")
        else:
            require(worktree is None, f"{method_id}: unexpected dirty-source allowance")
        for key in ("source_root", "environment", "run_root"):
            require(_absolute_posix(method.get(key)), f"{method_id}: {key} is not absolute")
        camera_root = method.get("camera_root")
        require(
            camera_root in {"shared.default_camera_root", "shared.graphdeco_camera_root"}
            or _absolute_posix(camera_root),
            f"{method_id}: camera root is invalid",
        )
        require(isinstance(method.get("appearance_policy"), str) and bool(method.get("appearance_policy")), f"{method_id}: appearance policy missing")
        require(isinstance(method.get("extra_cli"), list), f"{method_id}: extra CLI must be a list")
        require(
            all(isinstance(value, str) for value in method.get("extra_cli", [])),
            f"{method_id}: extra CLI contains a non-string",
        )
        require("evaluator" not in method and "metrics_script" not in method, f"{method_id}: method-specific evaluator registered")

        if method_id in {"citygaussian_v2", "citygs_x", "metrogs"}:
            require(
                method.get("input_class") == "rgb_colmap_external_geometry_prior",
                f"{method_id}: input stratum mismatch",
            )
        else:
            require(method.get("input_class") == "rgb_colmap_only", f"{method_id}: input stratum mismatch")
        if method_id == "qgs":
            require(_absolute_posix(method.get("config_path")), "qgs: config path missing")
            require(_absolute_posix(method.get("model_root")), "qgs: model root missing")
        elif method_id in {"citygaussian_v2", "metrogs"}:
            require(_absolute_posix(method.get("formal_checkpoint")), f"{method_id}: checkpoint missing")
        else:
            require(_absolute_posix(method.get("model_root")), f"{method_id}: model root missing")

    appearance = contract.get("appearance_policy", {})
    metro = next((method for method in methods if method.get("method_id") == "metrogs"), {})
    require(
        metro.get("appearance_policy") == "official_nearest_training_camera_geometry_alpha_0_7_v1",
        "MetroGS appearance rule mismatch",
    )
    require(_absolute_posix(metro.get("training_cameras_json")), "MetroGS training camera list missing")
    require("alpha=0.7" in str(appearance.get("metrogs", "")), "MetroGS alpha is not frozen")
    sof = next((method for method in methods if method.get("method_id") == "sof"), {})
    require("no_test_fit" in str(sof.get("appearance_policy", "")), "SOF heldout-fit ban missing")
    require(_absolute_posix(sof.get("splatting_config_path")), "SOF frozen splatting config missing")
    require(
        sof.get("splatting_config_sha256")
        == "39e7d3f021401602604d009d8b30182e4edf92e934f8df8e62fe0cd842bfafac",
        "SOF frozen splatting config SHA mismatch",
    )
    rade = next((method for method in methods if method.get("method_id") == "rade_gs"), {})
    require("canonical_base_render" in str(rade.get("appearance_policy", "")), "RaDe-GS canonical appearance rule missing")
    citygs = next((method for method in methods if method.get("method_id") == "citygs_x"), {})
    require(citygs.get("appearance_policy") == "appearance_dim_0", "CityGS-X appearance rule mismatch")

    execution = registry.get("execution_policy", {})
    require(execution.get("geometry_formal_completion_required_before_rgb") is True, "geometry-before-RGB gate disabled")
    require(execution.get("formal_contract_status_required") == "ACTIVE_FROZEN", "registry formal gate mismatch")
    require(execution.get("render_then_shared_metric") is True, "shared evaluator boundary missing")
    require(execution.get("model_files_remain_on_901") is True, "model retention policy changed")

    required_sources = [
        repo_root / "code" / "gcp" / "rgb_quality_contract.py",
        repo_root / "code" / "gcp" / "evaluate_m3m_native_quarter_rgb_quality.py",
        repo_root / "code" / "gcp" / "preflight_m3m_native_quarter_rgb_quality_3k.py",
        repo_root / "code" / "gcp" / "build_m3m_native_quarter_rgb_quality_3k_commands.py",
        repo_root / "docs" / "M3M_GCP_NATIVE_QUARTER_RGB_QUALITY_V1.md",
    ]
    for path in required_sources:
        require(path.is_file(), f"required RGB suite component missing: {path.relative_to(repo_root)}")

    review_preflight_path = (
        repo_root
        / "docs"
        / "protocol_evidence"
        / "m3m_native_quarter_rgb_quality_3k_review_preflight_v1.json"
    )
    require(review_preflight_path.is_file(), "review-candidate 901 preflight evidence missing")
    if review_preflight_path.is_file():
        review_preflight = json.loads(review_preflight_path.read_text(encoding="utf-8"))
        require(review_preflight.get("passed") is True, "review-candidate 901 preflight failed")
        require(
            review_preflight.get("status") == "PASS_STATIC_METRO_PENDING",
            "review-candidate 901 preflight status mismatch",
        )
        require(review_preflight.get("errors") == [], "review-candidate 901 preflight has errors")
        require(
            review_preflight.get("pending")
            == ["metrogs:geometry_formal_completion", "metrogs:formal_model"],
            "review-candidate 901 preflight pending set mismatch",
        )
        inputs = review_preflight.get("inputs", {})
        require(inputs.get("contract_sha256") == _sha256_file(contract_path), "preflight contract SHA binding mismatch")
        require(inputs.get("registry_sha256") == _sha256_file(registry_path), "preflight registry SHA binding mismatch")
        preflight_source = repo_root / "code" / "gcp" / "preflight_m3m_native_quarter_rgb_quality_3k.py"
        require(
            inputs.get("preflight_sha256") == _sha256_file(preflight_source),
            "preflight source SHA binding mismatch",
        )

    smoke_summary_path = (
        repo_root
        / "docs"
        / "protocol_evidence"
        / "m3m_native_quarter_rgb_quality_evaluator_gpu_smoke_summary_v1.json"
    )
    smoke_manifest_path = (
        repo_root
        / "docs"
        / "protocol_evidence"
        / "m3m_native_quarter_rgb_quality_evaluator_gpu_smoke_manifest_v1.json"
    )
    require(smoke_summary_path.is_file(), "GPU evaluator smoke summary missing")
    require(smoke_manifest_path.is_file(), "GPU evaluator smoke manifest missing")
    if smoke_summary_path.is_file() and smoke_manifest_path.is_file():
        smoke_summary = json.loads(smoke_summary_path.read_text(encoding="utf-8"))
        smoke_manifest = json.loads(smoke_manifest_path.read_text(encoding="utf-8"))
        require(
            smoke_summary.get("status") == "TECHNICAL_SMOKE_COMPLETE_UNRANKED",
            "GPU evaluator smoke is not explicitly unranked",
        )
        require(smoke_summary.get("ranking_eligible") is False, "GPU evaluator smoke became rankable")
        require(smoke_summary.get("formal_execution") is False, "GPU evaluator smoke became formal")
        require(smoke_summary.get("primary_scene_mean_available") is False, "GPU evaluator smoke exposes a primary mean")
        require(smoke_summary.get("evaluated_test_view_count") == 12, "GPU evaluator smoke coverage mismatch")
        require(smoke_manifest.get("status") == "PASS_TECHNICAL_SMOKE", "GPU evaluator smoke manifest status mismatch")
        require(smoke_manifest.get("ranking_eligible") is False, "GPU evaluator smoke manifest became rankable")
        require(
            smoke_manifest.get("evaluator_sha256")
            == _sha256_file(repo_root / "code" / "gcp" / "evaluate_m3m_native_quarter_rgb_quality.py"),
            "GPU evaluator smoke evaluator SHA binding mismatch",
        )
        require(
            smoke_manifest.get("contract_sha256") == _sha256_file(contract_path),
            "GPU evaluator smoke contract SHA binding mismatch",
        )
        require(
            smoke_manifest.get("outputs_sha256", {}).get("rgb_quality_summary.json")
            == _sha256_file(smoke_summary_path),
            "GPU evaluator smoke summary SHA binding mismatch",
        )

    return {
        "schema": "m3m_gcp_native_quarter_rgb_quality_validation_v1",
        "status": "PASS" if not errors else "FAIL",
        "passed": not errors,
        "suite_id": contract.get("suite_id"),
        "contract_status": contract.get("status"),
        "scene": registry.get("scene"),
        "method_ids": method_ids,
        "geometry_protocol_id": pointer.get("protocol_id"),
        "errors": errors,
    }


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
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
    parser.add_argument(
        "--current-pointer",
        type=Path,
        default=repo_root / "configs" / "m3m_gcp_native_quarter_current.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(
        args.repo_root.resolve(),
        args.contract.resolve(),
        args.registry.resolve(),
        args.current_pointer.resolve(),
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
