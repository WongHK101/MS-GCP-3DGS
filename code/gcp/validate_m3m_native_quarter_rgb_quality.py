#!/usr/bin/env python3
"""Validate the additive native-quarter RGB-quality contract and 3K registry."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
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
    "sof": ("export_sof_rgb.py", "sof_rgb_v1"),
    "citygaussian_v2": ("export_lightning_gaussian_rgb.py", "lightning_gaussian_rgb_v1"),
    "citygs_x": ("export_citygs_x_rgb.py", "citygs_x_rgb_v1"),
    "metrogs": ("export_lightning_gaussian_rgb.py", "lightning_gaussian_rgb_v1"),
}
EXPECTED_EXTRA_CLI = {
    method_id: []
    for method_id in EXPECTED_METHODS
}
EXPECTED_EXTRA_CLI["rade_gs"] = [
    "--kernel_size",
    "0.0",
    "--use_decoupled_appearance",
    "0",
]
EXPECTED_APPEARANCE_POLICIES = {
    "3dgs_original": "none",
    "2dgs": "none",
    "pgsr": "exposure_compensation_false_use_render_not_app_image",
    "rade_gs": "canonical_base_render_training_only_pgsr_appearance_no_test_fit",
    "qgs": "none",
    "gsprior": "exposure_compensation_false_use_render_not_app_image",
    "sof": "canonical_render_training_only_decoupled_appearance_no_test_fit",
    "citygaussian_v2": "frozen_renderer_has_no_appearance_module",
    "citygs_x": "appearance_dim_0",
    "metrogs": "official_nearest_training_camera_geometry_alpha_0_7_v1",
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
EXPECTED_FORMAL_MODEL_SHA256 = {
    "3dgs_original": "461b48e97f31ee6588b5ba3de52d29ed07b4709134f7a155c95bc7c38dba91ff",
    "2dgs": "3d13956ad22ede5cae6bb5899f51c358a9d5a2d5f6e853980f705b8966454193",
    "pgsr": "c545036afc31b7ca5430e81b011d8cb2e7fc6ce75eabe647f2c902b7c4652880",
    "rade_gs": "37168235bbf81084d1cb0763b92026b6eac4f866498bd90205ac08266d50f222",
    "qgs": "041c7a623dfd3cd9a9799ad7b785f54bfbc76d050d04bc315ee5316cf76bef81",
    "gsprior": "8aa5474fe45c7f99409fed16191b677325a2060c83584f702eac5f88699bb1a0",
    "sof": "0428d1da4b8fdfef93368e1cad7891bfd5003dd4a94379342acdbae520cfd032",
    "citygaussian_v2": "345b9277de5310a6d9f6cdc86b24cfda5805be982eab6fb8170bc975f5789a64",
    "citygs_x": "bf530190e953d8e84145f72bebe13457bc849c0340d6beb3dae0872187e8fb7d",
    "metrogs": "f45527e44a75e8745b6785981a43309803202e09a950c1b94259b25ab4f36274",
}
EXPECTED_CFG_ARGS_SHA256 = {
    "3dgs_original": "3ddde6f5c3b30c14752149c54dcad1f02f68e68dc2b546955eb8a177206276c0",
    "2dgs": "1fefa01fadfde6bba4db1c4f08838e45e3dc09c6832ab200a47e9cde32241db3",
    "pgsr": "6e0315d30d957f6d4df519fb32dd76d637e942e20f2f6a277898e476173996c3",
    "rade_gs": "6bd7681a876802b9aa11166ba43a687a81b705a3c5055f155b1eb1084eae4f79",
    "gsprior": "ab599a3002cdbe34b99d4632f54160277ee2580795d5f1ad06827e4a855e70de",
    "sof": "a2de2148d0a26c756e38c3868e167e1de14a0497f7490ed0e0033b5520a77835",
    "citygs_x": "026c5d9a2f99bbd1867738868c8cdbcd71ea2513d3489e51817438bd4c52aec2",
}
EXPECTED_QGS_CONFIG_SHA256 = "2787868ea446e3282e090f75cbbf170e877b3d560418bbd05b6e89b346cf07d8"
EXPECTED_CITYGS_X_AUX_SHA256 = {
    "additional_attributes.npz": "f9ac471f155ec503205bb409a16e9d0162fa03aa20f38500121f12b951ccc789",
    "checkpoints.pth": "eca8b10cfecfee85f2ad61eb9f1cd60a99db4cb1b51507e2f01917c9c597fb4d",
}
EXPECTED_COMMON_CAMERA_SHA256 = {
    "cameras.bin": "a627e4ecd29ea1afe44937b56719d0cb5f3f4d20b8b368542db64a395306567f",
    "images.bin": "478d9bacff13d778cbeb0b616fdef044f4bf332ed97a940c037c0e61aff902eb",
    "points3D.bin": "44f88eabb7e536416ff8bcf211b7c22f1bb6d2ca6eff2731099e771c97ca689f",
}
EXPECTED_GSPRIOR_CAMERA_SHA256 = {
    "cameras.bin": "a627e4ecd29ea1afe44937b56719d0cb5f3f4d20b8b368542db64a395306567f",
    "images.bin": "0f59fe69c46c862d59326b2acc70c5324844ba9c555cceffbb77aac90ce3741b",
    "points3D.bin": "ef416ae289f748d3731e867005ca9e3adccb7370adbae3676e6fef08f868d99a",
}
EXPECTED_PYTHONPATH_IDENTITIES = {
    "pgsr": [
        {
            "path": "/root/autodl-tmp/build/m3m-gcp-native-quarter/pgsr/de24f1a38b350387e8d8fe381b2cd70c1ae946e7/qualification-v1/compat",
            "file_count": 2,
            "manifest_sha256": "432f6706d7477d9611d141c863ef973fd438fc679e5d17b40ea235deb613ec8b",
        }
    ],
    "gsprior": [
        {
            "path": "/root/autodl-tmp/staging/m3m-gcp-native-quarter/batch-20260818/gsprior/prep-v1/compat",
            "file_count": 2,
            "manifest_sha256": "0c26588f70d2454278ad5cd8019c9ed9a2a9d1f719ea84dc883cf6c010a01053",
        }
    ],
    "citygs_x": [
        {
            "path": "/root/autodl-tmp/staging/m3m-gcp-native-quarter/batch-20260818/citygs_x/pytorch3d_compat",
            "file_count": 2,
            "manifest_sha256": "c7dee8e4f1f52a480bc577c6d6799abc48a596cf7c232c9ca28228df3b5c20bf",
        }
    ],
}
EXPECTED_METRO_TRAINING_CAMERAS_SHA256 = (
    "e5faae00adee6a3576cd7a358bcd7e809f70a202375931eea3925c08499e31fd"
)


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
    activation_preflight_path: Path | None = None,
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
    require(
        "clear original_image before" in str(prediction.get("heldout_rgb_renderer_boundary", "")),
        "heldout RGB renderer boundary is not frozen",
    )
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
    require(
        gate.get("review_candidate_render_default")
        == "reject before creating any artifact path",
        "review-candidate render hard lock is not frozen",
    )
    require("technical_smoke_root" in str(gate.get("technical_smoke_exception", "")), "technical-smoke confinement is missing")
    require("PASS_READY" in str(gate.get("activation_preflight", "")), "ACTIVE preflight gate is missing")
    require("commit and tree" in str(gate.get("benchmark_checkout_identity", "")), "benchmark checkout identity gate is missing")
    require(gate.get("no_overwrite") is True, "no-overwrite gate disabled")
    require(
        gate.get("camera_sparse_training_appearance_and_runtime_compat_hashes_required")
        is True,
        "runtime auxiliary-content hash gate is disabled",
    )

    methods = registry.get("methods", [])
    method_ids = [method.get("method_id") for method in methods if isinstance(method, dict)]
    require(method_ids == EXPECTED_METHODS, "active method order or membership mismatch")
    require(len(set(method_ids)) == len(method_ids), "duplicate method ID")
    require(registry.get("active_method_count") == 10, "active method count mismatch")
    require(registry.get("retired_methods_excluded") == ["gof"], "retired method exclusion mismatch")
    require("gof" not in method_ids, "retired GOF leaked into the active RGB suite")

    shared = registry.get("shared", {})
    require(
        shared.get("default_camera_sparse_sha256") == EXPECTED_COMMON_CAMERA_SHA256,
        "default camera sparse-model hashes mismatch",
    )
    require(
        shared.get("graphdeco_camera_sparse_sha256") == EXPECTED_COMMON_CAMERA_SHA256,
        "Graphdeco camera sparse-model hashes mismatch",
    )
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
    require(
        shared.get("registry_relative_path")
        == "configs/m3m_gcp_native_quarter_rgb_quality_3k_registry_v1.json",
        "registry relative path mismatch",
    )
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
        require(
            method.get("formal_model_sha256") == EXPECTED_FORMAL_MODEL_SHA256.get(method_id),
            f"{method_id}: formal model SHA mismatch",
        )
        if method_id in EXPECTED_CFG_ARGS_SHA256:
            require(
                method.get("cfg_args_sha256") == EXPECTED_CFG_ARGS_SHA256[method_id],
                f"{method_id}: cfg_args SHA mismatch",
            )
        else:
            require("cfg_args_sha256" not in method, f"{method_id}: unexpected cfg_args SHA")
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
        require(
            method.get("extra_cli") == EXPECTED_EXTRA_CLI[method_id],
            f"{method_id}: frozen extra CLI mismatch",
        )
        require(
            method.get("appearance_policy") == EXPECTED_APPEARANCE_POLICIES[method_id],
            f"{method_id}: frozen appearance policy mismatch",
        )
        require(
            method.get("pythonpath_content_identity", [])
            == EXPECTED_PYTHONPATH_IDENTITIES.get(method_id, []),
            f"{method_id}: runtime PYTHONPATH content identity mismatch",
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
            require("environment_variables" not in method, "qgs: unexpected environment override")
            require(_absolute_posix(method.get("config_path")), "qgs: config path missing")
            require(
                method.get("config_sha256") == EXPECTED_QGS_CONFIG_SHA256,
                "qgs: config SHA mismatch",
            )
            require(_absolute_posix(method.get("model_root")), "qgs: model root missing")
        elif method_id in {"citygaussian_v2", "metrogs"}:
            require(_absolute_posix(method.get("formal_checkpoint")), f"{method_id}: checkpoint missing")
            require(
                method.get("environment_variables")
                == {"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"},
                f"{method_id}: checkpoint-load environment mismatch",
            )
        else:
            require(_absolute_posix(method.get("model_root")), f"{method_id}: model root missing")
            require("environment_variables" not in method, f"{method_id}: unexpected environment override")
        if method_id == "gsprior":
            require(
                method.get("camera_sparse_sha256") == EXPECTED_GSPRIOR_CAMERA_SHA256,
                "gsprior: normalized camera sparse-model hashes mismatch",
            )
        elif str(method.get("camera_root", "")).startswith("/"):
            require(
                "camera_sparse_sha256" not in method,
                f"{method_id}: unexpected method-specific camera hash set",
            )

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
    require(
        citygs.get("formal_model_aux_sha256") == EXPECTED_CITYGS_X_AUX_SHA256,
        "CityGS-X auxiliary model hashes mismatch",
    )
    if contract.get("status") == "ACTIVE_FROZEN":
        require(
            metro.get("formal_model_sha256") != "PENDING_METRO_FORMAL_COMPLETION",
            "active registry retains pending MetroGS model identity",
        )
    require(
        metro.get("training_cameras_json_sha256")
        == EXPECTED_METRO_TRAINING_CAMERAS_SHA256,
        "MetroGS training cameras SHA mismatch",
    )

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
    status = contract.get("status")
    evidence_path = (
        review_preflight_path
        if status == "REVIEW_CANDIDATE_NOT_FORMAL"
        else activation_preflight_path
    )
    require(evidence_path is not None and evidence_path.is_file(), "state-specific 901 preflight evidence missing")
    if evidence_path is not None and evidence_path.is_file():
        preflight_evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        require(preflight_evidence.get("passed") is True, "state-specific 901 preflight failed")
        require(preflight_evidence.get("errors") == [], "state-specific 901 preflight has errors")
        if status == "REVIEW_CANDIDATE_NOT_FORMAL":
            require(
                preflight_evidence.get("status")
                == "PASS_REMEDIATION_REVIEW_PENDING",
                "review-candidate 901 preflight status mismatch",
            )
            require(
                preflight_evidence.get("formal_launch_ready") is False,
                "review-candidate preflight became launch-ready",
            )
            require(
                preflight_evidence.get("pending")
                == [
                    "archive_attempt1_to_immutable_superseded_root",
                    "activate_reviewed_remediation_commit",
                    "run_fresh_active_preflight_after_archive",
                ],
                "review-candidate 901 preflight pending set mismatch",
            )
        elif status == "ACTIVE_FROZEN":
            require(preflight_evidence.get("status") == "PASS_READY", "ACTIVE preflight is not PASS_READY")
            require(preflight_evidence.get("formal_launch_ready") is True, "ACTIVE preflight launch gate is false")
            require(preflight_evidence.get("pending") == [], "ACTIVE preflight retains pending items")
        inputs = preflight_evidence.get("inputs", {})
        require(inputs.get("contract_sha256") == _sha256_file(contract_path), "preflight contract SHA binding mismatch")
        require(inputs.get("registry_sha256") == _sha256_file(registry_path), "preflight registry SHA binding mismatch")
        preflight_source = repo_root / "code" / "gcp" / "preflight_m3m_native_quarter_rgb_quality_3k.py"
        require(inputs.get("preflight_sha256") == _sha256_file(preflight_source), "preflight source SHA binding mismatch")
        if status == "ACTIVE_FROZEN":
            try:
                actual_commit = subprocess.check_output(
                    ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
                ).strip()
                actual_tree = subprocess.check_output(
                    ["git", "-C", str(repo_root), "rev-parse", "HEAD^{tree}"], text=True
                ).strip()
                actual_status = subprocess.check_output(
                    ["git", "-C", str(repo_root), "status", "--porcelain=v1", "--untracked-files=all"],
                    text=True,
                ).strip()
            except Exception as exc:  # noqa: BLE001
                actual_commit = actual_tree = f"ERROR:{type(exc).__name__}:{exc}"
                actual_status = "ERROR"
            require(inputs.get("benchmark_commit") == actual_commit, "ACTIVE preflight benchmark commit mismatch")
            require(inputs.get("benchmark_tree") == actual_tree, "ACTIVE preflight benchmark tree mismatch")
            require(inputs.get("benchmark_clean") is True and actual_status == "", "ACTIVE benchmark checkout is not clean")

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
        if status == "ACTIVE_FROZEN":
            require(
                smoke_manifest.get("contract_sha256") == _sha256_file(contract_path),
                "GPU evaluator smoke contract SHA binding mismatch",
            )
        require(
            smoke_manifest.get("outputs_sha256", {}).get("rgb_quality_summary.json")
            == _sha256_file(smoke_summary_path),
            "GPU evaluator smoke summary SHA binding mismatch",
        )

    if status == "REVIEW_CANDIDATE_NOT_FORMAL":
        remediation_smoke_path = (
            repo_root
            / "docs"
            / "protocol_evidence"
            / "m3m_native_quarter_rgb_quality_adapter_remediation_smoke_v1.json"
        )
        require(remediation_smoke_path.is_file(), "adapter remediation smoke evidence missing")
        if remediation_smoke_path.is_file():
            remediation = json.loads(remediation_smoke_path.read_text(encoding="utf-8"))
            require(
                remediation.get("status") == "PASS_TECHNICAL_SMOKE",
                "adapter remediation smoke did not pass",
            )
            require(remediation.get("ranking_eligible") is False, "adapter remediation smoke became rankable")
            require(remediation.get("formal_execution") is False, "adapter remediation smoke became formal")
            require(
                remediation.get("contract_sha256") == _sha256_file(contract_path),
                "adapter remediation contract SHA binding mismatch",
            )
            require(
                remediation.get("registry_sha256") == _sha256_file(registry_path),
                "adapter remediation registry SHA binding mismatch",
            )
            source_sha = remediation.get("source_sha256", {})
            for name in (
                "export_qgs_rgb.py",
                "export_qgs_depth_maps.py",
                "export_gaussian_rgb.py",
                "export_sof_rgb.py",
                "evaluate_m3m_native_quarter_rgb_quality.py",
            ):
                require(
                    source_sha.get(name) == _sha256_file(repo_root / "code" / "gcp" / name),
                    f"adapter remediation source SHA mismatch: {name}",
                )
            smoke_methods = remediation.get("methods", {})
            for method_id in ("qgs", "sof"):
                method_smoke = smoke_methods.get(method_id, {})
                require(
                    method_smoke.get("status") == "COMPLETE_UNRANKED",
                    f"{method_id}: remediation smoke incomplete",
                )
                require(
                    method_smoke.get("render_returncode") == 0
                    and method_smoke.get("metric_returncode") == 0,
                    f"{method_id}: remediation smoke returned nonzero",
                )
                require(
                    method_smoke.get("required_test_view_count") == 12
                    and method_smoke.get("rendered_test_view_count") == 12
                    and method_smoke.get("evaluated_test_view_count") == 12
                    and method_smoke.get("complete_test_coverage") is True
                    and method_smoke.get("input_render_validation_passed") is True,
                    f"{method_id}: remediation smoke coverage/validation mismatch",
                )
            require(remediation.get("failed_methods") == [], "adapter remediation smoke retains failures")

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
    parser.add_argument("--activation-preflight", type=Path)
    args = parser.parse_args()
    result = validate(
        args.repo_root.resolve(),
        args.contract.resolve(),
        args.registry.resolve(),
        args.current_pointer.resolve(),
        args.activation_preflight.resolve() if args.activation_preflight else None,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
