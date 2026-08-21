#!/usr/bin/env python3
"""Build the ten hash-bound execution recipes for the 100K scene."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "configs" / "m3m_gcp_native_quarter_100k_recipes_v1"
SCENE = "gcp_100000_20260610"
FORMAL = f"/root/autodl-tmp/datasets/M3M-GCP-colmap-native-quarter-v1/formal_inputs/{SCENE}/train"
FORMAL_MANIFEST = f"/root/autodl-tmp/datasets/M3M-GCP-colmap-native-quarter-v1/formal_inputs/{SCENE}/NATIVE_QUARTER_INPUT_MANIFEST.json"
EVALUATION_CAMERA_ROOT = f"/root/autodl-tmp/datasets/M3M-GCP-100K-evaluation-camera-root-v1/{SCENE}"
EVALUATION_CAMERA_EVIDENCE = f"/root/autodl-tmp/runs/m3m-gcp-native-quarter/preparation/{SCENE}/evaluation-camera-root-v1.json"
EVALUATION_CAMERA_EVIDENCE_SHA = "6b31e460ba80b17e85ac284c55165bfbc6c6b3a85411ad88e785ed8fe6645aac"
ENV = "/root/autodl-tmp/envs/m3m-gcp-native-quarter/{method}/py310-torch2.7.1-cu128-v1/bin/python"
PACKET_ENV = {
    "3dgs_original": "/root/autodl-tmp/envs/m3m-gcp-native-quarter/3dgs-original/py310-torch2.7.1-cu128-v1/bin/python",
    "citygaussian_v2": "/root/autodl-tmp/envs/m3m-gcp-native-quarter/citygaussian_v2/eval-py310-torch2.7.1-cu128-v1/bin/python",
    "citygs_x": "/root/autodl-tmp/envs/m3m-gcp-native-quarter/citygs_x/eval-py310-torch2.7.1-cu128-v1/bin/python",
}
DA2_ROOT = "/root/autodl-tmp/build/m3m-gcp-native-quarter/depth-anything-v2/a561b849ebae10a6f5ef49e26c83cbbcd36c71bf/runtime-v1"
MOGE_WEIGHT = "/root/autodl-tmp/models/m3m-gcp-native-quarter/metrogs/moge-2-vitl-normal/b135031bae30b5ac2ae141a0e68717795ce38340/model.pt"
PI3_WEIGHT = "/root/autodl-tmp/models/m3m-gcp-native-quarter/metrogs/pi3/ae722e7039287d0c8fde9f11f197f804f44b510c/model.safetensors"
METHOD_INPUT_EVIDENCE = "/root/autodl-tmp/runs/m3m-gcp-native-quarter/preparation/gcp_100000_20260610/per-method-inputs-v2.json"
METHOD_INPUT_EVIDENCE_SHA = "080a1ef97ab5caadca70420d6e34b57681d793f874b2a43511480fbc09b30ab1"
FORMAL_CAMERAS_SHA = "6669584ba1ba326cf5b372b878a5abf182f8cfe0bfe0845da3a0c4f7aed8fe5e"
FORMAL_IMAGES_SHA = "dfc1a5d17532aebb3da670598635baea5c8fbf999592b6b567504251a01c9f72"
CITY_IMAGES_SHA = "825fb831886d96bb50d7d25f110909d6938a4a80afb29d3f047873d03d18dbe5"
FULL_POINTS_SHA = "09fc811f32558a11a47bada7393bf7bce2585cbe68eb4872ffce72025b0fc9aa"
METRO_POINTS_SHA = "fcbb06d2b52770281b2b2c88f6d1a9deb5b2435e4578e63ca77bb8f197c37e7f"
INITIAL_PLY_SHA = "9f653655a34c05007e58f339afec593136bd857a56b13a612c79d8e53913364e"
EMPTY_POINTS_SHA = "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc"
FORMAL_RUN_ROOT = f"/root/autodl-tmp/runs/m3m-gcp-native-quarter/formal-100k/{SCENE}"
PACKET_SCRATCH_ROOT = f"{FORMAL_RUN_ROOT}/packet-scratch"


REUSE_3DGS: dict[str, Any] = {
    "input_class": "rgb_colmap_only",
    "commit": "2eee0e26d2d5fd00ec462df47752223952f6bf4e",
    "tree": "5eee127dfc0942bf83d9fdd72328e03ec0cbf6c4",
    "renderer": "9fb339c043f893c80599b2d2a55dbe06f320e6a88085fd46d0eebff157edca81",
    "budget": {"type": "reuse_frozen_30k_model_no_retrain", "value": 30000},
    "dataset": FORMAL,
    "reuse_model": True,
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(payload: dict[str, Any]) -> str:
    clean = {key: value for key, value in payload.items() if key != "canonical_sha256"}
    raw = json.dumps(clean, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


METHODS: dict[str, dict[str, Any]] = {
    "2dgs": {
        "input_class": "rgb_colmap_only", "commit": "335ad612f2e783a4e57b9cbc4d1e167bd599fc98", "tree": "ad1da88f43447bde046712835db70e271816282e", "sub": "official-train", "status": "", "files": {"train.py": "ebc9f6e9def7e70733af913d07e00f350ae2e3fdd5cbdcf3e9abc37ccd347196"}, "renderer": "857f6a97f5d57933e97eab9ab5d7d5b50f345ae21a7c4c8dc78137d453c1437b", "budget": {"type": "iterations", "value": 30000},
        "command": [ENV.format(method="2dgs"), "-B", "{source_root}/train.py", "-s", "{dataset_root}", "-m", "{run_root}/model", "--resolution", "1", "--iterations", "30000", "--depth_ratio", "0", "--test_iterations", "7000", "30000", "--save_iterations", "7000", "30000", "--quiet"],
    },
    "pgsr": {
        "input_class": "rgb_colmap_only", "commit": "de24f1a38b350387e8d8fe381b2cd70c1ae946e7", "tree": "8504a351b4a7938ef0b15647c1e5efb01e7ea013", "sub": "train-loader-compat-v1", "status": " M scene/dataset_readers.py", "files": {"train.py": "02dbe7852e54f31f9ea31e3eb18807eb9a47a75eb1c8995186fd7d6bed34fa0a", "scene/dataset_readers.py": "2c51048f9358f2a86af863eeda3897f11fea10dc7d8234fd6e6953ec38872281"}, "renderer": "1cbebcf6239a74c9d54aa6c28067f6a837461badb548f22f957d50e24e23acc7", "budget": {"type": "iterations", "value": 30000},
        "command": [ENV.format(method="pgsr"), "-B", "{source_root}/train.py", "-s", "{dataset_root}", "-m", "{run_root}/model", "--resolution", "1", "--iterations", "30000", "--test_iterations", "7000", "30000", "--save_iterations", "7000", "30000", "--max_abs_split_points", "0", "--opacity_cull_threshold", "0.05", "--multi_view_max_dis", "1000000000", "--quiet"],
    },
    "rade_gs": {
        "input_class": "rgb_colmap_only", "commit": "d72f20792005ae1d6555a82aa2d15345f247604e", "tree": "e37a9f1bfec5b593371402d19fb5259cbcb6efa1", "sub": "official-train", "status": "", "files": {"train.py": "10dd1be2b912091db9f7d15afcf3ee088d9e550d6149cc536d7292a595da5328"}, "renderer": "7eb4afd438d360c880fd626acd01ef34886daddb110865454260bb955dc6cf3a", "budget": {"type": "iterations", "value": 30000},
        "command": [ENV.format(method="rade_gs"), "-B", "{source_root}/train.py", "-s", "{dataset_root}", "-m", "{run_root}/model", "--resolution", "1", "--iterations", "30000", "--test_iterations", "7000", "30000", "--save_iterations", "7000", "30000", "--checkpoint_iterations", "15000", "--use_decoupled_appearance", "3", "--multi_view_max_dis", "1000000000", "--quiet"],
    },
    "qgs": {
        "input_class": "rgb_colmap_only", "commit": "74d05c945e99fcaef7afe5a8831903be71ad9b55", "tree": "c20af6da770b9ecc9c4e1730b40671ea63ec1419", "sub": "official-train", "status": "", "files": {"train.py": "dabf257d7b37a3671731d804794d21b6a833f0f9f23101ae518d91d1ea00aeed"}, "renderer": "0d785d7c3442c86d0121dfc2782b577c199443073cb35122888f26221e3d4061", "budget": {"type": "iterations", "value": 30000}, "dataset": f"/root/autodl-tmp/datasets/M3M-GCP-qgs-formal-train-view-v1/{SCENE}",
        "command": [ENV.format(method="qgs"), "-B", "{repo}/code/gcp/run_qgs_training.py", "--qgs_repo", "{source_root}", "--conf_path", "{run_root}/formal_training_config.yaml", "--save_iterations", "30000", "--quiet"],
        "materialization": """case_name: gcp_100000_20260610\nroot_dir: {dataset_root}\nmodel_path: {run_root}/model\npipeline:\n  depth_ratio: 0.1\n  reciprocal_z: false\n  occlusion_awared_denom: false\ndataset:\n  use_alpha: false\n  eval: false\n  downsample: 1.0\n  ncc_scale: 1.0\n  type: all\n  kernel_size: -1\n  undistortion: true\n  white_background: false\n  multi_view_max_dis: 1000000000.0\n  multi_view_min_dis: 0.01\n  multi_view_max_angle: 30\ngs_model:\n  sigma: 3.0\n  sh_degree: 3\n  multi_view_num: 8\n  use_app: false\noptimizer:\n  iterations: 30000\n  position_lr_max_steps: 30000\n  percent_dense: 0.0005\n  densify_grad_threshold: 0.3\n  densification_interval: 100\n  opacity_reset_interval: 3000\n  densify_from_iter: 500\n  densify_until_iter: 15000\n  normal_from_iter: 7000\n  multi_view_weight_from_iter: 7000\n  multi_view_weight_until_iter: 30000\n  dist_from_iter: 3000\n  multi_view_ncc_weight: 0.5\n  multi_view_geo_weight: 0.03\n  curvature_clamp_threshold: -5.0\n  lambda_dist: 1000.0\n  lambda_normal: 0.05\n""",
    },
    "gsprior": {
        "input_class": "rgb_colmap_only", "commit": "dcb7c89fb6b60f068b440de45d064ecc7fbcba55", "tree": "779073585e88b85217d522d0ab345365346cd17f", "sub": "train-compat-v1", "status": " M scene/dataset_readers.py\n M train.py", "files": {"train.py": "44e434741b68075d7e3f92f2c71f471cab2c66e4d3bd129261e7701da03274f1", "scene/dataset_readers.py": "f0fadf22869091cafa7cec4cc14eb3e295f83c58bf5ad22c2557c3b3b9cee247"}, "renderer": "f6ceac12c6c04caa0ab56d2edbb62626ea8ab8a2777526a7c7cd14e9404046ee", "budget": {"type": "iterations", "value": 40000}, "dataset": f"/root/autodl-tmp/datasets/M3M-GCP-gsprior-normalized-v1/{SCENE}/train",
        "command": [ENV.format(method="gsprior"), "-B", "{source_root}/train.py", "-s", "{dataset_root}", "-m", "{run_root}/model", "--resolution", "1", "--iterations", "40000", "--test_iterations", "20000", "30000", "40000", "--save_iterations", "20000", "30000", "40000", "--checkpoint_iterations", "20000", "30000", "40000", "--quiet"],
        "prior_command": [ENV.format(method="gsprior"), "-B", "{repo}/code/gcp/materialize_gsprior_normalized_scene.py", "--source_scene", FORMAL, "--reference_train_scene", FORMAL, "--output_scene", "{dataset_root}", "--manifest", "{prior_root}/normalization_manifest.json", "--scene_id", SCENE, "--role", "train"],
    },
    "sof": {
        "input_class": "rgb_colmap_only", "commit": "b9eb4170c843014f5f96d54924976161bd675469", "tree": "d5ece75b8255c5dd6abf97482ddbf34d20dca707", "sub": "official-train", "status": "", "files": {"train.py": "1c58c67fca2017fb5a1b44c5d679a12dd9f176054b583aef33698ef44dbb7ddc"}, "renderer": "014f19cec9a8c2d44b619565a9abac70a5e97950469eec47b87b26aa96978371", "budget": {"type": "iterations", "value": 30000},
        "command": [ENV.format(method="sof"), "-B", "{source_root}/train.py", "-s", "{dataset_root}", "-m", "{run_root}/model", "-r", "1", "--iterations", "30000", "--test_iterations", "30000", "--save_iterations", "30000", "--lambda_distortion", "100", "--far_plane", "100", "--splatting_config", "{source_root}/configs/hierarchical.json", "--use_decoupled_appearance", "--detach_alpha", "False", "--quiet"],
    },
    "citygaussian_v2": {
        "input_class": "rgb_colmap_external_geometry_prior", "commit": "e84c7c8774dd11d3f4189be3488e1220afa20a86", "tree": "be088977358cb36bac000caec396eff3758c74b2", "sub": "official-train", "status": "", "files": {}, "renderer": "0d7946ee84f4c7f990d0f97e563973002f84a75cd06c8cb27e7b8de3d8bca4ab", "budget": {"type": "official_matrixcity_aerial_4x4", "coarse_steps": 30000, "fine_steps": 60000}, "dataset": f"/root/autodl-tmp/datasets/M3M-GCP-citygaussian-v2-prior-v4/{SCENE}",
        "command": [ENV.format(method="citygaussian_v2"), "-B", "{repo}/code/gcp/run_citygaussian_v2_100k_pipeline.py", "--repo", "{source_root}", "--python", ENV.format(method="citygaussian_v2"), "--dataset", "{dataset_root}", "--output_root", "{run_root}/pipeline", "--mode", "formal", "--coarse_steps", "30000", "--fine_steps", "60000"],
        "prior_command": [ENV.format(method="citygaussian_v2"), "-B", "{repo}/code/gcp/prepare_citygaussian_v2_100k_depth_prior.py", "--city_repo", "{source_root}", "--python", ENV.format(method="citygaussian_v2"), "--dataset", "{dataset_root}", "--formal_input_manifest", FORMAL_MANIFEST, "--da2_root", DA2_ROOT, "--manifest_output", "{prior_root}/depth_prior_v1.json", "--expected_city_commit", "e84c7c8774dd11d3f4189be3488e1220afa20a86", "--expected_da2_commit", "a561b849ebae10a6f5ef49e26c83cbbcd36c71bf", "--expected_da2_weight_sha256", "a7ea19fa0ed99244e67b624c72b8580b7e9553043245905be58796a608eb9345", "--input_size", "518", "--point_max_error", "1.5"],
        "prior_external": {f"{DA2_ROOT}/checkpoints/depth_anything_v2_vitl.pth": "a7ea19fa0ed99244e67b624c72b8580b7e9553043245905be58796a608eb9345", f"{DA2_ROOT}/run.py": "b8beab6341314bc864cfc0283e0e303f55571af4e2e456c3d3afbc4aae67249a"},
    },
    "citygs_x": {
        "input_class": "rgb_colmap_external_geometry_prior", "commit": "27617f2486505e3b6fe75345edf7c2b11161bc2a", "tree": "f8b1b5148c1f47420ab698fd069bdb78acf901ab", "sub": "train-runtime-v1", "status": " M scene/dataset_readers.py\n M utils/camera_utils.py", "files": {"train.py": "ba9658bc5bcbc7b0a7620aca776cc2736a045cccd5267276f21ed4e6557d4591", "scene/dataset_readers.py": "3d75bbeb16d47f7c078ba2d09a8612dbe4eb6139a865b1d15e1302aa8167a82c", "utils/camera_utils.py": "9326e6571685177543e34c903823b207b75258e96489d9398b08672637f5c9e3"}, "renderer": "65c3a18f6f88379b2a2add0775ac1e4d00b4c67e56d2347c4a3a47c088b04d43", "budget": {"type": "iterations", "value": 100000}, "dataset": f"/root/autodl-tmp/datasets/M3M-GCP-citygs-x-prior-v1/MatrixCity-{SCENE}/train/block_all",
        "command": [ENV.format(method="citygs_x"), "-B", "{repo}/code/gcp/run_citygs_x_100k_training.py", "--repo", "{source_root}", "--python", ENV.format(method="citygs_x"), "--dataset", "{dataset_root}", "--model_path", "{run_root}/model", "--prior_manifest", "{prior_root}/depth_and_multiview_prior_v1.json", "--pytorch3d_compat", "{repo}/compat/citygs_x/pytorch3d_transforms_minimal_v1", "--mode", "formal", "--iterations", "100000"],
        "prior_command": [ENV.format(method="citygs_x"), "-B", "{repo}/code/gcp/prepare_citygs_x_100k_depth_prior.py", "--city_repo", "{source_root}", "--python", ENV.format(method="citygs_x"), "--dataset", "{dataset_root}", "--formal_input_manifest", FORMAL_MANIFEST, "--da2_root", DA2_ROOT, "--pytorch3d_compat", "{repo}/compat/citygs_x/pytorch3d_transforms_minimal_v1", "--manifest_output", "{prior_root}/depth_and_multiview_prior_v1.json", "--expected_city_commit", "27617f2486505e3b6fe75345edf7c2b11161bc2a", "--expected_camera_utils_sha256", "9326e6571685177543e34c903823b207b75258e96489d9398b08672637f5c9e3", "--expected_dataset_readers_sha256", "3d75bbeb16d47f7c078ba2d09a8612dbe4eb6139a865b1d15e1302aa8167a82c", "--expected_da2_commit", "a561b849ebae10a6f5ef49e26c83cbbcd36c71bf", "--expected_da2_run_sha256", "b8beab6341314bc864cfc0283e0e303f55571af4e2e456c3d3afbc4aae67249a", "--expected_da2_weight_sha256", "a7ea19fa0ed99244e67b624c72b8580b7e9553043245905be58796a608eb9345", "--expected_cameras_sha256", FORMAL_CAMERAS_SHA, "--expected_images_sha256", CITY_IMAGES_SHA, "--expected_points3d_sha256", FULL_POINTS_SHA, "--input_size", "518", "--resolution", "1", "--pixel_thred", "1", "--multi_view_num", "8", "--multi_view_max_angle", "15", "--multi_view_min_dis", "0.01", "--multi_view_max_dis", "25"],
        "prior_external": {f"{DA2_ROOT}/checkpoints/depth_anything_v2_vitl.pth": "a7ea19fa0ed99244e67b624c72b8580b7e9553043245905be58796a608eb9345", f"{DA2_ROOT}/run.py": "b8beab6341314bc864cfc0283e0e303f55571af4e2e456c3d3afbc4aae67249a"},
    },
    "metrogs": {
        "input_class": "rgb_colmap_external_geometry_prior", "commit": "8cf9ac13c0c34b65c1a935d181c4634909e60f3f", "tree": "7e92b13095cf4a031d7eb8593e10616db154abbf", "sub": "official-train", "status": "", "files": {}, "renderer": "20aa83dc12ab0f3087b57c46597845cf2750beacc3b36ba119dd49a6bed78b82", "budget": {"type": "effective_image_iterations", "value": 150000, "optimizer_steps": 37500}, "dataset": f"/root/autodl-tmp/datasets/M3M-GCP-metrogs-prior-v2/{SCENE}",
        "command": [ENV.format(method="metrogs"), "-B", "{repo}/code/gcp/run_metrogs_100k_training.py", "--repo", "{source_root}", "--python", ENV.format(method="metrogs"), "--dataset", "{dataset_root}", "--model_path", "{run_root}/model", "--prior_manifest", "{prior_root}/training_priors.json", "--prior_pass_marker", "{prior_root}/TRAINING_PRIORS_PASS", "--additional_ply", "{prior_root}/additional_points/metrogs_pi3_merged.ply", "--mode", "formal", "--iterations", "150000"],
        "prior_source": {"sub": "prior-runtime-v1", "status": " M utils/get_mask_depth_scales.py", "files": {"utils/get_mask_depth_scales.py": "b48d68d1355140af9b37caf3fee55d135ae5b59277eb43eb5d245d0e60a67106"}},
        "prior_command": [ENV.format(method="metrogs"), "-B", "{repo}/code/gcp/prepare_metrogs_100k_training_priors.py", "--repo", "{source_root}", "--python", ENV.format(method="metrogs"), "--dataset", "{dataset_root}", "--formal_input_manifest", FORMAL_MANIFEST, "--moge_path", "{source_root}/utils/MoGe", "--moge_weight", MOGE_WEIGHT, "--pi3_weight", PI3_WEIGHT, "--compatibility_patch", "{repo}/patches/metrogs/numpy_bool_compat_get_mask_depth_scales_8cf9ac1_v1.patch", "--colmap_io", "{repo}/code/colmap/utils/read_write_model.py", "--subset_track_closure_tool", "{repo}/code/gcp/materialize_colmap_subset_track_closure.py", "--additional_ply", "{prior_root}/additional_points/metrogs_pi3_merged.ply", "--evidence_output", "{prior_root}/training_priors.json", "--pass_marker", "{prior_root}/TRAINING_PRIORS_PASS", "--expected_repo_commit", "8cf9ac13c0c34b65c1a935d181c4634909e60f3f", "--expected_repo_tree", "7e92b13095cf4a031d7eb8593e10616db154abbf", "--expected_runtime_status", " M utils/get_mask_depth_scales.py", "--expected_moge_sha256", "280741fd09bc3f403ccff9967784c2a391b52d2c0742ae3efdb21d9f90cc1a01", "--expected_pi3_sha256", "33580e4702ac671558aedeab1148fd08118f7ce45bdbeb99f3e3cf340062875d", "--expected_cameras_sha256", FORMAL_CAMERAS_SHA, "--expected_images_sha256", CITY_IMAGES_SHA, "--expected_points3d_sha256", METRO_POINTS_SHA, "--split_num", "4", "--multi_view_max_dis", "1.5"],
        "prior_external": {MOGE_WEIGHT: "280741fd09bc3f403ccff9967784c2a391b52d2c0742ae3efdb21d9f90cc1a01", PI3_WEIGHT: "33580e4702ac671558aedeab1148fd08118f7ce45bdbeb99f3e3cf340062875d"},
    },
}


EVAL_BINDINGS: dict[str, dict[str, Any]] = {
    "3dgs_original": {
        "root": "/root/autodl-tmp/worktrees/m3m-gcp-native-quarter/3dgs-original/2eee0e26d2d5fd00ec462df47752223952f6bf4e/eval-adapter-v1",
        "commit": REUSE_3DGS["commit"], "tree": REUSE_3DGS["tree"],
        "status": " M gaussian_renderer/__init__.py\n M submodules/diff-gaussian-rasterization",
        "files": {
            "gaussian_renderer/__init__.py": "1996207560bf23bf0df671bc1e801b5264ba08119e146e83c4a92493520dc062",
            "submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py": "61432710e2f150459d5f71b68529140d9ececa0d785d3ff91d24381696abfd89",
            "submodules/diff-gaussian-rasterization/rasterize_points.cu": "cc7a9fd3b0d809f28da952f54d36bba20d78906b54bfe3934013f953ed87f8ba",
            "submodules/diff-gaussian-rasterization/rasterize_points.h": "7fdf17df5880f2819551e70a162937abff526b7e6b0337ccb8d6fe184f18c3f2",
            "submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu": "af77279263d7cb39f09d7b2d5d089f396e14dbe3726a1a6af38cb4722f521ca2",
            "submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.h": "14e991a00e1be7958d3375675e49c8cba81ab67b634932ae1b552e9fda31db2e",
            "submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer.h": "ae6cb41d90c60f8d9d8e2c8401a9e777e0c32167b02a84436acd131c3fa5465b",
            "submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu": "d008f690da5cc27d1f74333df521b66532b6c56fe96e3fef29ce164e964291b2",
        },
    },
    "2dgs": {
        "root": "/root/autodl-tmp/worktrees/m3m-gcp-native-quarter/2dgs/335ad612f2e783a4e57b9cbc4d1e167bd599fc98/eval-adapter-v1",
        "commit": METHODS["2dgs"]["commit"], "tree": METHODS["2dgs"]["tree"],
        "status": " M gaussian_renderer/__init__.py\n M submodules/diff-surfel-rasterization",
        "files": {
            "gaussian_renderer/__init__.py": "3a27f0bbce9974210619a6aba1b0a19ec99649c95e525f324fe339c515be9c63",
            "submodules/diff-surfel-rasterization/cuda_rasterizer/auxiliary.h": "9695c1cd0f40047e155849c91aff14809150e811d89ab6106ce986c12e2ad7c4",
            "submodules/diff-surfel-rasterization/cuda_rasterizer/forward.cu": "b575f23158590bd68bba0b6bfb6caab7d28c06d1fe66f9efe9cf7f212cbb5841",
            "submodules/diff-surfel-rasterization/rasterize_points.cu": "1d2d33e760a2bae17637413d048c8768d5befe0cc47f009592f48d1d689fcdbd",
        },
    },
    "pgsr": {
        "root": "/root/autodl-tmp/worktrees/m3m-gcp-native-quarter/pgsr/de24f1a38b350387e8d8fe381b2cd70c1ae946e7/eval-adapter-v1",
        "commit": METHODS["pgsr"]["commit"], "tree": METHODS["pgsr"]["tree"],
        "status": " M gaussian_renderer/__init__.py\n M scene/dataset_readers.py\n M submodules/diff-plane-rasterization/cuda_rasterizer/config.h",
        "files": {
            "gaussian_renderer/__init__.py": "833a4ab591accf0365f3f4d4a12633beb82d175c516b605833919ba07e4b71e8",
            "scene/dataset_readers.py": "2c51048f9358f2a86af863eeda3897f11fea10dc7d8234fd6e6953ec38872281",
            "submodules/diff-plane-rasterization/cuda_rasterizer/config.h": "1f4268c9bdbc4bd9ec5c5d4a0460ae3a9e72e178e8e64e2a2438be035391b5d3",
        },
    },
    "rade_gs": {
        "root": "/root/autodl-tmp/worktrees/m3m-gcp-native-quarter/rade_gs/d72f20792005ae1d6555a82aa2d15345f247604e/eval-adapter-v1",
        "commit": METHODS["rade_gs"]["commit"], "tree": METHODS["rade_gs"]["tree"],
        "status": " M gaussian_renderer/__init__.py\n M submodules/diff-gaussian-rasterization/cuda_rasterizer/config.h\n M submodules/diff-gaussian-rasterization/cuda_rasterizer/render_forward.cu",
        "files": {
            "gaussian_renderer/__init__.py": "a0e9b8fd9970017822a817a949a4e3a9fa8c1fe68f71b1d1e67cde9cab833443",
            "submodules/diff-gaussian-rasterization/cuda_rasterizer/config.h": "4bdbf430b6d3c32090dfb05f3227c28b682d8b96736b3a5214e31ebae917b083",
            "submodules/diff-gaussian-rasterization/cuda_rasterizer/render_forward.cu": "597896677107334092ca6885dbecf109704f18d31aee03a2ea6bf152ac4db066",
        },
    },
    "qgs": {
        "root": "/root/autodl-tmp/worktrees/m3m-gcp-native-quarter/qgs/74d05c945e99fcaef7afe5a8831903be71ad9b55/eval-adapter-v1",
        "commit": METHODS["qgs"]["commit"], "tree": METHODS["qgs"]["tree"],
        "status": " M gaussian_renderer/__init__.py\n M submodules/diff-quadratic-rasterization/cuda_rasterizer/forward.cu\n M submodules/diff-quadratic-rasterization/cuda_rasterizer/stopthepop_QGS/resorted_render.cuh",
        "files": {
            "gaussian_renderer/__init__.py": "9b78e52eb0383669930477bf3b482b130089da04416d9968ee27cf69d95e1adf",
            "submodules/diff-quadratic-rasterization/cuda_rasterizer/forward.cu": "5a8574185204566504b56c3684655308637238b327b2a4e21f028c333fa325f5",
            "submodules/diff-quadratic-rasterization/cuda_rasterizer/stopthepop_QGS/resorted_render.cuh": "ea522c4dbc60bdd55f9863808778071bf7f909db52f0e5317acb3a979b1585b3",
        },
    },
    "gsprior": {
        "root": "/root/autodl-tmp/worktrees/m3m-gcp-native-quarter/gsprior/dcb7c89fb6b60f068b440de45d064ecc7fbcba55/eval-adapter-v1",
        "commit": METHODS["gsprior"]["commit"], "tree": METHODS["gsprior"]["tree"],
        "status": " M gaussian_renderer/__init__.py\n M scene/dataset_readers.py",
        "files": {
            "gaussian_renderer/__init__.py": "fb3dedbebd887c711639271e29ed7bf8e0796a85a36b811a56b6f98148880832",
            "scene/dataset_readers.py": "65af79d4de7c06197aab32a2c5ef35de4a7e1c1d43d7597aaedc0df88830b7d9",
        },
    },
    "sof": {
        "root": "/root/autodl-tmp/worktrees/m3m-gcp-native-quarter/sof/b9eb4170c843014f5f96d54924976161bd675469/eval-adapter-v1",
        "commit": METHODS["sof"]["commit"], "tree": METHODS["sof"]["tree"],
        "status": " M gaussian_renderer/__init__.py",
        "files": {"gaussian_renderer/__init__.py": "ae0f668bf4b8f49e78d7aa58caf640662de7451957655d7ff9d733fe3b0f0227"},
    },
    "citygaussian_v2": {
        "root": "/root/autodl-tmp/worktrees/m3m-gcp-native-quarter/citygaussian_v2/e84c7c8774dd11d3f4189be3488e1220afa20a86/eval-adapter-v1",
        "commit": METHODS["citygaussian_v2"]["commit"], "tree": METHODS["citygaussian_v2"]["tree"],
        "status": " M internal/renderers/sep_depth_trim_2dgs_renderer.py",
        "files": {"internal/renderers/sep_depth_trim_2dgs_renderer.py": "e1cf2d09c90d425d503770fe3342739787921b339b181dba19da6ac6e1d21209"},
    },
    "citygs_x": {
        "root": "/root/autodl-tmp/worktrees/m3m-gcp-native-quarter/citygs_x/27617f2486505e3b6fe75345edf7c2b11161bc2a/eval-adapter-v1",
        "commit": METHODS["citygs_x"]["commit"], "tree": METHODS["citygs_x"]["tree"],
        "status": " M gaussian_renderer/__init__.py\n M submodule_cityx/diff-gaussian-rasterization/cuda_rasterizer/config.h\n M submodule_cityx/diff-gaussian-rasterization/cuda_rasterizer/forward.cu",
        "files": {
            "gaussian_renderer/__init__.py": "7b802445a10de10f78c49ad9d96b62ae744abdd6400c87ae5197495f1d20c10c",
            "submodule_cityx/diff-gaussian-rasterization/cuda_rasterizer/config.h": "c1226b64d32ba3218224b6537bf25669e507fd82a8006054976cac4ac5be0dc0",
            "submodule_cityx/diff-gaussian-rasterization/cuda_rasterizer/forward.cu": "6c67049e65df40e43828eed3a0df5ee76b94949e6d9065eee0f73e26b108297d",
        },
    },
    "metrogs": {
        "root": "/root/autodl-tmp/worktrees/m3m-gcp-native-quarter/metrogs/8cf9ac13c0c34b65c1a935d181c4634909e60f3f/eval-adapter-v1",
        "commit": METHODS["metrogs"]["commit"], "tree": METHODS["metrogs"]["tree"],
        "status": " M internal/renderers/metrogs_renderer.py\n M submodules/dist-2dgs/cuda_rasterizer/auxiliary.h\n M submodules/dist-2dgs/cuda_rasterizer/forward.cu\n M submodules/dist-2dgs/rasterize_points.cu",
        "files": {
            "internal/renderers/metrogs_renderer.py": "5b2efbf811745cc0bac90f74d9a6fe2b804bdee05c7d2dfc9b560a838dcc90b4",
            "submodules/dist-2dgs/cuda_rasterizer/auxiliary.h": "6c05765817acd1c09a9c42aa4a6f810c373b0fdd8ff486b9da935c19fdc8f456",
            "submodules/dist-2dgs/cuda_rasterizer/forward.cu": "42b969a67cddbba76ff82ffff4b64a72c27f465067b21bb213242ab26b623917",
            "submodules/dist-2dgs/rasterize_points.cu": "8484fc8f1df69cddc5433cf1bf0ccd8c162ee865f9c723c12c27d849d3532bbf",
        },
    },
}


PACKET_EXPORTERS = {
    "3dgs_original": "code/gcp/export_gaussian_depth_maps.py",
    "2dgs": "code/gcp/export_gaussian_depth_maps.py",
    "pgsr": "code/gcp/export_gaussian_depth_maps.py",
    "rade_gs": "code/gcp/export_gaussian_depth_maps.py",
    "qgs": "code/gcp/export_qgs_depth_maps.py",
    "gsprior": "code/gcp/export_gaussian_depth_maps.py",
    "sof": "code/gcp/export_gaussian_depth_maps.py",
    "citygaussian_v2": "code/gcp/export_citygaussian_v2_depth_maps.py",
    "citygs_x": "code/gcp/export_citygs_x_depth_maps.py",
    "metrogs": "code/gcp/export_metrogs_depth_maps.py",
}


PACKET_REPORTS = {
    "2dgs": ("/root/autodl-tmp/build/m3m-gcp-native-quarter/2dgs/335ad612f2e783a4e57b9cbc4d1e167bd599fc98/qualification-v1/logs/raw_moment_cuda_conformance.json", "000e9597b673f86e3e014ff9c911f767821a9be90d15ba50fd1482285bf4d831"),
    "pgsr": ("/root/autodl-tmp/build/m3m-gcp-native-quarter/pgsr/de24f1a38b350387e8d8fe381b2cd70c1ae946e7/qualification-v1/logs/pgsr_raw_moment_cuda_conformance_v1.json", "d89771bc52aeda7e3cef40e07bce1da5c9bbd3149071e27708619b02c1bbf2a9"),
    "rade_gs": ("/root/autodl-tmp/build/m3m-gcp-native-quarter/rade_gs/d72f20792005ae1d6555a82aa2d15345f247604e/qualification-v1/rade_gs_raw_moment_cuda_conformance_v1.json", "e06e25640d6e753b8ac202e7d63f3b665fb18b4d2960a9f1fe98a5f4da2847c1"),
    "qgs": ("/root/autodl-tmp/build/m3m-gcp-native-quarter/qgs/74d05c945e99fcaef7afe5a8831903be71ad9b55/qualification-v1/raw_moment_conformance.json", "5c04974691588a8c9314ebb9bce0d325467ae95db642a03675feb4a90451ee73"),
    "gsprior": ("/root/autodl-tmp/build/m3m-gcp-native-quarter/gsprior/dcb7c89fb6b60f068b440de45d064ecc7fbcba55/qualification-v1/raw_moment_conformance.json", "4c6ba32b1471977d06a797f772d50ad7eccd8e7baab9236f92682d6ddce7d298"),
    "sof": ("/root/autodl-tmp/build/m3m-gcp-native-quarter/sof/b9eb4170c843014f5f96d54924976161bd675469/qualification-v1/raw_moment_conformance.json", "2e40d3b132cb58be69e92b4d66edfd65dc7396c138a3f2595e4090e6b0bb5aee"),
    "citygaussian_v2": ("/root/autodl-tmp/build/m3m-gcp-native-quarter/citygaussian_v2/e84c7c8774dd11d3f4189be3488e1220afa20a86/qualification-v1/raw_moment_conformance.json", "3c6f45d76fbe364ccdc8938a691206369687ecb8983af7f2d2b83911c6d5e5af"),
    "citygs_x": ("/root/autodl-tmp/build/m3m-gcp-native-quarter/citygs_x/27617f2486505e3b6fe75345edf7c2b11161bc2a/qualification-v1/raw_moment_conformance.json", "0d4f31c737f16c5da354564bd11b8008e372d2b0f4c9c26c7b71d67f4bbb61b4"),
    "metrogs": ("/root/autodl-tmp/build/m3m-gcp-native-quarter/metrogs/8cf9ac13c0c34b65c1a935d181c4634909e60f3f/qualification-v4/raw_moment_conformance.json", "76b32b0a5b7e45d1f88cb7d85d7906ec2975aaee61b7cb1693579bc42d69fcee"),
}


PACKET_PATCHES = {
    "3dgs_original": ["patches/3dgs_original/native_quarter_raw_moment_renderer_2eee0e26_v1.patch", "patches/3dgs_original/native_quarter_raw_moment_rasterizer_59f5f77_v1.patch"],
    "2dgs": ["patches/2dgs/native_quarter_raw_moments_renderer_335ad612_v1.patch", "patches/2dgs/native_quarter_raw_moments_rasterizer_e0ed020_v1.patch"],
    "pgsr": ["patches/pgsr/native_quarter_raw_moments_renderer_de24f1a_v1.patch", "patches/pgsr/native_quarter_raw_moments_rasterizer_de24f1a_v1.patch"],
    "rade_gs": ["patches/rade_gs/native_quarter_raw_moments_renderer_d72f207_v1.patch", "patches/rade_gs/native_quarter_raw_moments_rasterizer_d72f207_v1.patch"],
    "qgs": ["patches/qgs/native_quarter_raw_moments_renderer_74d05c9_v1.patch", "patches/qgs/native_quarter_raw_moments_rasterizer_74d05c9_v1.patch"],
    "gsprior": ["patches/gsprior/native_quarter_raw_moments_renderer_dcb7c89_v1.patch"],
    "sof": ["patches/sof/native_quarter_raw_moments_renderer_b9eb417_v1.patch"],
    "citygaussian_v2": ["patches/citygaussian_v2/native_quarter_raw_moments_renderer_e84c7c8_v1.patch", "patches/citygaussian_v2/native_quarter_raw_moments_rasterizer_9eefc03_v1.patch"],
    "citygs_x": ["patches/citygs_x/native_quarter_raw_moments_renderer_27617f2_v1.patch", "patches/citygs_x/native_quarter_raw_moments_rasterizer_27617f2_v1.patch"],
    "metrogs": ["patches/metrogs/native_quarter_raw_moments_renderer_8cf9ac1_v1.patch", "patches/metrogs/native_quarter_raw_moments_rasterizer_8cf9ac1_v1.patch"],
}


WRAPPERS = {
    "qgs": ["code/gcp/run_qgs_training.py"],
    "citygaussian_v2": ["code/gcp/run_citygaussian_v2_100k_pipeline.py", "code/gcp/prepare_citygaussian_v2_100k_depth_prior.py"],
    "citygs_x": [
        "code/gcp/run_citygs_x_100k_training.py",
        "code/gcp/prepare_citygs_x_100k_depth_prior.py",
        "compat/citygs_x/pytorch3d_transforms_minimal_v1/pytorch3d/__init__.py",
        "compat/citygs_x/pytorch3d_transforms_minimal_v1/pytorch3d/transforms/__init__.py",
    ],
    "metrogs": ["code/gcp/run_metrogs_100k_training.py", "code/gcp/prepare_metrogs_100k_training_priors.py"],
    "gsprior": ["code/gcp/materialize_gsprior_normalized_scene.py"],
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    all_methods = {"3dgs_original": REUSE_3DGS, **METHODS}
    for method, spec in all_methods.items():
        base_name = "3dgs" if method == "3dgs_original" else method
        base_path = ROOT / "configs" / f"m3m_gcp_native_quarter_{base_name}_3k_recipe_v1.json"
        base_recipe = json.loads(base_path.read_text(encoding="utf-8"))
        adapter_relative = str(base_recipe.get("evaluation_adapter", {}).get("config", ""))
        adapter_path = ROOT / adapter_relative
        if not adapter_path.is_file() or sha(adapter_path) != spec["renderer"]:
            raise RuntimeError(f"{method}: renderer-adapter config identity mismatch")
        source_root = (
            EVAL_BINDINGS[method]["root"]
            if method == "3dgs_original"
            else f"/root/autodl-tmp/worktrees/m3m-gcp-native-quarter/{method}/{spec['commit']}/{spec['sub']}"
        )
        dataset = spec.get("dataset", FORMAL)
        dependency_paths = [
            *WRAPPERS.get(method, []),
            "code/gcp/run_m3m_gcp_100k_packet_export.py",
            PACKET_EXPORTERS[method],
            *PACKET_PATCHES[method],
        ]
        dependency_paths.append("code/gcp/materialize_m3m_gcp_100k_method_inputs.py")
        dependency_paths.append(
            "code/gcp/materialize_m3m_gcp_100k_evaluation_camera_root.py"
        )
        if method in {"citygaussian_v2", "citygs_x"}:
            dependency_paths.append(
                "code/gcp/materialize_colmap_train_track_compatibility_streaming.py"
            )
        if method == "metrogs":
            dependency_paths.append(
                "code/gcp/filter_colmap_model_to_frozen_train_streaming.py"
            )
        dependencies = {
            path: sha(ROOT / path)
            for path in dependency_paths
        }
        phase_commands: dict[str, list[str]] = {}
        phase_roots: dict[str, dict[str, str]] = {}
        source_bindings: dict[str, dict[str, Any]] = {}
        if not spec.get("reuse_model"):
            phase_commands["training"] = spec["command"]
            phase_roots["training"] = {"dataset_root": dataset, "prior_root": dataset}
            source_bindings["training"] = {
                "root": source_root,
                "commit": spec["commit"],
                "tree": spec["tree"],
                "required_status": spec["status"],
                "required_files_sha256": spec["files"],
            }
        packet_python = PACKET_ENV.get(method, ENV.format(method=method))
        phase_commands["packet"] = [
            packet_python,
            "-B",
            "{repo}/code/gcp/run_m3m_gcp_100k_packet_export.py",
            "--method-id",
            method,
            "--benchmark-repo",
            "{repo}",
            "--evaluation-repo",
            "{source_root}",
            "--training-run-root",
            "{run_root}",
            "--dataset-root",
            "{dataset_root}",
            "--prior-root",
            "{prior_root}",
            "--camera-root",
            EVALUATION_CAMERA_ROOT,
            "--train-allowlist",
            "{repo}/configs/m3m_gcp_lidar_train_view_allowlists_v1/gcp_100000_20260610.csv",
            "--packet-set-root",
            "{packet_set_root}",
        ]
        phase_roots["packet"] = {"dataset_root": dataset, "prior_root": dataset}
        evaluation = EVAL_BINDINGS[method]
        source_bindings["packet"] = {
            "root": evaluation["root"],
            "commit": evaluation["commit"],
            "tree": evaluation["tree"],
            "required_status": evaluation["status"],
            "required_files_sha256": evaluation["files"],
        }
        phase_external: dict[str, dict[str, str]] = {}
        if method in PACKET_REPORTS:
            report_path, report_sha = PACKET_REPORTS[method]
            phase_external["packet"] = {report_path: report_sha}
        if "prior_command" in spec:
            prior_source = spec.get("prior_source", {})
            prior_root = (
                f"/root/autodl-tmp/worktrees/m3m-gcp-native-quarter/{method}/"
                f"{spec['commit']}/{prior_source.get('sub', spec['sub'])}"
            )
            phase_commands["prior"] = spec["prior_command"]
            phase_roots["prior"] = {"dataset_root": dataset, "prior_root": dataset}
            source_bindings["prior"] = {
                "root": prior_root,
                "commit": spec["commit"],
                "tree": spec["tree"],
                "required_status": prior_source.get("status", spec["status"]),
                "required_files_sha256": prior_source.get("files", spec["files"]),
            }
            phase_external["prior"] = spec.get("prior_external", {})
        payload: dict[str, Any] = {
            "schema": "m3m_gcp_native_quarter_100k_execution_recipe_v1",
            "status": "REVIEW_CANDIDATE_NOT_EXECUTION_AUTHORIZED",
            "method_id": method,
            "scene": SCENE,
            "seed": 0,
            "input_class": spec["input_class"],
            "authorized_run_root": (
                "/root/autodl-tmp/runs/m3m-gcp-native-quarter/3dgs-original/gcp_100000_20260610/seed0-30k-20260810T175634Z"
                if method == "3dgs_original"
                else f"{FORMAL_RUN_ROOT}/{method}/seed0-v1"
            ),
            "authorized_evidence_root": f"{FORMAL_RUN_ROOT}/{method}/evidence",
            "authorized_packet_set_root": f"{PACKET_SCRATCH_ROOT}/{method}",
            "authorized_packet_state": f"{PACKET_SCRATCH_ROOT}/ACTIVE_PACKET_STATE.json",
            "base_3k_recipe_path": base_path.relative_to(ROOT).as_posix(),
            "base_3k_recipe_sha256": sha(base_path),
            "formal_input_manifest": {"path": FORMAL_MANIFEST, "file_sha256": "c2cf9e951d95fee12a28d942e95c5c420df55bc364738b3f8737fed1c78bef3d", "canonical_sha256": "5b4fe34743310bd2225feb2dd236200606be933002fec19d2c9ecb9f3ba6769d", "train_views": 2196, "heldout_views": 314},
            "input_validation_phases": [
                phase for phase in phase_commands if phase in {"training", "packet"}
            ],
            "budget": spec["budget"],
            "source_bindings": source_bindings,
            "phase_roots": phase_roots,
            "phase_commands": phase_commands,
            "phase_external_required_files_sha256": phase_external,
            "materializations": {},
            "benchmark_required_files_sha256": dependencies,
            "renderer_adapter_sha256": spec["renderer"],
            "renderer_adapter_path": adapter_relative,
            "progress_monitor": {
                "regex": "(?i)(?:iteration|iter|step|progress)[^0-9]{0,32}([0-9]+)",
                "unit": "optimizer_steps" if method == "metrogs" else "iterations_or_stage_steps",
            },
            "progress_monitors": {
                "training": {
                    "regex": "(?i)(?:iteration|iter|step|progress)[^0-9]{0,32}([0-9]+)",
                    "unit": "optimizer_steps" if method == "metrogs" else "iterations_or_stage_steps",
                },
                "packet": {
                    "regex": "(?i)(?:exporting[^\\r\\n]{0,200}?)([0-9]+)/2196",
                    "unit": "train_views_exported",
                },
            },
            "retry_policy": "a guard rejection before child creation is not an attempt and may be relaunched; once the child starts every exit is final",
            "result_selection_from_metrics": "FORBIDDEN",
        }
        if spec.get("reuse_model"):
            payload["reuse_model_binding"] = {
                "run_root": "/root/autodl-tmp/runs/m3m-gcp-native-quarter/3dgs-original/gcp_100000_20260610/seed0-30k-20260810T175634Z",
                "point_cloud_relative_path": "model/point_cloud/iteration_30000/point_cloud.ply",
                "point_cloud_bytes": 2340432588,
                "point_cloud_sha256": "8d92360186d268d0e20a0e328122e8c2679cddd0c2d539c27a918ee4c972e1f5",
                "retrain_allowed": False,
            }
        if method in {"citygaussian_v2", "citygs_x"}:
            input_profile = "city_train_records_with_full_all_image_sfm_points"
            sparse_hashes = {
                "cameras.bin": FORMAL_CAMERAS_SHA,
                "images.bin": CITY_IMAGES_SHA,
                "points3D.bin": FULL_POINTS_SHA,
                "points3D.ply": INITIAL_PLY_SHA,
            }
        elif method == "metrogs":
            input_profile = "metrogs_reciprocal_train_track_closure_after_all_image_sfm"
            sparse_hashes = {
                "cameras.bin": FORMAL_CAMERAS_SHA,
                "images.bin": CITY_IMAGES_SHA,
                "points3D.bin": METRO_POINTS_SHA,
                "points3D.ply": INITIAL_PLY_SHA,
            }
        else:
            input_profile = "exact_formal_train_view_from_shared_all_image_sfm"
            sparse_hashes = {
                "cameras.bin": FORMAL_CAMERAS_SHA,
                "images.bin": FORMAL_IMAGES_SHA,
                "points3D.ply": INITIAL_PLY_SHA,
            }
        prepared_dataset = FORMAL if method == "gsprior" else dataset
        payload["prepared_method_input_binding"] = {
            "evidence_path": METHOD_INPUT_EVIDENCE,
            "evidence_sha256": METHOD_INPUT_EVIDENCE_SHA,
            "dataset_root": prepared_dataset,
            "input_profile": input_profile,
            "sparse_sha256": sparse_hashes,
            "all_image_sfm_precedes_train_test_split": True,
        }
        payload["evaluation_camera_root_binding"] = {
            "root": EVALUATION_CAMERA_ROOT,
            "evidence_path": EVALUATION_CAMERA_EVIDENCE,
            "evidence_sha256": EVALUATION_CAMERA_EVIDENCE_SHA,
            "status_required": "PASS_EVALUATION_CAMERA_ROOT_NO_TRAINING_NO_PRIOR_NO_EVALUATION",
            "view_count": 2196,
            "sparse_sha256": {
                "cameras.bin": FORMAL_CAMERAS_SHA,
                "images.bin": FORMAL_IMAGES_SHA,
                "points3D.bin": EMPTY_POINTS_SHA,
                "points3D.ply": INITIAL_PLY_SHA,
            },
            "points3d_bin_point_count": 0,
            "purpose": "pose-only all-train evaluation camera loader; never training or prior input",
        }
        if "materialization" in spec:
            payload["materializations"]["training"] = [{"relative_path": "formal_training_config.yaml", "content": spec["materialization"]}]
        payload["canonical_sha256"] = canonical(payload)
        path = OUT / f"{method}.json"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            )
        rows.append({"method_id": method, "path": path.relative_to(ROOT).as_posix(), "sha256": sha(path), "canonical_sha256": payload["canonical_sha256"]})
    manifest = {"schema": "m3m_gcp_native_quarter_100k_recipe_manifest_v1", "scene": SCENE, "seed": 0, "status": "REVIEW_CANDIDATE_NOT_EXECUTION_AUTHORIZED", "method_order": list(all_methods), "recipes": rows}
    manifest["canonical_sha256"] = canonical(manifest)
    output = ROOT / "configs" / "m3m_gcp_native_quarter_100k_recipe_manifest_v1.json"
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
