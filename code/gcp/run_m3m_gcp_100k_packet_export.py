#!/usr/bin/env python3
"""Dispatch one frozen 100K all-train-view packet export.

This wrapper is deliberately evaluation-only.  It validates the selected
training output and frozen adapter evidence, constructs the method-specific
exporter command, then replaces itself with that exporter so the outer
resource guard observes the actual GPU process.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


SCENE = "gcp_100000_20260610"
PROTOCOL_ID = "m3m_gcp_native_quarter_geometry_v2"
SOURCE_RELEASE_SHA256 = (
    "513f8999fe4b110f15bcbecad7932895781cee755ee9ccd7a14ff10298546d75"
)
IMAGE_DOMAIN = "colmap_4_0_4_image_undistorter_pinhole_max_1414"
PIXEL_CONVENTION = "zero_based_pixel_centers"
EXPECTED_TRAIN_VIEWS = 2196
FORMAL_TRAIN_ROOT = Path(
    "/root/autodl-tmp/datasets/M3M-GCP-colmap-native-quarter-v1/formal_inputs/"
    f"{SCENE}/train"
)
EVALUATION_CAMERA_ROOT = Path(
    f"/root/autodl-tmp/datasets/M3M-GCP-100K-evaluation-camera-root-v1/{SCENE}"
)
EVALUATION_CAMERA_MANIFEST_SHA256 = (
    "6b31e460ba80b17e85ac284c55165bfbc6c6b3a85411ad88e785ed8fe6645aac"
)
EVALUATION_CAMERA_SPARSE_SHA256 = {
    "cameras.bin": "6669584ba1ba326cf5b372b878a5abf182f8cfe0bfe0845da3a0c4f7aed8fe5e",
    "images.bin": "dfc1a5d17532aebb3da670598635baea5c8fbf999592b6b567504251a01c9f72",
    "points3D.bin": "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc",
    "points3D.ply": "9f653655a34c05007e58f339afec593136bd857a56b13a612c79d8e53913364e",
}
EXPECTED_3DGS_PLY_SHA256 = (
    "8d92360186d268d0e20a0e328122e8c2679cddd0c2d539c27a918ee4c972e1f5"
)
CITYGS_X_PYTORCH3D_COMPAT_RELATIVE = Path(
    "compat/citygs_x/pytorch3d_transforms_minimal_v1"
)


ADAPTERS: dict[str, dict[str, Any]] = {
    "3dgs_original": {
        "kind": "gaussian",
        "iteration": 30000,
        "rasterizer": "",
        "report": "{benchmark_repo}/docs/protocol_evidence/3dgs_native_quarter_adapter_gpu_preflight_v1.json",
        "report_sha256": "a605a9c045647d92bc34bcc30e05e5a9632038220245b36078c14cbcdacf898c",
        "patches": [
            "patches/3dgs_original/native_quarter_raw_moment_renderer_2eee0e26_v1.patch",
            "patches/3dgs_original/native_quarter_raw_moment_rasterizer_59f5f77_v1.patch",
        ],
    },
    "2dgs": {
        "kind": "gaussian",
        "iteration": 30000,
        "rasterizer": "submodules/diff-surfel-rasterization",
        "report": "/root/autodl-tmp/build/m3m-gcp-native-quarter/2dgs/335ad612f2e783a4e57b9cbc4d1e167bd599fc98/qualification-v1/logs/raw_moment_cuda_conformance.json",
        "report_sha256": "000e9597b673f86e3e014ff9c911f767821a9be90d15ba50fd1482285bf4d831",
        "patches": [
            "patches/2dgs/native_quarter_raw_moments_renderer_335ad612_v1.patch",
            "patches/2dgs/native_quarter_raw_moments_rasterizer_e0ed020_v1.patch",
        ],
    },
    "pgsr": {
        "kind": "gaussian",
        "iteration": 30000,
        "rasterizer": "submodules/diff-plane-rasterization",
        "report": "/root/autodl-tmp/build/m3m-gcp-native-quarter/pgsr/de24f1a38b350387e8d8fe381b2cd70c1ae946e7/qualification-v1/logs/pgsr_raw_moment_cuda_conformance_v1.json",
        "report_sha256": "d89771bc52aeda7e3cef40e07bce1da5c9bbd3149071e27708619b02c1bbf2a9",
        "patches": [
            "patches/pgsr/native_quarter_raw_moments_renderer_de24f1a_v1.patch",
            "patches/pgsr/native_quarter_raw_moments_rasterizer_de24f1a_v1.patch",
        ],
    },
    "rade_gs": {
        "kind": "gaussian",
        "iteration": 30000,
        "rasterizer": "submodules/diff-gaussian-rasterization",
        "report": "/root/autodl-tmp/build/m3m-gcp-native-quarter/rade_gs/d72f20792005ae1d6555a82aa2d15345f247604e/qualification-v1/rade_gs_raw_moment_cuda_conformance_v1.json",
        "report_sha256": "e06e25640d6e753b8ac202e7d63f3b665fb18b4d2960a9f1fe98a5f4da2847c1",
        "patches": [
            "patches/rade_gs/native_quarter_raw_moments_renderer_d72f207_v1.patch",
            "patches/rade_gs/native_quarter_raw_moments_rasterizer_d72f207_v1.patch",
        ],
        "extra": [
            "--kernel_size", "0.0", "--use_decoupled_appearance", "0",
            "--multi_view_max_dis", "1000000000",
        ],
    },
    "qgs": {
        "kind": "qgs",
        "iteration": 30000,
        "rasterizer": "submodules/diff-quadratic-rasterization",
        "report": "/root/autodl-tmp/build/m3m-gcp-native-quarter/qgs/74d05c945e99fcaef7afe5a8831903be71ad9b55/qualification-v1/raw_moment_conformance.json",
        "report_sha256": "5c04974691588a8c9314ebb9bce0d325467ae95db642a03675feb4a90451ee73",
        "patches": [
            "patches/qgs/native_quarter_raw_moments_renderer_74d05c9_v1.patch",
            "patches/qgs/native_quarter_raw_moments_rasterizer_74d05c9_v1.patch",
        ],
    },
    "gsprior": {
        "kind": "gaussian",
        "iteration": 40000,
        "rasterizer": "submodules/diff-plane-rasterization",
        "report": "/root/autodl-tmp/build/m3m-gcp-native-quarter/gsprior/dcb7c89fb6b60f068b440de45d064ecc7fbcba55/qualification-v1/raw_moment_conformance.json",
        "report_sha256": "4c6ba32b1471977d06a797f772d50ad7eccd8e7baab9236f92682d6ddce7d298",
        "patches": [
            "patches/gsprior/native_quarter_raw_moments_renderer_dcb7c89_v1.patch",
        ],
    },
    "sof": {
        "kind": "gaussian",
        "iteration": 30000,
        "rasterizer": "submodules/diff-gaussian-rasterization",
        "report": "/root/autodl-tmp/build/m3m-gcp-native-quarter/sof/b9eb4170c843014f5f96d54924976161bd675469/qualification-v1/raw_moment_conformance.json",
        "report_sha256": "2e40d3b132cb58be69e92b4d66edfd65dc7396c138a3f2595e4090e6b0bb5aee",
        "patches": [
            "patches/sof/native_quarter_raw_moments_renderer_b9eb417_v1.patch",
        ],
    },
    "citygaussian_v2": {
        "kind": "citygaussian_v2",
        "iteration": 60000,
        "rasterizer": "/root/autodl-tmp/build/m3m-gcp-native-quarter/citygaussian_v2/e84c7c8774dd11d3f4189be3488e1220afa20a86/qualification-v1/eval-sources/diff-surfel-city-9eefc03",
        "report": "/root/autodl-tmp/build/m3m-gcp-native-quarter/citygaussian_v2/e84c7c8774dd11d3f4189be3488e1220afa20a86/qualification-v1/raw_moment_conformance.json",
        "report_sha256": "3c6f45d76fbe364ccdc8938a691206369687ecb8983af7f2d2b83911c6d5e5af",
        "patches": [
            "patches/citygaussian_v2/native_quarter_raw_moments_renderer_e84c7c8_v1.patch",
            "patches/citygaussian_v2/native_quarter_raw_moments_rasterizer_9eefc03_v1.patch",
        ],
    },
    "citygs_x": {
        "kind": "citygs_x",
        "iteration": 100000,
        "rasterizer": "submodule_cityx/diff-gaussian-rasterization",
        "report": "/root/autodl-tmp/build/m3m-gcp-native-quarter/citygs_x/27617f2486505e3b6fe75345edf7c2b11161bc2a/qualification-v1/raw_moment_conformance.json",
        "report_sha256": "0d4f31c737f16c5da354564bd11b8008e372d2b0f4c9c26c7b71d67f4bbb61b4",
        "patches": [
            "patches/citygs_x/native_quarter_raw_moments_renderer_27617f2_v1.patch",
            "patches/citygs_x/native_quarter_raw_moments_rasterizer_27617f2_v1.patch",
        ],
    },
    "metrogs": {
        "kind": "metrogs",
        "iteration": 150000,
        "rasterizer": "submodules/dist-2dgs",
        "report": "/root/autodl-tmp/build/m3m-gcp-native-quarter/metrogs/8cf9ac13c0c34b65c1a935d181c4634909e60f3f/qualification-v4/raw_moment_conformance.json",
        "report_sha256": "76b32b0a5b7e45d1f88cb7d85d7906ec2975aaee61b7cb1693579bc42d69fcee",
        "patches": [
            "patches/metrogs/native_quarter_raw_moments_renderer_8cf9ac1_v1.patch",
            "patches/metrogs/native_quarter_raw_moments_rasterizer_8cf9ac1_v1.patch",
        ],
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, expected_sha256: str | None = None) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if expected_sha256 is not None and sha256(path) != expected_sha256:
        raise RuntimeError(f"file SHA mismatch: {path}")
    return path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(require_file(path).read_text(encoding="utf-8"))


def verify_allowlist(path: Path) -> None:
    with require_file(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    names = [str(row.get("image_name", "")).strip() for row in rows]
    if len(names) != EXPECTED_TRAIN_VIEWS or len(set(names)) != EXPECTED_TRAIN_VIEWS:
        raise RuntimeError("100K train allowlist must contain 2196 unique image names")
    if any(not name for name in names):
        raise RuntimeError("100K train allowlist contains an empty image name")


def canonical_sha256(payload: dict[str, Any]) -> str:
    body = dict(payload)
    body.pop("canonical_sha256", None)
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_camera_root(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path != EVALUATION_CAMERA_ROOT.resolve():
        raise RuntimeError("packet camera root differs from the frozen evaluation-only root")
    manifest_path = path / "EVALUATION_CAMERA_ROOT_MANIFEST.json"
    require_file(manifest_path, EVALUATION_CAMERA_MANIFEST_SHA256)
    manifest = read_json(manifest_path)
    output = manifest.get("output", {})
    if (
        manifest.get("schema") != "m3m_gcp_100k_evaluation_camera_root_v1"
        or manifest.get("status")
        != "PASS_EVALUATION_CAMERA_ROOT_NO_TRAINING_NO_PRIOR_NO_EVALUATION"
        or manifest.get("scene") != SCENE
        or manifest.get("canonical_sha256") != canonical_sha256(manifest)
        or output.get("root") != str(path)
        or output.get("view_count") != EXPECTED_TRAIN_VIEWS
        or output.get("points3d_bin_point_count") != 0
        or manifest.get("truth_boundary", {}).get("heldout_rgb_present") is not False
        or manifest.get("truth_boundary", {}).get("gcp_or_lidar_used") is not False
    ):
        raise RuntimeError("evaluation camera-root manifest identity mismatch")
    image_root = path / "images"
    if (
        not image_root.is_symlink()
        or image_root.resolve() != (FORMAL_TRAIN_ROOT / "images").resolve()
        or len([item for item in image_root.iterdir() if item.is_file()])
        != EXPECTED_TRAIN_VIEWS
    ):
        raise RuntimeError("evaluation camera-root RGB boundary mismatch")
    sparse = path / "sparse" / "0"
    manifest_files = output.get("files", {})
    if set(manifest_files) != set(EVALUATION_CAMERA_SPARSE_SHA256):
        raise RuntimeError("evaluation camera-root sparse inventory mismatch")
    for name, expected_sha in EVALUATION_CAMERA_SPARSE_SHA256.items():
        file_path = sparse / name
        row = manifest_files.get(name, {})
        if (
            not file_path.is_file()
            or sha256(file_path) != expected_sha
            or row.get("sha256") != expected_sha
            or row.get("bytes") != file_path.stat().st_size
        ):
            raise RuntimeError(f"evaluation camera-root identity mismatch: {name}")
    if (sparse / "points3D.bin").read_bytes() != (0).to_bytes(8, "little"):
        raise RuntimeError("evaluation camera-root compatibility points3D.bin is not empty")
    return manifest


def verify_summary_file(path: Path, *, method_id: str, status: str) -> dict[str, Any]:
    payload = read_json(path)
    if payload.get("method_id") != method_id or payload.get("status") != status:
        raise RuntimeError(f"training summary identity/status mismatch: {path}")
    if payload.get("mode") != "formal":
        raise RuntimeError(f"training summary is not formal: {path}")
    return payload


def verify_checkpoint(args: argparse.Namespace, spec: dict[str, Any]) -> tuple[Path, float]:
    run_root = args.training_run_root
    method_id = args.method_id
    iteration = int(spec["iteration"])
    if method_id == "citygaussian_v2":
        summary = verify_summary_file(
            run_root / "pipeline" / "pipeline_summary.json",
            method_id=method_id,
            status="PIPELINE_PASS",
        )
        if int(summary.get("fine_steps", -1)) != iteration:
            raise RuntimeError("CityGaussianV2 fine-step budget mismatch")
        row = summary.get("merged_checkpoint", {})
        checkpoint = require_file(Path(str(row.get("path", ""))), str(row.get("sha256", "")))
        return checkpoint, 1.0
    if method_id == "citygs_x":
        summary = verify_summary_file(
            run_root / "model" / "training_wrapper_summary.json",
            method_id=method_id,
            status="TRAINING_PASS",
        )
        if int(summary.get("iterations", -1)) != iteration:
            raise RuntimeError("CityGS-X iteration budget mismatch")
        model_path = Path(str(summary.get("model_path", ""))).resolve()
        checkpoint = Path(str(summary.get("checkpoint", {}).get("path", ""))).resolve()
        if model_path != (run_root / "model").resolve() or not checkpoint.is_dir():
            raise RuntimeError("CityGS-X model/checkpoint path mismatch")
        point_cloud = checkpoint / str(summary.get("checkpoint", {}).get("point_cloud_file", ""))
        require_file(point_cloud, str(summary.get("checkpoint", {}).get("point_cloud_sha256", "")))
        return model_path, 1.0
    if method_id == "metrogs":
        summary = verify_summary_file(
            run_root / "model" / "training_wrapper_summary.json",
            method_id=method_id,
            status="TRAINING_PASS",
        )
        if int(summary.get("effective_iterations", -1)) != iteration:
            raise RuntimeError("MetroGS effective-iteration budget mismatch")
        row = summary.get("checkpoint", {})
        checkpoint = require_file(Path(str(row.get("merged_path", ""))), str(row.get("merged_sha256", "")))
        return checkpoint, 1.0

    model_path = (run_root / "model").resolve()
    point_cloud = model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    if method_id == "3dgs_original":
        require_file(point_cloud, EXPECTED_3DGS_PLY_SHA256)
    else:
        require_file(point_cloud)
    scale = 1.0
    if method_id == "gsprior":
        manifest = read_json(args.prior_root / "normalization_manifest.json")
        if manifest.get("status") != "PASS" or manifest.get("scene_id") != SCENE:
            raise RuntimeError("GSPrior normalization manifest identity/status mismatch")
        scale = float(
            manifest.get("transform", {}).get(
                "original_colmap_units_per_normalized_unit", 0.0
            )
        )
        if not scale > 0.0:
            raise RuntimeError("invalid GSPrior camera-z unit scale")
    return model_path, scale


def common_tail(args: argparse.Namespace, report: Path, report_sha: str) -> list[str]:
    return [
        "--image_list_csv", str(args.train_allowlist),
        "--image_name_column", "image_name",
        "--image_domain", IMAGE_DOMAIN,
        "--pixel_coordinate_convention", PIXEL_CONVENTION,
        "--protocol_id", PROTOCOL_ID,
        "--protocol_scene", SCENE,
        "--source_data_release_root_digest_sha256", SOURCE_RELEASE_SHA256,
        "--camera_z_unit_contract", "frozen_colmap_model_camera_z_units",
        "--adapter_conformance_status", "PASS",
        "--adapter_conformance_report", str(report),
        "--adapter_conformance_report_sha256", report_sha,
    ]


def build_command(args: argparse.Namespace) -> list[str]:
    spec = ADAPTERS[args.method_id]
    benchmark_repo = args.benchmark_repo
    evaluation_repo = args.evaluation_repo
    report = Path(str(spec["report"]).format(benchmark_repo=benchmark_repo))
    require_file(report, str(spec["report_sha256"]))
    patches = [require_file(benchmark_repo / relative) for relative in spec["patches"]]
    model_or_checkpoint, scale = verify_checkpoint(args, spec)
    exporter_name = {
        "gaussian": "export_gaussian_depth_maps.py",
        "qgs": "export_qgs_depth_maps.py",
        "citygaussian_v2": "export_citygaussian_v2_depth_maps.py",
        "citygs_x": "export_citygs_x_depth_maps.py",
        "metrogs": "export_metrogs_depth_maps.py",
    }[str(spec["kind"])]
    exporter = require_file(benchmark_repo / "code" / "gcp" / exporter_name)
    iteration = str(spec["iteration"])
    packet_root = args.packet_set_root
    manifest = packet_root / "depth_export_manifest.json"
    mapping = packet_root / "depth_map_index.csv"

    if spec["kind"] == "gaussian":
        source_path = args.dataset_root if args.method_id == "gsprior" else args.camera_root
        command = [
            sys.executable, "-B", str(exporter),
            "--train_repo", str(evaluation_repo),
            "-s", str(source_path),
            "-m", str(model_or_checkpoint),
            "-r", "1",
            "--iteration", iteration,
            "--camera_sets", "train",
            "--depth_output_dir", str(packet_root),
            "--manifest_path", str(manifest),
            "--mapping_csv", str(mapping),
            "--raw_camera_z_to_protocol_scale", format(scale, ".17g"),
        ]
        rasterizer = str(spec.get("rasterizer", ""))
        if rasterizer:
            command.extend(["--rasterizer_repo", rasterizer])
        command.extend(spec.get("extra", []))
        command.extend(common_tail(args, report, str(spec["report_sha256"])))
        command.extend(["--renderer_adapter_patch", str(patches[0])])
        if len(patches) > 1:
            command.extend(["--rasterizer_adapter_patch", str(patches[1])])
        command.append("--quiet")
        return command

    common_without_explicit_report_sha = common_tail(
        args, report, str(spec["report_sha256"])
    )
    # The three dedicated exporters hash the report themselves and therefore
    # do not expose the generic explicit-report-SHA CLI option.
    index = common_without_explicit_report_sha.index("--adapter_conformance_report_sha256")
    del common_without_explicit_report_sha[index : index + 2]

    if spec["kind"] == "qgs":
        return [
            sys.executable, "-B", str(exporter),
            "--train_repo", str(evaluation_repo),
            "--qgs_config_path", str(args.training_run_root / "formal_training_config.yaml"),
            "--camera_source_path", str(args.camera_root),
            "--rasterizer_repo", str(spec["rasterizer"]),
            "--iteration", iteration,
            "--camera_sets", "train",
            "--depth_output_dir", str(packet_root),
            "--manifest_path", str(manifest),
            "--mapping_csv", str(mapping),
            *common_tail(args, report, str(spec["report_sha256"])),
            "--renderer_adapter_patch", str(patches[0]),
            "--rasterizer_adapter_patch", str(patches[1]),
            "--quiet",
        ]

    if spec["kind"] == "citygaussian_v2":
        rasterizer = Path(str(spec["rasterizer"])).resolve()
        if subprocess.check_output(["git", "-C", str(rasterizer), "rev-parse", "HEAD"], text=True).strip() != "9eefc03858a30bb3e5f98eccc56f077420ee2aaf":
            raise RuntimeError("CityGaussianV2 evaluation rasterizer commit mismatch")
        expected_rasterizer_files = {
            "cuda_rasterizer/auxiliary.h": "3262de5f888dc42d9b82dabbd122adc200058e19d3e7ed3eee83ed0d7931c1cf",
            "cuda_rasterizer/forward.cu": "a32600371e52f8b38ee3c98218d8870a822d514f99df8ea9e9e9178b879087ed",
            "rasterize_points.cu": "eb19ef9e6e93598f8f2ee8b8c16a3348c5d4235088dc2272adef772aada56a9d",
        }
        expected_status = (
            " M cuda_rasterizer/auxiliary.h\n"
            " M cuda_rasterizer/forward.cu\n"
            " M rasterize_points.cu"
        )
        actual_status = subprocess.check_output(
            ["git", "-C", str(rasterizer), "status", "--porcelain=v1", "--untracked-files=no"],
            text=True,
        ).rstrip("\n")
        if actual_status != expected_status:
            raise RuntimeError("CityGaussianV2 evaluation rasterizer status mismatch")
        for relative, expected_sha in expected_rasterizer_files.items():
            require_file(rasterizer / relative, expected_sha)
        command = [
            sys.executable, "-B", str(exporter), "--repo", str(evaluation_repo),
            "--checkpoint", str(model_or_checkpoint), "--rasterizer_repo", str(rasterizer),
        ]
    elif spec["kind"] == "citygs_x":
        command = [
            sys.executable, "-B", str(exporter), "--repo", str(evaluation_repo),
            "--model_path", str(model_or_checkpoint),
            "--rasterizer_repo", str(evaluation_repo / str(spec["rasterizer"])),
            "--pytorch3d_compat",
            str(args.benchmark_repo / CITYGS_X_PYTORCH3D_COMPAT_RELATIVE),
        ]
    else:
        command = [
            sys.executable, "-B", str(exporter), "--repo", str(evaluation_repo),
            "--checkpoint", str(model_or_checkpoint),
            "--rasterizer_repo", str(evaluation_repo / str(spec["rasterizer"])),
        ]
    command.extend([
        "--camera_root", str(args.camera_root),
        "--depth_output_dir", str(packet_root),
        "--manifest_path", str(manifest),
        "--mapping_csv", str(mapping),
        "--iteration", iteration,
        "--raw_camera_z_to_protocol_scale", "1.0",
        *common_without_explicit_report_sha,
    ])
    for patch in patches:
        command.extend(["--adapter_patch", str(patch)])
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-id", choices=tuple(ADAPTERS), required=True)
    parser.add_argument("--benchmark-repo", type=Path, required=True)
    parser.add_argument("--evaluation-repo", type=Path, required=True)
    parser.add_argument("--training-run-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--prior-root", type=Path, required=True)
    parser.add_argument("--camera-root", type=Path, required=True)
    parser.add_argument("--train-allowlist", type=Path, required=True)
    parser.add_argument("--packet-set-root", type=Path, required=True)
    args = parser.parse_args()
    for field in (
        "benchmark_repo", "evaluation_repo", "training_run_root", "dataset_root",
        "prior_root", "camera_root", "train_allowlist", "packet_set_root",
    ):
        setattr(args, field, getattr(args, field).expanduser().resolve())
    return args


def main() -> int:
    args = parse_args()
    if args.packet_set_root.exists():
        raise FileExistsError(f"packet set already exists: {args.packet_set_root}")
    if not (args.benchmark_repo / ".git").exists():
        raise FileNotFoundError("benchmark checkout is missing .git")
    if not (args.evaluation_repo / ".git").exists():
        raise FileNotFoundError("evaluation adapter checkout is missing .git")
    verify_allowlist(args.train_allowlist)
    verify_camera_root(args.camera_root)
    command = build_command(args)
    print(json.dumps({"status": "EXEC_PACKET_EXPORT", "method_id": args.method_id, "argv": command}), flush=True)
    os.execv(sys.executable, command)
    raise AssertionError("os.execv returned unexpectedly")


if __name__ == "__main__":
    raise SystemExit(main())
