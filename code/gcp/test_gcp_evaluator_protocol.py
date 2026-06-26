from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_gaussian_gcp_geometry import (  # noqa: E402
    camera_z_from_depth_value,
    file_sha256,
    load_depth_index,
    load_depth_manifest,
    load_release_config,
    reject_unsupported_depth_semantics,
    robust_depth_patch,
    run_roundtrip_unit_test,
    validate_annotation_rows_scene,
    validate_metric_packet_npz,
    verify_release_files,
)
from fit_gcp_sim3 import apply_similarity, fit_similarity_umeyama  # noqa: E402
from metric_depth_packet import (  # noqa: E402
    DEFAULT_NUMERICAL_SUPPORT_FLOOR,
    DEFAULT_VARIANCE_CLAMP_TOLERANCE,
    METRIC_PACKET_MANIFEST_SCHEMA,
    METRIC_PACKET_SCHEMA,
    METRIC_PACKET_TENSOR_NAMES,
    PRIMARY_DEPTH_SEMANTICS,
    PRIMARY_DEPTH_TENSOR,
    derive_metric_depth_packet,
    file_sha256 as packet_file_sha256,
    variance_validation_manifest_fields,
)
from triangulate_gcp_points import pixel_to_normalized, project_point  # noqa: E402


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_camera(model: str = "PINHOLE", width: int = 400, height: int = 300) -> Any:
    if model == "SIMPLE_RADIAL":
        params = np.asarray([300.0, width / 2.0, height / 2.0, 0.03], dtype=np.float64)
    else:
        params = np.asarray([300.0, 300.0, width / 2.0, height / 2.0], dtype=np.float64)
    return SimpleNamespace(model=model, width=width, height=height, params=params)


def assert_close(actual: Any, expected: Any, tol: float = 1e-8) -> None:
    if not np.allclose(actual, expected, atol=tol, rtol=0):
        raise AssertionError(f"not close:\nactual={actual}\nexpected={expected}")


def test_sim3_recovery() -> dict[str, Any]:
    source = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [2.0, -1.0, 0.5],
        ],
        dtype=np.float64,
    )
    theta = math.radians(25.0)
    rotation = np.asarray(
        [
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    scale = 2.5
    translation = np.asarray([10.0, -4.0, 3.0])
    target = apply_similarity(source, scale, rotation, translation)
    got_scale, got_rotation, got_translation = fit_similarity_umeyama(source, target)
    recovered = apply_similarity(source, got_scale, got_rotation, got_translation)
    assert_close(recovered, target, 1e-8)
    return {"max_error": float(np.max(np.linalg.norm(recovered - target, axis=1)))}


def test_depth_roundtrip() -> dict[str, Any]:
    result = run_roundtrip_unit_test()
    if not result["passed"]:
        raise AssertionError(result)
    return {"case_count": result["case_count"], "max_abs_error": result["max_abs_error"]}


def test_unnormalized_inverse_depth_opacity() -> dict[str, Any]:
    z = 20.0
    opacities = [0.25, 0.5, 1.0]
    naive_recovered = [1.0 / (alpha / z) for alpha in opacities]
    if abs(naive_recovered[0] - naive_recovered[-1]) < 1e-9:
        raise AssertionError("unnormalized inverse depth should depend on opacity")
    try:
        camera_z_from_depth_value(1.0 / z, 0.0, 0.0, "alpha_weighted_unnormalized_inverse_camera_z")
    except ValueError:
        rejected = True
    else:
        rejected = False
    if not rejected:
        raise AssertionError("unsupported unnormalized semantics were not rejected")
    return {"naive_recovered_z": naive_recovered, "unsupported_semantics_rejected": rejected}


def test_ray_distance_patch_per_pixel() -> dict[str, Any]:
    camera = make_camera()
    depth = np.ones((9, 9), dtype=np.float64) * 10.0
    ok, stats = robust_depth_patch(
        depth=depth,
        camera=camera,
        u=200.0,
        v=150.0,
        depth_u=4.0,
        depth_v=4.0,
        depth_pixel_scale_x=1.0,
        depth_pixel_scale_y=1.0,
        patch_size=3,
        min_valid_ratio=0.5,
        min_depth=1e-6,
        depth_semantics="camera_z",
    )
    if not ok:
        raise AssertionError(stats)
    assert_close(stats["camera_z"], 10.0, 1e-12)
    return {"camera_z": stats["camera_z"], "camera_z_mad": stats["camera_z_mad"]}


def test_simple_radial_roundtrip() -> dict[str, Any]:
    camera = make_camera("SIMPLE_RADIAL")
    image = SimpleNamespace(qvec=np.asarray([1.0, 0.0, 0.0, 0.0]), tvec=np.zeros(3))
    xyz = np.asarray([1.2, -0.6, 20.0], dtype=np.float64)
    uv = project_point(camera, image, xyz)
    if uv is None:
        raise AssertionError("projection failed")
    x_norm, y_norm = pixel_to_normalized(camera, uv[0], uv[1])
    assert_close([x_norm, y_norm], [xyz[0] / xyz[2], xyz[1] / xyz[2]], 1e-5)
    return {"u": uv[0], "v": uv[1], "x_norm": x_norm, "y_norm": y_norm}


def prepare_release_fixture(root: Path) -> tuple[Path, dict[str, Path]]:
    rows = {
        "gcp_points_primary_usable_cgcs2000_cm108_v1.csv": [
            {
                "point_name": "P1",
                "cgcs2000_gk_cm108_e_m": "1",
                "cgcs2000_gk_cm108_n_m": "2",
                "cgcs2000_normal_height_m": "3",
            }
        ],
        "gcp_control_checkpoint_splits_v1.csv": [
            {"scene": "scene_a", "point_name": "P1", "role": "control"}
        ],
        "scene_metadata_gcp_benchmark_v1_1.csv": [
            {"scene": "scene_a", "formal_accuracy_claim_role": "formal_benchmark_scene"}
        ],
        "scene_a_gcp_annotations_final_good_nadir_v1.csv": [
            {"scene": "scene_a", "point_name": "P1", "image_name": "im.jpg", "u_px": "1", "v_px": "2"}
        ],
    }
    paths: dict[str, Path] = {}
    for name, data in rows.items():
        path = root / name
        write_csv(path, data, list(data[0].keys()))
        paths[name] = path
    files = [
        {"path": name, "bytes": paths[name].stat().st_size, "sha256": file_sha256(paths[name])}
        for name in sorted(paths)
    ]
    config = {
        "schema": "ms_gcp_3dgs_benchmark_release_config_v1_1",
        "release_id": "unit_test_release",
        "gcp_csv": "gcp_points_primary_usable_cgcs2000_cm108_v1.csv",
        "split_csv": "gcp_control_checkpoint_splits_v1.csv",
        "scene_metadata_csv": "scene_metadata_gcp_benchmark_v1_1.csv",
        "files": files,
    }
    config_path = root / "gcp_benchmark_release_v1_1.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config_path, paths


def test_release_hash_and_missing_file() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as d:
        config_path, paths = prepare_release_fixture(Path(d))
        config = load_release_config(config_path)
        verified = verify_release_files(config_path, config)
        paths["scene_a_gcp_annotations_final_good_nadir_v1.csv"].write_text("corrupted\n", encoding="utf-8")
        try:
            verify_release_files(config_path, config)
        except ValueError:
            hash_rejected = True
        else:
            hash_rejected = False
        paths["scene_a_gcp_annotations_final_good_nadir_v1.csv"].unlink()
        try:
            verify_release_files(config_path, config)
        except FileNotFoundError:
            missing_rejected = True
        else:
            missing_rejected = False
    if not hash_rejected or not missing_rejected:
        raise AssertionError("release hash/missing-file rejection failed")
    return {"initial_verified_count": len(verified), "hash_rejected": hash_rejected, "missing_rejected": missing_rejected}


def test_annotation_scene_mismatch() -> dict[str, Any]:
    try:
        validate_annotation_rows_scene([{"scene": "other"}], "scene_a")
    except ValueError:
        return {"mismatch_rejected": True}
    raise AssertionError("annotation scene mismatch was not rejected")


def test_depth_manifest_index_and_mismatch() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        depth = np.ones((4, 5), dtype=np.float32)
        depth_path = root / "im.npy"
        np.save(depth_path, depth)
        mapping = root / "depth_map_index.csv"
        write_csv(
            mapping,
            [{"image_name": "im.jpg", "depth_path": str(depth_path), "height": "4", "width": "5"}],
            ["image_name", "depth_path", "height", "width"],
        )
        manifest_path = root / "depth_manifest.json"
        manifest_path.write_text(
            json.dumps({"depth_semantics": "camera_z", "mapping_csv": str(mapping), "depth_output_dir": str(root)}),
            encoding="utf-8",
        )
        manifest = load_depth_manifest(manifest_path)
        index = load_depth_index(manifest_path, manifest)
        if index["im.jpg"]["height"] != 4 or index["im.jpg"]["width"] != 5:
            raise AssertionError(index)
        bad_manifest = root / "bad_depth_manifest.json"
        bad_manifest.write_text(json.dumps({"mapping_csv": str(mapping)}), encoding="utf-8")
        try:
            load_depth_manifest(bad_manifest)
        except ValueError:
            bad_rejected = True
        else:
            bad_rejected = False
        try:
            reject_unsupported_depth_semantics("alpha_weighted_unnormalized_inverse_camera_z")
        except ValueError:
            unsupported_rejected = True
        else:
            unsupported_rejected = False
    if not bad_rejected or not unsupported_rejected:
        raise AssertionError("depth manifest/semantics rejection failed")
    return {"index_count": len(index), "bad_manifest_rejected": bad_rejected, "unsupported_rejected": unsupported_rejected}


def write_metric_packet_fixture(path: Path) -> dict[str, np.ndarray]:
    packet = derive_metric_depth_packet(
        np.asarray([[0.4, 0.8]], dtype=np.float32),
        np.asarray([[4.0, 16.0]], dtype=np.float32),
        np.asarray([[40.0, 320.0]], dtype=np.float32),
        np.asarray([[0.04, 0.04]], dtype=np.float32),
    )
    np.savez_compressed(path, **packet)
    return packet


def test_metric_packet_manifest_and_npz_validation() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        packet_path = root / "im_metric_depth_packet.npz"
        write_metric_packet_fixture(packet_path)
        manifest_path = root / "metric_manifest.json"
        manifest = {
            "schema": METRIC_PACKET_MANIFEST_SCHEMA,
            "packet_schema": METRIC_PACKET_SCHEMA,
            "primary_depth_tensor": PRIMARY_DEPTH_TENSOR,
            "primary_depth_semantics": PRIMARY_DEPTH_SEMANTICS,
            "tensor_names": METRIC_PACKET_TENSOR_NAMES,
            "model_content_hash": {"kind": "file", "sha256": "0" * 64},
            "renderer_commit": "renderer",
            "rasterizer_commit": "rasterizer",
            "exporter_commit": "exporter",
            "image_domain": "rendered_colmap_camera_domain",
            "pixel_coordinate_convention": "zero_indexed_pixel_centers",
            "numerical_support_floor": DEFAULT_NUMERICAL_SUPPORT_FLOOR,
            "variance_clamp_tolerance": DEFAULT_VARIANCE_CLAMP_TOLERANCE,
            **variance_validation_manifest_fields(),
            "depth_index": [
                {
                    "image_name": "im.jpg",
                    "packet_path": str(packet_path),
                    "packet_sha256": packet_file_sha256(packet_path),
                    "height": "1",
                    "width": "2",
                }
            ],
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        loaded = load_depth_manifest(manifest_path)
        index = load_depth_index(manifest_path, loaded)
        valid_packet = validate_metric_packet_npz(
            packet_path,
            index["im.jpg"],
            DEFAULT_NUMERICAL_SUPPORT_FLOOR,
            DEFAULT_VARIANCE_CLAMP_TOLERANCE,
            **variance_validation_manifest_fields(),
        )

        bad_manifest_path = root / "metric_manifest_missing_hash.json"
        bad_manifest = dict(manifest)
        bad_manifest.pop("model_content_hash")
        bad_manifest_path.write_text(json.dumps(bad_manifest), encoding="utf-8")
        try:
            load_depth_manifest(bad_manifest_path)
        except ValueError:
            missing_hash_rejected = True
        else:
            missing_hash_rejected = False

        bad_packet_path = root / "bad_missing_tensor.npz"
        partial = {name: valid_packet[name] for name in METRIC_PACKET_TENSOR_NAMES if name != "camera_z_variance"}
        np.savez_compressed(bad_packet_path, **partial)
        try:
            validate_metric_packet_npz(
                bad_packet_path,
                {"height": "1", "width": "2"},
                DEFAULT_NUMERICAL_SUPPORT_FLOOR,
                DEFAULT_VARIANCE_CLAMP_TOLERANCE,
                **variance_validation_manifest_fields(),
            )
        except ValueError:
            missing_tensor_rejected = True
        else:
            missing_tensor_rejected = False
        bad_policy_path = root / "metric_manifest_bad_policy.json"
        bad_policy = dict(manifest)
        bad_policy["variance_validation_abs_floor"] = -1.0
        bad_policy_path.write_text(json.dumps(bad_policy), encoding="utf-8")
        try:
            load_depth_manifest(bad_policy_path)
        except ValueError:
            bad_policy_rejected = True
        else:
            bad_policy_rejected = False
    if not missing_hash_rejected or not missing_tensor_rejected or not bad_policy_rejected:
        raise AssertionError("metric packet manifest/npz rejection failed")
    return {
        "valid_index_count": len(index),
        "valid_packet_tensor_count": len(valid_packet),
        "missing_model_hash_rejected": missing_hash_rejected,
        "missing_tensor_rejected": missing_tensor_rejected,
        "bad_variance_policy_rejected": bad_policy_rejected,
    }


def test_release_overrides_rejected() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as d:
        config_path, _paths = prepare_release_fixture(Path(d))
        rejected: dict[str, bool] = {}
        for flag, value in [
            ("--annotations_csv", str(Path(d) / "manual.csv")),
            ("--split_csv", str(Path(d) / "manual_split.csv")),
        ]:
            cmd = [
                sys.executable,
                str(REPO_ROOT / "code" / "gcp" / "evaluate_gaussian_gcp_geometry.py"),
                "--release_config",
                str(config_path),
                "--scene",
                "scene_a",
                flag,
                value,
                "--depth_manifest",
                str(Path(d) / "depth_manifest.json"),
            ]
            result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            rejected[flag] = result.returncode != 0 and "manual overrides" in (result.stderr + result.stdout)
    if not all(rejected.values()):
        raise AssertionError(rejected)
    return {"override_rejections": rejected}


def test_unknown_scene_rejected() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as d:
        config_path, _paths = prepare_release_fixture(Path(d))
        cmd = [
            sys.executable,
            str(REPO_ROOT / "code" / "gcp" / "evaluate_gaussian_gcp_geometry.py"),
            "--release_config",
            str(config_path),
            "--scene",
            "unknown_scene",
            "--depth_manifest",
            str(Path(d) / "depth_manifest.json"),
        ]
        result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode == 0 or "Unknown scene" not in (result.stderr + result.stdout):
        raise AssertionError(result.stderr + result.stdout)
    return {"unknown_scene_rejected": True, "returncode": result.returncode}


def run_tests() -> list[dict[str, Any]]:
    tests: list[tuple[str, Callable[[], dict[str, Any]]]] = [
        ("known_sim3_recovery", test_sim3_recovery),
        ("depth_semantics_roundtrip", test_depth_roundtrip),
        ("normalized_vs_unnormalized_inverse_depth", test_unnormalized_inverse_depth_opacity),
        ("plane_camera_z_patch", test_ray_distance_patch_per_pixel),
        ("simple_radial_projection_backprojection", test_simple_radial_roundtrip),
        ("release_hash_and_missing_file_rejection", test_release_hash_and_missing_file),
        ("annotation_scene_mismatch_rejection", test_annotation_scene_mismatch),
        ("depth_manifest_and_unsupported_semantics_rejection", test_depth_manifest_index_and_mismatch),
        ("metric_packet_manifest_and_npz_validation", test_metric_packet_manifest_and_npz_validation),
        ("release_manual_override_rejection", test_release_overrides_rejected),
        ("release_unknown_scene_rejection", test_unknown_scene_rejected),
    ]
    results: list[dict[str, Any]] = []
    for name, fn in tests:
        try:
            payload = fn()
            results.append({"test": name, "status": "PASS", **payload})
        except Exception as exc:  # noqa: BLE001
            results.append({"test": name, "status": "FAIL", "error": repr(exc)})
    return results


def main() -> None:
    results = run_tests()
    passed = sum(1 for row in results if row["status"] == "PASS")
    payload = {
        "schema": "gcp_evaluator_protocol_test_matrix_v1",
        "test_count": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "results": results,
    }
    print(json.dumps(payload, indent=2))
    if payload["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
