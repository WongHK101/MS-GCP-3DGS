from __future__ import annotations

import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np

from gcp_packet_camera_compatibility import (
    IMPLICIT_MAPPING_GATE_PX,
    MATRIX_EQUIVALENCE_TOL_PX,
    RAY_COORD_TOL,
    build_wrapper,
    file_sha256,
    focal2fov,
    load_json,
    load_model_cameras,
    load_target_pose_records,
    normalize_pixel_convention,
    packet_projection_for_row,
    packet_camera_hash,
    pose_from_cameras_json,
    project_packet,
    projection_matrix_equivalence,
    qvec_to_rotmat,
    recover_packet_camera,
    validate_compatibility_wrapper,
    validate_depth_manifest_contract,
    validate_npz_packet_headers,
    write_detached_sha256,
)
from gcp_pixel_domain_v1_2 import CameraRecord


RELEASE_CONFIG = Path(r"E:\datasets\M3M-GCP\scenes\gcp_manual_annotations_v1_2_2\gcp_benchmark_release_v1_2_2.json")
DEPTH_MANIFEST_3K = Path(
    r"E:\M3M-GCP-3DGS\outputs\stage3_v122_pkt_blocker_20260629_012143\evidence\metric_depth_manifest_gcp_3000_20260602.json"
)
MODEL_DIR_3K = Path(r"E:\M3M-GCP-3DGS\outputs\sibr_models_3scenes_20260624\models\gcp_3000_20260602\Model_RGB")
RENDERER_REPO = Path(r"E:\worktrees\Multispectral_gcp_regression_20260627_37698b4")
if not RENDERER_REPO.exists():
    RENDERER_REPO = Path(r"E:\Multispectral")
PACKET_SEARCH_ROOT = Path(
    r"E:\M3M-GCP-3DGS\outputs\gcp_3k_depth_semantics_inputs_20260628\packets\gcp_3000_20260602_full_reused_release"
)


def assert_close(actual: float, expected: float, tol: float = 1e-9) -> None:
    if abs(float(actual) - float(expected)) > tol:
        raise AssertionError(f"not close: actual={actual} expected={expected} tol={tol}")


def expect_raises(fn: Callable[[], Any]) -> str:
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"
    raise AssertionError("expected failure did not occur")


def load_depth_manifest_copy() -> dict[str, Any]:
    return json.loads(DEPTH_MANIFEST_3K.read_text(encoding="utf-8"))


def make_target_camera(width: int = 5654, height: int = 4098, focal: float = 3704.175422665665) -> CameraRecord:
    return CameraRecord(
        camera_id=1,
        model="PINHOLE",
        width=width,
        height=height,
        params=(focal, focal, width / 2.0, height / 2.0),
    )


def make_model_row(width: int = 5654, height: int = 4098, focal: float = 3704.175422665665) -> dict[str, Any]:
    return {
        "img_name": "im.jpg",
        "width": width,
        "height": height,
        "fx": focal,
        "fy": focal,
    }


def test_r8_camera_recovery() -> dict[str, Any]:
    target = make_target_camera()
    camera = recover_packet_camera(
        scene="unit",
        image_name="im.jpg",
        target_camera=target,
        model_camera_row=make_model_row(),
        packet_width=707,
        packet_height=512,
        cfg_args={"resolution": 8},
        manifest_pixel_convention="zero_indexed_pixel_centers",
    )
    expected_fx = target.params[0] * (707 / 5654)
    expected_fy = target.params[1] * (512 / 4098)
    assert_close(camera.fx, expected_fx, 1e-9)
    assert_close(camera.fy, expected_fy, 1e-9)
    assert_close(camera.cx, 353.5, 0.0)
    assert_close(camera.cy, 256.0, 0.0)
    return {"width": camera.width, "height": camera.height, "fx": camera.fx, "fy": camera.fy}


def test_rounding_tie_case() -> dict[str, Any]:
    target = make_target_camera(width=100, height=108, focal=50.0)
    camera = recover_packet_camera(
        scene="unit",
        image_name="tie.jpg",
        target_camera=target,
        model_camera_row=make_model_row(width=100, height=108, focal=50.0),
        packet_width=12,
        packet_height=14,
        cfg_args={"resolution": 8},
        manifest_pixel_convention="zero_based_pixel_centers",
    )
    if (camera.width, camera.height) != (12, 14):
        raise AssertionError("Python round ties-to-even semantics not honored")
    return {"source": "100x108", "resolution": 8, "packet": "12x14"}


def test_projection_matrix_equivalence() -> dict[str, Any]:
    camera = recover_packet_camera(
        scene="unit",
        image_name="im.jpg",
        target_camera=make_target_camera(),
        model_camera_row=make_model_row(),
        packet_width=707,
        packet_height=512,
        cfg_args={"resolution": 8},
        manifest_pixel_convention="zero_based_pixel_centers",
    )
    result = projection_matrix_equivalence(camera)
    if not result["passed"]:
        raise AssertionError(result)
    return {"max_error_px": result["max_error_px"], "tolerance_px": MATRIX_EQUIVALENCE_TOL_PX}


def test_generic_r1_r2_r4_r8_schema() -> dict[str, Any]:
    out = {}
    for resolution in [1, 2, 4, 8]:
        target = make_target_camera()
        width = round(5654 / resolution)
        height = round(4098 / resolution)
        camera = recover_packet_camera(
            scene="unit",
            image_name=f"r{resolution}.jpg",
            target_camera=target,
            model_camera_row=make_model_row(),
            packet_width=width,
            packet_height=height,
            cfg_args={"resolution": resolution},
            manifest_pixel_convention="zero_based_pixel_centers",
        )
        out[f"R{resolution}"] = {"width": camera.width, "height": camera.height}
    return out


def test_sx_not_sy() -> dict[str, Any]:
    target = make_target_camera()
    sx = 707 / target.width
    sy = 512 / target.height
    if abs(sx - sy) < 1e-6:
        raise AssertionError("fixture should have sx != sy")
    return {"sx": sx, "sy": sy}


def test_pixel_convention_alias_and_unknown() -> dict[str, Any]:
    if normalize_pixel_convention("zero_indexed_pixel_centers")[1] != "zero_based_pixel_centers":
        raise AssertionError("alias normalization failed")
    rejected = False
    try:
        normalize_pixel_convention("corner_pixels")
    except ValueError:
        rejected = True
    if not rejected:
        raise AssertionError("unknown pixel convention was accepted")
    return {"zero_indexed_alias": "zero_based_pixel_centers", "unknown_rejected": True}


def test_half_pixel_formula_distinction() -> dict[str, Any]:
    u = 100.25
    scale = 0.125
    direct = u * scale
    corner = (u + 0.5) * scale - 0.5
    if abs(direct - corner) < 1e-9:
        raise AssertionError("half-pixel fixture does not distinguish formulas")
    return {"direct": direct, "corner_formula": corner, "difference": abs(direct - corner)}


def test_pose_intrinsic_negative_cases() -> dict[str, Any]:
    target = make_target_camera()
    camera = recover_packet_camera(
        scene="unit",
        image_name="im.jpg",
        target_camera=target,
        model_camera_row=make_model_row(),
        packet_width=707,
        packet_height=512,
        cfg_args={"resolution": 8},
        manifest_pixel_convention="zero_based_pixel_centers",
    )
    good_x, good_y = project_packet(camera, 0.1, -0.05)
    bad_mapping = {
        "packet_camera_record_sha256": packet_camera_hash(camera),
        "packet_camera_record": {
            "camera_id": 1,
            "model": "PINHOLE",
            "width": camera.width,
            "height": camera.height,
            "params": [camera.fx * 1.01, camera.fy, camera.cx, camera.cy],
        },
        "sx": 707 / 5654,
        "sy": 512 / 4098,
    }
    row = {
        "image_name": "im.jpg",
        "u_px": str(target.params[0] * 0.1 + target.params[2]),
        "v_px": str(target.params[1] * -0.05 + target.params[3]),
        "target_camera_params": ";".join(str(v) for v in target.params),
    }
    rejected = False
    try:
        packet_projection_for_row(row, bad_mapping)
    except ValueError:
        rejected = True
    if not rejected:
        raise AssertionError("wrong packet intrinsics were not rejected")
    return {"good_packet_u": good_x, "good_packet_v": good_y, "wrong_intrinsics_rejected": True}


def test_packet_schema_formal_depth_negative_cases() -> dict[str, Any]:
    cases: dict[str, str] = {}
    mutations = {
        "packet_schema_mismatch": ("schema", "bad_manifest_schema"),
        "packet_schema_version_mismatch": ("packet_schema", "bad_packet_schema"),
        "primary_tensor_mismatch": ("primary_depth_tensor", "harmonic_camera_z"),
        "formal_semantics_mismatch": ("primary_depth_semantics", "ray_distance"),
        "formal_formula_mismatch": ("tensor_names", ["accumulated_alpha"]),
        "required_tensor_missing": ("tensor_names", ["accumulated_alpha", "weighted_camera_z_sum"]),
        "dtype_metadata_mismatch": ("dtype", "float64"),
    }
    for name, (key, value) in mutations.items():
        manifest = load_depth_manifest_copy()
        if name == "packet_schema_mismatch":
            manifest[key] = "bad_manifest_schema"
        elif name == "formal_formula_mismatch":
            manifest["packet_schema"] = "ms_gcp_metric_depth_packet_v2"
            manifest[key] = value
        else:
            manifest[key] = value
        cases[name] = expect_raises(lambda manifest=manifest: validate_depth_manifest_contract(manifest))

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad_packet.npz"
        np.savez_compressed(path, accumulated_alpha=np.zeros((2, 2), dtype=np.float32))
        cases["required_tensor_missing"] = expect_raises(
            lambda: validate_npz_packet_headers(path=path, expected_width=2, expected_height=2, expected_dtype="float32")
        )
        path_dtype = Path(tmp) / "bad_dtype.npz"
        arrays = {name: np.zeros((2, 2), dtype=np.float32) for name in [
            "accumulated_alpha",
            "weighted_camera_z_sum",
            "weighted_camera_z_second_moment",
            "weighted_inverse_camera_z_sum",
            "alpha_normalized_expected_camera_z",
            "alpha_normalized_expected_inverse_camera_z",
            "harmonic_camera_z",
            "camera_z_variance",
            "historical_invalid_unnormalized_inverse_depth",
        ]}
        arrays["metric_depth_valid_mask"] = np.zeros((2, 2), dtype=np.float32)
        np.savez_compressed(path_dtype, **arrays)
        cases["tensor_dtype_metadata_mismatch"] = expect_raises(
            lambda: validate_npz_packet_headers(path=path_dtype, expected_width=2, expected_height=2, expected_dtype="float32")
        )
        path_shape = Path(tmp) / "bad_shape.npz"
        arrays["metric_depth_valid_mask"] = np.zeros((2, 3), dtype=bool)
        np.savez_compressed(path_shape, **arrays)
        cases["tensor_shape_metadata_mismatch"] = expect_raises(
            lambda: validate_npz_packet_headers(path=path_shape, expected_width=2, expected_height=2, expected_dtype="float32")
        )
    return cases


def test_real_3k_golden_pose_conversion() -> dict[str, Any]:
    model_rows = load_model_cameras(MODEL_DIR_3K / "cameras.json")
    target_poses = load_target_pose_records(RELEASE_CONFIG.parent, "gcp_3000_20260602")
    name = "DJI_20260602165038_0001_D.JPG"
    pose = pose_from_cameras_json(model_rows[name], target_poses[name])
    if not pose["passed"]:
        raise AssertionError(pose)
    return {
        "image_name": name,
        "raw_rotation": pose["r_c2w"],
        "raw_position": pose["position_camera_center"],
        "converted_R_w2c": pose["r_w2c"],
        "converted_tvec": pose["tvec"],
        "target_qvec": pose["target_qvec"],
        "target_tvec": pose["target_tvec"],
        "center_difference_model_units": pose["center_difference_model_units"],
        "rotation_angular_difference_rad": pose["rotation_angular_difference_rad"],
        "pose_record_sha256": pose["pose_record_sha256"],
    }


def test_pose_conversion_negative_cases() -> dict[str, Any]:
    model_rows = load_model_cameras(MODEL_DIR_3K / "cameras.json")
    target_poses = load_target_pose_records(RELEASE_CONFIG.parent, "gcp_3000_20260602")
    name = "DJI_20260602165038_0001_D.JPG"
    good_model = dict(model_rows[name])
    good_target = dict(target_poses[name])
    cases: dict[str, str | bool] = {}

    transpose_model = dict(good_model)
    transpose_model["rotation"] = np.asarray(transpose_model["rotation"], dtype=float).T.tolist()
    cases["rotation_transpose_wrong"] = expect_raises(lambda: _require_pose_fail(transpose_model, good_target))
    cases["camera_to_world_world_to_camera_confusion"] = expect_raises(lambda: _require_pose_fail(transpose_model, good_target))
    cases["row_major_column_major_confusion"] = expect_raises(lambda: _require_pose_fail(transpose_model, good_target))

    tvec_position = dict(good_model)
    tvec_position["position"] = [float(x) for x in good_target["tvec"]]
    cases["position_treated_as_tvec_wrong"] = expect_raises(lambda: _require_pose_fail(tvec_position, good_target))

    axis_flip = dict(good_model)
    r = np.asarray(axis_flip["rotation"], dtype=float)
    r[:, 1] *= -1.0
    axis_flip["rotation"] = r.tolist()
    cases["axis_flip_wrong"] = expect_raises(lambda: _require_pose_fail(axis_flip, good_target))

    other_target = dict(target_poses["DJI_20260602165042_0002_D.JPG"])
    cases["another_view_pose_wrong"] = expect_raises(lambda: _require_pose_fail(good_model, other_target))
    shifted_model = dict(good_model)
    shifted_model["position"] = [float(x) + 1e-5 for x in shifted_model["position"]]
    cases["pose_tolerance_exceeded"] = expect_raises(lambda: _require_pose_fail(shifted_model, good_target))

    target_q = [float(x) for x in good_target["qvec"]]
    if not np.allclose(qvec_to_rotmat(target_q), qvec_to_rotmat([-x for x in target_q]), atol=0.0, rtol=0.0):
        raise AssertionError("quaternion sign equivalence failed")
    cases["quaternion_sign_equivalence_passed"] = True
    wrong_order = [target_q[1], target_q[2], target_q[3], target_q[0]]
    if np.allclose(qvec_to_rotmat(target_q), qvec_to_rotmat(wrong_order)):
        raise AssertionError("quaternion component order negative fixture failed")
    cases["quaternion_component_order_wrong_rejected"] = True
    return cases


def _require_pose_fail(model_row: dict[str, Any], target_row: dict[str, Any]) -> None:
    pose = pose_from_cameras_json(model_row, target_row)
    if pose["passed"]:
        raise AssertionError("pose unexpectedly passed")
    raise ValueError("pose rejected as expected")


def test_real_3k_coordinate_only_wrapper() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        result = build_wrapper(
            release_dir=RELEASE_CONFIG.parent,
            scene="gcp_3000_20260602",
            depth_manifest_path=DEPTH_MANIFEST_3K,
            model_dir=MODEL_DIR_3K,
            renderer_repo=RENDERER_REPO,
            out_dir=Path(tmp),
            packet_search_roots=[PACKET_SEARCH_ROOT],
            require_local_packets=False,
        )
        summary = result["coordinate_summary"]
        if summary["observation_count"] != 93:
            raise AssertionError(summary)
        if summary["packet_bounds_pass_count"] != 93:
            raise AssertionError(summary)
        if summary["max_ray_coordinate_error"] > RAY_COORD_TOL:
            raise AssertionError(summary)
        if summary["max_projection_vs_implicit_diff_px"] > IMPLICIT_MAPPING_GATE_PX:
            raise AssertionError(summary)
        wrapper = json.loads(Path(result["wrapper_path"]).read_text(encoding="utf-8"))
        validation = validate_compatibility_wrapper(
            Path(result["wrapper_path"]),
            depth_manifest=load_depth_manifest_copy(),
            depth_manifest_path=DEPTH_MANIFEST_3K,
            release_config=load_json(RELEASE_CONFIG),
            release_dir=RELEASE_CONFIG.parent,
            scene="gcp_3000_20260602",
            patch_size=7,
            packet_search_roots=[PACKET_SEARCH_ROOT],
        )
        packet_set = wrapper["packet_sets"][0]
        if packet_set["view_count"] != 24:
            raise AssertionError(packet_set["view_count"])
    return {
        "observations": 93,
        "views": 24,
        "max_projection_vs_implicit_diff_px": summary["max_projection_vs_implicit_diff_px"],
        "max_ray_coordinate_error": summary["max_ray_coordinate_error"],
        "runtime_validation_packet_sets": validation["packet_sets"],
    }


def test_wrapper_runtime_negative_cases() -> dict[str, Any]:
    cases: dict[str, str] = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        result = build_wrapper(
            release_dir=RELEASE_CONFIG.parent,
            scene="gcp_3000_20260602",
            depth_manifest_path=DEPTH_MANIFEST_3K,
            model_dir=MODEL_DIR_3K,
            renderer_repo=RENDERER_REPO,
            out_dir=tmp_path,
            packet_search_roots=[PACKET_SEARCH_ROOT],
            require_local_packets=False,
        )
        wrapper_path = Path(result["wrapper_path"])
        wrapper = load_json(wrapper_path)

        def write_mutated(name: str, mutate: Callable[[dict[str, Any]], None], refresh_sha: bool = True) -> Path:
            mutated = json.loads(json.dumps(wrapper))
            mutate(mutated)
            p = tmp_path / f"{name}.json"
            p.write_text(json.dumps(mutated, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            if refresh_sha:
                write_detached_sha256(p)
            else:
                shutil.copyfile(wrapper_path.with_suffix(wrapper_path.suffix + ".sha256"), p.with_suffix(p.suffix + ".sha256"))
            return p

        common = dict(
            depth_manifest=load_depth_manifest_copy(),
            depth_manifest_path=DEPTH_MANIFEST_3K,
            release_config=load_json(RELEASE_CONFIG),
            release_dir=RELEASE_CONFIG.parent,
            scene="gcp_3000_20260602",
            patch_size=7,
            packet_search_roots=[PACKET_SEARCH_ROOT],
        )
        cases["detached_wrapper_sha_mismatch"] = expect_raises(
            lambda: validate_compatibility_wrapper(write_mutated("sha_bad", lambda w: w.update({"detached_sha_tamper": True}), False), **common)
        )
        cases["records_root_tamper"] = expect_raises(
            lambda: validate_compatibility_wrapper(
                write_mutated("root_bad", lambda w: w["packet_sets"][0].update({"compatibility_records_root_sha256": "0" * 64})),
                **common,
            )
        )
        cases["mapping_hash_tamper"] = expect_raises(
            lambda: validate_compatibility_wrapper(
                write_mutated("mapping_bad", lambda w: w["packet_sets"][0]["view_mappings"][0].update({"source_target_packet_mapping_record_sha256": "1" * 64})),
                **common,
            )
        )
        cases["packet_camera_hash_tamper"] = expect_raises(
            lambda: validate_compatibility_wrapper(
                write_mutated("camera_bad", lambda w: w["packet_sets"][0]["view_mappings"][0].update({"packet_camera_record_sha256": "2" * 64})),
                **common,
            )
        )
        cases["patch_size_mismatch"] = expect_raises(lambda: validate_compatibility_wrapper(wrapper_path, **{**common, "patch_size": 5}))
        bad_manifest = load_depth_manifest_copy()
        bad_manifest["primary_depth_tensor"] = "harmonic_camera_z"
        cases["wrapper_vs_depth_manifest_tensor_declaration_mismatch"] = expect_raises(
            lambda: validate_compatibility_wrapper(wrapper_path, **{**common, "depth_manifest": bad_manifest})
        )
    return cases


TESTS: list[tuple[str, Callable[[], dict[str, Any]]]] = [
    ("r8_camera_recovery", test_r8_camera_recovery),
    ("rounding_tie_case", test_rounding_tie_case),
    ("projection_matrix_equivalence", test_projection_matrix_equivalence),
    ("generic_r1_r2_r4_r8_schema", test_generic_r1_r2_r4_r8_schema),
    ("sx_not_sy", test_sx_not_sy),
    ("pixel_convention_alias_and_unknown", test_pixel_convention_alias_and_unknown),
    ("half_pixel_formula_distinction", test_half_pixel_formula_distinction),
    ("pose_intrinsic_negative_cases", test_pose_intrinsic_negative_cases),
    ("packet_schema_formal_depth_negative_cases", test_packet_schema_formal_depth_negative_cases),
    ("real_3k_golden_pose_conversion", test_real_3k_golden_pose_conversion),
    ("pose_conversion_negative_cases", test_pose_conversion_negative_cases),
    ("real_3k_coordinate_only_wrapper", test_real_3k_coordinate_only_wrapper),
    ("wrapper_runtime_negative_cases", test_wrapper_runtime_negative_cases),
]


def main() -> None:
    rows = []
    failed = False
    for name, fn in TESTS:
        try:
            details = fn()
            status = "PASS"
        except Exception as exc:  # noqa: BLE001
            details = {"error": f"{type(exc).__name__}: {exc}"}
            status = "FAIL"
            failed = True
        rows.append({"test": name, "status": status, "details": details})
    print(json.dumps({"tests": rows, "passed": not failed}, indent=2, ensure_ascii=False, sort_keys=True))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
