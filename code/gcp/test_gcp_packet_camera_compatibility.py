from __future__ import annotations

import json
import math
import shutil
import tempfile
import copy
from pathlib import Path
from typing import Any, Callable

import numpy as np

from gcp_packet_camera_compatibility import (
    IMPLICIT_MAPPING_GATE_PX,
    MATRIX_EQUIVALENCE_TOL_PX,
    RAY_COORD_TOL,
    REQUIRED_TENSOR_DTYPES,
    build_wrapper,
    file_sha256,
    focal2fov,
    load_json,
    load_model_cameras,
    load_release_root_digest,
    load_release_rows,
    load_target_pose_records,
    metric_packet_contract_from_manifest,
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
from gcp_pixel_domain_v1_2 import canonical_record_sha256, canonical_records_root_sha256
from gcp_pixel_domain_v1_2 import CameraRecord


RELEASE_CONFIG = Path(r"E:\datasets\M3M-GCP\scenes\gcp_manual_annotations_v1_2_2\gcp_benchmark_release_v1_2_2.json")
RELEASE_CONFIG_V13 = Path(
    r"E:\datasets\M3M-GCP\scenes\gcp_manual_annotations_v1_3_0\gcp_benchmark_release_v1_3_0.json"
)
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
_WRAPPER_FIXTURE: dict[str, Any] | None = None


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


def wrapper_fixture() -> dict[str, Any]:
    global _WRAPPER_FIXTURE
    if _WRAPPER_FIXTURE is not None:
        return _WRAPPER_FIXTURE
    tmp = tempfile.mkdtemp(prefix="gcp_packet_compat_v11_")
    out_dir = Path(tmp)
    result = build_wrapper(
        release_dir=RELEASE_CONFIG.parent,
        scene="gcp_3000_20260602",
        depth_manifest_path=DEPTH_MANIFEST_3K,
        model_dir=MODEL_DIR_3K,
        renderer_repo=RENDERER_REPO,
        out_dir=out_dir,
        packet_search_roots=[PACKET_SEARCH_ROOT],
        require_local_packets=False,
    )
    wrapper_path = Path(result["wrapper_path"])
    _WRAPPER_FIXTURE = {
        "tmp": out_dir,
        "result": result,
        "wrapper_path": wrapper_path,
        "wrapper": load_json(wrapper_path),
        "common": dict(
            expected_wrapper_sha256=result["wrapper_sha256"],
            depth_manifest=load_depth_manifest_copy(),
            depth_manifest_path=DEPTH_MANIFEST_3K,
            release_config=load_json(RELEASE_CONFIG),
            release_dir=RELEASE_CONFIG.parent,
            scene="gcp_3000_20260602",
            patch_size=7,
            packet_search_roots=[PACKET_SEARCH_ROOT],
        ),
    }
    return _WRAPPER_FIXTURE


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
    explicit_formula_bad = load_depth_manifest_copy()
    explicit_formula_bad["formal_depth_formula"] = "A/M1"
    cases["formal_formula_mismatch"] = expect_raises(lambda: validate_depth_manifest_contract(explicit_formula_bad))
    for bad_formula in ["M1/A+1", "M1/AA", "M1/A malicious_suffix"]:
        manifest = load_depth_manifest_copy()
        manifest["tensor_formulas"]["alpha_normalized_expected_camera_z"] = bad_formula
        cases[f"formal_formula_reject_{bad_formula.replace('/', '_').replace(' ', '_')}"] = expect_raises(
            lambda manifest=manifest: validate_depth_manifest_contract(manifest)
        )
    approved_tensor_formula = load_depth_manifest_copy()
    validate_depth_manifest_contract(approved_tensor_formula)
    cases["approved_tensor_formula_pass"] = "PASS"
    exact_formula = load_depth_manifest_copy()
    exact_formula["tensor_formulas"]["alpha_normalized_expected_camera_z"] = "M1/A"
    validate_depth_manifest_contract(exact_formula)
    cases["exact_m1_over_a_formula_pass"] = "PASS"
    schema_explicit_formula_missing = load_depth_manifest_copy()
    schema_explicit_formula_missing["packet_schema"] = "ms_gcp_metric_depth_packet_requires_explicit_formula_v1"
    schema_explicit_formula_missing.pop("tensor_formulas", None)
    cases["missing_formula_under_explicit_formula_schema"] = expect_raises(
        lambda: metric_packet_contract_from_manifest(schema_explicit_formula_missing)
    )
    unknown_schema_formula = load_depth_manifest_copy()
    unknown_schema_formula["packet_schema"] = "unknown_metric_packet_schema"
    unknown_schema_formula.pop("tensor_formulas", None)
    cases["unknown_schema_formula"] = expect_raises(lambda: metric_packet_contract_from_manifest(unknown_schema_formula))

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
            expected_wrapper_sha256=result["wrapper_sha256"],
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
        def validate_local(path: Path, **kwargs: Any) -> dict[str, Any]:
            expected = kwargs.pop("expected_wrapper_sha256", file_sha256(path))
            call_args = dict(common)
            call_args.update(kwargs)
            return validate_compatibility_wrapper(path, expected_wrapper_sha256=expected, **call_args)

        cases["detached_wrapper_sha_mismatch"] = expect_raises(
            lambda: validate_local(write_mutated("sha_bad", lambda w: w.update({"detached_sha_tamper": True}), False), expected_wrapper_sha256=result["wrapper_sha256"])
        )
        cases["records_root_tamper"] = expect_raises(
            lambda: validate_local(write_mutated("root_bad", lambda w: w["packet_sets"][0].update({"compatibility_records_root_sha256": "0" * 64})))
        )
        cases["mapping_hash_tamper"] = expect_raises(
            lambda: validate_local(write_mutated("mapping_bad", lambda w: w["packet_sets"][0]["view_mappings"][0].update({"source_target_packet_mapping_record_sha256": "1" * 64})))
        )
        cases["packet_camera_hash_tamper"] = expect_raises(
            lambda: validate_local(write_mutated("camera_bad", lambda w: w["packet_sets"][0]["view_mappings"][0].update({"packet_camera_record_sha256": "2" * 64})))
        )
        cases["patch_size_mismatch"] = expect_raises(lambda: validate_local(wrapper_path, patch_size=5, expected_wrapper_sha256=result["wrapper_sha256"]))
        bad_manifest = load_depth_manifest_copy()
        bad_manifest["primary_depth_tensor"] = "harmonic_camera_z"
        cases["wrapper_vs_depth_manifest_tensor_declaration_mismatch"] = expect_raises(
            lambda: validate_local(wrapper_path, depth_manifest=bad_manifest, expected_wrapper_sha256=result["wrapper_sha256"])
        )
    return cases


def write_mutated_wrapper_case(case_name: str, mutate: Callable[[dict[str, Any]], None], refresh_sha: bool = True) -> Path:
    fixture = wrapper_fixture()
    wrapper = copy.deepcopy(fixture["wrapper"])
    mutate(wrapper)
    path = fixture["tmp"] / f"{case_name}.json"
    path.write_text(json.dumps(wrapper, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if refresh_sha:
        write_detached_sha256(path)
    else:
        shutil.copyfile(fixture["wrapper_path"].with_suffix(fixture["wrapper_path"].suffix + ".sha256"), path.with_suffix(path.suffix + ".sha256"))
    return path


def refresh_first_view_and_root(wrapper: dict[str, Any]) -> None:
    packet_set = wrapper["packet_sets"][0]
    row = packet_set["view_mappings"][0]
    row.pop("source_target_packet_mapping_record_sha256", None)
    row["source_target_packet_mapping_record_sha256"] = canonical_record_sha256(row)
    packet_set["compatibility_records_root_sha256"] = canonical_records_root_sha256(
        packet_set["view_mappings"],
        ["scene", "image_name", "packet_camera_record_sha256", "packet_sha256"],
    )


def validate_fixture_wrapper(path: Path | None = None, **overrides: Any) -> dict[str, Any]:
    fixture = wrapper_fixture()
    common = dict(fixture["common"])
    common.update(overrides)
    return validate_compatibility_wrapper(path or fixture["wrapper_path"], **common)


def write_synthetic_packet(path: Path, *, width: int, height: int, bad_shape: bool = False) -> None:
    shape = (height + 1, width) if bad_shape else (height, width)
    arrays: dict[str, Any] = {}
    for name, dtype in REQUIRED_TENSOR_DTYPES.items():
        arrays[name] = np.zeros(shape, dtype=np.dtype(dtype))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def completion_negative_test(name: str, mutate_wrapper: Callable[[dict[str, Any]], None] | None = None, **overrides: Any) -> dict[str, Any]:
    path = write_mutated_wrapper_case(name, mutate_wrapper) if mutate_wrapper else None
    if path is not None and "expected_wrapper_sha256" not in overrides:
        overrides["expected_wrapper_sha256"] = file_sha256(path)
    return {"case": name, "rejected_by": expect_raises(lambda: validate_fixture_wrapper(path, **overrides))}


def test_approved_wrapper_sha_runtime_gate() -> dict[str, Any]:
    fixture = wrapper_fixture()
    correct = validate_fixture_wrapper()
    cases = {
        "missing_approved_sha_rejection": expect_raises(lambda: validate_fixture_wrapper(expected_wrapper_sha256=None)),
        "empty_approved_sha_rejection": expect_raises(lambda: validate_fixture_wrapper(expected_wrapper_sha256="")),
        "malformed_length_sha_rejection": expect_raises(lambda: validate_fixture_wrapper(expected_wrapper_sha256="abc")),
        "non_hex_sha_rejection": expect_raises(lambda: validate_fixture_wrapper(expected_wrapper_sha256="g" * 64)),
        "correct_approved_sha_pass": correct["wrapper_sha256"] == fixture["result"]["wrapper_sha256"],
        "wrong_approved_sha_rejection": expect_raises(lambda: validate_fixture_wrapper(expected_wrapper_sha256="9" * 64)),
    }
    if cases["correct_approved_sha_pass"] is not True:
        raise AssertionError(cases)
    return cases


def test_release_root_digest_mismatch() -> dict[str, Any]:
    return completion_negative_test("release_root_digest_mismatch", lambda w: w.update({"release_root_digest_sha256": "0" * 64}))


def test_release_root_record_sha_mismatch() -> dict[str, Any]:
    return completion_negative_test("release_root_record_sha_mismatch", lambda w: w.update({"release_root_record_sha256": "1" * 64}))


def test_depth_manifest_sha_mismatch() -> dict[str, Any]:
    return completion_negative_test("depth_manifest_sha_mismatch", lambda w: w["packet_sets"][0].update({"original_depth_manifest_sha256": "2" * 64}))


def test_packet_sha_mismatch() -> dict[str, Any]:
    def mutate(w: dict[str, Any]) -> None:
        w["packet_sets"][0]["view_mappings"][0].update({"packet_sha256": "3" * 64})
        refresh_first_view_and_root(w)
    return completion_negative_test("packet_sha_mismatch", mutate)


def test_packet_width_height_mismatch() -> dict[str, Any]:
    manifest = load_depth_manifest_copy()
    manifest["depth_index"][0]["width"] = int(manifest["depth_index"][0]["width"]) + 1
    return completion_negative_test("packet_width_height_mismatch", depth_manifest=manifest)


def test_packet_image_list_mismatch() -> dict[str, Any]:
    manifest = load_depth_manifest_copy()
    manifest["depth_index"] = manifest["depth_index"][:-1]
    return completion_negative_test("packet_image_list_mismatch", depth_manifest=manifest)


def test_duplicate_view_rejection() -> dict[str, Any]:
    return completion_negative_test(
        "duplicate_view_rejection",
        lambda w: w["packet_sets"][0]["view_mappings"].append(copy.deepcopy(w["packet_sets"][0]["view_mappings"][0])),
    )


def test_missing_view_rejection() -> dict[str, Any]:
    return completion_negative_test("missing_view_rejection", lambda w: w["packet_sets"][0]["view_mappings"].pop())


def test_wrong_cx_cy_rejection() -> dict[str, Any]:
    def mutate(w: dict[str, Any]) -> None:
        rec = w["packet_sets"][0]["view_mappings"][0]["packet_camera_record"]
        rec["params"][2] = str(float(rec["params"][2]) + 1.0)
    return completion_negative_test("wrong_cx_cy_rejection", mutate)


def test_undeclared_crop_pad_rejection() -> dict[str, Any]:
    def mutate(w: dict[str, Any]) -> None:
        w["packet_sets"][0]["view_mappings"][0].update({"crop_pad_policy": "crop_undeclared"})
        refresh_first_view_and_root(w)
    return completion_negative_test("undeclared_crop_pad_rejection", mutate)


def test_packet_oob_rejection() -> dict[str, Any]:
    def mutate(w: dict[str, Any]) -> None:
        w["packet_sets"][0]["view_mappings"][0].update({"packet_x": "999999"})
        refresh_first_view_and_root(w)
    return completion_negative_test("packet_oob_rejection", mutate)


def test_renderer_head_mismatch() -> dict[str, Any]:
    return completion_negative_test("renderer_head_mismatch", lambda w: w["packet_sets"][0]["provenance_record"].update({"renderer_commit": "4" * 40}))


def test_renderer_dirty_worktree_rejection() -> dict[str, Any]:
    return completion_negative_test("renderer_dirty_worktree_rejection", lambda w: w["packet_sets"][0]["provenance_record"].update({"renderer_worktree_status_porcelain": " M file"}))


def test_renderer_source_hash_mismatch() -> dict[str, Any]:
    return completion_negative_test("renderer_source_hash_mismatch", lambda w: w["packet_sets"][0]["provenance_record"].update({"renderer_camera_loader_source_sha256": "5" * 64}))


def test_rasterizer_provenance_mismatch() -> dict[str, Any]:
    return completion_negative_test("rasterizer_provenance_mismatch", lambda w: w["packet_sets"][0]["provenance_record"].update({"rasterizer_tree_hash": "6" * 40}))


def test_exporter_provenance_mismatch() -> dict[str, Any]:
    return completion_negative_test("exporter_provenance_mismatch", lambda w: w["packet_sets"][0]["provenance_record"].update({"exporter_commit": "7" * 40}))


def test_packet_ref_protocol_mismatch() -> dict[str, Any]:
    manifest = load_depth_manifest_copy()
    manifest["packet_ref_consistency_protocol"] = "bad_protocol"
    return completion_negative_test("packet_ref_protocol_mismatch", depth_manifest=manifest)


def test_packet_ref_missing_required_field() -> dict[str, Any]:
    manifest = load_depth_manifest_copy()
    manifest["depth_index"][0].pop("variance_packet_ref_abs_error", None)
    return completion_negative_test("packet_ref_missing_required_field", depth_manifest=manifest)


def test_packet_ref_recompute_false() -> dict[str, Any]:
    manifest = load_depth_manifest_copy()
    manifest["depth_index"][0]["packet_recompute_passed"] = False
    return completion_negative_test("packet_ref_recompute_false", depth_manifest=manifest)


def test_packet_ref_failure_count_abnormal() -> dict[str, Any]:
    manifest = load_depth_manifest_copy()
    manifest["depth_index"][0]["variance_consistency_fail_count"] = "1"
    return completion_negative_test("packet_ref_failure_count_abnormal", depth_manifest=manifest)


def test_packet_ref_malformed_numeric_field() -> dict[str, Any]:
    manifest = load_depth_manifest_copy()
    manifest["depth_index"][0]["variance_packet_ref_allowed_error"] = "not_a_number"
    return completion_negative_test("packet_ref_malformed_numeric_field", depth_manifest=manifest)


def test_packet_ref_bound_negative_cases() -> dict[str, Any]:
    cases: dict[str, str] = {}

    def run_case(name: str, **updates: Any) -> None:
        manifest = load_depth_manifest_copy()
        manifest["depth_index"][0].update(updates)
        cases[name] = completion_negative_test(name, depth_manifest=manifest)["rejected_by"]

    run_case(
        "packet_ref_abs_error_gt_allowed",
        variance_packet_ref_abs_error="2.0",
        variance_packet_ref_allowed_error="1.0",
        variance_packet_ref_consistency_ratio="2.0",
    )
    run_case(
        "packet_ref_ratio_gt_one",
        variance_packet_ref_abs_error="0.5",
        variance_packet_ref_allowed_error="1.0",
        variance_packet_ref_consistency_ratio="1.5",
    )
    run_case(
        "packet_ref_negative_abs_error",
        variance_packet_ref_abs_error="-0.1",
        variance_packet_ref_allowed_error="1.0",
        variance_packet_ref_consistency_ratio="0.0",
    )
    run_case(
        "packet_ref_negative_allowed_error",
        variance_packet_ref_abs_error="0.0",
        variance_packet_ref_allowed_error="-1.0",
        variance_packet_ref_consistency_ratio="0.0",
    )
    run_case(
        "packet_ref_zero_allowed_nonzero_abs",
        variance_packet_ref_abs_error="0.1",
        variance_packet_ref_allowed_error="0.0",
        variance_packet_ref_consistency_ratio="0.0",
    )
    run_case(
        "packet_ref_fractional_failure_count",
        variance_consistency_fail_count="0.5",
    )
    run_case(
        "packet_ref_inconsistent_ratio",
        variance_packet_ref_abs_error="0.25",
        variance_packet_ref_allowed_error="1.0",
        variance_packet_ref_consistency_ratio="0.1",
    )
    return cases


def test_local_packet_runtime_header_gate() -> dict[str, Any]:
    fixture = wrapper_fixture()
    first = fixture["wrapper"]["packet_sets"][0]["view_mappings"][0]
    packet_name = Path(first["packet_path_original"]).name
    width = int(first["packet_width"])
    height = int(first["packet_height"])
    cases: dict[str, Any] = {}
    validation = validate_fixture_wrapper()
    packet_set = validation["packet_sets"][0]
    for key, expected in {
        "expected_packet_count": 24,
        "resolved_packet_count": 24,
        "sha_verified_packet_count": 24,
        "header_validated_packet_count": 24,
        "missing_packet_count": 0,
        "ambiguous_packet_count": 0,
    }.items():
        if packet_set.get(key) != expected:
            raise AssertionError({key: packet_set.get(key), "expected": expected})
    cases["real_3k_24_of_24_packet_header_validation_pass"] = dict(packet_set)
    cases["missing_local_packet_rejection"] = expect_raises(lambda: validate_fixture_wrapper(packet_search_roots=[]))

    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        for sub in ["a", "b"]:
            target = tmp_root / sub / packet_name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(first["packet_path_local"]), target)
        cases["ambiguous_local_packet_rejection"] = expect_raises(lambda: validate_fixture_wrapper(packet_search_roots=[tmp_root]))

    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        fake = tmp_root / packet_name
        fake.write_bytes(b"not a real packet")
        cases["local_packet_sha_mismatch_rejection"] = expect_raises(lambda: validate_fixture_wrapper(packet_search_roots=[tmp_root]))

    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        fake = tmp_root / packet_name
        write_synthetic_packet(fake, width=width, height=height, bad_shape=True)
        fake_sha = file_sha256(fake)

        def mutate(w: dict[str, Any]) -> None:
            w["packet_sets"][0]["view_mappings"][0].update({"packet_sha256": fake_sha})
            refresh_first_view_and_root(w)

        path = write_mutated_wrapper_case("local_packet_header_mismatch", mutate)
        manifest = load_depth_manifest_copy()
        manifest["depth_index"][0]["packet_sha256"] = fake_sha
        cases["local_packet_header_mismatch_rejection"] = expect_raises(
            lambda: validate_fixture_wrapper(path, expected_wrapper_sha256=file_sha256(path), depth_manifest=manifest, packet_search_roots=[tmp_root])
        )
    return cases


def test_evaluator_source_sha_mismatch() -> dict[str, Any]:
    return completion_negative_test("evaluator_source_sha_mismatch", lambda w: w.update({"evaluator_source_sha256": "8" * 64}))


def test_runtime_dirty_worktree_record_rejection() -> dict[str, Any]:
    return completion_negative_test("runtime_dirty_worktree_record_rejection", lambda w: w.update({"evaluator_worktree_status_porcelain": " M file"}))


def test_approved_external_wrapper_sha_mismatch() -> dict[str, Any]:
    return completion_negative_test("approved_external_wrapper_sha_mismatch", expected_wrapper_sha256="9" * 64)


def test_target_pose_hash_mismatch() -> dict[str, Any]:
    return completion_negative_test("target_pose_hash_mismatch", lambda w: w["packet_sets"][0]["view_mappings"][0].update({"target_pose_record_sha256": "a" * 64}))


def test_packet_pose_hash_mismatch() -> dict[str, Any]:
    return completion_negative_test("packet_pose_hash_mismatch", lambda w: w["packet_sets"][0]["packet_camera_records"][0].update({"packet_pose_record_sha256": "b" * 64}))


def test_stored_pose_difference_tamper() -> dict[str, Any]:
    return completion_negative_test("stored_pose_difference_tamper", lambda w: w["packet_sets"][0]["view_mappings"][0].update({"pose_center_difference_model_units": "999"}))


def test_stored_pose_pass_tamper() -> dict[str, Any]:
    return completion_negative_test("stored_pose_pass_tamper", lambda w: w["packet_sets"][0]["view_mappings"][0].update({"pose_equivalence_passed": False}))


def test_wrong_release_pose_reference() -> dict[str, Any]:
    def mutate(w: dict[str, Any]) -> None:
        rows = w["packet_sets"][0]["view_mappings"]
        rows[0]["target_pose_record_sha256"] = rows[1]["target_pose_record_sha256"]
    return completion_negative_test("wrong_release_pose_reference", mutate)


def test_orphan_packet_pose_record() -> dict[str, Any]:
    return completion_negative_test("orphan_packet_pose_record", lambda w: w["packet_sets"][0]["packet_camera_records"].pop(0))


def test_packet_camera_record_set_negative_cases() -> dict[str, Any]:
    cases = {
        "extra_packet_camera_record_rejection": completion_negative_test(
            "extra_packet_camera_record_rejection",
            lambda w: w["packet_sets"][0]["packet_camera_records"].append(
                {**copy.deepcopy(w["packet_sets"][0]["packet_camera_records"][0]), "image_name": "EXTRA_PACKET_CAMERA_RECORD.JPG"}
            ),
        )["rejected_by"],
        "duplicate_packet_camera_record_rejection": completion_negative_test(
            "duplicate_packet_camera_record_rejection",
            lambda w: w["packet_sets"][0]["packet_camera_records"].append(copy.deepcopy(w["packet_sets"][0]["packet_camera_records"][0])),
        )["rejected_by"],
        "missing_packet_camera_record_rejection": completion_negative_test(
            "missing_packet_camera_record_rejection",
            lambda w: w["packet_sets"][0]["packet_camera_records"].pop(0),
        )["rejected_by"],
    }
    return cases


def test_v13_release_layout_resolution() -> dict[str, Any]:
    config = load_json(RELEASE_CONFIG_V13)
    rows = load_release_rows(RELEASE_CONFIG_V13.parent, "gcp_3000_20260602", config)
    if len(rows) != 147 or any(str(row.get("formal_eligible", "")).lower() != "true" for row in rows):
        raise AssertionError(f"unexpected v1.3 formal row set: {len(rows)}")
    root = load_release_root_digest(RELEASE_CONFIG_V13.parent, config)
    if root.get("release_id") != config.get("release_id"):
        raise AssertionError("v1.3 release root/config identity mismatch")
    poses = load_target_pose_records(RELEASE_CONFIG_V13.parent, "gcp_3000_20260602", config)
    if len(poses) != 94:
        raise AssertionError(f"unexpected v1.3 target pose count: {len(poses)}")
    return {
        "formal_rows": len(rows),
        "target_poses": len(poses),
        "release_id": root["release_id"],
        "payload_root_digest_sha256": root["payload_root_digest_sha256"],
    }


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
    ("approved_wrapper_sha_runtime_gate", test_approved_wrapper_sha_runtime_gate),
    ("release_root_digest_mismatch", test_release_root_digest_mismatch),
    ("release_root_record_sha_mismatch", test_release_root_record_sha_mismatch),
    ("depth_manifest_sha_mismatch", test_depth_manifest_sha_mismatch),
    ("packet_sha_mismatch", test_packet_sha_mismatch),
    ("packet_width_height_mismatch", test_packet_width_height_mismatch),
    ("packet_image_list_mismatch", test_packet_image_list_mismatch),
    ("duplicate_view_rejection", test_duplicate_view_rejection),
    ("missing_view_rejection", test_missing_view_rejection),
    ("wrong_cx_cy_rejection", test_wrong_cx_cy_rejection),
    ("undeclared_crop_pad_rejection", test_undeclared_crop_pad_rejection),
    ("packet_oob_rejection", test_packet_oob_rejection),
    ("renderer_head_mismatch", test_renderer_head_mismatch),
    ("renderer_dirty_worktree_rejection", test_renderer_dirty_worktree_rejection),
    ("renderer_source_hash_mismatch", test_renderer_source_hash_mismatch),
    ("rasterizer_provenance_mismatch", test_rasterizer_provenance_mismatch),
    ("exporter_provenance_mismatch", test_exporter_provenance_mismatch),
    ("packet_ref_protocol_mismatch", test_packet_ref_protocol_mismatch),
    ("packet_ref_missing_required_field", test_packet_ref_missing_required_field),
    ("packet_ref_recompute_false", test_packet_ref_recompute_false),
    ("packet_ref_failure_count_abnormal", test_packet_ref_failure_count_abnormal),
    ("packet_ref_malformed_numeric_field", test_packet_ref_malformed_numeric_field),
    ("packet_ref_bound_negative_cases", test_packet_ref_bound_negative_cases),
    ("local_packet_runtime_header_gate", test_local_packet_runtime_header_gate),
    ("evaluator_source_sha_mismatch", test_evaluator_source_sha_mismatch),
    ("runtime_dirty_worktree_record_rejection", test_runtime_dirty_worktree_record_rejection),
    ("approved_external_wrapper_sha_mismatch", test_approved_external_wrapper_sha_mismatch),
    ("target_pose_hash_mismatch", test_target_pose_hash_mismatch),
    ("packet_pose_hash_mismatch", test_packet_pose_hash_mismatch),
    ("stored_pose_difference_tamper", test_stored_pose_difference_tamper),
    ("stored_pose_pass_tamper", test_stored_pose_pass_tamper),
    ("wrong_release_pose_reference", test_wrong_release_pose_reference),
    ("orphan_packet_pose_record", test_orphan_packet_pose_record),
    ("packet_camera_record_set_negative_cases", test_packet_camera_record_set_negative_cases),
    ("v13_release_layout_resolution", test_v13_release_layout_resolution),
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
