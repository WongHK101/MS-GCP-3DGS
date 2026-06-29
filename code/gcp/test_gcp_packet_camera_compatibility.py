from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np

from gcp_packet_camera_compatibility import (
    IMPLICIT_MAPPING_GATE_PX,
    MATRIX_EQUIVALENCE_TOL_PX,
    RAY_COORD_TOL,
    build_wrapper,
    focal2fov,
    normalize_pixel_convention,
    packet_projection_for_row,
    packet_camera_hash,
    project_packet,
    projection_matrix_equivalence,
    recover_packet_camera,
)
from gcp_pixel_domain_v1_2 import CameraRecord


RELEASE_CONFIG = Path(r"E:\datasets\M3M-GCP\scenes\gcp_manual_annotations_v1_2_2\gcp_benchmark_release_v1_2_2.json")
DEPTH_MANIFEST_3K = Path(
    r"E:\M3M-GCP-3DGS\outputs\stage3_v122_pkt_blocker_20260629_012143\evidence\metric_depth_manifest_gcp_3000_20260602.json"
)
MODEL_DIR_3K = Path(r"E:\M3M-GCP-3DGS\outputs\sibr_models_3scenes_20260624\models\gcp_3000_20260602\Model_RGB")
RENDERER_REPO = Path(r"E:\Multispectral")
PACKET_SEARCH_ROOT = Path(
    r"E:\M3M-GCP-3DGS\outputs\gcp_3k_depth_semantics_inputs_20260628\packets\gcp_3000_20260602_full_reused_release"
)


def assert_close(actual: float, expected: float, tol: float = 1e-9) -> None:
    if abs(float(actual) - float(expected)) > tol:
        raise AssertionError(f"not close: actual={actual} expected={expected} tol={tol}")


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
        packet_set = wrapper["packet_sets"][0]
        if packet_set["view_count"] != 24:
            raise AssertionError(packet_set["view_count"])
    return {
        "observations": 93,
        "views": 24,
        "max_projection_vs_implicit_diff_px": summary["max_projection_vs_implicit_diff_px"],
        "max_ray_coordinate_error": summary["max_ray_coordinate_error"],
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
    ("real_3k_coordinate_only_wrapper", test_real_3k_coordinate_only_wrapper),
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
