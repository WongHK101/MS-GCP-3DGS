#!/usr/bin/env python3
"""CPU-only synthetic conformance preflight for the 3DGS raw-moment adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from m3m_native_quarter_protocol import (
    DEFAULT_SUPPORT_FLOOR,
    PIXEL_CONVENTION,
    PROTOCOL_ID,
    geometric_median,
    half_pixel_sensitivity,
    sample_raw_moment_camera_z,
)


def case(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), **evidence}


def run_preflight() -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    yy, xx = np.mgrid[0:6, 0:6].astype(np.float64)
    u, v = 2.25, 2.40

    alpha = np.full((6, 6), 0.8, dtype=np.float32)
    flat_z = np.full((6, 6), 12.0, dtype=np.float32)
    flat = sample_raw_moment_camera_z(alpha, alpha * flat_z, u, v)
    cases.append(
        case(
            "single_plane_exact_shifted_ray",
            flat["valid"] and abs(flat["camera_z"] - 12.0) <= 5.0e-7,
            exact_camera_z=12.0,
            sampled_camera_z=flat["camera_z"],
            abs_error_m=abs(flat["camera_z"] - 12.0),
        )
    )

    tilted_z = 7.0 + 0.30 * xx - 0.20 * yy
    tilted_alpha = np.full((6, 6), 0.7, dtype=np.float32)
    tilted = sample_raw_moment_camera_z(
        tilted_alpha, (tilted_alpha * tilted_z).astype(np.float32), u, v
    )
    tilted_exact = 7.0 + 0.30 * u - 0.20 * v
    cases.append(
        case(
            "tilted_plane_exact_shifted_ray",
            tilted["valid"] and abs(tilted["camera_z"] - tilted_exact) <= 2.0e-6,
            exact_camera_z=tilted_exact,
            sampled_camera_z=tilted["camera_z"],
            abs_error_m=abs(tilted["camera_z"] - tilted_exact),
        )
    )

    layer_alpha = np.full((6, 6), 0.8, dtype=np.float32)
    layer_m1 = np.full((6, 6), 12.0, dtype=np.float32)
    layered = sample_raw_moment_camera_z(layer_alpha, layer_m1, u, v)
    cases.append(
        case(
            "front_back_two_layer_expected_coordinate",
            layered["valid"] and abs(layered["camera_z"] - 15.0) <= 5.0e-7,
            definition="(0.6*10 + 0.2*30) / (0.6+0.2)",
            expected_camera_z=15.0,
            sampled_camera_z=layered["camera_z"],
        )
    )

    boundary_z = np.where(xx <= 2.0, 10.0, 30.0).astype(np.float32)
    boundary_alpha = np.ones((6, 6), dtype=np.float32)
    boundary = sample_raw_moment_camera_z(boundary_alpha, boundary_z, 2.5, 2.25)
    boundary_sensitivity = half_pixel_sensitivity(
        boundary_alpha, boundary_z, 2.5, 2.25
    )
    exact_right_ray = 30.0
    boundary_difference = abs(boundary["camera_z"] - exact_right_ray)
    cases.append(
        case(
            "depth_boundary_mixing_is_detected_not_hidden",
            boundary["valid"]
            and boundary_difference >= 9.9
            and float(boundary_sensitivity["max_abs_camera_z_delta_model_units"]) >= 9.9,
            bilinear_camera_z=boundary["camera_z"],
            exact_right_shifted_ray_camera_z=exact_right_ray,
            exact_comparison_abs_difference_m=boundary_difference,
            half_pixel_max_abs_camera_z_delta_model_units=boundary_sensitivity[
                "max_abs_camera_z_delta_model_units"
            ],
            formal_interpretation="boundary sensitivity diagnostic; bilinear result is not relabelled as a physical first surface",
        )
    )

    below = np.full((6, 6), DEFAULT_SUPPORT_FLOOR, dtype=np.float32)
    above = np.full((6, 6), DEFAULT_SUPPORT_FLOOR * 2.0, dtype=np.float32)
    below_sample = sample_raw_moment_camera_z(below, below * 10.0, u, v)
    above_sample = sample_raw_moment_camera_z(above, above * 10.0, u, v)
    cases.append(
        case(
            "low_transparency_strict_support_floor_no_epsilon",
            (not below_sample["valid"])
            and below_sample["failure_reason"] == "interpolated_support_not_above_floor"
            and above_sample["valid"],
            support_floor=DEFAULT_SUPPORT_FLOOR,
            equal_to_floor_valid=below_sample["valid"],
            twice_floor_valid=above_sample["valid"],
        )
    )

    floating = sample_raw_moment_camera_z(
        tilted_alpha, (tilted_alpha * tilted_z).astype(np.float32), u, v
    )
    rounded = sample_raw_moment_camera_z(
        tilted_alpha,
        (tilted_alpha * tilted_z).astype(np.float32),
        float(round(u)),
        float(round(v)),
    )
    cases.append(
        case(
            "floating_pixel_is_not_rounded",
            floating["valid"]
            and rounded["valid"]
            and abs(floating["camera_z"] - tilted_exact) <= 2.0e-6
            and abs(floating["camera_z"] - rounded["camera_z"]) > 1.0e-3,
            floating_pixel=[u, v],
            floating_camera_z=floating["camera_z"],
            rounded_camera_z=rounded["camera_z"],
        )
    )

    fx, fy, cx, cy = 900.0, 910.0, 707.0, 512.0
    camera_z = 18.0
    xyz_camera = np.asarray(
        [(u - cx) / fx * camera_z, (v - cy) / fy * camera_z, camera_z],
        dtype=np.float64,
    )
    roundtrip_u = fx * xyz_camera[0] / xyz_camera[2] + cx
    roundtrip_v = fy * xyz_camera[1] / xyz_camera[2] + cy
    cases.append(
        case(
            "zero_based_pixel_camera_roundtrip",
            abs(roundtrip_u - u) <= 1.0e-12 and abs(roundtrip_v - v) <= 1.0e-12,
            pixel_convention=PIXEL_CONVENTION,
            input_pixel=[u, v],
            roundtrip_pixel=[roundtrip_u, roundtrip_v],
        )
    )

    variable_alpha = (0.2 + 0.05 * xx + 0.03 * yy).astype(np.float32)
    variable_z = (5.0 + 0.7 * xx - 0.1 * yy).astype(np.float32)
    raw_ratio = sample_raw_moment_camera_z(
        variable_alpha, variable_alpha * variable_z, u, v
    )
    normalised_neighbours = sample_raw_moment_camera_z(
        np.ones_like(variable_alpha), variable_z, u, v
    )
    operator_difference = abs(raw_ratio["camera_z"] - normalised_neighbours["camera_z"])
    cases.append(
        case(
            "raw_moments_are_interpolated_before_division",
            raw_ratio["valid"]
            and normalised_neighbours["valid"]
            and operator_difference > 1.0e-4,
            raw_moment_ratio=raw_ratio["camera_z"],
            bilinear_normalised_depth=normalised_neighbours["camera_z"],
            absolute_difference_m=operator_difference,
        )
    )

    points = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.1, 0.0], [0.2, 1.0, 0.2], [0.1, 0.2, 1.0]],
        dtype=np.float64,
    )
    angle = 0.73
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    centre = geometric_median(points)
    rotated_centre = geometric_median((rotation @ points.T).T)
    equivariance_error = float(np.linalg.norm(rotated_centre - rotation @ centre))
    cases.append(
        case(
            "geometric_median_rotation_equivariance",
            equivariance_error <= 1.0e-9,
            equivariance_error_m=equivariance_error,
        )
    )

    passed = all(item["passed"] for item in cases)
    return {
        "schema": "m3m_gcp_3dgs_native_quarter_adapter_cpu_preflight_v2",
        "protocol_id": PROTOCOL_ID,
        "adapter_id": "3dgs_raw_accumulated_alpha_weighted_camera_z_sum_v2",
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "device": "CPU",
        "case_count": len(cases),
        "cases": cases,
        "scope": "common operator and camera-coordinate contract only",
        "remaining_gate": "the patched evaluation renderer must build on the target CUDA runtime and pass a real packet-camera export/validation preflight on the frozen 3K cameras",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = run_preflight()
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
