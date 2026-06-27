from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metric_depth_packet import (  # noqa: E402
    DEFAULT_ALPHA_CUTOFF,
    DEFAULT_EARLY_TERMINATION_THRESHOLD,
    DEFAULT_NUMERICAL_SUPPORT_FLOOR,
    DEFAULT_VARIANCE_CLAMP_TOLERANCE,
    DIAGNOSTIC_VARIANCE_TENSOR,
    DIAGNOSTIC_VARIANCE_VALID_MASK_TENSOR,
    HISTORICAL_INVALID_TENSOR,
    METRIC_PACKET_TENSOR_NAMES,
    PRIMARY_DEPTH_TENSOR,
    cpu_reference_from_layers,
    derive_metric_depth_packet,
    recompute_and_compare_packet,
    variance_validation_manifest_fields,
)


ATOL = 1e-8
RTOL = 1e-8


def max_abs_rel(actual: Any, expected: Any) -> tuple[float, float]:
    a = np.asarray(actual, dtype=np.float64)
    e = np.asarray(expected, dtype=np.float64)
    diff = np.nan_to_num(a - e, nan=0.0)
    abs_err = float(np.max(np.abs(diff))) if diff.size else 0.0
    denom = np.maximum(np.abs(np.nan_to_num(e, nan=0.0)), 1e-12)
    rel_err = float(np.max(np.abs(diff) / denom)) if diff.size else 0.0
    return abs_err, rel_err


def assert_close_case(name: str, actual: Any, expected: Any, atol: float = ATOL, rtol: float = RTOL) -> dict[str, Any]:
    abs_err, rel_err = max_abs_rel(actual, expected)
    passed = bool(np.allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=True))
    if not passed:
        raise AssertionError(f"{name} failed: actual={actual}, expected={expected}, abs={abs_err}, rel={rel_err}")
    return {
        "name": name,
        "expected": np.asarray(expected).tolist(),
        "actual": np.asarray(actual).tolist(),
        "max_abs_error": abs_err,
        "max_rel_error": rel_err,
        "atol": atol,
        "rtol": rtol,
    }


def test_single_plane_opacity_invariance() -> dict[str, Any]:
    cases = []
    for opacity in [0.10, 0.35, 0.85]:
        packet = cpu_reference_from_layers(
            camera_z=np.asarray([[[20.0]]]),
            alpha=np.asarray([[[opacity]]]),
        )
        cases.append(
            assert_close_case(
                f"expected_camera_z_opacity_{opacity}",
                packet[PRIMARY_DEPTH_TENSOR],
                np.asarray([[20.0]]),
            )
        )
        cases.append(
            assert_close_case(
                f"harmonic_camera_z_opacity_{opacity}",
                packet["harmonic_camera_z"],
                np.asarray([[20.0]]),
            )
        )
    return {"cases": cases}


def test_historical_invalid_inverse_opacity_dependence() -> dict[str, Any]:
    z = 20.0
    values = [opacity / z for opacity in [0.10, 0.35, 0.85]]
    recovered = [1.0 / value for value in values]
    if max(recovered) - min(recovered) <= 1.0:
        raise AssertionError("historical unnormalized inverse-depth did not vary with opacity")
    return {
        "historical_tensor_name": HISTORICAL_INVALID_TENSOR,
        "unnormalized_inverse_values": values,
        "incorrect_1_over_values": recovered,
    }


def test_two_layer_expected_harmonic_variance() -> dict[str, Any]:
    z_near = 10.0
    z_far = 30.0
    alpha_near = 0.4
    alpha_far = 0.5
    weight_near = alpha_near
    weight_far = alpha_far * (1.0 - alpha_near)
    a = weight_near + weight_far
    m1 = weight_near * z_near + weight_far * z_far
    m2 = weight_near * z_near * z_near + weight_far * z_far * z_far
    h = weight_near / z_near + weight_far / z_far
    expected_z = m1 / a
    harmonic_z = a / h
    variance = m2 / a - expected_z * expected_z
    packet = cpu_reference_from_layers(
        camera_z=np.asarray([[[z_near]], [[z_far]]]),
        alpha=np.asarray([[[alpha_near]], [[alpha_far]]]),
    )
    return {
        "cases": [
            assert_close_case("A", packet["accumulated_alpha"], np.asarray([[a]])),
            assert_close_case("M1", packet["weighted_camera_z_sum"], np.asarray([[m1]])),
            assert_close_case("M2", packet["weighted_camera_z_second_moment"], np.asarray([[m2]])),
            assert_close_case("H", packet["weighted_inverse_camera_z_sum"], np.asarray([[h]])),
            assert_close_case("expected_z", packet[PRIMARY_DEPTH_TENSOR], np.asarray([[expected_z]]), atol=1e-6, rtol=1e-6),
            assert_close_case("harmonic_z", packet["harmonic_camera_z"], np.asarray([[harmonic_z]]), atol=1e-6, rtol=1e-6),
            assert_close_case("variance", packet["camera_z_variance"], np.asarray([[variance]]), atol=1e-6, rtol=1e-6),
        ]
    }


def test_zero_and_near_zero_alpha_invalid() -> dict[str, Any]:
    packet_zero = cpu_reference_from_layers(
        camera_z=np.asarray([[[20.0]]]),
        alpha=np.asarray([[[0.0]]]),
    )
    packet_low = cpu_reference_from_layers(
        camera_z=np.asarray([[[20.0]]]),
        alpha=np.asarray([[[DEFAULT_ALPHA_CUTOFF * 0.5]]]),
    )
    if bool(packet_zero["metric_depth_valid_mask"][0, 0]) or bool(packet_low["metric_depth_valid_mask"][0, 0]):
        raise AssertionError("zero / below-cutoff alpha produced a valid metric depth")
    if not math.isnan(float(packet_zero[PRIMARY_DEPTH_TENSOR][0, 0])):
        raise AssertionError("zero alpha expected depth must be NaN")
    return {
        "zero_valid_mask": bool(packet_zero["metric_depth_valid_mask"][0, 0]),
        "below_cutoff_valid_mask": bool(packet_low["metric_depth_valid_mask"][0, 0]),
        "support_floor": DEFAULT_NUMERICAL_SUPPORT_FLOOR,
        "alpha_cutoff": DEFAULT_ALPHA_CUTOFF,
    }


def test_off_axis_camera_z_semantics() -> dict[str, Any]:
    z = np.asarray(
        [
            [[12.0, 12.0], [12.0, 12.0]],
        ]
    )
    alpha = np.ones_like(z) * 0.5
    packet = cpu_reference_from_layers(z, alpha)
    return {
        "case": assert_close_case("off_axis_camera_z_constant", packet[PRIMARY_DEPTH_TENSOR], np.ones((2, 2)) * 12.0),
        "note": "camera-z remains camera-axis depth; evaluator converts ray-distance separately when semantics require it.",
    }


def test_early_termination() -> dict[str, Any]:
    packet = cpu_reference_from_layers(
        camera_z=np.asarray([[[10.0]], [[100.0]]]),
        alpha=np.asarray([[[0.99]], [[0.99]]]),
        early_termination_threshold=0.02,
    )
    return {
        "case": assert_close_case("early_termination_keeps_first_layer_only", packet[PRIMARY_DEPTH_TENSOR], np.asarray([[10.0]])),
        "early_termination_threshold": 0.02,
    }


def test_derived_recomputation() -> dict[str, Any]:
    packet = cpu_reference_from_layers(
        camera_z=np.asarray([[[8.0]], [[18.0]]]),
        alpha=np.asarray([[[0.3]], [[0.7]]]),
    )
    recompute = recompute_and_compare_packet(packet, atol=1e-6, rtol=1e-6)
    if not recompute["passed"]:
        raise AssertionError(recompute)
    return recompute


def test_high_depth_low_variance_forward_error_bound() -> dict[str, Any]:
    packet = derive_metric_depth_packet(
        np.asarray([[1.0]], dtype=np.float32),
        np.asarray([[100.0]], dtype=np.float32),
        np.asarray([[10000.002]], dtype=np.float32),
        np.asarray([[0.01]], dtype=np.float32),
    )
    strict_packet = {name: np.array(value, copy=True) for name, value in packet.items()}
    variance_ref = float(packet["camera_z_variance"][0, 0])
    strict_packet["camera_z_variance"][0, 0] = np.float32(variance_ref + 0.001)
    strict_fixed_tolerance_would_fail = not bool(
        np.allclose(
            strict_packet["camera_z_variance"],
            packet["camera_z_variance"],
            atol=1e-5,
            rtol=0,
            equal_nan=True,
        )
    )
    recompute = recompute_and_compare_packet(strict_packet, atol=1e-6, rtol=1e-6, **variance_validation_manifest_fields())
    variance_row = next(row for row in recompute["rows"] if row["tensor"] == "camera_z_variance")
    if not strict_fixed_tolerance_would_fail or not recompute["passed"]:
        raise AssertionError({"strict_fixed_tolerance_would_fail": strict_fixed_tolerance_would_fail, "recompute": recompute})
    return {
        "strict_fixed_tolerance_would_fail": strict_fixed_tolerance_would_fail,
        "variance_ref": variance_ref,
        "injected_error": 0.001,
        "variance_validation_row": variance_row,
    }


def test_corrupted_variance_beyond_bound_rejected() -> dict[str, Any]:
    packet = derive_metric_depth_packet(
        np.asarray([[1.0]], dtype=np.float32),
        np.asarray([[100.0]], dtype=np.float32),
        np.asarray([[10000.002]], dtype=np.float32),
        np.asarray([[0.01]], dtype=np.float32),
    )
    corrupted = {name: np.array(value, copy=True) for name, value in packet.items()}
    corrupted["camera_z_variance"][0, 0] = np.float32(float(packet["camera_z_variance"][0, 0]) + 0.1)
    recompute = recompute_and_compare_packet(corrupted, **variance_validation_manifest_fields())
    variance_row = next(row for row in recompute["rows"] if row["tensor"] == "camera_z_variance")
    if recompute["passed"] or variance_row["failing_pixel_count"] != 1 or variance_row["max_error_to_bound_ratio"] <= 1:
        raise AssertionError(recompute)
    return {"variance_validation_row": variance_row}


def _variance_row(recompute: dict[str, Any]) -> dict[str, Any]:
    return next(row for row in recompute["rows"] if row["tensor"] == "camera_z_variance")


def test_real_blocker_variance_cancellation_accepted() -> dict[str, Any]:
    a = np.asarray([[0.0999979]], dtype=np.float32)
    mu = 39.64956977159261
    variance_ref = -0.00042971063999175385
    m1 = np.asarray([[float(a[0, 0]) * mu]], dtype=np.float32)
    second = mu * mu + variance_ref
    m2 = np.asarray([[float(a[0, 0]) * second]], dtype=np.float32)
    h = np.asarray([[float(a[0, 0]) / mu]], dtype=np.float32)
    packet = derive_metric_depth_packet(a, m1, m2, h)
    packet["camera_z_variance"][0, 0] = np.float32(-0.0004248161567375064)
    recompute = recompute_and_compare_packet(packet, **variance_validation_manifest_fields())
    row = _variance_row(recompute)
    diag = recompute["diagnostic_tensors"][DIAGNOSTIC_VARIANCE_TENSOR]
    if not recompute["passed"]:
        raise AssertionError(recompute)
    if row["cancellation_accepted_count"] != 1 or row["cancellation_rejected_count"] != 0:
        raise AssertionError(row)
    if float(packet["camera_z_variance"][0, 0]) >= 0:
        raise AssertionError("raw packet variance was unexpectedly modified")
    if float(diag[0, 0]) != 0.0:
        raise AssertionError("diagnostic variance did not clamp accepted cancellation to zero")
    if not (0.0014 < row["max_allowed_error"] < 0.0017):
        raise AssertionError(row)
    return {
        "variance_validation_row": row,
        "raw_variance": float(packet["camera_z_variance"][0, 0]),
        "diagnostic_variance": float(diag[0, 0]),
    }


def test_100k_nonnegativity_unresolved_is_diagnostic_only() -> dict[str, Any]:
    a = np.asarray([[0.9998481273651123]], dtype=np.float32)
    m1 = np.asarray([[180.62217712402344]], dtype=np.float32)
    m2 = np.asarray([[32629.279296875]], dtype=np.float32)
    h = np.asarray([[0.005535]], dtype=np.float32)
    packet = derive_metric_depth_packet(a, m1, m2, h)
    packet["camera_z_variance"][0, 0] = np.float32(-0.046141814440488815)
    recompute = recompute_and_compare_packet(packet, **variance_validation_manifest_fields())
    row = _variance_row(recompute)
    diag = recompute["diagnostic_tensors"][DIAGNOSTIC_VARIANCE_TENSOR]
    diag_mask = recompute["diagnostic_tensors"][DIAGNOSTIC_VARIANCE_VALID_MASK_TENSOR]
    if not recompute["passed"]:
        raise AssertionError(recompute)
    if row["variance_consistency_fail_count"] != 0:
        raise AssertionError(row)
    if row["variance_nonnegativity_unresolved_count"] != 1:
        raise AssertionError(row)
    if not math.isnan(float(diag[0, 0])) or bool(diag_mask[0, 0]):
        raise AssertionError({"diagnostic": diag, "diagnostic_mask": diag_mask})
    if row["variance_packet_negative_max_ratio"] <= 1.0 or row["variance_ref_negative_max_ratio"] <= 1.0:
        raise AssertionError(row)
    return {
        "variance_validation_row": row,
        "raw_variance": float(packet["camera_z_variance"][0, 0]),
        "diagnostic_variance_is_nan": bool(math.isnan(float(diag[0, 0]))),
        "diagnostic_valid": bool(diag_mask[0, 0]),
    }


def test_negative_variance_beyond_forward_bound_rejected() -> dict[str, Any]:
    packet = derive_metric_depth_packet(
        np.asarray([[1.0]], dtype=np.float32),
        np.asarray([[10.0]], dtype=np.float32),
        np.asarray([[100.0]], dtype=np.float32),
        np.asarray([[0.1]], dtype=np.float32),
    )
    packet["camera_z_variance"][0, 0] = np.float32(-0.01)
    recompute = recompute_and_compare_packet(packet, **variance_validation_manifest_fields())
    row = _variance_row(recompute)
    if recompute["passed"] or row["cancellation_rejected_count"] != 1:
        raise AssertionError(recompute)
    return {"variance_validation_row": row}


def test_packet_ref_mismatch_beyond_bound_rejected() -> dict[str, Any]:
    packet = derive_metric_depth_packet(
        np.asarray([[1.0]], dtype=np.float32),
        np.asarray([[10.0]], dtype=np.float32),
        np.asarray([[110.0]], dtype=np.float32),
        np.asarray([[0.1]], dtype=np.float32),
    )
    packet["camera_z_variance"][0, 0] = np.float32(20.0)
    recompute = recompute_and_compare_packet(packet, **variance_validation_manifest_fields())
    row = _variance_row(recompute)
    if recompute["passed"] or row["failing_pixel_count"] != 1:
        raise AssertionError(recompute)
    return {"variance_validation_row": row}


def test_positive_variance_diagnostic_unchanged() -> dict[str, Any]:
    packet = derive_metric_depth_packet(
        np.asarray([[1.0]], dtype=np.float32),
        np.asarray([[10.0]], dtype=np.float32),
        np.asarray([[110.0]], dtype=np.float32),
        np.asarray([[0.1]], dtype=np.float32),
    )
    recompute = recompute_and_compare_packet(packet, **variance_validation_manifest_fields())
    diag = recompute["diagnostic_tensors"][DIAGNOSTIC_VARIANCE_TENSOR]
    if not recompute["passed"]:
        raise AssertionError(recompute)
    if not np.array_equal(packet["camera_z_variance"], diag):
        raise AssertionError("positive diagnostic variance changed")
    return {"raw_variance": float(packet["camera_z_variance"][0, 0]), "diagnostic_variance": float(diag[0, 0])}


def test_tiny_negative_raw_preserved_and_diagnostic_clamped() -> dict[str, Any]:
    packet = derive_metric_depth_packet(
        np.asarray([[1.0]], dtype=np.float32),
        np.asarray([[10.0]], dtype=np.float32),
        np.asarray([[100.0]], dtype=np.float32),
        np.asarray([[0.1]], dtype=np.float32),
        variance_clamp_tolerance=1e-5,
    )
    packet["camera_z_variance"][0, 0] = np.float32(-5e-7)
    raw = float(packet["camera_z_variance"][0, 0])
    if raw >= 0.0:
        raise AssertionError(f"tiny negative raw variance was not preserved: {raw}")
    recompute = recompute_and_compare_packet(packet, **variance_validation_manifest_fields())
    row = _variance_row(recompute)
    diag = recompute["diagnostic_tensors"][DIAGNOSTIC_VARIANCE_TENSOR]
    if not recompute["passed"] or row["diagnostic_zero_clamped_count"] != 1:
        raise AssertionError(recompute)
    if float(diag[0, 0]) != 0.0:
        raise AssertionError("tiny negative diagnostic variance was not clamped")
    return {
        "tiny_negative_raw_preserved": raw,
        "diagnostic_variance": float(diag[0, 0]),
        "variance_clamp_tolerance": DEFAULT_VARIANCE_CLAMP_TOLERANCE,
    }


def test_invalid_mask_nan_behavior_unchanged() -> dict[str, Any]:
    packet = derive_metric_depth_packet(
        np.asarray([[0.0]], dtype=np.float32),
        np.asarray([[0.0]], dtype=np.float32),
        np.asarray([[0.0]], dtype=np.float32),
        np.asarray([[0.0]], dtype=np.float32),
    )
    recompute = recompute_and_compare_packet(packet, **variance_validation_manifest_fields())
    row = _variance_row(recompute)
    if not recompute["passed"] or not row["invalid_nan_ok"]:
        raise AssertionError(recompute)
    return {"valid_mask": bool(packet["metric_depth_valid_mask"][0, 0]), "invalid_nan_ok": row["invalid_nan_ok"]}


def run_tests() -> list[dict[str, Any]]:
    tests: list[tuple[str, Callable[[], dict[str, Any]]]] = [
        ("single_plane_opacity_invariance", test_single_plane_opacity_invariance),
        ("historical_invalid_inverse_opacity_dependence", test_historical_invalid_inverse_opacity_dependence),
        ("two_layer_expected_harmonic_variance", test_two_layer_expected_harmonic_variance),
        ("zero_and_near_zero_alpha_invalid", test_zero_and_near_zero_alpha_invalid),
        ("off_axis_camera_z_semantics", test_off_axis_camera_z_semantics),
        ("alpha_cutoff_and_early_termination", test_early_termination),
        ("derived_tensor_recomputation", test_derived_recomputation),
        ("high_depth_low_variance_forward_error_bound", test_high_depth_low_variance_forward_error_bound),
        ("corrupted_variance_beyond_bound_rejected", test_corrupted_variance_beyond_bound_rejected),
        ("real_blocker_variance_cancellation_accepted", test_real_blocker_variance_cancellation_accepted),
        ("100k_nonnegativity_unresolved_is_diagnostic_only", test_100k_nonnegativity_unresolved_is_diagnostic_only),
        ("negative_variance_beyond_forward_bound_rejected", test_negative_variance_beyond_forward_bound_rejected),
        ("packet_ref_mismatch_beyond_bound_rejected", test_packet_ref_mismatch_beyond_bound_rejected),
        ("positive_variance_diagnostic_unchanged", test_positive_variance_diagnostic_unchanged),
        ("tiny_negative_raw_preserved_and_diagnostic_clamped", test_tiny_negative_raw_preserved_and_diagnostic_clamped),
        ("invalid_mask_nan_behavior_unchanged", test_invalid_mask_nan_behavior_unchanged),
    ]
    results = []
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
        "schema": "metric_depth_packet_cpu_test_matrix_v1",
        "primary_depth_tensor": PRIMARY_DEPTH_TENSOR,
        "required_tensors": METRIC_PACKET_TENSOR_NAMES,
        "test_count": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "results": results,
    }
    print(
        json.dumps(
            payload,
            indent=2,
            default=lambda obj: obj.tolist() if isinstance(obj, np.ndarray) else str(obj),
        )
    )
    if payload["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
