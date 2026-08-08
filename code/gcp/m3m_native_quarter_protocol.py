"""Core operators for the M3M-GCP native-quarter geometry protocol.

This module contains only representation-independent evaluation operations.  It
does not fit a registration transform and it never rounds an annotated pixel.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROTOCOL_ID = "m3m_gcp_native_quarter_geometry_v2"
PROTOCOL_RELEASE_SCHEMA = "m3m_gcp_native_quarter_protocol_release_v2"
PIXEL_DOMAIN = "colmap_4_0_4_image_undistorter_pinhole_max_1414"
PIXEL_CONVENTION = "zero_based_pixel_centers"
DEFAULT_SUPPORT_FLOOR = 1.0e-6
DEFAULT_MIN_VALID_FRACTION = 0.5
DEFAULT_MIN_VALID_OBSERVATIONS = 4
DEFAULT_MIN_NADIR_OBSERVATIONS = 2
DEFAULT_MIN_OBLIQUE_OBSERVATIONS = 2
DEFAULT_MIN_OBLIQUE_AZIMUTH_BINS = 2
DEFAULT_MIN_OBLIQUE_AZIMUTH_BIN_SEPARATION = 2
AZIMUTH_BIN_COUNT = 8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _bilinear_scalar(array: np.ndarray, u: float, v: float) -> tuple[bool, float, str]:
    """Sample a finite 2-D array with a four-neighbour bilinear stencil.

    Coordinates are zero-based pixel centres.  A full four-neighbour stencil is
    mandatory; no padding, clamping, or extrapolation is permitted.
    """

    values = np.asarray(array)
    if values.ndim != 2:
        raise ValueError(f"bilinear input must be 2-D, got shape={values.shape}")
    if not math.isfinite(float(u)) or not math.isfinite(float(v)):
        return False, math.nan, "nonfinite_pixel_coordinate"
    height, width = values.shape
    x0 = int(math.floor(float(u)))
    y0 = int(math.floor(float(v)))
    x1 = x0 + 1
    y1 = y0 + 1
    if x0 < 0 or y0 < 0 or x1 >= width or y1 >= height:
        return False, math.nan, "bilinear_stencil_out_of_bounds"
    neighbourhood = np.asarray(
        [values[y0, x0], values[y0, x1], values[y1, x0], values[y1, x1]],
        dtype=np.float64,
    )
    if not np.isfinite(neighbourhood).all():
        return False, math.nan, "nonfinite_bilinear_neighbour"
    dx = float(u) - x0
    dy = float(v) - y0
    top = (1.0 - dx) * neighbourhood[0] + dx * neighbourhood[1]
    bottom = (1.0 - dx) * neighbourhood[2] + dx * neighbourhood[3]
    return True, float((1.0 - dy) * top + dy * bottom), ""


def sample_raw_moment_camera_z(
    accumulated_alpha: np.ndarray,
    weighted_camera_z_sum: np.ndarray,
    u: float,
    v: float,
    support_floor: float = DEFAULT_SUPPORT_FLOOR,
) -> dict[str, Any]:
    """Return ``bilinear(M1) / bilinear(A)`` at a floating pixel coordinate.

    The denominator is used only after the strict ``A > support_floor`` gate.
    No epsilon is added.  Interpolating a pre-normalised M1/A image is not this
    operator and is intentionally unsupported here.
    """

    if not math.isfinite(float(support_floor)) or float(support_floor) < 0.0:
        raise ValueError("support_floor must be finite and non-negative")
    alpha = np.asarray(accumulated_alpha)
    moment = np.asarray(weighted_camera_z_sum)
    if alpha.shape != moment.shape:
        raise ValueError(f"A and M1 shapes differ: {alpha.shape} versus {moment.shape}")
    alpha_ok, alpha_interp, alpha_reason = _bilinear_scalar(alpha, u, v)
    if not alpha_ok:
        return {
            "valid": False,
            "failure_reason": alpha_reason,
            "u_px": float(u),
            "v_px": float(v),
            "accumulated_alpha_interp": math.nan,
            "weighted_camera_z_sum_interp": math.nan,
            "camera_z": math.nan,
        }
    moment_ok, moment_interp, moment_reason = _bilinear_scalar(moment, u, v)
    if not moment_ok:
        return {
            "valid": False,
            "failure_reason": moment_reason,
            "u_px": float(u),
            "v_px": float(v),
            "accumulated_alpha_interp": alpha_interp,
            "weighted_camera_z_sum_interp": math.nan,
            "camera_z": math.nan,
        }
    if not alpha_interp > float(support_floor):
        return {
            "valid": False,
            "failure_reason": "interpolated_support_not_above_floor",
            "u_px": float(u),
            "v_px": float(v),
            "accumulated_alpha_interp": alpha_interp,
            "weighted_camera_z_sum_interp": moment_interp,
            "camera_z": math.nan,
        }
    camera_z = moment_interp / alpha_interp
    if not math.isfinite(camera_z) or camera_z <= 0.0:
        return {
            "valid": False,
            "failure_reason": "nonpositive_or_nonfinite_camera_z",
            "u_px": float(u),
            "v_px": float(v),
            "accumulated_alpha_interp": alpha_interp,
            "weighted_camera_z_sum_interp": moment_interp,
            "camera_z": camera_z,
        }
    return {
        "valid": True,
        "failure_reason": "",
        "u_px": float(u),
        "v_px": float(v),
        "accumulated_alpha_interp": alpha_interp,
        "weighted_camera_z_sum_interp": moment_interp,
        "camera_z": camera_z,
    }


def half_pixel_sensitivity(
    accumulated_alpha: np.ndarray,
    weighted_camera_z_sum: np.ndarray,
    u: float,
    v: float,
    support_floor: float = DEFAULT_SUPPORT_FLOOR,
) -> dict[str, Any]:
    offsets = ((-0.5, 0.0), (0.5, 0.0), (0.0, -0.5), (0.0, 0.5))
    centre = sample_raw_moment_camera_z(
        accumulated_alpha,
        weighted_camera_z_sum,
        u,
        v,
        support_floor=support_floor,
    )
    shifted: list[dict[str, Any]] = []
    deltas: list[float] = []
    for du, dv in offsets:
        sample = sample_raw_moment_camera_z(
            accumulated_alpha,
            weighted_camera_z_sum,
            u + du,
            v + dv,
            support_floor=support_floor,
        )
        row = {"du_px": du, "dv_px": dv, **sample}
        shifted.append(row)
        if centre["valid"] and sample["valid"]:
            deltas.append(abs(float(sample["camera_z"]) - float(centre["camera_z"])))
    return {
        "centre": centre,
        "shifted": shifted,
        "valid_shift_count": len(deltas),
        "max_abs_camera_z_delta_model_units": max(deltas) if deltas else None,
        "median_abs_camera_z_delta_model_units": float(np.median(deltas)) if deltas else None,
    }


def geometric_median(
    points: np.ndarray | Sequence[Sequence[float]],
    tolerance: float = 1.0e-10,
    max_iterations: int = 512,
) -> np.ndarray:
    """Compute a rotation-equivariant geometric median with Weiszfeld updates."""

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] == 0:
        raise ValueError("points must be a non-empty Nx3 array")
    if not np.isfinite(values).all():
        raise ValueError("points contain non-finite coordinates")
    if values.shape[0] == 1:
        return values[0].copy()
    estimate = values.mean(axis=0)
    for _ in range(int(max_iterations)):
        distances = np.linalg.norm(values - estimate.reshape(1, 3), axis=1)
        coincident = distances <= float(tolerance)
        noncoincident = ~coincident
        if not np.any(noncoincident):
            return estimate
        weights = 1.0 / distances[noncoincident]
        weiszfeld = np.sum(values[noncoincident] * weights[:, None], axis=0) / float(
            np.sum(weights)
        )
        if np.any(coincident):
            # Modified Weiszfeld step (Vardi-Zhang): a data point is a valid
            # minimizer only when its subgradient ball contains the origin.
            directions = (
                values[noncoincident] - estimate.reshape(1, 3)
            ) / distances[noncoincident, None]
            residual_norm = float(np.linalg.norm(np.sum(directions, axis=0)))
            coincident_count = int(np.count_nonzero(coincident))
            eta = 1.0 if residual_norm <= coincident_count else coincident_count / residual_norm
            updated = (1.0 - eta) * weiszfeld + eta * estimate
        else:
            updated = weiszfeld
        if np.linalg.norm(updated - estimate) <= float(tolerance):
            return updated
        estimate = updated
    raise RuntimeError("geometric median did not converge")


def coverage_gate(
    expected_observation_count: int,
    valid_view_classes: Sequence[str],
    valid_azimuth_bins_45deg: Sequence[int],
    min_valid_fraction: float = DEFAULT_MIN_VALID_FRACTION,
    min_valid_observations: int = DEFAULT_MIN_VALID_OBSERVATIONS,
    min_nadir: int = DEFAULT_MIN_NADIR_OBSERVATIONS,
    min_oblique: int = DEFAULT_MIN_OBLIQUE_OBSERVATIONS,
    min_oblique_azimuth_bins: int = DEFAULT_MIN_OBLIQUE_AZIMUTH_BINS,
    min_oblique_azimuth_bin_separation: int = DEFAULT_MIN_OBLIQUE_AZIMUTH_BIN_SEPARATION,
) -> dict[str, Any]:
    if expected_observation_count <= 0:
        raise ValueError("expected_observation_count must be positive")
    required_total = max(
        int(min_valid_observations),
        int(math.ceil(float(min_valid_fraction) * int(expected_observation_count))),
    )
    normalised = [str(value).strip().lower() for value in valid_view_classes]
    azimuth_bins = [int(value) for value in valid_azimuth_bins_45deg]
    if len(normalised) != len(azimuth_bins):
        raise ValueError("view-class and azimuth-bin sequences must have equal length")
    if any(value not in range(AZIMUTH_BIN_COUNT) for value in azimuth_bins):
        raise ValueError("45-degree azimuth bins must be integers in [0, 7]")
    oblique_bins = sorted(
        {
            azimuth_bin
            for view_class, azimuth_bin in zip(normalised, azimuth_bins, strict=True)
            if view_class == "oblique"
        }
    )
    max_oblique_bin_separation = 0
    for left_index, left in enumerate(oblique_bins):
        for right in oblique_bins[left_index + 1 :]:
            direct = abs(left - right)
            max_oblique_bin_separation = max(
                max_oblique_bin_separation,
                min(direct, AZIMUTH_BIN_COUNT - direct),
            )
    counts = {
        "total": len(normalised),
        "nadir": normalised.count("nadir"),
        "oblique": normalised.count("oblique"),
    }
    failures: list[str] = []
    if counts["total"] < required_total:
        failures.append("insufficient_total_valid_observations")
    if counts["nadir"] < int(min_nadir):
        failures.append("insufficient_valid_nadir_observations")
    if counts["oblique"] < int(min_oblique):
        failures.append("insufficient_valid_oblique_observations")
    if len(oblique_bins) < int(min_oblique_azimuth_bins):
        failures.append("insufficient_valid_oblique_azimuth_bins")
    elif max_oblique_bin_separation < int(min_oblique_azimuth_bin_separation):
        failures.append("insufficient_valid_oblique_azimuth_bin_separation")
    return {
        "passed": not failures,
        "failure_reasons": failures,
        "expected_observation_count": int(expected_observation_count),
        "required_valid_observation_count": required_total,
        "valid_observation_count": counts["total"],
        "valid_nadir_count": counts["nadir"],
        "valid_oblique_count": counts["oblique"],
        "valid_oblique_azimuth_bin_count": len(oblique_bins),
        "valid_oblique_azimuth_bins_45deg": oblique_bins,
        "max_oblique_azimuth_circular_bin_separation": max_oblique_bin_separation,
        "required_oblique_azimuth_bin_count": int(min_oblique_azimuth_bins),
        "required_oblique_azimuth_circular_bin_separation": int(
            min_oblique_azimuth_bin_separation
        ),
    }


def scene_ranking_status(checkpoint_total: int, checkpoint_passed: int) -> dict[str, Any]:
    """Return the non-negotiable complete-scene ranking status.

    Subset residuals remain useful diagnostics, but a scene is eligible for an
    RMSE ranking only when every formal checkpoint passes its point-level
    coverage gate.
    """

    total = int(checkpoint_total)
    passed = int(checkpoint_passed)
    if total <= 0:
        raise ValueError("checkpoint_total must be positive")
    if passed < 0 or passed > total:
        raise ValueError("checkpoint_passed must be in [0, checkpoint_total]")
    complete = passed == total
    return {
        "status": "COMPLETE_RANKED" if complete else "INCOMPLETE_UNRANKED",
        "ranking_eligible": complete,
        "ranking_exclusion_reason": "" if complete else "formal_checkpoint_coverage_incomplete",
        "checkpoint_total": total,
        "checkpoint_passed": passed,
    }


def aggregate_view_groups(observations: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, dict[str, Any]]:
    """Aggregate views without allowing a densely sampled flight strip to dominate.

    First compute a geometric median per ``(view_class, azimuth_bin)`` group,
    then compute a geometric median across the group representatives.
    """

    grouped: dict[tuple[str, int], list[np.ndarray]] = {}
    for row in observations:
        xyz = np.asarray(row["model_xyz"], dtype=np.float64)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError("every observation must contain finite model_xyz[3]")
        key = (str(row["view_class"]).strip().lower(), int(row["azimuth_bin_45deg"]))
        grouped.setdefault(key, []).append(xyz)
    if not grouped:
        raise ValueError("cannot aggregate zero observations")
    representatives = []
    group_rows = []
    for key in sorted(grouped):
        stack = np.vstack(grouped[key])
        centre = geometric_median(stack)
        representatives.append(centre)
        distances = np.linalg.norm(stack - centre.reshape(1, 3), axis=1)
        group_rows.append(
            {
                "view_class": key[0],
                "azimuth_bin_45deg": key[1],
                "observation_count": int(stack.shape[0]),
                "representative_model_xyz": centre.tolist(),
                "within_group_scatter_max_m": float(np.max(distances)),
            }
        )
    representative_stack = np.vstack(representatives)
    aggregate = geometric_median(representative_stack)
    all_points = np.vstack([np.vstack(grouped[key]) for key in sorted(grouped)])
    distances = np.linalg.norm(all_points - aggregate.reshape(1, 3), axis=1)
    return aggregate, {
        "aggregation": "geometric_median_per_view_class_azimuth_bin_then_geometric_median",
        "group_count": len(group_rows),
        "groups": group_rows,
        "scatter_median_m": float(np.median(distances)),
        "scatter_p90_m": float(np.percentile(distances, 90)),
        "scatter_max_m": float(np.max(distances)),
    }


@dataclass(frozen=True)
class Sim3:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation, dtype=np.float64)
        translation = np.asarray(self.translation, dtype=np.float64)
        if rotation.shape != (3, 3) or translation.shape != (3,):
            raise ValueError("Sim3 rotation/translation shapes must be 3x3 and 3")
        if not math.isfinite(float(self.scale)) or float(self.scale) <= 0.0:
            raise ValueError("Sim3 scale must be positive and finite")
        if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
            raise ValueError("Sim3 contains non-finite values")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-8, rtol=0.0):
            raise ValueError("Sim3 rotation is not orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-8, rtol=0.0):
            raise ValueError("Sim3 rotation determinant is not +1")
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)

    def apply(self, points: np.ndarray | Sequence[float]) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        if values.shape == (3,):
            return float(self.scale) * (self.rotation @ values) + self.translation
        if values.ndim == 2 and values.shape[1] == 3:
            return (float(self.scale) * (self.rotation @ values.T)).T + self.translation.reshape(1, 3)
        raise ValueError("Sim3 input must be shape (3,) or (N,3)")

    def rotate_direction(self, direction: np.ndarray | Sequence[float]) -> np.ndarray:
        value = np.asarray(direction, dtype=np.float64)
        if value.shape != (3,):
            raise ValueError("direction must be shape (3,)")
        return self.rotation @ value


def sim3_from_mapping(value: Mapping[str, Any]) -> Sim3:
    transform = value.get("transform", value)
    return Sim3(
        scale=float(transform["scale"]),
        rotation=np.asarray(transform["rotation"], dtype=np.float64),
        translation=np.asarray(transform["translation"], dtype=np.float64),
    )


def residual_statistics(residuals: Iterable[Sequence[float]]) -> dict[str, Any]:
    values = np.asarray(list(residuals), dtype=np.float64)
    if values.size == 0:
        return {
            "count": 0,
            "rmse_h_m": None,
            "rmse_z_m": None,
            "rmse_3d_m": None,
            "median_3d_m": None,
            "p95_3d_m": None,
            "max_3d_m": None,
        }
    values = values.reshape(-1, 3)
    horizontal = np.linalg.norm(values[:, :2], axis=1)
    vertical = np.abs(values[:, 2])
    distance = np.linalg.norm(values, axis=1)
    return {
        "count": int(values.shape[0]),
        "rmse_h_m": float(np.sqrt(np.mean(horizontal * horizontal))),
        "rmse_z_m": float(np.sqrt(np.mean(vertical * vertical))),
        "rmse_3d_m": float(np.sqrt(np.mean(distance * distance))),
        "median_3d_m": float(np.median(distance)),
        "p95_3d_m": float(np.percentile(distance, 95)),
        "max_3d_m": float(np.max(distance)),
    }
