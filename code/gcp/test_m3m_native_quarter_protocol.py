from __future__ import annotations

import math

import numpy as np

from m3m_native_quarter_protocol import (
    aggregate_view_groups,
    coverage_gate,
    geometric_median,
    sample_raw_moment_camera_z,
    scene_ranking_status,
)
from preflight_3dgs_native_quarter_adapter import run_preflight


def test_raw_moment_ratio_is_not_bilinear_normalised_depth() -> None:
    alpha = np.asarray([[0.2, 0.9], [0.4, 0.8]], dtype=np.float32)
    z = np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    sample = sample_raw_moment_camera_z(alpha, alpha * z, 0.5, 0.5)
    assert sample["valid"]
    assert math.isclose(
        sample["camera_z"],
        float(np.mean(alpha.astype(np.float64) * z.astype(np.float64)) / np.mean(alpha.astype(np.float64))),
        rel_tol=0.0,
        abs_tol=1.0e-6,
    )
    assert not math.isclose(sample["camera_z"], float(np.mean(z)))


def test_bilinear_stencil_and_support_gate_are_strict() -> None:
    alpha = np.ones((3, 3), dtype=np.float32)
    moment = alpha * 10.0
    assert not sample_raw_moment_camera_z(alpha, moment, 2.0, 1.0)["valid"]
    floor = np.full((3, 3), 1.0e-6, dtype=np.float32)
    assert not sample_raw_moment_camera_z(floor, floor * 10.0, 1.0, 1.0)["valid"]


def test_geometric_median_rotation_equivariance() -> None:
    points = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.2, 0.1], [0.2, 1.0, 0.3], [0.1, 0.3, 1.0]]
    )
    angle = 0.6
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )
    assert np.allclose(
        geometric_median((rotation @ points.T).T),
        rotation @ geometric_median(points),
        atol=1.0e-9,
        rtol=0.0,
    )


def test_coverage_gate_uses_fraction_and_view_classes() -> None:
    passed = coverage_gate(
        10,
        ["nadir"] * 3 + ["oblique"] * 2,
        [0, 1, 2, 0, 2],
    )
    assert passed["passed"]
    assert passed["required_valid_observation_count"] == 5
    assert passed["valid_oblique_azimuth_bins_45deg"] == [0, 2]
    failed = coverage_gate(12, ["nadir"] * 5 + ["oblique"], [0, 1, 2, 3, 4, 6])
    assert not failed["passed"]
    assert "insufficient_valid_oblique_observations" in failed["failure_reasons"]


def test_coverage_gate_rejects_adjacent_oblique_azimuth_bins() -> None:
    result = coverage_gate(
        8,
        ["nadir", "nadir", "oblique", "oblique"],
        [0, 4, 2, 3],
    )
    assert not result["passed"]
    assert result["valid_oblique_azimuth_bin_count"] == 2
    assert result["max_oblique_azimuth_circular_bin_separation"] == 1
    assert "insufficient_valid_oblique_azimuth_bin_separation" in result["failure_reasons"]


def test_scene_ranking_requires_every_formal_checkpoint() -> None:
    assert scene_ranking_status(4, 4) == {
        "status": "COMPLETE_RANKED",
        "ranking_eligible": True,
        "ranking_exclusion_reason": "",
        "checkpoint_total": 4,
        "checkpoint_passed": 4,
    }
    incomplete = scene_ranking_status(4, 3)
    assert incomplete["status"] == "INCOMPLETE_UNRANKED"
    assert not incomplete["ranking_eligible"]
    assert incomplete["ranking_exclusion_reason"] == "formal_checkpoint_coverage_incomplete"


def test_view_group_aggregation_equalises_duplicate_frames() -> None:
    observations = [
        {"model_xyz": [0.0, 0.0, 0.0], "view_class": "nadir", "azimuth_bin_45deg": 0}
        for _ in range(20)
    ]
    observations += [
        {"model_xyz": [2.0, 0.0, 0.0], "view_class": "oblique", "azimuth_bin_45deg": 4}
    ]
    aggregate, diagnostics = aggregate_view_groups(observations)
    assert diagnostics["group_count"] == 2
    assert np.allclose(aggregate, [1.0, 0.0, 0.0], atol=1.0e-10, rtol=0.0)


def test_3dgs_cpu_preflight() -> None:
    report = run_preflight()
    assert report["passed"], report
    assert report["case_count"] >= 7
