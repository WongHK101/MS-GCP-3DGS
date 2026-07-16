#!/usr/bin/env python3
"""Focused tests for supplement completion and history-hint validation."""

from __future__ import annotations

import pandas as pd

from manual_gcp_annotator import history_hint_enabled
from validate_v13_uniform_supplement_completion import (
    legacy_history_correction,
    same_image_cross_point_collision_qc,
    status_coordinate_qc,
)


def test_history_hint_disabled_by_default_contract() -> None:
    assert history_hint_enabled("off") is False
    assert history_hint_enabled("legacy_history") is True
    try:
        history_hint_enabled("unknown")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown history hint mode must fail")


def test_legacy_history_can_double_shift_corrected_candidate() -> None:
    candidate = {
        "scene": "scene",
        "point_name": "P",
        "image_name": "new_0005_D.JPG",
        "pixel_x": "100",
        "pixel_y": "100",
    }
    history = []
    for index in range(4):
        history.append(
            {
                "scene": "scene",
                "point_name": f"H{index}",
                "image_name": f"old_{index:04d}_D.JPG",
                "projected_x": "100",
                "projected_y": "100",
                "manual_x": "110",
                "manual_y": "90",
                "visible": "1",
            }
        )
    correction = legacy_history_correction(candidate, history)
    assert correction is not None
    dx, dy, source = correction
    assert (dx, dy) == (10.0, -10.0)
    assert source == "weighted_scene_history"


def test_status_coordinate_contract() -> None:
    annotations = pd.DataFrame(
        [
            {"point_name": "G", "image_name": "a.jpg", "quality": "good", "visible": "1", "manual_x": "10", "manual_y": "20"},
            {"point_name": "A", "image_name": "a.jpg", "quality": "ambiguous", "visible": "1", "manual_x": "", "manual_y": ""},
            {"point_name": "N", "image_name": "a.jpg", "quality": "not_visible", "visible": "0", "manual_x": "", "manual_y": ""},
            {"point_name": "S", "image_name": "a.jpg", "quality": "not_visible", "visible": "0", "manual_x": "30", "manual_y": "40"},
        ]
    )
    cameras = {1: {"width": 100, "height": 80}}
    images = {"a.jpg": {"camera_id": 1}}
    qc = status_coordinate_qc("scene", annotations, cameras, images).set_index("point_name")
    assert qc.loc["G", "classification"] == "good_with_valid_coordinate"
    assert qc.loc["A", "classification"] == "ambiguous_missing_coordinate_recheck"
    assert qc.loc["A", "severity"] == "error"
    assert qc.loc["N", "classification"] == "not_visible_without_coordinate_expected"
    assert qc.loc["S", "classification"] == "not_visible_stale_coordinate_ignored"
    assert qc.loc["S", "severity"] == "warning"


def test_cross_point_same_marker_collision() -> None:
    annotations = pd.DataFrame(
        [
            {"point_name": "P1", "image_name": "a.jpg", "quality": "good", "visible": "1", "manual_x": "10", "manual_y": "20"},
            {"point_name": "P2", "image_name": "a.jpg", "quality": "good", "visible": "1", "manual_x": "13", "manual_y": "24"},
            {"point_name": "P3", "image_name": "a.jpg", "quality": "good", "visible": "1", "manual_x": "100", "manual_y": "100"},
        ]
    )
    qc = same_image_cross_point_collision_qc("scene", annotations, threshold_px=10.0)
    assert len(qc) == 1
    assert set(qc.loc[0, ["point_name_a", "point_name_b"]]) == {"P1", "P2"}
    assert qc.loc[0, "coordinate_distance_px"] == 5.0


def main() -> int:
    tests = [
        test_history_hint_disabled_by_default_contract,
        test_legacy_history_can_double_shift_corrected_candidate,
        test_status_coordinate_contract,
        test_cross_point_same_marker_collision,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS {len(tests)}/{len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
