#!/usr/bin/env python3
"""Focused tests for supplement completion and history-hint validation."""

from __future__ import annotations

from manual_gcp_annotator import history_hint_enabled
from validate_v13_uniform_supplement_completion import legacy_history_correction


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


def main() -> int:
    tests = [
        test_history_hint_disabled_by_default_contract,
        test_legacy_history_can_double_shift_corrected_candidate,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS {len(tests)}/{len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
