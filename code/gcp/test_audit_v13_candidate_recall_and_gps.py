#!/usr/bin/env python3
"""Focused tests for fixed candidate recall and GPS diagnostics."""

from __future__ import annotations

import math

import numpy as np

from audit_v13_candidate_recall_and_gps import (
    geodetic_to_enu,
    project_point,
    select_supplemental,
)


def test_geodetic_origin_is_zero() -> None:
    origin = [23.182032193425446, 108.27978223896685, 135.06314855072495]
    assert float(np.linalg.norm(geodetic_to_enu(*origin, origin))) < 1e-8


def test_folded_simple_radial_projection_is_rejected() -> None:
    camera = {
        "model": "SIMPLE_RADIAL",
        "params": [3700.0, 2640.0, 1978.0, -0.1148],
    }
    image = {
        "qvec": [1.0, 0.0, 0.0, 0.0],
        "tvec": [0.0, 0.0, 0.0],
    }
    assert project_point(np.asarray([3.0, 0.0, 1.0]), image, camera) is None


def test_candidate_selection_prefers_missing_view_type_and_new_bin() -> None:
    rows = []
    for index, (missing, new_bin, bin_id, center) in enumerate(
        [(False, False, 0, 0.9), (True, True, 3, 0.6), (True, False, 0, 0.8)]
    ):
        rows.append(
            {
                "scene": "s",
                "point_name": "p",
                "image_name": f"i{index}",
                "already_attempted": False,
                "adds_missing_view_type": missing,
                "adds_new_azimuth_bin": new_bin,
                "azimuth_bin_45deg": bin_id,
                "edge_margin_px": 500.0,
                "center_score": center,
            }
        )
    selected = select_supplemental(rows, 1)
    assert selected[0]["image_name"] == "i1"


def main() -> int:
    tests = [
        test_geodetic_origin_is_zero,
        test_folded_simple_radial_projection_is_rejected,
        test_candidate_selection_prefers_missing_view_type_and_new_bin,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)}/{len(tests)} PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
