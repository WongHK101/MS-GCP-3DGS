#!/usr/bin/env python3
"""Focused tests for the GPS/pose association audit."""

from __future__ import annotations

import numpy as np
import pandas as pd

from audit_annotation_gps_pose_association import rank_correlation, scene_image_gps_qc


def test_rank_correlation() -> None:
    frame = pd.DataFrame({"error": [1.0, 2.0, 3.0], "gps": [10.0, 20.0, 30.0]})
    assert rank_correlation(frame, "error", "gps") == 1.0
    frame["gps"] = [30.0, 20.0, 10.0]
    assert rank_correlation(frame, "error", "gps") == -1.0


def test_scene_image_percentiles() -> None:
    scene_record = {
        "raw_model": {
            "cameras": [{"camera_id": 1}],
            "images": [
                {
                    "image_id": 1,
                    "image_name": "a.jpg",
                    "camera_id": 1,
                    "qvec": [1.0, 0.0, 0.0, 0.0],
                    "tvec": [0.0, 0.0, 0.0],
                    "record_sha256": "a" * 64,
                },
                {
                    "image_id": 2,
                    "image_name": "b.jpg",
                    "camera_id": 1,
                    "qvec": [1.0, 0.0, 0.0, 0.0],
                    "tvec": [-1.0, 0.0, 0.0],
                    "record_sha256": "b" * 64,
                },
            ],
        }
    }
    metadata = {
        "a.jpg": {"lat": "0", "lon": "0", "ellipsoid_alt_m": "0"},
        "b.jpg": {"lat": "0", "lon": "0", "ellipsoid_alt_m": "0"},
    }
    result = scene_image_gps_qc(
        "scene",
        scene_record,
        metadata,
        {"enu_origin_lat_lon_alt": [0.0, 0.0, 0.0]},
    ).set_index("image_name")
    assert np.isclose(result.loc["a.jpg", "gps_to_colmap_3d_m"], 0.0)
    assert np.isclose(result.loc["b.jpg", "gps_to_colmap_3d_m"], 1.0)
    assert int(result.loc["b.jpg", "gps_residual_rank_desc"]) == 1
    assert np.isclose(result.loc["b.jpg", "gps_residual_percentile"], 100.0)


def main() -> int:
    tests = [test_rank_correlation, test_scene_image_percentiles]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS {len(tests)}/{len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
