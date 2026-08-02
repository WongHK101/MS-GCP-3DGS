#!/usr/bin/env python3
"""Tests for Stage 0.5 resolution, split, and shared-SfM assets."""

from __future__ import annotations

import copy
import json
import math
import tempfile
from pathlib import Path

import numpy as np

from gs_gcp_stage0_5 import (
    HOLDOUT_SEMANTICS,
    View,
    allocate_quotas,
    canonical_sha256,
    generate_scene_split,
    graphdeco_quarter_dimensions,
    parse_capture_order,
    segment_flight_strata,
    validate_split_manifest,
)


def make_view(index: int, x: float, y: float, *, scene: str = "synthetic") -> View:
    name = f"DJI_202607190101{index % 60:02d}_{index + 1:04d}_D.JPG"
    return View(
        scene=scene,
        image_id=index + 1,
        image_name=name,
        camera_id=1,
        qvec=(1.0, 0.0, 0.0, 0.0),
        tvec=(-x, -y, -10.0),
        center=(x, y, 10.0),
        capture_timestamp=f"202607190101{index % 60:02d}",
        capture_sequence=index + 1,
        image_sha256=f"{index:064x}"[-64:],
        image_bytes=1000 + index,
        decoded_width=5654,
        decoded_height=4098,
    )


def lawnmower_views() -> list[View]:
    rows = []
    index = 0
    for strip in range(4):
        xs = range(12) if strip % 2 == 0 else range(11, -1, -1)
        for x in xs:
            rows.append(make_view(index, float(x), float(strip * 4)))
            index += 1
        if strip < 3:
            rows.append(make_view(index, float(xs[-1]), float(strip * 4 + 2)))
            index += 1
    return rows


def test_quarter_resolution_golden_and_ties() -> None:
    assert graphdeco_quarter_dimensions(5654, 4098) == (1414, 1024)
    assert graphdeco_quarter_dimensions(10, 18) == (2, 4)
    assert graphdeco_quarter_dimensions(14, 22) == (4, 6)


def test_capture_order_and_rejection() -> None:
    assert parse_capture_order("DJI_20260602165038_0001_D.JPG", 7) == (
        "20260602165038", 1, "DJI_20260602165038_0001_D.JPG", 7
    )
    try:
        parse_capture_order("not-canonical.jpg", 1)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid capture name accepted")


def test_segmentation_and_exact_quota() -> None:
    views = lawnmower_views()
    strata, diagnostics = segment_flight_strata(views)
    assert diagnostics["positive_step_median_model_units"] > 0
    assert sum(row["image_count"] for row in strata) == len(views)
    quota = math.ceil(len(views) / 8)
    quotas = allocate_quotas(strata, quota)
    assert sum(quotas.values()) == quota
    split = generate_scene_split(views)
    assert split["test_view_count"] == quota
    assert split["train_view_count"] + split["test_view_count"] == len(views)
    assert set(split["train_image_names"]).isdisjoint(split["test_image_names"])


def test_zero_length_displacement_is_deterministic() -> None:
    views = [make_view(index, float(index), 0.0) for index in range(12)]
    views[5] = make_view(5, 4.0, 0.0)
    a = generate_scene_split(views)
    b = generate_scene_split(copy.deepcopy(views))
    assert canonical_sha256(a) == canonical_sha256(b)


def test_manifest_hash_and_leakage_rejections() -> None:
    scene = generate_scene_split(lawnmower_views())
    manifest = {
        "schema": "gs_gcp_rgb_holdout_split_manifest_v1",
        "split_protocol": "gs_gcp_rgb_holdout_split_v1",
        "holdout_semantics": HOLDOUT_SEMANTICS,
        "scenes": [scene],
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    assert validate_split_manifest(manifest)["passed"]
    bad = copy.deepcopy(manifest)
    bad["scenes"][0]["assignments"].append(copy.deepcopy(bad["scenes"][0]["assignments"][0]))
    bad["manifest_sha256"] = canonical_sha256({key: value for key, value in bad.items() if key != "manifest_sha256"})
    assert not validate_split_manifest(bad)["passed"]


def test_contract_files() -> None:
    root = Path(__file__).resolve().parents[2]
    resolution = json.loads((root / "configs" / "gs_gcp_quarter_resolution_v1.json").read_text(encoding="utf-8"))
    split = json.loads((root / "configs" / "gs_gcp_rgb_holdout_split_v1.json").read_text(encoding="utf-8"))
    suite = json.loads((root / "configs" / "gs_gcp_common_measurement_suite_v1.json").read_text(encoding="utf-8"))
    order = json.loads((root / "configs" / "gs_gcp_scene_execution_order_v1.json").read_text(encoding="utf-8"))
    materialization = json.loads(
        (root / "configs" / "gs_gcp_original_3dgs_camera_materialization_compatibility_v2.json").read_text(encoding="utf-8")
    )
    assert resolution["golden_case"] == {
        "decoded_width": 5654, "decoded_height": 4098, "loaded_width": 1414, "loaded_height": 1024
    }
    assert split["holdout_semantics"] == HOLDOUT_SEMANTICS
    assert suite["formal_depth_formula"] == "M1/A"
    assert len(suite["vgg16_weight_sha256"]) == 64
    assert len(suite["lpips_vgg_linear_weight_sha256"]) == 64
    assert order["resource_feasibility_order"] == [
        "gcp_100000_20260610", "gcp_50000_20260610"
    ]
    assert order["method_qualification_order"] == ["gcp_3000_20260602"]
    assert order["full_matrix_order_after_qualification"] == [
        "gcp_100000_20260610",
        "gcp_50000_20260610",
        "gcp_20000_20260602",
        "gcp_10000_20260610",
        "gcp_5000_20260602",
    ]
    assert materialization["host_allocator_policy"]["environment"] == {
        "MALLOC_TRIM_THRESHOLD_": "0"
    }
    assert materialization["host_allocator_policy"]["explicit_malloc_trim_calls"] is False


def test_frozen_real_split_counts() -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / "configs" / "gs_gcp_rgb_holdout_split_manifest_v1.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validation = validate_split_manifest(manifest)
    assert validation["passed"]
    assert validation["scene_counts"] == {
        "gcp_3000_20260602": {"full": 94, "train": 82, "test": 12},
        "gcp_5000_20260602": {"full": 101, "train": 88, "test": 13},
        "gcp_10000_20260610": {"full": 976, "train": 854, "test": 122},
        "gcp_20000_20260602": {"full": 298, "train": 260, "test": 38},
        "gcp_50000_20260610": {"full": 2208, "train": 1932, "test": 276},
        "gcp_100000_20260610": {"full": 2510, "train": 2196, "test": 314},
    }
    from gs_gcp_stage0_5 import sha256_file
    assert sha256_file(path) == "4535ce1b72dd36a0ba9a46fcf80843bba86b3af1f486ab11fa6d2ca636d1c37e"


TESTS = [
    test_quarter_resolution_golden_and_ties,
    test_capture_order_and_rejection,
    test_segmentation_and_exact_quota,
    test_zero_length_displacement_is_deterministic,
    test_manifest_hash_and_leakage_rejections,
    test_contract_files,
    test_frozen_real_split_counts,
]


def main() -> int:
    for test in TESTS:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS {len(TESTS)}/{len(TESTS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
