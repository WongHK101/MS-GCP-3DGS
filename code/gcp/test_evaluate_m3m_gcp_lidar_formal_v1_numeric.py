#!/usr/bin/env python3
"""Deterministic local-float64 voxel tests for the formal evaluator."""

from __future__ import annotations

import importlib.util
import sys
import types

import numpy as np

# The local protocol-test Python intentionally has no LiDAR/CRS stack.  Stub
# only missing heavy modules so the deterministic numeric core can be tested;
# the exact real dependency imports are exercised again in the frozen 901 env.
if importlib.util.find_spec("laspy") is None:
    sys.modules["laspy"] = types.ModuleType("laspy")
if importlib.util.find_spec("pyproj") is None:
    pyproj = types.ModuleType("pyproj")
    pyproj.Transformer = object
    sys.modules["pyproj"] = pyproj
if importlib.util.find_spec("shapely") is None:
    shapely = types.ModuleType("shapely")
    shapely.contains_xy = None
    geometry = types.ModuleType("shapely.geometry")
    geometry.MultiPoint = geometry.box = geometry.mapping = None
    sys.modules["shapely"] = shapely
    sys.modules["shapely.geometry"] = geometry

import evaluate_m3m_gcp_lidar_formal_v1 as evaluator


def main() -> None:
    voxel_m = 0.05
    audit = evaluator.run_numeric_self_tests(voxel_m)
    assert audit["status"] == "PASS"
    assert max(audit["centimetre_axis_max_abs_error_m"].values()) < 1e-4
    rng = np.random.default_rng(20260821)
    origin = np.asarray([221600.0, 2566300.0, 0.0], dtype=np.float64)
    points = origin + rng.uniform([0.0, 0.0, 10.0], [80.0, 80.0, 80.0], size=(20_000, 3))

    def accumulate(chunks: list[np.ndarray]) -> np.ndarray:
        ids = np.empty(0, dtype=np.uint64)
        for chunk in chunks:
            ids = evaluator.accumulate_voxels(ids, chunk, voxel_m, origin)
        return ids

    forward = accumulate([points[:1234], points[1234:8765], points[8765:]])
    reverse = points[::-1]
    backward = accumulate([reverse[:777], reverse[777:15000], reverse[15000:]])
    assert np.array_equal(forward, backward)

    batches = [
        evaluator.voxel_batch_ids(points[:1234], voxel_m, origin),
        evaluator.voxel_batch_ids(points[1234:8765], voxel_m, origin),
        evaluator.voxel_batch_ids(points[8765:], voxel_m, origin),
    ]
    batched = evaluator.merge_voxel_id_batches(
        np.empty(0, dtype=np.uint64), batches
    )
    staged = evaluator.merge_voxel_id_batches(
        evaluator.merge_voxel_id_batches(
            np.empty(0, dtype=np.uint64), batches[:2]
        ),
        batches[2:],
    )
    assert np.array_equal(forward, batched)
    assert np.array_equal(forward, staged)
    centres = evaluator.voxel_centers_local(forward, voxel_m)
    assert centres.dtype == np.float64
    assert len(centres) == len(np.unique(centres, axis=0))
    assert np.max(np.abs(centres)) < 1e5
    print("PASS_FORMAL_V1_NUMERIC")


if __name__ == "__main__":
    main()
