#!/usr/bin/env python3
"""Evaluate a frozen visible surface inside an explicit rectangular ROI.

The numeric construction is inherited from the verified UAVGS LiDAR evaluator,
but the spatial domain is read from a hash-bound, image-defined rectangle
rather than inferred from surveyed points. No method-specific registration or
ICP is permitted.
"""

from __future__ import annotations

import argparse
import ctypes
import csv
import gc
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import Point, box


THRESHOLDS_M = [0.05, 0.10, 0.20]
VERTICAL_SHIFT_M = 23.980600991639484
VOXEL_M = 0.05
ALPHA_MIN = 0.5
PIXEL_STRIDE = 4
THRESHOLD_EPSILON_M = 1e-9


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: dict[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("canonical_sha256", None)
    encoded = json.dumps(
        clean,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def identity(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def read_json(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def load_core(repo: Path) -> Any:
    code_dir = repo.expanduser().resolve() / "code/gcp"
    sys.path.insert(0, str(code_dir))
    import evaluate_m3m_gcp_lidar_formal_v1 as core  # type: ignore

    return core


def load_allowlist(path: Path) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    names = tuple(sorted(str(row["image_name"]) for row in rows))
    if not names or len(names) != len(set(names)):
        raise ValueError(f"invalid image allowlist: {path}")
    return names


def validate_vertical_sanity(path: Path, scene: str) -> dict[str, Any]:
    payload = read_json(path)
    row = payload["scenes"][scene]
    summary = row["summary"]
    limits = {
        "horizontal_distance_m_max": 0.10,
        "raw_minus_ellipsoid_m_abs_median": 0.05,
        "shifted_minus_normal_m_abs_median": 0.05,
        "shifted_minus_normal_m_p95_abs": 0.08,
    }
    actual = {
        "horizontal_distance_m_max": float(summary["horizontal_distance_m_max"]),
        "raw_minus_ellipsoid_m_abs_median": abs(
            float(summary["raw_minus_ellipsoid_m_median"])
        ),
        "shifted_minus_normal_m_abs_median": float(
            summary["shifted_minus_normal_m_abs_median"]
        ),
        "shifted_minus_normal_m_p95_abs": float(
            summary["shifted_minus_normal_m_p95_abs"]
        ),
    }
    failures = {key: {"actual": actual[key], "limit": limit} for key, limit in limits.items() if actual[key] > limit}
    if failures:
        raise ValueError(f"vertical datum sanity failed for {scene}: {failures}")
    return {"status": "PASS", "limits": limits, "actual": actual, "receipt": identity(path)}


def load_rectangular_roi(path: Path, scene: str) -> tuple[Any, dict[str, Any]]:
    """Load and validate one frozen, axis-aligned evaluation rectangle."""

    payload = read_json(path)
    if payload.get("schema") != "uavgs_image_defined_rectangular_roi_v1":
        raise ValueError("unsupported ROI configuration schema")
    if payload.get("status") != "FROZEN":
        raise ValueError("ROI configuration is not frozen")
    if payload.get("coordinate_reference_system") != "EPSG:32649":
        raise ValueError("ROI configuration must use EPSG:32649")
    recorded_sha = payload.get("canonical_sha256")
    if recorded_sha and recorded_sha != canonical_sha256(payload):
        raise ValueError("ROI configuration canonical hash mismatch")
    try:
        definition = dict(payload["scenes"][scene])
    except KeyError as exc:
        raise ValueError(f"scene missing from ROI configuration: {scene}") from exc
    bounds = np.asarray(definition.get("rectangle_bounds_utm49n_m"), dtype=np.float64)
    if bounds.shape != (4,) or not np.all(np.isfinite(bounds)):
        raise ValueError(f"invalid rectangular ROI bounds for {scene}")
    xmin, ymin, xmax, ymax = map(float, bounds)
    if not xmin < xmax or not ymin < ymax:
        raise ValueError(f"non-positive rectangular ROI for {scene}")
    expected_ring = np.asarray(
        [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax], [xmin, ymin]],
        dtype=np.float64,
    )
    ring = np.asarray(definition.get("rectangle_ring_utm49n_m"), dtype=np.float64)
    if ring.shape != (5, 2) or not np.array_equal(ring, expected_ring):
        raise ValueError(f"ROI ring is not the exact closed axis-aligned rectangle for {scene}")
    roi = box(xmin, ymin, xmax, ymax)
    stated_area = float(definition.get("area_m2"))
    if abs(float(roi.area) - stated_area) > 1e-6:
        raise ValueError(f"ROI area mismatch for {scene}: {roi.area} != {stated_area}")
    if definition.get("selection_basis") != "nadir_image_overlap":
        raise ValueError(f"unexpected ROI selection basis for {scene}")
    if not bool(definition.get("lidar_support_verified")):
        raise ValueError(f"LiDAR support was not verified for {scene}")
    definition["configuration"] = identity(path)
    return roi, definition


def reference_binding(
    *,
    scene: str,
    laz_path: Path,
    gcp_csv: Path,
    sim3_json: Path,
    vertical_sanity: Path,
    roi_config: Path,
    roi_definition: dict[str, Any],
    origin: np.ndarray,
    roi: Any,
) -> dict[str, Any]:
    payload = {
        "schema": "uavgs_lidar_rectangular_roi_reference_binding_v1",
        "scene": scene,
        "lidar": identity(laz_path),
        "gcp_csv": identity(gcp_csv),
        "sim3": identity(sim3_json),
        "vertical_sanity": identity(vertical_sanity),
        "roi_configuration": identity(roi_config),
        "roi_definition": roi_definition,
        "normal_minus_ellipsoid_m": VERTICAL_SHIFT_M,
        "roi_geometry": "axis_aligned_rectangle",
        "roi_selection_basis": "nadir_image_overlap",
        "roi_bounds_utm49n": list(map(float, roi.bounds)),
        "roi_area_m2": float(roi.area),
        "reference_voxel_m": VOXEL_M,
        "local_origin_utm49n_normal_height_m": origin.tolist(),
        "method_specific_registration": "FORBIDDEN",
        "icp": "FORBIDDEN",
    }
    payload["canonical_sha256"] = canonical_sha256(payload)
    return payload


def trim_transient_memory() -> None:
    """Return large NumPy temporaries to the constrained container promptly."""

    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def merge_sorted_unique(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Exact linear-memory union of two sorted unique uint64 arrays."""

    if left.dtype != np.uint64 or right.dtype != np.uint64:
        raise TypeError("voxel ID unions require uint64 arrays")
    if left.size == 0:
        return right
    if right.size == 0:
        return left
    insertion = np.searchsorted(left, right, side="left")
    unseen = insertion == left.size
    comparable = ~unseen
    unseen[comparable] = left[insertion[comparable]] != right[comparable]
    additions = right[unseen]
    if additions.size == 0:
        return left
    positions = np.searchsorted(left, additions, side="left")
    positions += np.arange(additions.size, dtype=positions.dtype)
    output = np.empty(left.size + additions.size, dtype=np.uint64)
    take_left = np.ones(output.size, dtype=bool)
    take_left[positions] = False
    output[positions] = additions
    output[take_left] = left
    return output


def merge_voxel_batches_memory_bounded(
    current: np.ndarray, pending: list[np.ndarray]
) -> np.ndarray:
    for batch in pending:
        current = merge_sorted_unique(current, batch)
    return current


def build_reference_batched(
    core: Any,
    laz_dir: Path,
    roi: Any,
    origin: np.ndarray,
    chunk_points: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return exact sorted voxel IDs and their construction audit."""

    selected: list[dict[str, Any]] = []
    voxel_ids = np.empty(0, dtype=np.uint64)
    pending: list[np.ndarray] = []
    # The AutoDL no-GPU container has a 2 GiB cgroup limit even though host
    # memory appears much larger. Merging each one-million-point chunk keeps
    # the exact sorted-set union below that limit even for the 100K scene.
    union_batch_chunks = 1
    raw_points = 0
    roi_points = 0
    for path in sorted(laz_dir.glob("*.laz")):
        with core.laspy.open(path) as reader:
            header = reader.header
            bounds = (
                float(header.mins[0]),
                float(header.mins[1]),
                float(header.maxs[0]),
                float(header.maxs[1]),
            )
            if not core.intersects_bounds(bounds, roi.bounds):
                continue
            crs = header.parse_crs()
            epsg = None if crs is None else crs.to_epsg()
            if epsg != 32649:
                raise ValueError(f"{path.name}: expected EPSG:32649, received {epsg}")
            tile_raw = 0
            tile_roi = 0
            classes: dict[int, int] = {}
            chunk_index = 0
            for points in reader.chunk_iterator(chunk_points):
                chunk_index += 1
                x = np.asarray(points.x, dtype=np.float64)
                y = np.asarray(points.y, dtype=np.float64)
                z = np.asarray(points.z, dtype=np.float64)
                tile_raw += len(x)
                mask = core.contains_xy(roi, x, y)
                if np.any(mask):
                    xyz = np.column_stack(
                        (x[mask], y[mask], z[mask] + VERTICAL_SHIFT_M)
                    )
                    batch_ids = core.voxel_batch_ids(xyz, VOXEL_M, origin)
                    if batch_ids.size:
                        pending.append(batch_ids)
                    tile_roi += int(mask.sum())
                    if hasattr(points, "classification"):
                        cls = np.asarray(points.classification)[mask]
                        values, counts = np.unique(cls, return_counts=True)
                        for value, count in zip(values, counts):
                            key = int(value)
                            classes[key] = classes.get(key, 0) + int(count)
                        del cls, values, counts
                    del xyz, batch_ids
                del points, x, y, z, mask
                if chunk_index % union_batch_chunks == 0:
                    voxel_ids = merge_voxel_batches_memory_bounded(voxel_ids, pending)
                    pending.clear()
                    trim_transient_memory()
                    print(
                        f"reference {path.name} chunks={chunk_index}: "
                        f"{len(voxel_ids):,} unique voxels",
                        flush=True,
                    )
            voxel_ids = merge_voxel_batches_memory_bounded(voxel_ids, pending)
            pending.clear()
            trim_transient_memory()
            raw_points += tile_raw
            roi_points += tile_roi
            selected.append(
                {
                    "path": str(path.resolve()),
                    "name": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "point_count_header": int(header.point_count),
                    "points_streamed": tile_raw,
                    "points_in_roi": tile_roi,
                    "bounds_xyz": [
                        *map(float, header.mins.tolist()),
                        *map(float, header.maxs.tolist()),
                    ],
                    "epsg": epsg,
                    "classification_counts_in_roi": classes,
                }
            )
            print(
                f"reference tile {path.name}: {tile_roi:,}/{tile_raw:,} points in ROI; "
                f"{len(voxel_ids):,} unique voxels",
                flush=True,
            )
    if not selected:
        raise ValueError(f"no LAZ tile intersects ROI under {laz_dir}")
    if len(voxel_ids) < 1000:
        raise ValueError(f"reference point cloud unexpectedly small: {len(voxel_ids)}")
    return voxel_ids, {
        "selected_tiles": selected,
        "raw_points_streamed": raw_points,
        "raw_points_in_roi": roi_points,
        "voxelized_points": int(len(voxel_ids)),
        "voxel_m": VOXEL_M,
        "voxel_representative": "deterministic_voxel_center",
        "voxel_union_batch_chunks": union_batch_chunks,
        "voxel_union_semantics": "exact_sorted_set_union",
        "point_coordinate_frame": "frozen_local_metric_frame",
        "local_origin_utm49n_normal_height_m": origin.tolist(),
        "voxel_id_dtype": str(voxel_ids.dtype),
        "point_dtype": "float64",
    }


def build_or_load_reference(
    core: Any,
    *,
    cache_root: Path,
    binding: dict[str, Any],
    laz_dir: Path,
    roi: Any,
    origin: np.ndarray,
    chunk_points: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    points_path = cache_root / "reference_voxel_centres_local_metric.npy"
    audit_path = cache_root / "reference_audit.json"
    manifest_path = cache_root / "reference_cache_manifest.json"
    if cache_root.exists():
        manifest = read_json(manifest_path)
        stored_points_path = Path(manifest.get("points", {}).get("path", "")).resolve()
        if (
            manifest.get("status") != "PASS_REFERENCE_CACHE"
            or manifest.get("binding") != binding
            or manifest.get("canonical_sha256") != canonical_sha256(manifest)
            or stored_points_path.parent != cache_root
            or manifest.get("points", {}).get("sha256") != sha256_file(stored_points_path)
            or manifest.get("audit", {}).get("sha256") != sha256_file(audit_path)
        ):
            raise ValueError(f"reference cache identity mismatch: {cache_root}")
        if stored_points_path.suffix == ".npz":
            with np.load(stored_points_path, allow_pickle=False) as packet:
                points = packet["points_local_m"]
                cached_origin = packet["local_origin_utm49n_normal_height_m"]
        elif stored_points_path.suffix == ".npy":
            points = np.load(stored_points_path, mmap_mode="r", allow_pickle=False)
            cached_origin = np.asarray(
                manifest["binding"]["local_origin_utm49n_normal_height_m"],
                dtype=np.float64,
            )
        else:
            raise ValueError(f"unsupported reference cache format: {stored_points_path}")
        if points.dtype != np.float64 or not np.array_equal(cached_origin, origin):
            raise ValueError("reference cache coordinate frame mismatch")
        if points.ndim != 2 or points.shape != (manifest["reference_point_count"], 3):
            raise ValueError("reference cache shape mismatch")
        return points, manifest

    cache_root.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix=cache_root.name + ".tmp-", dir=cache_root.parent))
    try:
        voxel_ids, audit = build_reference_batched(
            core,
            laz_dir,
            roi,
            origin,
            chunk_points,
        )
        temp_points = temp_root / points_path.name
        point_count = int(len(voxel_ids))
        point_map = np.lib.format.open_memmap(
            temp_points,
            mode="w+",
            dtype=np.float64,
            shape=(point_count, 3),
        )
        coordinate_chunk = 1_000_000
        for start in range(0, point_count, coordinate_chunk):
            stop = min(start + coordinate_chunk, point_count)
            point_map[start:stop] = core.voxel_centers_local(
                voxel_ids[start:stop], VOXEL_M
            )
            point_map.flush()
            print(
                f"reference coordinates {stop:,}/{point_count:,}",
                flush=True,
            )
        del point_map, voxel_ids
        trim_transient_memory()
        audit["point_storage"] = {
            "format": "npy_float64_nx3",
            "coordinate_write_chunk": coordinate_chunk,
            "memory_mapped": True,
        }
        write_json(temp_root / audit_path.name, audit)
        manifest = {
            "schema": "uavgs_lidar_rectangular_roi_reference_cache_v1",
            "status": "PASS_REFERENCE_CACHE",
            "created_at": now(),
            "binding": binding,
            "reference_point_count": point_count,
            "points": identity(temp_points),
            "audit": identity(temp_root / audit_path.name),
        }
        # Replace temporary absolute paths with the immutable final paths.
        manifest["points"]["path"] = str(points_path.resolve())
        manifest["points"]["format"] = "npy_float64_nx3"
        manifest["audit"]["path"] = str(audit_path.resolve())
        manifest["canonical_sha256"] = canonical_sha256(manifest)
        write_json(temp_root / manifest_path.name, manifest)
        os.rename(temp_root, cache_root)
        points = np.load(points_path, mmap_mode="r", allow_pickle=False)
        return points, manifest
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise


def exact_query_distances(tree: Any, points: np.ndarray, chunk: int, workers: int) -> np.ndarray:
    if points.dtype != np.float64:
        raise TypeError("nearest-neighbour queries require float64 coordinates")
    output = np.empty(len(points), dtype=np.float64)
    for start in range(0, len(points), chunk):
        stop = min(start + chunk, len(points))
        distance, _ = tree.query(points[start:stop], k=1, workers=workers)
        output[start:stop] = distance
    return output


def summarize_exact(
    core: Any,
    reconstruction: np.ndarray,
    reference: np.ndarray,
    query_chunk: int,
    workers: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    if workers != -1 and workers < 1:
        raise ValueError("query workers must be -1 or positive")
    frozen = core.query_distances

    def query(tree: Any, points: np.ndarray, chunk: int) -> np.ndarray:
        return exact_query_distances(tree, points, chunk, workers)

    core.query_distances = query
    try:
        result = core.summarize_distances(
            reconstruction,
            reference,
            THRESHOLDS_M,
            query_chunk,
            THRESHOLD_EPSILON_M,
        )
    finally:
        core.query_distances = frozen
    metrics, recon_to_ref, ref_to_recon = result
    metrics["nearest_neighbor_query_workers"] = workers
    return metrics, recon_to_ref, ref_to_recon


def normalize_packet_manifest(
    core: Any,
    path: Path,
    scene: str,
    expected_names: tuple[str, ...],
) -> tuple[Path, dict[str, Any] | None]:
    payload = read_json(path)
    if payload.get("camera_sets") == "train":
        core.validate_packet_manifest(payload, scene=scene, expected_image_names=expected_names)
        return path, None
    if payload.get("camera_sets") != "frozen_evaluation_allowlist":
        raise ValueError(f"unsupported packet camera_sets={payload.get('camera_sets')!r}")
    alias = dict(payload)
    alias["camera_sets"] = "train"
    if "canonical_sha256" in alias:
        alias["canonical_sha256"] = core.canonical_sha256(alias)
    core.validate_packet_manifest(alias, scene=scene, expected_image_names=expected_names)
    alias_path = path.parent / "depth_export_manifest_heldout_train_alias.json"
    if alias_path.exists() and read_json(alias_path) != alias:
        raise ValueError(f"stale packet alias: {alias_path}")
    if not alias_path.exists():
        write_json(alias_path, alias)
    receipt = {
        "status": "REPRESENTATION_ALIAS_ONLY_IMAGE_LIST_UNCHANGED",
        "source": identity(path),
        "alias": identity(alias_path),
        "changed_field": "camera_sets",
        "source_value": "frozen_evaluation_allowlist",
        "alias_value": "train",
    }
    return alias_path, receipt


def load_surface(path: Path, expected_origin: np.ndarray) -> np.ndarray:
    with np.load(path, allow_pickle=False) as packet:
        points = packet["points_local_m"]
        origin = packet["local_origin_utm49n_normal_height_m"]
    if points.dtype != np.float64 or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"invalid frozen surface array: {path}")
    if not np.array_equal(origin, expected_origin):
        raise ValueError(f"surface local origin differs from scene reference: {path}")
    return points


def self_test(core: Any) -> dict[str, Any]:
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(20260828)
    ref = rng.normal(size=(10000, 3)).astype(np.float64)
    qry = rng.normal(size=(3000, 3)).astype(np.float64)
    tree = cKDTree(ref)
    one = exact_query_distances(tree, qry, 257, 1)
    many = exact_query_distances(tree, qry, 257, -1)
    if not np.array_equal(one, many):
        raise AssertionError(f"parallel exact NN mismatch: {np.max(np.abs(one-many))}")
    raw_ids = rng.integers(0, 2_000_000, size=400_000, dtype=np.uint64)
    left = np.unique(raw_ids[:250_000])
    right = np.unique(raw_ids[150_000:])
    expected_union = np.union1d(left, right)
    actual_union = merge_sorted_unique(left, right)
    if not np.array_equal(expected_union, actual_union):
        raise AssertionError("memory-bounded sorted union differs from np.union1d")
    return {
        "status": "PASS",
        "parallel_exact_nn_array_equal": True,
        "memory_bounded_sorted_union_array_equal": True,
        "memory_bounded_sorted_union_count": int(actual_union.size),
        "max_abs_difference_m": float(np.max(np.abs(one - many))),
        "numeric_core": core.run_numeric_self_tests(VOXEL_M),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-repo", type=Path, required=True)
    parser.add_argument("--scene")
    parser.add_argument("--method-id")
    parser.add_argument("--lidar-root", type=Path)
    parser.add_argument("--gcp-csv", type=Path)
    parser.add_argument("--sim3-json", type=Path)
    parser.add_argument("--vertical-sanity", type=Path)
    parser.add_argument("--roi-config", type=Path)
    parser.add_argument("--reference-cache-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--surface-npz", type=Path)
    source.add_argument("--packet-manifest", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--colmap-model", type=Path)
    parser.add_argument("--allowlist-csv", type=Path)
    parser.add_argument("--laz-chunk-points", type=int, default=1_000_000)
    parser.add_argument("--query-chunk-points", type=int, default=250_000)
    parser.add_argument("--query-workers", type=int, default=-1)
    parser.add_argument("--build-reference-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.benchmark_repo.expanduser().resolve()
    core = load_core(repo)
    if args.self_test:
        print(json.dumps(self_test(core), indent=2, sort_keys=True))
        return 0
    required = [
        "scene",
        "lidar_root",
        "gcp_csv",
        "sim3_json",
        "vertical_sanity",
        "roi_config",
        "reference_cache_root",
    ]
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise ValueError(f"missing required arguments: {missing}")

    lidar_root = args.lidar_root.expanduser().resolve()
    laz_path = lidar_root / "lidars/terra_laz_1_4/cloud0.laz"
    gcp_csv = args.gcp_csv.expanduser().resolve()
    sim3_json = args.sim3_json.expanduser().resolve()
    vertical_sanity = args.vertical_sanity.expanduser().resolve()
    roi_config = args.roi_config.expanduser().resolve()
    cache_root = args.reference_cache_root.expanduser().resolve()
    sim3 = read_json(sim3_json)
    if sim3.get("scene") != args.scene or sim3.get("method_result_refit_forbidden") is not True:
        raise ValueError("common Sim(3) scene/policy mismatch")
    vertical = validate_vertical_sanity(vertical_sanity, args.scene)
    roi, roi_definition = load_rectangular_roi(roi_config, args.scene)
    _, gcp_rows = core.build_roi(gcp_csv, sim3, 0.0)
    outside = [
        row["point_name"]
        for row in gcp_rows
        if not roi.covers(Point(float(row["utm49n_e_m"]), float(row["utm49n_n_m"])))
    ]
    if outside:
        raise ValueError(f"surveyed points outside rectangular ROI for {args.scene}: {outside}")
    origin = core.freeze_local_origin(roi, VOXEL_M)
    binding = reference_binding(
        scene=args.scene,
        laz_path=laz_path,
        gcp_csv=gcp_csv,
        sim3_json=sim3_json,
        vertical_sanity=vertical_sanity,
        roi_config=roi_config,
        roi_definition=roi_definition,
        origin=origin,
        roi=roi,
    )
    reference, reference_manifest = build_or_load_reference(
        core,
        cache_root=cache_root,
        binding=binding,
        laz_dir=laz_path.parent,
        roi=roi,
        origin=origin,
        chunk_points=args.laz_chunk_points,
    )
    if args.build_reference_only:
        print(json.dumps(reference_manifest, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.output_root is None or args.method_id is None:
        raise ValueError("scoring requires --output-root and --method-id")
    if args.surface_npz is None and args.packet_manifest is None:
        raise ValueError("scoring requires --surface-npz or --packet-manifest")

    output_root = args.output_root.expanduser().resolve()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)
    started = time.monotonic()
    packet_alias = None
    surface_audit: dict[str, Any]
    if args.surface_npz is not None:
        surface_path = args.surface_npz.expanduser().resolve()
        reconstruction = load_surface(surface_path, origin)
        surface_audit = {
            "mode": "REUSED_FROZEN_VISIBLE_SURFACE",
            "source": identity(surface_path),
            "voxelized_points": int(len(reconstruction)),
            "local_origin_utm49n_normal_height_m": origin.tolist(),
        }
    else:
        if args.run_root is None or args.colmap_model is None or args.allowlist_csv is None:
            raise ValueError("packet scoring requires run root, COLMAP model and allowlist")
        expected_names = load_allowlist(args.allowlist_csv.expanduser().resolve())
        effective_manifest, packet_alias = normalize_packet_manifest(
            core,
            args.packet_manifest.expanduser().resolve(),
            args.scene,
            expected_names,
        )
        cameras, images = core.read_colmap_model(args.colmap_model.expanduser().resolve())
        reconstruction, surface_audit = core.build_reconstruction(
            args.run_root.expanduser().resolve(),
            cameras,
            images,
            sim3,
            roi,
            ALPHA_MIN,
            PIXEL_STRIDE,
            VOXEL_M,
            origin,
            args.scene,
            expected_names,
            effective_manifest,
        )
        surface_path = output_root / "surface_voxel_centres_local_metric.npz"
        np.savez_compressed(
            surface_path,
            points_local_m=reconstruction,
            local_origin_utm49n_normal_height_m=origin,
        )

    metrics, recon_to_ref, ref_to_recon = summarize_exact(
        core,
        reconstruction,
        reference,
        args.query_chunk_points,
        args.query_workers,
    )
    distance_path = output_root / "nearest_neighbor_distances.npz"
    np.savez_compressed(
        distance_path,
        reconstruction_to_lidar_m=recon_to_ref,
        lidar_to_reconstruction_m=ref_to_recon,
    )
    result = {
        "schema": "uavgs_lidar_rectangular_roi_method_result_v1",
        "protocol_id": "uavgs_lidar_visible_surface_rectangular_roi_v1",
        "status": "COMPLETE_RANKED",
        "created_at": now(),
        "scene": args.scene,
        "method_id": args.method_id,
        "interpretation": "visible-surface geometry within the frozen image-defined rectangular evaluation region",
        "method_specific_registration": "FORBIDDEN",
        "icp": "FORBIDDEN",
        "normal_minus_ellipsoid_m": VERTICAL_SHIFT_M,
        "roi_geometry": "axis_aligned_rectangle",
        "roi_selection_basis": "nadir_image_overlap",
        "roi_configuration": identity(roi_config),
        "roi_definition": roi_definition,
        "roi_bounds_utm49n": list(map(float, roi.bounds)),
        "roi_area_m2": float(roi.area),
        "local_origin_utm49n_normal_height_m": origin.tolist(),
        "thresholds_m": THRESHOLDS_M,
        "threshold_comparison_epsilon_m": THRESHOLD_EPSILON_M,
        "reconstruction_voxel_m": VOXEL_M,
        "reference_voxel_m": VOXEL_M,
        "vertical_datum_sanity": vertical,
        "gcp_rows": gcp_rows,
        "reference_cache": identity(cache_root / "reference_cache_manifest.json"),
        "surface": identity(surface_path),
        "surface_audit": surface_audit,
        "packet_manifest_camera_set_alias": packet_alias,
        "distances": identity(distance_path),
        "metrics": metrics,
        "total_seconds": time.monotonic() - started,
        "peak_rss_gib": core.peak_rss_gib(),
        "evaluator": identity(Path(__file__).resolve()),
        "numeric_self_test": self_test(core),
    }
    result["canonical_sha256"] = canonical_sha256(result)
    write_json(output_root / "metrics.json", result)
    with (output_root / "lidar_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        row = {"scene": args.scene, "method_id": args.method_id, "status": "COMPLETE_RANKED", **metrics}
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
