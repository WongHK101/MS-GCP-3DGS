#!/usr/bin/env python3
"""Evaluate frozen M3M-GCP rendered-depth surfaces under formal LiDAR v1.

The numeric core is inherited from the independently audited 3K pilot v0.2,
then wrapped in a six-scene fail-closed contract.  Formal execution uses exact
train-view allowlists, immutable source/implementation identities, no ICP or
method-specific registration, and a new output directory for every attempt.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import laspy
import numpy as np
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely import contains_xy
from shapely.geometry import MultiPoint, box, mapping

from check_m3m_gcp_lidar_formal_launch import validate_launch
from verify_m3m_gcp_lidar_formal_v1 import (
    METRIC_FIELDS,
    SCENE_RANK_KEYS,
    competition_rank_rows,
    sha256_file as verifier_sha256_file,
)

try:
    import resource
except ImportError:  # Windows-only local numeric tests; formal runs are Linux.
    resource = None  # type: ignore[assignment]


PROTOCOL_ID = "m3m_gcp_lidar_rendered_surface_v1"
SOURCE_PROTOCOL_ID = "m3m_gcp_native_quarter_geometry_v2"
PRIMARY_DEPTH = "alpha_normalized_expected_camera_z"
REQUIRED_PACKET_SCHEMA = "ms_gcp_metric_depth_packet_manifest_v2"
EXPECTED_IMAGE_DOMAIN = "colmap_4_0_4_image_undistorter_pinhole_max_1414"
EXPECTED_PIXEL_CONVENTION = "zero_based_pixel_centers"
THRESHOLD_EPSILON_M = 1e-9


@dataclass(frozen=True)
class Camera:
    camera_id: int
    model: str
    width: int
    height: int
    params: np.ndarray


@dataclass(frozen=True)
class Image:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str


CAMERA_MODELS: dict[int, tuple[str, int]] = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def peak_rss_gib() -> float | None:
    if resource is None:
        return None
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)


def read_exact(handle: Any, n: int) -> bytes:
    data = handle.read(n)
    if len(data) != n:
        raise EOFError(f"expected {n} bytes, received {len(data)}")
    return data


def read_c_string(handle: Any) -> str:
    chunks: list[bytes] = []
    while True:
        value = read_exact(handle, 1)
        if value == b"\x00":
            return b"".join(chunks).decode("utf-8")
        chunks.append(value)


def read_colmap_model(model_dir: Path) -> tuple[dict[int, Camera], dict[str, Image]]:
    cameras: dict[int, Camera] = {}
    with (model_dir / "cameras.bin").open("rb") as handle:
        count = struct.unpack("<Q", read_exact(handle, 8))[0]
        for _ in range(count):
            camera_id, model_id, width, height = struct.unpack(
                "<iiQQ", read_exact(handle, 24)
            )
            if model_id not in CAMERA_MODELS:
                raise ValueError(f"unsupported COLMAP camera model id: {model_id}")
            model, num_params = CAMERA_MODELS[model_id]
            params = np.asarray(
                struct.unpack(f"<{num_params}d", read_exact(handle, 8 * num_params)),
                dtype=np.float64,
            )
            cameras[camera_id] = Camera(camera_id, model, width, height, params)

    images: dict[str, Image] = {}
    with (model_dir / "images.bin").open("rb") as handle:
        count = struct.unpack("<Q", read_exact(handle, 8))[0]
        for _ in range(count):
            image_id = struct.unpack("<i", read_exact(handle, 4))[0]
            qvec = np.asarray(struct.unpack("<4d", read_exact(handle, 32)))
            tvec = np.asarray(struct.unpack("<3d", read_exact(handle, 24)))
            camera_id = struct.unpack("<i", read_exact(handle, 4))[0]
            name = read_c_string(handle)
            num_points = struct.unpack("<Q", read_exact(handle, 8))[0]
            handle.seek(24 * num_points, os.SEEK_CUR)
            images[name] = Image(image_id, qvec, tvec, camera_id, name)

    if not cameras or not images:
        raise ValueError("empty COLMAP model")
    for camera in cameras.values():
        if camera.model != "PINHOLE":
            raise ValueError(
                f"formal v1 expects an undistorted PINHOLE model, received {camera.model}"
            )
    return cameras, images


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = qvec
    return np.asarray(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("refusing to write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_roi(
    gcp_csv: Path,
    sim3_payload: dict[str, Any],
    buffer_m: float,
) -> tuple[Any, list[dict[str, Any]]]:
    names = set(sim3_payload["control_points"]) | set(sim3_payload["checkpoint_points"])
    rows = [row for row in load_csv(gcp_csv) if row["point_name"] in names]
    if {row["point_name"] for row in rows} != names:
        missing = sorted(names - {row["point_name"] for row in rows})
        raise ValueError(f"GCP coordinate rows missing: {missing}")

    transformer = Transformer.from_crs(4545, 32649, always_xy=True)
    points: list[tuple[float, float]] = []
    audit: list[dict[str, Any]] = []
    for row in rows:
        if row.get("derived_wgs84_utm49n_e_m") and row.get("derived_wgs84_utm49n_n_m"):
            e = float(row["derived_wgs84_utm49n_e_m"])
            n = float(row["derived_wgs84_utm49n_n_m"])
        else:
            e, n = transformer.transform(
                float(row["cgcs2000_gk_cm108_e_m"]),
                float(row["cgcs2000_gk_cm108_n_m"]),
            )
        points.append((e, n))
        audit.append(
            {
                "point_name": row["point_name"],
                "utm49n_e_m": e,
                "utm49n_n_m": n,
                "normal_height_m": float(row["cgcs2000_normal_height_m"]),
                "ellipsoid_height_m": float(row["wgs84_ellipsoid_height_m"]),
            }
        )
    hull = MultiPoint(points).convex_hull
    if hull.geom_type != "Polygon" or hull.area <= 0:
        raise ValueError("GCP ROI hull is degenerate")
    return hull.buffer(buffer_m), sorted(audit, key=lambda row: row["point_name"])


VOXEL_PACK_BITS = 21
VOXEL_PACK_OFFSET = 1 << (VOXEL_PACK_BITS - 1)
VOXEL_PACK_LIMIT = 1 << VOXEL_PACK_BITS


def freeze_local_origin(roi: Any, voxel_m: float) -> np.ndarray:
    """Return a stable UTM-aligned origin shared by reference and reconstruction."""
    if not math.isfinite(voxel_m) or voxel_m <= 0:
        raise ValueError("voxel size must be finite and positive")
    return np.asarray(
        [
            math.floor(float(roi.bounds[0]) / voxel_m) * voxel_m,
            math.floor(float(roi.bounds[1]) / voxel_m) * voxel_m,
            0.0,
        ],
        dtype=np.float64,
    )


def voxel_batch_ids(points: np.ndarray, voxel_m: float, origin: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.empty(0, dtype=np.uint64)
    if points.dtype != np.float64 or origin.dtype != np.float64:
        raise TypeError("voxel assignment requires float64 points and origin")
    keys = np.floor((points - origin) / voxel_m).astype(np.int64) + VOXEL_PACK_OFFSET
    if np.any(keys < 0) or np.any(keys >= VOXEL_PACK_LIMIT):
        raise ValueError("voxel key exceeds the signed 21-bit packing range")
    packed = (
        (keys[:, 0].astype(np.uint64) << np.uint64(2 * VOXEL_PACK_BITS))
        | (keys[:, 1].astype(np.uint64) << np.uint64(VOXEL_PACK_BITS))
        | keys[:, 2].astype(np.uint64)
    )
    return np.unique(packed)


def voxel_centers_local(voxel_ids: np.ndarray, voxel_m: float) -> np.ndarray:
    """Decode packed voxel IDs as deterministic float64 centres in the local frame."""
    if voxel_ids.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    mask = np.uint64((1 << VOXEL_PACK_BITS) - 1)
    keys = np.column_stack(
        (
            (voxel_ids >> np.uint64(2 * VOXEL_PACK_BITS)) & mask,
            (voxel_ids >> np.uint64(VOXEL_PACK_BITS)) & mask,
            voxel_ids & mask,
        )
    ).astype(np.int64)
    keys -= VOXEL_PACK_OFFSET
    return (keys.astype(np.float64) + 0.5) * voxel_m


def accumulate_voxels(
    current_ids: np.ndarray,
    new_points: np.ndarray,
    voxel_m: float,
    origin: np.ndarray,
) -> np.ndarray:
    new_ids = voxel_batch_ids(new_points, voxel_m, origin)
    if current_ids.size == 0:
        return new_ids
    if new_ids.size == 0:
        return current_ids
    return np.union1d(current_ids, new_ids)


def run_numeric_self_tests(voxel_m: float) -> dict[str, Any]:
    """Fail hard if centimetre distances or deterministic voxelization regress."""
    absolute_origin = np.asarray([221000.0, 2566000.0, 20.0], dtype=np.float64)
    expected = np.asarray([0.01, 0.05, 0.10, 0.20], dtype=np.float64)
    local_reference = np.zeros((1, 3), dtype=np.float64)
    axes: dict[str, float] = {}
    for axis in range(3):
        absolute = np.repeat(absolute_origin[None, :], len(expected), axis=0)
        absolute[:, axis] += expected
        local = absolute - absolute_origin
        measured, _ = cKDTree(local_reference).query(local, k=1, workers=1)
        error = float(np.max(np.abs(measured - expected)))
        axes["xyz"[axis]] = error
        if error > 1e-4:
            raise AssertionError(f"centimetre distance preservation failed on axis {axis}: {error}")

    grid_origin = np.asarray([220999.95, 2565999.95, 0.0], dtype=np.float64)
    samples = np.asarray(
        [
            [221000.001, 2566000.001, 20.001],
            [221000.049, 2566000.049, 20.049],
            [221000.051, 2566000.001, 20.001],
            [221000.001, 2566000.051, 20.001],
            [221000.001, 2566000.001, 20.051],
        ],
        dtype=np.float64,
    )
    ids_forward = accumulate_voxels(
        accumulate_voxels(np.empty(0, dtype=np.uint64), samples[:2], voxel_m, grid_origin),
        samples[2:],
        voxel_m,
        grid_origin,
    )
    ids_reverse = accumulate_voxels(
        accumulate_voxels(np.empty(0, dtype=np.uint64), samples[::-1][:3], voxel_m, grid_origin),
        samples[::-1][3:],
        voxel_m,
        grid_origin,
    )
    if not np.array_equal(ids_forward, ids_reverse):
        raise AssertionError("voxel IDs depend on point or chunk order")
    centers = voxel_centers_local(ids_forward, voxel_m)
    if centers.dtype != np.float64 or len(np.unique(centers, axis=0)) != len(centers):
        raise AssertionError("voxel centres are not unique float64 local coordinates")
    return {
        "status": "PASS",
        "centimetre_axis_max_abs_error_m": axes,
        "required_max_abs_error_m": 1e-4,
        "order_invariance": "PASS",
        "unique_voxel_centres": "PASS",
        "coordinate_dtype": "float64",
        "voxel_representative": "deterministic_voxel_center",
    }


def intersects_bounds(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def build_reference(
    laz_dir: Path,
    roi: Any,
    normal_minus_ellipsoid_m: float,
    voxel_m: float,
    chunk_points: int,
    origin: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    voxel_ids = np.empty(0, dtype=np.uint64)
    raw_points = 0
    roi_points = 0
    for path in sorted(laz_dir.glob("*.laz")):
        with laspy.open(path) as reader:
            header = reader.header
            bounds = (
                float(header.mins[0]),
                float(header.mins[1]),
                float(header.maxs[0]),
                float(header.maxs[1]),
            )
            if not intersects_bounds(bounds, roi.bounds):
                continue
            crs = header.parse_crs()
            epsg = None if crs is None else crs.to_epsg()
            if epsg != 32649:
                raise ValueError(f"{path.name}: expected EPSG:32649, received {epsg}")
            tile_raw = 0
            tile_roi = 0
            classes: dict[int, int] = {}
            for points in reader.chunk_iterator(chunk_points):
                x = np.asarray(points.x, dtype=np.float64)
                y = np.asarray(points.y, dtype=np.float64)
                z = np.asarray(points.z, dtype=np.float64)
                tile_raw += len(x)
                mask = contains_xy(roi, x, y)
                if not np.any(mask):
                    continue
                xyz = np.column_stack(
                    (x[mask], y[mask], z[mask] + normal_minus_ellipsoid_m)
                )
                voxel_ids = accumulate_voxels(voxel_ids, xyz, voxel_m, origin)
                tile_roi += int(mask.sum())
                if hasattr(points, "classification"):
                    cls = np.asarray(points.classification)[mask]
                    values, counts = np.unique(cls, return_counts=True)
                    for value, count in zip(values, counts):
                        key = int(value)
                        classes[key] = classes.get(key, 0) + int(count)
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
                f"reference tile {path.name}: {tile_roi:,}/{tile_raw:,} points in ROI",
                flush=True,
            )
    if not selected:
        raise ValueError(f"no LAZ tile intersects ROI under {laz_dir}")
    points = voxel_centers_local(voxel_ids, voxel_m)
    if len(points) < 1000:
        raise ValueError(f"reference point cloud unexpectedly small: {len(points)}")
    return points, {
        "selected_tiles": selected,
        "raw_points_streamed": raw_points,
        "raw_points_in_roi": roi_points,
        "voxelized_points": int(len(points)),
        "voxel_m": voxel_m,
        "voxel_representative": "deterministic_voxel_center",
        "point_coordinate_frame": "frozen_local_metric_frame",
        "local_origin_utm49n_normal_height_m": origin.tolist(),
        "point_dtype": str(points.dtype),
    }


def validate_packet_manifest(
    payload: dict[str, Any], *, scene: str, expected_image_names: tuple[str, ...]
) -> None:
    required = {
        "schema": REQUIRED_PACKET_SCHEMA,
        "protocol_id": SOURCE_PROTOCOL_ID,
        "scene": scene,
        "primary_depth_tensor": PRIMARY_DEPTH,
        "primary_depth_semantics": "camera_z",
        "image_domain": EXPECTED_IMAGE_DOMAIN,
        "pixel_coordinate_convention": EXPECTED_PIXEL_CONVENTION,
        "camera_z_unit_contract": "frozen_colmap_model_camera_z_units",
        "adapter_conformance_status": "PASS",
    }
    mismatch = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in required.items()
        if payload.get(key) != value
    }
    if mismatch:
        raise ValueError(f"packet manifest contract mismatch: {mismatch}")
    if int(payload.get("rendered_view_count", -1)) != len(expected_image_names):
        raise ValueError("packet count differs from the frozen all-train-view allowlist")
    if payload.get("camera_sets") not in {"train", "frozen_evaluation_allowlist"}:
        raise ValueError("formal v1 requires frozen training-view packets")
    packet_names = tuple(sorted(str(row["image_name"]) for row in payload.get("depth_index", [])))
    if packet_names != expected_image_names:
        raise ValueError("packet image names differ from the frozen scene train-view allowlist")


def build_reconstruction(
    run_root: Path,
    cameras: dict[int, Camera],
    images: dict[str, Image],
    sim3_payload: dict[str, Any],
    roi: Any,
    alpha_min: float,
    pixel_stride: int,
    voxel_m: float,
    origin: np.ndarray,
    scene: str,
    expected_image_names: tuple[str, ...],
) -> tuple[np.ndarray, dict[str, Any]]:
    packets_dir = run_root / "formal_evaluation" / "packets"
    manifest_path = packets_dir / "depth_export_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_packet_manifest(
        payload, scene=scene, expected_image_names=expected_image_names
    )
    entries = payload.get("depth_index", [])
    if len(entries) != len(expected_image_names):
        raise ValueError(
            f"expected {len(expected_image_names)} depth entries, received {len(entries)}"
        )

    transform = sim3_payload["transform"]
    scale = float(transform["scale"])
    rotation = np.asarray(transform["rotation"], dtype=np.float64)
    translation = np.asarray(transform["translation"], dtype=np.float64)
    transformer = Transformer.from_crs(4545, 32649, always_xy=True)
    voxel_ids = np.empty(0, dtype=np.uint64)
    sampled_pixels = 0
    supported_pixels = 0
    roi_samples = 0
    view_rows: list[dict[str, Any]] = []

    for entry in entries:
        image_name = str(entry["image_name"])
        if image_name not in images:
            raise ValueError(f"packet image absent from frozen COLMAP model: {image_name}")
        image = images[image_name]
        camera = cameras[image.camera_id]
        packet_path = packets_dir / Path(str(entry["packet_path"])).name
        with np.load(packet_path, allow_pickle=False) as packet:
            depth = packet[PRIMARY_DEPTH][::pixel_stride, ::pixel_stride]
            alpha = packet["accumulated_alpha"][::pixel_stride, ::pixel_stride]
            valid = packet["metric_depth_valid_mask"][::pixel_stride, ::pixel_stride]
        if depth.shape != alpha.shape or depth.shape != valid.shape:
            raise ValueError(f"packet array shape mismatch: {packet_path}")
        expected_shape = (
            math.ceil(camera.height / pixel_stride),
            math.ceil(camera.width / pixel_stride),
        )
        if depth.shape != expected_shape:
            raise ValueError(
                f"packet/camera shape mismatch: {packet_path.name} {depth.shape} != {expected_shape}"
            )
        v = np.arange(0, camera.height, pixel_stride, dtype=np.float64)[:, None]
        u = np.arange(0, camera.width, pixel_stride, dtype=np.float64)[None, :]
        mask = valid & np.isfinite(depth) & (depth > 0) & (alpha >= alpha_min)
        sampled_pixels += depth.size
        supported_pixels += int(mask.sum())
        if not np.any(mask):
            view_rows.append(
                {"image_name": image_name, "supported_samples": 0, "roi_samples": 0}
            )
            continue

        fx, fy, cx, cy = camera.params[:4]
        z = depth[mask].astype(np.float64)
        x_cam = ((np.broadcast_to(u, depth.shape)[mask] - cx) / fx) * z
        y_cam = ((np.broadcast_to(v, depth.shape)[mask] - cy) / fy) * z
        xyz_cam = np.column_stack((x_cam, y_cam, z))
        world = (xyz_cam - image.tvec) @ qvec_to_rotmat(image.qvec)
        target = scale * (world @ rotation.T) + translation
        e, n = transformer.transform(target[:, 0], target[:, 1])
        inside = contains_xy(roi, e, n)
        if np.any(inside):
            xyz = np.column_stack((e[inside], n[inside], target[inside, 2]))
            voxel_ids = accumulate_voxels(voxel_ids, xyz, voxel_m, origin)
        count_inside = int(inside.sum())
        roi_samples += count_inside
        view_rows.append(
            {
                "image_name": image_name,
                "supported_samples": int(mask.sum()),
                "roi_samples": count_inside,
            }
        )
    points = voxel_centers_local(voxel_ids, voxel_m)
    if len(points) < 1000:
        raise ValueError(f"reconstruction point cloud unexpectedly small: {len(points)}")
    return points, {
        "run_root": str(run_root.resolve()),
        "packet_manifest_path": str(manifest_path.resolve()),
        "packet_manifest_sha256": sha256_file(manifest_path),
        "packet_manifest_canonical_sha256": canonical_sha256(payload),
        "view_count": len(entries),
        "sampled_pixels": sampled_pixels,
        "supported_pixels": supported_pixels,
        "roi_samples_before_voxelization": roi_samples,
        "voxelized_points": int(len(points)),
        "alpha_min": alpha_min,
        "pixel_stride": pixel_stride,
        "voxel_m": voxel_m,
        "voxel_representative": "deterministic_voxel_center",
        "point_coordinate_frame": "frozen_local_metric_frame",
        "local_origin_utm49n_normal_height_m": origin.tolist(),
        "point_dtype": str(points.dtype),
        "view_rows": view_rows,
    }


def query_distances(tree: cKDTree, points: np.ndarray, chunk: int) -> np.ndarray:
    if points.dtype != np.float64:
        raise TypeError("nearest-neighbour queries require float64 local coordinates")
    output = np.empty(len(points), dtype=np.float64)
    for start in range(0, len(points), chunk):
        stop = min(start + chunk, len(points))
        distance, _ = tree.query(points[start:stop], k=1, workers=1)
        output[start:stop] = distance
    return output


def summarize_distances(
    reconstruction: np.ndarray,
    reference: np.ndarray,
    thresholds_m: list[float],
    query_chunk: int,
    threshold_epsilon_m: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    start = time.monotonic()
    reference_tree = cKDTree(reference)
    recon_to_ref = query_distances(reference_tree, reconstruction, query_chunk)
    del reference_tree
    reconstruction_tree = cKDTree(reconstruction)
    ref_to_recon = query_distances(reconstruction_tree, reference, query_chunk)
    del reconstruction_tree

    metrics: dict[str, Any] = {
        "reconstruction_points": int(len(reconstruction)),
        "reference_points": int(len(reference)),
        "accuracy_mean_m": float(np.mean(recon_to_ref)),
        "accuracy_median_m": float(np.median(recon_to_ref)),
        "accuracy_p95_m": float(np.quantile(recon_to_ref, 0.95)),
        "completeness_mean_m": float(np.mean(ref_to_recon)),
        "completeness_median_m": float(np.median(ref_to_recon)),
        "completeness_p95_m": float(np.quantile(ref_to_recon, 0.95)),
        "chamfer_l1_mean_m": float(
            0.5 * (np.mean(recon_to_ref) + np.mean(ref_to_recon))
        ),
        "symmetric_rmse_m": float(
            np.sqrt(0.5 * (np.mean(recon_to_ref**2) + np.mean(ref_to_recon**2)))
        ),
    }
    for threshold in thresholds_m:
        label = f"{int(round(threshold * 100)):02d}cm"
        comparison_threshold = threshold + threshold_epsilon_m
        precision = float(np.mean(recon_to_ref <= comparison_threshold))
        recall = float(np.mean(ref_to_recon <= comparison_threshold))
        fscore = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        metrics[f"precision_{label}"] = precision
        metrics[f"recall_{label}"] = recall
        metrics[f"fscore_{label}"] = fscore
    metrics["nearest_neighbor_seconds"] = time.monotonic() - start
    metrics["threshold_comparison_epsilon_m"] = threshold_epsilon_m
    return metrics, recon_to_ref, ref_to_recon


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--artifact-schema", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--geometry-release-root", type=Path, required=True)
    parser.add_argument("--formal-input-root", type=Path, required=True)
    parser.add_argument("--lidar-inventory", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--lidar-root", type=Path, required=True)
    parser.add_argument(
        "--laz-dir",
        type=Path,
        help="Optional LAZ directory override; defaults to lidar-root/lidars/terra_laz_1_4.",
    )
    parser.add_argument("--colmap-model", type=Path, required=True)
    parser.add_argument("--gcp-csv", type=Path, required=True)
    parser.add_argument("--sim3-json", type=Path, required=True)
    parser.add_argument("--methods-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--method-id", action="append", default=[])
    parser.add_argument("--roi-buffer-m", type=float, default=8.0)
    parser.add_argument("--normal-minus-ellipsoid-m", type=float, default=23.980600991639484)
    parser.add_argument("--alpha-min", type=float, default=0.5)
    parser.add_argument("--pixel-stride", type=int, default=4)
    parser.add_argument("--reconstruction-voxel-m", type=float, default=0.05)
    parser.add_argument("--reference-voxel-m", type=float, default=0.05)
    parser.add_argument("--laz-chunk-points", type=int, default=1_000_000)
    parser.add_argument("--query-chunk-points", type=int, default=250_000)
    parser.add_argument("--thresholds-m", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    parser.add_argument("--threshold-epsilon-m", type=float, default=THRESHOLD_EPSILON_M)
    parser.set_defaults(resume=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    contract_path = args.contract.resolve()
    activation_path = args.activation.resolve()
    schema_path = args.artifact_schema.resolve()
    split_path = args.split.resolve()
    registry_path = args.registry.resolve()
    sim3_path = args.sim3_json.resolve()
    methods_path = args.methods_json.resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    launch_errors = validate_launch(
        repo=repo,
        contract_path=contract_path,
        activation_path=activation_path,
        schema_path=schema_path,
        split_path=split_path,
        registry_path=registry_path,
        geometry_release_root=args.geometry_release_root.resolve(),
        formal_input_root=args.formal_input_root.resolve(),
        colmap_model=args.colmap_model.resolve(),
        lidar_inventory_path=args.lidar_inventory.resolve(),
        gcp_path=args.gcp_csv.resolve(),
        sim3_path=sim3_path,
        methods_path=methods_path,
        scene=args.scene,
        output_root=args.output_root,
    )
    if launch_errors:
        raise ValueError(f"formal launch gate failed before output creation: {launch_errors}")
    frozen_surface = contract["reconstruction_surface"]
    frozen_reference = contract["reference_surface"]
    frozen_metrics = contract["metrics"]
    frozen_lidar = contract["lidar_source"]
    exact_runtime_values = {
        "roi_buffer_m": (args.roi_buffer_m, frozen_reference["roi_buffer_m"]),
        "normal_minus_ellipsoid_m": (
            args.normal_minus_ellipsoid_m,
            frozen_lidar["normal_minus_ellipsoid_m"],
        ),
        "alpha_min": (args.alpha_min, frozen_surface["alpha_min_inclusive"]),
        "pixel_stride": (args.pixel_stride, frozen_surface["pixel_stride"]),
        "reconstruction_voxel_m": (
            args.reconstruction_voxel_m,
            frozen_surface["reconstruction_voxel_m"],
        ),
        "reference_voxel_m": (
            args.reference_voxel_m,
            frozen_reference["reference_voxel_m"],
        ),
        "threshold_epsilon_m": (
            args.threshold_epsilon_m,
            frozen_surface["threshold_comparison_epsilon_m"],
        ),
        "thresholds_m": (list(args.thresholds_m), frozen_metrics["thresholds_m"]),
    }
    changed = {
        key: {"actual": actual, "frozen": frozen}
        for key, (actual, frozen) in exact_runtime_values.items()
        if actual != frozen
    }
    if changed:
        raise ValueError(f"runtime values differ from formal contract: {changed}")
    numeric_self_test = run_numeric_self_tests(args.reference_voxel_m)
    args.output_root.mkdir(parents=True, exist_ok=True)
    laz_dir = (
        args.laz_dir
        if args.laz_dir is not None
        else args.lidar_root / "lidars" / "terra_laz_1_4"
    )

    sim3_payload = json.loads(args.sim3_json.read_text(encoding="utf-8"))
    if sim3_payload.get("protocol_id") != SOURCE_PROTOCOL_ID or sim3_payload.get("scene") != args.scene:
        raise ValueError("common Sim(3) does not match the frozen scene source protocol")
    if sim3_payload.get("method_result_refit_forbidden") is not True:
        raise ValueError("common Sim(3) does not explicitly forbid method-result refitting")

    roi, gcp_rows = build_roi(args.gcp_csv, sim3_payload, args.roi_buffer_m)
    local_origin = freeze_local_origin(roi, args.reference_voxel_m)
    cameras, images = read_colmap_model(args.colmap_model)
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    split_scene = next(row for row in split_payload["scenes"] if row["scene"] == args.scene)
    expected_image_names = tuple(sorted(str(name) for name in split_scene["train_image_names"]))
    methods_payload = json.loads(args.methods_json.read_text(encoding="utf-8"))
    methods = methods_payload["methods"]
    if args.method_id:
        selected_ids = set(args.method_id)
        methods = [method for method in methods if method["method_id"] in selected_ids]
        if {method["method_id"] for method in methods} != selected_ids:
            raise ValueError("one or more --method-id values are absent from methods JSON")
    common_packet_images: tuple[str, ...] | None = None
    packet_inputs: list[dict[str, Any]] = []
    for method in methods:
        manifest_path = (
            Path(method["run_root"])
            / "formal_evaluation"
            / "packets"
            / "depth_export_manifest.json"
        )
        if manifest_path.resolve() != Path(method["packet_manifest_path"]).resolve():
            raise ValueError(f"{method['method_id']}: packet manifest path differs from frozen methods manifest")
        if sha256_file(manifest_path) != method["packet_manifest_sha256"]:
            raise ValueError(f"{method['method_id']}: packet manifest SHA differs from frozen methods manifest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_packet_manifest(
            manifest, scene=args.scene, expected_image_names=expected_image_names
        )
        packet_inputs.append(
            {
                "method_id": method["method_id"],
                "manifest_path": str(manifest_path.resolve()),
                "manifest_sha256": sha256_file(manifest_path),
                "manifest_canonical_sha256": canonical_sha256(manifest),
            }
        )
        packet_images = tuple(sorted(str(row["image_name"]) for row in manifest["depth_index"]))
        if common_packet_images is None:
            common_packet_images = packet_images
        elif packet_images != common_packet_images:
            raise ValueError(
                f"{method['method_id']} does not use the same frozen all-train-view packet allowlist"
            )

    protocol = {
        "schema": "m3m_gcp_lidar_rendered_surface_protocol_v1",
        "protocol_id": PROTOCOL_ID,
        "status": "FORMAL_V1_EXECUTION",
        "scene": args.scene,
        "source_geometry_protocol_id": SOURCE_PROTOCOL_ID,
        "surface_representation": "deterministic_voxel_centres_of_backprojected_alpha_normalized_expected_camera_z",
        "surface_claim": "common rendered expected-depth surface samples; not raw Gaussian centres and not a universal physical mesh",
        "method_specific_registration": "forbidden",
        "icp": "forbidden",
        "lidar_training_access": "forbidden; evaluation only",
        "packet_view_split": "exactly every frozen train view; identical image-name set for every method",
        "packet_image_names": list(common_packet_images or ()),
        "alpha_min": args.alpha_min,
        "pixel_stride": args.pixel_stride,
        "reconstruction_voxel_m": args.reconstruction_voxel_m,
        "reference_voxel_m": args.reference_voxel_m,
        "voxel_representative": "deterministic_voxel_center",
        "voxel_grid_origin_utm49n_normal_height_m": local_origin.tolist(),
        "metric_coordinate_frame": "float64 local metres relative to the frozen voxel-grid origin",
        "absolute_utm_float32": "forbidden",
        "thresholds_m": args.thresholds_m,
        "threshold_comparison_epsilon_m": args.threshold_epsilon_m,
        "numeric_self_test": numeric_self_test,
        "horizontal_frame": "EPSG:32649 WGS 84 / UTM zone 49N",
        "reconstruction_horizontal_conversion": "frozen Sim(3) target EPSG:4545 -> EPSG:32649 with PROJ always_xy",
        "vertical_frame": "1985 National Height Datum normal-height approximation",
        "lidar_vertical_conversion": "LiDAR ellipsoid Z + frozen release-wide median(normal-minus-ellipsoid)",
        "normal_minus_ellipsoid_m": args.normal_minus_ellipsoid_m,
        "vertical_bridge_limitation": "constant release-wide datum bridge; explicitly declared and not a per-method fit",
        "roi_definition": "convex hull of the scene's frozen control+checkpoint GCPs in EPSG:32649, buffered by a fixed distance",
        "roi_buffer_m": args.roi_buffer_m,
        "roi_area_m2": float(roi.area),
        "roi_bounds_utm49n": list(map(float, roi.bounds)),
        "roi_geojson": mapping(roi),
        "gcp_rows": gcp_rows,
        "temporal_note": (
            f"RGB date {next(row for row in contract['scenes'] if row['scene'] == args.scene)['rgb_date']}; "
            f"LiDAR date {next(row for row in contract['scenes'] if row['scene'] == args.scene)['lidar_date']}; "
            "real temporal change remains declared reference uncertainty."
        ),
        "inputs": {
            "contract": str(contract_path),
            "contract_sha256": sha256_file(contract_path),
            "activation_manifest": str(activation_path),
            "activation_manifest_sha256": sha256_file(activation_path),
            "artifact_schema": str(schema_path),
            "artifact_schema_sha256": sha256_file(schema_path),
            "split_manifest": str(split_path),
            "split_manifest_sha256": sha256_file(split_path),
            "method_registry": str(registry_path),
            "method_registry_sha256": sha256_file(registry_path),
            "geometry_release_root": str(args.geometry_release_root.resolve()),
            "geometry_release_manifest_sha256": sha256_file(
                args.geometry_release_root.resolve()
                / contract["source_geometry_binding"]["release_manifest_relative_path"]
            ),
            "formal_input_root": str(args.formal_input_root.resolve()),
            "formal_input_manifest_sha256": sha256_file(
                args.formal_input_root.resolve() / "NATIVE_QUARTER_INPUT_MANIFEST.json"
            ),
            "lidar_inventory": str(args.lidar_inventory.resolve()),
            "lidar_inventory_sha256": sha256_file(args.lidar_inventory.resolve()),
            "laz_dir": str(laz_dir.resolve()),
            "evaluator_script": str(Path(__file__).resolve()),
            "evaluator_script_sha256": sha256_file(Path(__file__)),
            "sim3_json": str(args.sim3_json.resolve()),
            "sim3_sha256": sha256_file(args.sim3_json),
            "gcp_csv": str(args.gcp_csv.resolve()),
            "gcp_csv_sha256": sha256_file(args.gcp_csv),
            "colmap_model": str(args.colmap_model.resolve()),
            "cameras_bin_sha256": sha256_file(args.colmap_model / "cameras.bin"),
            "images_bin_sha256": sha256_file(args.colmap_model / "images.bin"),
            "methods_json": str(methods_path),
            "methods_json_sha256": sha256_file(methods_path),
            "packet_manifests": packet_inputs,
        },
        "runtime": {
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": importlib.metadata.version("scipy"),
            "laspy": importlib.metadata.version("laspy"),
            "lazrs": importlib.metadata.version("lazrs"),
            "pyproj": importlib.metadata.version("pyproj"),
            "shapely": importlib.metadata.version("shapely"),
        },
    }
    crop_manifest_path = laz_dir / "crop_manifest.json"
    if crop_manifest_path.is_file():
        protocol["inputs"]["laz_preselection_manifest"] = str(
            crop_manifest_path.resolve()
        )
        protocol["inputs"]["laz_preselection_manifest_sha256"] = sha256_file(
            crop_manifest_path
        )
    protocol["canonical_sha256_before_reference"] = canonical_sha256(protocol)
    protocol_path = args.output_root / "protocol_manifest.json"
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")

    reference_path = args.output_root / "reference_voxel_centres_local_metric.npz"
    reference_audit_path = args.output_root / "reference_audit.json"
    if args.resume and reference_path.is_file() and reference_audit_path.is_file():
        reference_audit = json.loads(reference_audit_path.read_text(encoding="utf-8"))
        if reference_audit.get("input_binding_canonical_sha256") != protocol["canonical_sha256_before_reference"]:
            raise ValueError("refusing stale --resume reference: protocol/input binding mismatch")
        if reference_audit.get("reference_points_file_sha256") != sha256_file(reference_path):
            raise ValueError("refusing stale --resume reference: point-file SHA256 mismatch")
        with np.load(reference_path, allow_pickle=False) as payload:
            reference = payload["points_local_m"]
            resumed_origin = payload["local_origin_utm49n_normal_height_m"]
        if reference.dtype != np.float64 or not np.array_equal(resumed_origin, local_origin):
            raise ValueError("refusing stale --resume reference: numeric frame mismatch")
        print(f"resumed reference: {len(reference):,} points", flush=True)
    else:
        reference, reference_audit = build_reference(
            laz_dir,
            roi,
            args.normal_minus_ellipsoid_m,
            args.reference_voxel_m,
            args.laz_chunk_points,
            local_origin,
        )
        np.savez_compressed(
            reference_path,
            points_local_m=reference,
            local_origin_utm49n_normal_height_m=local_origin,
        )
        reference_audit["input_binding_canonical_sha256"] = protocol[
            "canonical_sha256_before_reference"
        ]
        reference_audit["reference_points_file_sha256"] = sha256_file(reference_path)
        reference_audit_path.write_text(
            json.dumps(reference_audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    gcp_xyz = np.asarray(
        [
            [row["utm49n_e_m"], row["utm49n_n_m"], row["normal_height_m"]]
            for row in gcp_rows
        ],
        dtype=np.float64,
    )
    gcp_xyz_local = gcp_xyz - local_origin
    reference_tree = cKDTree(reference)
    gcp_distances, gcp_indices = reference_tree.query(gcp_xyz_local, k=1, workers=1)
    del reference_tree
    reference_audit["gcp_nearest_neighbor_diagnostics"] = [
        {
            "point_name": row["point_name"],
            "distance_3d_m": float(distance),
            "delta_e_m": float(reference[index, 0] - target[0]),
            "delta_n_m": float(reference[index, 1] - target[1]),
            "delta_z_m": float(reference[index, 2] - target[2]),
        }
        for row, target, distance, index in zip(
            gcp_rows, gcp_xyz_local, gcp_distances, gcp_indices
        )
    ]
    reference_audit_path.write_text(
        json.dumps(reference_audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    protocol["reference"] = reference_audit
    protocol["canonical_sha256"] = canonical_sha256(protocol)
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_rows: list[dict[str, Any]] = []
    for index, method in enumerate(methods, 1):
        method_id = method["method_id"]
        method_dir = args.output_root / "methods" / method_id
        method_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = method_dir / "metrics.json"
        if args.resume and metrics_path.is_file():
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            if payload.get("protocol_manifest_canonical_sha256") != protocol["canonical_sha256"]:
                raise ValueError(f"refusing stale --resume result for {method_id}: protocol binding mismatch")
            if payload.get("canonical_sha256") != canonical_sha256(
                {key: value for key, value in payload.items() if key != "canonical_sha256"}
            ):
                raise ValueError(f"refusing stale --resume result for {method_id}: canonical hash mismatch")
            summary_rows.append(payload["summary_row"])
            print(f"[{index}/{len(methods)}] resumed {method_id}", flush=True)
            continue
        print(f"[{index}/{len(methods)}] building {method_id} surface", flush=True)
        method_start = time.monotonic()
        reconstruction, surface_audit = build_reconstruction(
            Path(method["run_root"]),
            cameras,
            images,
            sim3_payload,
            roi,
            args.alpha_min,
            args.pixel_stride,
            args.reconstruction_voxel_m,
            local_origin,
            args.scene,
            expected_image_names,
        )
        surface_path = method_dir / "surface_voxel_centres_local_metric.npz"
        np.savez_compressed(
            surface_path,
            points_local_m=reconstruction,
            local_origin_utm49n_normal_height_m=local_origin,
        )
        print(
            f"[{index}/{len(methods)}] {method_id}: {len(reconstruction):,} voxels; querying LiDAR",
            flush=True,
        )
        metrics, recon_to_ref, ref_to_recon = summarize_distances(
            reconstruction,
            reference,
            args.thresholds_m,
            args.query_chunk_points,
            args.threshold_epsilon_m,
        )
        distance_path = method_dir / "nearest_neighbor_distances.npz"
        np.savez_compressed(
            distance_path,
            reconstruction_to_lidar_m=recon_to_ref,
            lidar_to_reconstruction_m=ref_to_recon,
        )
        formal_metrics = {field: metrics[field] for field in METRIC_FIELDS}
        summary_row: dict[str, Any] = {
            "method_id": method_id,
            "method": method["method_name"],
            "input_class": method["input_class"],
            "status": "COMPLETE_RANKED",
            **formal_metrics,
            "total_seconds": time.monotonic() - method_start,
            "nearest_neighbor_seconds": metrics["nearest_neighbor_seconds"],
            "peak_rss_gib": peak_rss_gib(),
            "oom": 0,
        }
        summary_row["method_evidence_sha256"] = canonical_sha256(
            {"surface_audit": surface_audit, "metrics": metrics}
        )
        payload = {
            "schema": "m3m_gcp_lidar_method_result_v1",
            "protocol_id": PROTOCOL_ID,
            "contract_file_sha256": sha256_file(contract_path),
            "activation_manifest_sha256": sha256_file(activation_path),
            "protocol_manifest_canonical_sha256": protocol["canonical_sha256"],
            "scene": args.scene,
            "method_id": method_id,
            "method": method["method_name"],
            "input_class": method["input_class"],
            "model_checkpoint_sha256": method["model_checkpoint_sha256"],
            "recipe_sha256": method["recipe_sha256"],
            "renderer_adapter_sha256": method["renderer_adapter_sha256"],
            "packet_manifest_sha256": surface_audit["packet_manifest_sha256"],
            "surface_npz_sha256": sha256_file(surface_path),
            "distance_npz_sha256": sha256_file(distance_path),
            "reference_npz_sha256": sha256_file(reference_path),
            "evaluator_sha256": sha256_file(Path(__file__).resolve()),
            "verifier_sha256": sha256_file(
                repo / contract["implementation"]["verifier_path"]
            ),
            "artifact_schema_sha256": sha256_file(schema_path),
            "train_view_count": len(expected_image_names),
            "reference_point_count": len(reference),
            "reconstruction_point_count": len(reconstruction),
            "reconstruction_to_lidar_distance_count": len(recon_to_ref),
            "lidar_to_reconstruction_distance_count": len(ref_to_recon),
            "surface_audit": surface_audit,
            "metrics": formal_metrics,
            "summary_row": summary_row,
        }
        payload["canonical_sha256"] = canonical_sha256(payload)
        metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        write_csv(method_dir / "view_surface_counts.csv", surface_audit["view_rows"])
        summary_rows.append(summary_row)
        write_csv(args.output_root / "lidar_metrics.csv", summary_rows)
        print(
            f"[{index}/{len(methods)}] {method_id}: F@10cm={metrics['fscore_10cm']:.6f}, "
            f"Chamfer-L1={metrics['chamfer_l1_mean_m']:.6f} m",
            flush=True,
        )

    for input_class in sorted({row["input_class"] for row in summary_rows}):
        class_rows = [row for row in summary_rows if row["input_class"] == input_class]
        class_ranked = competition_rank_rows(class_rows, SCENE_RANK_KEYS)
        ranks = {row["method_id"]: row["rank"] for row in class_ranked}
        for row in summary_rows:
            if row["input_class"] == input_class:
                row["official_input_class_rank"] = ranks[row["method_id"]]
    descriptive_order = competition_rank_rows(summary_rows, SCENE_RANK_KEYS)
    descriptive_positions = {
        row["method_id"]: index for index, row in enumerate(descriptive_order, 1)
    }
    for row in summary_rows:
        row["combined_descriptive_order_not_official_rank"] = descriptive_positions[
            row["method_id"]
        ]
    write_csv(args.output_root / "lidar_metrics.csv", summary_rows)
    final = {
        "schema": "m3m_gcp_lidar_scene_batch_result_v1",
        "protocol_id": PROTOCOL_ID,
        "contract_file_sha256": sha256_file(contract_path),
        "activation_manifest_sha256": sha256_file(activation_path),
        "scene": args.scene,
        "status": "COMPLETE_RANKED",
        "method_count": len(summary_rows),
        "failed_method_count": sum(row["status"] != "COMPLETE_RANKED" for row in summary_rows),
        "metrics_csv": str((args.output_root / "lidar_metrics.csv").resolve()),
        "protocol_manifest": str(protocol_path.resolve()),
        "protocol_manifest_sha256": sha256_file(protocol_path),
        "primary_ranking_metric": "fscore_10cm_descending",
        "ranking_tiebreakers": [
            "chamfer_l1_mean_m_ascending",
            "precision_10cm_descending",
        ],
        "ranking_numeric_tolerance": 1e-9,
        "all_keys_tied_rule": "same competition rank; method_id lexicographic display; next rank skips tie count",
        "official_ranking_scope": "within_input_class_only",
        "results": summary_rows,
    }
    final["canonical_sha256"] = canonical_sha256(final)
    (args.output_root / "batch_result.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(final, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
