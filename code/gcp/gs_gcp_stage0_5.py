#!/usr/bin/env python3
"""GS-GCP Stage 0.5 resolution, RGB holdout, subset, and GT utilities."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
COLMAP_UTILS = ROOT / "code" / "colmap" / "utils"
sys.path.insert(0, str(COLMAP_UTILS))

from read_write_model import (  # noqa: E402
    qvec2rotmat,
    read_cameras_binary,
    read_images_binary,
    write_cameras_binary,
    write_images_binary,
)


RELEASE_ID = "gcp_benchmark_release_v1_3_0_multiview_control_heavy_20260717"
RELEASE_ROOT_DIGEST = "513f8999fe4b110f15bcbecad7932895781cee755ee9ccd7a14ff10298546d75"
RESOLUTION_PROTOCOL = "graphdeco_quarter_resolution_v1"
SPLIT_PROTOCOL = "gs_gcp_rgb_holdout_split_v1"
HOLDOUT_SEMANTICS = "image_loss_holdout_under_shared_all_image_sfm_v1"
PIXEL_CONVENTION = "zero_based_pixel_centers"
TARGET_PIXEL_DOMAIN = "benchmark_colmap_undistorted_pinhole_pixel_domain"
PROJECTION_TOLERANCE_PX = 1e-9
RAY_COORDINATE_TOLERANCE = 1e-12
RAY_ANGLE_TOLERANCE_RAD = 1e-7
TURN_ANGLE_RAD = math.radians(45.0)
TURN_EXPANSION_FRAMES = 2
TURN_CHORD_FRAMES = 3
GAP_MULTIPLIER = 4.0
MIN_STRIP_LENGTH = 8
DJI_NAME = re.compile(
    r"^DJI_(?P<timestamp>\d{14})_(?P<sequence>\d{4})_[A-Za-z]\.JPG$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class View:
    scene: str
    image_id: int
    image_name: str
    camera_id: int
    qvec: tuple[float, float, float, float]
    tvec: tuple[float, float, float]
    center: tuple[float, float, float]
    capture_timestamp: str
    capture_sequence: int
    image_sha256: str
    image_bytes: int
    decoded_width: int
    decoded_height: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def graphdeco_quarter_dimensions(width: int, height: int) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError("decoded image dimensions must be positive")
    return round(width / 4), round(height / 4)


def fov2focal(fov: float, pixels: int) -> float:
    return float(pixels) / (2.0 * math.tan(float(fov) / 2.0))


def focal2fov(focal: float, pixels: int) -> float:
    return 2.0 * math.atan(float(pixels) / (2.0 * float(focal)))


def loaded_pinhole_record(camera: Any, width: int, height: int) -> dict[str, Any]:
    if str(camera.model) == "PINHOLE":
        source_fx, source_fy = float(camera.params[0]), float(camera.params[1])
    elif str(camera.model) == "SIMPLE_PINHOLE":
        source_fx = source_fy = float(camera.params[0])
    else:
        raise ValueError(f"unsupported benchmark camera model: {camera.model}")
    fovx = focal2fov(source_fx, int(camera.width))
    fovy = focal2fov(source_fy, int(camera.height))
    loaded_width, loaded_height = graphdeco_quarter_dimensions(width, height)
    return {
        "model": "PINHOLE",
        "width": loaded_width,
        "height": loaded_height,
        "fx": fov2focal(fovx, loaded_width),
        "fy": fov2focal(fovy, loaded_height),
        "cx": loaded_width / 2.0,
        "cy": loaded_height / 2.0,
        "fovx": fovx,
        "fovy": fovy,
    }


def ray_equivalence(camera: Any, loaded: dict[str, Any], u: float, v: float) -> dict[str, float]:
    if str(camera.model) != "PINHOLE":
        raise ValueError("ray equivalence requires benchmark PINHOLE camera")
    fx, fy, cx, cy = map(float, camera.params)
    x = (float(u) - cx) / fx
    y = (float(v) - cy) / fy
    ul = float(loaded["fx"]) * x + float(loaded["cx"])
    vl = float(loaded["fy"]) * y + float(loaded["cy"])
    xl = (ul - float(loaded["cx"])) / float(loaded["fx"])
    yl = (vl - float(loaded["cy"])) / float(loaded["fy"])
    coordinate_error = max(abs(xl - x), abs(yl - y))
    a = np.asarray([x, y, 1.0], dtype=np.float64)
    b = np.asarray([xl, yl, 1.0], dtype=np.float64)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    angle = math.acos(float(np.clip(np.dot(a, b), -1.0, 1.0)))
    return {"loaded_u": ul, "loaded_v": vl, "coordinate_error": coordinate_error, "angular_error_rad": angle}


def parse_capture_order(image_name: str, image_id: int) -> tuple[str, int, str, int]:
    match = DJI_NAME.fullmatch(Path(image_name).name)
    if match is None:
        raise ValueError(f"image name does not satisfy frozen DJI capture-order schema: {image_name}")
    return match.group("timestamp"), int(match.group("sequence")), Path(image_name).name, int(image_id)


def _camera_center(qvec: Sequence[float], tvec: Sequence[float]) -> tuple[float, float, float]:
    rotation = qvec2rotmat(np.asarray(qvec, dtype=np.float64))
    center = -rotation.T @ np.asarray(tvec, dtype=np.float64)
    return tuple(float(value) for value in center)


def _release_scene_map(camera_provenance: dict[str, Any]) -> dict[str, dict[str, Any]]:
    scenes = camera_provenance.get("scenes", {})
    if not isinstance(scenes, dict):
        raise ValueError("camera provenance scenes must be an object")
    return scenes


def load_release_views(training_manifest_path: Path, camera_provenance_path: Path) -> dict[str, list[View]]:
    training = json.loads(training_manifest_path.read_text(encoding="utf-8"))
    provenance = json.loads(camera_provenance_path.read_text(encoding="utf-8"))
    if training.get("release_id") != RELEASE_ID or provenance.get("release_id") != RELEASE_ID:
        raise ValueError("release identity mismatch")
    scene_map = _release_scene_map(provenance)
    by_scene_training: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in training.get("views", []):
        if str(row.get("training_view_included", "")).lower() != "true":
            continue
        scene = str(row["scene"])
        name = str(row["image_name"])
        if name in by_scene_training[scene]:
            raise ValueError(f"duplicate training-view name: {scene}/{name}")
        by_scene_training[scene][name] = row
    result: dict[str, list[View]] = {}
    for scene, rows in sorted(by_scene_training.items()):
        target_model = scene_map[scene]["target_model"]
        pose_rows = {str(row["image_name"]): row for row in target_model["images"]}
        views: list[View] = []
        seen_ids: set[int] = set()
        seen_order: set[tuple[str, int, str, int]] = set()
        for name, row in rows.items():
            pose = pose_rows.get(name)
            if pose is None:
                raise ValueError(f"training view has no target pose: {scene}/{name}")
            image_id = int(pose["image_id"])
            if image_id in seen_ids:
                raise ValueError(f"duplicate image ID: {scene}/{image_id}")
            order = parse_capture_order(name, image_id)
            if order in seen_order:
                raise ValueError(f"duplicate capture-order key: {scene}/{order}")
            seen_ids.add(image_id)
            seen_order.add(order)
            qvec = tuple(float(value) for value in pose["qvec"])
            tvec = tuple(float(value) for value in pose["tvec"])
            views.append(View(
                scene=scene,
                image_id=image_id,
                image_name=name,
                camera_id=int(pose["camera_id"]),
                qvec=qvec,
                tvec=tvec,
                center=_camera_center(qvec, tvec),
                capture_timestamp=order[0],
                capture_sequence=order[1],
                image_sha256=str(row["target_image_sha256"]),
                image_bytes=int(row["target_image_bytes"]),
                decoded_width=int(row["image_width"]),
                decoded_height=int(row["image_height"]),
            ))
        result[scene] = sorted(views, key=lambda view: parse_capture_order(view.image_name, view.image_id))
    return result


def _circular_difference(a: float, b: float) -> float:
    return math.atan2(math.sin(a - b), math.cos(a - b))


def _view_azimuth_octant(view: View) -> int:
    rotation = qvec2rotmat(np.asarray(view.qvec, dtype=np.float64))
    optical_axis_model = rotation.T @ np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    angle = math.atan2(float(optical_axis_model[1]), float(optical_axis_model[0])) % (2.0 * math.pi)
    return int(math.floor(angle / (math.pi / 4.0))) % 8


def segment_flight_strata(views: Sequence[View]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(views) < 2:
        raise ValueError("scene needs at least two views")
    centers = np.asarray([view.center for view in views], dtype=np.float64)
    deltas = centers[1:, :2] - centers[:-1, :2]
    distances = np.linalg.norm(deltas, axis=1)
    positive = distances[distances > 1e-9]
    if positive.size == 0:
        raise ValueError("all camera-center displacements are zero")
    d50 = float(np.median(positive))
    epsilon = max(1e-9, 1e-6 * d50)
    valid = distances > epsilon
    raw_heading: list[float | None] = [math.atan2(float(v[1]), float(v[0])) if ok else None for v, ok in zip(deltas, valid)]
    headings: list[float] = []
    for index, value in enumerate(raw_heading):
        if value is not None:
            headings.append(value)
            continue
        before = next((raw_heading[j] for j in range(index - 1, -1, -1) if raw_heading[j] is not None), None)
        after = next((raw_heading[j] for j in range(index + 1, len(raw_heading)) if raw_heading[j] is not None), None)
        if before is None and after is None:
            raise ValueError("near-zero displacement heading cannot be recovered")
        if before is None:
            headings.append(float(after))
        elif after is None:
            headings.append(float(before))
        else:
            headings.append(math.atan2(math.sin(before) + math.sin(after), math.cos(before) + math.cos(after)))
    turn_indices: set[int] = set()
    for index, distance in enumerate(distances):
        if float(distance) > GAP_MULTIPLIER * d50:
            turn_indices.update((index, index + 1))
    span = TURN_CHORD_FRAMES
    for index in range(span, len(views) - span):
        incoming = centers[index, :2] - centers[index - span, :2]
        outgoing = centers[index + span, :2] - centers[index, :2]
        if np.linalg.norm(incoming) <= epsilon or np.linalg.norm(outgoing) <= epsilon:
            continue
        hin = math.atan2(float(incoming[1]), float(incoming[0]))
        hout = math.atan2(float(outgoing[1]), float(outgoing[0]))
        if abs(_circular_difference(hout, hin)) > TURN_ANGLE_RAD:
            turn_indices.add(index)
    expanded: set[int] = set()
    for index in turn_indices:
        expanded.update(range(max(0, index - TURN_EXPANSION_FRAMES), min(len(views), index + TURN_EXPANSION_FRAMES + 1)))
    segment_rows: list[tuple[str, int, int]] = []
    cursor = 0
    segment_id = 0
    while cursor < len(views):
        is_transition = cursor in expanded
        end = cursor + 1
        while end < len(views) and ((end in expanded) == is_transition):
            end += 1
        kind = "transition" if is_transition or end - cursor < MIN_STRIP_LENGTH else "strip"
        segment_rows.append((kind, cursor, end))
        cursor = end
        segment_id += 1
    strata: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    image_segment: dict[int, tuple[str, int]] = {}
    for seq, (kind, start, end) in enumerate(segment_rows):
        for index in range(start, end):
            octant = _view_azimuth_octant(views[index])
            strata[(kind, seq, octant)].append(index)
            image_segment[index] = (kind, seq)
    rows = []
    for (kind, seq, octant), indices in sorted(strata.items()):
        rows.append({
            "stratum_id": f"{kind}_{seq:04d}_az{octant}",
            "segment_kind": kind,
            "segment_id": seq,
            "viewing_azimuth_octant": octant,
            "indices": indices,
            "image_count": len(indices),
            "minimum_quota": 1 if kind == "strip" else 0,
            "minimum_image_name": min(views[index].image_name for index in indices),
        })
    diagnostics = {
        "positive_step_median_model_units": d50,
        "near_zero_threshold_model_units": epsilon,
        "near_zero_step_count": int(np.sum(~valid)),
        "large_gap_count": int(np.sum(distances > GAP_MULTIPLIER * d50)),
        "turn_marker_count": len(turn_indices),
        "expanded_turn_image_count": len(expanded),
        "segment_count": len(segment_rows),
        "strip_segment_count": sum(1 for kind, _, _ in segment_rows if kind == "strip"),
        "transition_segment_count": sum(1 for kind, _, _ in segment_rows if kind == "transition"),
        "heading_definition": "camera_center_displacement_in_frozen_colmap_model_xy",
        "heading_count": len(headings),
    }
    return rows, diagnostics


def allocate_quotas(strata: list[dict[str, Any]], target: int) -> dict[str, int]:
    if target <= 0:
        raise ValueError("target test count must be positive")
    lower_total = sum(int(row["minimum_quota"]) for row in strata)
    if lower_total > target:
        raise ValueError("flight-strip lower-bound quotas exceed target test count")
    quotas = {row["stratum_id"]: int(row["minimum_quota"]) for row in strata}
    remaining = target - lower_total
    capacities = {row["stratum_id"]: int(row["image_count"]) - quotas[row["stratum_id"]] for row in strata}
    while remaining:
        eligible = [row for row in strata if capacities[row["stratum_id"]] > 0]
        if not eligible:
            raise ValueError("test quota exceeds available images")
        total_weight = sum(int(row["image_count"]) for row in eligible)
        exact = {row["stratum_id"]: remaining * int(row["image_count"]) / total_weight for row in eligible}
        allocated = 0
        for row in eligible:
            key = row["stratum_id"]
            amount = min(capacities[key], int(math.floor(exact[key])))
            quotas[key] += amount
            capacities[key] -= amount
            allocated += amount
        remaining -= allocated
        if remaining == 0:
            break
        ranked = sorted(
            (row for row in eligible if capacities[row["stratum_id"]] > 0),
            key=lambda row: (
                -(exact[row["stratum_id"]] - math.floor(exact[row["stratum_id"]])),
                row["stratum_id"].encode("utf-8"),
                row["minimum_image_name"].encode("utf-8"),
            ),
        )
        if not ranked:
            raise ValueError("quota remainder cannot be allocated")
        for row in ranked:
            if remaining == 0:
                break
            key = row["stratum_id"]
            quotas[key] += 1
            capacities[key] -= 1
            remaining -= 1
    if sum(quotas.values()) != target:
        raise AssertionError("quota total mismatch")
    return quotas


def _select_stratum_views(views: Sequence[View], indices: Sequence[int], quota: int) -> list[int]:
    if quota == 0:
        return []
    if quota > len(indices):
        raise ValueError("quota exceeds stratum size")
    ordered = list(indices)
    centers = np.asarray([views[index].center[:2] for index in ordered], dtype=np.float64)
    distances = np.linalg.norm(centers[1:] - centers[:-1], axis=1) if len(ordered) > 1 else np.asarray([])
    cumulative = np.concatenate(([0.0], np.cumsum(distances)))
    if float(cumulative[-1]) <= 1e-12:
        target_positions = [(slot + 0.5) * len(ordered) / quota - 0.5 for slot in range(quota)]
        chosen = [min(range(len(ordered)), key=lambda rank: (abs(rank - position), views[ordered[rank]].image_name)) for position in target_positions]
    else:
        target_positions = [(slot + 0.5) * float(cumulative[-1]) / quota for slot in range(quota)]
        chosen = [min(range(len(ordered)), key=lambda rank: (abs(float(cumulative[rank]) - position), views[ordered[rank]].image_name)) for position in target_positions]
    selected: list[int] = []
    used: set[int] = set()
    for preferred in chosen:
        candidates = sorted(
            (rank for rank in range(len(ordered)) if rank not in used),
            key=lambda rank: (abs(float(cumulative[rank]) - float(cumulative[preferred])), views[ordered[rank]].image_name),
        )
        rank = candidates[0]
        used.add(rank)
        selected.append(ordered[rank])
    return sorted(selected)


def generate_scene_split(views: Sequence[View]) -> dict[str, Any]:
    target = math.ceil(len(views) / 8)
    strata, trajectory = segment_flight_strata(views)
    quotas = allocate_quotas(strata, target)
    selected_indices: set[int] = set()
    for row in strata:
        chosen = _select_stratum_views(views, row["indices"], quotas[row["stratum_id"]])
        if selected_indices.intersection(chosen):
            raise ValueError("duplicate test selection across strata")
        selected_indices.update(chosen)
        row["test_quota"] = quotas[row["stratum_id"]]
        row["test_image_names"] = [views[index].image_name for index in chosen]
    if len(selected_indices) != target:
        raise ValueError("selected test count mismatch")
    assignments = []
    stratum_by_index = {}
    for row in strata:
        for index in row["indices"]:
            stratum_by_index[index] = row["stratum_id"]
    for index, view in enumerate(views):
        role = "test" if index in selected_indices else "train"
        assignments.append({
            "scene": view.scene,
            "image_id": view.image_id,
            "image_name": view.image_name,
            "camera_id": view.camera_id,
            "capture_timestamp": view.capture_timestamp,
            "capture_sequence": view.capture_sequence,
            "camera_center_x": format(view.center[0], ".17g"),
            "camera_center_y": format(view.center[1], ".17g"),
            "camera_center_z": format(view.center[2], ".17g"),
            "stratum_id": stratum_by_index[index],
            "split_role": role,
            "image_sha256": view.image_sha256,
            "image_bytes": view.image_bytes,
            "decoded_width": view.decoded_width,
            "decoded_height": view.decoded_height,
        })
    train = [row for row in assignments if row["split_role"] == "train"]
    test = [row for row in assignments if row["split_role"] == "test"]
    return {
        "scene": views[0].scene,
        "full_view_count": len(views),
        "train_view_count": len(train),
        "test_view_count": len(test),
        "target_test_count_rule": "ceil(full_view_count/8)",
        "trajectory_diagnostics": trajectory,
        "strata": [{key: value for key, value in row.items() if key != "indices"} for row in strata],
        "assignments": assignments,
        "train_image_ids": [row["image_id"] for row in train],
        "test_image_ids": [row["image_id"] for row in test],
        "train_image_names": [row["image_name"] for row in train],
        "test_image_names": [row["image_name"] for row in test],
    }


def validate_split_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if manifest.get("schema") != "gs_gcp_rgb_holdout_split_manifest_v1":
        errors.append("unknown split manifest schema")
    if manifest.get("split_protocol") != SPLIT_PROTOCOL or manifest.get("holdout_semantics") != HOLDOUT_SEMANTICS:
        errors.append("split protocol identity mismatch")
    all_names: set[tuple[str, str]] = set()
    counts: dict[str, dict[str, int]] = {}
    for scene in manifest.get("scenes", []):
        assignments = scene.get("assignments", [])
        names = [(str(row.get("scene")), str(row.get("image_name"))) for row in assignments]
        if len(names) != len(set(names)):
            errors.append(f"{scene.get('scene')}: duplicate image assignment")
        if all_names.intersection(names):
            errors.append(f"{scene.get('scene')}: cross-scene duplicate assignment")
        all_names.update(names)
        train = [row for row in assignments if row.get("split_role") == "train"]
        test = [row for row in assignments if row.get("split_role") == "test"]
        if len(assignments) != int(scene.get("full_view_count", -1)):
            errors.append(f"{scene.get('scene')}: full count mismatch")
        if len(test) != math.ceil(len(assignments) / 8):
            errors.append(f"{scene.get('scene')}: test count mismatch")
        if len(train) + len(test) != len(assignments):
            errors.append(f"{scene.get('scene')}: invalid split role")
        counts[str(scene.get("scene"))] = {"full": len(assignments), "train": len(train), "test": len(test)}
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest.get("manifest_sha256") != canonical_sha256(payload):
        errors.append("split manifest canonical hash mismatch")
    return {"passed": not errors, "errors": errors, "scene_counts": counts, "assignment_count": len(all_names)}


def generate_split_manifest(
    training_manifest: Path, camera_provenance: Path, generator_commit: str | None = None
) -> dict[str, Any]:
    views = load_release_views(training_manifest, camera_provenance)
    scenes = [generate_scene_split(rows) for _, rows in sorted(views.items())]
    manifest = {
        "schema": "gs_gcp_rgb_holdout_split_manifest_v1",
        "split_protocol": SPLIT_PROTOCOL,
        "holdout_semantics": HOLDOUT_SEMANTICS,
        "release_id": RELEASE_ID,
        "release_root_digest": RELEASE_ROOT_DIGEST,
        "training_view_manifest_sha256": sha256_file(training_manifest),
        "camera_provenance_manifest_sha256": sha256_file(camera_provenance),
        "generator_provenance": {
            "script_relative_path": "code/gcp/gs_gcp_stage0_5.py",
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "git_commit": generator_commit,
            "working_tree_requirement": "clean committed generator for frozen output",
        },
        "selection_inputs_allowlist": [
            "scene", "image_id", "image_name", "capture_timestamp", "capture_sequence",
            "frozen_colmap_camera_center", "frozen_colmap_pose",
        ],
        "selection_inputs_forbidden": [
            "image_pixels", "exposure_or_low_light_label", "gcp_identity", "gcp_residual",
            "psnr_ssim_lpips", "method_output", "manual_image_quality_selection",
        ],
        "capture_order_authority": "DJI filename timestamp, DJI sequence, image name, COLMAP image ID",
        "scene_xy_definition": "frozen COLMAP model coordinate X/Y axes before any GCP Sim3",
        "scenes": scenes,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    validation = validate_split_manifest(manifest)
    if not validation["passed"]:
        raise ValueError(f"generated split failed validation: {validation['errors']}")
    return manifest


def _copy_or_link(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        target.symlink_to(source.resolve())
    elif mode == "copy":
        shutil.copy2(source, target)
    else:
        raise ValueError(f"unsupported image materialization mode: {mode}")


def materialize_camera_subsets(
    split_manifest: dict[str, Any],
    scene: str,
    source_root: Path,
    output_root: Path,
    image_mode: str,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(output_root)
    scene_row = next((row for row in split_manifest["scenes"] if row["scene"] == scene), None)
    if scene_row is None:
        raise ValueError(f"scene is absent from split manifest: {scene}")
    sparse = source_root / "sparse" / "0"
    cameras = read_cameras_binary(sparse / "cameras.bin")
    images = read_images_binary(sparse / "images.bin")
    by_name = {image.name: image for image in images.values()}
    full_sfm = output_root / "common_full_sfm"
    train_root = output_root / "train_camera_subset"
    test_root = output_root / "test_camera_subset"
    output_root.mkdir(parents=True)
    for name in ("cameras.bin", "images.bin", "points3D.bin", "points3D.ply"):
        source = sparse / name
        if not source.is_file():
            raise FileNotFoundError(source)
        _copy_or_link(source, full_sfm / name, "copy")
    records = []
    for role, role_root in (("train", train_root), ("test", test_root)):
        assignments = [row for row in scene_row["assignments"] if row["split_role"] == role]
        selected_images = {}
        camera_ids = set()
        for row in assignments:
            name = row["image_name"]
            image = by_name.get(name)
            if image is None or int(image.id) != int(row["image_id"]):
                raise ValueError(f"COLMAP image identity mismatch: {name}")
            selected_images[int(image.id)] = image
            camera_ids.add(int(image.camera_id))
            source_image = source_root / "images" / name
            if not source_image.is_file() or sha256_file(source_image) != row["image_sha256"]:
                raise ValueError(f"source image hash mismatch: {name}")
            _copy_or_link(source_image, role_root / "images" / name, image_mode)
        selected_cameras = {camera_id: cameras[camera_id] for camera_id in sorted(camera_ids)}
        model_root = role_root / "sparse" / "0"
        model_root.mkdir(parents=True)
        write_cameras_binary(selected_cameras, model_root / "cameras.bin")
        write_images_binary(selected_images, model_root / "images.bin")
        shutil.copy2(sparse / "points3D.ply", model_root / "points3D.ply")
        if (model_root / "points3D.bin").exists():
            raise AssertionError("camera subset must not contain points3D.bin")
        records.append({
            "role": role,
            "root_relative_to_asset_root": role_root.relative_to(output_root).as_posix(),
            "image_count": len(assignments),
            "camera_count": len(selected_cameras),
            "image_names": [row["image_name"] for row in assignments],
            "cameras_bin_sha256": sha256_file(model_root / "cameras.bin"),
            "images_bin_sha256": sha256_file(model_root / "images.bin"),
            "shared_initial_ply_sha256": sha256_file(model_root / "points3D.ply"),
            "points3d_tracks_present": False,
        })
    manifest = {
        "schema": "gs_gcp_shared_all_image_sfm_camera_subsets_v1",
        "scene": scene,
        "holdout_semantics": HOLDOUT_SEMANTICS,
        "release_root_digest": RELEASE_ROOT_DIGEST,
        "split_manifest_sha256": split_manifest["manifest_sha256"],
        "image_materialization": image_mode,
        "common_full_sfm": {
            "role": "frozen_provenance_and_shared_initialization_not_direct_training_source",
            "cameras_bin_sha256": sha256_file(full_sfm / "cameras.bin"),
            "images_bin_sha256": sha256_file(full_sfm / "images.bin"),
            "points3D_bin_sha256": sha256_file(full_sfm / "points3D.bin"),
            "initial_ply_sha256": sha256_file(full_sfm / "points3D.ply"),
        },
        "training_allowlist": ["shared_point_xyz", "shared_point_rgb", "shared_point_error_if_admitted", "train_camera_intrinsics", "train_camera_extrinsics", "train_rgb"],
        "training_forbidden": ["test_rgb", "test_learned_features", "test_appearance_embeddings", "test_2d_track_coordinates", "test_visibility_statistics", "test_rgb_method_specific_priors"],
        "shared_initialization_limitation": "point RGB and sparse initialization may include contributions from test views in the frozen all-image SfM",
        "subsets": records,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    write_json(output_root / "CAMERA_SUBSET_MANIFEST.json", manifest)
    return manifest


def generate_benchmark_gt(
    split_manifest: dict[str, Any],
    scene: str,
    source_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    try:
        import PIL
        from PIL import Image, features
        import torch
        import torchvision
        from torchvision.utils import save_image
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"GT generation runtime dependencies unavailable: {exc}") from exc
    if output_root.exists():
        raise FileExistsError(output_root)
    scene_row = next((row for row in split_manifest["scenes"] if row["scene"] == scene), None)
    if scene_row is None:
        raise ValueError(f"scene is absent from split manifest: {scene}")
    rows = [row for row in scene_row["assignments"] if row["split_role"] == "test"]
    output_root.mkdir(parents=True)
    records = []
    for row in rows:
        source = source_root / "images" / row["image_name"]
        if sha256_file(source) != row["image_sha256"]:
            raise ValueError(f"GT source image hash mismatch: {source}")
        with Image.open(source) as image:
            if image.mode != "RGB":
                raise ValueError(f"official loader expects decoded RGB without convert(): {source} mode={image.mode}")
            if (image.width, image.height) != (int(row["decoded_width"]), int(row["decoded_height"])):
                raise ValueError(f"decoded GT dimensions mismatch: {source}")
            loaded_size = graphdeco_quarter_dimensions(image.width, image.height)
            resized = image.resize(loaded_size)
            array = np.array(resized)
        tensor = torch.from_numpy(array) / 255.0
        tensor = tensor.permute(2, 0, 1)
        target = output_root / "gt" / f"{Path(row['image_name']).stem}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        save_image(tensor, target)
        tensor_bytes = tensor.detach().cpu().contiguous().numpy().tobytes(order="C")
        records.append({
            "scene": scene,
            "image_id": row["image_id"],
            "image_name": row["image_name"],
            "source_image_sha256": row["image_sha256"],
            "source_decoded_mode": "RGB",
            "source_decoded_width": row["decoded_width"],
            "source_decoded_height": row["decoded_height"],
            "loaded_width": loaded_size[0],
            "loaded_height": loaded_size[1],
            "tensor_dtype": str(tensor.dtype),
            "tensor_sha256": hashlib.sha256(tensor_bytes).hexdigest(),
            "gt_relative_path": target.relative_to(output_root).as_posix(),
            "gt_png_sha256": sha256_file(target),
            "gt_png_bytes": target.stat().st_size,
        })
    manifest = {
        "schema": "gs_gcp_benchmark_rgb_gt_manifest_v1",
        "scene": scene,
        "resolution_protocol": RESOLUTION_PROTOCOL,
        "holdout_semantics": HOLDOUT_SEMANTICS,
        "split_manifest_sha256": split_manifest["manifest_sha256"],
        "orientation_policy": "PIL.Image.open decoded matrix; ignore EXIF orientation; no transpose",
        "decoded_mode_policy": "must already decode as RGB; no explicit convert",
        "resize_call": "PIL.Image.Image.resize(size) with resample omitted",
        "effective_rgb_resampling": "Pillow.Resampling.BICUBIC",
        "tensor_conversion": "torch.from_numpy(np.array(resized))/255.0 then CHW permute",
        "png_write": "torchvision.utils.save_image clamp [0,1], multiply 255, add 0.5, uint8 RGB PNG",
        "python": sys.version,
        "pillow_version": PIL.__version__,
        "libjpeg_version": features.version("jpg"),
        "zlib_version": features.version("zlib"),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "image_count": len(records),
        "images": records,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    write_json(output_root / "GT_MANIFEST.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    split = sub.add_parser("generate-split")
    split.add_argument("--training_manifest", type=Path, required=True)
    split.add_argument("--camera_provenance", type=Path, required=True)
    split.add_argument("--output", type=Path, required=True)
    split.add_argument("--generator_commit")
    validate = sub.add_parser("validate-split")
    validate.add_argument("--manifest", type=Path, required=True)
    subset = sub.add_parser("materialize-subsets")
    subset.add_argument("--split_manifest", type=Path, required=True)
    subset.add_argument("--scene", required=True)
    subset.add_argument("--source_root", type=Path, required=True)
    subset.add_argument("--output_root", type=Path, required=True)
    subset.add_argument("--image_mode", choices=("symlink", "copy"), default="symlink")
    gt = sub.add_parser("generate-gt")
    gt.add_argument("--split_manifest", type=Path, required=True)
    gt.add_argument("--scene", required=True)
    gt.add_argument("--source_root", type=Path, required=True)
    gt.add_argument("--output_root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "generate-split":
        payload = generate_split_manifest(
            args.training_manifest.resolve(), args.camera_provenance.resolve(), args.generator_commit
        )
        if args.output.exists():
            raise FileExistsError(args.output)
        write_json(args.output, payload)
    elif args.command == "validate-split":
        payload = validate_split_manifest(json.loads(args.manifest.read_text(encoding="utf-8")))
        if not payload["passed"]:
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 1
    elif args.command == "materialize-subsets":
        payload = materialize_camera_subsets(
            json.loads(args.split_manifest.read_text(encoding="utf-8")), args.scene,
            args.source_root.resolve(), args.output_root.resolve(), args.image_mode,
        )
    else:
        payload = generate_benchmark_gt(
            json.loads(args.split_manifest.read_text(encoding="utf-8")), args.scene,
            args.source_root.resolve(), args.output_root.resolve(),
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
