#!/usr/bin/env python3
"""Build COLMAP image pairs from estimated ground-footprint overlap."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ImagePose:
    name: str
    sequence: int
    east_m: float
    north_m: float
    altitude_m: float
    yaw_deg: float
    pitch_deg: float
    polygon: tuple[tuple[float, float], ...]
    area_m2: float
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--output-pairs", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--horizontal-fov-deg", type=float, required=True)
    parser.add_argument("--vertical-fov-deg", type=float, required=True)
    parser.add_argument("--max-footprint-neighbors", type=int, default=80)
    parser.add_argument("--sequential-overlap", type=int, default=20)
    parser.add_argument("--min-overlap-over-smaller", type=float, default=0.08)
    parser.add_argument("--max-center-distance-m", type=float, default=250.0)
    return parser.parse_args()


def get_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "").strip()
    if not value:
        raise ValueError(f"Missing {key} for {row.get('FileName', '<unknown>')}")
    return float(value)


def sequence_from_name(name: str) -> int:
    match = re.search(r"_(\d{4})_D\.JPG$", name, re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot parse image sequence from {name}")
    return int(match.group(1))


def cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def add_scaled(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    b_scale: float,
    c: tuple[float, float, float],
    c_scale: float,
) -> tuple[float, float, float]:
    return (
        a[0] + b_scale * b[0] + c_scale * c[0],
        a[1] + b_scale * b[1] + c_scale * c[1],
        a[2] + b_scale * b[2] + c_scale * c[2],
    )


def signed_area(poly: Iterable[tuple[float, float]]) -> float:
    points = list(poly)
    return 0.5 * sum(
        points[i][0] * points[(i + 1) % len(points)][1]
        - points[(i + 1) % len(points)][0] * points[i][1]
        for i in range(len(points))
    )


def polygon_area(poly: Iterable[tuple[float, float]]) -> float:
    return abs(signed_area(poly))


def footprint_polygon(
    east_m: float,
    north_m: float,
    altitude_m: float,
    yaw_deg: float,
    pitch_deg: float,
    horizontal_fov_deg: float,
    vertical_fov_deg: float,
) -> tuple[tuple[float, float], ...]:
    yaw = math.radians(yaw_deg)
    depression = math.radians(max(1.0, min(90.0, -pitch_deg)))

    optical = (
        math.sin(yaw) * math.cos(depression),
        math.cos(yaw) * math.cos(depression),
        -math.sin(depression),
    )
    right = (math.cos(yaw), -math.sin(yaw), 0.0)
    up = cross(right, optical)
    tx = math.tan(math.radians(horizontal_fov_deg) / 2.0)
    ty = math.tan(math.radians(vertical_fov_deg) / 2.0)

    rays = [
        add_scaled(optical, right, -tx, up, +ty),
        add_scaled(optical, right, -tx, up, -ty),
        add_scaled(optical, right, +tx, up, -ty),
        add_scaled(optical, right, +tx, up, +ty),
    ]
    polygon: list[tuple[float, float]] = []
    for ray in rays:
        if ray[2] >= -1e-6:
            raise ValueError(
                f"FOV reaches above the ground horizon for pitch={pitch_deg:.2f}"
            )
        distance = altitude_m / -ray[2]
        polygon.append((east_m + distance * ray[0], north_m + distance * ray[1]))

    if signed_area(polygon) < 0:
        polygon.reverse()
    return tuple(polygon)


def inside(
    point: tuple[float, float],
    edge_a: tuple[float, float],
    edge_b: tuple[float, float],
) -> bool:
    return (
        (edge_b[0] - edge_a[0]) * (point[1] - edge_a[1])
        - (edge_b[1] - edge_a[1]) * (point[0] - edge_a[0])
    ) >= -1e-9


def line_intersection(
    start: tuple[float, float],
    end: tuple[float, float],
    edge_a: tuple[float, float],
    edge_b: tuple[float, float],
) -> tuple[float, float]:
    dx1, dy1 = end[0] - start[0], end[1] - start[1]
    dx2, dy2 = edge_b[0] - edge_a[0], edge_b[1] - edge_a[1]
    denominator = dx1 * dy2 - dy1 * dx2
    if abs(denominator) < 1e-12:
        return end
    t = ((edge_a[0] - start[0]) * dy2 - (edge_a[1] - start[1]) * dx2) / denominator
    return (start[0] + t * dx1, start[1] + t * dy1)


def convex_intersection(
    subject: tuple[tuple[float, float], ...],
    clip: tuple[tuple[float, float], ...],
) -> tuple[tuple[float, float], ...]:
    output = list(subject)
    for index, edge_a in enumerate(clip):
        edge_b = clip[(index + 1) % len(clip)]
        input_points = output
        output = []
        if not input_points:
            break
        start = input_points[-1]
        for end in input_points:
            end_inside = inside(end, edge_a, edge_b)
            start_inside = inside(start, edge_a, edge_b)
            if end_inside:
                if not start_inside:
                    output.append(line_intersection(start, end, edge_a, edge_b))
                output.append(end)
            elif start_inside:
                output.append(line_intersection(start, end, edge_a, edge_b))
            start = end
    return tuple(output)


def bboxes_overlap(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def load_poses(
    metadata_csv: Path,
    horizontal_fov_deg: float,
    vertical_fov_deg: float,
) -> list[ImagePose]:
    with metadata_csv.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No metadata rows in {metadata_csv}")

    latitudes = [get_float(row, "GPSLatitude") for row in rows]
    longitudes = [get_float(row, "GPSLongitude") for row in rows]
    lat0 = sum(latitudes) / len(latitudes)
    lon0 = sum(longitudes) / len(longitudes)

    poses: list[ImagePose] = []
    for row in rows:
        name = row["FileName"].strip()
        latitude = get_float(row, "GPSLatitude")
        longitude = get_float(row, "GPSLongitude")
        east_m = (longitude - lon0) * 111_320.0 * math.cos(math.radians(lat0))
        north_m = (latitude - lat0) * 111_320.0
        altitude_m = get_float(row, "RelativeAltitude")
        yaw_deg = get_float(row, "GimbalYawDegree")
        pitch_deg = get_float(row, "GimbalPitchDegree")
        polygon = footprint_polygon(
            east_m,
            north_m,
            altitude_m,
            yaw_deg,
            pitch_deg,
            horizontal_fov_deg,
            vertical_fov_deg,
        )
        area = polygon_area(polygon)
        bbox = (
            min(point[0] for point in polygon),
            min(point[1] for point in polygon),
            max(point[0] for point in polygon),
            max(point[1] for point in polygon),
        )
        center = (
            sum(point[0] for point in polygon) / len(polygon),
            sum(point[1] for point in polygon) / len(polygon),
        )
        poses.append(
            ImagePose(
                name=name,
                sequence=sequence_from_name(name),
                east_m=east_m,
                north_m=north_m,
                altitude_m=altitude_m,
                yaw_deg=yaw_deg,
                pitch_deg=pitch_deg,
                polygon=polygon,
                area_m2=area,
                bbox=bbox,
                center=center,
            )
        )
    return sorted(poses, key=lambda pose: pose.sequence)


def make_pair(a: ImagePose, b: ImagePose) -> tuple[str, str]:
    return tuple(sorted((a.name, b.name)))


def main() -> None:
    args = parse_args()
    poses = load_poses(
        args.metadata_csv,
        args.horizontal_fov_deg,
        args.vertical_fov_deg,
    )
    pair_reasons: dict[tuple[str, str], set[str]] = {}
    scores_by_image: dict[str, list[tuple[float, float, ImagePose]]] = {
        pose.name: [] for pose in poses
    }

    for left_index, left in enumerate(poses):
        for right in poses[left_index + 1 :]:
            center_distance = math.dist(left.center, right.center)
            if center_distance > args.max_center_distance_m:
                continue
            if not bboxes_overlap(left.bbox, right.bbox):
                continue
            intersection = convex_intersection(left.polygon, right.polygon)
            intersection_area = polygon_area(intersection) if len(intersection) >= 3 else 0.0
            overlap = intersection_area / min(left.area_m2, right.area_m2)
            if overlap < args.min_overlap_over_smaller:
                continue
            scores_by_image[left.name].append((overlap, -center_distance, right))
            scores_by_image[right.name].append((overlap, -center_distance, left))

    name_to_pose = {pose.name: pose for pose in poses}
    for pose in poses:
        ranked = sorted(scores_by_image[pose.name], reverse=True)
        for overlap, _, neighbor in ranked[: args.max_footprint_neighbors]:
            pair_reasons.setdefault(make_pair(pose, neighbor), set()).add(
                f"footprint:{overlap:.4f}"
            )

    for index, pose in enumerate(poses):
        start = max(0, index - args.sequential_overlap)
        stop = min(len(poses), index + args.sequential_overlap + 1)
        for neighbor in poses[start:stop]:
            if pose.name == neighbor.name:
                continue
            pair_reasons.setdefault(make_pair(pose, neighbor), set()).add("sequential")

    args.output_pairs.parent.mkdir(parents=True, exist_ok=True)
    with args.output_pairs.open("w", encoding="utf-8", newline="\n") as handle:
        for left, right in sorted(pair_reasons):
            handle.write(f"{left} {right}\n")

    footprint_counts = []
    sequential_only = 0
    for pair, reasons in pair_reasons.items():
        if any(reason.startswith("footprint:") for reason in reasons):
            for name in pair:
                footprint_counts.append(name)
        elif reasons == {"sequential"}:
            sequential_only += 1

    per_image = {name: 0 for name in name_to_pose}
    for pair in pair_reasons:
        for name in pair:
            per_image[name] += 1
    counts = sorted(per_image.values())
    summary = {
        "schema": "m3m_gcp_fov_aware_pair_summary_v1",
        "metadata_csv": str(args.metadata_csv),
        "image_count": len(poses),
        "pair_count": len(pair_reasons),
        "sequential_only_pair_count": sequential_only,
        "parameters": {
            "horizontal_fov_deg": args.horizontal_fov_deg,
            "vertical_fov_deg": args.vertical_fov_deg,
            "max_footprint_neighbors": args.max_footprint_neighbors,
            "sequential_overlap": args.sequential_overlap,
            "min_overlap_over_smaller": args.min_overlap_over_smaller,
            "max_center_distance_m": args.max_center_distance_m,
        },
        "pair_degree": {
            "min": counts[0],
            "median": counts[len(counts) // 2],
            "mean": sum(counts) / len(counts),
            "max": counts[-1],
        },
        "altitude_m": {
            "min": min(pose.altitude_m for pose in poses),
            "max": max(pose.altitude_m for pose in poses),
        },
        "pitch_deg": {
            "min": min(pose.pitch_deg for pose in poses),
            "max": max(pose.pitch_deg for pose in poses),
        },
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
