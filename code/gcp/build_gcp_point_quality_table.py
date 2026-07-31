#!/usr/bin/env python3
"""Build a non-release RTK quality enrichment of the frozen v1.3.0 point table."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from gcp_pixel_domain_v1_2 import (
    canonical_record_sha256,
    file_sha256,
    read_csv,
    write_csv_deterministic,
    write_json_deterministic,
)


EPOCH_ENCODING = "gb18030"
EXPECTED_EPOCH_POINT_COUNT = 53
EXPECTED_EPOCHS_PER_POINT = 27
EXPECTED_FROZEN_POINT_COUNT = 53
EXPECTED_FORMAL_UNIQUE_POINT_COUNT = 50
KNOWN_BASE_POINTS_WITHOUT_EPOCH_RECORDS = {"K002", "NC08", "NC94", "NC96"}
GROSS_MIXED_TRAJECTORY_HORIZONTAL_M = 1.0

QUALITY_FIELDS = [
    "rtk_quality_source_status",
    "rtk_epoch_observation_count",
    "rtk_fixed_solution_count",
    "rtk_nonfixed_solution_count",
    "rtk_fixed_solution_rate",
    "rtk_observation_duration_s",
    "rtk_east_std_m",
    "rtk_north_std_m",
    "rtk_height_std_m",
    "rtk_horizontal_std_vector_m",
    "rtk_horizontal_pairwise_range_m",
    "rtk_height_range_m",
    "rtk_receiver_hrms_median_m",
    "rtk_receiver_hrms_max_m",
    "rtk_receiver_vrms_median_m",
    "rtk_receiver_vrms_max_m",
    "rtk_pdop_min",
    "rtk_pdop_median",
    "rtk_pdop_max",
    "rtk_satellite_min",
    "rtk_satellite_median",
    "rtk_satellite_max",
    "rtk_final_coordinate_minus_epoch_mean_3d_m",
    "rtk_gross_mixed_trajectory_flag",
    "rtk_absolute_accuracy_status",
    "rtk_quality_interpretation",
    "rtk_epoch_source_sha256",
    "rtk_final_coordinate_source_sha256",
    "rtk_quality_record_sha256_v2",
]


def point_name(value: str) -> str:
    return re.sub(r"_\d+$", "", value.strip())


def observation_index(value: str) -> int:
    match = re.search(r"_(\d+)$", value.strip())
    if not match:
        raise ValueError(f"RTK observation name lacks a numeric suffix: {value!r}")
    return int(match.group(1))


def parse_seconds(date_value: str, time_value: str) -> datetime:
    year, month, day = [int(part) for part in re.split(r"[-/]", date_value.strip())]
    hour, minute, second = [int(part) for part in time_value.strip().split(":")]
    return datetime(year, month, day, hour, minute, second)


def read_corrected_epochs(path: Path) -> dict[str, list[dict[str, Any]]]:
    with path.open("r", encoding=EPOCH_ENCODING, newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "点名",
        "东坐标",
        "北坐标",
        "高程",
        "日期",
        "时间",
        "VRMS",
        "HRMS",
        "PDOP",
        "解状态",
        "解算卫星",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"corrected epoch CSV lacks required fields: {sorted(required - set(rows[0] if rows else {}))}")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for row in rows:
        observation_name = row["点名"].strip()
        if observation_name in seen:
            raise ValueError(f"duplicate RTK observation identifier: {observation_name}")
        seen.add(observation_name)
        status = row["解状态"].strip()
        grouped[point_name(observation_name)].append(
            {
                "observation_name": observation_name,
                "observation_index": observation_index(observation_name),
                "east_m": float(row["东坐标"]),
                "north_m": float(row["北坐标"]),
                "height_m": float(row["高程"]),
                "timestamp": parse_seconds(row["日期"], row["时间"]),
                "vrms_m": float(row["VRMS"]),
                "hrms_m": float(row["HRMS"]),
                "pdop": float(row["PDOP"]),
                "solution_status": status,
                "satellite_count": int(float(row["解算卫星"])),
            }
        )
    if len(grouped) != EXPECTED_EPOCH_POINT_COUNT:
        raise ValueError(f"expected {EXPECTED_EPOCH_POINT_COUNT} corrected epoch points, got {len(grouped)}")
    for name, group in grouped.items():
        if len(group) != EXPECTED_EPOCHS_PER_POINT:
            raise ValueError(f"{name}: expected {EXPECTED_EPOCHS_PER_POINT} epochs, got {len(group)}")
        indices = sorted(item["observation_index"] for item in group)
        if indices != list(range(1, EXPECTED_EPOCHS_PER_POINT + 1)):
            raise ValueError(f"{name}: epoch indices are not exactly 1..{EXPECTED_EPOCHS_PER_POINT}")
    return grouped


def read_final_coordinates(path: Path) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    with path.open("r", encoding=EPOCH_ENCODING, newline="") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            name = row[0].strip()
            if name in result:
                raise ValueError(f"duplicate final coordinate point: {name}")
            result[name] = np.asarray([float(row[2]), float(row[3]), float(row[4])], dtype=np.float64)
    return result


def sample_std(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1))


def max_pairwise(points: np.ndarray) -> float:
    differences = points[:, None, :] - points[None, :, :]
    return float(np.max(np.linalg.norm(differences, axis=2)))


def summarize_quality(
    name: str,
    group: list[dict[str, Any]],
    final_coordinate: np.ndarray,
) -> dict[str, Any]:
    ordered = sorted(group, key=lambda item: item["observation_index"])
    xyz = np.asarray(
        [[item["east_m"], item["north_m"], item["height_m"]] for item in ordered],
        dtype=np.float64,
    )
    hrms = np.asarray([item["hrms_m"] for item in ordered], dtype=np.float64)
    vrms = np.asarray([item["vrms_m"] for item in ordered], dtype=np.float64)
    pdop = np.asarray([item["pdop"] for item in ordered], dtype=np.float64)
    satellites = np.asarray([item["satellite_count"] for item in ordered], dtype=np.float64)
    fixed_count = sum(item["solution_status"] == "固定解" for item in ordered)
    mean = xyz.mean(axis=0)
    elapsed = max(item["timestamp"] for item in ordered) - min(item["timestamp"] for item in ordered)
    duration_seconds = (
        float(elapsed.total_seconds())
        if hasattr(elapsed, "total_seconds")
        else float(elapsed / np.timedelta64(1, "s"))
    )
    horizontal_std = float(math.hypot(sample_std(xyz[:, 0]), sample_std(xyz[:, 1])))
    horizontal_range = max_pairwise(xyz[:, :2])
    quality = {
        "point_name": name,
        "observation_count": len(ordered),
        "fixed_solution_count": fixed_count,
        "nonfixed_solution_count": len(ordered) - fixed_count,
        "fixed_solution_rate": fixed_count / len(ordered),
        "duration_s": duration_seconds,
        "east_std_m": sample_std(xyz[:, 0]),
        "north_std_m": sample_std(xyz[:, 1]),
        "height_std_m": sample_std(xyz[:, 2]),
        "horizontal_std_vector_m": horizontal_std,
        "horizontal_pairwise_range_m": horizontal_range,
        "height_range_m": float(np.ptp(xyz[:, 2])),
        "hrms_median_m": float(np.median(hrms)),
        "hrms_max_m": float(np.max(hrms)),
        "vrms_median_m": float(np.median(vrms)),
        "vrms_max_m": float(np.max(vrms)),
        "pdop_min": float(np.min(pdop)),
        "pdop_median": float(np.median(pdop)),
        "pdop_max": float(np.max(pdop)),
        "satellite_min": int(np.min(satellites)),
        "satellite_median": float(np.median(satellites)),
        "satellite_max": int(np.max(satellites)),
        "final_coordinate_minus_epoch_mean_3d_m": float(np.linalg.norm(final_coordinate - mean)),
        "gross_mixed_trajectory_flag": horizontal_range > GROSS_MIXED_TRAJECTORY_HORIZONTAL_M,
    }
    return quality


def compact_quality_fields(
    quality: dict[str, Any],
    *,
    epoch_sha256: str,
    coordinate_sha256: str,
) -> dict[str, Any]:
    record = {
        "point_name": quality["point_name"],
        "observation_count": quality["observation_count"],
        "fixed_solution_count": quality["fixed_solution_count"],
        "nonfixed_solution_count": quality["nonfixed_solution_count"],
        "fixed_solution_rate": quality["fixed_solution_rate"],
        "duration_s": quality["duration_s"],
        "east_std_m": quality["east_std_m"],
        "north_std_m": quality["north_std_m"],
        "height_std_m": quality["height_std_m"],
        "horizontal_std_vector_m": quality["horizontal_std_vector_m"],
        "horizontal_pairwise_range_m": quality["horizontal_pairwise_range_m"],
        "height_range_m": quality["height_range_m"],
        "receiver_hrms_median_m": quality["hrms_median_m"],
        "receiver_hrms_max_m": quality["hrms_max_m"],
        "receiver_vrms_median_m": quality["vrms_median_m"],
        "receiver_vrms_max_m": quality["vrms_max_m"],
        "pdop_min": quality["pdop_min"],
        "pdop_median": quality["pdop_median"],
        "pdop_max": quality["pdop_max"],
        "satellite_min": quality["satellite_min"],
        "satellite_median": quality["satellite_median"],
        "satellite_max": quality["satellite_max"],
        "final_coordinate_minus_epoch_mean_3d_m": quality["final_coordinate_minus_epoch_mean_3d_m"],
        "gross_mixed_trajectory_flag": quality["gross_mixed_trajectory_flag"],
        "epoch_source_sha256": epoch_sha256,
        "final_coordinate_source_sha256": coordinate_sha256,
    }
    record["quality_record_sha256_v2"] = canonical_record_sha256(record)
    return record


def quality_join_fields(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "rtk_quality_source_status": "direct_corrected_epoch_quality_available",
        "rtk_epoch_observation_count": record["observation_count"],
        "rtk_fixed_solution_count": record["fixed_solution_count"],
        "rtk_nonfixed_solution_count": record["nonfixed_solution_count"],
        "rtk_fixed_solution_rate": record["fixed_solution_rate"],
        "rtk_observation_duration_s": record["duration_s"],
        "rtk_east_std_m": record["east_std_m"],
        "rtk_north_std_m": record["north_std_m"],
        "rtk_height_std_m": record["height_std_m"],
        "rtk_horizontal_std_vector_m": record["horizontal_std_vector_m"],
        "rtk_horizontal_pairwise_range_m": record["horizontal_pairwise_range_m"],
        "rtk_height_range_m": record["height_range_m"],
        "rtk_receiver_hrms_median_m": record["receiver_hrms_median_m"],
        "rtk_receiver_hrms_max_m": record["receiver_hrms_max_m"],
        "rtk_receiver_vrms_median_m": record["receiver_vrms_median_m"],
        "rtk_receiver_vrms_max_m": record["receiver_vrms_max_m"],
        "rtk_pdop_min": record["pdop_min"],
        "rtk_pdop_median": record["pdop_median"],
        "rtk_pdop_max": record["pdop_max"],
        "rtk_satellite_min": record["satellite_min"],
        "rtk_satellite_median": record["satellite_median"],
        "rtk_satellite_max": record["satellite_max"],
        "rtk_final_coordinate_minus_epoch_mean_3d_m": record["final_coordinate_minus_epoch_mean_3d_m"],
        "rtk_gross_mixed_trajectory_flag": str(record["gross_mixed_trajectory_flag"]).lower(),
        "rtk_absolute_accuracy_status": "not_independently_verified",
        "rtk_quality_interpretation": (
            "receiver_reported_precision_and_internal_repeatability_not_independent_absolute_accuracy"
        ),
        "rtk_epoch_source_sha256": record["epoch_source_sha256"],
        "rtk_final_coordinate_source_sha256": record["final_coordinate_source_sha256"],
        "rtk_quality_record_sha256_v2": record["quality_record_sha256_v2"],
    }


def missing_base_join_fields() -> dict[str, Any]:
    result = {field: "" for field in QUALITY_FIELDS}
    result.update(
        {
            "rtk_quality_source_status": "known_base_point_no_epoch_quality_record",
            "rtk_absolute_accuracy_status": "not_available_in_frozen_authoritative_artifacts",
            "rtk_quality_interpretation": (
                "known_base_coordinate_only_no_direct_epoch_hrms_vrms_or_repeatability_record"
            ),
        }
    )
    return result


def field_definitions() -> list[dict[str, str]]:
    return [
        {
            "field": "rtk_quality_source_status",
            "unit": "status",
            "source": "point-name join audit",
            "interpretation": (
                "direct corrected epoch evidence is available, or the point is a known base point "
                "without a directly attributable epoch-quality record"
            ),
        },
        {
            "field": "rtk_receiver_hrms_median_m / rtk_receiver_hrms_max_m",
            "unit": "m",
            "source": "corrected RTK epoch CSV HRMS",
            "interpretation": "receiver-reported horizontal solution precision; not independent absolute accuracy",
        },
        {
            "field": "rtk_receiver_vrms_median_m / rtk_receiver_vrms_max_m",
            "unit": "m",
            "source": "corrected RTK epoch CSV VRMS",
            "interpretation": "receiver-reported vertical solution precision; not independent absolute accuracy",
        },
        {
            "field": "rtk_horizontal_std_vector_m",
            "unit": "m",
            "source": "sample standard deviation of 27 corrected epoch coordinates",
            "interpretation": "internal horizontal repeatability: hypot(std(E), std(N))",
        },
        {
            "field": "rtk_height_std_m",
            "unit": "m",
            "source": "sample standard deviation of 27 corrected epoch heights",
            "interpretation": "internal vertical repeatability",
        },
        {
            "field": "rtk_horizontal_pairwise_range_m / rtk_height_range_m",
            "unit": "m",
            "source": "27 corrected epoch coordinates",
            "interpretation": "maximum horizontal pairwise distance / vertical peak-to-peak range",
        },
        {
            "field": "rtk_fixed_solution_rate",
            "unit": "ratio",
            "source": "corrected RTK epoch solution status",
            "interpretation": "fixed-solution epochs divided by all epochs",
        },
        {
            "field": "rtk_pdop_min / rtk_pdop_median / rtk_pdop_max",
            "unit": "dimensionless",
            "source": "corrected RTK epoch CSV PDOP",
            "interpretation": "satellite-geometry diagnostic; lower is generally better",
        },
        {
            "field": "rtk_satellite_min / rtk_satellite_median / rtk_satellite_max",
            "unit": "count",
            "source": "corrected RTK epoch CSV solved-satellite count",
            "interpretation": "satellites used by the receiver solution",
        },
        {
            "field": "rtk_final_coordinate_minus_epoch_mean_3d_m",
            "unit": "m",
            "source": "corrected final DAT and corrected epoch CSV",
            "interpretation": "3D difference between frozen final coordinate and mean of corrected epochs",
        },
        {
            "field": "rtk_absolute_accuracy_status",
            "unit": "status",
            "source": "provenance audit",
            "interpretation": "no independent higher-order truth is available for absolute-accuracy certification",
        },
    ]


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_root.exists():
        raise FileExistsError(f"output root already exists: {args.output_root}")
    point_rows = read_csv(args.frozen_point_table)
    split_rows = read_csv(args.frozen_split)
    if len(point_rows) != EXPECTED_FROZEN_POINT_COUNT:
        raise ValueError(f"expected {EXPECTED_FROZEN_POINT_COUNT} frozen point rows, got {len(point_rows)}")
    if len({row["point_name"] for row in point_rows}) != len(point_rows):
        raise ValueError("frozen point table contains duplicate point names")
    formal_points = {row["point_name"] for row in split_rows}
    if len(formal_points) != EXPECTED_FORMAL_UNIQUE_POINT_COUNT:
        raise ValueError(
            f"expected {EXPECTED_FORMAL_UNIQUE_POINT_COUNT} formal unique points, got {len(formal_points)}"
        )

    grouped = read_corrected_epochs(args.corrected_epoch_csv)
    final_coordinates = read_final_coordinates(args.corrected_coordinate_dat)
    missing_coordinates = sorted(set(grouped) - set(final_coordinates))
    if missing_coordinates:
        raise ValueError(f"corrected DAT lacks epoch-point coordinates: {missing_coordinates}")

    epoch_sha = file_sha256(args.corrected_epoch_csv)
    coordinate_sha = file_sha256(args.corrected_coordinate_dat)
    quality_by_name: dict[str, dict[str, Any]] = {}
    for name, group in sorted(grouped.items()):
        quality = summarize_quality(name, group, final_coordinates[name])
        quality_by_name[name] = compact_quality_fields(
            quality,
            epoch_sha256=epoch_sha,
            coordinate_sha256=coordinate_sha,
        )

    frozen_names = {row["point_name"] for row in point_rows}
    missing_quality = frozen_names - set(quality_by_name)
    if missing_quality != KNOWN_BASE_POINTS_WITHOUT_EPOCH_RECORDS:
        raise ValueError(f"unexpected frozen points without epoch quality: {sorted(missing_quality)}")
    formal_missing = formal_points - set(quality_by_name)
    if formal_missing != {"NC94"}:
        raise ValueError(f"unexpected formal points without epoch quality: {sorted(formal_missing)}")
    if "G47" not in quality_by_name or not quality_by_name["G47"]["gross_mixed_trajectory_flag"]:
        raise ValueError("G47 gross mixed trajectory was not recovered")
    if "G47" in frozen_names:
        raise ValueError("G47 must not be present in the frozen point table")

    args.output_root.mkdir(parents=True)
    enriched_rows = []
    original_fields = list(point_rows[0])
    for row in point_rows:
        item = dict(row)
        name = row["point_name"]
        if name in quality_by_name:
            item.update(quality_join_fields(quality_by_name[name]))
        else:
            item.update(missing_base_join_fields())
        enriched_rows.append(item)
    enriched_fields = original_fields + [field for field in QUALITY_FIELDS if field not in original_fields]

    enriched_path = args.output_root / "gcp_points_v1_3_0_rtk_quality_enriched_audit.csv"
    quality_path = args.output_root / "rtk_corrected_epoch_quality_all_points.csv"
    definitions_path = args.output_root / "rtk_quality_field_definitions.csv"
    readme_path = args.output_root / "README_RTK_QUALITY_ENRICHMENT.md"
    write_csv_deterministic(enriched_path, enriched_rows, enriched_fields)
    quality_fields = list(next(iter(quality_by_name.values())))
    write_csv_deterministic(
        quality_path,
        [quality_by_name[name] for name in sorted(quality_by_name)],
        quality_fields,
    )
    write_csv_deterministic(
        definitions_path,
        field_definitions(),
        ["field", "unit", "source", "interpretation"],
    )
    readme_path.write_text(
        """# GS-GCP v1.3.0 RTK quality enrichment audit

This directory is a non-release audit. It does not modify or supersede the
frozen v1.3.0 payload or root digest.

`gcp_points_v1_3_0_rtk_quality_enriched_audit.csv` preserves every existing
point-table field and appends measurement-quality fields derived from the
corrected authoritative RTK epoch file.

Interpretation:

- HRMS/VRMS are receiver-reported precision estimates.
- Coordinate standard deviations and ranges are internal repeatability.
- Neither quantity is an independently verified absolute accuracy.
- `NC94` is the only formal point without a directly attributable epoch-quality
  record. Its quality fields remain blank; nearby check-point observations are
  not substituted.
- The legacy frozen `mean_pdop` column is unchanged. Use the explicitly defined
  `rtk_pdop_median` field for this audit.

Promoting these fields into a formal benchmark payload requires a new
non-overwriting metadata release and external review.
""",
        encoding="utf-8",
        newline="\n",
    )

    for original, enriched in zip(point_rows, enriched_rows):
        for field in original_fields:
            if str(original.get(field, "")) != str(enriched.get(field, "")):
                raise ValueError(
                    f"frozen field changed during enrichment: {original['point_name']}:{field}"
                )

    direct_formal = formal_points & set(quality_by_name)
    valid_quality = [
        row
        for row in quality_by_name.values()
        if not row["gross_mixed_trajectory_flag"]
    ]
    summary = {
        "schema": "gs_gcp_v1_3_0_rtk_quality_enrichment_audit_v1",
        "status": "PASS_WITH_KNOWN_BASE_POINT_ACCURACY_GAP",
        "frozen_release_mutated": False,
        "artifact_role": "non_release_audit_sidecar",
        "original_field_value_mismatch_count": 0,
        "frozen_point_row_count": len(point_rows),
        "formal_unique_point_count": len(formal_points),
        "formal_points_with_direct_epoch_quality_count": len(direct_formal),
        "formal_points_without_direct_epoch_quality": sorted(formal_points - set(quality_by_name)),
        "frozen_points_with_direct_epoch_quality_count": len(frozen_names & set(quality_by_name)),
        "frozen_known_base_points_without_epoch_quality": sorted(missing_quality),
        "corrected_epoch_point_count": len(quality_by_name),
        "corrected_epoch_row_count": sum(len(group) for group in grouped.values()),
        "g47_gross_mixed_trajectory_recovered_and_excluded": True,
        "all_direct_quality_points_fixed_solution_rate_one": all(
            row["fixed_solution_rate"] == 1.0 for row in quality_by_name.values()
        ),
        "non_gross_quality_ranges": {
            "receiver_hrms_median_m": [
                min(row["receiver_hrms_median_m"] for row in valid_quality),
                max(row["receiver_hrms_median_m"] for row in valid_quality),
            ],
            "receiver_vrms_median_m": [
                min(row["receiver_vrms_median_m"] for row in valid_quality),
                max(row["receiver_vrms_median_m"] for row in valid_quality),
            ],
            "horizontal_std_vector_m": [
                min(row["horizontal_std_vector_m"] for row in valid_quality),
                max(row["horizontal_std_vector_m"] for row in valid_quality),
            ],
            "height_std_m": [
                min(row["height_std_m"] for row in valid_quality),
                max(row["height_std_m"] for row in valid_quality),
            ],
        },
        "legacy_field_note": (
            "The frozen mean_pdop column is preserved byte-for-lineage. "
            "Use rtk_pdop_median from this audit for an explicitly defined statistic."
        ),
        "accuracy_interpretation": (
            "HRMS/VRMS are receiver-reported precision and coordinate dispersion is internal repeatability; "
            "neither is independent absolute accuracy."
        ),
        "input_files": {
            "frozen_point_table": {
                "path": str(args.frozen_point_table),
                "sha256": file_sha256(args.frozen_point_table),
            },
            "frozen_split": {
                "path": str(args.frozen_split),
                "sha256": file_sha256(args.frozen_split),
            },
            "corrected_epoch_csv": {
                "path": str(args.corrected_epoch_csv),
                "sha256": epoch_sha,
            },
            "corrected_coordinate_dat": {
                "path": str(args.corrected_coordinate_dat),
                "sha256": coordinate_sha,
            },
        },
        "output_files": {},
    }
    for path in [enriched_path, quality_path, definitions_path, readme_path]:
        summary["output_files"][path.name] = {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    summary_path = args.output_root / "validation_summary.json"
    write_json_deterministic(summary_path, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_point_table", type=Path, required=True)
    parser.add_argument("--frozen_split", type=Path, required=True)
    parser.add_argument("--corrected_epoch_csv", type=Path, required=True)
    parser.add_argument("--corrected_coordinate_dat", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    args = parser.parse_args()
    summary = build(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
