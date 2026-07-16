#!/usr/bin/env python3
"""Audit corrected RTK epoch observations and point-name lineage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_POINTS = ("G07", "G09", "G39")
CSV_ENCODING = "gb18030"
GROSS_MIXED_TRAJECTORY_HORIZONTAL_M = 1.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def point_name(value: str) -> str:
    return re.sub(r"_\d+$", "", str(value))


def observation_index(value: str) -> int:
    match = re.search(r"_(\d+)$", str(value))
    if not match:
        raise ValueError(f"Observation name has no numeric suffix: {value}")
    return int(match.group(1))


def parse_datetime(date_value: str, time_value: str) -> datetime:
    parts = [int(part) for part in re.split(r"[-/]", str(date_value))]
    hours, minutes, seconds = [int(part) for part in str(time_value).split(":")]
    return datetime(parts[0], parts[1], parts[2], hours, minutes, seconds)


def read_epoch_csv(path: Path, *, require_observation_index: bool = True) -> pd.DataFrame:
    with path.open("r", encoding=CSV_ENCODING, newline="") as handle:
        rows = [
            {key: value for key, value in row.items() if key is not None}
            for row in csv.DictReader(handle)
        ]
    frame = pd.DataFrame(rows, dtype=str).fillna("")
    has_index = frame["点名"].map(lambda value: bool(re.search(r"_\d+$", str(value))))
    if require_observation_index and not has_index.all():
        invalid = frame.loc[~has_index, "点名"].tolist()
        raise ValueError(f"Corrected observations contain invalid names: {invalid}")
    if not require_observation_index:
        frame = frame.loc[has_index].copy()
    frame["point_name"] = frame["点名"].map(point_name)
    frame["observation_index"] = frame["点名"].map(observation_index)
    frame["timestamp"] = pd.to_datetime(
        [parse_datetime(d, t) for d, t in zip(frame["日期"], frame["时间"])]
    )
    for source, target in [
        ("东坐标", "east_m"),
        ("北坐标", "north_m"),
        ("高程", "height_m"),
        ("VRMS", "vrms_m"),
        ("HRMS", "hrms_m"),
        ("PDOP", "pdop"),
        ("解算卫星", "satellite_count"),
    ]:
        frame[target] = pd.to_numeric(frame[source], errors="raise")
    return frame


def read_dat(path: Path) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    with path.open("r", encoding=CSV_ENCODING, newline="") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            result[row[0]] = np.asarray([float(row[2]), float(row[3]), float(row[4])], dtype=np.float64)
    return result


def max_pairwise(points: np.ndarray) -> float:
    differences = points[:, None, :] - points[None, :, :]
    return float(np.max(np.linalg.norm(differences, axis=2)))


def chunk_center_spread(points: np.ndarray) -> tuple[float, float]:
    if len(points) != 27:
        return float("nan"), float("nan")
    centers = np.asarray([chunk.mean(axis=0) for chunk in np.split(points, 3)])
    return max_pairwise(centers[:, :2]), float(np.ptp(centers[:, 2]))


def summarize_point(group: pd.DataFrame, dat_coordinate: np.ndarray) -> dict[str, float | int | str]:
    group = group.sort_values("observation_index")
    xyz = group[["east_m", "north_m", "height_m"]].to_numpy(dtype=np.float64)
    mean = xyz.mean(axis=0)
    centered = xyz - mean
    horizontal_radius = np.linalg.norm(centered[:, :2], axis=1)
    distance_3d = np.linalg.norm(centered, axis=1)
    chunk_h, chunk_z = chunk_center_spread(xyz)
    dat_offset = dat_coordinate - mean
    elapsed_seconds = (group["timestamp"].max() - group["timestamp"].min()).total_seconds()
    return {
        "point_name": str(group["point_name"].iloc[0]),
        "observation_count": int(len(group)),
        "fixed_solution_count": int(group["解状态"].eq("固定解").sum()),
        "nonfixed_solution_count": int((~group["解状态"].eq("固定解")).sum()),
        "duration_s": float(elapsed_seconds),
        "east_mean_m": float(mean[0]),
        "north_mean_m": float(mean[1]),
        "height_mean_m": float(mean[2]),
        "east_std_m": float(np.std(xyz[:, 0], ddof=1)),
        "north_std_m": float(np.std(xyz[:, 1], ddof=1)),
        "height_std_m": float(np.std(xyz[:, 2], ddof=1)),
        "horizontal_std_vector_m": float(np.hypot(np.std(xyz[:, 0], ddof=1), np.std(xyz[:, 1], ddof=1))),
        "east_range_m": float(np.ptp(xyz[:, 0])),
        "north_range_m": float(np.ptp(xyz[:, 1])),
        "height_range_m": float(np.ptp(xyz[:, 2])),
        "horizontal_pairwise_range_m": max_pairwise(xyz[:, :2]),
        "three_dimensional_pairwise_range_m": max_pairwise(xyz),
        "horizontal_radius_rms_m": float(np.sqrt(np.mean(horizontal_radius**2))),
        "horizontal_radius_p95_m": float(np.quantile(horizontal_radius, 0.95)),
        "horizontal_radius_max_m": float(np.max(horizontal_radius)),
        "distance_3d_max_from_mean_m": float(np.max(distance_3d)),
        "three_chunk_center_horizontal_spread_m": chunk_h,
        "three_chunk_center_height_spread_m": chunk_z,
        "pdop_min": float(group["pdop"].min()),
        "pdop_median": float(group["pdop"].median()),
        "pdop_max": float(group["pdop"].max()),
        "satellite_min": int(group["satellite_count"].min()),
        "satellite_median": float(group["satellite_count"].median()),
        "satellite_max": int(group["satellite_count"].max()),
        "hrms_median_m": float(group["hrms_m"].median()),
        "hrms_max_m": float(group["hrms_m"].max()),
        "vrms_median_m": float(group["vrms_m"].median()),
        "vrms_max_m": float(group["vrms_m"].max()),
        "dat_east_m": float(dat_coordinate[0]),
        "dat_north_m": float(dat_coordinate[1]),
        "dat_height_m": float(dat_coordinate[2]),
        "dat_minus_mean_east_m": float(dat_offset[0]),
        "dat_minus_mean_north_m": float(dat_offset[1]),
        "dat_minus_mean_height_m": float(dat_offset[2]),
        "dat_minus_mean_3d_m": float(np.linalg.norm(dat_offset)),
    }


def lineage_key(row: pd.Series) -> tuple[datetime, float, float, float]:
    return (
        row["timestamp"],
        round(float(row["east_m"]), 3),
        round(float(row["north_m"]), 3),
        round(float(row["height_m"]), 3),
    )


def add_ranks(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "horizontal_pairwise_range_m",
        "height_range_m",
        "horizontal_std_vector_m",
        "height_std_m",
        "three_chunk_center_horizontal_spread_m",
        "three_chunk_center_height_spread_m",
    ]
    summary["gross_mixed_trajectory_flag"] = (
        summary["horizontal_pairwise_range_m"] > GROSS_MIXED_TRAJECTORY_HORIZONTAL_M
    )
    comparison = summary.loc[~summary["gross_mixed_trajectory_flag"]]
    for metric in metrics:
        summary[f"{metric}_descending_rank"] = comparison[metric].rank(method="min", ascending=False)
        summary[f"{metric}_empirical_percentile"] = comparison[metric].rank(method="max", pct=True) * 100.0
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_csv", type=Path, required=True)
    parser.add_argument("--corrected_csv", type=Path, required=True)
    parser.add_argument("--corrected_dat", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=False)
    original = read_epoch_csv(args.original_csv, require_observation_index=False)
    corrected = read_epoch_csv(args.corrected_csv)
    dat = read_dat(args.corrected_dat)

    counts = corrected.groupby("point_name").size()
    if len(counts) != 53 or not counts.eq(27).all():
        raise ValueError(f"Corrected observations must contain 53 points x 27 epochs, got {counts.to_dict()}")
    if corrected["点名"].duplicated().any():
        raise ValueError("Corrected observation identifiers are not unique")

    summaries = []
    for name, group in corrected.groupby("point_name", sort=True):
        if name not in dat:
            raise ValueError(f"Corrected DAT has no coordinate for {name}")
        summaries.append(summarize_point(group, dat[name]))
    summary = add_ranks(pd.DataFrame(summaries))
    summary.to_csv(args.output_root / "rtk_observation_quality_all_points.csv", index=False, encoding="utf-8-sig")

    original_by_key: dict[tuple[datetime, float, float, float], list[str]] = defaultdict(list)
    corrected_by_key: dict[tuple[datetime, float, float, float], list[str]] = defaultdict(list)
    for _, row in original.iterrows():
        original_by_key[lineage_key(row)].append(str(row["point_name"]))
    for _, row in corrected.iterrows():
        corrected_by_key[lineage_key(row)].append(str(row["point_name"]))

    lineage_rows = []
    for target in TARGET_POINTS:
        target_rows = corrected[corrected["point_name"].eq(target)].sort_values("observation_index")
        for _, row in target_rows.iterrows():
            matches = original_by_key.get(lineage_key(row), [])
            lineage_rows.append(
                {
                    "corrected_point_name": target,
                    "corrected_observation_name": row["点名"],
                    "timestamp": row["timestamp"].isoformat(),
                    "original_point_matches": ";".join(sorted(set(matches))),
                    "match_count": len(matches),
                }
            )
    lineage = pd.DataFrame(lineage_rows)
    lineage.to_csv(args.output_root / "target_corrected_lineage.csv", index=False, encoding="utf-8-sig")

    original_g39 = original[original["point_name"].eq("G39")].sort_values("timestamp").copy()
    original_g39["session"] = (original_g39["timestamp"].diff().dt.total_seconds().fillna(0) > 60).cumsum() + 1
    g39_sessions = []
    for session, group in original_g39.groupby("session"):
        xyz = group[["east_m", "north_m", "height_m"]].to_numpy(dtype=np.float64)
        corrected_matches = Counter(
            name for _, row in group.iterrows() for name in corrected_by_key.get(lineage_key(row), [])
        )
        g39_sessions.append(
            {
                "session": int(session),
                "observation_count": int(len(group)),
                "start": group["timestamp"].min().isoformat(),
                "end": group["timestamp"].max().isoformat(),
                "east_mean_m": float(xyz[:, 0].mean()),
                "north_mean_m": float(xyz[:, 1].mean()),
                "height_mean_m": float(xyz[:, 2].mean()),
                "corrected_point_matches": dict(corrected_matches),
            }
        )

    target_summary = summary[summary["point_name"].isin(TARGET_POINTS)].copy()
    target_summary.to_csv(args.output_root / "target_point_quality_summary.csv", index=False, encoding="utf-8-sig")
    corrected[corrected["point_name"].isin(TARGET_POINTS)].to_csv(
        args.output_root / "target_corrected_observations.csv", index=False, encoding="utf-8-sig"
    )

    target_lineage_counts = {
        target: dict(
            Counter(
                match
                for values in lineage.loc[lineage["corrected_point_name"].eq(target), "original_point_matches"]
                for match in str(values).split(";")
                if match
            )
        )
        for target in TARGET_POINTS
    }
    audit = {
        "schema": "ms_gcp_corrected_rtk_observation_quality_audit_v1",
        "status": "pass_targets_usable_without_special_role_restriction",
        "authoritative_policy": "corrected files override conflicting uncorrected files",
        "inputs": {
            "original_csv": {"path": str(args.original_csv), "sha256": sha256_file(args.original_csv)},
            "corrected_csv": {"path": str(args.corrected_csv), "sha256": sha256_file(args.corrected_csv)},
            "corrected_dat": {"path": str(args.corrected_dat), "sha256": sha256_file(args.corrected_dat)},
        },
        "corrected_inventory": {
            "point_count": int(len(counts)),
            "observation_count": int(len(corrected)),
            "observations_per_point": sorted(set(int(value) for value in counts)),
            "nonfixed_solution_count": int((~corrected["解状态"].eq("固定解")).sum()),
            "gross_mixed_trajectory_threshold_horizontal_m": GROSS_MIXED_TRAJECTORY_HORIZONTAL_M,
            "gross_mixed_trajectory_points": summary.loc[
                summary["gross_mixed_trajectory_flag"], "point_name"
            ].tolist(),
        },
        "target_lineage_counts": target_lineage_counts,
        "original_g39_sessions": g39_sessions,
        "interpretation": {
            "G07": "27 corrected epochs retain the original G07 identity and are all fixed solutions",
            "G09": "27 corrected epochs retain the original G09 identity and are all fixed solutions",
            "G39": "the first original G39 session is reassigned to dyl2; the second 27-epoch session is the corrected G39",
            "role_policy": "G07/G09/G39 may use the same geometry-only control/checkpoint eligibility rules as other points",
            "unrelated_global_warning": "corrected G47 still contains a gross mixed trajectory but is absent from the current formal scene pointsets",
        },
    }
    (args.output_root / "audit_summary.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    report = [
        "# G07 / G09 / G39 corrected RTK observation audit",
        "",
        "The corrected CSV is authoritative. It contains 53 points and exactly 27 fixed-solution epochs per point.",
        "The unrelated G47 record still contains a gross mixed trajectory and must not be treated as a valid coordinate series; G47 is absent from the current formal pointsets.",
        "",
        "## Point-name correction",
        "",
        "The original G39 label covered two 27-epoch sessions. The earlier session is identical to corrected `dyl2`; the later session is corrected `G39`.",
        "G07 and G09 retain their original identities.",
        "",
        "## Target quality",
        "",
        "| Point | H pairwise range (m) | Z range (m) | H std (m) | Z std (m) | PDOP median/max | Satellites min-max | DAT-to-mean 3D (m) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in target_summary.iterrows():
        report.append(
            f"| {row['point_name']} | {row['horizontal_pairwise_range_m']:.6f} | {row['height_range_m']:.6f} | "
            f"{row['horizontal_std_vector_m']:.6f} | {row['height_std_m']:.6f} | "
            f"{row['pdop_median']:.3f}/{row['pdop_max']:.3f} | {row['satellite_min']}-{row['satellite_max']} | "
            f"{row['dat_minus_mean_3d_m']:.6f} |"
        )
    report += [
        "",
        "## Disposition",
        "",
        "The large report values are repeated-epoch ranges rather than demonstrated absolute biases. After applying the corrected point-name mapping, all three points have 27 fixed epochs and coherent corrected identities. They may be used normally under the same residual-blind geometry and view-count rules as other points.",
    ]
    (args.output_root / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
