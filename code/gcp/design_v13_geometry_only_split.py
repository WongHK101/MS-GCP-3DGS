#!/usr/bin/env python3
"""Design a residual-blind, spatially distributed v1.3 GCP split candidate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONTROL_HEIGHT_RANGE_TARGET = 0.80
USER_ACCEPTED_FULL_RTK_STATUS = "full_rtk_27_observation_mean_user_accepted"
FORMAL_COORDINATE_STATUSES = {"primary_usable", USER_ACCEPTED_FULL_RTK_STATUS}


SCENE_RULES = {
    "gcp_3000_20260602": {
        "label": "3K",
        "points": ["G11", "G12", "G13", "G14", "G15", "G16", "G17", "G18", "NC94"],
        "controls": 5,
        "forced_checkpoints": {},
    },
    "gcp_5000_20260602": {
        "label": "5K",
        "points": [f"G{i:02d}" for i in range(1, 11)],
        "controls": 6,
        "forced_checkpoints": {},
    },
    "gcp_10000_20260610": {
        "label": "10K",
        "points": [f"G{i}" for i in range(19, 28)] + ["G49"],
        "controls": 6,
        "forced_checkpoints": {},
    },
    "gcp_20000_20260602": {
        "label": "20K",
        "points": ["G28", "G29", "G30", "G31", "G33", "G35", "G36", "G37", "G38", "dyl2"],
        "controls": 6,
        "forced_checkpoints": {},
    },
    "gcp_50000_20260610": {
        "label": "50K",
        "points": None,
        "controls": 12,
        "forced_checkpoints": {},
    },
    "gcp_100000_20260610": {
        "label": "100K",
        "points": None,
        "controls": 13,
        "forced_checkpoints": {},
    },
}


ANNOTATION_RELATIVE_PATHS = {
    "gcp_3000_20260602": "outputs/gcp_multiview_direct_annotation_tasks_20260713_annotate_direct_v2/working_annotations/gcp_3000_20260602_manual_annotations_v1_3_draft_working.csv",
    "gcp_5000_20260602": "outputs/gcp_map_defined_core_annotation_tasks_20260714_map_core_G04_G07_G09_v1/working_annotations/gcp_5000_20260602_map_core_v1_3_draft_working.csv",
    "gcp_10000_20260610": "outputs/gcp_multiview_direct_annotation_tasks_20260713_annotate_direct_v2/working_annotations/gcp_10000_20260610_manual_annotations_v1_3_draft_working.csv",
    "gcp_20000_20260602": "outputs/gcp_multiview_direct_annotation_tasks_20260713_annotate_direct_v2/working_annotations/gcp_20000_20260602_manual_annotations_v1_3_draft_working.csv",
    "gcp_50000_20260610": "outputs/gcp_followup_annotation_tasks_50k100k_20260715_followup_after_map_core_v1/working_annotations/gcp_50000_20260610_followup_v1_3_draft_working.csv",
    "gcp_100000_20260610": "outputs/gcp_followup_annotation_tasks_50k100k_20260715_followup_after_map_core_v1/working_annotations/gcp_100000_20260610_followup_v1_3_draft_working.csv",
}


@dataclass(frozen=True)
class SplitMetrics:
    checkpoint_inside_count: int
    checkpoint_count: int
    hull_area_ratio: float
    height_range_ratio: float
    max_point_to_control_norm: float
    mean_point_to_control_norm: float
    checkpoint_min_pair_norm: float
    control_min_pair_norm: float

    @property
    def score(self) -> tuple[float, ...]:
        inside_fraction = self.checkpoint_inside_count / max(1, self.checkpoint_count)
        vertical_gate = min(1.0, self.height_range_ratio / CONTROL_HEIGHT_RANGE_TARGET)
        return (
            vertical_gate,
            inside_fraction,
            self.hull_area_ratio,
            self.height_range_ratio,
            -self.max_point_to_control_norm,
            -self.mean_point_to_control_norm,
            self.checkpoint_min_pair_norm,
            self.control_min_pair_norm,
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generator_git_provenance(script_path: Path) -> dict[str, str | bool]:
    repo_root = subprocess.run(
        ["git", "-C", str(script_path.parent), "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", repo_root, *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    status = git("status", "--porcelain")
    if status:
        raise ValueError("Split candidate generation requires a clean Git worktree")
    return {
        "repo_root": repo_root,
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "worktree_clean": True,
        "script_path": str(script_path),
        "script_sha256": sha256_file(script_path),
    }


def convex_hull(points: np.ndarray) -> np.ndarray:
    unique = sorted({(float(x), float(y)) for x, y in points})
    if len(unique) <= 1:
        return np.asarray(unique, dtype=np.float64)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def polygon_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    return float(abs(np.dot(poly[:, 0], np.roll(poly[:, 1], 1)) - np.dot(poly[:, 1], np.roll(poly[:, 0], 1))) / 2.0)


def point_in_convex_polygon(point: np.ndarray, poly: np.ndarray, tolerance: float = 1e-9) -> bool:
    if len(poly) < 3:
        return False
    signs = []
    for idx in range(len(poly)):
        a = poly[idx]
        b = poly[(idx + 1) % len(poly)]
        edge = b - a
        offset = point - a
        signs.append(edge[0] * offset[1] - edge[1] * offset[0])
    values = np.asarray(signs, dtype=np.float64)
    return bool(np.all(values >= -tolerance) or np.all(values <= tolerance))


def min_pair_distance(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    distances[distances == 0] = np.inf
    return float(np.min(distances))


def normalized_geometry(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    xy = frame[["x_m", "y_m"]].to_numpy(dtype=np.float64)
    z = frame[["z_m"]].to_numpy(dtype=np.float64)
    xy_span = np.maximum(np.ptp(xy, axis=0), 1e-12)
    z_span = max(float(np.ptp(z[:, 0])), 1e-12)
    xy_norm = (xy - np.min(xy, axis=0)) / xy_span
    xyz_norm = np.column_stack((xy_norm, 0.35 * (z[:, 0] - np.min(z[:, 0])) / z_span))
    return xy_norm, xyz_norm


def split_metrics(frame: pd.DataFrame, control_indices: Sequence[int]) -> SplitMetrics:
    controls = sorted(control_indices)
    checkpoints = sorted(set(range(len(frame))) - set(controls))
    xy_norm, _ = normalized_geometry(frame)
    control_xy = xy_norm[controls]
    checkpoint_xy = xy_norm[checkpoints]
    hull = convex_hull(control_xy)
    all_hull = convex_hull(xy_norm)
    inside = sum(point_in_convex_polygon(point, hull) for point in checkpoint_xy)
    nearest = np.min(np.linalg.norm(xy_norm[:, None, :] - control_xy[None, :, :], axis=2), axis=1)
    z_all = frame["z_m"].to_numpy(dtype=np.float64)
    z_range = float(np.ptp(z_all[controls])) if len(controls) else 0.0
    z_all_range = max(float(np.ptp(z_all)), 1e-12)
    return SplitMetrics(
        checkpoint_inside_count=int(inside),
        checkpoint_count=len(checkpoints),
        hull_area_ratio=polygon_area(hull) / max(polygon_area(all_hull), 1e-12),
        height_range_ratio=z_range / z_all_range,
        max_point_to_control_norm=float(np.max(nearest)),
        mean_point_to_control_norm=float(np.mean(nearest)),
        checkpoint_min_pair_norm=min_pair_distance(checkpoint_xy),
        control_min_pair_norm=min_pair_distance(control_xy),
    )


def improve_by_swaps(frame: pd.DataFrame, selected: set[int], eligible: set[int]) -> set[int]:
    current = split_metrics(frame, selected)
    while True:
        best = None
        best_metrics = current
        for removed in sorted(selected):
            for added in sorted(eligible - selected):
                proposal = (selected - {removed}) | {added}
                metrics = split_metrics(frame, proposal)
                if metrics.score > best_metrics.score:
                    best = proposal
                    best_metrics = metrics
        if best is None:
            return selected
        selected = best
        current = best_metrics


def choose_controls(frame: pd.DataFrame, control_count: int) -> tuple[set[int], SplitMetrics]:
    eligible = set(frame.index[frame["control_eligible"]].tolist())
    if len(eligible) < control_count:
        raise ValueError(f"Only {len(eligible)} control-eligible points for requested {control_count}")
    _, xyz_norm = normalized_geometry(frame)
    candidates = []
    for seed in sorted(eligible):
        selected = {seed}
        while len(selected) < control_count:
            remaining = sorted(eligible - selected)
            distances = [float(np.min(np.linalg.norm(xyz_norm[idx] - xyz_norm[list(selected)], axis=1))) for idx in remaining]
            selected.add(remaining[int(np.argmax(distances))])
        candidates.append(improve_by_swaps(frame, selected, eligible))
    unique = {tuple(sorted(candidate)): candidate for candidate in candidates}
    best = max(unique.values(), key=lambda candidate: split_metrics(frame, candidate).score)
    return best, split_metrics(frame, best)


def annotation_summary(path: Path) -> pd.DataFrame:
    annotations = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "residual" in " ".join(annotations.columns).lower():
        raise ValueError(f"Forbidden residual field in annotation input: {path}")
    good = annotations[annotations["quality"].str.lower() == "good"].copy()
    return good.groupby("point_name", as_index=False).agg(good_view_count=("image_name", "nunique"))


def coordinate_is_formal_usable(status: str) -> bool:
    return status in FORMAL_COORDINATE_STATUSES


def load_coordinates(release_dir: Path, review_source: Path) -> pd.DataFrame:
    primary = pd.read_csv(release_dir / "gcp_points_primary_usable_cgcs2000_cm108_v1.csv", dtype=str)
    primary = primary.rename(
        columns={
            "cgcs2000_gk_cm108_e_m": "x_m",
            "cgcs2000_gk_cm108_n_m": "y_m",
            "cgcs2000_normal_height_m": "z_m",
        }
    )
    primary["coordinate_status"] = "primary_usable"
    if review_source.suffix.lower() in {".xlsx", ".xls"}:
        review = pd.read_excel(review_source, dtype=str).rename(
            columns={"点名": "point_name", "东坐标": "x_m", "北坐标": "y_m", "高程": "z_m"}
        )
        review = review[review["point_name"].isin({"G07", "G09", "G39"})].copy()
    else:
        review = pd.read_csv(review_source, dtype=str).rename(
            columns={
                "cgcs2000_gk_cm108_e_m": "x_m",
                "cgcs2000_gk_cm108_n_m": "y_m",
                "cgcs2000_normal_height_m": "z_m",
            }
        )
    review = review[review["point_name"].isin({"G07", "G09", "G39"})].copy()
    review["coordinate_status"] = USER_ACCEPTED_FULL_RTK_STATUS
    review = review[["point_name", "x_m", "y_m", "z_m", "coordinate_status"]]
    if set(review["point_name"]) != {"G07", "G09", "G39"}:
        raise ValueError(f"Review-coordinate source must resolve G07/G09/G39 exactly: {review_source}")
    coordinates = pd.concat(
        [primary[["point_name", "x_m", "y_m", "z_m", "coordinate_status"]], review],
        ignore_index=True,
    ).drop_duplicates("point_name", keep="last")
    for column in ["x_m", "y_m", "z_m"]:
        coordinates[column] = pd.to_numeric(coordinates[column], errors="raise")
    return coordinates


def write_csv(path: Path, rows: Iterable[dict], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\r\n")
        writer.writeheader()
        writer.writerows(rows)


def render_scene_plot(frame: pd.DataFrame, controls: set[int], output: Path, title: str) -> None:
    fig, (ax_xy, ax_z) = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    control_frame = frame.loc[sorted(controls)]
    checkpoint_frame = frame.drop(index=sorted(controls))
    hull = convex_hull(control_frame[["x_m", "y_m"]].to_numpy(dtype=np.float64))
    if len(hull) >= 3:
        closed = np.vstack([hull, hull[0]])
        ax_xy.fill(closed[:, 0], closed[:, 1], color="#2878b5", alpha=0.10)
        ax_xy.plot(closed[:, 0], closed[:, 1], color="#2878b5", linewidth=1.5, label="control hull")
    ax_xy.scatter(control_frame["x_m"], control_frame["y_m"], marker="^", s=95, c="#2878b5", label="control")
    colors = ["#ef8a17" if coordinate_is_formal_usable(status) else "#d62728" for status in checkpoint_frame["coordinate_status"]]
    ax_xy.scatter(checkpoint_frame["x_m"], checkpoint_frame["y_m"], marker="o", s=70, c=colors, label="checkpoint")
    for _, row in frame.iterrows():
        ax_xy.annotate(row["point_name"], (row["x_m"], row["y_m"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax_xy.set_title(f"{title}: spatial split")
    ax_xy.set_xlabel("CGCS2000 easting (m)")
    ax_xy.set_ylabel("CGCS2000 northing (m)")
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.grid(alpha=0.25)
    ax_xy.legend(loc="best")

    ordered = frame.sort_values(["role", "point_name"]).reset_index(drop=True)
    z_colors = ["#2878b5" if role == "control" else "#ef8a17" for role in ordered["role"]]
    ax_z.bar(range(len(ordered)), ordered["z_m"], color=z_colors)
    ax_z.set_xticks(range(len(ordered)), ordered["point_name"], rotation=70)
    ax_z.set_ylabel("Normal height (m)")
    ax_z.set_title("Height coverage")
    ax_z.grid(axis="y", alpha=0.25)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact_repo_root", type=Path, required=True)
    parser.add_argument("--release_dir", type=Path, required=True)
    parser.add_argument("--review_coordinate_source", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=False)
    plots_dir = args.output_root / "plots"
    plots_dir.mkdir()
    coordinates = load_coordinates(args.release_dir, args.review_coordinate_source)
    coordinate_by_name = coordinates.set_index("point_name")
    generator_provenance = generator_git_provenance(Path(__file__).resolve())

    split_rows = []
    disposition_rows = []
    summaries = []
    input_records = []
    for scene, rule in SCENE_RULES.items():
        annotation_path = args.artifact_repo_root / ANNOTATION_RELATIVE_PATHS[scene]
        counts = annotation_summary(annotation_path)
        count_by_name = dict(zip(counts["point_name"], counts["good_view_count"]))
        input_records.append({"scene": scene, "path": str(annotation_path), "sha256": sha256_file(annotation_path)})
        if rule["points"] is None:
            points = sorted(name for name, count in count_by_name.items() if int(count) >= 4)
            if scene == "gcp_50000_20260610":
                points = [name for name in points if name != "dyl2"]
            if scene == "gcp_100000_20260610":
                points = [name for name in points if name != "G33"]
        else:
            points = list(rule["points"])
        expected_total = rule["controls"] + ({"3K": 4, "5K": 4, "10K": 4, "20K": 4, "50K": 11, "100K": 12}[rule["label"]])
        if len(points) != expected_total:
            raise ValueError(f"{scene}: expected {expected_total} points, found {len(points)}: {points}")

        rows = []
        for name in points:
            if name not in coordinate_by_name.index:
                raise ValueError(f"Missing coordinate for {scene}/{name}")
            coordinate = coordinate_by_name.loc[name]
            good_count = int(count_by_name.get(name, 0))
            if good_count < 4:
                raise ValueError(f"Formal candidate {scene}/{name} has only {good_count} Good views")
            coordinate_status = str(coordinate["coordinate_status"])
            forced_reason = rule["forced_checkpoints"].get(name, "")
            control_eligible = good_count >= 6 and coordinate_is_formal_usable(coordinate_status) and not forced_reason
            rows.append(
                {
                    "scene": scene,
                    "point_name": name,
                    "x_m": float(coordinate["x_m"]),
                    "y_m": float(coordinate["y_m"]),
                    "z_m": float(coordinate["z_m"]),
                    "good_view_count": good_count,
                    "coordinate_status": coordinate_status,
                    "control_eligible": control_eligible,
                    "forced_checkpoint_reason": forced_reason,
                }
            )
        frame = pd.DataFrame(rows)
        controls, metrics = choose_controls(frame, int(rule["controls"]))
        frame["role"] = ["control" if idx in controls else "checkpoint" for idx in frame.index]
        if any(not coordinate_is_formal_usable(status) for status in frame.loc[list(controls), "coordinate_status"]):
            raise AssertionError(f"{scene}: non-formal coordinate selected as control")
        if any(frame.loc[list(controls), "good_view_count"] < 6):
            raise AssertionError(f"{scene}: low-view point selected as control")

        for _, row in frame.iterrows():
            provisional = not coordinate_is_formal_usable(str(row["coordinate_status"]))
            split_rows.append(
                {
                    "scene": scene,
                    "point_name": row["point_name"],
                    "role": row["role"],
                    "cgcs2000_gk_cm108_e_m": f"{row['x_m']:.3f}",
                    "cgcs2000_gk_cm108_n_m": f"{row['y_m']:.3f}",
                    "cgcs2000_normal_height_m": f"{row['z_m']:.3f}",
                    "good_view_count": int(row["good_view_count"]),
                    "coordinate_status": row["coordinate_status"],
                    "control_eligible": str(bool(row["control_eligible"])).lower(),
                    "split_row_status": "provisional_coordinate_review_required" if provisional else "ready_candidate_not_frozen",
                    "role_reason": row["forced_checkpoint_reason"] or "geometry_only_uniform_distribution_selection",
                }
            )
        selected_names = set(frame["point_name"])
        for name, good_count in sorted(count_by_name.items()):
            if name in selected_names:
                disposition = "selected_for_split_candidate"
                reason = "user_approved_scene_pointset_and_view_threshold"
            elif int(good_count) < 4:
                disposition = "diagnostic_only"
                reason = "insufficient_good_views_lt4"
                if scene == "gcp_100000_20260610" and name == "G33":
                    reason = "insufficient_good_views_lt4_and_no_valid_nadir_coverage"
            else:
                disposition = "excluded_from_formal_scene_pointset"
                reason = "outside_user_approved_survey_area_pointset"
            disposition_rows.append(
                {
                    "scene": scene,
                    "point_name": name,
                    "good_view_count": int(good_count),
                    "disposition": disposition,
                    "reason": reason,
                }
            )

        xy = frame[["x_m", "y_m"]].to_numpy(dtype=np.float64)
        control_xy = frame.loc[sorted(controls), ["x_m", "y_m"]].to_numpy(dtype=np.float64)
        nearest_m = np.min(np.linalg.norm(xy[:, None, :] - control_xy[None, :, :], axis=2), axis=1)
        review_points = sorted(
            frame.loc[~frame["coordinate_status"].map(coordinate_is_formal_usable), "point_name"].tolist()
        )
        scene_status = "provisional_coordinate_review_required" if review_points else "ready_candidate_not_frozen"
        summaries.append(
            {
                "scene": scene,
                "label": rule["label"],
                "total_points": len(frame),
                "control_count": int((frame["role"] == "control").sum()),
                "checkpoint_count": int((frame["role"] == "checkpoint").sum()),
                "checkpoint_inside_control_hull_count": metrics.checkpoint_inside_count,
                "control_hull_area_ratio": f"{metrics.hull_area_ratio:.9f}",
                "control_height_range_ratio": f"{metrics.height_range_ratio:.9f}",
                "max_point_to_nearest_control_m": f"{float(np.max(nearest_m)):.6f}",
                "mean_point_to_nearest_control_m": f"{float(np.mean(nearest_m)):.6f}",
                "coordinate_review_points": ";".join(review_points),
                "scene_split_status": scene_status,
            }
        )
        render_scene_plot(frame, controls, plots_dir / f"{scene}_geometry_only_split.png", rule["label"])

    split_fields = list(split_rows[0])
    disposition_fields = list(disposition_rows[0])
    summary_fields = list(summaries[0])
    write_csv(args.output_root / "gcp_control_checkpoint_split_v1_3_candidate.csv", split_rows, split_fields)
    write_csv(args.output_root / "point_disposition_v1_3_candidate.csv", disposition_rows, disposition_fields)
    write_csv(args.output_root / "split_scene_summary.csv", summaries, summary_fields)

    manifest = {
        "schema": "ms_gcp_geometry_only_split_candidate_v1",
        "status": "candidate_not_release_frozen",
        "generator": generator_provenance,
        "selection_policy": {
            "forbidden_inputs": ["model residual", "RMSE", "depth", "alpha", "variance", "multiview model scatter"],
            "allowed_inputs": ["surveyed XYZ", "Good view count", "coordinate QC", "user-approved scene boundary"],
            "formal_coordinate_statuses": sorted(FORMAL_COORDINATE_STATUSES),
            "full_rtk_coordinate_policy": {
                "points": ["G07", "G09", "G39"],
                "status": USER_ACCEPTED_FULL_RTK_STATUS,
                "decision": "use_normally_without_role_restriction",
                "basis": "reported dispersion is a repeated-observation range, not evidence of absolute coordinate bias; user accepted the 27-observation means on 2026-07-17",
            },
            "control_min_good_views": 6,
            "formal_point_min_good_views": 4,
            "control_height_range_target_ratio": CONTROL_HEIGHT_RANGE_TARGET,
            "objective_order": [
                "reach at least 80 percent of the candidate-set height range when geometrically feasible",
                "checkpoint fraction inside control hull",
                "control hull area ratio",
                "control height range ratio",
                "minimize maximum point-to-control distance",
                "minimize mean point-to-control distance",
                "checkpoint and control spacing",
            ],
        },
        "approved_counts": {scene: {"controls": rule["controls"], "checkpoints": len(rule["points"]) - rule["controls"] if rule["points"] else next(int(x["checkpoint_count"]) for x in summaries if x["scene"] == scene)} for scene, rule in SCENE_RULES.items()},
        "inputs": input_records
        + [
            {"path": str(args.release_dir / "gcp_points_primary_usable_cgcs2000_cm108_v1.csv"), "sha256": sha256_file(args.release_dir / "gcp_points_primary_usable_cgcs2000_cm108_v1.csv")},
            {"path": str(args.review_coordinate_source), "sha256": sha256_file(args.review_coordinate_source)},
        ],
    }
    (args.output_root / "split_design_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    report_lines = [
        "# v1.3 Geometry-only Control/Checkpoint Split Candidate",
        "",
        "This is a candidate split, not a frozen release. No model residual, RMSE, depth, alpha, variance, or model scatter was read.",
        "",
        "## Scene status",
        "",
        "| Scene | Control | Checkpoint | Checkpoints inside control hull | Coordinate review | Status |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in summaries:
        report_lines.append(
            f"| {row['label']} | {row['control_count']} | {row['checkpoint_count']} | "
            f"{row['checkpoint_inside_control_hull_count']}/{row['checkpoint_count']} | "
            f"{row['coordinate_review_points'] or '-'} | {row['scene_split_status']} |"
        )
    report_lines += [
        "",
        "## Important eligibility notes",
        "",
        "- 5K G07/G09 and 50K G39 use the 27-observation RTK mean coordinates accepted by the user on 2026-07-17. Their report provenance is retained, but they receive no special control/checkpoint restriction.",
        "- 50K dyl2 remains diagnostic-only because it has fewer than four Good views and no corrected nadir coverage.",
        "- 100K G33 now has sufficient multi-view annotations but remains outside the user-approved 25-point formal scene pointset.",
        "- 20K G36 and 100K dyl2 now exceed the six-Good-view control threshold and are no longer forced checkpoints.",
        "- v1.2.2 remains unchanged.",
    ]
    (args.output_root / "README.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    output_files = sorted(path for path in args.output_root.rglob("*") if path.is_file())
    hashes = [{"path": path.relative_to(args.output_root).as_posix(), "size": path.stat().st_size, "sha256": sha256_file(path)} for path in output_files]
    (args.output_root / "OUTPUT_SHA256_MANIFEST.json").write_text(json.dumps({"files": hashes}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_root": str(args.output_root), "scenes": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
