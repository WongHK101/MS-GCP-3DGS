#!/usr/bin/env python3
"""Summarize an isolated COLMAP mapper benchmark without modifying its outputs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from statistics import mean


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code" / "colmap" / "utils"))

from read_write_model import read_model  # noqa: E402


def summarize_model(path: Path) -> dict:
    model = read_model(str(path))
    if model is None:
        raise ValueError(f"Unreadable COLMAP model: {path}")
    cameras, images, points = model
    point_errors = [float(point.error) for point in points.values()]
    track_lengths = [len(point.image_ids) for point in points.values()]
    observations = sum(
        int((image.point3D_ids >= 0).sum()) for image in images.values()
    )
    return {
        "model_path": str(path),
        "camera_count": len(cameras),
        "registered_image_count": len(images),
        "point3D_count": len(points),
        "observation_count": observations,
        "mean_reprojection_error_px": mean(point_errors) if point_errors else None,
        "mean_track_length": mean(track_lengths) if track_lengths else None,
    }


def select_largest_model(sparse_root: Path) -> tuple[Path, dict, list[dict]]:
    candidates = []
    for path in sorted(sparse_root.iterdir()):
        if not path.is_dir():
            continue
        try:
            stats = summarize_model(path)
        except ValueError:
            continue
        candidates.append(
            (
                stats["registered_image_count"],
                stats["point3D_count"],
                path,
                stats,
            )
        )
    if not candidates:
        raise FileNotFoundError(f"No readable COLMAP model under {sparse_root}")

    _, _, path, stats = max(candidates, key=lambda row: row[:2])
    all_models = [row[3] for row in sorted(candidates, key=lambda row: row[2].name)]
    return path, stats, all_models


def parse_time_file(path: Path) -> dict:
    values = {}
    if not path.exists():
        return values
    for line in path.read_text(errors="replace").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    if "elapsed_seconds" in values:
        values["elapsed_seconds"] = int(values["elapsed_seconds"])
        values["elapsed_minutes"] = values["elapsed_seconds"] / 60.0
    if "mapper_rc" in values:
        values["mapper_rc"] = int(values["mapper_rc"])
    return values


def parse_baseline_elapsed_minutes(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    text = log_path.read_text(errors="replace")
    matches = re.findall(r"Elapsed time:\s*([0-9.]+)\s*\[?minutes\]?", text)
    return max(map(float, matches)) if matches else None


def read_json(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def parse_gpu_trace(path: Path) -> dict:
    if not path.exists():
        return {}
    memory = []
    utilization = []
    power = []
    with path.open(newline="", errors="replace") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                memory.append(float(row["memory.used"]))
                utilization.append(float(row["utilization.gpu"]))
                power.append(float(row["power.draw"]))
            except (KeyError, TypeError, ValueError):
                continue
    return {
        "sample_count": len(memory),
        "max_memory_used_mib": max(memory) if memory else None,
        "max_gpu_utilization_pct": max(utilization) if utilization else None,
        "max_power_draw_w": max(power) if power else None,
        "nonzero_gpu_utilization_samples": sum(value > 0 for value in utilization),
    }


def build_summary(args: argparse.Namespace) -> dict:
    candidate_run = args.candidate_run.resolve()
    baseline_run = args.baseline_run.resolve()
    candidate_model, candidate_stats, candidate_models = select_largest_model(
        candidate_run / "RGB" / "distorted" / "sparse"
    )
    baseline_model, baseline_stats, baseline_models = select_largest_model(
        baseline_run / "RGB" / "distorted" / "sparse"
    )
    candidate_time = parse_time_file(candidate_run / "logs" / "time.txt")
    baseline_elapsed = parse_baseline_elapsed_minutes(
        baseline_run / "logs" / "colmap_spatial_gpu.log"
    )
    speedup = None
    if baseline_elapsed and candidate_time.get("elapsed_minutes"):
        speedup = baseline_elapsed / candidate_time["elapsed_minutes"]

    candidate_alignment = read_json(args.candidate_alignment)
    baseline_alignment = read_json(args.baseline_alignment)
    alignment_comparison = None
    if candidate_alignment and baseline_alignment:
        alignment_comparison = {
            "mean_error_delta_m": (
                candidate_alignment["alignment_error_m"]["mean"]
                - baseline_alignment["alignment_error_m"]["mean"]
            ),
            "median_error_delta_m": (
                candidate_alignment["alignment_error_m"]["median"]
                - baseline_alignment["alignment_error_m"]["median"]
            ),
            "candidate_mean_error_m": candidate_alignment["alignment_error_m"][
                "mean"
            ],
            "baseline_mean_error_m": baseline_alignment["alignment_error_m"]["mean"],
            "candidate_median_error_m": candidate_alignment["alignment_error_m"][
                "median"
            ],
            "baseline_median_error_m": baseline_alignment["alignment_error_m"][
                "median"
            ],
        }

    summary = {
        "schema": "ms_gcp_colmap_mapper_benchmark_v1",
        "scene": args.scene,
        "candidate": {
            "run_root": str(candidate_run),
            "model": candidate_stats,
            "model_count": len(candidate_models),
            "models": candidate_models,
            "timing": candidate_time,
            "gpu_trace": parse_gpu_trace(candidate_run / "logs" / "gpu_trace.csv"),
            "alignment": candidate_alignment,
        },
        "baseline": {
            "run_root": str(baseline_run),
            "model": baseline_stats,
            "model_count": len(baseline_models),
            "models": baseline_models,
            "elapsed_minutes": baseline_elapsed,
            "alignment": baseline_alignment,
        },
        "comparison": {
            "registered_image_delta": (
                candidate_stats["registered_image_count"]
                - baseline_stats["registered_image_count"]
            ),
            "point3D_ratio_candidate_over_baseline": (
                candidate_stats["point3D_count"] / baseline_stats["point3D_count"]
                if baseline_stats["point3D_count"]
                else None
            ),
            "reprojection_error_delta_px": (
                candidate_stats["mean_reprojection_error_px"]
                - baseline_stats["mean_reprojection_error_px"]
            ),
            "mapper_speedup_candidate_over_baseline": speedup,
            "alignment": alignment_comparison,
        },
        "interpretation_boundary": (
            "The baseline uses COLMAP 3.9.1 CPU bundle adjustment while the "
            "candidate uses COLMAP 4.0.4 GPU bundle adjustment. Runtime "
            "differences therefore combine COLMAP-version and solver-backend effects."
        ),
        "selected_models": {
            "candidate": str(candidate_model),
            "baseline": str(baseline_model),
        },
    }
    return summary


def write_markdown(summary: dict, path: Path) -> None:
    candidate = summary["candidate"]
    baseline = summary["baseline"]
    comparison = summary["comparison"]
    lines = [
        "# COLMAP Mapper GPU-BA Feasibility Summary",
        "",
        f"- Scene: `{summary['scene']}`",
        (
            "- Candidate registered images: "
            f"{candidate['model']['registered_image_count']}"
        ),
        (
            "- Baseline registered images: "
            f"{baseline['model']['registered_image_count']}"
        ),
        (
            "- Candidate mapper time: "
            f"{candidate['timing'].get('elapsed_minutes', 'unavailable')} min"
        ),
        f"- Baseline mapper time: {baseline['elapsed_minutes']} min",
        (
            "- Candidate/baseline speedup: "
            f"{comparison['mapper_speedup_candidate_over_baseline']}"
        ),
        (
            "- Candidate points / baseline points: "
            f"{comparison['point3D_ratio_candidate_over_baseline']}"
        ),
        (
            "- Mean reprojection-error delta: "
            f"{comparison['reprojection_error_delta_px']} px"
        ),
        (
            "- WGS84-derived ENU mean alignment error, candidate / baseline: "
            f"{comparison.get('alignment', {}).get('candidate_mean_error_m')} m / "
            f"{comparison.get('alignment', {}).get('baseline_mean_error_m')} m"
            if comparison.get("alignment")
            else "- WGS84-derived ENU alignment comparison: unavailable"
        ),
        (
            "- GPU peak memory / utilization / power: "
            f"{candidate['gpu_trace'].get('max_memory_used_mib')} MiB / "
            f"{candidate['gpu_trace'].get('max_gpu_utilization_pct')}% / "
            f"{candidate['gpu_trace'].get('max_power_draw_w')} W"
        ),
        "",
        "## Interpretation Boundary",
        "",
        summary["interpretation_boundary"],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--scene", default="gcp_10000_20260610")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--candidate-alignment", type=Path)
    parser.add_argument("--baseline-alignment", type=Path)
    args = parser.parse_args()

    summary = build_summary(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    write_markdown(summary, args.output_md)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
