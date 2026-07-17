from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


STAT_FIELDS = ("rmse_h_m", "rmse_z_m", "rmse_3d_m", "median_3d_m", "p90_3d_m", "p95_3d_m", "max_3d_m")


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def residual_stats(rows: list[dict[str, str]]) -> dict[str, float | int]:
    horizontal = np.asarray([float(row["error_h_m"]) for row in rows], dtype=np.float64)
    vertical = np.asarray([float(row["error_z_m"]) for row in rows], dtype=np.float64)
    distance = np.asarray([float(row["error_3d_m"]) for row in rows], dtype=np.float64)
    if not len(rows):
        raise ValueError("cannot recompute residual statistics from an empty role")
    return {
        "count": len(rows),
        "rmse_h_m": float(np.sqrt(np.mean(horizontal * horizontal))),
        "rmse_z_m": float(np.sqrt(np.mean(vertical * vertical))),
        "rmse_3d_m": float(np.sqrt(np.mean(distance * distance))),
        "median_3d_m": float(np.median(distance)),
        "p90_3d_m": float(np.percentile(distance, 90)),
        "p95_3d_m": float(np.percentile(distance, 95)),
        "max_3d_m": float(np.max(distance)),
    }


def fit_sim3(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or source.shape[0] < 3:
        raise ValueError(f"invalid Sim(3) fixture shapes: {source.shape} {target.shape}")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / source.shape[0]
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(3, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vt
    source_variance = float(np.mean(np.sum(source_centered * source_centered, axis=1)))
    if source_variance <= 0:
        raise ValueError("degenerate control source variance")
    scale = float(np.sum(singular * np.diag(correction)) / source_variance)
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def row_xyz(row: dict[str, str], prefix: str) -> np.ndarray:
    return np.asarray([float(row[f"{prefix}_x"]), float(row[f"{prefix}_y"]), float(row[f"{prefix}_z"])], dtype=np.float64)


def verify(eval_dir: Path, tolerance_m: float = 1e-9) -> dict[str, Any]:
    summary_path = eval_dir / "method_gcp_eval_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    controls = load_csv(eval_dir / "method_gcp_sim3_control_residuals.csv")
    checkpoints = load_csv(eval_dir / "method_gcp_checkpoint_residuals.csv")
    observations = load_csv(eval_dir / "method_gcp_observation_points.csv")
    aggregated = load_csv(eval_dir / "method_gcp_aggregated_points.csv")

    source = np.stack([row_xyz(row, "model") for row in controls])
    target = np.stack([row_xyz(row, "target") for row in controls])
    scale, rotation, translation = fit_sim3(source, target)
    declared = summary["transform"]
    declared_rotation = np.asarray(declared["rotation"], dtype=np.float64)
    declared_translation = np.asarray(declared["translation"], dtype=np.float64)
    transform_errors = {
        "scale_abs_error": abs(scale - float(declared["scale"])),
        "rotation_max_abs_error": float(np.max(np.abs(rotation - declared_rotation))),
        "translation_max_abs_error_m": float(np.max(np.abs(translation - declared_translation))),
    }

    max_row_error = 0.0
    for row in controls + checkpoints:
        model = row_xyz(row, "model")
        expected_target = row_xyz(row, "target")
        predicted = scale * (rotation @ model) + translation
        residual = predicted - expected_target
        horizontal = float(math.hypot(residual[0], residual[1]))
        vertical = float(abs(residual[2]))
        distance = float(np.linalg.norm(residual))
        declared_values = np.asarray(
            [
                float(row["predicted_x"]),
                float(row["predicted_y"]),
                float(row["predicted_z"]),
                float(row["residual_x_m"]),
                float(row["residual_y_m"]),
                float(row["residual_z_m"]),
                float(row["error_h_m"]),
                float(row["error_z_m"]),
                float(row["error_3d_m"]),
            ],
            dtype=np.float64,
        )
        recomputed_values = np.concatenate([predicted, residual, [horizontal, vertical, distance]])
        max_row_error = max(max_row_error, float(np.max(np.abs(recomputed_values - declared_values))))

    recomputed_stats = {
        "control": residual_stats(controls),
        "checkpoint": residual_stats(checkpoints),
        "all": residual_stats(controls + checkpoints),
    }
    max_summary_error = 0.0
    for role, values in recomputed_stats.items():
        declared_values = summary["residual_stats"][role]
        if int(declared_values["count"]) != int(values["count"]):
            raise ValueError(f"{role} residual count mismatch")
        for field in STAT_FIELDS:
            max_summary_error = max(max_summary_error, abs(float(values[field]) - float(declared_values[field])))

    control_names = [row["point_name"] for row in controls]
    checkpoint_names = [row["point_name"] for row in checkpoints]
    identity_passed = (
        control_names == list(summary["control_points_used"])
        and checkpoint_names == list(summary["checkpoint_points_used"])
    )
    valid_observations = sum(str(row.get("valid", "")).strip().lower() in {"1", "true"} for row in observations)
    valid_aggregated = sum(str(row.get("valid", "")).strip().lower() in {"1", "true"} for row in aggregated)
    coverage_passed = (
        len(observations) == int(summary["raw_observation_rows"])
        and valid_observations == int(summary["valid_observation_rows"])
        and valid_aggregated == int(summary["aggregated_gcp_count"])
        and not summary.get("missing_control_points")
        and not summary.get("missing_checkpoint_points")
    )
    max_numeric_error = max(
        max_row_error,
        max_summary_error,
        transform_errors["translation_max_abs_error_m"],
        transform_errors["scale_abs_error"],
        transform_errors["rotation_max_abs_error"],
    )
    passed = identity_passed and coverage_passed and max_numeric_error <= tolerance_m
    result = {
        "schema": "ms_gcp_eval_output_independent_recomputation_v1",
        "scene": summary["scene"],
        "method_id": summary["method_id"],
        "passed": passed,
        "tolerance": tolerance_m,
        "identity_passed": identity_passed,
        "coverage_passed": coverage_passed,
        "control_names": control_names,
        "checkpoint_names": checkpoint_names,
        "observation_rows": len(observations),
        "valid_observation_rows": valid_observations,
        "aggregated_valid_gcp_count": valid_aggregated,
        "transform_recomputed": {
            "scale": scale,
            "rotation": rotation.tolist(),
            "translation": translation.tolist(),
        },
        "transform_errors": transform_errors,
        "max_per_point_numeric_abs_error": max_row_error,
        "max_summary_stat_abs_error": max_summary_error,
        "recomputed_residual_stats": recomputed_stats,
    }
    if not passed:
        raise ValueError(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Independently recompute and verify formal GCP evaluator outputs.")
    parser.add_argument("--eval_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args()
    result = verify(Path(args.eval_dir), tolerance_m=float(args.tolerance))
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
