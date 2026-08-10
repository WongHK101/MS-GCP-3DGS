#!/usr/bin/env python3
"""Independently verify M3M native-quarter evaluator output tables."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PROTOCOL_ID = "m3m_gcp_native_quarter_geometry_v2"
STAT_FIELDS = ("rmse_h_m", "rmse_z_m", "rmse_3d_m", "median_3d_m", "p95_3d_m", "max_3d_m")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def vector(row: dict[str, str], fields: tuple[str, str, str]) -> np.ndarray:
    return np.asarray([float(row[field]) for field in fields], dtype=np.float64)


def residual_stats(residuals: list[np.ndarray]) -> dict[str, Any]:
    if not residuals:
        return {"count": 0, **{field: None for field in STAT_FIELDS}}
    values = np.vstack(residuals).astype(np.float64)
    horizontal = np.linalg.norm(values[:, :2], axis=1)
    vertical = np.abs(values[:, 2])
    distance = np.linalg.norm(values, axis=1)
    return {
        "count": int(values.shape[0]),
        "rmse_h_m": float(np.sqrt(np.mean(horizontal * horizontal))),
        "rmse_z_m": float(np.sqrt(np.mean(vertical * vertical))),
        "rmse_3d_m": float(np.sqrt(np.mean(distance * distance))),
        "median_3d_m": float(np.median(distance)),
        "p95_3d_m": float(np.percentile(distance, 95)),
        "max_3d_m": float(np.max(distance)),
    }


def _max_circular_bin_separation(bins: list[int]) -> int:
    maximum = 0
    for index, left in enumerate(bins):
        for right in bins[index + 1 :]:
            direct = abs(left - right)
            maximum = max(maximum, min(direct, 8 - direct))
    return maximum


def verify(eval_dir: Path, *, tolerance: float = 1e-9) -> dict[str, Any]:
    summary_path = eval_dir / "evaluation_summary.json"
    point_path = eval_dir / "point_results.csv"
    observation_path = eval_dir / "observation_samples.csv"
    manifest_path = eval_dir / "evaluator_manifest.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    points = load_csv(point_path)
    observations = load_csv(observation_path)
    errors: list[str] = []
    max_numeric_error = 0.0

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    def compare(actual: float, expected: float, label: str) -> None:
        nonlocal max_numeric_error
        error = abs(float(actual) - float(expected))
        max_numeric_error = max(max_numeric_error, error)
        if not math.isfinite(error) or error > tolerance:
            errors.append(f"{label} differs by {error}")

    require(summary.get("schema") == "m3m_gcp_native_quarter_method_evaluation_v2", "summary schema mismatch")
    require(manifest.get("schema") == "m3m_gcp_native_quarter_evaluator_run_manifest_v2", "manifest schema mismatch")
    require(summary.get("protocol_id") == PROTOCOL_ID, "protocol ID mismatch")
    require(summary.get("method_specific_sim3_fitted") is False, "method-specific Sim(3) was fitted")
    require(summary.get("physical_surface_claim") is False, "unexpected physical-surface claim")
    require(manifest.get("sim3_policy") == "frozen_common_transform_no_method_refit", "Sim(3) policy mismatch")
    require(manifest.get("operator") == "bilinear_raw_moment_ratio_v1", "operator mismatch")
    require(manifest.get("ranking_policy") == "complete_checkpoint_coverage_only_v1", "ranking policy mismatch")

    for name in ("observation_samples.csv", "point_results.csv", "evaluation_summary.json"):
        require(sha256_file(eval_dir / name) == manifest.get("outputs", {}).get(name), f"output SHA mismatch: {name}")
    for path_field, sha_field in (
        ("protocol_release_manifest", "protocol_release_manifest_sha256"),
        ("source_data_contract", "source_data_contract_sha256"),
        ("packet_manifest", "packet_manifest_sha256"),
    ):
        path = Path(str(manifest.get(path_field, "")))
        require(path.is_file(), f"manifest dependency is missing: {path_field}")
        if path.is_file():
            require(sha256_file(path) == manifest.get(sha_field), f"manifest dependency SHA mismatch: {path_field}")
    require(summary.get("packet_manifest_sha256") == manifest.get("packet_manifest_sha256"), "packet manifest binding differs")

    sim3_path = Path(str(summary.get("common_sim3_path", "")))
    require(sim3_path.is_file(), "common Sim(3) file is missing")
    sim3 = json.loads(sim3_path.read_text(encoding="utf-8")) if sim3_path.is_file() else {}
    require(sha256_file(sim3_path) == summary.get("common_sim3_sha256") if sim3_path.is_file() else False, "common Sim(3) SHA mismatch")
    require(sim3.get("protocol_id") == PROTOCOL_ID, "common Sim(3) protocol mismatch")
    require(sim3.get("scene") == summary.get("scene"), "common Sim(3) scene mismatch")
    require(sim3.get("method_result_refit_forbidden") is True, "common Sim(3) does not forbid method refits")
    transform = sim3.get("transform", {})
    scale = float(transform.get("scale", math.nan))
    rotation = np.asarray(transform.get("rotation", []), dtype=np.float64)
    translation = np.asarray(transform.get("translation", []), dtype=np.float64)
    require(rotation.shape == (3, 3) and translation.shape == (3,), "common Sim(3) shape mismatch")

    point_by_name: dict[str, dict[str, str]] = {}
    for row in points:
        name = row["point_name"]
        require(name not in point_by_name, f"duplicate point row: {name}")
        point_by_name[name] = row
        require(row.get("scene") == summary.get("scene"), f"point scene mismatch: {name}")
        require(row.get("role") in {"control", "checkpoint"}, f"point role mismatch: {name}")
    require(bool(point_by_name), "point result table is empty")

    observations_by_point: dict[str, list[dict[str, str]]] = defaultdict(list)
    failure_counts: Counter[str] = Counter()
    for row in observations:
        name = row["point_name"]
        observations_by_point[name].append(row)
        require(name in point_by_name, f"observation references an unknown point: {name}")
        require(row.get("scene") == summary.get("scene"), f"observation scene mismatch: {name}")
        if name in point_by_name:
            require(row.get("role") == point_by_name[name].get("role"), f"observation role mismatch: {name}")
        if not truthy(row.get("valid")):
            failure_counts[str(row.get("failure_reason", ""))] += 1
    require(dict(sorted(failure_counts.items())) == summary.get("observation_failure_counts"), "observation failure counts mismatch")

    residuals: dict[str, list[np.ndarray]] = {"control": [], "checkpoint": [], "all": []}
    role_totals: Counter[str] = Counter()
    role_passed: Counter[str] = Counter()
    for name, row in point_by_name.items():
        role = row["role"]
        role_totals[role] += 1
        point_observations = observations_by_point.get(name, [])
        valid = [item for item in point_observations if truthy(item.get("valid"))]
        valid_nadir = sum(item.get("view_class", "").lower() == "nadir" for item in valid)
        valid_oblique = sum(item.get("view_class", "").lower() == "oblique" for item in valid)
        oblique_bins = sorted({int(item["azimuth_bin_45deg"]) for item in valid if item.get("view_class", "").lower() == "oblique"})
        declared_bins = list(ast.literal_eval(row.get("valid_oblique_azimuth_bins_45deg", "[]")))
        require(int(row["expected_observation_count"]) == len(point_observations), f"expected observation count mismatch: {name}")
        require(int(row["valid_observation_count"]) == len(valid), f"valid observation count mismatch: {name}")
        require(int(row["valid_nadir_count"]) == valid_nadir, f"valid nadir count mismatch: {name}")
        require(int(row["valid_oblique_count"]) == valid_oblique, f"valid oblique count mismatch: {name}")
        require(int(row["valid_oblique_azimuth_bin_count"]) == len(oblique_bins), f"oblique-bin count mismatch: {name}")
        require(declared_bins == oblique_bins, f"oblique-bin inventory mismatch: {name}")
        require(int(row["max_oblique_azimuth_circular_bin_separation"]) == _max_circular_bin_separation(oblique_bins), f"oblique-bin separation mismatch: {name}")
        passed = truthy(row.get("passed"))
        expected_pass = (
            len(valid) >= max(4, math.ceil(0.5 * len(point_observations)))
            and valid_nadir >= 2
            and valid_oblique >= 2
            and len(oblique_bins) >= 2
            and _max_circular_bin_separation(oblique_bins) >= 2
        )
        require(passed == expected_pass, f"coverage decision mismatch: {name}")
        if not passed:
            continue
        role_passed[role] += 1
        model = vector(row, ("model_x", "model_y", "model_z"))
        predicted = vector(row, ("predicted_e_m", "predicted_n_m", "predicted_z_m"))
        target = vector(row, ("target_e_m", "target_n_m", "target_z_m"))
        residual = vector(row, ("residual_e_m", "residual_n_m", "residual_z_m"))
        if rotation.shape == (3, 3) and translation.shape == (3,):
            expected_predicted = scale * (rotation @ model) + translation
            max_numeric_error = max(max_numeric_error, float(np.max(np.abs(predicted - expected_predicted))))
            require(np.allclose(predicted, expected_predicted, atol=tolerance, rtol=0), f"common Sim(3) application mismatch: {name}")
        expected_residual = predicted - target
        max_numeric_error = max(max_numeric_error, float(np.max(np.abs(residual - expected_residual))))
        require(np.allclose(residual, expected_residual, atol=tolerance, rtol=0), f"residual vector mismatch: {name}")
        compare(float(row["error_h_m"]), float(np.linalg.norm(residual[:2])), f"horizontal error: {name}")
        compare(float(row["error_z_m"]), float(abs(residual[2])), f"vertical error: {name}")
        compare(float(row["error_3d_m"]), float(np.linalg.norm(residual)), f"3D error: {name}")
        residuals[role].append(residual)
        residuals["all"].append(residual)

    declared_counts = summary.get("point_counts", {})
    for role in ("control", "checkpoint"):
        require(int(declared_counts.get(f"{role}_total", -1)) == role_totals[role], f"{role} total mismatch")
        require(int(declared_counts.get(f"{role}_passed", -1)) == role_passed[role], f"{role} passed mismatch")
    complete = role_totals["checkpoint"] > 0 and role_passed["checkpoint"] == role_totals["checkpoint"]
    require(summary.get("status") == ("COMPLETE_RANKED" if complete else "INCOMPLETE_UNRANKED"), "ranking status mismatch")
    require(summary.get("ranking_eligible") is complete, "ranking eligibility mismatch")
    compare(float(summary["checkpoint_coverage_rate"]), role_passed["checkpoint"] / role_totals["checkpoint"], "checkpoint coverage rate")

    recomputed_stats = {role: residual_stats(residuals[role]) for role in ("control", "checkpoint", "all")}
    for role, stats in recomputed_stats.items():
        declared = summary.get("residual_statistics", {}).get(role, {})
        require(int(declared.get("count", -1)) == stats["count"], f"{role} residual count mismatch")
        for field in STAT_FIELDS:
            actual = declared.get(field)
            expected = stats[field]
            if expected is None:
                require(actual is None, f"{role} {field} should be null")
            else:
                compare(float(actual), float(expected), f"{role} {field}")

    passed = not errors and max_numeric_error <= tolerance
    result = {
        "schema": "m3m_gcp_native_quarter_evaluator_output_independent_verification_v1",
        "scene": summary.get("scene"),
        "method_id": summary.get("method_id"),
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "tolerance": tolerance,
        "errors": errors,
        "observation_count": len(observations),
        "point_count": len(points),
        "point_counts": dict(declared_counts),
        "ranking_status": summary.get("status"),
        "ranking_eligible": summary.get("ranking_eligible"),
        "method_specific_sim3_fitted": summary.get("method_specific_sim3_fitted"),
        "output_hashes_passed": not any(error.startswith("output SHA mismatch") for error in errors),
        "dependency_hashes_passed": not any("dependency SHA mismatch" in error for error in errors),
        "coverage_recomputation_passed": not any("coverage" in error or "observation count mismatch" in error or "bin" in error for error in errors),
        "common_sim3_recomputation_passed": not any("Sim(3)" in error for error in errors),
        "residual_recomputation_passed": not any("error:" in error or "residual" in error for error in errors),
        "recomputed_residual_statistics": recomputed_stats,
        "max_numeric_abs_error": max_numeric_error,
    }
    if not passed:
        raise ValueError(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval_dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args()
    result = verify(args.eval_dir, tolerance=float(args.tolerance))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
