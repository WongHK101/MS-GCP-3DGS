#!/usr/bin/env python3
"""Independent artifact and ranking verifier for M3M-GCP LiDAR formal v1."""

from __future__ import annotations

import argparse
import hashlib
import json
from functools import cmp_to_key
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np


PROTOCOL_ID = "m3m_gcp_lidar_rendered_surface_v1"
METRIC_FIELDS = (
    "accuracy_mean_m",
    "accuracy_median_m",
    "accuracy_p95_m",
    "completeness_mean_m",
    "completeness_median_m",
    "completeness_p95_m",
    "chamfer_l1_mean_m",
    "symmetric_rmse_m",
    "precision_5cm",
    "recall_5cm",
    "fscore_5cm",
    "precision_10cm",
    "recall_10cm",
    "fscore_10cm",
    "precision_20cm",
    "recall_20cm",
    "fscore_20cm",
)
SCENE_RANK_KEYS = (
    ("fscore_10cm", "descending"),
    ("chamfer_l1_mean_m", "ascending"),
    ("precision_10cm", "descending"),
)
OVERALL_RANK_KEYS = (
    ("macro_fscore_10cm", "descending"),
    ("macro_chamfer_l1_mean_m", "ascending"),
    ("macro_precision_10cm", "descending"),
)
RANK_TOLERANCE = 1e-9


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: Any, *, self_field: str = "canonical_sha256") -> str:
    clean = dict(payload)
    clean.pop(self_field, None)
    raw = json.dumps(clean, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def recompute_metrics(
    reconstruction_to_lidar_m: np.ndarray,
    lidar_to_reconstruction_m: np.ndarray,
    *,
    threshold_epsilon_m: float = 1e-9,
) -> dict[str, float]:
    accuracy = reconstruction_to_lidar_m
    completeness = lidar_to_reconstruction_m
    metrics = {
        "accuracy_mean_m": float(np.mean(accuracy)),
        "accuracy_median_m": float(np.median(accuracy)),
        "accuracy_p95_m": float(np.percentile(accuracy, 95)),
        "completeness_mean_m": float(np.mean(completeness)),
        "completeness_median_m": float(np.median(completeness)),
        "completeness_p95_m": float(np.percentile(completeness, 95)),
        "chamfer_l1_mean_m": float((np.mean(accuracy) + np.mean(completeness)) / 2),
        "symmetric_rmse_m": float(
            np.sqrt((np.mean(np.square(accuracy)) + np.mean(np.square(completeness))) / 2)
        ),
    }
    for threshold, label in ((0.05, "5cm"), (0.10, "10cm"), (0.20, "20cm")):
        precision = float(np.mean(accuracy <= threshold + threshold_epsilon_m))
        recall = float(np.mean(completeness <= threshold + threshold_epsilon_m))
        fscore = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        metrics[f"precision_{label}"] = precision
        metrics[f"recall_{label}"] = recall
        metrics[f"fscore_{label}"] = fscore
    return metrics


def compare_rows(
    left: dict[str, Any],
    right: dict[str, Any],
    keys: Iterable[tuple[str, str]],
    *,
    tolerance: float = RANK_TOLERANCE,
) -> int:
    for field, direction in keys:
        delta = float(left[field]) - float(right[field])
        if abs(delta) <= tolerance:
            continue
        if direction == "descending":
            return -1 if delta > 0 else 1
        if direction == "ascending":
            return -1 if delta < 0 else 1
        raise ValueError(f"unknown ranking direction: {direction}")
    return 0


def competition_rank_rows(
    rows: Iterable[dict[str, Any]],
    keys: Iterable[tuple[str, str]],
    *,
    tolerance: float = RANK_TOLERANCE,
) -> list[dict[str, Any]]:
    keys = tuple(keys)
    ordered = sorted(
        (dict(row) for row in rows),
        key=cmp_to_key(lambda a, b: compare_rows(a, b, keys, tolerance=tolerance)),
    )
    grouped: list[list[dict[str, Any]]] = []
    for row in ordered:
        if not grouped or compare_rows(grouped[-1][0], row, keys, tolerance=tolerance) != 0:
            grouped.append([row])
        else:
            grouped[-1].append(row)
    ranked: list[dict[str, Any]] = []
    consumed = 0
    for group in grouped:
        rank = consumed + 1
        for row in sorted(group, key=lambda item: str(item["method_id"])):
            row["rank"] = rank
            ranked.append(row)
        consumed += len(group)
    return ranked


def _load_npz_exact(path: Path, keys: set[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        actual = set(payload.files)
        if actual != keys:
            raise ValueError(f"{path.name}: NPZ keys {sorted(actual)} != {sorted(keys)}")
        return {key: np.asarray(payload[key]) for key in keys}


def validate_point_npz(path: Path, *, expected_origin: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    payload = _load_npz_exact(path, {"points_local_m", "local_origin_utm49n_normal_height_m"})
    points = payload["points_local_m"]
    origin = payload["local_origin_utm49n_normal_height_m"]
    if points.dtype != np.float64 or points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 1:
        raise ValueError(f"{path.name}: points must be nonempty float64 [N,3]")
    if origin.dtype != np.float64 or origin.shape != (3,):
        raise ValueError(f"{path.name}: origin must be float64 [3]")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(origin)):
        raise ValueError(f"{path.name}: nonfinite point/origin value")
    if expected_origin is not None and not np.array_equal(origin, expected_origin):
        raise ValueError(f"{path.name}: local origin differs from reference")
    return points, origin


def validate_distance_npz(path: Path, *, reconstruction_count: int, reference_count: int) -> tuple[np.ndarray, np.ndarray]:
    payload = _load_npz_exact(path, {"reconstruction_to_lidar_m", "lidar_to_reconstruction_m"})
    accuracy = payload["reconstruction_to_lidar_m"]
    completeness = payload["lidar_to_reconstruction_m"]
    if accuracy.dtype != np.float64 or accuracy.shape != (reconstruction_count,):
        raise ValueError("reconstruction_to_lidar_m must be float64 [N_reconstruction]")
    if completeness.dtype != np.float64 or completeness.shape != (reference_count,):
        raise ValueError("lidar_to_reconstruction_m must be float64 [N_reference]")
    if not np.all(np.isfinite(accuracy)) or not np.all(np.isfinite(completeness)):
        raise ValueError("distance arrays contain nonfinite values")
    if np.any(accuracy < 0) or np.any(completeness < 0):
        raise ValueError("distance arrays contain negative values")
    return accuracy, completeness


def validate_archive_manifest(path: Path, root: Path) -> list[str]:
    errors: list[str] = []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "m3m_gcp_lidar_lightweight_archive_manifest_v1":
        errors.append("archive schema mismatch")
    if payload.get("canonical_sha256") != canonical_sha256(payload):
        errors.append("archive canonical SHA mismatch")
    for row in payload.get("inventory", []):
        if set(row) != {"relative_path", "bytes", "sha256"}:
            errors.append("archive inventory row fields mismatch")
            continue
        rel = PurePosixPath(str(row["relative_path"]))
        if rel.is_absolute() or ".." in rel.parts:
            errors.append(f"unsafe archive path: {rel}")
            continue
        target = root.joinpath(*rel.parts)
        if not target.is_file():
            errors.append(f"archive file missing: {rel}")
            continue
        if target.stat().st_size != int(row["bytes"]):
            errors.append(f"archive byte count mismatch: {rel}")
        if sha256_file(target) != row["sha256"]:
            errors.append(f"archive SHA mismatch: {rel}")
    return errors


def verify_method_result(*, result_path: Path, reference_path: Path, schema_path: Path) -> dict[str, Any]:
    errors: list[str] = []
    result = json.loads(result_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    method_dir = result_path.parent
    surface_path = method_dir / "surface_voxel_centres_local_metric.npz"
    distance_path = method_dir / "nearest_neighbor_distances.npz"
    try:
        reference, origin = validate_point_npz(reference_path)
        surface, _ = validate_point_npz(surface_path, expected_origin=origin)
        accuracy, completeness = validate_distance_npz(
            distance_path,
            reconstruction_count=len(surface),
            reference_count=len(reference),
        )
    except (OSError, ValueError) as exc:
        return {"status": "FAIL", "errors": [str(exc)]}

    if result.get("schema") != "m3m_gcp_lidar_method_result_v1":
        errors.append("method result schema mismatch")
    if result.get("protocol_id") != PROTOCOL_ID:
        errors.append("method result protocol mismatch")
    if result.get("canonical_sha256") != canonical_sha256(result):
        errors.append("method result canonical SHA mismatch")
    if set(result.get("metrics", {})) != set(METRIC_FIELDS):
        errors.append("method metric field inventory mismatch")
    if result.get("surface_npz_sha256") != sha256_file(surface_path):
        errors.append("surface NPZ SHA mismatch")
    if result.get("distance_npz_sha256") != sha256_file(distance_path):
        errors.append("distance NPZ SHA mismatch")
    if result.get("reference_npz_sha256") != sha256_file(reference_path):
        errors.append("reference NPZ SHA mismatch")
    if result.get("artifact_schema_sha256") != sha256_file(schema_path):
        errors.append("artifact schema SHA mismatch")
    if schema.get("protocol_id") != PROTOCOL_ID:
        errors.append("artifact schema protocol mismatch")
    counts = {
        "reference_point_count": len(reference),
        "reconstruction_point_count": len(surface),
        "reconstruction_to_lidar_distance_count": len(accuracy),
        "lidar_to_reconstruction_distance_count": len(completeness),
    }
    for field, expected in counts.items():
        if int(result.get(field, -1)) != expected:
            errors.append(f"{field} mismatch")
    recomputed = recompute_metrics(accuracy, completeness)
    for field in METRIC_FIELDS:
        if not np.isclose(float(result.get("metrics", {}).get(field, np.nan)), recomputed[field], atol=1e-12, rtol=0):
            errors.append(f"metric mismatch: {field}")
    return {
        "status": "PASS_VERIFIED_FORMAL_V1" if not errors else "FAIL",
        "method_id": result.get("method_id"),
        "scene": result.get("scene"),
        "errors": errors,
        "recomputed_metrics": recomputed,
        "distance_npz_sha256": sha256_file(distance_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--artifact-schema", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify_method_result(
        result_path=args.result.resolve(),
        reference_path=args.reference.resolve(),
        schema_path=args.artifact_schema.resolve(),
    )
    report["verifier_sha256"] = sha256_file(Path(__file__).resolve())
    report["canonical_sha256"] = canonical_sha256(report)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.exists():
            raise FileExistsError("refusing to overwrite verification report")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
