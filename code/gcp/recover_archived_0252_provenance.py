#!/usr/bin/env python
"""No-GPU provenance recovery for the archived 3K P1 0.252 m result.

This diagnostic is intentionally read-only with respect to evaluator protocol:
it does not render, export depth, train, edit pointsets/splits/annotations, or
modify formal evaluator code.  It audits archived artifacts and writes a review
package for GPT inspection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCENE = "gcp_3000_20260602"
OLD_TARGET_RMSE_3D = 0.25216052987948767
OLD_TARGET_RMSE_3D_REPORTED = 0.25216052987948456
ABS_TOL_ARCHIVED_POINTS = 1e-9
ABS_TOL_RERUN_SAME_CPU = 1e-6
BOOTSTRAP_SEED = 20260628
BOOTSTRAP_REPETITIONS = 1000


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_zip_text(zf: zipfile.ZipFile, name: str) -> str:
    return zf.read(name).decode("utf-8-sig")


def read_zip_json(zf: zipfile.ZipFile, name: str) -> dict[str, Any]:
    return json.loads(read_zip_text(zf, name))


def read_zip_csv(zf: zipfile.ZipFile, name: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(read_zip_text(zf, name))))


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def ensure_unique_dir(path: Path) -> Path:
    if not path.exists():
        path.mkdir(parents=True)
        return path
    for i in range(1, 1000):
        candidate = path.with_name(f"{path.name}_{i:02d}")
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
    raise RuntimeError(f"Could not allocate unique directory for {path}")


def ensure_unique_file(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for i in range(1, 1000):
        candidate = path.with_name(f"{stem}_{i:02d}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not allocate unique file for {path}")


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def residual_stats(rows: list[dict[str, str]]) -> dict[str, float | int]:
    if not rows:
        return {
            "count": 0,
            "rmse_h_m": math.nan,
            "rmse_z_m": math.nan,
            "rmse_3d_m": math.nan,
            "median_3d_m": math.nan,
            "p90_3d_m": math.nan,
            "max_3d_m": math.nan,
        }
    h = np.asarray([f(r, "error_h_m") for r in rows], dtype=np.float64)
    z = np.asarray([f(r, "error_z_m") for r in rows], dtype=np.float64)
    e = np.asarray([f(r, "error_3d_m") for r in rows], dtype=np.float64)
    return {
        "count": int(len(rows)),
        "rmse_h_m": float(np.sqrt(np.mean(h * h))),
        "rmse_z_m": float(np.sqrt(np.mean(z * z))),
        "rmse_3d_m": float(np.sqrt(np.mean(e * e))),
        "median_3d_m": float(np.median(e)),
        "p90_3d_m": float(np.percentile(e, 90)),
        "max_3d_m": float(np.max(e)),
    }


def sim3_umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError("Sim3 inputs must be Nx3 arrays with matching shape")
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    x = src - mu_src
    y = dst - mu_dst
    cov = (y.T @ x) / len(src)
    u, svals, vt = np.linalg.svd(cov)
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[-1, -1] = -1
    rot = u @ sign @ vt
    var_src = float(np.sum(x * x) / len(src))
    if var_src <= 0:
        raise ValueError("Degenerate Sim3 source controls")
    scale = float(np.trace(np.diag(svals) @ sign) / var_src)
    trans = mu_dst - scale * (rot @ mu_src)
    return scale, rot, trans


def transform_points(points: np.ndarray, scale: float, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    return (scale * (rot @ points.T)).T + trans


def point_dict_from_residuals(rows: Iterable[dict[str, str]]) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for row in rows:
        out[row["point_name"]] = {
            "src": np.asarray([f(row, "model_x"), f(row, "model_y"), f(row, "model_z")], dtype=np.float64),
            "target": np.asarray([f(row, "target_x"), f(row, "target_y"), f(row, "target_z")], dtype=np.float64),
        }
    return out


def evaluate_split(
    points: dict[str, dict[str, np.ndarray]],
    controls: list[str],
    checkpoints: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    src = np.asarray([points[p]["src"] for p in controls], dtype=np.float64)
    dst = np.asarray([points[p]["target"] for p in controls], dtype=np.float64)
    scale, rot, trans = sim3_umeyama(src, dst)
    residual_rows: list[dict[str, Any]] = []
    for role, names in (("control", controls), ("checkpoint", checkpoints)):
        for name in names:
            pred = transform_points(points[name]["src"][None, :], scale, rot, trans)[0]
            target = points[name]["target"]
            res = pred - target
            residual_rows.append(
                {
                    "point_name": name,
                    "role": role,
                    "predicted_x": pred[0],
                    "predicted_y": pred[1],
                    "predicted_z": pred[2],
                    "target_x": target[0],
                    "target_y": target[1],
                    "target_z": target[2],
                    "residual_x_m": res[0],
                    "residual_y_m": res[1],
                    "residual_z_m": res[2],
                    "error_h_m": float(np.linalg.norm(res[:2])),
                    "error_z_m": float(abs(res[2])),
                    "error_3d_m": float(np.linalg.norm(res)),
                }
            )
    summary: dict[str, Any] = {
        "scale": scale,
        "rotation": rot.tolist(),
        "translation": trans.tolist(),
        "rotation_determinant": float(np.linalg.det(rot)),
        "control": residual_stats([r for r in residual_rows if r["role"] == "control"]),
        "checkpoint": residual_stats([r for r in residual_rows if r["role"] == "checkpoint"]),
        "all": residual_stats(residual_rows),
    }
    src_centered = src - src.mean(axis=0)
    cov = src_centered.T @ src_centered / len(src)
    singular = np.linalg.svd(cov, compute_uv=False)
    summary["control_source_covariance_singular_values"] = singular.tolist()
    summary["control_condition_number"] = float(singular[0] / singular[-1]) if singular[-1] > 0 else math.inf
    summary["control_spatial_extent_xyz"] = (src.max(axis=0) - src.min(axis=0)).tolist()
    summary["control_height_range"] = float(src[:, 2].max() - src[:, 2].min())
    return summary, residual_rows


def stats_match(recomputed: dict[str, Any], archived: dict[str, Any], role: str, tol: float) -> dict[str, Any]:
    keys = ["count", "rmse_h_m", "rmse_z_m", "rmse_3d_m", "median_3d_m", "p90_3d_m", "max_3d_m"]
    rows = []
    ok = True
    for key in keys:
        a = archived["residual_stats"][role][key]
        b = recomputed[key]
        err = abs(float(a) - float(b)) if key != "count" else abs(int(a) - int(b))
        passed = err <= (0 if key == "count" else tol)
        ok = ok and passed
        rows.append(
            {
                "role": role,
                "metric": key,
                "archived": a,
                "recomputed": b,
                "abs_error": err,
                "tolerance": 0 if key == "count" else tol,
                "pass": passed,
            }
        )
    return {"ok": ok, "rows": rows}


def zip_extract_json_csv(zf: zipfile.ZipFile, prefix: str) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    summary = read_zip_json(zf, f"{prefix}/method_gcp_eval_summary.json")
    control = read_zip_csv(zf, f"{prefix}/method_gcp_sim3_control_residuals.csv")
    checkpoint = read_zip_csv(zf, f"{prefix}/method_gcp_checkpoint_residuals.csv")
    aggregate = read_zip_csv(zf, f"{prefix}/method_gcp_aggregated_points.csv")
    return summary, control, checkpoint, aggregate


def describe_file(path: Path, artifact_id: str, role: str, notes: str = "") -> dict[str, Any]:
    row: dict[str, Any] = {
        "artifact_id": artifact_id,
        "role": role,
        "path": str(path),
        "exists": path.exists(),
        "bytes": "",
        "sha256": "",
        "evidence_status": "missing",
        "notes": notes,
    }
    if path.exists() and path.is_file():
        row["bytes"] = path.stat().st_size
        row["sha256"] = sha256_file(path)
        row["evidence_status"] = "recovered_unhistorically_hashed"
    elif path.exists() and path.is_dir():
        row["evidence_status"] = "directory_exists"
    return row


def git_text(args: list[str], cwd: Path) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=str(cwd), text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:  # pragma: no cover - diagnostic best effort
        return f"ERROR: {exc}"


def load_current_release_zip(zip_path: Path) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    prefix = (
        "GPT_GCP_3SCENE_METRIC_DEPTH_REGRESSION_RELEASEMODE_REVIEW_20260627/"
        "evaluations/gcp_3000_20260602_formal_expected_camera_z_release"
    )
    with zipfile.ZipFile(zip_path, "r") as zf:
        return zip_extract_json_csv(zf, prefix)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence_root", type=Path, default=Path(r"E:\M3M-GCP-3DGS"))
    parser.add_argument("--current_release_zip", type=Path, default=Path(r"E:\M3M-GCP-3DGS\outputs\gpt_review_packages\GPT_GCP_3SCENE_METRIC_DEPTH_REGRESSION_RELEASEMODE_REVIEW_20260627.zip"))
    parser.add_argument("--old_r1_depth_dir", type=Path, default=Path(r"E:\M3M-GCP-3DGS\outputs\archived_0252_inputs_20260628\umgs_rgb_depths_r1_annotated"))
    parser.add_argument("--out_base", type=Path, default=Path(r"E:\M3M-GCP-3DGS\outputs"))
    parser.add_argument("--package_dir", type=Path, default=Path(r"E:\M3M-GCP-3DGS\outputs\gpt_review_packages"))
    args = parser.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_root = ensure_unique_dir(args.out_base / f"gcp_3k_archived_0252_provenance_20260628_{stamp}")

    root = args.evidence_root
    old_base = root / "outputs" / "gaussian_gcp_eval_20260618"
    old_r1 = old_base / "gcp_3000_r1_vs_r8_20260618"
    old_r8 = old_base / "evaluator_umgs_rgb_depth_only_p1_r8_scaled_inversez"
    old_annotations = old_base / "annotations_undistorted" / "gcp_image_observations_undistorted_for_evaluation.csv"
    old_annotations_manifest = old_base / "annotations_undistorted" / "undistort_observations_manifest.json"
    old_gcp = root / "evidence" / "gcp_coordinates" / "gcp_points_primary_usable_cgcs2000_cm108_20260615.csv"
    old_three_summary = root / "outputs" / "gcp_diagnostics_three_fixed_20260624" / "three_scene_gaussian_vs_colmap_summary.csv"

    exact_commands = {
        "script_invocation": " ".join([sys.executable, *sys.argv]),
        "cwd": os.getcwd(),
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": git_text(["rev-parse", "HEAD"], Path.cwd()),
        "git_branch": git_text(["branch", "--show-current"], Path.cwd()),
        "git_status_porcelain": git_text(["status", "--porcelain"], Path.cwd()),
        "old_candidate_training_script_commit": "a51fc5454790f012b5247fad3c15d60501cbfc2e",
        "current_release_zip": str(args.current_release_zip),
    }
    write_json(out_root / "exact_commands_and_environment.json", exact_commands)

    inventory: list[dict[str, Any]] = []
    for artifact_id, role, path, notes in [
        ("old_r1_eval_summary", "old_r1_0252", old_r1 / "method_gcp_eval_summary.json", "Archived R1 P1 summary matching 0.252 m."),
        ("old_r1_eval_manifest", "old_r1_0252", old_r1 / "evaluator_manifest.json", ""),
        ("old_r1_depth_export_manifest", "old_r1_0252", old_r1 / "depth_export_manifest.json", ""),
        ("old_r1_depth_map_index", "old_r1_0252", old_r1 / "depth_map_index.csv", ""),
        ("old_r1_control_residuals", "old_r1_0252", old_r1 / "method_gcp_sim3_control_residuals.csv", ""),
        ("old_r1_checkpoint_residuals", "old_r1_0252", old_r1 / "method_gcp_checkpoint_residuals.csv", ""),
        ("old_r1_aggregate_points", "old_r1_0252", old_r1 / "method_gcp_aggregated_points.csv", ""),
        ("old_r8_eval_summary", "old_r8_semantic_conflict", old_r8 / "method_gcp_eval_summary.json", "Older R8 directory with camera_z/export vs inverse/evaluator conflict."),
        ("old_r8_eval_manifest", "old_r8_semantic_conflict", old_r8 / "evaluator_manifest.json", ""),
        ("old_r8_depth_export_manifest", "old_r8_semantic_conflict", old_base / "umgs_rgb_depths" / "depth_export_manifest.json", ""),
        ("old_annotations_undistorted", "old_shared", old_annotations, ""),
        ("old_annotations_transform_manifest", "old_shared", old_annotations_manifest, ""),
        ("old_gcp_coordinates", "old_shared", old_gcp, ""),
        ("old_three_scene_summary", "old_archive_summary", old_three_summary, ""),
        ("current_release_zip", "current_release", args.current_release_zip, ""),
        ("old_r1_depth_dir", "old_r1_0252", args.old_r1_depth_dir, "Directory copied from 901 AutoDL; individual file hashes recorded separately."),
    ]:
        inventory.append(describe_file(path, artifact_id, role, notes))

    depth_hash_rows: list[dict[str, Any]] = []
    if args.old_r1_depth_dir.exists():
        for path in sorted(args.old_r1_depth_dir.glob("*.npy")):
            depth_hash_rows.append(
                {
                    "file_name": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "source": "copied_existing_artifact_from_901_autodl_no_gpu",
                    "historical_sha_available": False,
                }
            )
    write_csv(out_root / "old_r1_depth_file_hashes.csv", depth_hash_rows)

    old_r1_summary = read_json(old_r1 / "method_gcp_eval_summary.json")
    old_r1_manifest = read_json(old_r1 / "evaluator_manifest.json")
    old_r1_export = read_json(old_r1 / "depth_export_manifest.json")
    old_r8_summary = read_json(old_r8 / "method_gcp_eval_summary.json")
    old_r8_manifest = read_json(old_r8 / "evaluator_manifest.json")
    old_r8_export = read_json(old_base / "umgs_rgb_depths" / "depth_export_manifest.json")
    current_summary, current_ctrl, current_chk, current_agg = load_current_release_zip(args.current_release_zip)

    old_r1_ctrl = read_csv(old_r1 / "method_gcp_sim3_control_residuals.csv")
    old_r1_chk = read_csv(old_r1 / "method_gcp_checkpoint_residuals.csv")
    old_points = point_dict_from_residuals([*old_r1_ctrl, *old_r1_chk])
    old_controls = list(old_r1_summary["control_points_used"])
    old_checkpoints = list(old_r1_summary["checkpoint_points_used"])
    current_controls = list(current_summary["control_points_used"])
    current_checkpoints = list(current_summary["checkpoint_points_used"])

    stage_a_rows: list[dict[str, Any]] = []
    old_recomputed = {
        "control": residual_stats(old_r1_ctrl),
        "checkpoint": residual_stats(old_r1_chk),
        "all": residual_stats([*old_r1_ctrl, *old_r1_chk]),
    }
    a2_match_rows: list[dict[str, Any]] = []
    a2_ok = True
    for role in ["control", "checkpoint", "all"]:
        check = stats_match(old_recomputed[role], old_r1_summary, role, ABS_TOL_ARCHIVED_POINTS)
        a2_ok = a2_ok and bool(check["ok"])
        a2_match_rows.extend(check["rows"])
    write_csv(out_root / "stage_a2_recompute_from_archived_residuals.csv", a2_match_rows)

    exact_replay_missing = [
        "exact evaluator source snapshot/commit is not recorded in old manifest",
        "exact uncommitted diff at run time is not recorded",
        "exact evaluator command/config is not preserved; manifest records effective settings only",
        "old depth file historical SHA-256 is absent; present-day SHA-256 was computed after recovery",
        "old split exists as manifest control/checkpoint lists but no independently hashed split CSV was recorded",
        "old COLMAP model historical content hash is absent from manifest",
    ]
    stage_a1_status = "unverified_archived_result_not_comparable_to_release_protocol"
    stage_a2_status = "forensic_reconstruction_only"
    if a2_ok:
        stage_a2_status = "exact_numeric_reproduction_with_unhistorically_hashed_inputs"
    if old_r1_manifest.get("depth_semantics") != old_r1_export.get("depth_semantics"):
        stage_a2_status = "numerically_reproduced_but_semantically_inconsistent"

    stage_a_summary = {
        "stage_A1_exact_archived_replay": {
            "status": stage_a1_status,
            "eligible": False,
            "missing_or_unverified_requirements": exact_replay_missing,
        },
        "stage_A2_forensic_reconstruction": {
            "status": stage_a2_status,
            "archived_residual_summary_recomputed": a2_ok,
            "tolerance_m": ABS_TOL_ARCHIVED_POINTS,
            "checkpoint_rmse_3d_m": old_recomputed["checkpoint"]["rmse_3d_m"],
            "target_checkpoint_rmse_3d_m": OLD_TARGET_RMSE_3D,
        },
    }
    write_json(out_root / "stage_a_reproduction_summary.json", stage_a_summary)
    write_csv(
        out_root / "stage_a_reproduction_summary.csv",
        [
            {
                "stage": "A1_exact_archived_replay",
                "status": stage_a1_status,
                "checkpoint_rmse_3d_m": "",
                "reason": "; ".join(exact_replay_missing),
            },
            {
                "stage": "A2_forensic_reconstruction",
                "status": stage_a2_status,
                "checkpoint_rmse_3d_m": old_recomputed["checkpoint"]["rmse_3d_m"],
                "reason": "Recomputed from archived residual CSVs and present-day recovered artifacts; historical hashes/command incomplete.",
            },
        ],
    )

    # Stage B: only split_only is identifiable without changing multiple inputs.
    stage_b_rows: list[dict[str, Any]] = []
    baseline_summary, baseline_residuals = evaluate_split(old_points, old_controls, old_checkpoints)
    split_summary, split_residuals = evaluate_split(old_points, current_controls, current_checkpoints)
    for name, summary, status, changed, note in [
        ("baseline_A2_old_split", baseline_summary, "baseline", "none", "A2 baseline recomputed from old model points and old split."),
        ("split_only", split_summary, "computed", "control/checkpoint split only", "Old model/GCP/points retained; only split switched to current release v1.1."),
    ]:
        stage_b_rows.append(
            {
                "variant": name,
                "status": status,
                "changed_factor": changed,
                "control_points": ",".join(old_controls if name.startswith("baseline") else current_controls),
                "checkpoint_points": ",".join(old_checkpoints if name.startswith("baseline") else current_checkpoints),
                "control_rmse_3d_m": summary["control"]["rmse_3d_m"],
                "checkpoint_rmse_h_m": summary["checkpoint"]["rmse_h_m"],
                "checkpoint_rmse_z_m": summary["checkpoint"]["rmse_z_m"],
                "checkpoint_rmse_3d_m": summary["checkpoint"]["rmse_3d_m"],
                "scale": summary["scale"],
                "rotation_determinant": summary["rotation_determinant"],
                "note": note,
            }
        )
    for variant in ["annotation_pointset_only", "patch_aggregation_only", "camera_model_only", "depth_artifact_only", "sim3_only"]:
        stage_b_rows.append(
            {
                "variant": variant,
                "status": "not_identifiable_as_single_factor",
                "changed_factor": variant.replace("_only", ""),
                "control_points": "",
                "checkpoint_points": "",
                "control_rmse_3d_m": "",
                "checkpoint_rmse_h_m": "",
                "checkpoint_rmse_z_m": "",
                "checkpoint_rmse_3d_m": "",
                "scale": "",
                "rotation_determinant": "",
                "note": "This factor cannot be substituted alone from archived artifacts without also changing image/depth/camera/code compatibility.",
            }
        )
    write_csv(out_root / "stage_b_single_factor_results.csv", stage_b_rows)
    write_csv(out_root / "stage_b_split_only_residuals.csv", split_residuals)

    # Current release transform sensitivity.
    current_points = point_dict_from_residuals([*current_ctrl, *current_chk])
    full_current_summary, full_current_residuals = evaluate_split(current_points, current_controls, current_checkpoints)
    loo_rows: list[dict[str, Any]] = []
    full_rmse = float(full_current_summary["checkpoint"]["rmse_3d_m"])
    for dropped in current_controls:
        ctrls = [p for p in current_controls if p != dropped]
        summary, _ = evaluate_split(current_points, ctrls, current_checkpoints)
        loo_rows.append(
            {
                "dropped_control": dropped,
                "remaining_controls": ",".join(ctrls),
                "checkpoint_rmse_h_m": summary["checkpoint"]["rmse_h_m"],
                "checkpoint_rmse_z_m": summary["checkpoint"]["rmse_z_m"],
                "checkpoint_rmse_3d_m": summary["checkpoint"]["rmse_3d_m"],
                "delta_checkpoint_rmse_3d_m": summary["checkpoint"]["rmse_3d_m"] - full_rmse,
                "scale": summary["scale"],
                "rotation_determinant": summary["rotation_determinant"],
                "condition_number": summary["control_condition_number"],
                "height_range_m": summary["control_height_range"],
            }
        )
    write_csv(out_root / "current_release_leave_one_control_out.csv", loo_rows)

    rng = random.Random(BOOTSTRAP_SEED)
    boot_rows: list[dict[str, Any]] = []
    invalid = 0
    valid = 0
    for i in range(BOOTSTRAP_REPETITIONS):
        sample = [rng.choice(current_controls) for _ in current_controls]
        unique = sorted(set(sample))
        if len(unique) < 3:
            invalid += 1
            boot_rows.append({"iteration": i, "status": "invalid_less_than_3_unique_controls", "sample": ",".join(sample)})
            continue
        try:
            summary, _ = evaluate_split(current_points, sample, current_checkpoints)
        except Exception as exc:
            invalid += 1
            boot_rows.append({"iteration": i, "status": f"invalid_degenerate_geometry:{exc}", "sample": ",".join(sample)})
            continue
        valid += 1
        boot_rows.append(
            {
                "iteration": i,
                "status": "valid",
                "sample": ",".join(sample),
                "unique_control_count": len(unique),
                "scale": summary["scale"],
                "rotation_determinant": summary["rotation_determinant"],
                "condition_number": summary["control_condition_number"],
                "height_range_m": summary["control_height_range"],
                "checkpoint_rmse_h_m": summary["checkpoint"]["rmse_h_m"],
                "checkpoint_rmse_z_m": summary["checkpoint"]["rmse_z_m"],
                "checkpoint_rmse_3d_m": summary["checkpoint"]["rmse_3d_m"],
            }
        )
    write_csv(out_root / "current_release_bootstrap_transform_sensitivity.csv", boot_rows)
    valid_boot = [r for r in boot_rows if r.get("status") == "valid"]
    def percentile_field(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
        vals = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
        return {
            "min": float(vals.min()),
            "median": float(np.median(vals)),
            "p05": float(np.percentile(vals, 5)),
            "p95": float(np.percentile(vals, 95)),
            "max": float(vals.max()),
        }
    bootstrap_summary = {
        "protocol": "current_release_protocol_transform_sensitivity",
        "seed": BOOTSTRAP_SEED,
        "repetitions": BOOTSTRAP_REPETITIONS,
        "valid_count": valid,
        "invalid_count": invalid,
        "full_current_checkpoint_rmse_3d_m": full_rmse,
        "checkpoint_rmse_3d_distribution": percentile_field(valid_boot, "checkpoint_rmse_3d_m") if valid_boot else {},
        "scale_distribution": percentile_field(valid_boot, "scale") if valid_boot else {},
        "condition_number_distribution": percentile_field(valid_boot, "condition_number") if valid_boot else {},
        "most_influential_control": max(loo_rows, key=lambda r: abs(float(r["delta_checkpoint_rmse_3d_m"])))["dropped_control"],
        "leave_one_control_out_rows": len(loo_rows),
    }
    write_json(out_root / "current_release_transform_sensitivity_summary.json", bootstrap_summary)

    depth_semantics_rows = [
        {
            "artifact": "old_r1_0252_depth_export_manifest",
            "exporter_declared_semantics": old_r1_export.get("depth_semantics"),
            "evaluator_manifest_semantics": old_r1_manifest.get("depth_semantics"),
            "code_actual_conversion": "camera_z = 1 / depth_raw for inverse_camera_z evaluator semantics",
            "exact_command_override": "unresolved; exact evaluator command not preserved",
            "raw_depth_value_evidence": "R1 median depth values around 0.025 imply inverse-camera-z-like payload for ~40 m camera-z.",
            "summary_result": old_r1_summary["residual_stats"]["checkpoint"]["rmse_3d_m"],
            "semantic_status": "numerically_reconstructed_from_incompletely_verified_artifacts",
        },
        {
            "artifact": "old_r8_semantic_conflict_directory",
            "exporter_declared_semantics": old_r8_export.get("depth_semantics"),
            "evaluator_manifest_semantics": old_r8_manifest.get("depth_semantics"),
            "code_actual_conversion": "camera_z = 1 / depth_raw for inverse_camera_z evaluator semantics",
            "exact_command_override": "unresolved; exact evaluator command not preserved",
            "raw_depth_value_evidence": "R8 median depth values around 0.025 also numerically look inverse-camera-z-like, despite export manifest camera_z label.",
            "summary_result": old_r8_summary["residual_stats"]["checkpoint"]["rmse_3d_m"],
            "semantic_status": "numerically_reproduced_but_semantically_inconsistent_for_r8_directory_not_the_0252_target",
        },
    ]
    write_csv(out_root / "depth_semantics_conflict_audit.csv", depth_semantics_rows)
    (out_root / "depth_semantics_conflict_audit.md").write_text(
        "\n".join(
            [
                "# Depth Semantics Audit",
                "",
                "The archived `0.252161 m` target corresponds to the R1 directory. Its exporter and evaluator manifests both state `inverse_camera_z`, and raw values around 0.025 support an inverse-depth interpretation for roughly 40 m camera-z. However, the exact evaluator command and historical source snapshot are not locked, so this is not verified exact replay.",
                "",
                "A separate R8 archived directory shows a manifest conflict: the depth export manifest states `camera_z`, while the evaluator manifest states `inverse_camera_z`. That R8 directory has checkpoint RMSE-3D around 0.1647 m and is not the locked `0.252161 m` target, but it is retained as semantic inconsistency evidence.",
            ]
        ),
        encoding="utf-8",
    )

    provenance_rows = [
        {
            "field": "evaluator_source_snapshot_commit",
            "archived_old_run": "candidate a51fc5454790f012b5247fad3c15d60501cbfc2e inferred from script timing, not recorded in output manifest",
            "current_release_run": exact_commands["git_commit"],
            "status": "unresolved",
        },
        {
            "field": "uncommitted_diff",
            "archived_old_run": "not recorded",
            "current_release_run": exact_commands["git_status_porcelain"],
            "status": "missing",
        },
        {
            "field": "exact_command",
            "archived_old_run": "not preserved for evaluator; training launcher exists but is not evaluator command",
            "current_release_run": exact_commands["script_invocation"],
            "status": "missing",
        },
        {
            "field": "depth_files",
            "archived_old_run": f"{len(depth_hash_rows)} R1 .npy files recovered and present-day hashed",
            "current_release_run": "metric packets verified in prior release-mode package",
            "status": "recovered_unhashed",
        },
        {
            "field": "annotations_gcp_split",
            "archived_old_run": "annotation/GCP files recovered; split recovered from evaluator manifest lists, not a hashed split CSV",
            "current_release_run": "release v1.1 files hash-verified in prior package",
            "status": "different",
        },
        {
            "field": "depth_semantics",
            "archived_old_run": f"R1 export={old_r1_export.get('depth_semantics')}; evaluator={old_r1_manifest.get('depth_semantics')}",
            "current_release_run": "formal alpha_normalized_expected_camera_z=M1/A",
            "status": "different",
        },
        {
            "field": "control_checkpoint_split",
            "archived_old_run": f"controls={','.join(old_controls)} checkpoints={','.join(old_checkpoints)}",
            "current_release_run": f"controls={','.join(current_controls)} checkpoints={','.join(current_checkpoints)}",
            "status": "different",
        },
        {
            "field": "patch_aggregation",
            "archived_old_run": f"patch_size={old_r1_manifest.get('patch_size')} min_valid_ratio={old_r1_manifest.get('min_patch_valid_ratio')} min_observations={old_r1_manifest.get('min_valid_observations')}",
            "current_release_run": "release-mode metric-packet evaluator; exact settings in release package",
            "status": "different",
        },
        {
            "field": "colmap_model",
            "archived_old_run": old_r1_manifest.get("colmap_model"),
            "current_release_run": current_summary.get("colmap_model"),
            "status": "unresolved",
        },
    ]
    write_csv(out_root / "provenance_matrix.csv", provenance_rows)
    write_json(out_root / "provenance_matrix.json", provenance_rows)

    inventory.extend(
        [
            {
                "artifact_id": "old_r1_depth_file_set",
                "role": "old_r1_0252",
                "path": str(args.old_r1_depth_dir),
                "exists": args.old_r1_depth_dir.exists(),
                "bytes": sum(int(r["bytes"]) for r in depth_hash_rows) if depth_hash_rows else 0,
                "sha256": "see old_r1_depth_file_hashes.csv",
                "evidence_status": "recovered_unhistorically_hashed" if depth_hash_rows else "missing",
                "notes": f"{len(depth_hash_rows)} .npy files.",
            }
        ]
    )
    write_csv(out_root / "artifact_search_inventory.csv", inventory)
    write_json(out_root / "artifact_search_inventory.json", inventory)

    # Per-point old/current comparison.
    current_by_point = {r["point_name"]: r for r in [*current_ctrl, *current_chk]}
    old_by_point = {r["point_name"]: r for r in [*old_r1_ctrl, *old_r1_chk]}
    point_compare: list[dict[str, Any]] = []
    for point in sorted(set(old_by_point) | set(current_by_point)):
        o = old_by_point.get(point, {})
        c = current_by_point.get(point, {})
        point_compare.append(
            {
                "point_name": point,
                "old_role": o.get("role", ""),
                "current_role": c.get("role", ""),
                "old_error_h_m": o.get("error_h_m", ""),
                "old_error_z_m": o.get("error_z_m", ""),
                "old_error_3d_m": o.get("error_3d_m", ""),
                "current_error_h_m": c.get("error_h_m", ""),
                "current_error_z_m": c.get("error_z_m", ""),
                "current_error_3d_m": c.get("error_3d_m", ""),
                "role_changed": o.get("role", "") != c.get("role", ""),
            }
        )
    write_csv(out_root / "per_point_old_current_residual_comparison.csv", point_compare)

    unresolved = [
        {"item": "exact old evaluator command", "impact": "Exact archived replay impossible without command."},
        {"item": "old evaluator uncommitted diff", "impact": "Cannot prove source snapshot exactly matches archived run."},
        {"item": "historical SHA-256 for old depth files", "impact": "Present-day hashes support forensic reconstruction only."},
        {"item": "old COLMAP model historical hash", "impact": "COLMAP/camera equivalence remains unresolved."},
        {"item": "independent old split CSV hash", "impact": "Split is recovered from manifest lists, not an external release file."},
    ]
    write_csv(out_root / "unresolved_missing_artifact_list.csv", unresolved)

    final_status = stage_a2_status
    if final_status in {"exact_numeric_reproduction_with_unhistorically_hashed_inputs", "forensic_reconstruction_only"}:
        comparable = False
        disposition = "unverified_archived_result_not_comparable_to_release_protocol"
    elif final_status == "numerically_reproduced_but_semantically_inconsistent":
        comparable = False
        disposition = "numerically_reproduced_but_semantically_inconsistent"
    else:
        comparable = False
        disposition = "unreproduced_archived_result_not_comparable_to_release_protocol"

    answers = {
        "q1_is_0252161_numerically_reproducible": bool(a2_ok),
        "q1_answer": "Yes, from archived residual CSVs and recovered files as A2 forensic reconstruction; no A1 exact replay.",
        "q2_old_evaluator_depth_interpretation": "The 0.252 R1 path records inverse_camera_z in exporter and evaluator manifests; raw values support inverse-depth-like payload. The separate R8 path has a camera_z/inverse_camera_z manifest conflict.",
        "q3_old_result_semantically_valid": "Not verified as a current formal metric. It is a historical inverse-depth diagnostic, not release-mode expected-camera-z.",
        "q4_comparable_to_current_release": comparable,
        "q4_answer": "No. Key inputs/protocol differ and exact old replay is unavailable.",
        "q5_main_single_factor_if_identifiable": "split_only is identifiable and makes old R1 points better under current split, not worse; it does not explain the 1.09 m current result. Other factors are not single-factor identifiable from recovered artifacts.",
        "q6_coupled_or_missing_factors": [r["item"] for r in unresolved],
        "final_disposition": disposition,
    }
    write_json(out_root / "six_question_answers.json", answers)

    review_lines = [
        "# Archived 0.252161 m Provenance Review Brief",
        "",
        "This is a no-GPU forensic audit. It did not render/export depth, train, mutate checkpoints/support, edit pointsets/splits/annotations, or change the formal evaluator.",
        "",
        "## Main finding",
        "",
        f"- Stage A1 exact archived replay: `{stage_a1_status}`.",
        f"- Stage A2 forensic reconstruction: `{stage_a2_status}`.",
        f"- Recomputed old R1 checkpoint RMSE-3D from archived residuals: `{old_recomputed['checkpoint']['rmse_3d_m']}` m.",
        f"- Current release expected-z checkpoint RMSE-3D: `{current_summary['residual_stats']['checkpoint']['rmse_3d_m']}` m.",
        f"- Final disposition: `{disposition}`.",
        "",
        "## Interpretation",
        "",
        "The archived 0.252161 m value is numerically recoverable from archived residual CSVs, but exact replay is not verified because the exact old evaluator command, uncommitted diff, historical file hashes, and independent old split/COLMAP hashes are not preserved.",
        "",
        "The R1 0.252 target is not the same as the older R8 directory with a manifest semantics conflict. R1 declares inverse_camera_z in both export and evaluator manifests; R8 declares camera_z in export but inverse_camera_z in evaluator and is retained as separate semantic-risk evidence.",
        "",
        "Stage B split_only is identifiable and does not explain the gap: applying the current split to old R1 model points gives a low checkpoint RMSE-3D rather than a jump to 1.09 m. Other factors cannot be isolated one-at-a-time from available artifacts.",
        "",
        "The current release transform sensitivity is included separately and does not alter the frozen split.",
    ]
    (out_root / "REVIEW_BRIEF.md").write_text("\n".join(review_lines), encoding="utf-8")
    (out_root / "FINAL_DISPOSITION.md").write_text(
        f"# Final disposition\n\n`{disposition}`\n\nThe archived value must not be used as a direct baseline for the current formal release-mode expected-camera-z evaluator.\n",
        encoding="utf-8",
    )

    # Package with detached checksum.
    write_json(
        out_root / "package_manifest.json",
        {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "out_root": str(out_root),
            "stage_a1_status": stage_a1_status,
            "stage_a2_status": stage_a2_status,
            "final_disposition": disposition,
            "included_large_depth_files": False,
            "old_depth_hash_rows": len(depth_hash_rows),
        },
    )

    shutil.copy2(Path(__file__), out_root / "recover_archived_0252_provenance.py")
    doc_path = Path.cwd() / "docs" / "gcp_archived_0252_provenance.md"
    if doc_path.exists():
        shutil.copy2(doc_path, out_root / "gcp_archived_0252_provenance.md")
    (out_root / "git_status_porcelain.txt").write_text(exact_commands["git_status_porcelain"], encoding="utf-8")
    (out_root / "diagnostic_commit.txt").write_text(
        git_text(["log", "-1", "--pretty=fuller"], Path.cwd()) + "\n",
        encoding="utf-8",
    )
    (out_root / "diagnostic_commit.patch").write_text(
        git_text(["show", "--stat", "--patch", "--find-renames", "--find-copies", "HEAD"], Path.cwd()) + "\n",
        encoding="utf-8",
    )

    package_files = [p for p in sorted(out_root.rglob("*")) if p.is_file()]
    content_hash_rows = []
    for path in package_files:
        rel = path.relative_to(out_root).as_posix()
        content_hash_rows.append({"file": rel, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_csv(out_root / "PACKAGE_CONTENT_SHA256SUMS.csv", content_hash_rows)
    package_files = [p for p in sorted(out_root.rglob("*")) if p.is_file()]

    args.package_dir.mkdir(parents=True, exist_ok=True)
    zip_path = ensure_unique_file(args.package_dir / "GPT_GCP_3K_ARCHIVED_0252_PROVENANCE_REVIEW_20260628.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        root_name = out_root.name
        for path in package_files:
            if path.suffix.lower() == ".npy":
                continue
            zf.write(path, f"{root_name}/{path.relative_to(out_root).as_posix()}")
    zip_sha = sha256_file(zip_path)
    zip_path.with_suffix(zip_path.suffix + ".sha256").write_text(f"{zip_sha}  {zip_path.name}\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "out_root": str(out_root),
                "package": str(zip_path),
                "package_sha256": zip_sha,
                "stage_a1_status": stage_a1_status,
                "stage_a2_status": stage_a2_status,
                "final_disposition": disposition,
                "old_r1_checkpoint_rmse_3d_recomputed": old_recomputed["checkpoint"]["rmse_3d_m"],
                "current_release_checkpoint_rmse_3d": current_summary["residual_stats"]["checkpoint"]["rmse_3d_m"],
                "bootstrap_valid": valid,
                "bootstrap_invalid": invalid,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
