from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


DEFAULT_TARGET_FIELDS = (
    "cgcs2000_gk_cm108_e_m",
    "cgcs2000_gk_cm108_n_m",
    "cgcs2000_normal_height_m",
)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_name_set(text: str | None) -> set[str]:
    if not text:
        return set()
    return {item.strip() for item in text.split(",") if item.strip()}


def load_role_csv(path: Path | None) -> Dict[str, str]:
    if path is None:
        return {}
    roles: Dict[str, str] = {}
    for row in read_csv(path):
        name = row.get("point_name", "").strip()
        role = row.get("role", "").strip().lower()
        if name and role:
            roles[name] = role
    return roles


def load_model_points(path: Path) -> Dict[str, np.ndarray]:
    rows = read_csv(path)
    points: Dict[str, np.ndarray] = {}
    for row in rows:
        name = row.get("point_name", "").strip()
        if not name:
            continue
        if all(k in row for k in ("model_x", "model_y", "model_z")):
            xyz = [row["model_x"], row["model_y"], row["model_z"]]
        elif all(k in row for k in ("x", "y", "z")):
            xyz = [row["x"], row["y"], row["z"]]
        else:
            raise ValueError(
                f"Model-point CSV must contain model_x/model_y/model_z or x/y/z columns: {path}"
            )
        points[name] = np.asarray([float(v) for v in xyz], dtype=np.float64)
    return points


def load_target_points(
    path: Path,
    target_fields: Sequence[str] = DEFAULT_TARGET_FIELDS,
) -> Dict[str, np.ndarray]:
    rows = read_csv(path)
    points: Dict[str, np.ndarray] = {}
    for row in rows:
        name = row.get("point_name", "").strip()
        if not name:
            continue
        missing = [field for field in target_fields if field not in row or row[field] == ""]
        if missing:
            raise ValueError(f"Missing target coordinate fields for {name}: {missing}")
        points[name] = np.asarray([float(row[field]) for field in target_fields], dtype=np.float64)
    return points


def fit_similarity_umeyama(
    source_xyz: np.ndarray,
    target_xyz: np.ndarray,
    estimate_scale: bool = True,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Fit target ~= scale * rotation @ source + translation.

    This is the Umeyama/Procrustes closed-form least-squares solution. It uses
    only a global similarity transform and intentionally does not support
    non-rigid warping, because checkpoints must remain independent evidence.
    """

    if source_xyz.shape != target_xyz.shape:
        raise ValueError("source and target arrays must have the same shape")
    if source_xyz.ndim != 2 or source_xyz.shape[1] != 3:
        raise ValueError("source and target arrays must be Nx3")
    if source_xyz.shape[0] < 3:
        raise ValueError("At least three non-collinear control points are required")

    n = source_xyz.shape[0]
    mu_x = source_xyz.mean(axis=0)
    mu_y = target_xyz.mean(axis=0)
    x = source_xyz - mu_x
    y = target_xyz - mu_y
    var_x = float(np.sum(x * x) / n)
    if var_x <= 0:
        raise ValueError("Degenerate source control points")

    covariance = (y.T @ x) / n
    u, singular_values, vt = np.linalg.svd(covariance)
    det = np.linalg.det(u @ vt)
    sign = np.ones(3, dtype=np.float64)
    if det < 0:
        sign[-1] = -1.0
    s_mat = np.diag(sign)
    rotation = u @ s_mat @ vt
    if np.linalg.det(rotation) < 0.0:
        raise ValueError("Estimated rotation is a reflection; check control point geometry")
    scale = float(np.sum(singular_values * sign) / var_x) if estimate_scale else 1.0
    translation = mu_y - scale * (rotation @ mu_x)
    return scale, rotation, translation


def apply_similarity(points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return (scale * (rotation @ points.T)).T + translation.reshape(1, 3)


def residual_stats(residuals: np.ndarray) -> Dict[str, float | int | None]:
    if residuals.size == 0:
        return {
            "count": 0,
            "rmse_h_m": None,
            "rmse_z_m": None,
            "rmse_3d_m": None,
            "median_3d_m": None,
            "p90_3d_m": None,
            "p95_3d_m": None,
            "max_3d_m": None,
        }
    h = np.linalg.norm(residuals[:, :2], axis=1)
    z = np.abs(residuals[:, 2])
    d = np.linalg.norm(residuals, axis=1)
    return {
        "count": int(residuals.shape[0]),
        "rmse_h_m": float(math.sqrt(np.mean(h * h))),
        "rmse_z_m": float(math.sqrt(np.mean(z * z))),
        "rmse_3d_m": float(math.sqrt(np.mean(d * d))),
        "median_3d_m": float(np.median(d)),
        "p90_3d_m": float(np.percentile(d, 90)),
        "p95_3d_m": float(np.percentile(d, 95)),
        "max_3d_m": float(np.max(d)),
    }


def make_markdown(summary: Dict[str, Any], residual_rows: List[Dict[str, Any]]) -> str:
    lines = [
        "# GCP Sim(3) Registration Evaluation",
        "",
        "This report fits a single global similarity transform from model-space GCP points",
        "to surveyed GCP coordinates. It does not apply local stretching or non-rigid",
        "warping.",
        "",
        "## Transform",
        "",
        f"- Scale: `{summary['transform']['scale']}`",
        f"- Control points: `{summary['control_count']}`",
        f"- Checkpoints: `{summary['checkpoint_count']}`",
        f"- Target fields: `{', '.join(summary['target_fields'])}`",
        "",
        "## Residual summary",
        "",
        "| Role | Count | RMSE-H (m) | RMSE-Z (m) | RMSE-3D (m) | Median-3D (m) | P95-3D (m) | Max-3D (m) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for role in ["control", "checkpoint", "all"]:
        stats = summary["residual_stats"][role]

        def fmt(value: Any) -> str:
            return "" if value is None else f"{float(value):.4f}"

        lines.append(
            f"| {role} | {stats['count']} | {fmt(stats['rmse_h_m'])} | "
            f"{fmt(stats['rmse_z_m'])} | {fmt(stats['rmse_3d_m'])} | "
            f"{fmt(stats['median_3d_m'])} | {fmt(stats['p95_3d_m'])} | {fmt(stats['max_3d_m'])} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Control residuals describe transform fit quality.",
            "- Checkpoint residuals are the independent accuracy evidence when checkpoints were not used in fitting.",
            "- If all points are controls, the report is a registration fit report rather than an independent checkpoint evaluation.",
            "",
            "## Per-point residuals",
            "",
            "| Point | Role | eH (m) | eZ (m) | e3D (m) |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in residual_rows:
        lines.append(
            f"| {row['point_name']} | {row['role']} | {float(row['error_h_m']):.4f} | "
            f"{float(row['error_z_m']):.4f} | {float(row['error_3d_m']):.4f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a global Sim(3) transform from model GCPs to surveyed GCPs.")
    parser.add_argument("--model_points_csv", required=True, help="CSV with point_name and model_x/model_y/model_z.")
    parser.add_argument("--gcp_csv", required=True, help="Canonical surveyed GCP CSV.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--target_fields",
        default=",".join(DEFAULT_TARGET_FIELDS),
        help="Comma-separated target x/y/z fields in the GCP CSV.",
    )
    parser.add_argument("--control_points", default="", help="Comma-separated point names used for fitting.")
    parser.add_argument("--checkpoint_points", default="", help="Optional comma-separated held-out checkpoint names.")
    parser.add_argument("--role_csv", default="", help="Optional CSV with point_name,role where role is control/checkpoint.")
    parser.add_argument("--no_scale", action="store_true", help="Fit rigid transform only, with scale fixed to 1.")
    args = parser.parse_args()

    model_points = load_model_points(Path(args.model_points_csv))
    target_fields = [field.strip() for field in args.target_fields.split(",") if field.strip()]
    target_points = load_target_points(Path(args.gcp_csv), target_fields=target_fields)
    common_names = sorted(set(model_points) & set(target_points))
    if len(common_names) < 3:
        raise ValueError("At least three common GCPs are required")

    control_points = parse_name_set(args.control_points)
    checkpoint_points = parse_name_set(args.checkpoint_points)
    role_csv = Path(args.role_csv) if args.role_csv else None
    roles = load_role_csv(role_csv)
    for name, role in roles.items():
        if role in {"control", "fit", "registration"}:
            control_points.add(name)
        elif role in {"checkpoint", "check", "validation", "heldout", "held-out"}:
            checkpoint_points.add(name)

    if not control_points:
        raise ValueError("Specify control points with --control_points or --role_csv. Do not fit on all points implicitly.")
    control_names = sorted(control_points & set(common_names))
    if len(control_names) < 3:
        raise ValueError(f"At least three common control points are required, got {len(control_names)}")

    if checkpoint_points:
        checkpoint_names = sorted(checkpoint_points & set(common_names))
    else:
        checkpoint_names = [name for name in common_names if name not in set(control_names)]

    source_control = np.vstack([model_points[name] for name in control_names])
    target_control = np.vstack([target_points[name] for name in control_names])
    scale, rotation, translation = fit_similarity_umeyama(
        source_control,
        target_control,
        estimate_scale=not args.no_scale,
    )

    residual_rows: List[Dict[str, Any]] = []
    residuals_by_role: Dict[str, List[np.ndarray]] = {"control": [], "checkpoint": [], "all": []}
    for name in common_names:
        model_xyz = model_points[name]
        target_xyz = target_points[name]
        pred_xyz = apply_similarity(model_xyz.reshape(1, 3), scale, rotation, translation)[0]
        residual = pred_xyz - target_xyz
        if name in set(control_names):
            role = "control"
        elif name in set(checkpoint_names):
            role = "checkpoint"
        else:
            role = "unused"
        if role in residuals_by_role:
            residuals_by_role[role].append(residual)
        residuals_by_role["all"].append(residual)
        h = float(np.linalg.norm(residual[:2]))
        z = float(abs(residual[2]))
        d = float(np.linalg.norm(residual))
        residual_rows.append(
            {
                "point_name": name,
                "role": role,
                "model_x": model_xyz[0],
                "model_y": model_xyz[1],
                "model_z": model_xyz[2],
                "target_x": target_xyz[0],
                "target_y": target_xyz[1],
                "target_z": target_xyz[2],
                "predicted_x": pred_xyz[0],
                "predicted_y": pred_xyz[1],
                "predicted_z": pred_xyz[2],
                "residual_x_m": residual[0],
                "residual_y_m": residual[1],
                "residual_z_m": residual[2],
                "error_h_m": h,
                "error_z_m": z,
                "error_3d_m": d,
            }
        )

    residual_arrays = {
        role: np.vstack(values) if values else np.empty((0, 3), dtype=np.float64)
        for role, values in residuals_by_role.items()
    }
    summary = {
        "schema": "m3m_gcp_sim3_registration_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_points_csv": str(Path(args.model_points_csv)),
        "gcp_csv": str(Path(args.gcp_csv)),
        "target_fields": target_fields,
        "common_point_count": len(common_names),
        "control_points": control_names,
        "checkpoint_points": checkpoint_names,
        "control_count": len(control_names),
        "checkpoint_count": len(checkpoint_names),
        "estimate_scale": not args.no_scale,
        "transform": {
            "scale": scale,
            "rotation": rotation.tolist(),
            "translation": translation.tolist(),
            "definition": "target_xyz = scale * rotation @ model_xyz + translation",
        },
        "residual_stats": {
            role: residual_stats(values) for role, values in residual_arrays.items()
        },
        "notes": [
            "Uses a single global Sim(3)/Helmert-style transform.",
            "Does not perform local stretching or non-rigid warping.",
            "Checkpoint residuals are independent only if those points were not used in fitting.",
        ],
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        out_dir / "gcp_sim3_residuals.csv",
        residual_rows,
        [
            "point_name",
            "role",
            "model_x",
            "model_y",
            "model_z",
            "target_x",
            "target_y",
            "target_z",
            "predicted_x",
            "predicted_y",
            "predicted_z",
            "residual_x_m",
            "residual_y_m",
            "residual_z_m",
            "error_h_m",
            "error_z_m",
            "error_3d_m",
        ],
    )
    (out_dir / "gcp_sim3_transform.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "gcp_sim3_report.md").write_text(make_markdown(summary, residual_rows), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
