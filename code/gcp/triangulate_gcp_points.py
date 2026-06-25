from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
COLMAP_UTILS = REPO_ROOT / "code" / "colmap" / "utils"
sys.path.insert(0, str(COLMAP_UTILS))

from read_write_model import qvec2rotmat, read_model  # noqa: E402


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


def camera_params(camera: Any) -> tuple[float, float, float, float, List[float]]:
    model = camera.model.upper()
    params = np.asarray(camera.params, dtype=np.float64)
    if model == "SIMPLE_PINHOLE":
        f, cx, cy = params[:3]
        return float(f), float(f), float(cx), float(cy), []
    if model == "PINHOLE":
        fx, fy, cx, cy = params[:4]
        return float(fx), float(fy), float(cx), float(cy), []
    if model == "SIMPLE_RADIAL":
        f, cx, cy, k = params[:4]
        return float(f), float(f), float(cx), float(cy), [float(k)]
    if model == "RADIAL":
        f, cx, cy, k1, k2 = params[:5]
        return float(f), float(f), float(cx), float(cy), [float(k1), float(k2)]
    if model == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = params[:8]
        return float(fx), float(fy), float(cx), float(cy), [float(k1), float(k2), float(p1), float(p2)]
    raise NotImplementedError(f"Unsupported camera model for GCP triangulation: {camera.model}")


def distort_normalized(x: float, y: float, distortion: Sequence[float]) -> tuple[float, float]:
    if not distortion:
        return x, y
    r2 = x * x + y * y
    if len(distortion) == 1:
        radial = 1.0 + distortion[0] * r2
        return x * radial, y * radial
    if len(distortion) == 2:
        radial = 1.0 + distortion[0] * r2 + distortion[1] * r2 * r2
        return x * radial, y * radial
    k1, k2, p1, p2 = distortion[:4]
    radial = 1.0 + k1 * r2 + k2 * r2 * r2
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return xd, yd


def undistort_normalized(xd: float, yd: float, distortion: Sequence[float], iterations: int = 8) -> tuple[float, float]:
    if not distortion:
        return xd, yd
    x, y = float(xd), float(yd)
    for _ in range(iterations):
        x_proj, y_proj = distort_normalized(x, y, distortion)
        x += xd - x_proj
        y += yd - y_proj
    return x, y


def pixel_to_normalized(camera: Any, u: float, v: float) -> tuple[float, float]:
    fx, fy, cx, cy, distortion = camera_params(camera)
    xd = (float(u) - cx) / fx
    yd = (float(v) - cy) / fy
    return undistort_normalized(xd, yd, distortion)


def project_point(camera: Any, image: Any, xyz_world: np.ndarray) -> tuple[float, float] | None:
    rotation = qvec2rotmat(image.qvec)
    xyz_cam = rotation @ xyz_world + image.tvec
    if xyz_cam[2] <= 1e-9:
        return None
    x = float(xyz_cam[0] / xyz_cam[2])
    y = float(xyz_cam[1] / xyz_cam[2])
    fx, fy, cx, cy, distortion = camera_params(camera)
    xd, yd = distort_normalized(x, y, distortion)
    return fx * xd + cx, fy * yd + cy


def triangulate_point(observations: Sequence[Dict[str, Any]], cameras: Dict[int, Any], images_by_name: Dict[str, Any]) -> np.ndarray:
    rows: List[np.ndarray] = []
    for obs in observations:
        image = images_by_name[obs["image_name"]]
        camera = cameras[image.camera_id]
        x, y = pixel_to_normalized(camera, float(obs["u_px"]), float(obs["v_px"]))
        rotation = qvec2rotmat(image.qvec)
        projection = np.hstack([rotation, image.tvec.reshape(3, 1)])
        rows.append(x * projection[2, :] - projection[0, :])
        rows.append(y * projection[2, :] - projection[1, :])
    a = np.vstack(rows)
    _, _, vt = np.linalg.svd(a)
    homogeneous = vt[-1, :]
    if abs(float(homogeneous[3])) <= 1e-12:
        raise ValueError("Degenerate triangulation result")
    return homogeneous[:3] / homogeneous[3]


def observation_is_usable(row: Dict[str, str], min_confidence: float) -> bool:
    if "u_px" in row and "v_px" in row:
        u, v = row.get("u_px", ""), row.get("v_px", "")
    else:
        u, v = row.get("manual_x", ""), row.get("manual_y", "")
    if not u or not v:
        return False
    visible = str(row.get("visible", "1")).strip()
    if visible not in {"", "1", "true", "True", "yes", "Y"}:
        return False
    quality = str(row.get("quality", "")).strip()
    if quality in {"not_visible", "reject", "rejected"}:
        return False
    if quality and quality != "good":
        return False
    confidence_text = str(row.get("confidence", "")).strip()
    if confidence_text:
        try:
            if float(confidence_text) < min_confidence:
                return False
        except ValueError:
            pass
    return True


def normalize_observation(row: Dict[str, str]) -> Dict[str, Any]:
    u = row.get("u_px", row.get("manual_x", ""))
    v = row.get("v_px", row.get("manual_y", ""))
    return {
        "scene": row.get("scene", ""),
        "point_name": row["point_name"],
        "image_name": row["image_name"],
        "u_px": float(u),
        "v_px": float(v),
        "quality": row.get("quality", ""),
        "confidence": row.get("confidence", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Triangulate model-space GCP points from manual 2D observations.")
    parser.add_argument("--colmap_model", required=True, help="COLMAP sparse model directory.")
    parser.add_argument("--observations_csv", required=True, help="Manual/evaluation-ready 2D GCP observation CSV.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--scene", default="", help="Optional scene id written to outputs.")
    parser.add_argument("--min_observations", type=int, default=2)
    parser.add_argument("--min_confidence", type=float, default=1.0)
    args = parser.parse_args()

    cameras, images, _points3d = read_model(args.colmap_model)
    images_by_name = {image.name: image for image in images.values()}
    raw_rows = read_csv(Path(args.observations_csv))
    usable_rows = [
        normalize_observation(row)
        for row in raw_rows
        if observation_is_usable(row, min_confidence=float(args.min_confidence))
        and row.get("image_name", "") in images_by_name
    ]
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in usable_rows:
        grouped[row["point_name"]].append(row)

    point_rows: List[Dict[str, Any]] = []
    observation_rows: List[Dict[str, Any]] = []
    for point_name in sorted(grouped):
        observations = sorted(grouped[point_name], key=lambda row: row["image_name"])
        if len(observations) < int(args.min_observations):
            continue
        xyz = triangulate_point(observations, cameras, images_by_name)
        reproj_errors: List[float] = []
        for obs in observations:
            image = images_by_name[obs["image_name"]]
            camera = cameras[image.camera_id]
            projected = project_point(camera, image, xyz)
            if projected is None:
                error_px = math.nan
                reproj_u = math.nan
                reproj_v = math.nan
            else:
                reproj_u, reproj_v = projected
                error_px = math.hypot(reproj_u - obs["u_px"], reproj_v - obs["v_px"])
                reproj_errors.append(error_px)
            observation_rows.append(
                {
                    "scene": obs.get("scene") or args.scene,
                    "point_name": point_name,
                    "image_name": obs["image_name"],
                    "u_px": obs["u_px"],
                    "v_px": obs["v_px"],
                    "reprojected_u_px": reproj_u,
                    "reprojected_v_px": reproj_v,
                    "reprojection_error_px": error_px,
                    "quality": obs.get("quality", ""),
                    "confidence": obs.get("confidence", ""),
                }
            )
        point_rows.append(
            {
                "scene": args.scene or observations[0].get("scene", ""),
                "point_name": point_name,
                "model_x": xyz[0],
                "model_y": xyz[1],
                "model_z": xyz[2],
                "observation_count": len(observations),
                "mean_reprojection_error_px": float(np.mean(reproj_errors)) if reproj_errors else "",
                "max_reprojection_error_px": float(np.max(reproj_errors)) if reproj_errors else "",
                "used_image_names": ";".join(obs["image_name"] for obs in observations),
            }
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        out_dir / "triangulated_gcp_model_points.csv",
        point_rows,
        [
            "scene",
            "point_name",
            "model_x",
            "model_y",
            "model_z",
            "observation_count",
            "mean_reprojection_error_px",
            "max_reprojection_error_px",
            "used_image_names",
        ],
    )
    write_csv(
        out_dir / "triangulated_gcp_observation_residuals.csv",
        observation_rows,
        [
            "scene",
            "point_name",
            "image_name",
            "u_px",
            "v_px",
            "reprojected_u_px",
            "reprojected_v_px",
            "reprojection_error_px",
            "quality",
            "confidence",
        ],
    )
    mean_errors = [
        float(row["mean_reprojection_error_px"])
        for row in point_rows
        if row["mean_reprojection_error_px"] != ""
    ]
    manifest = {
        "schema": "m3m_gcp_triangulated_model_points_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "colmap_model": str(Path(args.colmap_model)),
        "observations_csv": str(Path(args.observations_csv)),
        "scene": args.scene,
        "registered_images_in_model": len(images),
        "raw_observation_rows": len(raw_rows),
        "usable_observation_rows": len(usable_rows),
        "triangulated_point_count": len(point_rows),
        "min_observations": int(args.min_observations),
        "min_confidence": float(args.min_confidence),
        "mean_point_reprojection_error_px": float(np.mean(mean_errors)) if mean_errors else None,
        "outputs": {
            "model_points": str(out_dir / "triangulated_gcp_model_points.csv"),
            "observation_residuals": str(out_dir / "triangulated_gcp_observation_residuals.csv"),
        },
        "notes": [
            "Triangulates model-space GCP coordinates from manual 2D observations and COLMAP camera poses.",
            "Does not fit GCP georeferencing transform; use fit_gcp_sim3.py for control/checkpoint residuals.",
            "Camera distortion is handled for SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, and OPENCV models.",
        ],
    }
    (out_dir / "triangulated_gcp_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
