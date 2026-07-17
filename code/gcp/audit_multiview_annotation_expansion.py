#!/usr/bin/env python
"""Six-scene multi-view annotation expansion and control-heavy design audit.

This is a read-only audit. It does not modify any release, packet, split,
survey coordinate, or evaluator artifact. The generated outputs are intended
for protocol/design review before any v1.3.0 data work.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import shutil
import subprocess
import sys
import time
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from triangulate_gcp_points import simple_radial_principal_branch_is_valid

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover - surfaced in runtime preflight.
    Image = None
    ImageDraw = None
    ImageFont = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plots become an explicit blocker.
    plt = None


SCENES = [
    "gcp_3000_20260602",
    "gcp_5000_20260602",
    "gcp_10000_20260610",
    "gcp_20000_20260602",
    "gcp_50000_20260610",
    "gcp_100000_20260610",
]

SCENE_LABELS = {
    "gcp_3000_20260602": "3K",
    "gcp_5000_20260602": "5K",
    "gcp_10000_20260610": "10K",
    "gcp_20000_20260602": "20K",
    "gcp_50000_20260610": "50K",
    "gcp_100000_20260610": "100K",
}

LOW_VIEW_FOCUS = {
    ("gcp_5000_20260602", "G11"),
    ("gcp_5000_20260602", "G13"),
    ("gcp_5000_20260602", "G18"),
    ("gcp_5000_20260602", "NC94"),
    ("gcp_20000_20260602", "wy3_1"),
}

FORMAL_MIN_VIEWS = 4
CONTROL_PREFERRED_MIN_VIEWS = 6
PREFERRED_VIEW_TARGET = 8
SEARCH_MARGIN_PX = 1200.0
TRIANGULATED_MAX_PER_POINT = 24
COARSE_MAX_PER_POINT = 24
CONTACT_MAX_PER_POINT = 36


@dataclass(frozen=True)
class Paths:
    repo: Path
    dataset: Path
    release: Path
    candidates: Path
    residual_diag: Path
    output_root: Path
    review_root: Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, obj: Any) -> None:
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def run_cmd(args: list[str], cwd: Path) -> dict[str, Any]:
    proc = subprocess.run(args, cwd=str(cwd), text=True, capture_output=True)
    return {
        "command": " ".join(args),
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def qvec_to_rotmat(qvec: Iterable[float]) -> np.ndarray:
    qw, qx, qy, qz = [float(x) for x in qvec]
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n == 0:
        raise ValueError("zero quaternion")
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array(
        [
            [
                1 - 2 * qy * qy - 2 * qz * qz,
                2 * qx * qy - 2 * qz * qw,
                2 * qx * qz + 2 * qy * qw,
            ],
            [
                2 * qx * qy + 2 * qz * qw,
                1 - 2 * qx * qx - 2 * qz * qz,
                2 * qy * qz - 2 * qx * qw,
            ],
            [
                2 * qx * qz - 2 * qy * qw,
                2 * qy * qz + 2 * qx * qw,
                1 - 2 * qx * qx - 2 * qy * qy,
            ],
        ],
        dtype=np.float64,
    )


def camera_center_from_pose(qvec: Iterable[float], tvec: Iterable[float]) -> np.ndarray:
    r = qvec_to_rotmat(qvec)
    t = np.array([float(x) for x in tvec], dtype=np.float64)
    return -r.T @ t


def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 0 or not math.isfinite(n):
        return v * np.nan
    return v / n


def triangulate_rays(centers: list[np.ndarray], dirs: list[np.ndarray]) -> tuple[np.ndarray, float]:
    """Least-squares point closest to a set of 3D rays."""
    a = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    eye = np.eye(3)
    for c, d in zip(centers, dirs):
        d = normalize(d)
        m = eye - np.outer(d, d)
        a += m
        b += m @ c
    x = np.linalg.solve(a, b)
    distances = []
    for c, d in zip(centers, dirs):
        d = normalize(d)
        distances.append(float(np.linalg.norm(np.cross(x - c, d))))
    return x, float(np.median(distances)) if distances else float("nan")


def project_simple_radial(
    xyz_world: np.ndarray,
    qvec: Iterable[float],
    tvec: Iterable[float],
    params: Iterable[float],
) -> tuple[float, float, float]:
    f, cx, cy, k = [float(x) for x in params]
    r = qvec_to_rotmat(qvec)
    t = np.array([float(x) for x in tvec], dtype=np.float64)
    xyz_cam = r @ xyz_world + t
    z = float(xyz_cam[2])
    if z <= 0:
        return float("nan"), float("nan"), z
    x = float(xyz_cam[0] / z)
    y = float(xyz_cam[1] / z)
    r2 = x * x + y * y
    if not simple_radial_principal_branch_is_valid(x, y, k):
        return float("nan"), float("nan"), z
    scale = 1.0 + k * r2
    return f * x * scale + cx, f * y * scale + cy, z


def point_to_ray_world(row: pd.Series, image_record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    ray_cam = np.array(
        [float(row["normalized_x"]), float(row["normalized_y"]), 1.0],
        dtype=np.float64,
    )
    ray_cam = normalize(ray_cam)
    r = qvec_to_rotmat(image_record["qvec"])
    c = camera_center_from_pose(image_record["qvec"], image_record["tvec"])
    ray_world = normalize(r.T @ ray_cam)
    return c, ray_world


def euclidean_xy(dx: float, dy: float) -> float:
    return float(math.sqrt(dx * dx + dy * dy))


def percentile(values: Iterable[float], p: float) -> float:
    vals = [float(v) for v in values if pd.notna(v) and math.isfinite(float(v))]
    if not vals:
        return float("nan")
    return float(np.percentile(vals, p))


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def azimuth_deg(dx: float, dy: float) -> float:
    return (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0


def max_azimuth_gap(angles: list[float]) -> float:
    vals = sorted(float(a) % 360.0 for a in angles if math.isfinite(float(a)))
    if len(vals) < 2:
        return 360.0
    gaps = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
    gaps.append(vals[0] + 360.0 - vals[-1])
    return float(max(gaps))


def unique_direction_bins(angles: list[float], bin_degrees: int = 45) -> int:
    bins = {int((float(a) % 360.0) // bin_degrees) for a in angles if math.isfinite(float(a))}
    return len(bins)


def convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    pts = sorted(set((float(x), float(y)) for x, y in points if math.isfinite(x) and math.isfinite(y)))
    if len(pts) <= 1:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def load_release_annotations(release_dir: Path) -> pd.DataFrame:
    frames = []
    for scene in SCENES:
        p = release_dir / f"{scene}_gcp_annotations_pixel_domain_v1_2_2.csv"
        df = pd.read_csv(p, dtype={"point_name": str, "raw_image_name": str})
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_candidate_seed(candidates_root: Path) -> pd.DataFrame:
    frames = []
    for scene in SCENES:
        p = candidates_root / scene / "gcp_projection_candidates.csv"
        if p.exists():
            df = pd.read_csv(p, dtype={"point_name": str, "image_name": str})
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_image_metadata(candidates_root: Path) -> pd.DataFrame:
    frames = []
    for scene in SCENES:
        p = candidates_root / scene / "image_metadata.csv"
        if p.exists():
            df = pd.read_csv(p, dtype={"image_name": str})
            df["scene"] = scene
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_camera_provenance(release_dir: Path) -> dict[str, Any]:
    return read_json(release_dir / "camera_provenance_manifest_v1_2_2.json")


def build_camera_indices(camera_manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    scene_idx: dict[str, dict[str, Any]] = {}
    for scene, scene_data in camera_manifest["scenes"].items():
        source = scene_data["source_model"]
        target = scene_data["target_model"]
        source_cameras = {int(c["camera_id"]): c for c in source["cameras"]}
        source_images = {img["image_name"]: img for img in source["images"]}
        target_cameras = {int(c["camera_id"]): c for c in target["cameras"]}
        target_images = {img["image_name"]: img for img in target["images"]}
        scene_idx[scene] = {
            "source_cameras": source_cameras,
            "source_images": source_images,
            "target_cameras": target_cameras,
            "target_images": target_images,
        }
    return scene_idx


def build_current_annotation_candidates(ann: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "scene",
        "point_name",
        "raw_image_name",
        "raw_manual_x",
        "raw_manual_y",
        "source_image_width",
        "source_image_height",
        "normalized_x",
        "normalized_y",
        "normalized_unit_ray_x",
        "normalized_unit_ray_y",
        "normalized_unit_ray_z",
    ]
    df = ann[cols].copy()
    df = df.rename(
        columns={
            "raw_image_name": "image_name",
            "raw_manual_x": "candidate_x",
            "raw_manual_y": "candidate_y",
            "source_image_width": "image_width",
            "source_image_height": "image_height",
        }
    )
    df["candidate_source"] = "current_annotation"
    df["already_annotated"] = True
    df["inside_image"] = True
    df["edge_margin_px"] = df.apply(
        lambda r: min(
            float(r["candidate_x"]),
            float(r["candidate_y"]),
            float(r["image_width"]) - float(r["candidate_x"]),
            float(r["image_height"]) - float(r["candidate_y"]),
        ),
        axis=1,
    )
    df["candidate_rank_for_point"] = 0
    df["projection_uncertainty_px"] = 0.0
    df["visibility_classification"] = "already_annotated_usable"
    df["recommended_action"] = "keep_existing"
    df["reject_reason"] = ""
    return df


def build_coarse_candidates(cand: pd.DataFrame, ann_keys: set[tuple[str, str, str]]) -> pd.DataFrame:
    if cand.empty:
        return cand
    df = cand.copy()
    df = df.rename(
        columns={
            "image_name": "image_name",
            "pixel_x": "candidate_x",
            "pixel_y": "candidate_y",
            "rank_for_gcp": "candidate_rank_for_point",
        }
    )
    df["candidate_source"] = "coarse_exif_gimbal_seed"
    df["already_annotated"] = [
        (str(r.scene), str(r.point_name), str(r.image_name)) in ann_keys for r in df.itertuples()
    ]
    df["projection_uncertainty_px"] = 800.0
    df["visibility_classification"] = "requires_visual_review"
    df["reject_reason"] = np.where(df["inside_image"], "", "outside_fov_by_coarse_seed")
    return df


def build_triangulation_candidates(
    ann: pd.DataFrame,
    camera_idx: dict[str, dict[str, Any]],
    image_meta: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    loo_rows: list[dict[str, Any]] = []

    meta_keys = set(zip(image_meta.get("scene", []), image_meta.get("image_name", [])))

    for (scene, point_name), group in ann.groupby(["scene", "point_name"], sort=True):
        scene_cam = camera_idx.get(scene)
        if not scene_cam or len(group) < 2:
            summaries.append(
                {
                    "scene": scene,
                    "point_name": point_name,
                    "triangulation_status": "not_enough_annotation_views",
                    "annotation_views": int(len(group)),
                    "median_ray_distance_model_units": np.nan,
                    "median_annotation_reprojection_error_px": np.nan,
                    "p95_annotation_reprojection_error_px": np.nan,
                    "triangulated_candidate_images": 0,
                }
            )
            continue
        source_images = scene_cam["source_images"]
        source_cameras = scene_cam["source_cameras"]
        centers: list[np.ndarray] = []
        dirs: list[np.ndarray] = []
        usable_rows = []
        for _, row in group.iterrows():
            img = source_images.get(str(row["raw_image_name"]))
            if img is None:
                continue
            c, d = point_to_ray_world(row, img)
            centers.append(c)
            dirs.append(d)
            usable_rows.append(row)
        if len(centers) < 2:
            summaries.append(
                {
                    "scene": scene,
                    "point_name": point_name,
                    "triangulation_status": "missing_pose_for_annotation_views",
                    "annotation_views": int(len(group)),
                    "median_ray_distance_model_units": np.nan,
                    "median_annotation_reprojection_error_px": np.nan,
                    "p95_annotation_reprojection_error_px": np.nan,
                    "triangulated_candidate_images": 0,
                }
            )
            continue
        try:
            xyz, median_ray_dist = triangulate_rays(centers, dirs)
        except Exception as exc:
            summaries.append(
                {
                    "scene": scene,
                    "point_name": point_name,
                    "triangulation_status": f"failed:{exc}",
                    "annotation_views": int(len(group)),
                    "median_ray_distance_model_units": np.nan,
                    "median_annotation_reprojection_error_px": np.nan,
                    "p95_annotation_reprojection_error_px": np.nan,
                    "triangulated_candidate_images": 0,
                }
            )
            continue

        reproj_errors = []
        for row in usable_rows:
            img = source_images[str(row["raw_image_name"])]
            cam = source_cameras[int(img["camera_id"])]
            u, v, z = project_simple_radial(xyz, img["qvec"], img["tvec"], cam["params"])
            if math.isfinite(u) and math.isfinite(v) and z > 0:
                reproj_errors.append(
                    euclidean_xy(float(u) - float(row["raw_manual_x"]), float(v) - float(row["raw_manual_y"]))
                )
        uncertainty = max(300.0, min(1500.0, percentile(reproj_errors, 95) * 2.0 + 200.0))
        image_rows = []
        for image_name, img in source_images.items():
            cam = source_cameras[int(img["camera_id"])]
            width = int(cam["width"])
            height = int(cam["height"])
            u, v, z = project_simple_radial(xyz, img["qvec"], img["tvec"], cam["params"])
            if not (math.isfinite(u) and math.isfinite(v) and z > 0):
                continue
            inside = (-SEARCH_MARGIN_PX <= u <= width + SEARCH_MARGIN_PX) and (
                -SEARCH_MARGIN_PX <= v <= height + SEARCH_MARGIN_PX
            )
            if not inside:
                continue
            edge = min(float(u), float(v), width - float(u), height - float(v))
            center_score = euclidean_xy(float(u) - width / 2.0, float(v) - height / 2.0)
            image_rows.append(
                {
                    "scene": scene,
                    "point_name": point_name,
                    "image_name": image_name,
                    "candidate_x": float(u),
                    "candidate_y": float(v),
                    "inside_image": (0 <= u < width and 0 <= v < height),
                    "edge_margin_px": edge,
                    "image_width": width,
                    "image_height": height,
                    "candidate_source": "triangulated_annotation_rays",
                    "already_annotated": image_name in set(group["raw_image_name"].astype(str)),
                    "candidate_rank_for_point": np.nan,
                    "projection_uncertainty_px": uncertainty,
                    "visibility_classification": "requires_visual_review",
                    "recommended_action": "",
                    "reject_reason": "" if edge >= -SEARCH_MARGIN_PX else "outside_search_margin",
                    "triangulated_model_x": float(xyz[0]),
                    "triangulated_model_y": float(xyz[1]),
                    "triangulated_model_z": float(xyz[2]),
                    "triangulation_reprojection_error_p95_px": percentile(reproj_errors, 95),
                    "has_image_metadata": (scene, image_name) in meta_keys,
                    "camera_z_model_units": z,
                    "center_distance_px": center_score,
                }
            )
        image_rows.sort(key=lambda r: (not r["already_annotated"], -float(r["edge_margin_px"]), r["center_distance_px"]))
        selected = []
        annotated_rows = [r for r in image_rows if r["already_annotated"]]
        non_ann_rows = [r for r in image_rows if not r["already_annotated"]]
        selected.extend(annotated_rows)
        selected.extend(non_ann_rows[:TRIANGULATED_MAX_PER_POINT])
        for i, r in enumerate(selected, start=1):
            r["candidate_rank_for_point"] = i if not r["already_annotated"] else 0
            rows.append(r)
        summaries.append(
            {
                "scene": scene,
                "point_name": point_name,
                "triangulation_status": "ok",
                "annotation_views": int(len(group)),
                "median_ray_distance_model_units": median_ray_dist,
                "median_annotation_reprojection_error_px": percentile(reproj_errors, 50),
                "p95_annotation_reprojection_error_px": percentile(reproj_errors, 95),
                "projection_uncertainty_px": uncertainty,
                "triangulated_candidate_images": int(len(image_rows)),
                "triangulated_selected_images": int(len(selected)),
            }
        )

        for hide_idx, hidden in group.iterrows():
            rest = group.drop(index=hide_idx)
            status = "not_testable_less_than_two_remaining_views"
            recalled = False
            hidden_proj_error = np.nan
            if len(rest) >= 2:
                rest_centers = []
                rest_dirs = []
                for _, rr in rest.iterrows():
                    img = source_images.get(str(rr["raw_image_name"]))
                    if img is not None:
                        c, d = point_to_ray_world(rr, img)
                        rest_centers.append(c)
                        rest_dirs.append(d)
                if len(rest_centers) >= 2:
                    try:
                        rest_xyz, _ = triangulate_rays(rest_centers, rest_dirs)
                        hidden_img = source_images.get(str(hidden["raw_image_name"]))
                        if hidden_img is not None:
                            cam = source_cameras[int(hidden_img["camera_id"])]
                            u, v, z = project_simple_radial(
                                rest_xyz,
                                hidden_img["qvec"],
                                hidden_img["tvec"],
                                cam["params"],
                            )
                            hidden_proj_error = euclidean_xy(
                                float(u) - float(hidden["raw_manual_x"]),
                                float(v) - float(hidden["raw_manual_y"]),
                            )
                            width = int(cam["width"])
                            height = int(cam["height"])
                            recalled = (
                                z > 0
                                and -SEARCH_MARGIN_PX <= u <= width + SEARCH_MARGIN_PX
                                and -SEARCH_MARGIN_PX <= v <= height + SEARCH_MARGIN_PX
                            )
                            status = "recalled" if recalled else "missed"
                    except Exception as exc:
                        status = f"failed:{exc}"
            loo_rows.append(
                {
                    "scene": scene,
                    "point_name": point_name,
                    "hidden_image_name": hidden["raw_image_name"],
                    "remaining_views": int(len(rest)),
                    "status": status,
                    "recalled_by_triangulation_with_margin": bool(recalled),
                    "hidden_projection_error_px": hidden_proj_error,
                    "search_margin_px": SEARCH_MARGIN_PX,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(summaries), pd.DataFrame(loo_rows)


def dedupe_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    priority = {
        "current_annotation": 0,
        "triangulated_annotation_rays": 1,
        "coarse_exif_gimbal_seed": 2,
    }
    df = candidates.copy()
    df["_priority"] = df["candidate_source"].map(priority).fillna(99)
    df["_edge_sort"] = pd.to_numeric(df.get("edge_margin_px", np.nan), errors="coerce").fillna(-999999.0)
    df = df.sort_values(
        ["scene", "point_name", "image_name", "_priority", "_edge_sort"],
        ascending=[True, True, True, True, False],
    )
    df = df.drop_duplicates(["scene", "point_name", "image_name"], keep="first")
    return df.drop(columns=["_priority", "_edge_sort"])


def add_image_metadata(candidates: pd.DataFrame, image_meta: pd.DataFrame, points: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    meta_cols = [
        c
        for c in [
            "scene",
            "image_name",
            "image_path",
            "projected_e",
            "projected_n",
            "yaw_deg",
            "pitch_deg",
            "roll_deg",
            "rel_alt_m",
            "ellipsoid_alt_m",
        ]
        if c in image_meta.columns
    ]
    df = candidates.merge(image_meta[meta_cols], on=["scene", "image_name"], how="left")
    pt = points[
        [
            "point_name",
            "cgcs2000_gk_cm108_e_m",
            "cgcs2000_gk_cm108_n_m",
            "cgcs2000_normal_height_m",
            "point_category",
            "quality_evaluation",
        ]
    ].copy()
    df = df.merge(pt, on="point_name", how="left")
    dx = pd.to_numeric(df["projected_e"], errors="coerce") - pd.to_numeric(
        df["cgcs2000_gk_cm108_e_m"], errors="coerce"
    )
    dy = pd.to_numeric(df["projected_n"], errors="coerce") - pd.to_numeric(
        df["cgcs2000_gk_cm108_n_m"], errors="coerce"
    )
    df["camera_ground_distance_m"] = np.sqrt(dx * dx + dy * dy)
    df["camera_azimuth_deg"] = [azimuth_deg(x, y) if pd.notna(x) and pd.notna(y) else np.nan for x, y in zip(dx, dy)]
    df["off_nadir_or_pitch_abs_deg"] = np.abs(pd.to_numeric(df.get("pitch_deg", np.nan), errors="coerce") + 90.0)
    return df


def classify_candidates(df: pd.DataFrame, ann_counts: pd.DataFrame) -> pd.DataFrame:
    count_map = {
        (r.scene, r.point_name): int(r.annotation_side_usable_count) for r in ann_counts.itertuples(index=False)
    }
    out = df.copy()
    actions = []
    visibility = []
    reject = []
    for r in out.itertuples(index=False):
        current_count = count_map.get((r.scene, r.point_name), 0)
        already = bool(getattr(r, "already_annotated", False))
        edge = safe_float(getattr(r, "edge_margin_px", np.nan))
        inside = bool(getattr(r, "inside_image", False))
        source = str(getattr(r, "candidate_source", ""))
        if already:
            actions.append("keep_existing")
            visibility.append("already_annotated_usable")
            reject.append("")
        elif not inside:
            actions.append("reject")
            visibility.append("outside_fov_or_search_margin")
            reject.append("outside-FOV")
        elif edge < 80:
            actions.append("review")
            visibility.append("requires_visual_review_edge")
            reject.append("near-edge")
        elif current_count < FORMAL_MIN_VIEWS:
            actions.append("label")
            visibility.append("requires_visual_review")
            reject.append("")
        elif source == "triangulated_annotation_rays" and edge >= 150:
            actions.append("review")
            visibility.append("requires_visual_review")
            reject.append("")
        else:
            actions.append("review")
            visibility.append("requires_visual_review")
            reject.append("")
    out["recommended_action"] = actions
    out["visibility_classification"] = visibility
    out["reject_reason"] = reject
    return out


def make_annotation_counts(ann: pd.DataFrame, residual_diag: Path) -> pd.DataFrame:
    counts = ann.groupby(["scene", "point_name"]).size().reset_index(name="raw_annotation_count")
    counts["annotation_side_usable_count"] = counts["raw_annotation_count"]
    residual_path = residual_diag / "per_point_residual_diagnostics.csv"
    if residual_path.exists():
        rd = pd.read_csv(residual_path, dtype={"point_name": str})
        rd = rd[
            [
                "scene",
                "point_name",
                "role",
                "valid_observation_count",
                "raw_observation_count",
                "aggregation_mode",
            ]
        ].copy()
        rd = rd.rename(
            columns={
                "valid_observation_count": "historical_packet_valid_count",
                "raw_observation_count": "current_eval_raw_observation_count",
            }
        )
        counts = counts.merge(rd, on=["scene", "point_name"], how="left")
    return counts


def build_inventory(
    scenes: list[str],
    points: pd.DataFrame,
    ann_counts: pd.DataFrame,
    splits: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    split_map = {(r.scene, r.point_name): r.role for r in splits.itertuples(index=False)}
    ann_map = {
        (r.scene, r.point_name): r._asdict() for r in ann_counts.itertuples(index=False)
    }
    candidate_counts = candidates.groupby(["scene", "point_name"]).agg(
        candidate_image_count=("image_name", "nunique"),
        suggested_label_count=("recommended_action", lambda s: int((s == "label").sum())),
        suggested_review_count=("recommended_action", lambda s: int((s == "review").sum())),
    )
    candidate_map = {idx: row.to_dict() for idx, row in candidate_counts.iterrows()}
    for scene in scenes:
        scene_points = points.copy()
        xs = pd.to_numeric(scene_points["cgcs2000_gk_cm108_e_m"], errors="coerce")
        ys = pd.to_numeric(scene_points["cgcs2000_gk_cm108_n_m"], errors="coerce")
        zs = pd.to_numeric(scene_points["cgcs2000_normal_height_m"], errors="coerce")
        minx, maxx = float(xs.min()), float(xs.max())
        miny, maxy = float(ys.min()), float(ys.max())
        zq15, zq85 = float(zs.quantile(0.15)), float(zs.quantile(0.85))
        for r in scene_points.itertuples(index=False):
            point_name = str(r.point_name)
            key = (scene, point_name)
            ann_info = ann_map.get(key, {})
            cand_info = candidate_map.get(key, {})
            x = safe_float(getattr(r, "cgcs2000_gk_cm108_e_m"))
            y = safe_float(getattr(r, "cgcs2000_gk_cm108_n_m"))
            z = safe_float(getattr(r, "cgcs2000_normal_height_m"))
            nx = (x - minx) / (maxx - minx) if maxx > minx else np.nan
            ny = (y - miny) / (maxy - miny) if maxy > miny else np.nan
            edge = nx < 0.15 or nx > 0.85 or ny < 0.15 or ny > 0.85
            height_special = z <= zq15 or z >= zq85
            rows.append(
                {
                    "scene": scene,
                    "point_name": point_name,
                    "survey_x_m": x,
                    "survey_y_m": y,
                    "survey_z_m": z,
                    "point_category": getattr(r, "point_category", ""),
                    "quality_evaluation": getattr(r, "quality_evaluation", ""),
                    "current_v1_2_2_role": split_map.get(key, "not_in_current_scene_split"),
                    "has_current_annotation": key in ann_map,
                    "current_raw_annotation_count": int(ann_info.get("raw_annotation_count", 0) or 0),
                    "annotation_side_usable_count": int(ann_info.get("annotation_side_usable_count", 0) or 0),
                    "historical_packet_valid_count": (
                        int(ann_info.get("historical_packet_valid_count"))
                        if pd.notna(ann_info.get("historical_packet_valid_count", np.nan))
                        else ""
                    ),
                    "candidate_image_count": int(cand_info.get("candidate_image_count", 0) or 0),
                    "suggested_label_count": int(cand_info.get("suggested_label_count", 0) or 0),
                    "suggested_review_count": int(cand_info.get("suggested_review_count", 0) or 0),
                    "scene_position_class": "edge" if edge else "interior",
                    "height_class": "height_special" if height_special else "typical_height",
                    "selection_note": "selection_inputs_exclude_residual_depth_alpha_variance_scatter",
                }
            )
    return pd.DataFrame(rows)


def compute_view_diversity(candidates: pd.DataFrame, ann: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ann_keys = set(zip(ann["scene"], ann["point_name"], ann["raw_image_name"]))
    for (scene, point), g in candidates.groupby(["scene", "point_name"], sort=True):
        usable = g[g["recommended_action"].isin(["keep_existing", "label", "review"])].copy()
        current = g[g.apply(lambda r: (r["scene"], r["point_name"], r["image_name"]) in ann_keys, axis=1)]
        angles = [safe_float(v) for v in usable.get("camera_azimuth_deg", [])]
        ann_angles = [safe_float(v) for v in current.get("camera_azimuth_deg", [])]
        dists = [safe_float(v) for v in usable.get("camera_ground_distance_m", [])]
        current_count = int(current["image_name"].nunique()) if not current.empty else 0
        potential_count = int(usable["image_name"].nunique()) if not usable.empty else 0
        rows.append(
            {
                "scene": scene,
                "point_name": point,
                "current_annotation_views": current_count,
                "potential_usable_or_review_views": potential_count,
                "potential_new_views": max(0, potential_count - current_count),
                "current_azimuth_bins_45deg": unique_direction_bins(ann_angles),
                "potential_azimuth_bins_45deg": unique_direction_bins(angles),
                "current_max_azimuth_gap_deg": max_azimuth_gap(ann_angles),
                "potential_max_azimuth_gap_deg": max_azimuth_gap(angles),
                "median_camera_ground_distance_m": percentile(dists, 50),
                "p95_camera_ground_distance_m": percentile(dists, 95),
                "view_diversity_status": (
                    "control_candidate_strong"
                    if potential_count >= CONTROL_PREFERRED_MIN_VIEWS and unique_direction_bins(angles) >= 3
                    else "formal_candidate"
                    if potential_count >= FORMAL_MIN_VIEWS
                    else "insufficient_views_or_diversity"
                ),
            }
        )
    return pd.DataFrame(rows)


def classify_low_view_causes(inventory: pd.DataFrame, diversity: pd.DataFrame) -> pd.DataFrame:
    df = inventory.merge(
        diversity[
            [
                "scene",
                "point_name",
                "potential_usable_or_review_views",
                "potential_new_views",
                "potential_azimuth_bins_45deg",
                "view_diversity_status",
            ]
        ],
        on=["scene", "point_name"],
        how="left",
    )
    rows = []
    for r in df.itertuples(index=False):
        current = safe_int(getattr(r, "annotation_side_usable_count", 0), 0)
        potential = safe_int(getattr(r, "potential_usable_or_review_views", 0), 0)
        new_views = safe_int(getattr(r, "potential_new_views", 0), 0)
        has_current = bool(getattr(r, "has_current_annotation", False))
        candidate_count = safe_int(getattr(r, "candidate_image_count", 0), 0)
        if not has_current and candidate_count == 0:
            cause = "not_in_scene_candidate_pool_from_available_evidence"
            disposition = "not_recommended_without_new_coverage_evidence"
        elif current > 3 and potential >= FORMAL_MIN_VIEWS:
            cause = "sufficient_current_or_not_low_view"
            disposition = "keep_for_candidate_pool"
        elif potential >= FORMAL_MIN_VIEWS and new_views > 0:
            cause = "annotation_selection_incomplete"
            disposition = "generate_supplemental_label_worklist"
        elif potential >= FORMAL_MIN_VIEWS:
            cause = "additional_annotation_feasible"
            disposition = "review_candidate_views"
        elif int(getattr(r, "candidate_image_count", 0) or 0) > 0:
            cause = "visible_but_ambiguous"
            disposition = "manual_review_required_before_formal_exclusion"
        else:
            cause = "original_image_coverage_insufficient"
            disposition = "exclude_from_future_formal_primary_keep_diagnostic_only"
        rows.append(
            {
                "scene": r.scene,
                "point_name": r.point_name,
                "current_v1_2_2_role": r.current_v1_2_2_role,
                "has_current_annotation": has_current,
                "annotation_side_usable_count": current,
                "historical_packet_valid_count": r.historical_packet_valid_count,
                "candidate_image_count": candidate_count,
                "potential_usable_or_review_views": potential,
                "potential_new_views": new_views,
                "potential_azimuth_bins_45deg": safe_int(getattr(r, "potential_azimuth_bins_45deg", 0), 0),
                "low_view_cause_primary": cause,
                "future_formal_primary_disposition": disposition,
                "is_user_focus_point": (r.scene, r.point_name) in LOW_VIEW_FOCUS,
            }
        )
    out = pd.DataFrame(rows)
    return out[
        ((out["has_current_annotation"]) & (out["annotation_side_usable_count"] <= 3))
        | ((~out["has_current_annotation"]) & (out["candidate_image_count"] > 0))
        | (out["future_formal_primary_disposition"].str.contains("label|review|exclude", regex=True))
    ].copy()


def build_annotation_worklist(candidates: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    inv = inventory.set_index(["scene", "point_name"])
    rows = []
    for r in candidates.itertuples(index=False):
        if str(r.recommended_action) not in {"label", "review"}:
            continue
        key = (r.scene, r.point_name)
        inv_row = inv.loc[key] if key in inv.index else None
        current = int(inv_row["annotation_side_usable_count"]) if inv_row is not None else 0
        is_new_point = bool(inv_row is not None and not bool(inv_row["has_current_annotation"]))
        priority = 1
        if (r.scene, r.point_name) in LOW_VIEW_FOCUS:
            priority = 0
        elif current < FORMAL_MIN_VIEWS:
            priority = 1
        elif is_new_point:
            priority = 2
        else:
            priority = 3
        rows.append(
            {
                "priority": priority,
                "scene": r.scene,
                "point_name": r.point_name,
                "image_name": r.image_name,
                "candidate_source": r.candidate_source,
                "recommended_action": r.recommended_action,
                "visibility_classification": r.visibility_classification,
                "reject_reason": r.reject_reason,
                "raw_candidate_x": safe_float(r.candidate_x),
                "raw_candidate_y": safe_float(r.candidate_y),
                "projection_uncertainty_px": safe_float(getattr(r, "projection_uncertainty_px", np.nan)),
                "camera_azimuth_deg": safe_float(getattr(r, "camera_azimuth_deg", np.nan)),
                "off_nadir_or_pitch_abs_deg": safe_float(getattr(r, "off_nadir_or_pitch_abs_deg", np.nan)),
                "camera_ground_distance_m": safe_float(getattr(r, "camera_ground_distance_m", np.nan)),
                "image_path": getattr(r, "image_path", ""),
                "current_annotation_count": current,
                "is_new_surveyed_point_for_scene": is_new_point,
                "annotation_tool_image_domain": "raw_dji_decoded_pixel_matrix_ignore_exif_orientation",
                "annotation_output_coordinate_domain": "raw_image_zero_based_pixel_centers",
            }
        )
    return pd.DataFrame(rows).sort_values(["priority", "scene", "point_name", "image_name"])


def build_control_design(inventory: pd.DataFrame, diversity: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = inventory.merge(diversity, on=["scene", "point_name"], how="left")
    rows = []
    per_point = []
    for scene, g in df.groupby("scene", sort=True):
        formal = g[g["potential_usable_or_review_views"].fillna(0) >= FORMAL_MIN_VIEWS].copy()
        control = formal[
            (formal["potential_usable_or_review_views"].fillna(0) >= CONTROL_PREFERRED_MIN_VIEWS)
            & (formal["potential_azimuth_bins_45deg"].fillna(0) >= 3)
        ].copy()
        checkpoint = formal.copy()
        total = len(formal)
        checkpoint_floor = max(4, math.ceil(total * 0.3)) if total else 0
        control_low = min(len(control), max(4, math.floor(total * 0.45))) if total else 0
        control_high = min(len(control), max(control_low, total - checkpoint_floor)) if total else 0
        if control_high < control_low:
            control_high = control_low
        rows.append(
            {
                "scene": scene,
                "current_formal_points_in_v1_2_2": int(g["has_current_annotation"].sum()),
                "formal_candidate_points_after_possible_annotation": int(total),
                "control_candidate_points_strict": int(len(control)),
                "checkpoint_candidate_points": int(len(checkpoint)),
                "recommended_control_count_range": f"{control_low}-{control_high}",
                "recommended_checkpoint_count_range": f"{max(0, total-control_high)}-{max(0, total-control_low)}",
                "design_status": (
                    "candidate_pool_supports_control_heavy_design"
                    if len(control) >= 6 and total >= 10 and total - control_high >= 4
                    else "needs_more_points_or_annotations_before_freezing_split"
                ),
                "split_freeze_status": "not_frozen_design_audit_only",
                "if_controls_reduce_checkpoints_too_much": "increase_total_points_not_checkpoint_sacrifice",
            }
        )
        for rr in g.itertuples(index=False):
            potential = safe_int(getattr(rr, "potential_usable_or_review_views", 0), 0)
            bins = safe_int(getattr(rr, "potential_azimuth_bins_45deg", 0), 0)
            if potential >= CONTROL_PREFERRED_MIN_VIEWS and bins >= 3:
                eligibility = "strict_control_candidate"
            elif potential >= FORMAL_MIN_VIEWS:
                eligibility = "checkpoint_or_secondary_control_candidate"
            elif potential > 0:
                eligibility = "diagnostic_or_needs_more_annotation"
            else:
                eligibility = "not_candidate_without_new_coverage"
            per_point.append(
                {
                    "scene": rr.scene,
                    "point_name": rr.point_name,
                    "current_v1_2_2_role": rr.current_v1_2_2_role,
                    "potential_usable_or_review_views": potential,
                    "potential_azimuth_bins_45deg": bins,
                    "scene_position_class": rr.scene_position_class,
                    "height_class": rr.height_class,
                    "v1_3_candidate_eligibility": eligibility,
                    "selection_inputs": "XYZ+usable_count+view_diversity+visibility_QC+scene_boundary_only",
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(per_point)


def draw_scene_plots(out_dir: Path, points: pd.DataFrame, inventory: pd.DataFrame, control_design: pd.DataFrame) -> list[Path]:
    if plt is None:
        raise RuntimeError("matplotlib is not available")
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for scene in SCENES:
        inv = inventory[inventory["scene"] == scene].copy()
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.scatter(
            points["cgcs2000_gk_cm108_e_m"],
            points["cgcs2000_gk_cm108_n_m"],
            s=20,
            c="#cfcfcf",
            label="all surveyed points",
        )
        cur = inv[inv["has_current_annotation"]]
        cand = inv[(~inv["has_current_annotation"]) & (inv["candidate_image_count"] > 0)]
        if not cur.empty:
            ax.scatter(cur["survey_x_m"], cur["survey_y_m"], s=55, c="#1f77b4", label="current v1.2.2 points")
        if not cand.empty:
            ax.scatter(cand["survey_x_m"], cand["survey_y_m"], s=45, c="#ff7f0e", marker="^", label="new surveyed candidates")
        for _, rr in inv.iterrows():
            if rr["has_current_annotation"] or rr["candidate_image_count"] > 0:
                ax.text(rr["survey_x_m"], rr["survey_y_m"], rr["point_name"], fontsize=7)
        ax.set_title(f"{scene} surveyed point distribution")
        ax.set_xlabel("CGCS2000 GK CM108 E (m)")
        ax.set_ylabel("CGCS2000 GK CM108 N (m)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        path = plot_dir / f"{scene}_spatial_distribution.png"
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

        fig, ax = plt.subplots(figsize=(8, 4))
        inv2 = inv[inv["candidate_image_count"] > 0].copy()
        inv2 = inv2.sort_values("survey_z_m")
        ax.bar(inv2["point_name"], inv2["survey_z_m"], color="#4c78a8")
        ax.set_title(f"{scene} candidate height distribution")
        ax.set_ylabel("normal height (m)")
        ax.tick_params(axis="x", rotation=90, labelsize=7)
        fig.tight_layout()
        path = plot_dir / f"{scene}_height_distribution.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def load_font() -> Any:
    if ImageFont is None:
        return None
    for name in ["arial.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, 13)
        except Exception:
            continue
    return ImageFont.load_default()


def make_contact_sheets(
    out_dir: Path,
    candidates: pd.DataFrame,
    worklist: pd.DataFrame,
    low_view: pd.DataFrame,
    inventory: pd.DataFrame,
) -> tuple[pd.DataFrame, Path]:
    if Image is None:
        raise RuntimeError("Pillow is not available")
    contact_dir = out_dir / "contact_sheets"
    contact_dir.mkdir(parents=True, exist_ok=True)
    font = load_font()
    low_keys = set(zip(low_view["scene"], low_view["point_name"]))
    work_keys = set(zip(worklist["scene"], worklist["point_name"]))
    new_keys = set(
        zip(
            inventory[(~inventory["has_current_annotation"]) & (inventory["candidate_image_count"] > 0)]["scene"],
            inventory[(~inventory["has_current_annotation"]) & (inventory["candidate_image_count"] > 0)]["point_name"],
        )
    )
    target_keys = low_keys | work_keys | new_keys
    records = []
    for scene, point in sorted(target_keys):
        g = candidates[(candidates["scene"] == scene) & (candidates["point_name"] == point)].copy()
        if g.empty:
            continue
        g["_action_rank"] = g["recommended_action"].map({"keep_existing": 0, "label": 1, "review": 2, "reject": 9}).fillna(5)
        g["_edge"] = pd.to_numeric(g["edge_margin_px"], errors="coerce").fillna(-9999.0)
        g = g.sort_values(["_action_rank", "_edge", "image_name"], ascending=[True, False, True]).head(CONTACT_MAX_PER_POINT)
        tile_w, tile_h = 420, 300
        cols = 2
        rows = int(math.ceil(len(g) / cols))
        sheet = Image.new("RGB", (cols * tile_w, rows * tile_h), "white")
        draw = ImageDraw.Draw(sheet)
        for idx, rr in enumerate(g.itertuples(index=False)):
            col = idx % cols
            row = idx // cols
            x0 = col * tile_w
            y0 = row * tile_h
            image_path = getattr(rr, "image_path", "")
            if not image_path or not Path(str(image_path)).exists():
                fallback = Path(r"E:\datasets\M3M-GCP\scenes") / scene / str(rr.image_name)
                image_path = str(fallback)
            try:
                with Image.open(image_path) as im0:
                    im = im0.convert("RGB")
                    w, h = im.size
                    cx = safe_float(rr.candidate_x, w / 2)
                    cy = safe_float(rr.candidate_y, h / 2)
                    thumb = im.copy()
                    thumb.thumbnail((190, 130))
                    crop_size = 520
                    left = int(max(0, min(w - 1, cx) - crop_size // 2))
                    top = int(max(0, min(h - 1, cy) - crop_size // 2))
                    right = int(min(w, left + crop_size))
                    bottom = int(min(h, top + crop_size))
                    crop = im.crop((left, top, right, bottom))
                    crop = crop.resize((190, 190))
                    cdraw = ImageDraw.Draw(crop)
                    px = int((cx - left) / max(1, right - left) * 190)
                    py = int((cy - top) / max(1, bottom - top) * 190)
                    radius = max(5, min(70, int(safe_float(getattr(rr, "projection_uncertainty_px", 0.0), 0.0) / max(1, right - left) * 190)))
                    cdraw.ellipse((px - radius, py - radius, px + radius, py + radius), outline="magenta", width=2)
                    cdraw.line((px - 8, py, px + 8, py), fill="yellow", width=2)
                    cdraw.line((px, py - 8, px, py + 8), fill="yellow", width=2)
                    sheet.paste(thumb, (x0 + 8, y0 + 8))
                    sheet.paste(crop, (x0 + 218, y0 + 8))
            except Exception as exc:
                draw.rectangle((x0 + 8, y0 + 8, x0 + 408, y0 + 198), outline="red")
                draw.text((x0 + 12, y0 + 12), f"image load failed: {exc}", fill="red", font=font)
            text = (
                f"{scene} | {point}\n"
                f"{rr.image_name}\n"
                f"source={rr.candidate_source} action={rr.recommended_action}\n"
                f"x={safe_float(rr.candidate_x):.1f}, y={safe_float(rr.candidate_y):.1f}, "
                f"az={safe_float(getattr(rr, 'camera_azimuth_deg', np.nan)):.1f}, "
                f"baseline={safe_float(getattr(rr, 'camera_ground_distance_m', np.nan)):.1f}m\n"
                f"class={rr.visibility_classification} reason={rr.reject_reason}"
            )
            draw.multiline_text((x0 + 8, y0 + 206), text, fill="black", font=font, spacing=3)
            draw.rectangle((x0, y0, x0 + tile_w - 1, y0 + tile_h - 1), outline="#cccccc")
        rel_name = f"{scene}_{point}_contact_sheet.png"
        out_path = contact_dir / rel_name
        sheet.save(out_path)
        records.append(
            {
                "scene": scene,
                "point_name": point,
                "contact_sheet": str(out_path),
                "candidate_tiles": int(len(g)),
                "contains_low_view_point": (scene, point) in low_keys,
                "contains_new_surveyed_point_candidate": (scene, point) in new_keys,
            }
        )
    index = contact_dir / "index.html"
    html_lines = [
        "<!doctype html><html><head><meta charset='utf-8'><title>GS-GCP contact sheets</title>",
        "<style>body{font-family:Arial,sans-serif} img{max-width:100%;border:1px solid #ddd} .card{margin:24px 0}</style></head><body>",
        "<h1>GS-GCP Multi-view Annotation Audit Contact Sheets</h1>",
        "<p>All coordinates are raw decoded-image pixel coordinates. Magenta circles indicate projection/search uncertainty, not ground truth visibility.</p>",
    ]
    for rec in records:
        rel = Path(rec["contact_sheet"]).name
        html_lines.append(
            f"<div class='card'><h2>{html.escape(rec['scene'])} | {html.escape(rec['point_name'])}</h2>"
            f"<p>tiles={rec['candidate_tiles']} low_view={rec['contains_low_view_point']} "
            f"new_point_candidate={rec['contains_new_surveyed_point_candidate']}</p>"
            f"<img src='{html.escape(rel)}'></div>"
        )
    html_lines.append("</body></html>")
    index.write_text("\n".join(html_lines), encoding="utf-8")
    return pd.DataFrame(records), index


def write_protocol_docs(out_dir: Path, run_meta: dict[str, Any], summary: pd.DataFrame, low_view: pd.DataFrame) -> None:
    brief = out_dir / "REVIEW_BRIEF.md"
    low_view_display = low_view[
        (low_view["has_current_annotation"] & (low_view["annotation_side_usable_count"] <= 3))
        | (low_view["is_user_focus_point"])
        | (low_view["future_formal_primary_disposition"].str.contains("label|review", regex=True))
    ].copy()
    lines = [
        "# GS-GCP Multi-View Annotation Expansion And Control-Heavy Design Audit",
        "",
        "## Scope",
        "",
        "- This package is a design audit only.",
        "- `v1.2.2` remains frozen as the sparse-control diagnostic benchmark.",
        "- No release, split, survey coordinate, packet, evaluator, residual, or formal metric was modified.",
        "- Candidate discovery deliberately uses multiple evidence paths and does not depend solely on a single current model/Sim(3).",
        "",
        "## Candidate Discovery Sources",
        "",
        "1. Current v1.2.2 annotation image list.",
        "2. Coarse `gcp_annotation_candidates_20260617_all` seed, not treated as visibility proof.",
        "3. Annotation-ray triangulation/reprojection using COLMAP source camera poses for points with at least two views.",
        "4. Wide-margin review for low-view or non-triangulatable points.",
        "",
        "## Key Scene Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Low-View Focus Points",
        "",
        low_view_display[
            [
                "scene",
                "point_name",
                "current_v1_2_2_role",
                "has_current_annotation",
                "annotation_side_usable_count",
                "potential_usable_or_review_views",
                "low_view_cause_primary",
                "future_formal_primary_disposition",
            ]
        ].to_markdown(index=False),
        "",
        "## v1.3.0 Draft Annotation Contract",
        "",
        "- Human annotation must be performed in raw DJI decoded-image pixel domain.",
        "- Pixel convention is zero-based pixel centers.",
        "- Store raw image SHA-256, dimensions, camera identity, raw x/y, and raw-to-target provenance for each new observation.",
        "- Conversion chain: raw pixel -> normalized source-camera ray -> undistorted benchmark target pixel -> packet-native pixel.",
        "- Direct labeling on R8 packets or undistorted render images must not be presented as raw-domain annotation.",
        "- Primary annotation and independent review passes should be separate; reviewers must not see model residuals or point error.",
        "",
        "## Audit Metadata",
        "",
        "```json",
        json.dumps(run_meta, ensure_ascii=False, indent=2, sort_keys=True),
        "```",
    ]
    brief.write_text("\n".join(lines), encoding="utf-8")

    gpt = out_dir / "MESSAGE_TO_GPT.txt"
    gpt.write_text(
        "\n".join(
            [
                "GPT：",
                "本包为 GS-GCP 六场景 Multi-View Annotation Expansion and Control-Heavy Benchmark Design Audit。",
                "",
                "请重点审核：",
                "1. 候选影像发现是否足以避免 a single current model/Sim(3) transform sensitivity 导致的漏图；",
                "2. 低视图点分类和补标工作清单是否足够支持 v1.3.0 设计；",
                "3. view-diversity 指标和 control-heavy 候选范围是否适合作为下一步 split 设计依据；",
                "4. raw-domain annotation + raw→target→packet conversion protocol 是否足以避免 v1.1/v1.2 发生过的 pixel-domain 问题；",
                "5. 是否同意保留 v1.2.2 sparse-control diagnostic track，并将新增 observations/points 的正式版本规划为 v1.3.0。",
                "",
                "本轮未修改 v1.2.2 release、split、survey coordinates、packets、evaluator，也未重新评测或使用 GPU。",
            ]
        ),
        encoding="utf-8",
    )

    protocol = out_dir / "V1_3_0_DRAFT_PIXEL_DOMAIN_AND_QC_PROTOCOL.md"
    protocol.write_text(
        "\n".join(
            [
                "# v1.3.0 Draft Pixel-Domain And Annotation QC Protocol",
                "",
                "## Pixel Domain",
                "",
                "All new human annotations are raw DJI decoded-image pixel coordinates with EXIF orientation ignored, matching v1.2.2 source-domain policy. Coordinates use zero-based pixel centers.",
                "",
                "## Stored Fields",
                "",
                "- scene, point_name, raw_image_name",
                "- raw image SHA-256, decoded width/height, camera identity",
                "- raw_manual_x, raw_manual_y as decimal strings",
                "- raw-to-target transform provenance, round-trip error, target bounds status",
                "- reviewer identity fields may be pseudonymous if needed",
                "",
                "## Conversion Chain",
                "",
                "raw pixel -> normalized source-camera ray -> undistorted benchmark target pixel -> packet-native pixel",
                "",
                "## QC Passes",
                "",
                "1. Primary annotation pass.",
                "2. Independent review pass with residuals and current method errors hidden.",
                "3. Disagreement review. Ambiguous observations are not retained by majority guessing.",
                "4. Cross-view point identity consistency check.",
                "5. For low-light 5K, enhanced display is allowed only as a visual aid; saved coordinates remain raw-domain.",
                "",
                "## Formal Candidate View Count",
                "",
                "Formal primary points should have at least 4 usable observations and preferably 6-8+ diverse observations. Controls should preferentially have at least 6 usable diverse observations. Single-view, two-view, and three-view controls are not allowed in the future formal primary subset.",
            ]
        ),
        encoding="utf-8",
    )


def package_outputs(out_dir: Path, review_root: Path, stamp: str) -> tuple[Path, Path]:
    review_root.mkdir(parents=True, exist_ok=True)
    base = review_root / f"GPT_GCP_MULTIVIEW_ANNOTATION_EXPANSION_CONTROL_HEAVY_AUDIT_REVIEW_{stamp}.zip"
    zip_path = base
    i = 1
    while zip_path.exists():
        zip_path = review_root / f"{base.stem}_{i:02d}.zip"
        i += 1
    manifest_rows = []
    for p in sorted(out_dir.rglob("*")):
        if p.is_file():
            rel = p.relative_to(out_dir).as_posix()
            manifest_rows.append({"path": rel, "size": p.stat().st_size, "sha256": sha256_file(p)})
    manifest_path = out_dir / "PACKAGE_FILE_MANIFEST.json"
    write_json(manifest_path, manifest_rows)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in sorted(out_dir.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(out_dir).as_posix())
    sha_path = zip_path.with_suffix(zip_path.suffix + ".sha256")
    sha_path.write_text(f"{sha256_file(zip_path)}  {zip_path.name}\n", encoding="ascii")
    return zip_path, sha_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--dataset", type=Path, default=Path(r"E:\datasets\M3M-GCP"))
    parser.add_argument(
        "--release",
        type=Path,
        default=Path(r"E:\datasets\M3M-GCP\scenes\gcp_manual_annotations_v1_2_2"),
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path(r"E:\M3M-GCP-3DGS\outputs\gcp_annotation_candidates_20260617_all"),
    )
    parser.add_argument(
        "--residual_diag",
        type=Path,
        default=Path(r"E:\M3M-GCP-3DGS\outputs\six_scene_residual_outlier_diagnostics_20260702_101500"),
    )
    parser.add_argument("--stamp", default=time.strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    paths = Paths(
        repo=args.repo,
        dataset=args.dataset,
        release=args.release,
        candidates=args.candidates,
        residual_diag=args.residual_diag,
        output_root=args.repo / "outputs" / f"gcp_multiview_annotation_expansion_control_heavy_audit_{args.stamp}",
        review_root=args.repo / "outputs" / "gpt_review_packages",
    )
    paths.output_root.mkdir(parents=True, exist_ok=False)

    if Image is None:
        raise RuntimeError("Pillow is required for contact sheets")
    if plt is None:
        raise RuntimeError("matplotlib is required for plots")

    git_head = run_cmd(["git", "rev-parse", "HEAD"], paths.repo)
    git_status = run_cmd(["git", "status", "--short"], paths.repo)
    run_meta = {
        "stamp": args.stamp,
        "scope": "design_audit_only_no_release_or_evaluator_mutation",
        "repo": str(paths.repo),
        "dataset": str(paths.dataset),
        "release": str(paths.release),
        "candidate_seed": str(paths.candidates),
        "git_head": git_head,
        "git_status_short": git_status,
        "python": sys.version,
        "hard_boundaries": [
            "v1.2.2 release not modified",
            "split not modified",
            "survey coordinates not modified",
            "packets not modified",
            "evaluator not modified",
            "no GPU",
            "no formal regression",
            "residuals not used for candidate/control selection",
        ],
    }

    ann = load_release_annotations(paths.release)
    points = pd.read_csv(paths.release / "gcp_points_primary_usable_cgcs2000_cm108_v1.csv", dtype={"point_name": str})
    splits = pd.read_csv(paths.release / "gcp_control_checkpoint_splits_v1.csv", dtype={"point_name": str})
    coarse = load_candidate_seed(paths.candidates)
    image_meta = load_image_metadata(paths.candidates)
    camera_manifest = load_camera_provenance(paths.release)
    camera_idx = build_camera_indices(camera_manifest)

    ann_counts = make_annotation_counts(ann, paths.residual_diag)
    current_candidates = build_current_annotation_candidates(ann)
    ann_keys = set(zip(ann["scene"], ann["point_name"], ann["raw_image_name"]))
    coarse_candidates = build_coarse_candidates(coarse, ann_keys)
    if not coarse_candidates.empty:
        coarse_candidates = coarse_candidates.sort_values(["scene", "point_name", "candidate_rank_for_point"])
        coarse_candidates = (
            coarse_candidates.groupby(["scene", "point_name"], group_keys=False)
            .head(COARSE_MAX_PER_POINT)
            .copy()
        )
    tri_candidates, tri_summary, loo = build_triangulation_candidates(ann, camera_idx, image_meta)
    all_candidates = pd.concat(
        [current_candidates, coarse_candidates, tri_candidates],
        ignore_index=True,
        sort=False,
    )
    all_candidates = dedupe_candidates(all_candidates)
    all_candidates = add_image_metadata(all_candidates, image_meta, points)
    all_candidates = classify_candidates(all_candidates, ann_counts)

    inventory = build_inventory(SCENES, points, ann_counts, splits, all_candidates)
    diversity = compute_view_diversity(all_candidates, ann)
    low_view = classify_low_view_causes(inventory, diversity)
    worklist = build_annotation_worklist(all_candidates, inventory)
    control_summary, control_per_point = build_control_design(inventory, diversity)

    annotated_recall_rows = []
    candidate_key_set = set(zip(all_candidates["scene"], all_candidates["point_name"], all_candidates["image_name"]))
    coarse_key_set = set(
        zip(
            coarse_candidates.get("scene", []),
            coarse_candidates.get("point_name", []),
            coarse_candidates.get("image_name", []),
        )
    )
    tri_key_set = set(
        zip(
            tri_candidates.get("scene", []),
            tri_candidates.get("point_name", []),
            tri_candidates.get("image_name", []),
        )
    )
    for scene, g in ann.groupby("scene", sort=True):
        total = len(g)
        union_recalled = sum((r.scene, r.point_name, r.raw_image_name) in candidate_key_set for r in g.itertuples())
        coarse_recalled = sum((r.scene, r.point_name, r.raw_image_name) in coarse_key_set for r in g.itertuples())
        tri_recalled = sum((r.scene, r.point_name, r.raw_image_name) in tri_key_set for r in g.itertuples())
        annotated_recall_rows.append(
            {
                "scene": scene,
                "annotated_observation_count": total,
                "union_recalled_count": union_recalled,
                "union_recall_rate": union_recalled / total if total else np.nan,
                "coarse_seed_recalled_count": coarse_recalled,
                "coarse_seed_recall_rate": coarse_recalled / total if total else np.nan,
                "triangulation_recalled_count": tri_recalled,
                "triangulation_recall_rate": tri_recalled / total if total else np.nan,
                "completeness_rule": "coverage_insufficient_not_claimed_when_recall_or_visual_QC_incomplete",
            }
        )
    recall = pd.DataFrame(annotated_recall_rows)

    scene_summary_rows = []
    for scene in SCENES:
        inv = inventory[inventory["scene"] == scene]
        wl = worklist[worklist["scene"] == scene]
        lv = low_view[
            (low_view["scene"] == scene)
            & (low_view["has_current_annotation"])
            & (low_view["annotation_side_usable_count"] <= 3)
        ]
        scene_summary_rows.append(
            {
                "scene": scene,
                "current_v1_2_2_points": int(inv["has_current_annotation"].sum()),
                "all_surveyed_points_considered": int(len(inv)),
                "points_with_any_candidate_images": int((inv["candidate_image_count"] > 0).sum()),
                "low_view_points_le_3_current_views": int(len(lv)),
                "suggested_label_pairs": int((wl["recommended_action"] == "label").sum()),
                "suggested_review_pairs": int((wl["recommended_action"] == "review").sum()),
                "new_surveyed_point_candidates": int(((~inv["has_current_annotation"]) & (inv["candidate_image_count"] > 0)).sum()),
                "current_annotated_image_recall_union": float(
                    recall.loc[recall["scene"] == scene, "union_recall_rate"].iloc[0]
                ),
                "v1_3_split_status": "not_frozen_design_audit_only",
            }
        )
    scene_summary = pd.DataFrame(scene_summary_rows)

    # Write tables.
    tables = {
        "six_scene_surveyed_point_inventory.csv": inventory,
        "point_image_candidate_coverage_matrix.csv": all_candidates,
        "annotation_sufficiency_matrix.csv": diversity,
        "low_view_cause_classification.csv": low_view,
        "candidate_discovery_recall_summary.csv": recall,
        "leave_one_annotation_out_candidate_recall.csv": loo,
        "triangulation_candidate_summary.csv": tri_summary,
        "manual_supplemental_annotation_worklist.csv": worklist,
        "control_heavy_scene_design_options.csv": control_summary,
        "control_heavy_per_point_candidate_roles.csv": control_per_point,
        "scene_audit_summary.csv": scene_summary,
    }
    for name, df in tables.items():
        df.to_csv(paths.output_root / name, index=False, encoding="utf-8-sig")

    draw_scene_plots(paths.output_root, points, inventory, control_summary)
    contact_manifest, contact_index = make_contact_sheets(paths.output_root, all_candidates, worklist, low_view, inventory)
    contact_manifest.to_csv(paths.output_root / "contact_sheet_manifest.csv", index=False, encoding="utf-8-sig")

    write_protocol_docs(paths.output_root, run_meta, scene_summary, low_view)
    write_json(paths.output_root / "audit_run_manifest.json", run_meta)

    zip_path, sha_path = package_outputs(paths.output_root, paths.review_root, args.stamp.split("_")[0])
    print(json.dumps({"output_root": str(paths.output_root), "zip": str(zip_path), "sha256": str(sha_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
