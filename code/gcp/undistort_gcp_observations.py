from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
COLMAP_UTILS = REPO_ROOT / "code" / "colmap" / "utils"
sys.path.insert(0, str(COLMAP_UTILS))

from read_write_model import read_model  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def camera_normalized_from_pixel(camera: Any, u: float, v: float, max_iter: int = 20) -> tuple[float, float]:
    model = camera.model.upper()
    params = [float(x) for x in camera.params]
    if model == "SIMPLE_PINHOLE":
        f, cx, cy = params
        return (u - cx) / f, (v - cy) / f
    if model == "PINHOLE":
        fx, fy, cx, cy = params
        return (u - cx) / fx, (v - cy) / fy
    if model == "SIMPLE_RADIAL":
        f, cx, cy, k = params
        xd = (u - cx) / f
        yd = (v - cy) / f
        x = xd
        y = yd
        for _ in range(max_iter):
            r2 = x * x + y * y
            scale = 1.0 + k * r2
            if abs(scale) < 1e-12:
                break
            x = xd / scale
            y = yd / scale
        return x, y
    raise ValueError(f"unsupported source camera model for pixel normalization: {camera.model}")


def camera_pixel_from_normalized(camera: Any, x: float, y: float) -> tuple[float, float]:
    model = camera.model.upper()
    params = [float(p) for p in camera.params]
    if model == "SIMPLE_PINHOLE":
        f, cx, cy = params
        return f * x + cx, f * y + cy
    if model == "PINHOLE":
        fx, fy, cx, cy = params
        return fx * x + cx, fy * y + cy
    if model == "SIMPLE_RADIAL":
        f, cx, cy, k = params
        r2 = x * x + y * y
        scale = 1.0 + k * r2
        return f * x * scale + cx, f * y * scale + cy
    raise ValueError(f"unsupported target camera model for pixel projection: {camera.model}")


def parse_float(row: dict[str, str], *names: str) -> float:
    for name in names:
        value = row.get(name, "")
        if str(value).strip() != "":
            return float(value)
    raise ValueError(f"missing numeric field, tried {names}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transform manually annotated GCP observations from a source COLMAP pixel domain to a target COLMAP pixel domain."
    )
    parser.add_argument("--annotations_csv", required=True)
    parser.add_argument("--source_colmap_model", required=True)
    parser.add_argument("--target_colmap_model", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_manifest", default="")
    parser.add_argument("--source_domain_label", default="distorted_original_colmap")
    parser.add_argument("--target_domain_label", default="undistorted_training_colmap")
    args = parser.parse_args()

    annotations_csv = Path(args.annotations_csv)
    source_model = Path(args.source_colmap_model)
    target_model = Path(args.target_colmap_model)
    out_csv = Path(args.out_csv)
    out_manifest = Path(args.out_manifest) if args.out_manifest else out_csv.with_suffix(".manifest.json")

    source_cameras, source_images, _source_points = read_model(source_model)
    target_cameras, target_images, _target_points = read_model(target_model)
    source_by_name = {image.name: image for image in source_images.values()}
    target_by_name = {image.name: image for image in target_images.values()}

    rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    for row in read_csv(annotations_csv):
        out: dict[str, Any] = dict(row)
        image_name = str(row.get("image_name", "")).strip()
        out["source_pixel_domain"] = args.source_domain_label
        out["target_pixel_domain"] = args.target_domain_label
        out["source_u_px"] = row.get("u_px", row.get("manual_x", ""))
        out["source_v_px"] = row.get("v_px", row.get("manual_y", ""))
        out["source_colmap_model"] = str(source_model)
        out["target_colmap_model"] = str(target_model)
        out["pixel_transform_status"] = "failed"
        out["pixel_transform_reason"] = ""

        if image_name not in source_by_name:
            out["pixel_transform_reason"] = "missing_source_image"
            counters["missing_source_image"] += 1
            rows.append(out)
            continue
        if image_name not in target_by_name:
            out["pixel_transform_reason"] = "missing_target_image"
            counters["missing_target_image"] += 1
            rows.append(out)
            continue

        try:
            u = parse_float(row, "u_px", "manual_x")
            v = parse_float(row, "v_px", "manual_y")
            source_image = source_by_name[image_name]
            target_image = target_by_name[image_name]
            source_camera = source_cameras[source_image.camera_id]
            target_camera = target_cameras[target_image.camera_id]
            x_norm, y_norm = camera_normalized_from_pixel(source_camera, u, v)
            u_target, v_target = camera_pixel_from_normalized(target_camera, x_norm, y_norm)
            in_bounds = (
                math.isfinite(u_target)
                and math.isfinite(v_target)
                and 0.0 <= u_target < float(target_camera.width)
                and 0.0 <= v_target < float(target_camera.height)
            )
            out["u_px"] = f"{u_target:.6f}"
            out["v_px"] = f"{v_target:.6f}"
            out["undistorted_u_px"] = f"{u_target:.6f}"
            out["undistorted_v_px"] = f"{v_target:.6f}"
            out["normalized_x"] = f"{x_norm:.12g}"
            out["normalized_y"] = f"{y_norm:.12g}"
            out["target_width"] = int(target_camera.width)
            out["target_height"] = int(target_camera.height)
            out["undistorted_in_bounds"] = int(in_bounds)
            out["pixel_transform_status"] = "ok" if in_bounds else "out_of_bounds"
            out["pixel_transform_reason"] = "" if in_bounds else "target_pixel_out_of_bounds"
            counters[out["pixel_transform_status"]] += 1
        except Exception as exc:  # noqa: BLE001
            out["pixel_transform_reason"] = type(exc).__name__ + ": " + str(exc)
            counters["exception"] += 1
        rows.append(out)

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    write_csv(out_csv, rows, fieldnames)

    source_camera_models = sorted({cam.model for cam in source_cameras.values()})
    target_camera_models = sorted({cam.model for cam in target_cameras.values()})
    manifest = {
        "schema": "ms_gcp_observation_pixel_domain_transform_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "annotations_csv": str(annotations_csv),
        "source_colmap_model": str(source_model),
        "target_colmap_model": str(target_model),
        "out_csv": str(out_csv),
        "source_domain_label": args.source_domain_label,
        "target_domain_label": args.target_domain_label,
        "source_camera_models": source_camera_models,
        "target_camera_models": target_camera_models,
        "row_count": len(rows),
        "status_counts": dict(sorted(counters.items())),
        "notes": [
            "This transform maps raw manually annotated pixels into the undistorted training/rendering pixel domain.",
            "It does not change GCP coordinates, camera poses, Gaussian checkpoints, or evaluation metrics.",
        ],
    }
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
