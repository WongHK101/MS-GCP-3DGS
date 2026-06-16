from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[2]
CGCS2000_A = 6378137.0
CGCS2000_F = 1.0 / 298.257222101
CGCS2000_E2 = CGCS2000_F * (2.0 - CGCS2000_F)
CGCS2000_EP2 = CGCS2000_E2 / (1.0 - CGCS2000_E2)
CGCS2000_CM_DEG = 108.0
CGCS2000_FALSE_EASTING_M = 500000.0


@dataclass
class CameraMeta:
    image_path: Path
    image_name: str
    width: int
    height: int
    lat: float
    lon: float
    projected_e: float
    projected_n: float
    ellipsoid_alt_m: float
    rel_alt_m: float | None
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    focal_px: float


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(str(value).replace("+", ""))
    except Exception:
        return None


def latlon_to_cgcs2000_gk_cm108(lat_deg: float, lon_deg: float) -> tuple[float, float]:
    """Project geographic coordinates to CGCS2000 / 3-degree GK CM 108E.

    This implements the EPSG:4545 Transverse Mercator parameters used by the
    RTK project: central meridian 108 E, scale 1, false easting 500000 m, and
    false northing 0 m. WGS84 and CGCS2000 geographic coordinates are treated
    as coincident for this sub-kilometre candidate-search use; final GCP
    evaluation must use surveyed image observations and the canonical table.
    """
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    lon0 = math.radians(CGCS2000_CM_DEG)
    n = CGCS2000_A / math.sqrt(1.0 - CGCS2000_E2 * math.sin(lat) ** 2)
    t = math.tan(lat) ** 2
    c = CGCS2000_EP2 * math.cos(lat) ** 2
    a = math.cos(lat) * (lon - lon0)
    m = CGCS2000_A * (
        (1 - CGCS2000_E2 / 4 - 3 * CGCS2000_E2**2 / 64 - 5 * CGCS2000_E2**3 / 256) * lat
        - (
            3 * CGCS2000_E2 / 8
            + 3 * CGCS2000_E2**2 / 32
            + 45 * CGCS2000_E2**3 / 1024
        )
        * math.sin(2 * lat)
        + (15 * CGCS2000_E2**2 / 256 + 45 * CGCS2000_E2**3 / 1024) * math.sin(4 * lat)
        - (35 * CGCS2000_E2**3 / 3072) * math.sin(6 * lat)
    )
    easting = CGCS2000_FALSE_EASTING_M + n * (
        a
        + (1 - t + c) * a**3 / 6
        + (5 - 18 * t + t**2 + 72 * c - 58 * CGCS2000_EP2) * a**5 / 120
    )
    northing = (
        m
        + n
        * math.tan(lat)
        * (
            a**2 / 2
            + (5 - t + 9 * c + 4 * c**2) * a**4 / 24
            + (61 - 58 * t + t**2 + 600 * c - 330 * CGCS2000_EP2) * a**6 / 720
        )
    )
    return easting, northing


def load_gcps(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "point_name": row["point_name"],
                    "projected_e": float(row["cgcs2000_gk_cm108_e_m"]),
                    "projected_n": float(row["cgcs2000_gk_cm108_n_m"]),
                    "normal_height_m": float(row["cgcs2000_normal_height_m"]),
                    "ellipsoid_height_m": float(row["wgs84_ellipsoid_height_m"]),
                    "point_category": row.get("point_category", ""),
                    "quality_evaluation": row.get("quality_evaluation", ""),
                }
            )
    return rows


def run_exiftool(exiftool: str, images: Sequence[Path], chunk_size: int = 120) -> List[Dict[str, Any]]:
    fields = [
        "-n",
        "-GPSLatitude",
        "-GPSLongitude",
        "-AbsoluteAltitude",
        "-RelativeAltitude",
        "-FlightYawDegree",
        "-FlightPitchDegree",
        "-FlightRollDegree",
        "-GimbalYawDegree",
        "-GimbalPitchDegree",
        "-GimbalRollDegree",
        "-CalibratedFocalLength",
        "-ExifImageWidth",
        "-ExifImageHeight",
        "-ImageWidth",
        "-ImageHeight",
    ]
    all_rows: List[Dict[str, Any]] = []
    for start in range(0, len(images), chunk_size):
        chunk = images[start : start + chunk_size]
        cmd = [exiftool, "-json", *fields, *[str(p) for p in chunk]]
        proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        all_rows.extend(json.loads(proc.stdout))
    return all_rows


def parse_camera_meta(row: Dict[str, Any]) -> CameraMeta | None:
    src = row.get("SourceFile")
    lat = _float_or_none(row.get("GPSLatitude"))
    lon = _float_or_none(row.get("GPSLongitude"))
    abs_alt = _float_or_none(row.get("AbsoluteAltitude"))
    if not src or lat is None or lon is None or abs_alt is None:
        return None
    width = int(row.get("ExifImageWidth") or row.get("ImageWidth") or 0)
    height = int(row.get("ExifImageHeight") or row.get("ImageHeight") or 0)
    focal_px = _float_or_none(row.get("CalibratedFocalLength"))
    if width <= 0 or height <= 0 or focal_px is None or focal_px <= 0:
        return None
    yaw = _float_or_none(row.get("GimbalYawDegree"))
    if yaw is None:
        yaw = _float_or_none(row.get("FlightYawDegree")) or 0.0
    pitch = _float_or_none(row.get("GimbalPitchDegree"))
    if pitch is None:
        pitch = _float_or_none(row.get("FlightPitchDegree")) or -90.0
    roll = _float_or_none(row.get("GimbalRollDegree"))
    if roll is None:
        roll = _float_or_none(row.get("FlightRollDegree")) or 0.0
    e, n = latlon_to_cgcs2000_gk_cm108(lat, lon)
    p = Path(src)
    return CameraMeta(
        image_path=p,
        image_name=p.name,
        width=width,
        height=height,
        lat=lat,
        lon=lon,
        projected_e=e,
        projected_n=n,
        ellipsoid_alt_m=abs_alt,
        rel_alt_m=_float_or_none(row.get("RelativeAltitude")),
        yaw_deg=float(yaw),
        pitch_deg=float(pitch),
        roll_deg=float(roll),
        focal_px=float(focal_px),
    )


def _rotate_axes_by_roll(right: np.ndarray, down: np.ndarray, roll_deg: float) -> tuple[np.ndarray, np.ndarray]:
    r = math.radians(roll_deg)
    cr, sr = math.cos(r), math.sin(r)
    return cr * right + sr * down, -sr * right + cr * down


def project_gcp(cam: CameraMeta, gcp: Dict[str, Any]) -> Dict[str, Any] | None:
    yaw = math.radians(cam.yaw_deg)
    pitch = math.radians(-cam.pitch_deg)
    heading = np.asarray([math.sin(yaw), math.cos(yaw), 0.0], dtype=np.float64)
    down_world = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
    forward = math.cos(pitch) * heading + math.sin(pitch) * down_world
    forward = forward / np.linalg.norm(forward)
    right = np.asarray([math.cos(yaw), -math.sin(yaw), 0.0], dtype=np.float64)
    right = right / np.linalg.norm(right)
    img_down = np.cross(forward, right)
    img_down = img_down / np.linalg.norm(img_down)
    right, img_down = _rotate_axes_by_roll(right, img_down, cam.roll_deg)
    delta = np.asarray(
        [
            float(gcp["projected_e"]) - cam.projected_e,
            float(gcp["projected_n"]) - cam.projected_n,
            float(gcp["ellipsoid_height_m"]) - cam.ellipsoid_alt_m,
        ],
        dtype=np.float64,
    )
    z = float(np.dot(delta, forward))
    if z <= 1e-6:
        return None
    x = float(np.dot(delta, right))
    y = float(np.dot(delta, img_down))
    u = cam.width / 2.0 + cam.focal_px * x / z
    v = cam.height / 2.0 + cam.focal_px * y / z
    nx = abs(u - cam.width / 2.0) / (cam.width / 2.0)
    ny = abs(v - cam.height / 2.0) / (cam.height / 2.0)
    inside = 0 <= u < cam.width and 0 <= v < cam.height
    edge_margin_px = min(u, v, cam.width - 1 - u, cam.height - 1 - v) if inside else -1.0
    score = max(0.0, 1.0 - max(nx, ny))
    off_nadir_deg = abs(float(cam.pitch_deg) + 90.0)
    return {
        "pixel_x": u,
        "pixel_y": v,
        "camera_z_m": z,
        "ground_dx_e_m": delta[0],
        "ground_dy_n_m": delta[1],
        "ground_dz_m": delta[2],
        "inside_image": inside,
        "edge_margin_px": edge_margin_px,
        "center_score": score,
        "off_nadir_deg": off_nadir_deg,
    }


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _font(size: int) -> ImageFont.ImageFont:
    for candidate in [r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\calibri.ttf"]:
        p = Path(candidate)
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def make_crop_sheet(
    out_path: Path,
    candidates: Sequence[Dict[str, Any]],
    crop_size: int,
    thumb_size: int,
    max_items: int,
) -> None:
    selected = list(candidates[:max_items])
    if not selected:
        return
    cols = min(4, len(selected))
    rows = math.ceil(len(selected) / cols)
    pad = 18
    label_h = 44
    title_h = 44
    w = cols * thumb_size + (cols + 1) * pad
    h = title_h + rows * (thumb_size + label_h) + (rows + 1) * pad
    sheet = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(sheet)
    font = _font(18)
    small = _font(14)
    title = f"{selected[0]['scene']}  {selected[0]['point_name']}  top projected candidates"
    draw.text((pad, 10), title, fill="black", font=font)
    half = crop_size // 2
    for idx, cand in enumerate(selected):
        r, c = divmod(idx, cols)
        x0 = pad + c * (thumb_size + pad)
        y0 = title_h + pad + r * (thumb_size + label_h + pad)
        img = Image.open(cand["image_path"]).convert("RGB")
        px, py = float(cand["pixel_x"]), float(cand["pixel_y"])
        left = int(round(px - half))
        top = int(round(py - half))
        crop = Image.new("RGB", (crop_size, crop_size), "black")
        src_box = (
            max(0, left),
            max(0, top),
            min(img.width, left + crop_size),
            min(img.height, top + crop_size),
        )
        paste_xy = (max(0, -left), max(0, -top))
        if src_box[2] > src_box[0] and src_box[3] > src_box[1]:
            crop.paste(img.crop(src_box), paste_xy)
        cx, cy = px - left, py - top
        cd = ImageDraw.Draw(crop)
        cd.line([(cx - 20, cy), (cx + 20, cy)], fill=(255, 230, 0), width=3)
        cd.line([(cx, cy - 20), (cx, cy + 20)], fill=(255, 230, 0), width=3)
        crop.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
        sheet.paste(crop, (x0, y0))
        label = f"{cand['rank_for_gcp']:02d} {cand['image_name']}  score={float(cand['center_score']):.2f}"
        draw.text((x0, y0 + thumb_size + 4), label, fill="black", font=small)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build coarse GCP projection candidates from DJI image metadata.")
    parser.add_argument("--scenes_root", default=r"E:\datasets\M3M-GCP\scenes")
    parser.add_argument(
        "--gcp_csv",
        default=str(REPO_ROOT / "evidence" / "gcp_coordinates" / "gcp_points_primary_usable_cgcs2000_cm108_20260615.csv"),
    )
    parser.add_argument("--out_root", default=str(REPO_ROOT / "outputs" / "gcp_annotation_candidates_20260616"))
    parser.add_argument("--exiftool", default="exiftool")
    parser.add_argument(
        "--scene",
        action="append",
        default=[],
        help="Optional scene id to process. Repeat for multiple scenes. Defaults to all scene directories.",
    )
    parser.add_argument("--max_candidates_per_gcp", type=int, default=12)
    parser.add_argument(
        "--max_off_nadir_deg",
        type=float,
        default=10.0,
        help=(
            "Only keep near-nadir images by default. DJI M3M flights often mix "
            "-90 deg orthographic images and -45 deg oblique images; oblique "
            "projections can include GCPs hidden behind structures and should "
            "not drive the first-pass manual annotation candidate list."
        ),
    )
    parser.add_argument("--crop_size", type=int, default=720)
    parser.add_argument("--thumb_size", type=int, default=360)
    parser.add_argument("--skip_contact_sheets", action="store_true")
    args = parser.parse_args()

    scenes_root = Path(args.scenes_root)
    out_root = Path(args.out_root)
    gcps = load_gcps(Path(args.gcp_csv))
    if args.scene:
        scene_dirs = [scenes_root / s for s in args.scene]
        missing = [str(p) for p in scene_dirs if not p.is_dir()]
        if missing:
            raise FileNotFoundError(f"Missing scene directories: {missing}")
    else:
        scene_dirs = sorted([p for p in scenes_root.iterdir() if p.is_dir()], key=lambda p: p.name)
    global_manifest: Dict[str, Any] = {
        "schema": "m3m_gcp_projection_candidates_v3",
        "scenes_root": str(scenes_root),
        "gcp_csv": str(Path(args.gcp_csv)),
        "candidate_projection_backend": "coarse DJI EXIF/Gimbal projection for manual annotation candidate discovery",
        "candidate_projection_warning": (
            "These candidates are not visibility proof and are not final GCP observations. "
            "Use contact sheets and manual annotation before quantitative GCP evaluation."
        ),
        "horizontal_crs": "CGCS2000 / 3-degree Gauss-Kruger CM 108E",
        "horizontal_crs_epsg": 4545,
        "vertical_frame_for_projection": (
            "DJI AbsoluteAltitude and WGS84 ellipsoid_height_m; "
            "1985 National Height Datum normal height is retained for surveyed deliverables."
        ),
        "max_off_nadir_deg": float(args.max_off_nadir_deg),
        "scene_count": len(scene_dirs),
        "gcp_count": len(gcps),
        "scenes": [],
    }
    candidate_fields = [
        "scene",
        "point_name",
        "point_category",
        "quality_evaluation",
        "image_name",
        "image_path",
        "rank_for_gcp",
        "pixel_x",
        "pixel_y",
        "inside_image",
        "edge_margin_px",
        "center_score",
        "camera_z_m",
        "ground_dx_e_m",
        "ground_dy_n_m",
        "ground_dz_m",
        "off_nadir_deg",
        "camera_cgcs2000_gk_cm108_e_m",
        "camera_cgcs2000_gk_cm108_n_m",
        "camera_wgs84_ellipsoid_alt_m",
        "camera_rel_alt_m",
        "camera_lat",
        "camera_lon",
        "yaw_deg",
        "pitch_deg",
        "roll_deg",
        "focal_px",
        "image_width",
        "image_height",
    ]
    summary_fields = [
        "scene",
        "point_name",
        "candidate_count",
        "best_image_name",
        "best_center_score",
        "best_edge_margin_px",
        "best_pixel_x",
        "best_pixel_y",
    ]
    for scene_dir in scene_dirs:
        scene = scene_dir.name
        out_dir = out_root / scene
        out_dir.mkdir(parents=True, exist_ok=True)
        images = sorted(scene_dir.glob("*_D.JPG"))
        exif_rows = run_exiftool(args.exiftool, images)
        cameras: List[CameraMeta] = []
        for row in exif_rows:
            meta = parse_camera_meta(row)
            if meta is not None:
                cameras.append(meta)
        metadata_fields = [
            "image_name",
            "image_path",
            "width",
            "height",
            "lat",
            "lon",
            "projected_e",
            "projected_n",
            "ellipsoid_alt_m",
            "rel_alt_m",
            "yaw_deg",
            "pitch_deg",
            "roll_deg",
            "focal_px",
        ]
        write_csv(out_dir / "image_metadata.csv", [c.__dict__ | {"image_path": str(c.image_path)} for c in cameras], metadata_fields)
        all_candidates: List[Dict[str, Any]] = []
        summary_rows: List[Dict[str, Any]] = []
        for gcp in gcps:
            projected: List[Dict[str, Any]] = []
            for cam in cameras:
                proj = project_gcp(cam, gcp)
                if not proj or not proj["inside_image"]:
                    continue
                if float(proj["off_nadir_deg"]) > float(args.max_off_nadir_deg):
                    continue
                projected.append(
                    {
                        "scene": scene,
                        "point_name": gcp["point_name"],
                        "point_category": gcp.get("point_category", ""),
                        "quality_evaluation": gcp.get("quality_evaluation", ""),
                        "image_name": cam.image_name,
                        "image_path": str(cam.image_path),
                        **proj,
                        "camera_cgcs2000_gk_cm108_e_m": cam.projected_e,
                        "camera_cgcs2000_gk_cm108_n_m": cam.projected_n,
                        "camera_wgs84_ellipsoid_alt_m": cam.ellipsoid_alt_m,
                        "camera_rel_alt_m": cam.rel_alt_m if cam.rel_alt_m is not None else "",
                        "camera_lat": cam.lat,
                        "camera_lon": cam.lon,
                        "yaw_deg": cam.yaw_deg,
                        "pitch_deg": cam.pitch_deg,
                        "roll_deg": cam.roll_deg,
                        "focal_px": cam.focal_px,
                        "image_width": cam.width,
                        "image_height": cam.height,
                    }
                )
            projected.sort(key=lambda r: (-float(r["center_score"]), -float(r["edge_margin_px"])))
            top = projected[: int(args.max_candidates_per_gcp)]
            for idx, row in enumerate(top, start=1):
                row["rank_for_gcp"] = idx
            all_candidates.extend(top)
            if top:
                best = top[0]
                summary_rows.append(
                    {
                        "scene": scene,
                        "point_name": gcp["point_name"],
                        "candidate_count": len(projected),
                        "best_image_name": best["image_name"],
                        "best_center_score": f"{float(best['center_score']):.6f}",
                        "best_edge_margin_px": f"{float(best['edge_margin_px']):.3f}",
                        "best_pixel_x": f"{float(best['pixel_x']):.3f}",
                        "best_pixel_y": f"{float(best['pixel_y']):.3f}",
                    }
                )
                if not args.skip_contact_sheets:
                    make_crop_sheet(
                        out_dir / "contact_sheets" / f"{gcp['point_name']}.jpg",
                        top,
                        crop_size=int(args.crop_size),
                        thumb_size=int(args.thumb_size),
                        max_items=int(args.max_candidates_per_gcp),
                    )
            else:
                summary_rows.append(
                    {
                        "scene": scene,
                        "point_name": gcp["point_name"],
                        "candidate_count": 0,
                        "best_image_name": "",
                        "best_center_score": "",
                        "best_edge_margin_px": "",
                        "best_pixel_x": "",
                        "best_pixel_y": "",
                    }
                )
        write_csv(out_dir / "gcp_projection_candidates.csv", all_candidates, candidate_fields)
        write_csv(out_dir / "gcp_visibility_summary.csv", summary_rows, summary_fields)
        scene_manifest = {
            "scene": scene,
            "image_count": len(images),
            "camera_metadata_count": len(cameras),
            "candidate_rows": len(all_candidates),
            "gcps_with_candidates": sum(1 for r in summary_rows if int(r["candidate_count"]) > 0),
            "outputs": {
                "image_metadata": str(out_dir / "image_metadata.csv"),
                "projection_candidates": str(out_dir / "gcp_projection_candidates.csv"),
                "visibility_summary": str(out_dir / "gcp_visibility_summary.csv"),
                "contact_sheets": str(out_dir / "contact_sheets"),
            },
        }
        (out_dir / "projection_manifest.json").write_text(json.dumps(scene_manifest, indent=2), encoding="utf-8")
        global_manifest["scenes"].append(scene_manifest)
        print(json.dumps(scene_manifest, ensure_ascii=False))
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "gcp_projection_candidates_manifest.json").write_text(
        json.dumps(global_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
