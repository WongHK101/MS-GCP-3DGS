#!/usr/bin/env python3
"""Build residual-blind RGB annotation tasks for the six TGS-GCP scenes.

The task generator is intentionally independent from any Gaussian model,
metric, or residual.  It uses surveyed GCP coordinates and DJI RGB EXIF pose
metadata for broad candidate discovery, then selects camera/strip-diverse
views.  The predicted pixel is only a search hint in the raw decoded JPEG
pixel matrix.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from build_gcp_projection_candidates import latlon_to_cgcs2000_gk_cm108


SCENES = {
    "gcp_3000_20260602": ("InternalRoad", "GCP-3K"),
    "gcp_5000_20260602": ("Garden", "GCP-5K"),
    "gcp_10000_20260610": ("Plaza", "GCP-10K"),
    "gcp_20000_20260602": ("Urban20K", "GCP-20K"),
    "gcp_50000_20260610": ("Urban50K", "GCP-50K"),
    "gcp_100000_20260610": ("Urban100K", "GCP-100K"),
}

KNOWN_VISIBLE_ANCHORS = {
    ("InternalRoad", "NC94"): "0085.jpg",
    ("Garden", "G04"): "0102.jpg",
    ("Plaza", "G25"): "0220.jpg",
    ("Urban20K", "G31"): "0187.jpg",
    ("Urban50K", "G39"): "0068.jpg",
    ("Urban100K", "G42"): "0103.jpg",
}

TASK_SCHEMA = "gs_gcp_tgs_rgb_outsourcing_candidate_v1"
ANNOTATION_SCHEMA = "gs_gcp_tgs_rgb_manual_image_observation_v1"
COORDINATE_DOMAIN = "raw_dji_decoded_pixel_matrix_ignore_exif_orientation"
COORDINATE_CONVENTION = "raw_image_zero_based_pixel_centers"
PROJECTION_METHOD = "dji_exif_gimbal_raw_rgb_coarse_projection_v2"
SELECTION_METHOD = "broad_projection_strip_azimuth_round_robin_v1"
TARGET_CANDIDATES_PER_POINT = 20
SEARCH_RADIUS_PX = 900.0
EXPANDED_IMAGE_MARGIN_PX = 900.0
FOCAL_35MM_SENSOR_WIDTH_MM = 36.0

CANDIDATE_FIELDS = [
    "schema",
    "annotation_schema",
    "task_id",
    "scene",
    "benchmark_scene",
    "dataset_scene_dir",
    "point_name",
    "point_role",
    "point_ellipsoid_height_source",
    "image_name",
    "image_path",
    "rank_for_gcp",
    "candidate_source",
    "selection_method",
    "projection_method",
    "pixel_x",
    "pixel_y",
    "projected_x",
    "projected_y",
    "projection_uncertainty_px",
    "search_radius_px",
    "inside_image",
    "edge_margin_px",
    "center_score",
    "ground_distance_m",
    "camera_z_m",
    "camera_azimuth_deg",
    "azimuth_bin_45deg",
    "flight_strip_id",
    "capture_order",
    "flight_yaw_deg",
    "gimbal_yaw_deg",
    "gimbal_pitch_deg",
    "gimbal_roll_deg",
    "source_image_width",
    "source_image_height",
    "source_image_sha256",
    "source_orientation_value",
    "orientation_policy",
    "focal_length_mm",
    "focal_length_35mm_equivalent_mm",
    "focal_px",
    "principal_point_x",
    "principal_point_y",
    "coordinate_domain",
    "coordinate_convention",
    "known_visible_anchor",
]


@dataclass(frozen=True)
class Point:
    name: str
    e: float
    n: float
    h_ellipsoid: float
    h_ellipsoid_source: str
    role: str


@dataclass
class Camera:
    scene: str
    dataset_scene_dir: str
    image_path: Path
    image_name: str
    capture_order: int
    width: int
    height: int
    orientation: int
    e: float
    n: float
    h_ellipsoid: float
    flight_yaw_deg: float
    gimbal_yaw_deg: float
    gimbal_pitch_deg: float
    gimbal_roll_deg: float
    focal_length_mm: float
    focal_35mm_mm: float
    focal_px: float
    strip_id: str = ""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def compact_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def wrap_degrees(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


def angular_difference_deg(a: float, b: float) -> float:
    return abs(wrap_degrees(a - b))


def extract_exif(exiftool: str, paths: list[Path], chunk_size: int = 180) -> list[dict[str, Any]]:
    fields = [
        "-n",
        "-ImageWidth",
        "-ImageHeight",
        "-Orientation",
        "-DateTimeOriginal",
        "-GPSLatitude",
        "-GPSLongitude",
        "-AbsoluteAltitude",
        "-RelativeAltitude",
        "-FlightYawDegree",
        "-GimbalYawDegree",
        "-GimbalPitchDegree",
        "-GimbalRollDegree",
        "-FocalLength",
        "-FocalLengthIn35mmFormat",
    ]
    output: list[dict[str, Any]] = []
    for start in range(0, len(paths), chunk_size):
        command = [exiftool, "-j", *fields, *map(str, paths[start : start + chunk_size])]
        proc = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output.extend(json.loads(proc.stdout))
    return output


def required_float(row: dict[str, Any], key: str, source: Path) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Missing or invalid {key} in {source}") from exc
    if not math.isfinite(value):
        raise RuntimeError(f"Non-finite {key} in {source}")
    return value


def build_cameras(scene: str, dataset_scene_dir: str, rows: list[dict[str, Any]]) -> list[Camera]:
    cameras: list[Camera] = []
    for row in rows:
        path = Path(row["SourceFile"])
        width = int(required_float(row, "ImageWidth", path))
        height = int(required_float(row, "ImageHeight", path))
        orientation = int(required_float(row, "Orientation", path))
        focal_35 = required_float(row, "FocalLengthIn35mmFormat", path)
        if orientation != 1:
            raise RuntimeError(f"{path}: EXIF orientation {orientation} is unsupported for this frozen task")
        if (width, height) != (4032, 3024):
            raise RuntimeError(f"{path}: unexpected RGB dimensions {width}x{height}")
        if abs(focal_35 - 24.0) > 1e-12:
            raise RuntimeError(f"{path}: unexpected 35mm-equivalent focal length {focal_35}")
        lat = required_float(row, "GPSLatitude", path)
        lon = required_float(row, "GPSLongitude", path)
        e, n = latlon_to_cgcs2000_gk_cm108(lat, lon)
        name_stem = path.stem
        if not name_stem.isdigit():
            raise RuntimeError(f"{path}: normalized RGB filename is not numeric")
        cameras.append(
            Camera(
                scene=scene,
                dataset_scene_dir=dataset_scene_dir,
                image_path=path,
                image_name=path.name,
                capture_order=int(name_stem),
                width=width,
                height=height,
                orientation=orientation,
                e=e,
                n=n,
                h_ellipsoid=required_float(row, "AbsoluteAltitude", path),
                flight_yaw_deg=required_float(row, "FlightYawDegree", path),
                gimbal_yaw_deg=required_float(row, "GimbalYawDegree", path),
                gimbal_pitch_deg=required_float(row, "GimbalPitchDegree", path),
                gimbal_roll_deg=required_float(row, "GimbalRollDegree", path),
                focal_length_mm=required_float(row, "FocalLength", path),
                focal_35mm_mm=focal_35,
                focal_px=width * focal_35 / FOCAL_35MM_SENSOR_WIDTH_MM,
            )
        )
    cameras.sort(key=lambda camera: (camera.capture_order, camera.image_name))
    if len({camera.image_name for camera in cameras}) != len(cameras):
        raise RuntimeError(f"{scene}: duplicate RGB image name")
    assign_flight_strips(cameras)
    return cameras


def assign_flight_strips(cameras: list[Camera]) -> None:
    distances = [
        math.hypot(current.e - previous.e, current.n - previous.n)
        for previous, current in zip(cameras, cameras[1:])
        if math.hypot(current.e - previous.e, current.n - previous.n) > 1e-6
    ]
    median_step = float(np.median(np.asarray(distances))) if distances else 0.0
    strip_index = 1
    previous: Camera | None = None
    for camera in cameras:
        if previous is not None:
            ground_step = math.hypot(camera.e - previous.e, camera.n - previous.n)
            sequence_gap = camera.capture_order - previous.capture_order
            yaw_change = angular_difference_deg(camera.flight_yaw_deg, previous.flight_yaw_deg)
            if sequence_gap != 1 or yaw_change > 30.0 or (median_step > 0 and ground_step > 4.0 * median_step):
                strip_index += 1
        camera.strip_id = f"strip_{strip_index:03d}"
        previous = camera


def camera_axes(camera: Camera) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yaw = math.radians(camera.gimbal_yaw_deg)
    pitch = math.radians(-camera.gimbal_pitch_deg)
    heading = np.asarray([math.sin(yaw), math.cos(yaw), 0.0], dtype=np.float64)
    forward = math.cos(pitch) * heading + math.sin(pitch) * np.asarray([0.0, 0.0, -1.0])
    forward /= np.linalg.norm(forward)
    right = np.asarray([math.cos(yaw), -math.sin(yaw), 0.0], dtype=np.float64)
    right /= np.linalg.norm(right)
    image_down = np.cross(forward, right)
    image_down /= np.linalg.norm(image_down)
    roll = math.radians(camera.gimbal_roll_deg)
    rolled_right = math.cos(roll) * right + math.sin(roll) * image_down
    rolled_down = -math.sin(roll) * right + math.cos(roll) * image_down
    return rolled_right, rolled_down, forward


def project(camera: Camera, point: Point) -> dict[str, float | bool] | None:
    right, image_down, forward = camera_axes(camera)
    delta = np.asarray([point.e - camera.e, point.n - camera.n, point.h_ellipsoid - camera.h_ellipsoid])
    z = float(np.dot(delta, forward))
    if z <= 1e-6:
        return None
    x = float(np.dot(delta, right))
    y = float(np.dot(delta, image_down))
    cx = camera.width / 2.0
    cy = camera.height / 2.0
    u = cx + camera.focal_px * x / z
    v = cy + camera.focal_px * y / z
    inside = 0.0 <= u < camera.width and 0.0 <= v < camera.height
    edge = min(u, v, camera.width - 1.0 - u, camera.height - 1.0 - v)
    center_score = max(0.0, 1.0 - max(abs(u - cx) / cx, abs(v - cy) / cy))
    azimuth = (math.degrees(math.atan2(camera.e - point.e, camera.n - point.n)) + 360.0) % 360.0
    return {
        "u": u,
        "v": v,
        "z": z,
        "inside": inside,
        "edge": edge,
        "center_score": center_score,
        "ground_distance": math.hypot(camera.e - point.e, camera.n - point.n),
        "camera_azimuth": azimuth,
    }


def candidate_pool(camera_rows: list[Camera], point: Point) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for camera in camera_rows:
        result = project(camera, point)
        if result is None:
            continue
        u, v = float(result["u"]), float(result["v"])
        broad_inside = (
            -EXPANDED_IMAGE_MARGIN_PX <= u < camera.width + EXPANDED_IMAGE_MARGIN_PX
            and -EXPANDED_IMAGE_MARGIN_PX <= v < camera.height + EXPANDED_IMAGE_MARGIN_PX
        )
        if not broad_inside:
            continue
        output.append(
            {
                "camera": camera,
                "pixel_x": u,
                "pixel_y": v,
                "inside_image": bool(result["inside"]),
                "edge_margin_px": float(result["edge"]),
                "center_score": float(result["center_score"]),
                "ground_distance_m": float(result["ground_distance"]),
                "camera_z_m": float(result["z"]),
                "camera_azimuth_deg": float(result["camera_azimuth"]),
                "azimuth_bin_45deg": int(float(result["camera_azimuth"]) // 45.0) % 8,
            }
        )
    return output


def select_diverse(pool: list[dict[str, Any]], count: int, mandatory_image: str | None) -> list[dict[str, Any]]:
    by_name = {row["camera"].image_name: row for row in pool}
    selected: list[dict[str, Any]] = []
    if mandatory_image:
        if mandatory_image not in by_name:
            raise RuntimeError(f"Known-visible anchor {mandatory_image} was not recalled by broad candidate discovery")
        selected.append(by_name.pop(mandatory_image))
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in by_name.values():
        key = (row["camera"].strip_id, int(row["azimuth_bin_45deg"]))
        groups.setdefault(key, []).append(row)
    for rows in groups.values():
        rows.sort(
            key=lambda row: (
                not bool(row["inside_image"]),
                -float(row["center_score"]),
                -float(row["edge_margin_px"]),
                float(row["ground_distance_m"]),
                row["camera"].image_name,
            )
        )
    group_counts: dict[tuple[str, int], int] = {key: 0 for key in groups}
    while len(selected) < count and any(groups.values()):
        available_keys = [key for key, rows in groups.items() if rows]
        key = min(
            available_keys,
            key=lambda item: (
                group_counts[item],
                not bool(groups[item][0]["inside_image"]),
                -float(groups[item][0]["center_score"]),
                item,
            ),
        )
        row = groups[key].pop(0)
        selected.append(row)
        group_counts[key] += 1
    return selected


def corrected_epoch_ellipsoid_heights(point_table: Path) -> dict[str, tuple[float, float, float, float]]:
    authoritative_dir = point_table.parent / "rtk_authoritative"
    candidates = sorted(authoritative_dir.glob("*\u6539\u6b63.csv"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one authoritative corrected epoch CSV, found {candidates}")
    values: dict[str, list[tuple[float, float, float, float]]] = {}
    with candidates[0].open("r", encoding="gb18030", newline="") as handle:
        for row in csv.DictReader(handle):
            epoch_name = row["\u70b9\u540d"]
            if "_" not in epoch_name:
                continue
            point_name = epoch_name.rsplit("_", 1)[0]
            values.setdefault(point_name, []).append(
                (
                    float(row["\u4e1c\u5750\u6807"]),
                    float(row["\u5317\u5750\u6807"]),
                    float(row["\u9ad8\u7a0b"]),
                    float(row["\u692d\u7403\u9ad8"]),
                )
            )
    return {
        point_name: tuple(float(value) for value in np.mean(np.asarray(rows, dtype=np.float64), axis=0))
        for point_name, rows in values.items()
    }


def load_points(point_table: Path, split_csv: Path) -> dict[str, list[Point]]:
    coordinates = {row["point_name"]: row for row in read_csv(point_table)}
    corrected_epochs = corrected_epoch_ellipsoid_heights(point_table)
    output: dict[str, list[Point]] = {scene: [] for scene in SCENES}
    seen: set[tuple[str, str]] = set()
    for row in read_csv(split_csv):
        key = (row["scene"], row["point_name"])
        if key in seen:
            raise RuntimeError(f"Duplicate scene-point split row: {key}")
        seen.add(key)
        coord = coordinates[row["point_name"]]
        ellipsoid_text = coord["wgs84_ellipsoid_height_m"].strip()
        if ellipsoid_text:
            ellipsoid_height = float(ellipsoid_text)
            ellipsoid_source = "frozen_v1_3_0_point_table"
        else:
            if row["point_name"] not in corrected_epochs:
                raise RuntimeError(f"{row['point_name']}: missing ellipsoid height and corrected epoch fallback")
            epoch_e, epoch_n, epoch_normal_h, ellipsoid_height = corrected_epochs[row["point_name"]]
            frozen = (
                float(coord["cgcs2000_gk_cm108_e_m"]),
                float(coord["cgcs2000_gk_cm108_n_m"]),
                float(coord["cgcs2000_normal_height_m"]),
            )
            if math.dist((epoch_e, epoch_n, epoch_normal_h), frozen) > 0.05:
                raise RuntimeError(f"{row['point_name']}: corrected epoch mean does not match frozen point table")
            ellipsoid_source = "authoritative_corrected_epoch_mean_fallback"
        output[row["scene"]].append(
            Point(
                name=row["point_name"],
                e=float(coord["cgcs2000_gk_cm108_e_m"]),
                n=float(coord["cgcs2000_gk_cm108_n_m"]),
                h_ellipsoid=ellipsoid_height,
                h_ellipsoid_source=ellipsoid_source,
                role=row["role"],
            )
        )
    for rows in output.values():
        rows.sort(key=lambda point: point.name.lower())
    return output


def candidate_record(
    point: Point,
    row: dict[str, Any],
    rank: int,
    benchmark_scene: str,
    source_sha: str,
    mandatory_image: str | None,
) -> dict[str, Any]:
    camera: Camera = row["camera"]
    relative_path = f"{camera.dataset_scene_dir}/rgb/{camera.image_name}"
    task_id = compact_hash([TASK_SCHEMA, camera.scene, point.name, camera.image_name, source_sha])
    return {
        "schema": TASK_SCHEMA,
        "annotation_schema": ANNOTATION_SCHEMA,
        "task_id": task_id,
        "scene": camera.scene,
        "benchmark_scene": benchmark_scene,
        "dataset_scene_dir": camera.dataset_scene_dir,
        "point_name": point.name,
        "point_role": point.role,
        "point_ellipsoid_height_source": point.h_ellipsoid_source,
        "image_name": camera.image_name,
        "image_path": relative_path,
        "rank_for_gcp": rank,
        "candidate_source": "known_visible_anchor+exif_broad_projection" if camera.image_name == mandatory_image else "exif_broad_projection",
        "selection_method": SELECTION_METHOD,
        "projection_method": PROJECTION_METHOD,
        "pixel_x": f"{float(row['pixel_x']):.9f}",
        "pixel_y": f"{float(row['pixel_y']):.9f}",
        "projected_x": f"{float(row['pixel_x']):.9f}",
        "projected_y": f"{float(row['pixel_y']):.9f}",
        "projection_uncertainty_px": f"{SEARCH_RADIUS_PX:.1f}",
        "search_radius_px": f"{SEARCH_RADIUS_PX:.1f}",
        "inside_image": str(bool(row["inside_image"])).lower(),
        "edge_margin_px": f"{float(row['edge_margin_px']):.6f}",
        "center_score": f"{float(row['center_score']):.9f}",
        "ground_distance_m": f"{float(row['ground_distance_m']):.6f}",
        "camera_z_m": f"{float(row['camera_z_m']):.6f}",
        "camera_azimuth_deg": f"{float(row['camera_azimuth_deg']):.6f}",
        "azimuth_bin_45deg": int(row["azimuth_bin_45deg"]),
        "flight_strip_id": camera.strip_id,
        "capture_order": camera.capture_order,
        "flight_yaw_deg": f"{camera.flight_yaw_deg:.6f}",
        "gimbal_yaw_deg": f"{camera.gimbal_yaw_deg:.6f}",
        "gimbal_pitch_deg": f"{camera.gimbal_pitch_deg:.6f}",
        "gimbal_roll_deg": f"{camera.gimbal_roll_deg:.6f}",
        "source_image_width": camera.width,
        "source_image_height": camera.height,
        "source_image_sha256": source_sha,
        "source_orientation_value": camera.orientation,
        "orientation_policy": "ignore_exif_orientation_no_transpose",
        "focal_length_mm": f"{camera.focal_length_mm:.6f}",
        "focal_length_35mm_equivalent_mm": f"{camera.focal_35mm_mm:.6f}",
        "focal_px": f"{camera.focal_px:.9f}",
        "principal_point_x": f"{camera.width / 2.0:.9f}",
        "principal_point_y": f"{camera.height / 2.0:.9f}",
        "coordinate_domain": COORDINATE_DOMAIN,
        "coordinate_convention": COORDINATE_CONVENTION,
        "known_visible_anchor": str(camera.image_name == mandatory_image).lower(),
    }


def git_text(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def build_readme(task_counts: dict[str, int]) -> str:
    counts = "\n".join(f"- {scene}: {count} 条候选" for scene, count in sorted(task_counts.items()))
    return f"""# TGS-GCP RGB 外包标注说明

## 1. 本批任务包含什么

本工具包只包含候选清单、标注程序和校验脚本，**不包含任何 RGB 或热红外图像**。
只标注 RGB。候选任务统计：

{counts}

## 2. 数据目录要求

请选择包含以下六个目录的 `TGS-GCP` 根目录；不要选择某一个 `rgb` 子目录：

```text
TGS-GCP/
  GCP-3K/rgb/0001.jpg ...
  GCP-5K/rgb/0001.jpg ...
  GCP-10K/rgb/0001.jpg ...
  GCP-20K/rgb/0001.jpg ...
  GCP-50K/rgb/0001.jpg ...
  GCP-100K/rgb/0001.jpg ...
```

`thermal` 不参与本批任务。首次使用先双击 `01_校验本地RGB数据.bat`，选择上述根目录并等待校验通过。

## 3. 开始标注

1. 双击 `02_启动标注工具.bat`。
2. 选择 `TGS-GCP` 根目录，填写标注人员代号。
3. 按场景打开任务。程序会把结果持续保存到 `results/`，可随时关闭后继续。

界面含义：

- 黄色十字和黄色圆：GPS/姿态给出的**粗搜索提示，不是真值**；允许存在数百像素偏差。
- 青色十字：你实际点击的位置。
- 鼠标左键：点击指定点名对应的像控点中心。
- 鼠标滚轮或 `+/-`：缩放；右键拖动：平移；`0`：复位。
- `1 Good`：点名能够唯一确认且中心可准确点击。
- `2 Ambiguous`：看到了疑似目标，但点名或中心不能可靠确认。
- `3 Not visible`：图中不可见、遮挡、模糊、出画或无法确认。
- `4/5`：上一张/下一张；`6`：保存。

**身份优先于几何位置。** 同一画面可能出现多个红色 L 形像控标志，必须看清地面手写点名；不要因为黄色准心附近有另一个像控点就点击它。看不清点名时选 Ambiguous 或 Not visible，不要猜。

Good/Ambiguous 必须先点击坐标；Not visible 会自动清空坐标。所有坐标均保存为原始 RGB JPEG 解码矩阵中的 zero-based pixel-center 坐标。

## 4. 完成后回传

六场景全部完成后，双击 `03_生成最小回传包.bat`。脚本只有在以下条件全部满足时才会生成 ZIP：

- 每条候选均已选择 Good、Ambiguous 或 Not visible；
- Good/Ambiguous 均有图内点击坐标；
- Not visible 无坐标；
- task ID、图像 SHA、点名和图片名与任务清单一致。

请只回传新生成的 `return/TGS_GCP_RGB_ANNOTATIONS_MINIMAL_*.zip`。不需要回传图像、整个工具包或本地数据。

## 5. 冻结原则

- 本批只使用 surveyed point、RGB EXIF/GPS/姿态和空间覆盖选图，不读取 3DGS residual、RMSE、depth、alpha 或 variance。
- 候选按航带和相机中心方位做多样化抽样，避免只选连续近重复图。
- 人工标注域始终是 raw RGB；不得在 thermal、CFR crop、undistorted render 或低分辨率 packet 上代替标注。
"""


def copy_runtime_files(repo: Path, task_root: Path) -> None:
    tool_dir = task_root / "tool"
    tool_dir.mkdir(parents=True, exist_ok=True)
    for name in ["manual_gcp_annotator.py", "tgs_gcp_outsourcing_runtime.py"]:
        shutil.copy2(repo / "code" / "gcp" / name, tool_dir / name)
    (task_root / "requirements.txt").write_text("Pillow==11.1.0\n", encoding="ascii")
    batch_files = {
        "00_安装依赖.bat": '@echo off\r\ncd /d "%~dp0"\r\npython -m pip install -r requirements.txt\r\npause\r\n',
        "01_校验本地RGB数据.bat": '@echo off\r\ncd /d "%~dp0"\r\npython tool\\tgs_gcp_outsourcing_runtime.py verify\r\npause\r\n',
        "02_启动标注工具.bat": '@echo off\r\ncd /d "%~dp0"\r\npython tool\\tgs_gcp_outsourcing_runtime.py launch\r\nif errorlevel 1 pause\r\n',
        "03_生成最小回传包.bat": '@echo off\r\ncd /d "%~dp0"\r\npython tool\\tgs_gcp_outsourcing_runtime.py pack\r\npause\r\n',
    }
    for name, content in batch_files.items():
        (task_root / name).write_text(content, encoding="utf-8-sig", newline="")


def package_directory(task_root: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        raise FileExistsError(zip_path)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(task_root.rglob("*"), key=lambda item: item.relative_to(task_root).as_posix().encode("utf-8")):
            if path.is_file():
                archive.write(path, path.relative_to(task_root).as_posix())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", type=Path, default=Path(r"E:\datasets\TGS-GCP"))
    parser.add_argument(
        "--point_table",
        type=Path,
        default=Path(r"E:\datasets\M3M-GCP\scenes\gcp_manual_annotations_v1_3_0\gcp_points_cgcs2000_cm108_v1_3_0.csv"),
    )
    parser.add_argument(
        "--split_csv",
        type=Path,
        default=Path(r"E:\datasets\M3M-GCP\scenes\gcp_manual_annotations_v1_3_0\gcp_control_checkpoint_split_v1_3_0.csv"),
    )
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--package_zip", type=Path, required=True)
    parser.add_argument("--exiftool", default="exiftool")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    if git_text(repo, "status", "--porcelain"):
        raise RuntimeError("Task generation requires a clean committed worktree")
    if args.output_root.exists() or args.package_zip.exists():
        raise FileExistsError("Output root or package ZIP already exists")
    args.output_root.mkdir(parents=True)
    candidate_dir = args.output_root / "candidate_lists"
    candidate_dir.mkdir()
    (args.output_root / "results").mkdir()
    (args.output_root / "return").mkdir()

    points = load_points(args.point_table, args.split_csv)
    all_candidates: list[dict[str, Any]] = []
    scene_summaries: list[dict[str, Any]] = []
    all_image_count = 0
    all_selected_images: set[Path] = set()
    exif_audit: list[dict[str, Any]] = []

    for benchmark_scene, (scene, dataset_scene_dir) in SCENES.items():
        rgb_dir = args.dataset_root / dataset_scene_dir / "rgb"
        image_paths = sorted(rgb_dir.glob("*.jpg"), key=lambda path: (int(path.stem), path.name))
        if not image_paths:
            raise RuntimeError(f"No RGB images found: {rgb_dir}")
        all_image_count += len(image_paths)
        cameras = build_cameras(scene, dataset_scene_dir, extract_exif(args.exiftool, image_paths))
        selected_scene: list[dict[str, Any]] = []
        point_summaries: list[dict[str, Any]] = []
        for point in points[benchmark_scene]:
            pool = candidate_pool(cameras, point)
            anchor = KNOWN_VISIBLE_ANCHORS.get((scene, point.name))
            selected = select_diverse(pool, TARGET_CANDIDATES_PER_POINT, anchor)
            if len(selected) < 8:
                raise RuntimeError(f"{scene}/{point.name}: only {len(selected)} broad candidates")
            selected_paths = {row["camera"].image_path for row in selected}
            all_selected_images.update(selected_paths)
            hashes = {path: sha256_file(path) for path in selected_paths}
            records = [
                candidate_record(point, row, rank, benchmark_scene, hashes[row["camera"].image_path], anchor)
                for rank, row in enumerate(selected, 1)
            ]
            selected_scene.extend(records)
            point_summaries.append(
                {
                    "scene": scene,
                    "benchmark_scene": benchmark_scene,
                    "point_name": point.name,
                    "point_role": point.role,
                    "broad_candidate_count": len(pool),
                    "selected_candidate_count": len(records),
                    "selected_unique_strip_count": len({row["flight_strip_id"] for row in records}),
                    "selected_azimuth_bin_count": len({row["azimuth_bin_45deg"] for row in records}),
                    "known_visible_anchor": anchor or "",
                    "known_visible_anchor_recalled": str(not anchor or any(row["image_name"] == anchor for row in records)).lower(),
                }
            )
        selected_scene.sort(key=lambda row: (row["point_name"].lower(), int(row["rank_for_gcp"]), row["image_name"]))
        if len({row["task_id"] for row in selected_scene}) != len(selected_scene):
            raise RuntimeError(f"{scene}: duplicate task ID")
        path = candidate_dir / f"{scene}_rgb_candidates.csv"
        write_csv(path, selected_scene, CANDIDATE_FIELDS)
        all_candidates.extend(selected_scene)
        scene_summaries.extend(point_summaries)
        exif_audit.append(
            {
                "scene": scene,
                "dataset_scene_dir": dataset_scene_dir,
                "rgb_image_count": len(cameras),
                "unique_strip_count": len({camera.strip_id for camera in cameras}),
                "widths": sorted({camera.width for camera in cameras}),
                "heights": sorted({camera.height for camera in cameras}),
                "orientations": sorted({camera.orientation for camera in cameras}),
                "focal_35mm_values": sorted({camera.focal_35mm_mm for camera in cameras}),
                "gimbal_pitch_range": [min(camera.gimbal_pitch_deg for camera in cameras), max(camera.gimbal_pitch_deg for camera in cameras)],
                "gimbal_roll_values": sorted({camera.gimbal_roll_deg for camera in cameras}),
            }
        )

    write_csv(args.output_root / "point_task_summary.csv", scene_summaries, list(scene_summaries[0]))
    task_counts: dict[str, int] = {}
    for row in all_candidates:
        task_counts[row["scene"]] = task_counts.get(row["scene"], 0) + 1
    (args.output_root / "README_外包标注说明.md").write_text(build_readme(task_counts), encoding="utf-8")
    copy_runtime_files(repo, args.output_root)

    candidate_hashes = []
    for path in sorted(candidate_dir.glob("*.csv"), key=lambda item: item.name.encode("utf-8")):
        candidate_hashes.append({"path": path.relative_to(args.output_root).as_posix(), "size": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = {
        "schema": "gs_gcp_tgs_rgb_outsourcing_package_v1",
        "status": "annotation_working_package_not_release",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "generator": {
            "commit": git_text(repo, "rev-parse", "HEAD"),
            "branch": git_text(repo, "branch", "--show-current"),
            "clean": True,
            "script": "code/gcp/prepare_tgs_gcp_rgb_outsourcing.py",
            "script_sha256": sha256_file(Path(__file__)),
        },
        "inputs": {
            "dataset_root": str(args.dataset_root),
            "point_table_sha256": sha256_file(args.point_table),
            "split_csv_sha256": sha256_file(args.split_csv),
            "rgb_image_count": all_image_count,
            "selected_unique_rgb_image_count": len(all_selected_images),
            "thermal_image_count_used": 0,
        },
        "protocol": {
            "coordinate_domain": COORDINATE_DOMAIN,
            "coordinate_convention": COORDINATE_CONVENTION,
            "projection_method": PROJECTION_METHOD,
            "selection_method": SELECTION_METHOD,
            "target_candidates_per_point": TARGET_CANDIDATES_PER_POINT,
            "search_radius_px": SEARCH_RADIUS_PX,
            "model_residual_used": False,
            "thermal_used": False,
        },
        "scene_task_counts": task_counts,
        "total_task_count": len(all_candidates),
        "unique_task_id_count": len({row["task_id"] for row in all_candidates}),
        "known_visible_anchor_count": sum(row["known_visible_anchor"] == "true" for row in all_candidates),
        "exif_audit": exif_audit,
        "candidate_files": candidate_hashes,
    }
    manifest_path = args.output_root / "task_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    payload_rows = []
    excluded = {"FILE_SHA256SUMS.csv"}
    for path in sorted(args.output_root.rglob("*"), key=lambda item: item.relative_to(args.output_root).as_posix().encode("utf-8")):
        if path.is_file() and path.name not in excluded:
            payload_rows.append(
                {"path": path.relative_to(args.output_root).as_posix(), "size": path.stat().st_size, "sha256": sha256_file(path)}
            )
    write_csv(args.output_root / "FILE_SHA256SUMS.csv", payload_rows, ["path", "size", "sha256"])
    package_directory(args.output_root, args.package_zip)
    (args.package_zip.with_suffix(args.package_zip.suffix + ".sha256")).write_text(
        f"{sha256_file(args.package_zip)}  {args.package_zip.name}\n", encoding="ascii"
    )
    print(json.dumps({"task_root": str(args.output_root), "package_zip": str(args.package_zip), **manifest["inputs"], **task_counts}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
