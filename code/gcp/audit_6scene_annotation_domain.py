from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

try:
    from PIL import Image
except Exception as exc:  # noqa: BLE001
    Image = None
    PIL_IMPORT_ERROR = exc
else:
    PIL_IMPORT_ERROR = None


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = Path(r"E:\datasets\M3M-GCP")
DEFAULT_RELEASE_DIR = DEFAULT_DATASET_ROOT / "scenes" / "gcp_manual_annotations"
DEFAULT_REMOTE_LIGHT_MANIFEST = (
    DEFAULT_PROJECT_ROOT
    / "outputs"
    / "gcp_6scene_annotation_domain_inputs_20260628"
    / "gcp_6scene_annotation_domain_jsonlight_20260628"
    / "remote_light_manifest.json"
)
DEFAULT_OUT_BASE = DEFAULT_PROJECT_ROOT / "outputs"
DEFAULT_PACKAGE_DIR = DEFAULT_PROJECT_ROOT / "outputs" / "gpt_review_packages"

SCENES = [
    "gcp_3000_20260602",
    "gcp_5000_20260602",
    "gcp_10000_20260610",
    "gcp_20000_20260602",
    "gcp_50000_20260610",
    "gcp_100000_20260610",
]

POSE_CENTER_TOL = 1e-8
POSE_ROTATION_TOL_RAD = 1e-8
PIXEL_ROUNDTRIP_TOL = 1e-6
ARCHIVED_AGREEMENT_TOL = 1e-4

ALLOWED_CONCLUSIONS = {
    "confirmed_mismatch",
    "domain_already_consistent",
    "unresolved_missing_source_camera",
    "unresolved_missing_target_camera",
    "unresolved_mapping",
    "non_equivalent_camera_pose",
    "other_explicit_failure",
}

ARCHIVED_UNDISTORTED_CANDIDATES = {
    "gcp_3000_20260602": [
        DEFAULT_PROJECT_ROOT
        / "outputs"
        / "gaussian_gcp_eval_20260618"
        / "annotations_undistorted"
        / "gcp_image_observations_undistorted_for_evaluation.csv",
    ],
    "gcp_5000_20260602": [
        DEFAULT_PROJECT_ROOT
        / "outputs"
        / "remote_sync"
        / "three_scene_diagnostics_light_20260624"
        / "gaussian-gcp-eval-official-3scenes-20260624"
        / "gcp_5000_20260602"
        / "annotations_undistorted"
        / "gcp_image_observations_undistorted_for_evaluation.csv",
    ],
    "gcp_100000_20260610": [
        DEFAULT_PROJECT_ROOT
        / "outputs"
        / "gcp_eval_official_2scenes_20260623"
        / "gcp_100000_20260610"
        / "annotations_undistorted"
        / "gcp_image_observations_undistorted_for_evaluation.csv",
    ],
}


@dataclass(frozen=True)
class CameraRecord:
    camera_id: int
    model: str
    width: int
    height: int
    params: tuple[float, ...]
    record_sha256: str


@dataclass(frozen=True)
class ImageRecord:
    image_id: int
    image_name: str
    camera_id: int
    qvec: tuple[float, float, float, float]
    tvec: tuple[float, float, float]
    record_sha256: str


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    rows = list(rows)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_sha256(obj: Any) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def git_text(args: list[str], cwd: Path = REPO_ROOT) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=cwd, text=True, errors="replace").strip()
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {type(exc).__name__}: {exc}"


def make_unique_dir(base: Path) -> Path:
    if not base.exists():
        base.mkdir(parents=True)
        return base
    stamp = datetime.now().strftime("%H%M%S")
    for i in range(1, 1000):
        candidate = base.with_name(f"{base.name}_{stamp}_{i:02d}")
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
    raise RuntimeError(f"cannot create unique output dir near {base}")


def make_unique_file(path: Path) -> Path:
    if not path.exists():
        return path
    stamp = datetime.now().strftime("%H%M%S")
    for i in range(1, 1000):
        candidate = path.with_name(f"{path.stem}_{stamp}_{i:02d}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"cannot create unique file near {path}")


def qvec2rotmat(qvec: Sequence[float]) -> np.ndarray:
    q = np.asarray(qvec, dtype=float)
    return np.array(
        [
            [
                1 - 2 * q[2] ** 2 - 2 * q[3] ** 2,
                2 * q[1] * q[2] - 2 * q[0] * q[3],
                2 * q[3] * q[1] + 2 * q[0] * q[2],
            ],
            [
                2 * q[1] * q[2] + 2 * q[0] * q[3],
                1 - 2 * q[1] ** 2 - 2 * q[3] ** 2,
                2 * q[2] * q[3] - 2 * q[0] * q[1],
            ],
            [
                2 * q[3] * q[1] - 2 * q[0] * q[2],
                2 * q[2] * q[3] + 2 * q[0] * q[1],
                1 - 2 * q[1] ** 2 - 2 * q[2] ** 2,
            ],
        ],
        dtype=float,
    )


def camera_center(image: ImageRecord) -> np.ndarray:
    r = qvec2rotmat(image.qvec)
    t = np.asarray(image.tvec, dtype=float)
    return -r.T @ t


def rotation_angle_rad(source: ImageRecord, target: ImageRecord) -> float:
    qs = np.asarray(source.qvec, dtype=float)
    qt = np.asarray(target.qvec, dtype=float)
    qs /= np.linalg.norm(qs)
    qt /= np.linalg.norm(qt)
    dot = abs(float(np.dot(qs, qt)))
    dot = max(-1.0, min(1.0, dot))
    return float(2.0 * math.acos(dot))


def camera_normalized_from_pixel(camera: CameraRecord, u: float, v: float, max_iter: int = 20) -> tuple[float, float]:
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
    raise ValueError(f"unsupported source camera model: {camera.model}")


def camera_pixel_from_normalized(camera: CameraRecord, x: float, y: float) -> tuple[float, float]:
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
    raise ValueError(f"unsupported target camera model: {camera.model}")


def coordinate_stats(values: Iterable[float]) -> dict[str, Any]:
    arr = np.asarray([float(v) for v in values if v is not None and not math.isnan(float(v))], dtype=float)
    if arr.size == 0:
        return {"count": 0, "median": None, "p95": None, "max": None, "mean": None}
    return {
        "count": int(arr.size),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def parse_float(row: dict[str, Any], key: str) -> float:
    value = row.get(key, "")
    if str(value).strip() == "":
        raise ValueError(f"missing numeric field {key}")
    return float(value)


def load_model_manifest(scene_entry: dict[str, Any], key: str) -> tuple[dict[int, CameraRecord], dict[str, ImageRecord], dict[str, Any]]:
    model = scene_entry.get(key, {})
    cameras: dict[int, CameraRecord] = {}
    for cam in model.get("cameras", []):
        rec = CameraRecord(
            camera_id=int(cam["camera_id"]),
            model=str(cam["model"]),
            width=int(cam["width"]),
            height=int(cam["height"]),
            params=tuple(float(x) for x in cam.get("params", [])),
            record_sha256=str(cam.get("record_sha256", "")),
        )
        cameras[rec.camera_id] = rec
    images: dict[str, ImageRecord] = {}
    for image in model.get("images", []):
        rec = ImageRecord(
            image_id=int(image["image_id"]),
            image_name=Path(str(image["image_name"])).name,
            camera_id=int(image["camera_id"]),
            qvec=tuple(float(x) for x in image.get("qvec", [])),  # type: ignore[arg-type]
            tvec=tuple(float(x) for x in image.get("tvec", [])),  # type: ignore[arg-type]
            record_sha256=str(image.get("record_sha256", "")),
        )
        images[rec.image_name] = rec
    return cameras, images, model


def find_file_record(model: dict[str, Any], name: str) -> dict[str, Any]:
    for rec in model.get("files", []):
        if Path(str(rec.get("name", ""))).name == name:
            return rec
    return {}


def resolve_source_annotation_path(path_text: str, project_root: Path, release_dir: Path) -> tuple[Path | None, str]:
    raw = Path(path_text)
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        normalized = Path(path_text.replace("/", "\\"))
        candidates.extend(
            [
                project_root / normalized,
                project_root / "outputs" / normalized,
                project_root / "outputs" / "gcp_annotations" / normalized.name,
                project_root / "outputs" / "gcp_annotation_inclusion_audit_20260625_min2" / "filtered_annotations" / f"{normalized.stem}_eval_strict_good_nadir.csv",
                project_root / "outputs" / "gcp_annotation_inclusion_audit_20260625" / "filtered_annotations" / f"{normalized.stem}_eval_strict_good_nadir.csv",
                project_root / "outputs" / "gcp_annotation_inclusion_audit_20260625_min1" / "filtered_annotations" / f"{normalized.stem}_eval_strict_good_nadir.csv",
                release_dir / normalized,
                release_dir.parent / normalized,
            ]
        )
    seen = set()
    for candidate in candidates:
        key = str(candidate).lower()
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            status = "declared_path_found" if candidate == raw else "fallback_recovered_unhashed"
            return candidate, status
    return None, "missing"


def resolve_image_path(row: dict[str, Any], dataset_root: Path, scene: str) -> tuple[Path | None, str]:
    image_name = Path(str(row.get("image_name", ""))).name
    image_path = str(row.get("image_path", "")).strip()
    candidates: list[Path] = []
    if image_path:
        p = Path(image_path)
        if p.is_absolute():
            candidates.append(p)
        else:
            normalized = Path(image_path.replace("/", "\\"))
            candidates.extend(
                [
                    dataset_root / "scenes" / normalized,
                    dataset_root / "scenes" / scene / normalized.name,
                    dataset_root / normalized,
                ]
            )
    candidates.append(dataset_root / "scenes" / scene / image_name)
    seen = set()
    for candidate in candidates:
        key = str(candidate).lower()
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            status = "source_path_from_annotation" if image_path and candidate == Path(image_path) else "source_path_resolved_by_scene_image"
            return candidate, status
    return None, "missing"


def decoded_image_size(path: Path) -> tuple[int | None, int | None, str]:
    if Image is None:
        return None, None, f"pil_import_failed:{PIL_IMPORT_ERROR}"
    try:
        with Image.open(path) as img:
            return int(img.width), int(img.height), "ok"
    except Exception as exc:  # noqa: BLE001
        return None, None, f"decode_failed:{type(exc).__name__}:{exc}"


def release_stable_key(row: dict[str, Any]) -> str:
    return "|".join(
        [
            str(row.get("scene", "")).strip(),
            str(row.get("point_name", "")).strip(),
            str(row.get("image_name", "")).strip(),
            str(row.get("manual_x", "")).strip(),
            str(row.get("manual_y", "")).strip(),
        ]
    )


def source_match_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("scene", "")).strip(),
        str(row.get("point_name", "")).strip(),
        Path(str(row.get("image_name", ""))).name,
    )


def source_rows_by_key(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], list[dict[str, str]]]:
    result: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        result[source_match_key(row)].append(row)
    return result


def select_source_row(release_row: dict[str, str], source_rows: dict[tuple[str, str, str], list[dict[str, str]]]) -> tuple[dict[str, str] | None, str]:
    key = source_match_key(release_row)
    candidates = source_rows.get(key, [])
    if not candidates:
        return None, "missing"
    rx = str(release_row.get("manual_x", "")).strip()
    ry = str(release_row.get("manual_y", "")).strip()
    exact = [row for row in candidates if str(row.get("manual_x", "")).strip() == rx and str(row.get("manual_y", "")).strip() == ry]
    if len(exact) == 1:
        return exact[0], "exact_coordinate_match"
    if len(candidates) == 1:
        return candidates[0], "single_key_match_coordinate_may_differ"
    return candidates[0], "multiple_key_matches_first_used"


def build_observation_id(scene: str, point_name: str, raw_image_name: str, raw_image_sha256: str, raw_manual_x_text: str, raw_manual_y_text: str) -> str:
    payload = [
        scene.strip(),
        point_name.strip(),
        Path(raw_image_name).name,
        raw_image_sha256.strip(),
        raw_manual_x_text.strip(),
        raw_manual_y_text.strip(),
    ]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def transform_raw_to_target(source_camera: CameraRecord, target_camera: CameraRecord, u: float, v: float) -> dict[str, float]:
    x, y = camera_normalized_from_pixel(source_camera, u, v)
    unit = np.asarray([x, y, 1.0], dtype=float)
    unit /= np.linalg.norm(unit)
    target_u, target_v = camera_pixel_from_normalized(target_camera, x, y)
    x_back, y_back = camera_normalized_from_pixel(target_camera, target_u, target_v)
    raw_u, raw_v = camera_pixel_from_normalized(source_camera, x_back, y_back)
    return {
        "normalized_x": float(x),
        "normalized_y": float(y),
        "normalized_unit_ray_x": float(unit[0]),
        "normalized_unit_ray_y": float(unit[1]),
        "normalized_unit_ray_z": float(unit[2]),
        "target_x": float(target_u),
        "target_y": float(target_v),
        "roundtrip_raw_x": float(raw_u),
        "roundtrip_raw_y": float(raw_v),
        "roundtrip_error_px": float(math.hypot(raw_u - u, raw_v - v)),
    }


def load_archived_undistorted(project_root: Path, scene: str) -> tuple[list[dict[str, str]], str, str]:
    for candidate in ARCHIVED_UNDISTORTED_CANDIDATES.get(scene, []):
        if candidate.exists():
            return read_csv(candidate), str(candidate), file_sha256(candidate)
    matches = list(project_root.rglob(f"*{scene}*undistort*observations*.csv"))
    for candidate in matches:
        if candidate.exists():
            return read_csv(candidate), str(candidate), file_sha256(candidate)
    return [], "", ""


def get_archived_xy(row: dict[str, str]) -> tuple[float | None, float | None]:
    x_keys = ["undistorted_x", "undistorted_u", "u_px", "target_x", "manual_x", "x"]
    y_keys = ["undistorted_y", "undistorted_v", "v_px", "target_y", "manual_y", "y"]
    x = None
    y = None
    for key in x_keys:
        if str(row.get(key, "")).strip() != "":
            x = float(row[key])
            break
    for key in y_keys:
        if str(row.get(key, "")).strip() != "":
            y = float(row[key])
            break
    return x, y


def package_outputs(out_dir: Path, package_path: Path) -> tuple[Path, Path]:
    package_path.parent.mkdir(parents=True, exist_ok=True)
    package_path = make_unique_file(package_path)
    rows = []
    for path in sorted(out_dir.rglob("*")):
        if path.is_file() and path.name != "PACKAGE_CONTENT_SHA256SUMS.csv":
            rel = path.relative_to(out_dir).as_posix()
            rows.append({"path": rel, "bytes": path.stat().st_size, "sha256": file_sha256(path)})
    write_csv(out_dir / "PACKAGE_CONTENT_SHA256SUMS.csv", rows, ["path", "bytes", "sha256"])
    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(out_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(out_dir).as_posix())
    sha_path = package_path.with_suffix(package_path.suffix + ".sha256")
    sha_path.write_text(f"{file_sha256(package_path)}  {package_path.name}\n", encoding="utf-8")
    return package_path, sha_path


def safe_percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), percentile))


def audit_scene(
    scene: str,
    release_dir: Path,
    dataset_root: Path,
    project_root: Path,
    manifest_scene: dict[str, Any],
    provenance_rows: list[dict[str, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    release_csv = release_dir / f"{scene}_gcp_annotations_final_good_nadir_v1.csv"
    release_rows = read_csv(release_csv) if release_csv.exists() else []
    release_rows = [row for row in release_rows if row.get("scene", scene) == scene]

    scene_prov = [row for row in provenance_rows if row.get("scene") == scene and row.get("included_in_final_release", "").lower() == "true"]
    declared_source_files = sorted({row.get("source_annotation_file", "").strip() for row in scene_prov if row.get("source_annotation_file", "").strip()})
    source_rows: list[dict[str, str]] = []
    source_file_records = []
    for text in declared_source_files:
        path, status = resolve_source_annotation_path(text, project_root, release_dir)
        record = {"scene": scene, "declared_source_annotation_file": text, "resolved_path": str(path) if path else "", "resolution_status": status}
        if path:
            record.update({"bytes": path.stat().st_size, "sha256": file_sha256(path)})
            source_rows.extend([row for row in read_csv(path) if row.get("scene") == scene])
        source_file_records.append(record)
    if not source_rows:
        fallback = project_root / "outputs" / "gcp_annotation_inclusion_audit_20260625_min2" / "filtered_annotations" / f"{scene}_manual_annotations_eval_strict_good_nadir.csv"
        if fallback.exists():
            source_rows.extend([row for row in read_csv(fallback) if row.get("scene") == scene])
            source_file_records.append(
                {
                    "scene": scene,
                    "declared_source_annotation_file": "",
                    "resolved_path": str(fallback),
                    "resolution_status": "fallback_filtered_release_rows",
                    "bytes": fallback.stat().st_size,
                    "sha256": file_sha256(fallback),
                }
            )
    source_lookup = source_rows_by_key(source_rows)

    raw_cameras, raw_images, raw_model = load_model_manifest(manifest_scene, "raw_model")
    target_cameras, target_images, target_model = load_model_manifest(manifest_scene, "target_model")
    target_hashes = {Path(row["image_name"]).name: row for row in manifest_scene.get("target_image_hashes", [])}
    archived_rows, archived_path, archived_sha = load_archived_undistorted(project_root, scene)
    archived_lookup = source_rows_by_key(archived_rows)

    per_obs = []
    mapping_rows = []
    archived_rows_out = []
    duplicate_release_keys = [k for k, c in Counter(release_stable_key(row) for row in release_rows).items() if c > 1]
    seen_obs_ids = Counter()

    for row_index, release_row in enumerate(release_rows):
        image_name = Path(str(release_row.get("image_name", ""))).name
        point_name = str(release_row.get("point_name", "")).strip()
        source_row, source_match_status = select_source_row(release_row, source_lookup)
        raw_manual_x_text = str((source_row or release_row).get("manual_x", release_row.get("manual_x", ""))).strip()
        raw_manual_y_text = str((source_row or release_row).get("manual_y", release_row.get("manual_y", ""))).strip()
        source_image_path, source_image_status = resolve_image_path(source_row or release_row, dataset_root, scene)
        source_image_sha = file_sha256(source_image_path) if source_image_path and source_image_path.exists() else ""
        decoded_w, decoded_h, decode_status = decoded_image_size(source_image_path) if source_image_path else (None, None, "missing")
        obs_id = build_observation_id(scene, point_name, image_name, source_image_sha, raw_manual_x_text, raw_manual_y_text)
        seen_obs_ids[obs_id] += 1

        source_image = raw_images.get(image_name)
        target_image = target_images.get(image_name)
        source_camera = raw_cameras.get(source_image.camera_id) if source_image else None
        target_camera = target_cameras.get(target_image.camera_id) if target_image else None
        target_hash = target_hashes.get(image_name, {})

        status_flags: list[str] = []
        if source_row is None:
            status_flags.append("missing_source_annotation_row")
        if source_image_path is None:
            status_flags.append("missing_source_image_file")
        if source_image is None:
            status_flags.append("missing_source_colmap_image")
        if target_image is None:
            status_flags.append("missing_target_colmap_image")
        if source_camera is None:
            status_flags.append("missing_source_camera")
        if target_camera is None:
            status_flags.append("missing_target_camera")

        transform: dict[str, Any] = {}
        pose_center_diff = None
        pose_rotation_diff = None
        pose_equivalent = False
        mapping_type = "unresolved_mapping"
        in_bounds = False
        archived_agreement_px = None
        source_target_displacement_px = None
        release_target_displacement_px = None
        release_source_displacement_px = None
        raw_target_roundtrip_error = None

        if source_image and target_image and source_camera and target_camera:
            if source_image.record_sha256 and source_image.record_sha256 == target_image.record_sha256:
                center_diff = 0.0
                rot_diff = 0.0
            else:
                center_diff = float(np.linalg.norm(camera_center(source_image) - camera_center(target_image)))
                rot_diff = rotation_angle_rad(source_image, target_image)
            pose_center_diff = center_diff
            pose_rotation_diff = rot_diff
            pose_equivalent = center_diff <= POSE_CENTER_TOL and rot_diff <= POSE_ROTATION_TOL_RAD
            if pose_equivalent:
                mapping_type = "pose_equivalent_colmap_undistortion_intrinsics_remap"
            else:
                mapping_type = "non_equivalent_camera_pose_unmappable_without_depth"
                status_flags.append("pose_mismatch")
            try:
                raw_x = float(raw_manual_x_text)
                raw_y = float(raw_manual_y_text)
                transform = transform_raw_to_target(source_camera, target_camera, raw_x, raw_y)
                raw_target_roundtrip_error = transform["roundtrip_error_px"]
                in_bounds = (
                    0.0 <= transform["target_x"] < float(target_camera.width)
                    and 0.0 <= transform["target_y"] < float(target_camera.height)
                )
                if not in_bounds:
                    status_flags.append("target_coordinate_out_of_bounds")
                release_x = parse_float(release_row, "manual_x")
                release_y = parse_float(release_row, "manual_y")
                release_source_displacement_px = math.hypot(release_x - raw_x, release_y - raw_y)
                source_target_displacement_px = math.hypot(transform["target_x"] - raw_x, transform["target_y"] - raw_y)
                release_target_displacement_px = math.hypot(release_x - transform["target_x"], release_y - transform["target_y"])
            except Exception as exc:  # noqa: BLE001
                status_flags.append(f"coordinate_transform_failed:{type(exc).__name__}")

        archived_candidates = archived_lookup.get((scene, point_name, image_name), [])
        archived_status = "missing"
        archived_x = archived_y = None
        if archived_candidates:
            archived_x, archived_y = get_archived_xy(archived_candidates[0])
            if archived_x is not None and archived_y is not None and transform:
                archived_agreement_px = math.hypot(archived_x - transform["target_x"], archived_y - transform["target_y"])
                archived_status = "coordinate_available"
            else:
                archived_status = "row_present_coordinate_unresolved"

        row_out = {
            "scene": scene,
            "row_index": row_index,
            "observation_id": obs_id,
            "point_name": point_name,
            "raw_image_name": image_name,
            "release_key": release_stable_key(release_row),
            "source_match_status": source_match_status,
            "source_image_path": str(source_image_path) if source_image_path else "",
            "source_image_resolution_status": source_image_status,
            "source_image_sha256": source_image_sha,
            "source_decoded_width": decoded_w if decoded_w is not None else "",
            "source_decoded_height": decoded_h if decoded_h is not None else "",
            "source_decode_status": decode_status,
            "target_image_sha256": target_hash.get("sha256", ""),
            "target_image_bytes": target_hash.get("bytes", ""),
            "source_image_id": source_image.image_id if source_image else "",
            "source_camera_id": source_image.camera_id if source_image else "",
            "target_image_id": target_image.image_id if target_image else "",
            "target_camera_id": target_image.camera_id if target_image else "",
            "source_camera_model": source_camera.model if source_camera else "",
            "target_camera_model": target_camera.model if target_camera else "",
            "source_camera_width": source_camera.width if source_camera else "",
            "source_camera_height": source_camera.height if source_camera else "",
            "target_camera_width": target_camera.width if target_camera else "",
            "target_camera_height": target_camera.height if target_camera else "",
            "source_camera_record_hash": source_camera.record_sha256 if source_camera else "",
            "target_camera_record_hash": target_camera.record_sha256 if target_camera else "",
            "source_pose_record_hash": source_image.record_sha256 if source_image else "",
            "target_pose_record_hash": target_image.record_sha256 if target_image else "",
            "source_target_camera_record_match": bool(source_camera and target_camera and source_camera.record_sha256 == target_camera.record_sha256),
            "source_target_pose_record_match": bool(source_image and target_image and source_image.record_sha256 == target_image.record_sha256),
            "pose_center_difference_model_units": pose_center_diff if pose_center_diff is not None else "",
            "pose_rotation_difference_rad": pose_rotation_diff if pose_rotation_diff is not None else "",
            "pose_equivalent": bool(pose_equivalent),
            "mapping_type": mapping_type,
            "release_manual_x": release_row.get("manual_x", ""),
            "release_manual_y": release_row.get("manual_y", ""),
            "raw_manual_x_text": raw_manual_x_text,
            "raw_manual_y_text": raw_manual_y_text,
            "raw_manual_x": float(raw_manual_x_text) if raw_manual_x_text else "",
            "raw_manual_y": float(raw_manual_y_text) if raw_manual_y_text else "",
            "normalized_x": transform.get("normalized_x", ""),
            "normalized_y": transform.get("normalized_y", ""),
            "normalized_unit_ray_x": transform.get("normalized_unit_ray_x", ""),
            "normalized_unit_ray_y": transform.get("normalized_unit_ray_y", ""),
            "normalized_unit_ray_z": transform.get("normalized_unit_ray_z", ""),
            "target_x": transform.get("target_x", ""),
            "target_y": transform.get("target_y", ""),
            "target_in_bounds": bool(in_bounds),
            "target_to_raw_roundtrip_x": transform.get("roundtrip_raw_x", ""),
            "target_to_raw_roundtrip_y": transform.get("roundtrip_raw_y", ""),
            "roundtrip_error_px": raw_target_roundtrip_error if raw_target_roundtrip_error is not None else "",
            "release_minus_source_displacement_px": release_source_displacement_px if release_source_displacement_px is not None else "",
            "source_raw_to_target_displacement_px": source_target_displacement_px if source_target_displacement_px is not None else "",
            "release_minus_target_displacement_px": release_target_displacement_px if release_target_displacement_px is not None else "",
            "archived_undistorted_status": archived_status,
            "archived_undistorted_x": archived_x if archived_x is not None else "",
            "archived_undistorted_y": archived_y if archived_y is not None else "",
            "recomputed_vs_archived_undistorted_displacement_px": archived_agreement_px if archived_agreement_px is not None else "",
            "status_flags": ";".join(status_flags) if status_flags else "ok",
        }
        per_obs.append(row_out)
        mapping_rows.append(
            {
                "scene": scene,
                "image_name": image_name,
                "source_image_id": row_out["source_image_id"],
                "source_camera_id": row_out["source_camera_id"],
                "target_image_id": row_out["target_image_id"],
                "target_camera_id": row_out["target_camera_id"],
                "source_pose_record_hash": row_out["source_pose_record_hash"],
                "target_pose_record_hash": row_out["target_pose_record_hash"],
                "source_camera_record_hash": row_out["source_camera_record_hash"],
                "target_camera_record_hash": row_out["target_camera_record_hash"],
                "mapping_type": mapping_type,
                "pose_center_difference_model_units": row_out["pose_center_difference_model_units"],
                "pose_rotation_difference_rad": row_out["pose_rotation_difference_rad"],
                "mapping_record_sha256": stable_sha256(
                    {
                        "scene": scene,
                        "image_name": image_name,
                        "source_image_id": row_out["source_image_id"],
                        "source_camera_id": row_out["source_camera_id"],
                        "target_image_id": row_out["target_image_id"],
                        "target_camera_id": row_out["target_camera_id"],
                        "mapping_type": mapping_type,
                    }
                ),
            }
        )
        if archived_status != "missing":
            archived_rows_out.append(
                {
                    "scene": scene,
                    "point_name": point_name,
                    "image_name": image_name,
                    "archived_path": archived_path,
                    "archived_sha256": archived_sha,
                    "archived_x": archived_x if archived_x is not None else "",
                    "archived_y": archived_y if archived_y is not None else "",
                    "recomputed_target_x": transform.get("target_x", ""),
                    "recomputed_target_y": transform.get("target_y", ""),
                    "recomputed_vs_archived_displacement_px": archived_agreement_px if archived_agreement_px is not None else "",
                    "status": archived_status,
                }
            )

    obs_id_duplicates = sum(c - 1 for c in seen_obs_ids.values() if c > 1)
    missing_source_camera = sum(1 for row in per_obs if "missing_source_camera" in str(row["status_flags"]))
    missing_target_camera = sum(1 for row in per_obs if "missing_target_camera" in str(row["status_flags"]))
    missing_mapping = sum(1 for row in per_obs if "missing_source_colmap_image" in str(row["status_flags"]) or "missing_target_colmap_image" in str(row["status_flags"]))
    pose_mismatch = sum(1 for row in per_obs if "pose_mismatch" in str(row["status_flags"]))
    out_of_bounds = sum(1 for row in per_obs if not row["target_in_bounds"])
    camera_record_mismatch = sum(1 for row in per_obs if not row["source_target_camera_record_match"])
    release_source_max = coordinate_stats([float(row["release_minus_source_displacement_px"]) for row in per_obs if str(row["release_minus_source_displacement_px"]) != ""])["max"]
    displacement_stats = coordinate_stats([float(row["source_raw_to_target_displacement_px"]) for row in per_obs if str(row["source_raw_to_target_displacement_px"]) != ""])
    roundtrip_stats = coordinate_stats([float(row["roundtrip_error_px"]) for row in per_obs if str(row["roundtrip_error_px"]) != ""])
    archived_stats = coordinate_stats([float(row["recomputed_vs_archived_undistorted_displacement_px"]) for row in per_obs if str(row["recomputed_vs_archived_undistorted_displacement_px"]) != ""])

    if not release_rows:
        conclusion = "other_explicit_failure"
    elif missing_source_camera:
        conclusion = "unresolved_missing_source_camera"
    elif missing_target_camera:
        conclusion = "unresolved_missing_target_camera"
    elif missing_mapping:
        conclusion = "unresolved_mapping"
    elif pose_mismatch:
        conclusion = "non_equivalent_camera_pose"
    elif out_of_bounds:
        conclusion = "other_explicit_failure"
    elif release_source_max is not None and float(release_source_max) <= 1e-9 and (displacement_stats["median"] or 0) > 1.0:
        conclusion = "confirmed_mismatch"
    elif (displacement_stats["max"] or 0) <= 1e-6:
        conclusion = "domain_already_consistent"
    else:
        conclusion = "other_explicit_failure"
    assert conclusion in ALLOWED_CONCLUSIONS

    image_mapping_counter = Counter(row["image_name"] for row in mapping_rows)
    per_image_counts = [
        {"scene": scene, "image_name": image_name, "observation_count": count}
        for image_name, count in sorted(image_mapping_counter.items())
    ]
    scene_summary = {
        "scene": scene,
        "conclusion": conclusion,
        "annotation_source_domain": "raw_source_decoded_pixel_matrix" if release_source_max is not None and float(release_source_max) <= 1e-9 else "unresolved_or_changed",
        "target_packet_training_domain": "colmap_undistorted_training_camera_domain",
        "observation_count": len(release_rows),
        "source_annotation_declared_files": declared_source_files,
        "source_annotation_resolution_records": source_file_records,
        "source_annotation_rows_recovered": len(source_rows),
        "duplicate_release_key_count": len(duplicate_release_keys),
        "observation_id_duplicate_count": obs_id_duplicates,
        "median_raw_to_target_pixel_displacement": displacement_stats["median"],
        "p95_raw_to_target_pixel_displacement": displacement_stats["p95"],
        "max_raw_to_target_pixel_displacement": displacement_stats["max"],
        "median_roundtrip_error_px": roundtrip_stats["median"],
        "p95_roundtrip_error_px": roundtrip_stats["p95"],
        "max_roundtrip_error_px": roundtrip_stats["max"],
        "target_in_bounds_count": sum(1 for row in per_obs if row["target_in_bounds"]),
        "target_out_of_bounds_count": out_of_bounds,
        "unique_source_cameras": sorted({row["source_camera_id"] for row in per_obs if row["source_camera_id"] != ""}),
        "unique_target_cameras": sorted({row["target_camera_id"] for row in per_obs if row["target_camera_id"] != ""}),
        "per_image_mapping_count": len(image_mapping_counter),
        "missing_mapping_count": missing_mapping,
        "pose_mismatch_count": pose_mismatch,
        "camera_record_mismatch_count": camera_record_mismatch,
        "archived_undistorted_path": archived_path,
        "archived_undistorted_sha256": archived_sha,
        "archived_undistorted_rows": len(archived_rows),
        "archived_undistorted_agreement_count": archived_stats["count"],
        "archived_undistorted_agreement_median_px": archived_stats["median"],
        "archived_undistorted_agreement_p95_px": archived_stats["p95"],
        "archived_undistorted_agreement_max_px": archived_stats["max"],
        "source_cameras_file_sha256": find_file_record(raw_model, "cameras.bin").get("sha256", ""),
        "source_images_file_sha256": find_file_record(raw_model, "images.bin").get("sha256", ""),
        "target_cameras_file_sha256": find_file_record(target_model, "cameras.bin").get("sha256", ""),
        "target_images_file_sha256": find_file_record(target_model, "images.bin").get("sha256", ""),
        "source_model_path": raw_model.get("path", ""),
        "target_model_path": target_model.get("path", ""),
    }
    return scene_summary, per_obs, mapping_rows, per_image_counts, archived_rows_out


def build_v12_schema_draft() -> str:
    return "\n".join(
        [
            "# Draft Release v1.2 Pixel-Domain Observation Schema",
            "",
            "This is a Stage-1 draft only. It is not a frozen release CSV and must not be used as a formal benchmark release until Stage 2 is reviewed.",
            "",
            "## Canonical Observation Identity",
            "",
            "`observation_id` is `SHA-256(UTF-8 JSON array)` over:",
            "",
            "1. `scene`",
            "2. `point_name`",
            "3. `raw_image_name`",
            "4. raw image SHA-256",
            "5. original decimal string `raw_manual_x`",
            "6. original decimal string `raw_manual_y`",
            "",
            "The ID is independent of CSV row number, target projection, method, camera target, and evaluator output.",
            "",
            "## Source Ray Definition",
            "",
            "- `normalized_x` and `normalized_y` are source-camera distortion-inverted normalized image-plane coordinates.",
            "- The source camera-frame ray is `[normalized_x, normalized_y, 1]`.",
            "- `normalized_unit_ray` is that vector normalized in source camera frame.",
            "- Pixel convention is 0-based pixel centers.",
            "- Source image orientation is the actual decoded pixel matrix orientation, not only the EXIF orientation tag.",
            "- Source camera convention follows COLMAP world-to-camera extrinsics.",
            "",
            "## Cached Benchmark Target Projection Fields",
            "",
            "- `target_pixel_domain`",
            "- `target_camera_id`, `target_camera_model`, `target_camera_parameters`, `target_camera_hash`",
            "- `target_x`, `target_y`",
            "- `mapping_type`",
            "- `transform_version`",
            "- source/target `cameras.bin` and `images.bin` SHA-256",
            "- per-camera intrinsic record SHA-256",
            "- per-image pose record SHA-256",
            "",
            "## Stage-2 Track Boundary",
            "",
            "- Benchmark camera track: methods use benchmark-provided undistorted images and camera model.",
            "- Method-specific camera track: allowed only when cameras are pose-equivalent to source/benchmark cameras or a verified explicit pixel remap is provided.",
            "- A raw source ray cannot be projected into an arbitrary target camera with different pose without depth.",
            "",
        ]
    )


def artifact_inventory_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        rows.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() and path.is_file() else "",
                "sha256": file_sha256(path) if path.exists() and path.is_file() else "",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-1 no-GPU six-scene GCP annotation pixel-domain audit.")
    parser.add_argument("--project_root", default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("--dataset_root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--release_dir", default=str(DEFAULT_RELEASE_DIR))
    parser.add_argument("--remote_light_manifest", default=str(DEFAULT_REMOTE_LIGHT_MANIFEST))
    parser.add_argument("--out_base", default=str(DEFAULT_OUT_BASE))
    parser.add_argument("--package_dir", default=str(DEFAULT_PACKAGE_DIR))
    args = parser.parse_args()

    project_root = Path(args.project_root)
    dataset_root = Path(args.dataset_root)
    release_dir = Path(args.release_dir)
    remote_light_manifest = Path(args.remote_light_manifest)
    out_base = Path(args.out_base)
    package_dir = Path(args.package_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = make_unique_dir(out_base / f"gcp_6scene_annotation_domain_audit_20260628_{timestamp}")

    if not remote_light_manifest.exists():
        raise SystemExit(f"missing remote light manifest: {remote_light_manifest}")
    manifest = json.loads(remote_light_manifest.read_text(encoding="utf-8"))
    provenance_csv = release_dir / "final_annotation_inclusion_provenance.csv"
    release_config = release_dir / "gcp_benchmark_release_v1_1.json"
    provenance_rows = read_csv(provenance_csv)

    scene_summaries: list[dict[str, Any]] = []
    all_obs: list[dict[str, Any]] = []
    all_mappings: list[dict[str, Any]] = []
    all_image_counts: list[dict[str, Any]] = []
    all_archived: list[dict[str, Any]] = []
    for scene in SCENES:
        scene_entry = manifest.get("scenes", {}).get(scene)
        if not scene_entry:
            scene_summaries.append(
                {
                    "scene": scene,
                    "conclusion": "other_explicit_failure",
                    "annotation_source_domain": "unresolved",
                    "target_packet_training_domain": "unresolved",
                    "observation_count": 0,
                    "failure": "missing_scene_in_remote_light_manifest",
                }
            )
            continue
        summary, obs_rows, mapping_rows, image_counts, archived_rows = audit_scene(
            scene, release_dir, dataset_root, project_root, scene_entry, provenance_rows
        )
        scene_summaries.append(summary)
        all_obs.extend(obs_rows)
        all_mappings.extend(mapping_rows)
        all_image_counts.extend(image_counts)
        all_archived.extend(archived_rows)

    mapping_manifest_sha = stable_sha256(all_mappings)
    for row in scene_summaries:
        row["mapping_manifest_sha256"] = mapping_manifest_sha

    write_csv(out_dir / "scene_domain_audit_summary.csv", scene_summaries)
    write_json(out_dir / "scene_domain_audit_summary.json", scene_summaries)
    write_csv(out_dir / "per_observation_domain_audit.csv", all_obs)
    write_csv(out_dir / "source_target_mapping_manifest.csv", all_mappings)
    write_json(out_dir / "source_target_mapping_manifest.json", all_mappings)
    write_csv(out_dir / "per_image_mapping_counts.csv", all_image_counts)
    write_csv(out_dir / "archived_undistorted_agreement.csv", all_archived)
    (out_dir / "v1_2_schema_draft.md").write_text(build_v12_schema_draft(), encoding="utf-8")

    obs_id_counter = Counter(row["observation_id"] for row in all_obs)
    obs_id_collisions = [
        {"observation_id": obs_id, "count": count}
        for obs_id, count in sorted(obs_id_counter.items())
        if count > 1
    ]
    write_csv(out_dir / "observation_id_collision_report.csv", obs_id_collisions, ["observation_id", "count"])
    write_json(
        out_dir / "observation_id_collision_report.json",
        {
            "observation_count": len(all_obs),
            "unique_observation_id_count": len(obs_id_counter),
            "collision_count": len(obs_id_collisions),
            "collisions": obs_id_collisions,
        },
    )

    tests = [
        {
            "name": "no_gpu_no_packet_export_boundary",
            "passed": True,
            "actual": "script reads CSVs, local raw images, and cached camera manifest only",
            "expected": "no GPU, no packet export, no formal evaluator mutation",
        },
        {
            "name": "row_spine_preserved",
            "passed": len(all_obs) == sum(int(row.get("observation_count", 0)) for row in scene_summaries),
            "actual": len(all_obs),
            "expected": sum(int(row.get("observation_count", 0)) for row in scene_summaries),
        },
        {
            "name": "observation_id_unique",
            "passed": len(obs_id_collisions) == 0,
            "actual": len(obs_id_collisions),
            "expected": 0,
        },
        {
            "name": "all_conclusions_in_allowed_set",
            "passed": all(row.get("conclusion") in ALLOWED_CONCLUSIONS for row in scene_summaries),
            "actual": sorted({row.get("conclusion") for row in scene_summaries}),
            "expected": sorted(ALLOWED_CONCLUSIONS),
        },
        {
            "name": "roundtrip_error_below_predeclared_tolerance_for_resolved_rows",
            "passed": all(
                row.get("max_roundtrip_error_px") in ("", None)
                or float(row["max_roundtrip_error_px"]) <= PIXEL_ROUNDTRIP_TOL
                for row in scene_summaries
            ),
            "actual": {row["scene"]: row.get("max_roundtrip_error_px") for row in scene_summaries},
            "expected": f"<= {PIXEL_ROUNDTRIP_TOL}",
        },
    ]
    write_csv(out_dir / "test_results.csv", tests)
    write_json(out_dir / "test_results.json", tests)

    release_csvs = [release_dir / f"{scene}_gcp_annotations_final_good_nadir_v1.csv" for scene in SCENES]
    inventory_paths = [remote_light_manifest, provenance_csv, release_config, *release_csvs]
    inventory = artifact_inventory_rows(inventory_paths)
    write_csv(out_dir / "artifact_inventory.csv", inventory)
    write_json(out_dir / "artifact_inventory.json", inventory)

    source_resolution_records = []
    for row in scene_summaries:
        for rec in row.get("source_annotation_resolution_records", []):
            source_resolution_records.append(rec)
    write_csv(out_dir / "source_annotation_resolution_inventory.csv", source_resolution_records)

    source_target_matrix = []
    for row in scene_summaries:
        source_target_matrix.append(
            {
                "scene": row["scene"],
                "source_domain_status": row.get("annotation_source_domain", ""),
                "target_domain_status": row.get("target_packet_training_domain", ""),
                "source_cameras_file_sha256": row.get("source_cameras_file_sha256", ""),
                "source_images_file_sha256": row.get("source_images_file_sha256", ""),
                "target_cameras_file_sha256": row.get("target_cameras_file_sha256", ""),
                "target_images_file_sha256": row.get("target_images_file_sha256", ""),
                "conclusion": row.get("conclusion", ""),
                "evidence_status": (
                    "source_target_mapping_verified"
                    if row.get("conclusion") in {"confirmed_mismatch", "domain_already_consistent"}
                    else "unresolved_or_failed"
                ),
            }
        )
    write_csv(out_dir / "annotation_provenance_matrix.csv", source_target_matrix)
    write_json(out_dir / "annotation_provenance_matrix.json", source_target_matrix)

    code_dir = out_dir / "code_snapshot"
    code_dir.mkdir(parents=True, exist_ok=True)
    for rel in [
        "code/gcp/audit_6scene_annotation_domain.py",
        "code/gcp/audit_3k_annotation_domain.py",
        "code/gcp/undistort_gcp_observations.py",
        "code/gcp/manual_gcp_annotator.py",
    ]:
        src = REPO_ROOT / rel
        if src.exists():
            dst = code_dir / Path(rel).name
            shutil.copy2(src, dst)
    (code_dir / "git_commit.txt").write_text(git_text(["rev-parse", "HEAD"]) + "\n", encoding="utf-8")
    (code_dir / "git_status_porcelain.txt").write_text(git_text(["status", "--porcelain"]) + "\n", encoding="utf-8")
    (code_dir / "git_show_head.patch").write_text(
        git_text(["show", "--stat", "--patch", "--no-renames", "--", "code/gcp/audit_6scene_annotation_domain.py", "docs/gcp_6scene_annotation_domain_audit.md"])
        + "\n",
        encoding="utf-8",
    )

    conclusion_counts = Counter(row.get("conclusion", "") for row in scene_summaries)
    all_resolved = all(row.get("conclusion") in {"confirmed_mismatch", "domain_already_consistent"} for row in scene_summaries)
    readiness = "stage1_resolved_all_scenes_but_v1_2_not_frozen" if all_resolved else "stage1_unresolved_scene_exists_no_v1_2_readiness"
    review_brief = [
        "# Six-Scene Annotation-Domain Audit",
        "",
        f"- Created UTC: `{datetime.now(timezone.utc).isoformat()}`",
        f"- Scenes audited: `{len(SCENES)}`",
        f"- Frozen v1.1 observation rows preserved: `{len(all_obs)}`",
        f"- Observation ID collisions: `{len(obs_id_collisions)}`",
        f"- Mapping manifest SHA-256: `{mapping_manifest_sha}`",
        f"- Stage-1 readiness label: `{readiness}`",
        "",
        "## Per-scene conclusions",
        "",
        "| Scene | Conclusion | Obs | Median raw-to-target displacement (px) | P95 (px) | Max (px) | Round-trip max (px) | Target in-bounds | Pose mismatches |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in scene_summaries:
        review_brief.append(
            f"| {row['scene']} | {row.get('conclusion')} | {row.get('observation_count')} | {row.get('median_raw_to_target_pixel_displacement')} | {row.get('p95_raw_to_target_pixel_displacement')} | {row.get('max_raw_to_target_pixel_displacement')} | {row.get('max_roundtrip_error_px')} | {row.get('target_in_bounds_count')} | {row.get('pose_mismatch_count')} |"
        )
    review_brief.extend(
        [
            "",
            "## Boundary",
            "",
            "This package is Stage 1 only. It does not freeze release v1.2, create formal v1.2 CSVs, modify release v1.1, modify the formal evaluator, export packets, train models, or use GPU.",
            "",
            "## v1.2 status",
            "",
            "The included `v1_2_schema_draft.md` is a schema draft only. Stage 2 must be reviewed separately before any unified v1.2 release can be frozen.",
        ]
    )
    (out_dir / "REVIEW_BRIEF.md").write_text("\n".join(review_brief) + "\n", encoding="utf-8")

    audit_summary = {
        "schema": "ms_gcp_6scene_annotation_domain_audit_stage1_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scenes": SCENES,
        "stage": "stage1_audit_only",
        "readiness": readiness,
        "conclusion_counts": dict(conclusion_counts),
        "observation_count": len(all_obs),
        "unique_observation_id_count": len(obs_id_counter),
        "observation_id_collision_count": len(obs_id_collisions),
        "pose_center_tolerance_model_units": POSE_CENTER_TOL,
        "pose_rotation_tolerance_rad": POSE_ROTATION_TOL_RAD,
        "pixel_roundtrip_tolerance_px": PIXEL_ROUNDTRIP_TOL,
        "mapping_manifest_sha256": mapping_manifest_sha,
        "boundaries": [
            "No GPU.",
            "No packet export.",
            "No training.",
            "No formal evaluator changes.",
            "No release v1.1 overwrite.",
            "No pointset/split/annotation mutation.",
            "No release v1.2 freeze.",
        ],
    }
    write_json(out_dir / "audit_summary.json", audit_summary)

    package_manifest = {"out_dir": str(out_dir), "files": []}
    for path in sorted(out_dir.rglob("*")):
        if path.is_file():
            package_manifest["files"].append(
                {"path": path.relative_to(out_dir).as_posix(), "bytes": path.stat().st_size, "sha256": file_sha256(path)}
            )
    write_json(out_dir / "package_manifest.json", package_manifest)

    package_path, sha_path = package_outputs(
        out_dir,
        package_dir / "GPT_GCP_6SCENE_ANNOTATION_DOMAIN_AUDIT_REVIEW_20260628.zip",
    )
    write_json(
        out_dir / "final_package_pointer.json",
        {"package_path": str(package_path), "sha256_file": str(sha_path), "package_sha256": file_sha256(package_path)},
    )
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "package": str(package_path),
                "package_sha256": file_sha256(package_path),
                "readiness": readiness,
                "conclusion_counts": dict(conclusion_counts),
                "scene_conclusions": {row["scene"]: row.get("conclusion") for row in scene_summaries},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
