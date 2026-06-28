from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

try:
    from PIL import Image, features
except Exception as exc:  # noqa: BLE001
    Image = None
    features = None
    PIL_IMPORT_ERROR = exc
else:
    PIL_IMPORT_ERROR = None


SCENES = [
    "gcp_3000_20260602",
    "gcp_5000_20260602",
    "gcp_10000_20260610",
    "gcp_20000_20260602",
    "gcp_50000_20260610",
    "gcp_100000_20260610",
]

RELEASE_V12_SCHEMA = "ms_gcp_3dgs_benchmark_release_config_v1_2"
RELEASE_V12_ID = "gcp_benchmark_release_v1_2_pixel_domain_20260628"
RELEASE_V121_SCHEMA = "ms_gcp_3dgs_benchmark_release_config_v1_2_1"
RELEASE_V121_ID = "gcp_benchmark_release_v1_2_1_pixel_domain_20260628"
PIXEL_DOMAIN_RELEASE_SCHEMAS = {RELEASE_V12_SCHEMA, RELEASE_V121_SCHEMA}
TRANSFORM_VERSION = "raw_simple_radial_to_benchmark_pinhole_v1"
SOURCE_PIXEL_DOMAIN = "raw_dji_decoded_pixel_matrix_ignore_exif_orientation"
TARGET_PIXEL_DOMAIN = "benchmark_colmap_undistorted_pinhole_pixel_domain"
ORIENTATION_POLICY = "ignore_exif_orientation_no_transpose"
PIXEL_CONVENTION = "zero_based_pixel_centers"

POSE_CENTER_TOL = 1e-8
POSE_ROTATION_TOL_RAD = 1e-8
CACHED_TARGET_TOL_PX = 1e-9
ROUNDTRIP_TOL_PX = 1e-6
ARCHIVED_UNDISTORTED_TOL_PX = 1e-4
SIMPLE_RADIAL_MAX_ITER = 20
SIMPLE_RADIAL_CONVERGENCE_ABS = 1e-12
SIMPLE_RADIAL_CONVERGENCE_REL = 1e-12

RGB_MATRIX_HASH_MAGIC = b"MS_GCP_RGB_MATRIX_HASH_V1\x00"
OBSERVATION_ID_SCHEMA = "ms_gcp_observation_id_stage1_v1"


@dataclass(frozen=True)
class CameraRecord:
    camera_id: int
    model: str
    width: int
    height: int
    params: tuple[float, ...]
    record_sha256: str = ""


@dataclass(frozen=True)
class ImageRecord:
    image_id: int
    image_name: str
    camera_id: int
    qvec: tuple[float, float, float, float]
    tvec: tuple[float, float, float]
    record_sha256: str = ""


def fmt_float(value: float) -> str:
    return format(float(value), ".17g")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_record_sha256(obj: Any) -> str:
    return sha256_bytes(canonical_json_bytes(obj))


def canonical_records_root_sha256(
    rows: Sequence[dict[str, Any]],
    sort_keys: Sequence[str],
) -> str:
    normalized = sorted((dict(row) for row in rows), key=lambda row: tuple(str(row.get(k, "")) for k in sort_keys))
    return sha256_bytes(canonical_json_bytes(normalized))


def observation_id_payload(
    scene: str,
    point_name: str,
    raw_image_name: str,
    raw_image_sha256: str,
    raw_manual_x_text: str,
    raw_manual_y_text: str,
) -> list[str]:
    if not all([scene, point_name, raw_image_name, raw_image_sha256, raw_manual_x_text, raw_manual_y_text]):
        raise ValueError("observation_id fields must be non-empty")
    return [
        scene.strip(),
        point_name.strip(),
        Path(raw_image_name).name,
        raw_image_sha256.strip().lower(),
        raw_manual_x_text.strip(),
        raw_manual_y_text.strip(),
    ]


def serialize_observation_id_payload(payload: Sequence[str]) -> bytes:
    return json.dumps(list(payload), ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def observation_id_from_payload(payload: Sequence[str]) -> str:
    return sha256_bytes(serialize_observation_id_payload(payload))


def observation_id_from_fields(
    scene: str,
    point_name: str,
    raw_image_name: str,
    raw_image_sha256: str,
    raw_manual_x_text: str,
    raw_manual_y_text: str,
) -> str:
    return observation_id_from_payload(
        observation_id_payload(scene, point_name, raw_image_name, raw_image_sha256, raw_manual_x_text, raw_manual_y_text)
    )


def serialize_rgb_pixel_matrix(mode: str, width: int, height: int, raw_rgb_bytes: bytes) -> bytes:
    if mode != "RGB":
        raise ValueError(f"RGB pixel matrix hash only supports mode RGB, got {mode!r}")
    expected = int(width) * int(height) * 3
    if len(raw_rgb_bytes) != expected:
        raise ValueError(f"RGB byte length mismatch: expected {expected}, got {len(raw_rgb_bytes)}")
    mode_bytes = mode.encode("ascii")
    return (
        RGB_MATRIX_HASH_MAGIC
        + struct.pack("<H", len(mode_bytes))
        + mode_bytes
        + struct.pack("<II", int(width), int(height))
        + raw_rgb_bytes
    )


def rgb_pixel_matrix_sha256(mode: str, width: int, height: int, raw_rgb_bytes: bytes) -> str:
    return sha256_bytes(serialize_rgb_pixel_matrix(mode, width, height, raw_rgb_bytes))


def image_decoder_versions() -> dict[str, Any]:
    payload = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "pillow_import_error": repr(PIL_IMPORT_ERROR) if PIL_IMPORT_ERROR else "",
    }
    if Image is not None:
        payload["pillow_version"] = getattr(Image, "__version__", "")
    if features is not None:
        for key in ["jpg", "jpg_2000", "zlib", "libtiff"]:
            try:
                payload[f"{key}_version"] = features.version(key)
            except Exception as exc:  # noqa: BLE001
                payload[f"{key}_version"] = f"ERROR:{type(exc).__name__}:{exc}"
    return payload


def load_raw_image_orientation_record(path: Path) -> dict[str, Any]:
    if Image is None:
        raise RuntimeError(f"Pillow is unavailable: {PIL_IMPORT_ERROR}")
    byte_hash = file_sha256(path)
    with Image.open(path) as img:
        exif_orientation = ""
        try:
            exif = img.getexif()
            value = exif.get(274)
            exif_orientation = "" if value is None else str(value)
        except Exception as exc:  # noqa: BLE001
            exif_orientation = f"ERROR:{type(exc).__name__}:{exc}"
        rgb = img.convert("RGB")
        width, height = int(rgb.width), int(rgb.height)
        raw = rgb.tobytes()
        matrix_hash = rgb_pixel_matrix_sha256("RGB", width, height, raw)
    return {
        "raw_image_path": str(path),
        "raw_image_sha256": byte_hash,
        "exif_orientation_raw_value": exif_orientation,
        "applied_orientation_policy": ORIENTATION_POLICY,
        "decoded_mode": "RGB",
        "decoded_width": width,
        "decoded_height": height,
        "rgb_pixel_matrix_hash_schema": "ms_gcp_rgb_matrix_hash_v1",
        "rgb_pixel_matrix_sha256": matrix_hash,
        "decoder_versions": image_decoder_versions(),
    }


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
    return -qvec2rotmat(image.qvec).T @ np.asarray(image.tvec, dtype=float)


def rotation_angle_rad(source: ImageRecord, target: ImageRecord) -> float:
    if source.record_sha256 and source.record_sha256 == target.record_sha256:
        return 0.0
    qs = np.asarray(source.qvec, dtype=float)
    qt = np.asarray(target.qvec, dtype=float)
    qs /= np.linalg.norm(qs)
    qt /= np.linalg.norm(qt)
    dot = abs(float(np.dot(qs, qt)))
    dot = max(-1.0, min(1.0, dot))
    return float(2.0 * math.acos(dot))


def pose_equivalence(source: ImageRecord, target: ImageRecord) -> dict[str, Any]:
    if source.record_sha256 and source.record_sha256 == target.record_sha256:
        center_diff = 0.0
        rot_diff = 0.0
    else:
        center_diff = float(np.linalg.norm(camera_center(source) - camera_center(target)))
        rot_diff = rotation_angle_rad(source, target)
    return {
        "camera_center_difference_model_units": center_diff,
        "rotation_angular_difference_rad": rot_diff,
        "pose_equivalent": bool(center_diff <= POSE_CENTER_TOL and rot_diff <= POSE_ROTATION_TOL_RAD),
    }


def invert_simple_radial(camera: CameraRecord, u: float, v: float) -> tuple[float, float, dict[str, Any]]:
    if camera.model.upper() != "SIMPLE_RADIAL" or len(camera.params) != 4:
        raise ValueError(f"source camera must be SIMPLE_RADIAL [f,cx,cy,k], got {camera.model} {camera.params}")
    f, cx, cy, k = [float(x) for x in camera.params]
    xd = (float(u) - cx) / f
    yd = (float(v) - cy) / f
    x = xd
    y = yd
    max_delta = math.inf
    for _ in range(SIMPLE_RADIAL_MAX_ITER):
        r2 = x * x + y * y
        denom = 1.0 + k * r2
        if denom == 0 or not math.isfinite(denom):
            raise ValueError("SIMPLE_RADIAL inversion produced invalid denominator")
        x_new = xd / denom
        y_new = yd / denom
        max_delta = max(abs(x_new - x), abs(y_new - y))
        x, y = x_new, y_new
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("SIMPLE_RADIAL inversion produced non-finite coordinates")
    tol = SIMPLE_RADIAL_CONVERGENCE_ABS + SIMPLE_RADIAL_CONVERGENCE_REL * max(abs(x), abs(y), 1.0)
    if max_delta > tol:
        raise ValueError(f"SIMPLE_RADIAL inversion did not converge: delta={max_delta} tol={tol}")
    return x, y, {"xd": xd, "yd": yd, "iterations": SIMPLE_RADIAL_MAX_ITER, "final_delta": max_delta, "tolerance": tol}


def distort_simple_radial(camera: CameraRecord, x: float, y: float) -> tuple[float, float]:
    if camera.model.upper() != "SIMPLE_RADIAL" or len(camera.params) != 4:
        raise ValueError(f"source camera must be SIMPLE_RADIAL [f,cx,cy,k], got {camera.model} {camera.params}")
    f, cx, cy, k = [float(v) for v in camera.params]
    r2 = float(x) * float(x) + float(y) * float(y)
    scale = 1.0 + k * r2
    xd = float(x) * scale
    yd = float(y) * scale
    u = f * xd + cx
    v = f * yd + cy
    if not all(math.isfinite(z) for z in [u, v]):
        raise ValueError("SIMPLE_RADIAL forward distortion produced non-finite pixel")
    return u, v


def project_pinhole(camera: CameraRecord, x: float, y: float) -> tuple[float, float]:
    if camera.model.upper() != "PINHOLE" or len(camera.params) != 4:
        raise ValueError(f"target camera must be PINHOLE [fx,fy,cx,cy], got {camera.model} {camera.params}")
    fx, fy, cx, cy = [float(v) for v in camera.params]
    u = fx * float(x) + cx
    v = fy * float(y) + cy
    if not all(math.isfinite(z) for z in [u, v]):
        raise ValueError("PINHOLE projection produced non-finite pixel")
    return u, v


def unproject_pinhole(camera: CameraRecord, u: float, v: float) -> tuple[float, float]:
    if camera.model.upper() != "PINHOLE" or len(camera.params) != 4:
        raise ValueError(f"target camera must be PINHOLE [fx,fy,cx,cy], got {camera.model} {camera.params}")
    fx, fy, cx, cy = [float(z) for z in camera.params]
    return (float(u) - cx) / fx, (float(v) - cy) / fy


def raw_to_target_projection(source_camera: CameraRecord, target_camera: CameraRecord, raw_x: float, raw_y: float) -> dict[str, Any]:
    x, y, inversion = invert_simple_radial(source_camera, raw_x, raw_y)
    unit = np.asarray([x, y, 1.0], dtype=np.float64)
    unit /= np.linalg.norm(unit)
    target_x, target_y = project_pinhole(target_camera, x, y)
    back_x, back_y = unproject_pinhole(target_camera, target_x, target_y)
    raw_back_x, raw_back_y = distort_simple_radial(source_camera, back_x, back_y)
    roundtrip = math.hypot(raw_back_x - float(raw_x), raw_back_y - float(raw_y))
    return {
        "normalized_x": x,
        "normalized_y": y,
        "normalized_unit_ray_x": float(unit[0]),
        "normalized_unit_ray_y": float(unit[1]),
        "normalized_unit_ray_z": float(unit[2]),
        "target_x": target_x,
        "target_y": target_y,
        "roundtrip_raw_x": raw_back_x,
        "roundtrip_raw_y": raw_back_y,
        "roundtrip_error_px": roundtrip,
        "inversion": inversion,
    }


def camera_canonical_record(camera: CameraRecord) -> dict[str, Any]:
    return {
        "camera_id": int(camera.camera_id),
        "height": int(camera.height),
        "model": camera.model,
        "params": [fmt_float(v) for v in camera.params],
        "width": int(camera.width),
    }


def image_pose_canonical_record(image: ImageRecord) -> dict[str, Any]:
    return {
        "camera_id": int(image.camera_id),
        "image_id": int(image.image_id),
        "image_name": image.image_name,
        "qvec": [fmt_float(v) for v in image.qvec],
        "tvec": [fmt_float(v) for v in image.tvec],
    }


def camera_record_hash(camera: CameraRecord) -> str:
    return canonical_record_sha256(camera_canonical_record(camera))


def image_pose_record_hash(image: ImageRecord) -> str:
    return canonical_record_sha256(image_pose_canonical_record(image))


def load_manifest_model(scene_entry: dict[str, Any], key: str) -> tuple[dict[int, CameraRecord], dict[str, ImageRecord], dict[str, Any]]:
    model = scene_entry.get(key, {})
    cameras = {}
    for raw in model.get("cameras", []):
        camera = CameraRecord(
            camera_id=int(raw["camera_id"]),
            model=str(raw["model"]),
            width=int(raw["width"]),
            height=int(raw["height"]),
            params=tuple(float(v) for v in raw.get("params", [])),
            record_sha256=str(raw.get("record_sha256", "")),
        )
        cameras[camera.camera_id] = camera
    images = {}
    for raw in model.get("images", []):
        image = ImageRecord(
            image_id=int(raw["image_id"]),
            image_name=Path(str(raw["image_name"])).name,
            camera_id=int(raw["camera_id"]),
            qvec=tuple(float(v) for v in raw.get("qvec", [])),  # type: ignore[arg-type]
            tvec=tuple(float(v) for v in raw.get("tvec", [])),  # type: ignore[arg-type]
            record_sha256=str(raw.get("record_sha256", "")),
        )
        images[image.image_name] = image
    return cameras, images, model


def model_file_record(model: dict[str, Any], name: str) -> dict[str, Any]:
    for item in model.get("files", []):
        if Path(str(item.get("name", ""))).name == name:
            return dict(item)
    return {}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_deterministic(path: Path, rows: Iterable[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), lineterminator="\r\n", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json_deterministic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")


def relative_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def payload_manifest_entries(root: Path, exclude: set[str]) -> list[dict[str, Any]]:
    rows = []
    for path in sorted([p for p in root.rglob("*") if p.is_file()], key=lambda p: relative_posix(p, root).encode("utf-8")):
        rel = relative_posix(path, root)
        if rel in exclude:
            continue
        rows.append({"path": rel, "bytes": path.stat().st_size, "sha256": file_sha256(path)})
    return rows


def payload_root_digest(entries: Sequence[dict[str, Any]]) -> str:
    return sha256_bytes(json.dumps(list(entries), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def git_text(args: list[str], cwd: Path) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=cwd, text=True, errors="replace").strip()
    except Exception as exc:  # noqa: BLE001
        return f"ERROR:{type(exc).__name__}:{exc}"


def generator_provenance(repo_root: Path, script_relative_path: str, command_manifest: dict[str, Any]) -> dict[str, Any]:
    status = git_text(["status", "--porcelain"], repo_root)
    return {
        "command_manifest_sha256": canonical_record_sha256(command_manifest),
        "generator_git_commit": git_text(["rev-parse", "HEAD"], repo_root),
        "generator_script_relative_path": script_relative_path.replace("\\", "/"),
        "generator_worktree_clean": status == "",
        "generator_worktree_status_porcelain": status,
        "image_decoder_versions": image_decoder_versions(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "transform_version": TRANSFORM_VERSION,
    }


def verify_payload_integrity(root: Path, manifest_path: Path, root_record_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root_record = json.loads(root_record_path.read_text(encoding="utf-8"))
    entries = manifest["files"]
    problems: list[str] = []
    excluded = {relative_posix(manifest_path, root), relative_posix(root_record_path, root)}
    listed = {item["path"]: item for item in entries}
    for rel, item in listed.items():
        path = root / rel
        if not path.exists():
            problems.append(f"missing:{rel}")
            continue
        if path.stat().st_size != int(item["bytes"]):
            problems.append(f"size_mismatch:{rel}")
        if file_sha256(path) != item["sha256"]:
            problems.append(f"sha_mismatch:{rel}")
    actual_files = {
        relative_posix(path, root)
        for path in root.rglob("*")
        if path.is_file()
        and relative_posix(path, root) not in excluded
    }
    unregistered = sorted(actual_files - set(listed))
    if unregistered:
        problems.extend([f"unregistered:{rel}" for rel in unregistered])
    manifest_hash = file_sha256(manifest_path)
    if manifest_hash != root_record["payload_manifest_sha256"]:
        problems.append("payload_manifest_sha_mismatch")
    digest = payload_root_digest(entries)
    if digest != root_record["payload_root_digest_sha256"]:
        problems.append("payload_root_digest_mismatch")
    return {
        "passed": not problems,
        "problem_count": len(problems),
        "problems": problems,
        "payload_file_count": len(entries),
        "payload_manifest_sha256": manifest_hash,
        "payload_root_digest_sha256": digest,
    }


def detect_pixel_domain_release_token(release_base: Path) -> str:
    if (release_base / "v1_2_1_release_file_manifest.json").exists():
        return "v1_2_1"
    if (release_base / "v1_2_release_file_manifest.json").exists():
        return "v1_2"
    raise FileNotFoundError(f"No v1.2/v1.2.1 release manifest found in {release_base}")


def release_sidecar_name(token: str, stem: str, suffix: str) -> str:
    return f"{stem}_{token}.{suffix}"


def colmap_camera_to_record(camera_id: int, camera: Any) -> CameraRecord:
    return CameraRecord(
        camera_id=int(camera_id),
        model=str(camera.model),
        width=int(camera.width),
        height=int(camera.height),
        params=tuple(float(x) for x in camera.params),
    )


def colmap_image_to_record(image_id: int, image: Any) -> ImageRecord:
    return ImageRecord(
        image_id=int(image_id),
        image_name=Path(str(image.name)).name,
        camera_id=int(image.camera_id),
        qvec=tuple(float(x) for x in image.qvec),  # type: ignore[arg-type]
        tvec=tuple(float(x) for x in image.tvec),  # type: ignore[arg-type]
    )


def load_release_v12_sidecars(release_base: Path) -> dict[str, Any]:
    token = detect_pixel_domain_release_token(release_base)
    sidecars = {
        "release_token": token,
        "projection": json.loads((release_base / release_sidecar_name(token, "projection_manifest", "json")).read_text(encoding="utf-8")),
        "orientation": json.loads((release_base / release_sidecar_name(token, "raw_image_orientation_manifest", "json")).read_text(encoding="utf-8")),
        "mapping": json.loads((release_base / release_sidecar_name(token, "source_target_mapping_manifest", "json")).read_text(encoding="utf-8")),
        "camera": json.loads((release_base / release_sidecar_name(token, "camera_provenance_manifest", "json")).read_text(encoding="utf-8")),
        "root_record": json.loads((release_base / f"{token}_release_root_digest.json").read_text(encoding="utf-8")),
        "payload_manifest": json.loads((release_base / f"{token}_release_file_manifest.json").read_text(encoding="utf-8")),
    }
    integrity = verify_payload_integrity(
        release_base,
        release_base / f"{token}_release_file_manifest.json",
        release_base / f"{token}_release_root_digest.json",
    )
    if not integrity["passed"]:
        raise ValueError(f"v1.2 release integrity failed: {integrity}")
    sidecars["integrity"] = integrity
    return sidecars


def camera_provenance_lookup(camera_manifest: dict[str, Any]) -> tuple[dict[tuple[str, str, int], dict[str, Any]], dict[tuple[str, str, str], dict[str, Any]]]:
    camera_lookup: dict[tuple[str, str, int], dict[str, Any]] = {}
    pose_lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for scene, scene_payload in camera_manifest.get("scenes", {}).items():
        for role, model_key in [("source", "source_model"), ("target", "target_model")]:
            model = scene_payload.get(model_key, {})
            for raw in model.get("cameras", []):
                key = (scene, role, int(raw["camera_id"]))
                if key in camera_lookup:
                    raise ValueError(f"duplicate camera provenance record: {key}")
                camera_lookup[key] = raw
            for raw in model.get("images", []):
                key = (scene, role, Path(str(raw["image_name"])).name)
                if key in pose_lookup:
                    raise ValueError(f"duplicate pose provenance record: {key}")
                pose_lookup[key] = raw
    return camera_lookup, pose_lookup


def _rows_by_key(rows: Sequence[dict[str, Any]], keys: Sequence[str]) -> dict[tuple[str, ...], dict[str, Any]]:
    result = {}
    for row in rows:
        key = tuple(str(row.get(k, "")) for k in keys)
        if key in result:
            raise ValueError(f"duplicate v1.2 sidecar key: {keys}={key}")
        result[key] = dict(row)
    return result


def validate_release_v12_rows_for_evaluator(
    *,
    release_base: Path,
    scene: str,
    rows: Sequence[dict[str, str]],
    colmap_cameras: dict[int, Any],
    colmap_images: dict[int, Any],
    depth_manifest: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    sidecars = load_release_v12_sidecars(release_base)
    orientation = _rows_by_key(sidecars["orientation"], ["scene", "image_name"])
    mapping = _rows_by_key(sidecars["mapping"], ["scene", "source_image_name", "target_image_name"])
    camera_lookup, pose_lookup = camera_provenance_lookup(sidecars["camera"])
    strict_provenance = sidecars.get("release_token") == "v1_2_1"
    cameras = {int(cid): colmap_camera_to_record(int(cid), cam) for cid, cam in colmap_cameras.items()}
    images = {Path(str(img.name)).name: colmap_image_to_record(int(iid), img) for iid, img in colmap_images.items()}
    if depth_manifest is not None:
        for required in ["target_cameras_bin_sha256", "target_images_bin_sha256", "pixel_coordinate_convention"]:
            if required not in depth_manifest:
                raise ValueError(f"v1.2 release mode requires depth manifest field {required}")
        if str(depth_manifest.get("pixel_coordinate_convention")) != PIXEL_CONVENTION:
            raise ValueError(f"depth manifest pixel convention mismatch: {depth_manifest.get('pixel_coordinate_convention')}")
    validated: list[dict[str, str]] = []
    for row in rows:
        if row.get("scene") != scene:
            raise ValueError(f"annotation scene mismatch: expected {scene}, got {row.get('scene')}")
        image_name = Path(str(row["raw_image_name"])).name
        orient = orientation.get((scene, image_name))
        if orient is None:
            raise ValueError(f"missing orientation record for {scene} {image_name}")
        if orient.get("raw_image_sha256", "").lower() != row.get("source_image_sha256", "").lower():
            raise ValueError(f"source image SHA mismatch for {scene} {image_name}")
        expected_observation_id = observation_id_from_fields(
            scene=scene,
            point_name=str(row["point_name"]),
            raw_image_name=image_name,
            raw_image_sha256=str(row["source_image_sha256"]),
            raw_manual_x_text=str(row["raw_manual_x"]),
            raw_manual_y_text=str(row["raw_manual_y"]),
        )
        if expected_observation_id != row.get("observation_id"):
            raise ValueError(f"observation_id mismatch for {scene} {image_name}")
        if orient.get("rgb_pixel_matrix_sha256") != row.get("source_rgb_pixel_matrix_sha256"):
            raise ValueError(f"source RGB pixel-matrix hash mismatch for {scene} {image_name}")
        if orient.get("applied_orientation_policy") != ORIENTATION_POLICY:
            raise ValueError(f"source orientation policy mismatch for {scene} {image_name}")
        if row.get("source_orientation_policy") != ORIENTATION_POLICY:
            raise ValueError(f"annotation source orientation policy mismatch for {scene} {image_name}")
        target_image = images.get(image_name)
        if target_image is None:
            raise ValueError(f"target image missing in evaluator COLMAP model: {scene} {image_name}")
        target_camera = cameras.get(target_image.camera_id)
        if target_camera is None:
            raise ValueError(f"target camera missing in evaluator COLMAP model: {scene} {image_name}")
        if depth_manifest is not None:
            if str(depth_manifest.get("target_cameras_bin_sha256")) != row.get("target_cameras_bin_sha256"):
                raise ValueError(f"depth manifest target cameras hash mismatch for {scene} {image_name}")
            if str(depth_manifest.get("target_images_bin_sha256")) != row.get("target_images_bin_sha256"):
                raise ValueError(f"depth manifest target images hash mismatch for {scene} {image_name}")
        if camera_record_hash(target_camera) != row.get("target_camera_record_sha256"):
            raise ValueError(f"target camera record hash mismatch for {scene} {image_name}")
        if image_pose_record_hash(target_image) != row.get("target_pose_record_sha256"):
            raise ValueError(f"target pose record hash mismatch for {scene} {image_name}")
        if strict_provenance:
            source_camera_record = camera_lookup.get((scene, "source", int(row["source_camera_id"])))
            target_camera_record = camera_lookup.get((scene, "target", int(row["target_camera_id"])))
            source_pose_record = pose_lookup.get((scene, "source", image_name))
            target_pose_record = pose_lookup.get((scene, "target", image_name))
            if source_camera_record is None or target_camera_record is None:
                raise ValueError(f"camera provenance record missing for {scene} {image_name}")
            if source_pose_record is None or target_pose_record is None:
                raise ValueError(f"pose provenance record missing for {scene} {image_name}")
            if source_camera_record.get("record_sha256") != row.get("source_camera_record_sha256"):
                raise ValueError(f"source camera provenance hash mismatch for {scene} {image_name}")
            if target_camera_record.get("record_sha256") != row.get("target_camera_record_sha256"):
                raise ValueError(f"target camera provenance hash mismatch for {scene} {image_name}")
            if source_pose_record.get("record_sha256") != row.get("source_pose_record_sha256"):
                raise ValueError(f"source pose provenance hash mismatch for {scene} {image_name}")
            if target_pose_record.get("record_sha256") != row.get("target_pose_record_sha256"):
                raise ValueError(f"target pose provenance hash mismatch for {scene} {image_name}")
        if int(row["target_image_width"]) != int(target_camera.width) or int(row["target_image_height"]) != int(target_camera.height):
            raise ValueError(f"target dimensions mismatch for {scene} {image_name}")
        m = mapping.get((scene, image_name, image_name))
        if m is None:
            raise ValueError(f"missing source-target mapping record for {scene} {image_name}")
        m_without_hash = dict(m)
        claimed_mapping_hash = str(m_without_hash.pop("source_target_mapping_record_sha256", ""))
        if canonical_record_sha256(m_without_hash) != claimed_mapping_hash:
            raise ValueError(f"mapping record self-hash mismatch for {scene} {image_name}")
        if m.get("source_target_mapping_record_sha256") != row.get("source_target_mapping_record_sha256"):
            raise ValueError(f"source-target mapping hash mismatch for {scene} {image_name}")
        if str(m.get("source_image_sha256", "")).lower() != str(row.get("source_image_sha256", "")).lower():
            raise ValueError(f"mapping source image SHA mismatch for {scene} {image_name}")
        if str(m.get("target_image_sha256", "")).lower() != str(row.get("target_image_sha256", "")).lower():
            raise ValueError(f"mapping target image SHA mismatch for {scene} {image_name}")
        for field in [
            "source_camera_record_sha256",
            "target_camera_record_sha256",
            "source_pose_record_sha256",
            "target_pose_record_sha256",
        ]:
            if str(m.get(field, "")) != str(row.get(field, "")):
                raise ValueError(f"mapping {field} mismatch for {scene} {image_name}")
        if m.get("transform_version") != row.get("transform_version") or row.get("transform_version") != TRANSFORM_VERSION:
            raise ValueError(f"transform version mismatch for {scene} {image_name}")
        if str(m.get("mapping_type")) != str(row.get("mapping_type")):
            raise ValueError(f"mapping type mismatch for {scene} {image_name}")
        if not bool(m.get("pose_equivalent", False)):
            raise ValueError(f"source-target pose is not equivalent for {scene} {image_name}")
        source_camera = CameraRecord(
            camera_id=int(row["source_camera_id"]),
            model=row["source_camera_model"],
            width=int(row["source_camera_width"]),
            height=int(row["source_camera_height"]),
            params=tuple(float(x) for x in row["source_camera_params"].split(";")),
        )
        if camera_record_hash(source_camera) != row.get("source_camera_record_sha256"):
            raise ValueError(f"source camera record hash mismatch for {scene} {image_name}")
        projection = raw_to_target_projection(source_camera, target_camera, float(row["raw_manual_x"]), float(row["raw_manual_y"]))
        dx = abs(projection["target_x"] - float(row["target_x"]))
        dy = abs(projection["target_y"] - float(row["target_y"]))
        if dx > CACHED_TARGET_TOL_PX or dy > CACHED_TARGET_TOL_PX:
            raise ValueError(f"cached target projection mismatch for {scene} {image_name}: dx={dx} dy={dy}")
        if projection["roundtrip_error_px"] > ROUNDTRIP_TOL_PX:
            raise ValueError(f"roundtrip error exceeds tolerance for {scene} {image_name}: {projection['roundtrip_error_px']}")
        if not (0.0 <= float(row["target_x"]) < target_camera.width and 0.0 <= float(row["target_y"]) < target_camera.height):
            raise ValueError(f"target coordinates out of bounds for {scene} {image_name}")
        out = dict(row)
        out["u_px"] = row["target_x"]
        out["v_px"] = row["target_y"]
        out["manual_x"] = row["target_x"]
        out["manual_y"] = row["target_y"]
        out["image_name"] = row.get("target_image_name", image_name)
        out["raw_manual_x"] = row["raw_manual_x"]
        out["raw_manual_y"] = row["raw_manual_y"]
        validated.append(out)
    return validated
