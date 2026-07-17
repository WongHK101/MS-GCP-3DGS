from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gcp_pixel_domain_v1_2 import (  # noqa: E402
    CameraRecord,
    PIXEL_CONVENTION,
    camera_record_hash,
    canonical_record_sha256,
    canonical_records_root_sha256,
    fmt_float,
    project_pinhole,
    read_csv,
    verify_payload_integrity,
)


# Compatibility v1.1 and packet v2 identifiers are frozen wire tokens.
COMPAT_SCHEMA = "ms_gcp_metric_depth_packet_resolution_compatibility_v1_1"
WITHDRAWN_COMPAT_SCHEMA_V1 = "ms_gcp_metric_depth_packet_resolution_compatibility_v1"
WITHDRAWN_COMPAT_V1_DISPOSITION = "withdrawn_before_stage3_due_to_uncommitted_generator_and_incomplete_runtime_validation"
COMPAT_TRANSFORM_VERSION = "benchmark_pinhole_to_packet_native_camera_v1"
PACKET_PIXEL_DOMAIN = "metric_depth_packet_native_render_pixel_domain"
CANONICAL_PIXEL_CONVENTION = PIXEL_CONVENTION
ACCEPTED_PIXEL_CONVENTION_ALIASES = {
    "zero_based_pixel_centers": "zero_based_pixel_centers",
    "zero_indexed_pixel_centers": "zero_based_pixel_centers",
}
PATCH_PROTOCOL = "native_packet_pixel_patch_v1"
PROJECTION_TOL_PX = 1e-9
RAY_COORD_TOL = 1e-12
RAY_ANGLE_TOL_RAD = 1e-7
IMPLICIT_MAPPING_GATE_PX = 1e-6
CAMERA_INTRINSIC_TOL = 1e-9
MATRIX_EQUIVALENCE_TOL_PX = 1e-9
POSE_CENTER_TOL_MODEL_UNITS = 1e-8
POSE_ROTATION_TOL_RAD = 1e-8
METRIC_PACKET_MANIFEST_SCHEMA = "ms_gcp_metric_depth_packet_manifest_v2"
METRIC_PACKET_SCHEMA = "ms_gcp_metric_depth_packet_v2"
PRIMARY_DEPTH_TENSOR = "alpha_normalized_expected_camera_z"
PRIMARY_DEPTH_SEMANTICS = "camera_z"
FORMAL_DEPTH_FORMULA = "M1/A"
PACKET_REF_CONSISTENCY_PROTOCOL = "raw_accumulator_recompute_v2_with_variance_forward_error_bound"
FORMULA_SOURCE_EXPLICIT_FIELD = "manifest_explicit_field"
FORMULA_SOURCE_TENSOR_FORMULAS = "manifest_tensor_formulas"
FORMULA_SOURCE_SCHEMA_IMPLIED = "packet_schema_implied_contract"
APPROVED_FORMULA_COMPACT_ALLOWLIST = {
    "M1/A",
    "M1/AforA>floorelseNaN",
}
PACKET_REF_NUMERIC_TOL = 1e-9
REQUIRED_TENSOR_DTYPES = {
    "accumulated_alpha": "float32",
    "weighted_camera_z_sum": "float32",
    "weighted_camera_z_second_moment": "float32",
    "weighted_inverse_camera_z_sum": "float32",
    "alpha_normalized_expected_camera_z": "float32",
    "alpha_normalized_expected_inverse_camera_z": "float32",
    "harmonic_camera_z": "float32",
    "camera_z_variance": "float32",
    "metric_depth_valid_mask": "bool",
    "historical_invalid_unnormalized_inverse_depth": "float32",
}
POSE_CONVENTION_VERSION = "cameras_json_c2w_position_to_colmap_w2c_v1"


@dataclass(frozen=True)
class PacketCamera:
    scene: str
    image_name: str
    camera_id: int
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    resolution_value: int
    resolution_scale: float
    source_width: int
    source_height: int
    source_fx: float
    source_fy: float
    fovx: float
    fovy: float
    rounding_rule: str
    principal_point_rule: str
    resize_rule: str
    crop_pad_policy: str
    pixel_convention_original: str
    pixel_convention_canonical: str


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_detached_sha256(path: Path) -> Path:
    sha_path = path.with_suffix(path.suffix + ".sha256")
    sha_path.write_text(f"{file_sha256(path)}  {path.name}\n", encoding="utf-8")
    return sha_path


def verify_detached_sha256(path: Path) -> dict[str, Any]:
    sha_path = path.with_suffix(path.suffix + ".sha256")
    if not sha_path.exists():
        raise ValueError(f"missing detached sha256 file: {sha_path}")
    token = sha_path.read_text(encoding="utf-8").strip().split()[0]
    actual = file_sha256(path)
    passed = token.lower() == actual.lower()
    if not passed:
        raise ValueError(f"detached sha256 mismatch for {path}: {token} vs {actual}")
    return {"path": str(path), "sha256_path": str(sha_path), "sha256": actual, "passed": True}


def validate_approved_sha256(value: str | None) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        raise ValueError("approved packet compatibility wrapper SHA-256 is required")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", text):
        raise ValueError(f"approved packet compatibility wrapper SHA-256 must be 64 hex characters: {text!r}")
    return text.lower()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={repo.as_posix()}", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return ""


def git_status_porcelain(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={repo.as_posix()}", "-C", str(repo), "status", "--porcelain=v1"],
            text=True,
        ).strip()
    except Exception as exc:  # noqa: BLE001
        return f"ERROR:{type(exc).__name__}:{exc}"


def git_show_file_sha256(repo: Path, rel_path: str) -> str:
    try:
        payload = subprocess.check_output(
            ["git", "-c", f"safe.directory={repo.as_posix()}", "-C", str(repo), "show", f"HEAD:{rel_path}"]
        )
        return sha256_bytes(payload)
    except Exception:
        path = repo / rel_path
        return file_sha256(path) if path.exists() else ""


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_cfg_args(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8").strip()
    if not text.startswith("Namespace(") or not text.endswith(")"):
        raise ValueError(f"Unsupported cfg_args format: {path}")
    body = text[len("Namespace(") : -1]
    out: dict[str, Any] = {}
    for match in re.finditer(r"(\w+)=", body):
        key = match.group(1)
        start = match.end()
        next_match = re.search(r",\s*\w+=", body[start:])
        end = len(body) if next_match is None else start + next_match.start()
        raw_value = body[start:end].strip().rstrip(",")
        try:
            out[key] = ast.literal_eval(raw_value)
        except Exception:
            out[key] = raw_value
    return out


def fov2focal(fov: float, pixels: int) -> float:
    return float(pixels) / (2.0 * math.tan(float(fov) / 2.0))


def focal2fov(focal: float, pixels: int) -> float:
    return 2.0 * math.atan(float(pixels) / (2.0 * float(focal)))


def normalize_pixel_convention(value: str) -> tuple[str, str]:
    original = str(value or "").strip()
    if original not in ACCEPTED_PIXEL_CONVENTION_ALIASES:
        raise ValueError(f"Unsupported packet pixel convention token: {original!r}")
    return original, ACCEPTED_PIXEL_CONVENTION_ALIASES[original]


def packet_camera_record(camera: PacketCamera) -> dict[str, Any]:
    return {
        "camera_id": int(camera.camera_id),
        "height": int(camera.height),
        "model": "PINHOLE",
        "params": [fmt_float(camera.fx), fmt_float(camera.fy), fmt_float(camera.cx), fmt_float(camera.cy)],
        "width": int(camera.width),
    }


def packet_camera_hash(camera: PacketCamera) -> str:
    return canonical_record_sha256(packet_camera_record(camera))


def intrinsic_ray(camera: CameraRecord | PacketCamera, u: float, v: float) -> tuple[float, float]:
    if isinstance(camera, PacketCamera):
        fx, fy, cx, cy = camera.fx, camera.fy, camera.cx, camera.cy
    else:
        if camera.model.upper() != "PINHOLE" or len(camera.params) != 4:
            raise ValueError(f"Expected PINHOLE camera, got {camera.model} {camera.params}")
        fx, fy, cx, cy = [float(x) for x in camera.params]
    return (float(u) - cx) / fx, (float(v) - cy) / fy


def project_packet(camera: PacketCamera, x_norm: float, y_norm: float) -> tuple[float, float]:
    return camera.fx * float(x_norm) + camera.cx, camera.fy * float(y_norm) + camera.cy


def unit_ray(x_norm: float, y_norm: float) -> np.ndarray:
    vec = np.asarray([float(x_norm), float(y_norm), 1.0], dtype=np.float64)
    vec /= np.linalg.norm(vec)
    return vec


def ray_angle(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(np.dot(a, b))
    dot = max(-1.0, min(1.0, dot))
    return float(math.acos(dot))


def qvec_to_rotmat(qvec: Sequence[float]) -> np.ndarray:
    qw, qx, qy, qz = [float(x) for x in qvec]
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm == 0.0:
        raise ValueError("zero quaternion")
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    return np.asarray(
        [
            [1.0 - 2.0 * qy * qy - 2.0 * qz * qz, 2.0 * qx * qy - 2.0 * qz * qw, 2.0 * qx * qz + 2.0 * qy * qw],
            [2.0 * qx * qy + 2.0 * qz * qw, 1.0 - 2.0 * qx * qx - 2.0 * qz * qz, 2.0 * qy * qz - 2.0 * qx * qw],
            [2.0 * qx * qz - 2.0 * qy * qw, 2.0 * qy * qz + 2.0 * qx * qw, 1.0 - 2.0 * qx * qx - 2.0 * qy * qy],
        ],
        dtype=np.float64,
    )


def rotmat_to_qvec(rot: np.ndarray) -> tuple[float, float, float, float]:
    m = np.asarray(rot, dtype=np.float64)
    q = np.empty(4, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q[0] = 0.25 * s
        q[1] = (m[2, 1] - m[1, 2]) / s
        q[2] = (m[0, 2] - m[2, 0]) / s
        q[3] = (m[1, 0] - m[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(m)))
        if i == 0:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            q[0] = (m[2, 1] - m[1, 2]) / s
            q[1] = 0.25 * s
            q[2] = (m[0, 1] + m[1, 0]) / s
            q[3] = (m[0, 2] + m[2, 0]) / s
        elif i == 1:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            q[0] = (m[0, 2] - m[2, 0]) / s
            q[1] = (m[0, 1] + m[1, 0]) / s
            q[2] = 0.25 * s
            q[3] = (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            q[0] = (m[1, 0] - m[0, 1]) / s
            q[1] = (m[0, 2] + m[2, 0]) / s
            q[2] = (m[1, 2] + m[2, 1]) / s
            q[3] = 0.25 * s
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q *= -1.0
    return tuple(float(x) for x in q)


def camera_center_from_colmap(qvec: Sequence[float], tvec: Sequence[float]) -> np.ndarray:
    rot = qvec_to_rotmat(qvec)
    t = np.asarray([float(x) for x in tvec], dtype=np.float64)
    return -rot.T @ t


def rotation_angle_rad(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
    def _orthonormalize(rot: np.ndarray) -> np.ndarray:
        u, _s, vh = np.linalg.svd(np.asarray(rot, dtype=np.float64))
        out = u @ vh
        if np.linalg.det(out) < 0:
            u[:, -1] *= -1.0
            out = u @ vh
        return out

    delta = _orthonormalize(rot_a) @ _orthonormalize(rot_b).T
    value = (float(np.trace(delta)) - 1.0) / 2.0
    if 0.0 <= 1.0 - value <= 8.0 * np.finfo(np.float64).eps:
        value = 1.0
    return float(math.acos(max(-1.0, min(1.0, value))))


def parse_float_sequence(values: Sequence[Any] | str) -> list[float]:
    if isinstance(values, str):
        if ";" in values:
            return [float(x) for x in values.split(";") if x != ""]
        return [float(x) for x in values.split() if x != ""]
    return [float(x) for x in values]


def format_float_list(values: Sequence[float]) -> list[str]:
    return [fmt_float(float(x)) for x in values]


def pose_record_hash(record: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(record))


def pose_from_cameras_json(model_camera_row: dict[str, Any], target_pose_row: dict[str, Any]) -> dict[str, Any]:
    rotation_raw = model_camera_row.get("rotation")
    position_raw = model_camera_row.get("position")
    if rotation_raw is None or position_raw is None:
        raise ValueError(f"cameras.json row missing rotation/position for {model_camera_row.get('img_name')}")
    r_c2w = np.asarray(rotation_raw, dtype=np.float64)
    if r_c2w.shape != (3, 3):
        raise ValueError(f"cameras.json rotation is not 3x3 for {model_camera_row.get('img_name')}")
    center = np.asarray(position_raw, dtype=np.float64)
    if center.shape != (3,):
        raise ValueError(f"cameras.json position is not length-3 for {model_camera_row.get('img_name')}")
    r_w2c = r_c2w.T
    tvec = -r_w2c @ center
    qvec = rotmat_to_qvec(r_w2c)
    target_qvec = parse_float_sequence(target_pose_row["qvec"])
    target_tvec = parse_float_sequence(target_pose_row["tvec"])
    target_r_w2c = qvec_to_rotmat(target_qvec)
    target_center = camera_center_from_colmap(target_qvec, target_tvec)
    center_diff = float(np.linalg.norm(center - target_center))
    canonical_r_w2c = qvec_to_rotmat(qvec)
    angle = rotation_angle_rad(canonical_r_w2c, target_r_w2c)
    record = {
        "camera_id": int(target_pose_row["camera_id"]),
        "camera_center": format_float_list(center),
        "image_id": int(target_pose_row["image_id"]),
        "image_name": str(target_pose_row["image_name"]),
        "pose_convention": POSE_CONVENTION_VERSION,
        "qvec": format_float_list(qvec),
        "tvec": format_float_list(tvec),
    }
    passed = bool(center_diff <= POSE_CENTER_TOL_MODEL_UNITS and angle <= POSE_ROTATION_TOL_RAD)
    return {
        "passed": passed,
        "pose_convention": POSE_CONVENTION_VERSION,
        "cameras_json_rotation_semantics": "camera_to_world_row_major_3x3",
        "cameras_json_position_semantics": "camera_center_model_world_coordinates",
        "colmap_pose_semantics": "world_to_camera_qvec_qw_qx_qy_qz_and_tvec",
        "axis_flip_policy": "none",
        "r_c2w": [[float(x) for x in row] for row in r_c2w.tolist()],
        "position_camera_center": [float(x) for x in center.tolist()],
        "r_w2c": [[float(x) for x in row] for row in r_w2c.tolist()],
        "tvec": [float(x) for x in tvec.tolist()],
        "qvec": [float(x) for x in qvec],
        "target_qvec": [float(x) for x in target_qvec],
        "target_tvec": [float(x) for x in target_tvec],
        "target_camera_center": [float(x) for x in target_center.tolist()],
        "center_difference_model_units": center_diff,
        "center_tolerance_model_units": POSE_CENTER_TOL_MODEL_UNITS,
        "rotation_angular_difference_rad": angle,
        "rotation_tolerance_rad": POSE_ROTATION_TOL_RAD,
        "pose_record": record,
        "pose_record_sha256": pose_record_hash(record),
        "target_pose_record_sha256": str(target_pose_row.get("record_sha256", "")),
    }


def renderer_projection_matrix(fovx: float, fovy: float, znear: float = 0.01, zfar: float = 100.0) -> np.ndarray:
    tan_half_y = math.tan(fovy / 2.0)
    tan_half_x = math.tan(fovx / 2.0)
    top = tan_half_y * znear
    bottom = -top
    right = tan_half_x * znear
    left = -right
    matrix = np.zeros((4, 4), dtype=np.float64)
    matrix[0, 0] = 2.0 * znear / (right - left)
    matrix[1, 1] = 2.0 * znear / (top - bottom)
    matrix[0, 2] = (right + left) / (right - left)
    matrix[1, 2] = (top + bottom) / (top - bottom)
    matrix[3, 2] = 1.0
    matrix[2, 2] = zfar / (zfar - znear)
    matrix[2, 3] = -(zfar * znear) / (zfar - znear)
    return matrix


def renderer_matrix_to_pixel(camera: PacketCamera, x_norm: float, y_norm: float, z: float = 10.0) -> tuple[float, float]:
    # scene.cameras.Camera stores getProjectionMatrix(...).transpose(0, 1)
    # and the renderer utility multiplies row vectors by that transposed matrix.
    matrix = renderer_projection_matrix(camera.fovx, camera.fovy).T
    point = np.asarray([x_norm * z, y_norm * z, z, 1.0], dtype=np.float64)
    clip = point @ matrix
    ndc = clip[:3] / clip[3]
    # This is the symmetric-frustum pixel-center convention implied by the renderer projection.
    u = (ndc[0] + 1.0) * camera.width / 2.0
    v = (ndc[1] + 1.0) * camera.height / 2.0
    return float(u), float(v)


def projection_matrix_equivalence(camera: PacketCamera) -> dict[str, Any]:
    samples = [
        (0.0, 0.0),
        (-0.15, -0.11),
        (0.17, -0.09),
        (-0.13, 0.12),
        (0.19, 0.13),
        (0.0712345, -0.043219),
    ]
    rows = []
    max_error = 0.0
    for x_norm, y_norm in samples:
        matrix_u, matrix_v = renderer_matrix_to_pixel(camera, x_norm, y_norm)
        intr_u, intr_v = project_packet(camera, x_norm, y_norm)
        error = math.hypot(matrix_u - intr_u, matrix_v - intr_v)
        max_error = max(max_error, error)
        rows.append(
            {
                "x_norm": x_norm,
                "y_norm": y_norm,
                "renderer_matrix_u": matrix_u,
                "renderer_matrix_v": matrix_v,
                "intrinsic_u": intr_u,
                "intrinsic_v": intr_v,
                "error_px": error,
            }
        )
    return {
        "passed": bool(max_error <= MATRIX_EQUIVALENCE_TOL_PX),
        "max_error_px": max_error,
        "tolerance_px": MATRIX_EQUIVALENCE_TOL_PX,
        "samples": rows,
    }


def npz_member_headers_header_only(path: Path) -> dict[str, dict[str, Any]]:
    headers: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(path, "r") as zf:
        members = sorted(name for name in zf.namelist() if name.endswith(".npy"))
        if not members:
            raise ValueError(f"NPZ has no .npy members: {path}")
        for member in members:
            key = Path(member).stem
            info = zf.getinfo(member)
            with zf.open(member, "r") as f:
                version = np.lib.format.read_magic(f)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
                elif version == (2, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
                else:
                    shape, fortran, dtype = np.lib.format._read_array_header(f, version)  # type: ignore[attr-defined]
            headers[key] = {
                "shape": [int(x) for x in shape],
                "dtype": str(np.dtype(dtype)),
                "fortran_order": bool(fortran),
                "zip_member": member,
                "zip_compressed_size": int(info.compress_size),
                "zip_file_size": int(info.file_size),
            }
    return headers


def npz_member_shape_header_only(path: Path, preferred_key: str = PRIMARY_DEPTH_TENSOR) -> tuple[int, int]:
    headers = npz_member_headers_header_only(path)
    member = preferred_key if preferred_key in headers else sorted(headers)[0]
    shape = headers[member]["shape"]
    if len(shape) != 2:
        raise ValueError(f"Expected 2D packet tensor header in {path}, got shape={shape}")
    return int(shape[0]), int(shape[1])


def validate_npz_packet_headers(
    *,
    path: Path,
    expected_width: int,
    expected_height: int,
    expected_dtype: str,
) -> dict[str, Any]:
    headers = npz_member_headers_header_only(path)
    missing = sorted(set(REQUIRED_TENSOR_DTYPES) - set(headers))
    if missing:
        raise ValueError(f"packet required tensor missing in {path.name}: {missing}")
    shape = [int(expected_height), int(expected_width)]
    checked: dict[str, dict[str, Any]] = {}
    for name, dtype in REQUIRED_TENSOR_DTYPES.items():
        header = headers[name]
        if header["shape"] != shape:
            raise ValueError(f"packet tensor shape mismatch {path.name}:{name}: {header['shape']} vs {shape}")
        if header["dtype"] != dtype:
            raise ValueError(f"packet tensor dtype mismatch {path.name}:{name}: {header['dtype']} vs {dtype}")
        if name != "metric_depth_valid_mask" and expected_dtype and header["dtype"] != expected_dtype:
            raise ValueError(f"packet tensor dtype mismatch against manifest {path.name}:{name}: {header['dtype']} vs {expected_dtype}")
        checked[name] = header
    return {
        "packet_path": str(path),
        "checked_tensor_count": len(checked),
        "required_tensors": sorted(REQUIRED_TENSOR_DTYPES),
        "headers": checked,
        "depth_tensor_values_read": False,
    }


def resolve_packet_path(packet_path: str, search_roots: Sequence[Path]) -> Path | None:
    raw = str(packet_path)
    candidate = Path(raw)
    foreign_absolute = (
        (os.name == "nt" and raw.startswith("/") and not raw.startswith("//"))
        or (os.name != "nt" and PureWindowsPath(raw).is_absolute())
    )
    if not foreign_absolute and candidate.exists():
        return candidate
    name = candidate.name
    for root in search_roots:
        if not root.exists():
            continue
        matches = list(root.rglob(name))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous local packet matches for {name}: {matches[:5]}")
    return None


def recover_packet_camera(
    *,
    scene: str,
    image_name: str,
    target_camera: CameraRecord,
    model_camera_row: dict[str, Any],
    packet_width: int,
    packet_height: int,
    cfg_args: dict[str, Any],
    manifest_pixel_convention: str,
) -> PacketCamera:
    resolution = cfg_args.get("resolution")
    if not isinstance(resolution, int) or resolution not in {1, 2, 4, 8}:
        raise ValueError(f"Only audited integer resolution values [1,2,4,8] are supported, got {resolution!r}")
    resolution_scale = 1.0
    source_width = int(model_camera_row["width"])
    source_height = int(model_camera_row["height"])
    expected_width = round(source_width / (resolution_scale * resolution))
    expected_height = round(source_height / (resolution_scale * resolution))
    if (expected_width, expected_height) != (int(packet_width), int(packet_height)):
        raise ValueError(
            f"Packet shape does not match renderer rounding rule for {scene} {image_name}: "
            f"expected {expected_width}x{expected_height}, got {packet_width}x{packet_height}"
        )
    source_fx = float(model_camera_row["fx"])
    source_fy = float(model_camera_row["fy"])
    fovx = focal2fov(source_fx, source_width)
    fovy = focal2fov(source_fy, source_height)
    fx = fov2focal(fovx, packet_width)
    fy = fov2focal(fovy, packet_height)
    # The renderer source uses a symmetric frustum with P[0,2]=P[1,2]=0; this
    # implies an image-center principal point in the packet-native raster.
    cx = packet_width / 2.0
    cy = packet_height / 2.0
    original_conv, canonical_conv = normalize_pixel_convention(manifest_pixel_convention)
    camera = PacketCamera(
        scene=scene,
        image_name=Path(image_name).name,
        camera_id=int(target_camera.camera_id),
        width=int(packet_width),
        height=int(packet_height),
        fx=float(fx),
        fy=float(fy),
        cx=float(cx),
        cy=float(cy),
        resolution_value=int(resolution),
        resolution_scale=float(resolution_scale),
        source_width=source_width,
        source_height=source_height,
        source_fx=source_fx,
        source_fy=source_fy,
        fovx=float(fovx),
        fovy=float(fovy),
        rounding_rule="python_builtin_round_ties_to_even_on_width_and_height_independently",
        principal_point_rule="renderer_symmetric_projection_matrix_image_center_width_over_2_height_over_2",
        resize_rule="utils.camera_utils.loadCam_integer_resolution_round_orig_size_div_resolution",
        crop_pad_policy="none_verified_by_renderer_camera_loader_source",
        pixel_convention_original=original_conv,
        pixel_convention_canonical=canonical_conv,
    )
    matrix_check = projection_matrix_equivalence(camera)
    if not matrix_check["passed"]:
        raise ValueError(f"Renderer matrix equivalence failed for {scene} {image_name}: {matrix_check}")
    return camera


def load_model_cameras(path: Path) -> dict[str, dict[str, Any]]:
    payload = load_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"cameras.json must be a list: {path}")
    out: dict[str, dict[str, Any]] = {}
    for row in payload:
        name = Path(str(row.get("img_name", ""))).name
        if not name:
            continue
        if name in out:
            raise ValueError(f"duplicate cameras.json image name: {name}")
        out[name] = dict(row)
    return out


def model_camera_for_image(model_cameras: dict[str, dict[str, Any]], image_name: str) -> dict[str, Any]:
    aliases = {str(image_name), Path(str(image_name)).name, Path(str(image_name)).stem}
    matches = [row for key, row in model_cameras.items() if key in aliases or Path(key).stem in aliases]
    unique = {id(row): row for row in matches}
    if len(unique) != 1:
        raise ValueError(f"cannot uniquely resolve cameras.json image {image_name!r}: {sorted(model_cameras)[:8]}")
    return next(iter(unique.values()))


def resolve_release_payload_path(release_dir: Path, value: str, label: str) -> Path:
    relative = Path(str(value))
    if not str(value).strip() or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"invalid release-owned {label} path: {value!r}")
    path = (release_dir / relative).resolve()
    if path.parent != release_dir.resolve():
        raise ValueError(f"release-owned {label} must be in the release root: {value!r}")
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def resolve_scene_annotation_path(release_dir: Path, release_config: dict[str, Any], scene: str) -> Path:
    pattern = str(release_config.get("annotation_csv_pattern", "")).strip()
    if not pattern:
        raise ValueError("release config has no annotation_csv_pattern")
    matches = [
        path
        for path in release_dir.glob(pattern)
        if path.is_file() and path.name.startswith(f"{scene}_gcp_annotations_")
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one release annotation for {scene}, found {matches}")
    return matches[0]


def load_release_rows(
    release_dir: Path,
    scene: str,
    release_config: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    if release_config is None:
        path = release_dir / f"{scene}_gcp_annotations_pixel_domain_v1_2_2.csv"
    else:
        path = resolve_scene_annotation_path(release_dir, release_config, scene)
    if not path.exists():
        raise FileNotFoundError(path)
    rows = read_csv(path)
    if rows and "formal_eligible" in rows[0]:
        rows = [row for row in rows if str(row.get("formal_eligible", "")).strip().lower() == "true"]
    if not rows:
        raise ValueError(f"release has no formal annotation rows for {scene}: {path}")
    return rows


def camera_from_release_row(row: dict[str, str]) -> CameraRecord:
    return CameraRecord(
        camera_id=int(row["target_camera_id"]),
        model=row["target_camera_model"],
        width=int(row["target_camera_width"]),
        height=int(row["target_camera_height"]),
        params=tuple(float(x) for x in str(row["target_camera_params"]).split(";")),
    )


def depth_index_by_name(depth_manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in depth_manifest.get("depth_index", []):
        name = Path(str(row.get("image_name", ""))).name
        if not name:
            continue
        if name in result:
            raise ValueError(f"duplicate depth_index image_name: {name}")
        result[name] = dict(row)
    if not result:
        raise ValueError("depth_manifest has no depth_index records")
    return result


def metric_packet_contract_from_manifest(depth_manifest: dict[str, Any]) -> dict[str, Any]:
    tensor_names = depth_manifest.get("tensor_names", [])
    if isinstance(tensor_names, str):
        tensor_names = [x for x in tensor_names.split("|") if x]
    formula_source, formula_value, formula_normalized = formal_formula_from_manifest(depth_manifest)
    protocol_source, protocol_value = packet_ref_protocol_from_manifest(depth_manifest)
    return {
        "manifest_schema": depth_manifest.get("schema", ""),
        "packet_schema": depth_manifest.get("packet_schema", ""),
        "primary_depth_tensor": depth_manifest.get("primary_depth_tensor", ""),
        "primary_depth_semantics": depth_manifest.get("primary_depth_semantics", depth_manifest.get("depth_semantics", "")),
        "formal_depth_formula": formula_normalized,
        "formal_depth_formula_raw": formula_value,
        "formal_formula_source": formula_source,
        "dtype": depth_manifest.get("dtype", ""),
        "required_tensors": sorted(REQUIRED_TENSOR_DTYPES),
        "tensor_names": list(tensor_names),
        "packet_ref_consistency_protocol": protocol_value,
        "packet_ref_consistency_protocol_source": protocol_source,
        "packet_ref_consistency_required_fields": [
            "packet_recompute_passed",
            "variance_packet_ref_abs_error",
            "variance_packet_ref_allowed_error",
            "variance_packet_ref_consistency_ratio",
            "variance_consistency_fail_count",
        ],
    }


def formal_formula_from_manifest(depth_manifest: dict[str, Any]) -> tuple[str, str, str]:
    explicit_keys = ["formal_depth_formula", "primary_depth_formula", "formal_formula"]
    for key in explicit_keys:
        if key in depth_manifest and str(depth_manifest.get(key, "")).strip():
            value = str(depth_manifest[key]).strip()
            normalized = normalize_formal_formula(value)
            return FORMULA_SOURCE_EXPLICIT_FIELD, value, normalized
    tensor_formulas = depth_manifest.get("tensor_formulas", {})
    if isinstance(tensor_formulas, dict) and str(tensor_formulas.get(PRIMARY_DEPTH_TENSOR, "")).strip():
        value = str(tensor_formulas[PRIMARY_DEPTH_TENSOR]).strip()
        normalized = normalize_formal_formula(value)
        return FORMULA_SOURCE_TENSOR_FORMULAS, value, normalized
    if depth_manifest.get("packet_schema") == METRIC_PACKET_SCHEMA:
        return FORMULA_SOURCE_SCHEMA_IMPLIED, "", FORMAL_DEPTH_FORMULA
    raise ValueError(f"cannot infer formal formula from unknown packet schema: {depth_manifest.get('packet_schema')}")


def normalize_formal_formula(value: str) -> str:
    compact = str(value).strip().replace(" ", "")
    if compact in APPROVED_FORMULA_COMPACT_ALLOWLIST:
        return FORMAL_DEPTH_FORMULA
    raise ValueError(f"formal depth formula mismatch: {value!r}")


def packet_ref_protocol_from_manifest(depth_manifest: dict[str, Any]) -> tuple[str, str]:
    for key in ["packet_ref_consistency_protocol", "packet_recompute_protocol", "ref_consistency_protocol"]:
        if key in depth_manifest and str(depth_manifest.get(key, "")).strip():
            value = str(depth_manifest[key]).strip()
            if value != PACKET_REF_CONSISTENCY_PROTOCOL:
                raise ValueError(f"packet/ref consistency protocol mismatch: {value}")
            return "manifest_explicit_field", value
    if depth_manifest.get("packet_schema") == METRIC_PACKET_SCHEMA:
        return "packet_schema_implied_contract", PACKET_REF_CONSISTENCY_PROTOCOL
    raise ValueError(f"cannot infer packet/ref protocol from unknown packet schema: {depth_manifest.get('packet_schema')}")


def validate_depth_manifest_contract(depth_manifest: dict[str, Any]) -> dict[str, Any]:
    contract = metric_packet_contract_from_manifest(depth_manifest)
    if contract["manifest_schema"] != METRIC_PACKET_MANIFEST_SCHEMA:
        raise ValueError(f"metric packet manifest schema mismatch: {contract['manifest_schema']}")
    if contract["packet_schema"] != METRIC_PACKET_SCHEMA:
        raise ValueError(f"metric packet schema/version mismatch: {contract['packet_schema']}")
    if contract["primary_depth_tensor"] != PRIMARY_DEPTH_TENSOR:
        raise ValueError(f"primary tensor mismatch: {contract['primary_depth_tensor']}")
    if contract["primary_depth_semantics"] != PRIMARY_DEPTH_SEMANTICS:
        raise ValueError(f"formal depth semantics mismatch: {contract['primary_depth_semantics']}")
    if contract["formal_depth_formula"] != FORMAL_DEPTH_FORMULA:
        raise ValueError(f"formal formula mismatch: {contract['formal_depth_formula']}")
    if contract["packet_ref_consistency_protocol"] != PACKET_REF_CONSISTENCY_PROTOCOL:
        raise ValueError(f"packet/ref consistency protocol mismatch: {contract['packet_ref_consistency_protocol']}")
    tensor_names = set(contract["tensor_names"])
    missing = sorted(set(REQUIRED_TENSOR_DTYPES) - tensor_names)
    if missing:
        raise ValueError(f"depth manifest required tensor declarations missing: {missing}")
    if str(contract["dtype"]) != "float32":
        raise ValueError(f"depth manifest dtype mismatch: {contract['dtype']}")
    return contract


def validate_depth_index_row_contract(row: dict[str, Any], expected_width: int, expected_height: int) -> None:
    if str(row.get("primary_depth_tensor", "")) != PRIMARY_DEPTH_TENSOR:
        raise ValueError(f"depth index primary tensor mismatch for {row.get('image_name')}: {row.get('primary_depth_tensor')}")
    if str(row.get("primary_depth_semantics", "")) != PRIMARY_DEPTH_SEMANTICS:
        raise ValueError(f"depth index semantics mismatch for {row.get('image_name')}: {row.get('primary_depth_semantics')}")
    if str(row.get("dtype", "")) != "float32":
        raise ValueError(f"depth index dtype mismatch for {row.get('image_name')}: {row.get('dtype')}")
    if int(row.get("width", -1)) != int(expected_width) or int(row.get("height", -1)) != int(expected_height):
        raise ValueError(f"depth index shape mismatch for {row.get('image_name')}")
    tensor_names = set(str(row.get("tensor_names", "")).split("|"))
    missing = sorted(set(REQUIRED_TENSOR_DTYPES) - tensor_names)
    if missing:
        raise ValueError(f"depth index tensor declarations missing for {row.get('image_name')}: {missing}")
    packet_recompute = row.get("packet_recompute_passed")
    if not (packet_recompute is True or (isinstance(packet_recompute, str) and packet_recompute.strip().lower() == "true")):
        raise ValueError(f"packet/ref consistency did not pass for {row.get('image_name')}")
    required_ref_fields = [
        "variance_packet_ref_abs_error",
        "variance_packet_ref_allowed_error",
        "variance_packet_ref_consistency_ratio",
        "variance_consistency_fail_count",
    ]
    missing = [name for name in required_ref_fields if name not in row or str(row.get(name, "")).strip() == ""]
    if missing:
        raise ValueError(f"packet/ref required fields missing for {row.get('image_name')}: {missing}")
    abs_error = float(row["variance_packet_ref_abs_error"])
    allowed_error = float(row["variance_packet_ref_allowed_error"])
    ratio = float(row["variance_packet_ref_consistency_ratio"])
    fail_count_raw = row["variance_consistency_fail_count"]
    if isinstance(fail_count_raw, bool):
        raise ValueError(f"packet/ref failure count must be integer zero for {row.get('image_name')}: {fail_count_raw!r}")
    if isinstance(fail_count_raw, int):
        fail_count = fail_count_raw
    elif isinstance(fail_count_raw, str) and re.fullmatch(r"[+-]?\d+", fail_count_raw.strip()):
        fail_count = int(fail_count_raw.strip())
    else:
        raise ValueError(f"packet/ref failure count must be strict integer zero for {row.get('image_name')}: {fail_count_raw!r}")
    if not all(math.isfinite(v) for v in [abs_error, allowed_error, ratio]):
        raise ValueError(f"packet/ref numeric field is not finite for {row.get('image_name')}")
    if allowed_error < 0 or abs_error < 0 or fail_count != 0:
        raise ValueError(f"packet/ref consistency failure fields invalid for {row.get('image_name')}")
    if allowed_error == 0.0 and abs_error != 0.0:
        raise ValueError(f"packet/ref allowed error is zero with nonzero abs error for {row.get('image_name')}")
    if allowed_error > 0.0 and abs_error > allowed_error + PACKET_REF_NUMERIC_TOL:
        raise ValueError(
            f"packet/ref abs error exceeds allowed error for {row.get('image_name')}: "
            f"{abs_error} > {allowed_error}"
        )
    if ratio < 0.0 or ratio > 1.0 + PACKET_REF_NUMERIC_TOL:
        raise ValueError(f"packet/ref consistency ratio outside [0,1] for {row.get('image_name')}: {ratio}")
    expected_ratio = 0.0 if allowed_error == 0.0 else abs_error / allowed_error
    if abs(expected_ratio - ratio) > max(PACKET_REF_NUMERIC_TOL, PACKET_REF_NUMERIC_TOL * abs(expected_ratio)):
        raise ValueError(
            f"packet/ref consistency ratio mismatch for {row.get('image_name')}: "
            f"{ratio} vs {expected_ratio}"
        )


def compatibility_record(
    *,
    scene: str,
    image_name: str,
    target_camera: CameraRecord,
    packet_camera: PacketCamera,
    depth_row: dict[str, Any],
    model_camera_row: dict[str, Any],
    release_row: dict[str, str],
    packet_path_local: Path | None,
) -> dict[str, Any]:
    target_x = float(release_row["target_x"])
    target_y = float(release_row["target_y"])
    x_b, y_b = intrinsic_ray(target_camera, target_x, target_y)
    packet_u, packet_v = project_packet(packet_camera, x_b, y_b)
    x_p, y_p = intrinsic_ray(packet_camera, packet_u, packet_v)
    ray_b = unit_ray(x_b, y_b)
    ray_p = unit_ray(x_p, y_p)
    ray_error = math.hypot(x_p - x_b, y_p - y_b)
    angle = ray_angle(ray_b, ray_p)
    if ray_error > RAY_COORD_TOL or angle > RAY_ANGLE_TOL_RAD:
        raise ValueError(f"packet ray equivalence failed for {scene} {image_name}: coord={ray_error} angle={angle}")
    implicit_u = target_x * (packet_camera.width / target_camera.width)
    implicit_v = target_y * (packet_camera.height / target_camera.height)
    implicit_diff = math.hypot(packet_u - implicit_u, packet_v - implicit_v)
    in_bounds = 0.0 <= packet_u < packet_camera.width and 0.0 <= packet_v < packet_camera.height
    if not in_bounds:
        raise ValueError(f"packet-native coordinate out of bounds for {scene} {image_name}: {packet_u},{packet_v}")
    shape_status = "not_local"
    shape_height = int(depth_row["height"])
    shape_width = int(depth_row["width"])
    if packet_path_local is not None:
        header_height, header_width = npz_member_shape_header_only(packet_path_local)
        if (header_width, header_height) != (packet_camera.width, packet_camera.height):
            raise ValueError(
                f"packet header shape mismatch for {scene} {image_name}: "
                f"header {header_width}x{header_height}, camera {packet_camera.width}x{packet_camera.height}"
            )
        shape_status = "header_verified_no_depth_values_read"
    packet_record = {
        "scene": scene,
        "image_name": image_name,
        "target_camera_record_sha256": release_row["target_camera_record_sha256"],
        "packet_camera_record_sha256": packet_camera_hash(packet_camera),
        "packet_camera_record": packet_camera_record(packet_camera),
        "packet_width": packet_camera.width,
        "packet_height": packet_camera.height,
        "packet_fx": fmt_float(packet_camera.fx),
        "packet_fy": fmt_float(packet_camera.fy),
        "packet_cx": fmt_float(packet_camera.cx),
        "packet_cy": fmt_float(packet_camera.cy),
        "source_width": packet_camera.source_width,
        "source_height": packet_camera.source_height,
        "source_fx": fmt_float(packet_camera.source_fx),
        "source_fy": fmt_float(packet_camera.source_fy),
        "fovx": fmt_float(packet_camera.fovx),
        "fovy": fmt_float(packet_camera.fovy),
        "resolution_label": f"R{packet_camera.resolution_value}",
        "resolution_value": packet_camera.resolution_value,
        "rounding_rule": packet_camera.rounding_rule,
        "principal_point_rule": packet_camera.principal_point_rule,
        "resize_rule": packet_camera.resize_rule,
        "crop_pad_policy": packet_camera.crop_pad_policy,
        "pixel_convention_original": packet_camera.pixel_convention_original,
        "pixel_convention_canonical": packet_camera.pixel_convention_canonical,
        "benchmark_target_x": fmt_float(target_x),
        "benchmark_target_y": fmt_float(target_y),
        "packet_x": fmt_float(packet_u),
        "packet_y": fmt_float(packet_v),
        "packet_normalized_x": fmt_float(x_p),
        "packet_normalized_y": fmt_float(y_p),
        "canonical_normalized_x": fmt_float(x_b),
        "canonical_normalized_y": fmt_float(y_b),
        "ray_coordinate_error": fmt_float(ray_error),
        "ray_angle_error_rad": fmt_float(angle),
        "packet_in_bounds": bool(in_bounds),
        "sx": fmt_float(packet_camera.width / target_camera.width),
        "sy": fmt_float(packet_camera.height / target_camera.height),
        "implicit_scale_packet_x": fmt_float(implicit_u),
        "implicit_scale_packet_y": fmt_float(implicit_v),
        "camera_projection_vs_implicit_mapping_diff_px": fmt_float(implicit_diff),
        "packet_path_original": depth_row.get("packet_path", depth_row.get("depth_path", "")),
        "packet_path_local": str(packet_path_local) if packet_path_local is not None else "",
        "packet_sha256": str(depth_row.get("packet_sha256") or depth_row.get("sha256") or depth_row.get("file_sha256") or ""),
        "packet_shape_manifest_width": shape_width,
        "packet_shape_manifest_height": shape_height,
        "packet_shape_header_status": shape_status,
        "model_camera_width": int(model_camera_row["width"]),
        "model_camera_height": int(model_camera_row["height"]),
        "model_camera_fx": fmt_float(float(model_camera_row["fx"])),
        "model_camera_fy": fmt_float(float(model_camera_row["fy"])),
    }
    packet_record["source_target_packet_mapping_record_sha256"] = canonical_record_sha256(packet_record)
    return packet_record


def summarize_coordinate_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    diffs = np.asarray([float(r["camera_projection_vs_implicit_mapping_diff_px"]) for r in rows], dtype=np.float64)
    ray_diffs = np.asarray([float(r["ray_coordinate_error"]) for r in rows], dtype=np.float64)
    if diffs.size == 0:
        raise ValueError("no coordinate rows to summarize")
    return {
        "observation_count": int(len(rows)),
        "packet_bounds_pass_count": int(sum(1 for r in rows if bool(r["packet_in_bounds"]))),
        "median_projection_vs_implicit_diff_px": float(np.median(diffs)),
        "p95_projection_vs_implicit_diff_px": float(np.percentile(diffs, 95)),
        "max_projection_vs_implicit_diff_px": float(np.max(diffs)),
        "median_ray_coordinate_error": float(np.median(ray_diffs)),
        "p95_ray_coordinate_error": float(np.percentile(ray_diffs, 95)),
        "max_ray_coordinate_error": float(np.max(ray_diffs)),
        "implicit_mapping_gate_px": IMPLICIT_MAPPING_GATE_PX,
        "implicit_mapping_gate_passed": bool(float(np.max(diffs)) <= IMPLICIT_MAPPING_GATE_PX),
    }


def load_release_root_digest(
    release_dir: Path,
    release_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if release_config is None:
        manifest_path = release_dir / "v1_2_2_release_file_manifest.json"
        root_path = release_dir / "v1_2_2_release_root_digest.json"
    else:
        manifest_path = resolve_release_payload_path(
            release_dir,
            str(release_config.get("payload_manifest", "")),
            "payload manifest",
        )
        root_path = resolve_release_payload_path(
            release_dir,
            str(release_config.get("root_digest_record", "")),
            "root digest record",
        )
    root = load_json(root_path)
    integrity = verify_payload_integrity(release_dir, manifest_path, root_path)
    if not integrity["passed"]:
        raise ValueError(f"release integrity failed: {integrity}")
    root["_root_record_sha256"] = file_sha256(root_path)
    root["_payload_manifest_sha256"] = file_sha256(manifest_path)
    return root


def load_target_pose_records(
    release_dir: Path,
    scene: str,
    release_config: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    if release_config is None:
        path = release_dir / "camera_provenance_manifest_v1_2_2.json"
    else:
        path = resolve_release_payload_path(
            release_dir,
            str(release_config.get("camera_provenance_manifest", "")),
            "camera provenance manifest",
        )
    manifest = load_json(path)
    try:
        rows = manifest["scenes"][scene]["target_model"]["images"]
    except KeyError as exc:
        raise ValueError(f"camera provenance manifest missing target images for {scene}") from exc
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = Path(str(row.get("image_name", ""))).name
        if not name:
            continue
        if name in out:
            raise ValueError(f"duplicate target pose record in camera provenance: {scene} {name}")
        out[name] = dict(row)
    if not out:
        raise ValueError(f"no target pose records loaded for {scene}")
    return out


def build_provenance_record(
    *,
    renderer_repo: Path,
    depth_manifest: dict[str, Any],
) -> dict[str, Any]:
    evaluator_source = REPO_ROOT / "code" / "gcp" / "evaluate_gaussian_gcp_geometry.py"
    validator_source = Path(__file__).resolve()
    record = {
        "schema": "ms_gcp_packet_camera_compatibility_provenance_v1",
        "generator_commit": git_commit(REPO_ROOT),
        "evaluator_runtime_commit": git_commit(REPO_ROOT),
        "evaluator_worktree_status_porcelain": git_status_porcelain(REPO_ROOT),
        "generator_source_sha256": file_sha256(validator_source),
        "compatibility_validator_source_sha256": file_sha256(validator_source),
        "evaluator_source_sha256": file_sha256(evaluator_source),
        "renderer_repo": str(renderer_repo),
        "renderer_commit": git_commit(renderer_repo),
        "renderer_worktree_status_porcelain": git_status_porcelain(renderer_repo),
        "renderer_commit_declared_by_depth_manifest": depth_manifest.get("renderer_commit", ""),
        "renderer_camera_loader_source_sha256": file_sha256(renderer_repo / "utils" / "camera_utils.py") if (renderer_repo / "utils" / "camera_utils.py").exists() else "",
        "renderer_projection_source_sha256": file_sha256(renderer_repo / "utils" / "graphics_utils.py") if (renderer_repo / "utils" / "graphics_utils.py").exists() else "",
        "renderer_scene_camera_source_sha256": file_sha256(renderer_repo / "scene" / "cameras.py") if (renderer_repo / "scene" / "cameras.py").exists() else "",
        "rasterizer_tree_hash": depth_manifest.get("rasterizer_tree_hash", ""),
        "exporter_commit": depth_manifest.get("exporter_commit", ""),
    }
    required_nonempty = [
        "generator_commit",
        "evaluator_runtime_commit",
        "generator_source_sha256",
        "compatibility_validator_source_sha256",
        "evaluator_source_sha256",
        "renderer_commit",
        "renderer_camera_loader_source_sha256",
        "renderer_projection_source_sha256",
        "renderer_scene_camera_source_sha256",
        "rasterizer_tree_hash",
        "exporter_commit",
    ]
    missing = [name for name in required_nonempty if not str(record.get(name, "")).strip()]
    if missing:
        raise ValueError(f"provenance record required fields missing: {missing}")
    if record["renderer_commit"] != record["renderer_commit_declared_by_depth_manifest"]:
        raise ValueError("renderer commit mismatch between runtime renderer repo and depth manifest")
    if record["evaluator_worktree_status_porcelain"] != "":
        raise ValueError("evaluator/generator worktree must be clean to build wrapper")
    if record["renderer_worktree_status_porcelain"] != "":
        raise ValueError("renderer worktree must be clean to build wrapper")
    return record


def build_wrapper(
    *,
    release_dir: Path,
    scene: str,
    depth_manifest_path: Path,
    model_dir: Path,
    renderer_repo: Path,
    out_dir: Path,
    packet_search_roots: Sequence[Path],
    require_local_packets: bool = False,
    release_config_path: Path | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if release_config_path is None:
        candidates = sorted(release_dir.glob("gcp_benchmark_release_v*.json"))
        if len(candidates) != 1:
            raise ValueError(f"cannot uniquely resolve release config in {release_dir}: {candidates}")
        release_config_path = candidates[0]
    release_config = load_json(release_config_path)
    release_root = load_release_root_digest(release_dir, release_config)
    release_rows = load_release_rows(release_dir, scene, release_config)
    target_pose_records = load_target_pose_records(release_dir, scene, release_config)
    by_image: dict[str, list[dict[str, str]]] = {}
    for row in release_rows:
        by_image.setdefault(Path(row["target_image_name"]).name, []).append(row)
    depth_manifest = load_json(depth_manifest_path)
    validate_depth_manifest_contract(depth_manifest)
    provenance_record = build_provenance_record(renderer_repo=renderer_repo, depth_manifest=depth_manifest)
    provenance_record_sha256 = canonical_record_sha256(provenance_record)
    depth_index = depth_index_by_name(depth_manifest)
    cfg_path = model_dir / "cfg_args"
    cameras_json_path = model_dir / "cameras.json"
    cfg_args = parse_cfg_args(cfg_path)
    model_cameras = load_model_cameras(cameras_json_path)
    missing = sorted(set(by_image) - set(depth_index))
    if missing:
        raise ValueError(f"depth manifest is missing annotated images for {scene}: {missing[:8]}")
    camera_records: dict[str, dict[str, Any]] = {}
    view_rows: list[dict[str, Any]] = []
    coordinate_rows: list[dict[str, Any]] = []
    local_packet_failures = []
    for image_name in sorted(by_image):
        depth_row = depth_index[image_name]
        release_row = by_image[image_name][0]
        target_camera = camera_from_release_row(release_row)
        model_row = model_camera_for_image(model_cameras, image_name)
        target_pose = target_pose_records.get(image_name)
        if target_pose is None:
            raise ValueError(f"camera provenance target pose missing {scene} {image_name}")
        pose_check = pose_from_cameras_json(model_row, target_pose)
        if not pose_check["passed"]:
            raise ValueError(f"packet/benchmark pose equivalence failed for {scene} {image_name}: {pose_check}")
        packet_width = int(depth_row["width"])
        packet_height = int(depth_row["height"])
        validate_depth_index_row_contract(depth_row, packet_width, packet_height)
        packet_camera = recover_packet_camera(
            scene=scene,
            image_name=image_name,
            target_camera=target_camera,
            model_camera_row=model_row,
            packet_width=packet_width,
            packet_height=packet_height,
            cfg_args=cfg_args,
            manifest_pixel_convention=str(depth_manifest.get("pixel_coordinate_convention", "")),
        )
        camera_records[image_name] = {
            "image_name": image_name,
            "packet_camera_record_sha256": packet_camera_hash(packet_camera),
            "packet_camera_record": packet_camera_record(packet_camera),
            "packet_pose_record_sha256": pose_check["pose_record_sha256"],
            "packet_pose_record": pose_check["pose_record"],
            "target_pose_record_sha256": pose_check["target_pose_record_sha256"],
            "pose_conversion_full": pose_check,
            "pose_equivalence": {
                "passed": pose_check["passed"],
                "center_difference_model_units": pose_check["center_difference_model_units"],
                "center_tolerance_model_units": POSE_CENTER_TOL_MODEL_UNITS,
                "rotation_angular_difference_rad": pose_check["rotation_angular_difference_rad"],
                "rotation_tolerance_rad": POSE_ROTATION_TOL_RAD,
                "pose_convention": POSE_CONVENTION_VERSION,
            },
            "projection_matrix_equivalence": projection_matrix_equivalence(packet_camera),
        }
        packet_path = resolve_packet_path(str(depth_row.get("packet_path", depth_row.get("depth_path", ""))), packet_search_roots)
        if packet_path is None:
            if require_local_packets:
                local_packet_failures.append(image_name)
        else:
            expected_sha = str(depth_row.get("packet_sha256") or depth_row.get("sha256") or depth_row.get("file_sha256") or "")
            if expected_sha and file_sha256(packet_path) != expected_sha:
                raise ValueError(f"local packet SHA mismatch: {packet_path}")
            validate_npz_packet_headers(
                path=packet_path,
                expected_width=packet_width,
                expected_height=packet_height,
                expected_dtype=str(depth_row.get("dtype", "float32")),
            )
        view_record = compatibility_record(
            scene=scene,
            image_name=image_name,
            target_camera=target_camera,
            packet_camera=packet_camera,
            depth_row=depth_row,
            model_camera_row=model_row,
            release_row=release_row,
            packet_path_local=packet_path,
        )
        view_record.update(
            {
                "packet_pose_record_sha256": pose_check["pose_record_sha256"],
                "target_pose_record_sha256": pose_check["target_pose_record_sha256"],
                "pose_center_difference_model_units": fmt_float(pose_check["center_difference_model_units"]),
                "pose_rotation_angular_difference_rad": fmt_float(pose_check["rotation_angular_difference_rad"]),
                "pose_convention": POSE_CONVENTION_VERSION,
                "pose_equivalence_passed": bool(pose_check["passed"]),
                "provenance_record_sha256": provenance_record_sha256,
            }
        )
        hash_payload = dict(view_record)
        hash_payload.pop("source_target_packet_mapping_record_sha256", None)
        view_record["source_target_packet_mapping_record_sha256"] = canonical_record_sha256(hash_payload)
        view_rows.append(view_record)
        for row in by_image[image_name]:
            coord = compatibility_record(
                scene=scene,
                image_name=image_name,
                target_camera=target_camera,
                packet_camera=packet_camera,
                depth_row=depth_row,
                model_camera_row=model_row,
                release_row=row,
                packet_path_local=packet_path,
            )
            coord["observation_id"] = row["observation_id"]
            coord["point_name"] = row["point_name"]
            coord["packet_pose_record_sha256"] = pose_check["pose_record_sha256"]
            coord["target_pose_record_sha256"] = pose_check["target_pose_record_sha256"]
            coord["pose_center_difference_model_units"] = fmt_float(pose_check["center_difference_model_units"])
            coord["pose_rotation_angular_difference_rad"] = fmt_float(pose_check["rotation_angular_difference_rad"])
            coord["provenance_record_sha256"] = provenance_record_sha256
            coordinate_rows.append(coord)
    if local_packet_failures:
        raise ValueError(f"local packet files missing: {local_packet_failures[:8]}")
    coordinate_summary = summarize_coordinate_rows(coordinate_rows)
    if not coordinate_summary["implicit_mapping_gate_passed"]:
        raise ValueError(f"implicit mapping gate failed: {coordinate_summary}")
    records_root = canonical_records_root_sha256(
        view_rows,
        ["scene", "image_name", "packet_camera_record_sha256", "packet_sha256"],
    )
    wrapper = {
        "schema": COMPAT_SCHEMA,
        "supersedes_schema": WITHDRAWN_COMPAT_SCHEMA_V1,
        "withdrawn_previous_wrapper_disposition": WITHDRAWN_COMPAT_V1_DISPOSITION,
        "created_by": "code/gcp/gcp_packet_camera_compatibility.py",
        "evaluator_repo_commit": git_commit(REPO_ROOT),
        "evaluator_worktree_status_porcelain": git_status_porcelain(REPO_ROOT),
        "exact_generation_command": " ".join(sys.argv),
        "generator_source_sha256": file_sha256(Path(__file__).resolve()),
        "evaluator_source_sha256": provenance_record["evaluator_source_sha256"],
        "compatibility_validator_source_sha256": provenance_record["compatibility_validator_source_sha256"],
        "generator_commit": provenance_record["generator_commit"],
        "evaluator_runtime_commit": provenance_record["evaluator_runtime_commit"],
        "release_id": str(release_config.get("release_id", "")),
        "release_schema": str(release_config.get("schema", "")),
        "release_config_path": str(release_config_path),
        "release_config_sha256": file_sha256(release_config_path),
        "release_root_digest_sha256": release_root.get("payload_root_digest_sha256", ""),
        "release_root_record_sha256": release_root.get("_root_record_sha256", ""),
        "projection_tolerance_px": PROJECTION_TOL_PX,
        "ray_coordinate_tolerance": RAY_COORD_TOL,
        "ray_angle_tolerance_rad": RAY_ANGLE_TOL_RAD,
        "accepted_pixel_convention_aliases": ACCEPTED_PIXEL_CONVENTION_ALIASES,
        "metric_packet_contract": metric_packet_contract_from_manifest(depth_manifest),
        "pose_conversion_protocol": {
            "version": POSE_CONVENTION_VERSION,
            "cameras_json_rotation": "camera_to_world_row_major_3x3",
            "cameras_json_position": "camera_center_model_world_coordinates",
            "conversion": "R_w2c=R_c2w^T; t=-R_w2c*C",
            "colmap_convention": "world_to_camera_qvec_qw_qx_qy_qz_and_tvec",
            "axis_flip_policy": "none",
            "center_difference_tolerance_model_units": POSE_CENTER_TOL_MODEL_UNITS,
            "rotation_angular_difference_tolerance_rad": POSE_ROTATION_TOL_RAD,
            "pose_hash_serialization": "UTF-8 no BOM, sorted keys, compact separators, .17g float strings",
        },
        "depth_tensor_values_read": False,
        "patch_sampling_performed": False,
        "residual_generation_performed": False,
        "sim3_performed": False,
        "formal_metric_computation_performed": False,
        "packet_sets": [
            {
                "scene": scene,
                "status": "PASS",
                "original_depth_manifest_path": str(depth_manifest_path),
                "original_depth_manifest_sha256": file_sha256(depth_manifest_path),
                "model_dir": str(model_dir),
                "model_cameras_json_sha256": file_sha256(cameras_json_path),
                "model_cfg_args_sha256": file_sha256(cfg_path),
                "provenance_record": provenance_record,
                "provenance_record_sha256": provenance_record_sha256,
                "renderer_repo": str(renderer_repo),
                "renderer_commit": provenance_record["renderer_commit"],
                "renderer_worktree_status_porcelain": provenance_record["renderer_worktree_status_porcelain"],
                "renderer_commit_declared_by_depth_manifest": provenance_record["renderer_commit_declared_by_depth_manifest"],
                "rasterizer_tree_hash": provenance_record["rasterizer_tree_hash"],
                "exporter_commit": provenance_record["exporter_commit"],
                "renderer_camera_loader_source": "utils/camera_utils.py::loadCam and scene/cameras.py::Camera",
                "renderer_projection_source": "utils/graphics_utils.py::getProjectionMatrix symmetric frustum",
                "renderer_camera_loader_source_sha256": provenance_record["renderer_camera_loader_source_sha256"],
                "renderer_projection_source_sha256": provenance_record["renderer_projection_source_sha256"],
                "renderer_scene_camera_source_sha256": provenance_record["renderer_scene_camera_source_sha256"],
                "resolution_label": f"R{cfg_args.get('resolution')}",
                "patch_protocol": PATCH_PROTOCOL,
                "packet_patch_size": 7,
                "packet_patch_radius": 3,
                "view_count": len(view_rows),
                "observation_count": len(coordinate_rows),
                "packet_camera_records": list(camera_records.values()),
                "view_mappings": view_rows,
                "coordinate_summary": coordinate_summary,
                "compatibility_records_root_sha256": records_root,
            }
        ],
    }
    wrapper_path = out_dir / "metric_depth_packet_resolution_compatibility_v1_1.json"
    coordinate_csv = out_dir / f"{scene}_coordinate_only_validation.csv"
    summary_path = out_dir / f"{scene}_coordinate_only_summary.json"
    golden_pose = camera_records.get("DJI_20260602165038_0001_D.JPG") or next(iter(camera_records.values()))
    golden_pose_path = out_dir / f"{scene}_golden_pose_conversion.json"
    write_json(wrapper_path, wrapper)
    sha_path = write_detached_sha256(wrapper_path)
    write_csv(coordinate_csv, coordinate_rows, sorted({k for row in coordinate_rows for k in row.keys()}))
    write_json(summary_path, coordinate_summary)
    write_json(golden_pose_path, golden_pose)
    return {
        "wrapper_path": str(wrapper_path),
        "wrapper_sha256": file_sha256(wrapper_path),
        "wrapper_sha256_path": str(sha_path),
        "coordinate_csv": str(coordinate_csv),
        "coordinate_summary": coordinate_summary,
        "coordinate_summary_path": str(summary_path),
        "golden_pose_path": str(golden_pose_path),
    }


def validate_provenance_record(record: dict[str, Any]) -> None:
    required = [
        "generator_commit",
        "evaluator_runtime_commit",
        "generator_source_sha256",
        "compatibility_validator_source_sha256",
        "evaluator_source_sha256",
        "renderer_commit",
        "renderer_commit_declared_by_depth_manifest",
        "renderer_camera_loader_source_sha256",
        "renderer_projection_source_sha256",
        "renderer_scene_camera_source_sha256",
        "rasterizer_tree_hash",
        "exporter_commit",
    ]
    missing = [name for name in required if not str(record.get(name, "")).strip()]
    if missing:
        raise ValueError(f"provenance required fields missing: {missing}")
    if record.get("generator_commit") != git_commit(REPO_ROOT):
        raise ValueError("provenance generator commit mismatch")
    if record.get("evaluator_runtime_commit") != git_commit(REPO_ROOT):
        raise ValueError("provenance evaluator runtime commit mismatch")
    if record.get("evaluator_worktree_status_porcelain", "") != "":
        raise ValueError("provenance records dirty evaluator worktree")
    if record.get("renderer_worktree_status_porcelain", "") != "":
        raise ValueError("provenance records dirty renderer worktree")
    if record.get("renderer_commit") != record.get("renderer_commit_declared_by_depth_manifest"):
        raise ValueError("provenance renderer commit mismatch")
    if record.get("generator_source_sha256") != file_sha256(Path(__file__).resolve()):
        raise ValueError("provenance generator source SHA mismatch")
    if record.get("compatibility_validator_source_sha256") != file_sha256(Path(__file__).resolve()):
        raise ValueError("provenance validator source SHA mismatch")
    evaluator_source = REPO_ROOT / "code" / "gcp" / "evaluate_gaussian_gcp_geometry.py"
    if record.get("evaluator_source_sha256") != file_sha256(evaluator_source):
        raise ValueError("provenance evaluator source SHA mismatch")


def compare_view_to_depth_index(wrapper_row: dict[str, Any], depth_row: dict[str, Any]) -> None:
    name = Path(str(wrapper_row.get("image_name", ""))).name
    depth_name = Path(str(depth_row.get("image_name", ""))).name
    if name != depth_name:
        raise ValueError(f"depth manifest image-name mismatch: {name} vs {depth_name}")
    wrapper_packet = Path(str(wrapper_row.get("packet_path_original", ""))).name
    depth_packet = Path(str(depth_row.get("packet_path", depth_row.get("depth_path", "")))).name
    if wrapper_packet != depth_packet:
        raise ValueError(f"packet basename/path identity mismatch for {name}: {wrapper_packet} vs {depth_packet}")
    checks = {
        "packet_sha256": str(depth_row.get("packet_sha256", depth_row.get("sha256", depth_row.get("file_sha256", "")))),
        "packet_width": int(depth_row.get("width", -1)),
        "packet_height": int(depth_row.get("height", -1)),
        "dtype": str(depth_row.get("dtype", "")),
        "primary_depth_tensor": str(depth_row.get("primary_depth_tensor", "")),
        "primary_depth_semantics": str(depth_row.get("primary_depth_semantics", "")),
    }
    if str(wrapper_row.get("packet_sha256", "")) != checks["packet_sha256"]:
        raise ValueError(f"packet SHA mismatch between wrapper and depth manifest for {name}")
    if int(wrapper_row.get("packet_width", -1)) != checks["packet_width"] or int(wrapper_row.get("packet_height", -1)) != checks["packet_height"]:
        raise ValueError(f"packet width/height mismatch between wrapper and depth manifest for {name}")
    if checks["dtype"] != "float32":
        raise ValueError(f"depth index dtype mismatch for {name}: {checks['dtype']}")
    if checks["primary_depth_tensor"] != PRIMARY_DEPTH_TENSOR:
        raise ValueError(f"depth index primary tensor mismatch for {name}: {checks['primary_depth_tensor']}")
    if checks["primary_depth_semantics"] != PRIMARY_DEPTH_SEMANTICS:
        raise ValueError(f"depth index primary semantics mismatch for {name}: {checks['primary_depth_semantics']}")
    wrapper_tensors = set(str(wrapper_row.get("tensor_names", "")).split("|")) if wrapper_row.get("tensor_names") else None
    depth_tensors = set(str(depth_row.get("tensor_names", "")).split("|"))
    if wrapper_tensors is not None and wrapper_tensors != depth_tensors:
        raise ValueError(f"tensor names mismatch between wrapper and depth manifest for {name}")


def coordinate_check_for_release_row(release_row: dict[str, str], view_mapping: dict[str, Any]) -> dict[str, Any]:
    packet_camera = packet_camera_from_view(view_mapping)
    target_camera = camera_from_release_row(release_row)
    target_x = float(release_row["target_x"])
    target_y = float(release_row["target_y"])
    x_b, y_b = intrinsic_ray(target_camera, target_x, target_y)
    packet_u, packet_v = project_packet_from_camera_record(packet_camera, x_b, y_b)
    x_p, y_p = intrinsic_ray(packet_camera, packet_u, packet_v)
    ray_error = math.hypot(x_p - x_b, y_p - y_b)
    angle = ray_angle(unit_ray(x_p, y_p), unit_ray(x_b, y_b))
    if ray_error > RAY_COORD_TOL or angle > RAY_ANGLE_TOL_RAD:
        raise ValueError(f"runtime ray equivalence failed for {release_row.get('observation_id')}")
    in_bounds = 0.0 <= packet_u < packet_camera.width and 0.0 <= packet_v < packet_camera.height
    implicit_u = target_x * (packet_camera.width / target_camera.width)
    implicit_v = target_y * (packet_camera.height / target_camera.height)
    implicit_diff = math.hypot(packet_u - implicit_u, packet_v - implicit_v)
    return {
        "packet_in_bounds": bool(in_bounds),
        "packet_x": packet_u,
        "packet_y": packet_v,
        "ray_coordinate_error": ray_error,
        "ray_angle_error_rad": angle,
        "camera_projection_vs_implicit_mapping_diff_px": implicit_diff,
    }


def project_packet_from_camera_record(camera: CameraRecord, x_norm: float, y_norm: float) -> tuple[float, float]:
    if camera.model.upper() != "PINHOLE" or len(camera.params) != 4:
        raise ValueError(f"Expected PINHOLE packet camera, got {camera.model} {camera.params}")
    fx, fy, cx, cy = [float(x) for x in camera.params]
    return fx * float(x_norm) + cx, fy * float(y_norm) + cy


def validate_compatibility_wrapper(
    path: Path,
    *,
    expected_wrapper_sha256: str | None = None,
    depth_manifest: dict[str, Any] | None = None,
    depth_manifest_path: Path | None = None,
    release_config: dict[str, Any] | None = None,
    release_dir: Path | None = None,
    scene: str | None = None,
    patch_size: int | None = None,
    packet_search_roots: Sequence[Path] = (),
    require_local_packets: bool = True,
) -> dict[str, Any]:
    detached = verify_detached_sha256(path)
    actual_wrapper_sha = detached["sha256"]
    approved_wrapper_sha = validate_approved_sha256(expected_wrapper_sha256)
    if actual_wrapper_sha.lower() != approved_wrapper_sha:
        raise ValueError(f"approved wrapper SHA mismatch: {actual_wrapper_sha} vs {approved_wrapper_sha}")
    wrapper = load_json(path)
    if wrapper.get("schema") != COMPAT_SCHEMA:
        raise ValueError(f"Unsupported packet compatibility schema: {wrapper.get('schema')}")
    if wrapper.get("evaluator_repo_commit") != git_commit(REPO_ROOT):
        raise ValueError("wrapper evaluator commit does not match runtime evaluator commit")
    if git_status_porcelain(REPO_ROOT) != "":
        raise ValueError("runtime evaluator worktree is dirty")
    if wrapper.get("evaluator_worktree_status_porcelain", "") != "":
        raise ValueError("wrapper was generated from a dirty evaluator worktree")
    evaluator_source = REPO_ROOT / "code" / "gcp" / "evaluate_gaussian_gcp_geometry.py"
    validator_source = Path(__file__).resolve()
    if wrapper.get("evaluator_source_sha256") != file_sha256(evaluator_source):
        raise ValueError("evaluator source SHA mismatch")
    if wrapper.get("compatibility_validator_source_sha256") != file_sha256(validator_source):
        raise ValueError("compatibility validator source SHA mismatch")
    if wrapper.get("depth_tensor_values_read") is not False:
        raise ValueError("compatibility wrapper must declare depth_tensor_values_read=false")
    if release_config is not None and wrapper.get("release_id") != release_config.get("release_id"):
        raise ValueError(f"compatibility release mismatch: {wrapper.get('release_id')} vs {release_config.get('release_id')}")
    if release_dir is not None:
        release_root = load_release_root_digest(release_dir, release_config)
        if wrapper.get("release_root_digest_sha256") != release_root.get("payload_root_digest_sha256"):
            raise ValueError("wrapper release payload root digest mismatch")
        if wrapper.get("release_root_record_sha256") != release_root.get("_root_record_sha256"):
            raise ValueError("wrapper release root-record SHA mismatch")
    if patch_size is not None and int(patch_size) != 7:
        raise ValueError(f"packet compatibility requires patch_size=7, got {patch_size}")
    if depth_manifest is not None:
        contract = validate_depth_manifest_contract(depth_manifest)
        if wrapper.get("metric_packet_contract", {}) != contract:
            raise ValueError("wrapper metric packet contract does not match depth manifest")
        depth_index = depth_index_by_name(depth_manifest)
    else:
        depth_index = {}
    release_rows_by_image: dict[str, list[dict[str, str]]] = {}
    target_pose_records: dict[str, dict[str, Any]] = {}
    if release_dir is not None and scene is not None:
        for row in load_release_rows(release_dir, scene, release_config):
            release_rows_by_image.setdefault(Path(row["target_image_name"]).name, []).append(row)
        target_pose_records = load_target_pose_records(release_dir, scene, release_config)
    packet_sets = wrapper.get("packet_sets", [])
    seen_scenes: set[str] = set()
    validation: dict[str, Any] = {
        "schema": wrapper.get("schema"),
        "wrapper_sha256": file_sha256(path),
        "packet_sets": [],
        "depth_tensor_values_read": False,
    }
    for packet_set in packet_sets:
        set_scene = str(packet_set.get("scene", ""))
        if not set_scene:
            raise ValueError("packet set missing scene")
        if set_scene in seen_scenes:
            raise ValueError(f"duplicate packet set scene: {set_scene}")
        seen_scenes.add(set_scene)
        if scene is not None and set_scene != scene:
            continue
        if packet_set.get("status") != "PASS":
            raise ValueError(f"packet set status is not PASS for {set_scene}: {packet_set.get('status')}")
        if packet_set.get("patch_protocol") != PATCH_PROTOCOL:
            raise ValueError(f"patch protocol mismatch: {packet_set.get('patch_protocol')}")
        if int(packet_set.get("packet_patch_size", -1)) != 7 or int(packet_set.get("packet_patch_radius", -1)) != 3:
            raise ValueError("packet patch gate requires packet_patch_size=7 and radius=3")
        if depth_manifest_path is not None and packet_set.get("original_depth_manifest_sha256") != file_sha256(depth_manifest_path):
            raise ValueError("wrapper original depth manifest SHA mismatch")
        if packet_set.get("renderer_worktree_status_porcelain", "") != "":
            raise ValueError("renderer worktree was not clean when wrapper was generated")
        if packet_set.get("renderer_commit") != packet_set.get("renderer_commit_declared_by_depth_manifest"):
            raise ValueError("renderer commit mismatch between wrapper and depth manifest")
        provenance_record = packet_set.get("provenance_record")
        if not isinstance(provenance_record, dict):
            raise ValueError(f"packet set missing provenance record for {set_scene}")
        provenance_record_sha = canonical_record_sha256(provenance_record)
        if provenance_record_sha != packet_set.get("provenance_record_sha256"):
            raise ValueError(f"provenance record hash mismatch for {set_scene}")
        validate_provenance_record(provenance_record)
        view_rows = list(packet_set.get("view_mappings", []))
        if int(packet_set.get("view_count", -1)) != len(view_rows):
            raise ValueError(f"view_count mismatch for {set_scene}")
        by_view: dict[str, dict[str, Any]] = {}
        packet_camera_records = packet_set.get("packet_camera_records", [])
        if not isinstance(packet_camera_records, list):
            raise ValueError(f"packet_camera_records must be a list for {set_scene}")
        packet_camera_records_by_name: dict[str, dict[str, Any]] = {}
        for cam_record in packet_camera_records:
            rec_name = Path(str(cam_record.get("image_name", ""))).name
            if not rec_name:
                raise ValueError(f"packet camera record missing image_name in {set_scene}")
            if rec_name in packet_camera_records_by_name:
                raise ValueError(f"duplicate packet camera record in wrapper: {set_scene} {rec_name}")
            packet_camera_records_by_name[rec_name] = cam_record
        coordinate_counts = {
            "observation_count": 0,
            "packet_bounds_pass_count": 0,
            "max_projection_vs_implicit_diff_px": 0.0,
            "max_ray_coordinate_error": 0.0,
            "all_observations_uniquely_mapped": True,
        }
        packet_counts = {
            "expected_packet_count": len(view_rows),
            "resolved_packet_count": 0,
            "sha_verified_packet_count": 0,
            "header_validated_packet_count": 0,
            "missing_packet_count": 0,
            "ambiguous_packet_count": 0,
        }
        for row in view_rows:
            name = Path(str(row.get("image_name", ""))).name
            if not name:
                raise ValueError(f"view mapping missing image_name in {set_scene}")
            if name in by_view:
                raise ValueError(f"duplicate packet view mapping in wrapper: {set_scene} {name}")
            by_view[name] = row
            if row.get("provenance_record_sha256") != provenance_record_sha:
                raise ValueError(f"view provenance record hash mismatch for {set_scene} {name}")
            if row.get("crop_pad_policy") != "none_verified_by_renderer_camera_loader_source":
                raise ValueError(f"undeclared crop/pad policy for {set_scene} {name}: {row.get('crop_pad_policy')}")
            if not (0.0 <= float(row.get("packet_x", "nan")) < float(row.get("packet_width", -1))):
                raise ValueError(f"stored packet_x out of bounds for {set_scene} {name}")
            if not (0.0 <= float(row.get("packet_y", "nan")) < float(row.get("packet_height", -1))):
                raise ValueError(f"stored packet_y out of bounds for {set_scene} {name}")
            if row.get("pose_equivalence_passed") is not True:
                raise ValueError(f"pose equivalence did not pass for {set_scene} {name}")
            cam_record = packet_camera_records_by_name.get(name)
            if cam_record is None:
                raise ValueError(f"missing packet pose/camera record for {set_scene} {name}")
            packet_pose_record = cam_record.get("packet_pose_record")
            if pose_record_hash(packet_pose_record) != cam_record.get("packet_pose_record_sha256"):
                raise ValueError(f"packet pose record hash mismatch for {set_scene} {name}")
            if cam_record.get("packet_pose_record_sha256") != row.get("packet_pose_record_sha256"):
                raise ValueError(f"view packet pose reference mismatch for {set_scene} {name}")
            if cam_record.get("packet_camera_record_sha256") != row.get("packet_camera_record_sha256"):
                raise ValueError(f"view packet camera reference mismatch for {set_scene} {name}")
            if target_pose_records:
                target_pose = target_pose_records.get(name)
                if target_pose is None:
                    raise ValueError(f"release target pose missing for {set_scene} {name}")
                target_hash = canonical_record_sha256(
                    {
                        "image_id": target_pose["image_id"],
                        "image_name": target_pose["image_name"],
                        "camera_id": target_pose["camera_id"],
                        "qvec": target_pose["qvec"],
                        "tvec": target_pose["tvec"],
                    }
                )
                if target_hash != target_pose.get("record_sha256"):
                    raise ValueError(f"release target pose record hash mismatch for {set_scene} {name}")
                if target_hash != row.get("target_pose_record_sha256"):
                    raise ValueError(f"view target pose hash mismatch for {set_scene} {name}")
                center = np.asarray([float(x) for x in packet_pose_record["camera_center"]], dtype=np.float64)
                target_center = camera_center_from_colmap(target_pose["qvec"], target_pose["tvec"])
                center_diff = float(np.linalg.norm(center - target_center))
                angle = rotation_angle_rad(qvec_to_rotmat(packet_pose_record["qvec"]), qvec_to_rotmat(target_pose["qvec"]))
                if abs(center_diff - float(row.get("pose_center_difference_model_units", "nan"))) > max(1e-12, POSE_CENTER_TOL_MODEL_UNITS):
                    raise ValueError(f"stored pose center difference tamper/mismatch for {set_scene} {name}")
                if abs(angle - float(row.get("pose_rotation_angular_difference_rad", "nan"))) > max(1e-12, POSE_ROTATION_TOL_RAD):
                    raise ValueError(f"stored pose rotation difference tamper/mismatch for {set_scene} {name}")
                if center_diff > POSE_CENTER_TOL_MODEL_UNITS or angle > POSE_ROTATION_TOL_RAD:
                    raise ValueError(f"runtime pose equivalence failed for {set_scene} {name}")
            if float(row.get("pose_center_difference_model_units", "inf")) > POSE_CENTER_TOL_MODEL_UNITS:
                raise ValueError(f"pose center tolerance exceeded for {set_scene} {name}")
            if float(row.get("pose_rotation_angular_difference_rad", "inf")) > POSE_ROTATION_TOL_RAD:
                raise ValueError(f"pose rotation tolerance exceeded for {set_scene} {name}")
            packet_camera = packet_camera_from_view(row)
            if str(row.get("packet_camera_record_sha256", "")) != camera_record_hash(packet_camera):
                raise ValueError(f"packet camera hash mismatch for {set_scene} {name}")
            hash_payload = dict(row)
            hash_payload.pop("source_target_packet_mapping_record_sha256", None)
            recomputed = canonical_record_sha256(hash_payload)
            if str(row.get("source_target_packet_mapping_record_sha256", "")) != recomputed:
                raise ValueError(f"per-view mapping record hash mismatch for {set_scene} {name}")
            if depth_index:
                depth_row = depth_index.get(name)
                if depth_row is None:
                    raise ValueError(f"wrapper view missing from depth manifest: {set_scene} {name}")
                validate_depth_index_row_contract(depth_row, int(row["packet_width"]), int(row["packet_height"]))
                compare_view_to_depth_index(row, depth_row)
            packet_path = resolve_packet_path(str(row.get("packet_path_original", "")), packet_search_roots)
            if packet_path is None:
                packet_counts["missing_packet_count"] += 1
                if require_local_packets:
                    raise ValueError(f"local packet is required but missing for {set_scene} {name}")
            else:
                packet_counts["resolved_packet_count"] += 1
                if row.get("packet_sha256") and file_sha256(packet_path) != row.get("packet_sha256"):
                    raise ValueError(f"packet SHA mismatch for {set_scene} {name}")
                packet_counts["sha_verified_packet_count"] += 1
                validate_npz_packet_headers(
                    path=packet_path,
                    expected_width=int(row["packet_width"]),
                    expected_height=int(row["packet_height"]),
                    expected_dtype="float32",
                )
                packet_counts["header_validated_packet_count"] += 1
            if release_rows_by_image:
                obs_rows = release_rows_by_image.get(name, [])
                if not obs_rows:
                    raise ValueError(f"wrapper view missing release observations: {set_scene} {name}")
                for release_row in obs_rows:
                    coord = coordinate_check_for_release_row(release_row, row)
                    coordinate_counts["observation_count"] += 1
                    coordinate_counts["packet_bounds_pass_count"] += int(bool(coord["packet_in_bounds"]))
                    coordinate_counts["max_projection_vs_implicit_diff_px"] = max(
                        coordinate_counts["max_projection_vs_implicit_diff_px"],
                        float(coord["camera_projection_vs_implicit_mapping_diff_px"]),
                    )
                    coordinate_counts["max_ray_coordinate_error"] = max(
                        coordinate_counts["max_ray_coordinate_error"],
                        float(coord["ray_coordinate_error"]),
                    )
        if depth_index:
            depth_names = set(depth_index)
            view_names = set(by_view)
            if depth_names != view_names:
                raise ValueError(
                    f"wrapper/depth manifest image-list mismatch for {set_scene}: "
                    f"missing={sorted(depth_names - view_names)[:5]} extra={sorted(view_names - depth_names)[:5]}"
                )
        camera_record_names = set(packet_camera_records_by_name)
        view_names = set(by_view)
        if camera_record_names != view_names:
            raise ValueError(
                f"packet camera record set mismatch for {set_scene}: "
                f"missing={sorted(view_names - camera_record_names)[:5]} extra={sorted(camera_record_names - view_names)[:5]}"
            )
        if require_local_packets and (
            packet_counts["resolved_packet_count"] != packet_counts["expected_packet_count"]
            or packet_counts["sha_verified_packet_count"] != packet_counts["expected_packet_count"]
            or packet_counts["header_validated_packet_count"] != packet_counts["expected_packet_count"]
            or packet_counts["missing_packet_count"] != 0
            or packet_counts["ambiguous_packet_count"] != 0
        ):
            raise ValueError(f"local packet validation counts incomplete for {set_scene}: {packet_counts}")
        if release_rows_by_image:
            release_view_names = set(release_rows_by_image)
            if release_view_names != view_names:
                raise ValueError(
                    f"wrapper/release image-list mismatch for {set_scene}: "
                    f"missing={sorted(release_view_names - view_names)[:5]} extra={sorted(view_names - release_view_names)[:5]}"
                )
            if coordinate_counts["observation_count"] != int(packet_set.get("observation_count", -1)):
                raise ValueError(f"runtime observation count mismatch for {set_scene}")
            if coordinate_counts["packet_bounds_pass_count"] != coordinate_counts["observation_count"]:
                raise ValueError(f"packet coordinate bounds failure for {set_scene}")
            frozen_summary = packet_set.get("coordinate_summary", {})
            if coordinate_counts["max_projection_vs_implicit_diff_px"] > min(
                IMPLICIT_MAPPING_GATE_PX,
                float(frozen_summary.get("implicit_mapping_gate_px", IMPLICIT_MAPPING_GATE_PX)),
            ):
                raise ValueError(f"projection-vs-implicit gate failed for {set_scene}")
            if coordinate_counts["max_ray_coordinate_error"] > RAY_COORD_TOL:
                raise ValueError(f"ray coordinate gate failed for {set_scene}")
        records_root = canonical_records_root_sha256(
            view_rows,
            ["scene", "image_name", "packet_camera_record_sha256", "packet_sha256"],
        )
        if records_root != packet_set.get("compatibility_records_root_sha256"):
            raise ValueError(f"compatibility records root mismatch for {set_scene}")
        validation["packet_sets"].append(
            {
                "scene": set_scene,
                "view_count": len(view_rows),
                "records_root_sha256": records_root,
                "patch_protocol": packet_set.get("patch_protocol"),
                "packet_patch_size": packet_set.get("packet_patch_size"),
                "packet_patch_radius": packet_set.get("packet_patch_radius"),
                "packet_headers_checked_when_local": bool(packet_counts["header_validated_packet_count"] == packet_counts["expected_packet_count"]),
                **packet_counts,
                "runtime_coordinate_gate": coordinate_counts if release_rows_by_image else {},
            }
        )
    if scene is not None and not validation["packet_sets"]:
        raise ValueError(f"wrapper has no packet set for scene {scene}")
    return validation


def load_compatibility_wrapper(path: Path) -> dict[str, Any]:
    wrapper = load_json(path)
    if wrapper.get("schema") != COMPAT_SCHEMA:
        raise ValueError(f"Unsupported packet compatibility schema: {wrapper.get('schema')}")
    return wrapper


def packet_set_lookup(wrapper: dict[str, Any], scene: str) -> dict[str, Any]:
    matches = [packet_set for packet_set in wrapper.get("packet_sets", []) if packet_set.get("scene") == scene]
    if len(matches) != 1:
        raise ValueError(f"Expected one packet compatibility set for scene {scene}, found {len(matches)}")
    return matches[0]


def packet_view_lookup(packet_set: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in packet_set.get("view_mappings", []):
        name = Path(str(row.get("image_name", ""))).name
        if not name:
            continue
        if name in result:
            raise ValueError(f"duplicate packet compatibility view mapping: {name}")
        result[name] = row
    return result


def packet_camera_from_view(row: dict[str, Any]) -> CameraRecord:
    record = row.get("packet_camera_record") or {}
    params = record.get("params", [])
    return CameraRecord(
        camera_id=int(record.get("camera_id", 0)),
        model=str(record.get("model", "PINHOLE")),
        width=int(record.get("width", row.get("packet_width", 0))),
        height=int(record.get("height", row.get("packet_height", 0))),
        params=tuple(float(x) for x in params),
    )


def packet_projection_for_row(row: dict[str, str], view_mapping: dict[str, Any]) -> dict[str, Any]:
    packet_camera = packet_camera_from_view(view_mapping)
    expected_hash = str(view_mapping.get("packet_camera_record_sha256", ""))
    if expected_hash and camera_record_hash(packet_camera) != expected_hash:
        raise ValueError(f"packet camera record hash mismatch for {row.get('image_name')}")
    target_x = float(row["u_px"])
    target_y = float(row["v_px"])
    target_fx, target_fy, target_cx, target_cy = [float(x) for x in str(row["target_camera_params"]).split(";")]
    x_norm = (target_x - target_cx) / target_fx
    y_norm = (target_y - target_cy) / target_fy
    packet_fx, packet_fy, packet_cx, packet_cy = [float(x) for x in packet_camera.params]
    packet_x = packet_fx * x_norm + packet_cx
    packet_y = packet_fy * y_norm + packet_cy
    packet_x_norm = (packet_x - packet_cx) / packet_fx
    packet_y_norm = (packet_y - packet_cy) / packet_fy
    coord_error = math.hypot(packet_x_norm - x_norm, packet_y_norm - y_norm)
    angle_error = ray_angle(unit_ray(packet_x_norm, packet_y_norm), unit_ray(x_norm, y_norm))
    if coord_error > RAY_COORD_TOL or angle_error > RAY_ANGLE_TOL_RAD:
        raise ValueError(f"packet ray equivalence failed for {row.get('image_name')}: {coord_error} {angle_error}")
    if not (0.0 <= packet_x < packet_camera.width and 0.0 <= packet_y < packet_camera.height):
        raise ValueError(f"packet coordinate out of bounds for {row.get('image_name')}: {packet_x},{packet_y}")
    return {
        "packet_u_px": packet_x,
        "packet_v_px": packet_y,
        "packet_normalized_x": packet_x_norm,
        "packet_normalized_y": packet_y_norm,
        "packet_camera": packet_camera,
        "ray_coordinate_error": coord_error,
        "ray_angle_error_rad": angle_error,
        "depth_pixel_scale_x": float(view_mapping["sx"]),
        "depth_pixel_scale_y": float(view_mapping["sy"]),
        "packet_patch_protocol": PATCH_PROTOCOL,
    }


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="Build no-depth packet-camera compatibility wrapper.")
    parser.add_argument("--release_config", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--depth_manifest", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--renderer_repo", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--packet_search_root", action="append", default=[])
    parser.add_argument("--require_local_packets", action="store_true")
    args = parser.parse_args()
    release_config = Path(args.release_config)
    packet_roots = [Path(p) for p in args.packet_search_root]
    result = build_wrapper(
        release_config_path=release_config,
        release_dir=release_config.parent,
        scene=args.scene,
        depth_manifest_path=Path(args.depth_manifest),
        model_dir=Path(args.model_dir),
        renderer_repo=Path(args.renderer_repo),
        out_dir=Path(args.out_dir),
        packet_search_roots=packet_roots,
        require_local_packets=bool(args.require_local_packets),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    run_cli()
