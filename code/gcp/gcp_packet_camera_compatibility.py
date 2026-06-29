from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
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


COMPAT_SCHEMA = "ms_gcp_metric_depth_packet_resolution_compatibility_v1"
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
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return ""


def git_status_porcelain(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain=v1"], text=True).strip()
    except Exception as exc:  # noqa: BLE001
        return f"ERROR:{type(exc).__name__}:{exc}"


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


def npz_member_shape_header_only(path: Path, preferred_key: str = "alpha_normalized_expected_camera_z") -> tuple[int, int]:
    with zipfile.ZipFile(path, "r") as zf:
        members = sorted(zf.namelist())
        member = f"{preferred_key}.npy"
        if member not in members:
            member = members[0]
        with zf.open(member, "r") as f:
            version = np.lib.format.read_magic(f)
            if version == (1, 0):
                shape, _fortran, _dtype = np.lib.format.read_array_header_1_0(f)
            elif version == (2, 0):
                shape, _fortran, _dtype = np.lib.format.read_array_header_2_0(f)
            else:
                shape, _fortran, _dtype = np.lib.format._read_array_header(f, version)  # type: ignore[attr-defined]
    if len(shape) != 2:
        raise ValueError(f"Expected 2D packet tensor header in {path}, got shape={shape}")
    return int(shape[0]), int(shape[1])


def resolve_packet_path(packet_path: str, search_roots: Sequence[Path]) -> Path | None:
    candidate = Path(packet_path)
    if candidate.exists():
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


def load_release_rows(release_dir: Path, scene: str) -> list[dict[str, str]]:
    path = release_dir / f"{scene}_gcp_annotations_pixel_domain_v1_2_2.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return read_csv(path)


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


def load_release_root_digest(release_dir: Path) -> dict[str, Any]:
    root = load_json(release_dir / "v1_2_2_release_root_digest.json")
    manifest_path = release_dir / "v1_2_2_release_file_manifest.json"
    integrity = verify_payload_integrity(release_dir, manifest_path, release_dir / "v1_2_2_release_root_digest.json")
    if not integrity["passed"]:
        raise ValueError(f"release integrity failed: {integrity}")
    root["_root_record_sha256"] = file_sha256(release_dir / "v1_2_2_release_root_digest.json")
    return root


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
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    release_root = load_release_root_digest(release_dir)
    release_rows = load_release_rows(release_dir, scene)
    by_image: dict[str, list[dict[str, str]]] = {}
    for row in release_rows:
        by_image.setdefault(Path(row["target_image_name"]).name, []).append(row)
    depth_manifest = load_json(depth_manifest_path)
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
        model_row = model_cameras.get(image_name)
        if model_row is None:
            raise ValueError(f"model cameras.json missing image {image_name}")
        packet_width = int(depth_row["width"])
        packet_height = int(depth_row["height"])
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
        "created_by": "code/gcp/gcp_packet_camera_compatibility.py",
        "evaluator_repo_commit": git_commit(REPO_ROOT),
        "evaluator_worktree_status_porcelain": git_status_porcelain(REPO_ROOT),
        "release_id": "gcp_benchmark_release_v1_2_2_pixel_domain_20260628",
        "release_root_digest_sha256": release_root.get("payload_root_digest_sha256", ""),
        "release_root_record_sha256": release_root.get("_root_record_sha256", ""),
        "projection_tolerance_px": PROJECTION_TOL_PX,
        "ray_coordinate_tolerance": RAY_COORD_TOL,
        "ray_angle_tolerance_rad": RAY_ANGLE_TOL_RAD,
        "accepted_pixel_convention_aliases": ACCEPTED_PIXEL_CONVENTION_ALIASES,
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
                "renderer_repo": str(renderer_repo),
                "renderer_commit": depth_manifest.get("renderer_commit", ""),
                "renderer_camera_loader_source": "utils/camera_utils.py::loadCam and scene/cameras.py::Camera",
                "renderer_projection_source": "utils/graphics_utils.py::getProjectionMatrix symmetric frustum",
                "renderer_camera_loader_source_sha256": file_sha256(renderer_repo / "utils" / "camera_utils.py") if (renderer_repo / "utils" / "camera_utils.py").exists() else "",
                "renderer_projection_source_sha256": file_sha256(renderer_repo / "utils" / "graphics_utils.py") if (renderer_repo / "utils" / "graphics_utils.py").exists() else "",
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
    wrapper_path = out_dir / "metric_depth_packet_resolution_compatibility_v1.json"
    coordinate_csv = out_dir / f"{scene}_coordinate_only_validation.csv"
    summary_path = out_dir / f"{scene}_coordinate_only_summary.json"
    write_json(wrapper_path, wrapper)
    write_csv(coordinate_csv, coordinate_rows, sorted({k for row in coordinate_rows for k in row.keys()}))
    write_json(summary_path, coordinate_summary)
    return {
        "wrapper_path": str(wrapper_path),
        "wrapper_sha256": file_sha256(wrapper_path),
        "coordinate_csv": str(coordinate_csv),
        "coordinate_summary": coordinate_summary,
        "coordinate_summary_path": str(summary_path),
    }


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
