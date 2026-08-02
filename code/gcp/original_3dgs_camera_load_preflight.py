#!/usr/bin/env python3
"""Load official Graphdeco cameras only and report memory/ray/FD evidence."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import signal
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jpeg_fds() -> list[str]:
    rows = []
    root = Path("/proc/self/fd")
    for path in root.iterdir():
        try:
            target = os.readlink(path)
        except OSError:
            continue
        if target.lower().endswith((".jpg", ".jpeg")):
            rows.append(target)
    return sorted(rows)


def _storage_bytes(tensor: Any) -> int:
    return int(tensor.untyped_storage().nbytes())


def _tensor_sha256(tensor: Any) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _process_rss_kib() -> int:
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    return 0


def _cgroup_memory_current() -> int | None:
    path = Path("/sys/fs/cgroup/memory.current")
    try:
        return int(path.read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method_root", type=Path, required=True)
    parser.add_argument("--source_root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=4)
    parser.add_argument("--data_device", default="cuda")
    parser.add_argument("--stabilization_seconds", type=int, default=30)
    parser.add_argument("--lifecycle_report", type=Path, required=True)
    parser.add_argument("--expected_materialization", choices=("path_backed", "eager"), default="path_backed")
    parser.add_argument("--include_tensor_hashes", action="store_true")
    parser.add_argument(
        "--host_allocator_policy",
        choices=("glibc_malloc_trim_threshold_zero_v1",),
        required=True,
    )
    args = parser.parse_args()
    method_root = args.method_root.resolve()
    source_root = args.source_root.resolve()
    if (source_root / "sparse" / "0" / "points3D.bin").exists():
        raise ValueError("camera subset must not expose full points3D tracks")
    sys.path.insert(0, str(method_root))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "colmap" / "utils"))
    import torch
    from PIL import Image
    from scene.dataset_readers import readColmapSceneInfo
    from utils.camera_utils import cameraList_from_camInfos
    from read_write_model import read_cameras_binary, read_images_binary
    if args.resolution != 4 or args.data_device not in {"cuda", "cpu"}:
        raise ValueError("formal resource preflight requires resolution=4 and data_device=cuda|cpu")
    if os.environ.get("MALLOC_TRIM_THRESHOLD_") != "0":
        raise ValueError("glibc_malloc_trim_threshold_zero_v1 requires MALLOC_TRIM_THRESHOLD_=0")
    lifecycle_path = args.lifecycle_report.resolve()
    lifecycle_path.parent.mkdir(parents=True, exist_ok=True)
    lifecycle_handle = lifecycle_path.open("x", encoding="utf-8", buffering=1, newline="\n")
    progress: dict[str, Any] = {
        "schema": "gs_gcp_original_3dgs_camera_load_partial_v2",
        "status": "RUNNING",
        "camera_records_read_count": 0,
        "camera_tensors_materialized_count": 0,
        "currently_open_source_image_count": 0,
        "data_device": args.data_device,
        "host_allocator_policy": args.host_allocator_policy,
        "malloc_trim_threshold_env": os.environ.get("MALLOC_TRIM_THRESHOLD_"),
    }

    def persist_partial(status: str, reason: str | None = None) -> None:
        payload = {**progress, "status": status, "failure_reason": reason}
        _write_json(args.report, payload)

    def handle_term(signum: int, _frame: Any) -> None:
        persist_partial("CONTROLLED_TERMINATION", f"signal_{signum}")
        lifecycle_handle.flush()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, handle_term)
    fd_before = len(list(Path("/proc/self/fd").iterdir()))
    allocated_before = int(torch.cuda.memory_allocated())
    reserved_before = int(torch.cuda.memory_reserved())
    scene_info = readColmapSceneInfo(str(source_root), "images", False)
    progress["camera_records_read_count"] = len(scene_info.train_cameras)
    if args.expected_materialization == "path_backed" and any(camera.image is not None for camera in scene_info.train_cameras):
        persist_partial("CAMERA_LOAD_FAILURE", "COLMAP CameraInfo retains live image backing")
        raise ValueError("path-backed compatibility source did not produce image=None CameraInfo records")
    if args.expected_materialization == "eager" and any(camera.image is None for camera in scene_info.train_cameras):
        persist_partial("CAMERA_LOAD_FAILURE", "eager reference CameraInfo is not image-backed")
        raise ValueError("eager reference did not produce live image-backed CameraInfo records")
    model_args = SimpleNamespace(resolution=4, data_device="cuda")
    model_args.data_device = args.data_device
    total_cameras = len(scene_info.train_cameras)

    def lifecycle_observer(event: str, index: int, cam_info: Any) -> None:
        if event == "source_image_opened":
            progress["currently_open_source_image_count"] += 1
        elif event == "source_image_closed":
            progress["currently_open_source_image_count"] -= 1
            progress["camera_tensors_materialized_count"] = index + 1
        if event != "source_image_closed" or not (index == 0 or (index + 1) % 32 == 0 or index + 1 == total_cameras):
            return
        row = {
            "event": "camera_materialized",
            "camera_index": index,
            "camera_image_name": cam_info.image_name,
            "camera_records_read_count": progress["camera_records_read_count"],
            "camera_tensors_materialized_count": progress["camera_tensors_materialized_count"],
            "currently_open_source_image_count": progress["currently_open_source_image_count"],
            "jpeg_fd_count": len(_jpeg_fds()),
            "live_pil_image_object_count": sum(1 for obj in gc.get_objects() if isinstance(obj, Image.Image)),
            "process_rss_kib": _process_rss_kib(),
            "cgroup_memory_current_bytes": _cgroup_memory_current(),
            "torch_cuda_allocated_bytes": int(torch.cuda.memory_allocated()),
            "torch_cuda_reserved_bytes": int(torch.cuda.memory_reserved()),
        }
        lifecycle_handle.write(json.dumps(row, sort_keys=True) + "\n")
        lifecycle_handle.flush()

    try:
        if args.expected_materialization == "path_backed":
            cameras = cameraList_from_camInfos(scene_info.train_cameras, 1.0, model_args, lifecycle_observer)
        else:
            cameras = cameraList_from_camInfos(scene_info.train_cameras, 1.0, model_args)
            progress["camera_tensors_materialized_count"] = len(cameras)
    except SystemExit:
        raise
    except Exception as exc:
        persist_partial("CAMERA_LOAD_FAILURE", f"{type(exc).__name__}: {exc}")
        lifecycle_handle.close()
        raise
    del scene_info
    gc.collect()
    torch.cuda.synchronize()
    sparse = source_root / "sparse" / "0"
    source_cameras = read_cameras_binary(sparse / "cameras.bin")
    source_images = read_images_binary(sparse / "images.bin")
    source_by_stem = {Path(image.name).stem: image for image in source_images.values()}
    theoretical_bytes = 0
    actual_tensor_bytes = 0
    storages = set()
    records = []
    max_ray_error = 0.0
    for camera in cameras:
        tensor = camera.original_image
        width = int(tensor.shape[-1])
        height = int(tensor.shape[-2])
        channels = int(tensor.shape[-3])
        theoretical = width * height * channels * int(tensor.element_size())
        theoretical_bytes += theoretical
        storage_key = (int(tensor.untyped_storage().data_ptr()), int(tensor.untyped_storage().nbytes()))
        if storage_key not in storages:
            storages.add(storage_key)
            actual_tensor_bytes += _storage_bytes(tensor)
        image = source_by_stem.get(camera.image_name)
        if image is None:
            raise ValueError(f"loaded camera has no COLMAP image: {camera.image_name}")
        source = source_cameras[int(image.camera_id)]
        expected_dims = (round(int(source.width) / 4), round(int(source.height) / 4))
        if (width, height) != expected_dims:
            raise ValueError(f"loaded dimension mismatch for {camera.image_name}: {(width,height)} != {expected_dims}")
        source_fx, source_fy = float(source.params[0]), float(source.params[1])
        fovx = 2.0 * math.atan(float(source.width) / (2.0 * source_fx))
        fovy = 2.0 * math.atan(float(source.height) / (2.0 * source_fy))
        loaded_fx = width / (2.0 * math.tan(fovx / 2.0))
        loaded_fy = height / (2.0 * math.tan(fovy / 2.0))
        for x, y in ((0.0, 0.0), (0.2, -0.15), (-0.25, 0.1)):
            u = loaded_fx * x + width / 2.0
            v = loaded_fy * y + height / 2.0
            xr = (u - width / 2.0) / loaded_fx
            yr = (v - height / 2.0) / loaded_fy
            max_ray_error = max(max_ray_error, abs(xr - x), abs(yr - y))
        record = {
            "image_id": int(image.id),
            "image_name": image.name,
            "camera_id": int(image.camera_id),
            "loaded_width": width,
            "loaded_height": height,
            "channels": channels,
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "tensor_bytes": theoretical,
            "loaded_fx": format(loaded_fx, ".17g"),
            "loaded_fy": format(loaded_fy, ".17g"),
            "loaded_cx": format(width / 2.0, ".17g"),
            "loaded_cy": format(height / 2.0, ".17g"),
            "R": [format(float(value), ".17g") for value in camera.R.reshape(-1)],
            "T": [format(float(value), ".17g") for value in camera.T.reshape(-1)],
            "FoVx": format(float(camera.FoVx), ".17g"),
            "FoVy": format(float(camera.FoVy), ".17g"),
        }
        if args.include_tensor_hashes:
            record.update({
                "tensor_sha256": _tensor_sha256(tensor),
                "world_view_transform_sha256": _tensor_sha256(camera.world_view_transform),
                "projection_matrix_sha256": _tensor_sha256(camera.projection_matrix),
                "full_proj_transform_sha256": _tensor_sha256(camera.full_proj_transform),
                "camera_center_sha256": _tensor_sha256(camera.camera_center),
            })
        records.append(record)
    if actual_tensor_bytes != theoretical_bytes:
        raise ValueError(f"camera tensor storage mismatch: actual={actual_tensor_bytes} theoretical={theoretical_bytes}")
    live_pil = sum(1 for obj in gc.get_objects() if isinstance(obj, Image.Image))
    jpeg_after_load = _jpeg_fds()
    allocated_after = int(torch.cuda.memory_allocated())
    reserved_after = int(torch.cuda.memory_reserved())
    for _ in range(args.stabilization_seconds):
        time.sleep(1)
    gc.collect()
    jpeg_after_stable = _jpeg_fds()
    fd_after = len(list(Path("/proc/self/fd").iterdir()))
    report = {
        "schema": "gs_gcp_original_3dgs_camera_load_preflight_v2",
        "status": "PASS",
        "source_root": str(source_root),
        "source_manifest_sha256": sha256_file(source_root.parent / "CAMERA_SUBSET_MANIFEST.json") if (source_root.parent / "CAMERA_SUBSET_MANIFEST.json").is_file() else None,
        "resolution": 4,
        "data_device": args.data_device,
        "materialization_mode": args.expected_materialization,
        "host_allocator_policy": args.host_allocator_policy,
        "malloc_trim_threshold_env": os.environ.get("MALLOC_TRIM_THRESHOLD_"),
        "libc_runtime": list(platform.libc_ver()),
        "tensor_hashes_included": bool(args.include_tensor_hashes),
        "camera_count": len(cameras),
        "camera_records_read_count": progress["camera_records_read_count"],
        "camera_tensors_materialized_count": progress["camera_tensors_materialized_count"],
        "currently_open_source_image_count": progress["currently_open_source_image_count"],
        "actual_camera_tensor_bytes": actual_tensor_bytes,
        "theoretical_camera_tensor_bytes": theoretical_bytes,
        "camera_tensor_byte_match": True,
        "torch_cuda_allocated_before": allocated_before,
        "torch_cuda_allocated_after": allocated_after,
        "torch_cuda_reserved_before": reserved_before,
        "torch_cuda_reserved_after": reserved_after,
        "fd_before": fd_before,
        "fd_after": fd_after,
        "jpeg_fds_after_load": jpeg_after_load,
        "jpeg_fds_after_stabilization": jpeg_after_stable,
        "live_pil_image_object_count": live_pil,
        "max_normalized_ray_coordinate_error": max_ray_error,
        "points3d_tracks_read": False,
        "camera_records": records,
    }
    if jpeg_after_stable:
        report["status"] = "CAMERA_LOAD_FAILURE"
        report["failure_reason"] = "source JPEG file descriptors remain open after stabilization"
    _write_json(args.report, report)
    lifecycle_handle.close()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
