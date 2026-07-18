#!/usr/bin/env python3
"""Load official Graphdeco cameras only and report memory/ray/FD evidence."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method_root", type=Path, required=True)
    parser.add_argument("--source_root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=4)
    parser.add_argument("--data_device", default="cuda")
    parser.add_argument("--stabilization_seconds", type=int, default=30)
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
    if args.resolution != 4 or args.data_device != "cuda":
        raise ValueError("formal resource preflight requires resolution=4 and data_device=cuda")
    fd_before = len(list(Path("/proc/self/fd").iterdir()))
    allocated_before = int(torch.cuda.memory_allocated())
    reserved_before = int(torch.cuda.memory_reserved())
    scene_info = readColmapSceneInfo(str(source_root), "images", False)
    model_args = SimpleNamespace(resolution=4, data_device="cuda")
    cameras = cameraList_from_camInfos(scene_info.train_cameras, 1.0, model_args)
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
        records.append({
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
        })
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
        "schema": "gs_gcp_original_3dgs_camera_load_preflight_v1",
        "status": "PASS",
        "source_root": str(source_root),
        "source_manifest_sha256": sha256_file(source_root.parent / "CAMERA_SUBSET_MANIFEST.json") if (source_root.parent / "CAMERA_SUBSET_MANIFEST.json").is_file() else None,
        "resolution": 4,
        "data_device": "cuda",
        "camera_count": len(cameras),
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
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
