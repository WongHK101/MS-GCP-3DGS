#!/usr/bin/env python3
"""Materialize and audit a train-ready COLMAP native-quarter scene."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image as PILImage


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def load_colmap_module(path: Path):
    spec = importlib.util.spec_from_file_location("read_write_model", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import COLMAP model utility: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_tracks(images, points) -> dict[str, int]:
    point_ids = set(points)
    observation_count = 0
    linked_count = 0
    for image in images.values():
        if len(image.xys) != len(image.point3D_ids):
            raise ValueError(f"Observation arrays differ for image {image.id}")
        observation_count += len(image.xys)
        for point_id in image.point3D_ids:
            numeric_id = int(point_id)
            if numeric_id == -1:
                continue
            linked_count += 1
            if numeric_id not in point_ids:
                raise ValueError(
                    f"Image {image.id} references absent point3D {numeric_id}"
                )
    track_count = 0
    for point in points.values():
        if len(point.image_ids) != len(point.point2D_idxs):
            raise ValueError(f"Track arrays differ for point3D {point.id}")
        track_count += len(point.image_ids)
        for image_id, point2d_idx in zip(point.image_ids, point.point2D_idxs):
            image = images.get(int(image_id))
            if image is None:
                raise ValueError(
                    f"Point3D {point.id} references absent image {image_id}"
                )
            index = int(point2d_idx)
            if index < 0 or index >= len(image.point3D_ids):
                raise ValueError(
                    f"Point3D {point.id} has invalid point2D index {index}"
                )
            if int(image.point3D_ids[index]) != int(point.id):
                raise ValueError(
                    f"Track mismatch: point={point.id}, image={image_id}, "
                    f"point2D_idx={index}"
                )
    if linked_count != track_count:
        raise ValueError(
            f"Linked observations and track elements differ: "
            f"{linked_count} != {track_count}"
        )
    return {
        "observation_count": observation_count,
        "linked_observation_count": linked_count,
        "track_element_count": track_count,
    }


def inspect_image(path: Path) -> dict[str, object]:
    with PILImage.open(path) as image:
        width, height = image.size
        mode = image.mode
        image.verify()
    return {
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "width": width,
        "height": height,
        "mode": mode,
    }


def reprojection_audit(images, points, camera) -> dict[str, object]:
    if camera.model != "PINHOLE" or len(camera.params) != 4:
        raise ValueError(f"Unsupported audit camera: {camera}")
    fx, fy, cx, cy = [float(value) for value in camera.params]
    residual_chunks = []
    linked_count = 0
    behind_count = 0
    for image_id in sorted(images):
        image = images[image_id]
        valid_indices = np.flatnonzero(image.point3D_ids != -1)
        if not len(valid_indices):
            continue
        point_ids = image.point3D_ids[valid_indices]
        xyz = np.asarray(
            [points[int(point_id)].xyz for point_id in point_ids],
            dtype=np.float64,
        )
        camera_xyz = (image.qvec2rotmat() @ xyz.T).T + image.tvec
        behind_count += int(np.count_nonzero(camera_xyz[:, 2] <= 0.0))
        projected = np.column_stack(
            (
                fx * camera_xyz[:, 0] / camera_xyz[:, 2] + cx,
                fy * camera_xyz[:, 1] / camera_xyz[:, 2] + cy,
            )
        )
        residuals = np.linalg.norm(
            projected - image.xys[valid_indices], axis=1
        )
        if not np.isfinite(residuals).all():
            raise ValueError(f"Non-finite residual in image {image_id}")
        residual_chunks.append(residuals)
        linked_count += len(residuals)
    if not residual_chunks:
        raise ValueError("No linked observations for reprojection audit")
    residuals = np.concatenate(residual_chunks)
    result = {
        "linked_observation_count": linked_count,
        "behind_camera_count": behind_count,
        "mean_px": float(np.mean(residuals)),
        "rmse_px": float(np.sqrt(np.mean(np.square(residuals)))),
        "median_px": float(np.median(residuals)),
        "p95_px": float(np.quantile(residuals, 0.95)),
        "p99_px": float(np.quantile(residuals, 0.99)),
        "max_px": float(np.max(residuals)),
    }
    if behind_count != 0:
        raise ValueError(f"Points behind cameras: {behind_count}")
    if result["mean_px"] >= 1.0 or result["p95_px"] >= 2.0:
        raise ValueError(f"Reprojection audit exceeded guardrails: {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--frozen-model", required=True, type=Path)
    parser.add_argument("--read-write-model", required=True, type=Path)
    parser.add_argument("--local-colmap", required=True, type=Path)
    parser.add_argument("--image-workers", type=int, default=4)
    args = parser.parse_args()

    for path in (
        args.candidate_root,
        args.frozen_model,
        args.read_write_model,
        args.local_colmap,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    preparation_path = args.candidate_root / "evidence" / "PREPARATION.json"
    download_path = args.candidate_root / "evidence" / "DOWNLOAD.json"
    remote_launch_path = args.candidate_root / "evidence" / "REMOTE_LAUNCH.json"
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    download = json.loads(download_path.read_text(encoding="utf-8"))
    remote_launch = json.loads(remote_launch_path.read_text(encoding="utf-8"))
    if preparation.get("scene") != args.scene or download.get("scene") != args.scene:
        raise ValueError("Scene identity differs across preparation/download evidence")
    if remote_launch.get("scene") != args.scene or remote_launch.get("status") != "launched":
        raise ValueError("Remote launch evidence has an invalid scene or status")
    generation_threads = int(remote_launch["command_contract"]["num_threads"])
    if generation_threads < 1:
        raise ValueError(f"Invalid generation thread count: {generation_threads}")
    expected_images = int(preparation["raw_images"]["count"])

    colmap = load_colmap_module(args.read_write_model)
    native_root = args.candidate_root / "evidence" / "native_output" / "sparse"
    images_root = args.candidate_root / "images"
    output_root = args.candidate_root / "sparse" / "0"
    output_root.mkdir(parents=True, exist_ok=True)
    if any(output_root.iterdir()):
        raise FileExistsError(f"Refusing non-empty final model root: {output_root}")

    native_cameras = colmap.read_cameras_binary(str(native_root / "cameras.bin"))
    native_images = colmap.read_images_binary(str(native_root / "images.bin"))
    native_points = colmap.read_points3D_binary(str(native_root / "points3D.bin"))
    frozen_cameras = colmap.read_cameras_binary(
        str(args.frozen_model / "cameras.bin")
    )
    frozen_images = colmap.read_images_binary(str(args.frozen_model / "images.bin"))
    if len(native_cameras) != 1 or len(frozen_cameras) != 1:
        raise ValueError("Expected one native and one frozen camera")
    if len(native_images) != expected_images or len(frozen_images) != expected_images:
        raise ValueError(
            f"Image count mismatch: native={len(native_images)}, "
            f"frozen={len(frozen_images)}, expected={expected_images}"
        )
    if native_points:
        raise ValueError("Native pose-only output unexpectedly contains points3D")
    native_camera = next(iter(native_cameras.values()))
    frozen_camera = next(iter(frozen_cameras.values()))
    if native_camera.model != "PINHOLE" or frozen_camera.model != "PINHOLE":
        raise ValueError("Native/frozen camera models must both be PINHOLE")
    if max(native_camera.width, native_camera.height) != 1414:
        raise ValueError(f"Native output did not honor max_image_size=1414: {native_camera}")
    scale_x = native_camera.width / frozen_camera.width
    scale_y = native_camera.height / frozen_camera.height
    expected_params = np.asarray(frozen_camera.params, dtype=np.float64) * np.array(
        [scale_x, scale_y, scale_x, scale_y], dtype=np.float64
    )
    camera_error = np.abs(
        np.asarray(native_camera.params, dtype=np.float64) - expected_params
    )
    if float(camera_error.max()) > 1e-10:
        raise ValueError(f"Native camera is not a frozen-camera scale: {camera_error}")

    actual_image_names = {
        path.name for path in images_root.iterdir() if path.is_file()
    }
    if actual_image_names != {image.name for image in native_images.values()}:
        raise ValueError("Native image files and model names differ")

    scaled_images = {}
    max_qvec_error = 0.0
    max_tvec_error = 0.0
    for image_id in sorted(frozen_images):
        frozen = frozen_images[image_id]
        native = native_images.get(image_id)
        if native is None:
            raise ValueError(f"Missing native image ID {image_id}")
        if frozen.name != native.name or frozen.camera_id != native.camera_id:
            raise ValueError(f"Image identity mismatch for ID {image_id}")
        qvec_error = float(np.max(np.abs(frozen.qvec - native.qvec)))
        tvec_error = float(np.max(np.abs(frozen.tvec - native.tvec)))
        max_qvec_error = max(max_qvec_error, qvec_error)
        max_tvec_error = max(max_tvec_error, tvec_error)
        if qvec_error != 0.0 or tvec_error != 0.0:
            raise ValueError(f"Pose mismatch for image ID {image_id}")
        scaled_xys = np.asarray(frozen.xys, dtype=np.float64).copy()
        scaled_xys[:, 0] *= scale_x
        scaled_xys[:, 1] *= scale_y
        scaled_images[image_id] = colmap.Image(
            id=frozen.id,
            qvec=np.asarray(native.qvec, dtype=np.float64),
            tvec=np.asarray(native.tvec, dtype=np.float64),
            camera_id=native.camera_id,
            name=native.name,
            xys=scaled_xys,
            point3D_ids=np.asarray(frozen.point3D_ids, dtype=np.int64),
        )

    print(
        json.dumps({"status": "loading_points3D", "scene": args.scene}),
        flush=True,
    )
    frozen_points = colmap.read_points3D_binary(
        str(args.frozen_model / "points3D.bin")
    )
    track_report = validate_tracks(scaled_images, frozen_points)

    colmap.write_cameras_binary(native_cameras, str(output_root / "cameras.bin"))
    colmap.write_images_binary(scaled_images, str(output_root / "images.bin"))
    shutil.copy2(args.frozen_model / "points3D.bin", output_root / "points3D.bin")
    shutil.copy2(args.frozen_model / "points3D.ply", output_root / "points3D.ply")
    modern_files = []
    for name in ("rigs.bin", "frames.bin"):
        source = native_root / name
        if source.is_file():
            shutil.copy2(source, output_root / name)
            modern_files.append(name)
    if sha256_file(output_root / "points3D.bin") != sha256_file(
        args.frozen_model / "points3D.bin"
    ):
        raise ValueError("points3D.bin was not preserved byte-for-byte")
    if sha256_file(output_root / "points3D.ply") != sha256_file(
        args.frozen_model / "points3D.ply"
    ):
        raise ValueError("points3D.ply was not preserved byte-for-byte")

    check_cameras = colmap.read_cameras_binary(str(output_root / "cameras.bin"))
    check_images = colmap.read_images_binary(str(output_root / "images.bin"))
    if set(check_cameras) != set(native_cameras) or set(check_images) != set(scaled_images):
        raise ValueError("Final camera/image IDs changed after binary round trip")
    for image_id, checked in check_images.items():
        expected = scaled_images[image_id]
        if not (
            checked.name == expected.name
            and checked.camera_id == expected.camera_id
            and np.array_equal(checked.qvec, expected.qvec)
            and np.array_equal(checked.tvec, expected.tvec)
            and np.array_equal(checked.xys, expected.xys)
            and np.array_equal(checked.point3D_ids, expected.point3D_ids)
        ):
            raise ValueError(f"Final image changed after binary round trip: {image_id}")

    analyzer = subprocess.run(
        [str(args.local_colmap), "model_analyzer", "--path", str(output_root)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=1800,
        check=False,
    )
    analyzer_stdout = args.candidate_root / "evidence" / "model_analyzer.stdout.log"
    analyzer_stderr = args.candidate_root / "evidence" / "model_analyzer.stderr.log"
    analyzer_stdout.write_text(analyzer.stdout, encoding="utf-8", newline="\n")
    analyzer_stderr.write_text(analyzer.stderr, encoding="utf-8", newline="\n")
    if analyzer.returncode != 0:
        raise RuntimeError(
            f"Local COLMAP model_analyzer failed ({analyzer.returncode}): "
            f"{analyzer.stderr[-1000:]}"
        )

    print(
        json.dumps({"status": "reprojection_audit", "scene": args.scene}),
        flush=True,
    )
    reprojection = reprojection_audit(check_images, frozen_points, native_camera)
    if reprojection["linked_observation_count"] != track_report["linked_observation_count"]:
        raise ValueError("Track and reprojection linked-observation counts differ")

    image_paths = sorted(path for path in images_root.iterdir() if path.is_file())
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.image_workers
    ) as pool:
        image_records = list(pool.map(inspect_image, image_paths))
    decoded_sizes = {(item["width"], item["height"]) for item in image_records}
    decoded_modes = {item["mode"] for item in image_records}
    if decoded_sizes != {(native_camera.width, native_camera.height)}:
        raise ValueError(f"Unexpected decoded image dimensions: {decoded_sizes}")
    if decoded_modes != {"RGB"}:
        raise ValueError(f"Unexpected decoded image modes: {decoded_modes}")
    image_manifest = args.candidate_root / "evidence" / "images.sha256"
    with image_manifest.open("w", encoding="utf-8", newline="\n") as handle:
        for item in image_records:
            handle.write(f"{item['sha256']}  images/{item['name']}\n")

    report = {
        "schema": "gs-gcp-colmap-native-quarter-package-audit-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scene": args.scene,
        "status": "pass",
        "candidate_root": str(args.candidate_root.resolve()),
        "standard_colmap_layout": True,
        "train_ready_for_standard_colmap_loaders": True,
        "gpu_used": False,
        "training_started": False,
        "image_generation": {
            "generator": "COLMAP 4.0.4 image_undistorter",
            "CUDA_VISIBLE_DEVICES": "",
            "max_image_size": 1414,
            "num_threads": generation_threads,
        },
        "camera": {
            "id": native_camera.id,
            "model": native_camera.model,
            "width": native_camera.width,
            "height": native_camera.height,
            "params": native_camera.params.tolist(),
            "scale_x_from_frozen": scale_x,
            "scale_y_from_frozen": scale_y,
            "param_abs_error_from_exact_scale": camera_error.tolist(),
        },
        "counts": {
            "cameras": len(check_cameras),
            "images": len(check_images),
            "points3D": len(frozen_points),
            "image_bytes": sum(int(item["bytes"]) for item in image_records),
        },
        "identity": {
            "max_qvec_abs_error": max_qvec_error,
            "max_tvec_abs_error": max_tvec_error,
            "points3D_bin_byte_preserved": True,
            "points3D_ply_byte_preserved": True,
            "modern_colmap_files": modern_files,
        },
        "tracks": track_report,
        "reprojection": reprojection,
        "decoded_images": {
            "count": len(image_records),
            "sizes": [list(size) for size in sorted(decoded_sizes)],
            "modes": sorted(decoded_modes),
            "sha256_mismatch_count": 0,
        },
        "model_analyzer": {
            "executable": str(args.local_colmap.resolve()),
            "returncode": analyzer.returncode,
            "stdout": file_record(analyzer_stdout),
            "stderr": file_record(analyzer_stderr),
        },
        "model_files": {
            path.name: file_record(path)
            for path in sorted(output_root.iterdir())
            if path.is_file()
        },
        "image_sha256_manifest": file_record(image_manifest),
        "sources": {
            "preparation": file_record(preparation_path),
            "download": file_record(download_path),
            "remote_launch": file_record(remote_launch_path),
            "native_camera_model": file_record(native_root / "cameras.bin"),
            "native_image_model": file_record(native_root / "images.bin"),
            "frozen_camera_model": file_record(args.frozen_model / "cameras.bin"),
            "frozen_image_model": file_record(args.frozen_model / "images.bin"),
            "frozen_points3D": file_record(args.frozen_model / "points3D.bin"),
            "frozen_points3D_ply": file_record(args.frozen_model / "points3D.ply"),
            "read_write_model": file_record(args.read_write_model),
            "audit_script": file_record(Path(__file__)),
        },
    }
    report_path = args.candidate_root / "evidence" / "PACKAGE_AUDIT.json"
    with report_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")

    readme = args.candidate_root / "README.md"
    readme.write_text(
        "\n".join(
            [
                f"# {args.scene} COLMAP-native quarter candidate",
                "",
                "- `images/`: COLMAP 4.0.4 `image_undistorter` output with "
                f"`--max_image_size 1414 --num_threads {generation_threads}` "
                "and GPU hidden.",
                "- `sparse/0/`: complete train-ready COLMAP model.",
                "- `evidence/PACKAGE_AUDIT.json`: integrity, identity, track, "
                "reprojection, and loader-layout audit.",
                "- The frozen full-resolution source remains unchanged.",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "scene": args.scene,
                "images": len(check_images),
                "points3D": len(frozen_points),
                "reprojection": reprojection,
                "report": str(report_path.resolve()),
            },
            ensure_ascii=False,
            allow_nan=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
