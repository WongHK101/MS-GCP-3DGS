#!/usr/bin/env python3
"""Prepare a pose-only COLMAP input without copying the raw scene images."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import re
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image as PILImage


SCENE_PATTERN = re.compile(r"^gcp_[0-9]+_[0-9]{8}$")
CAMERA_ARCHIVE_PREFIX = (
    "root/autodl-tmp/runs/ms-gcp-3dgs/"
    "colmap-4.0.4-global-formal-20260616"
)


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


def load_scene_hashes(path: Path, scene: str) -> dict[str, str]:
    prefix = f"./{scene}/"
    hashes: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        digest, relative = raw_line.split(maxsplit=1)
        relative = relative.strip()
        if not relative.startswith(prefix):
            continue
        name = relative[len(prefix) :]
        if Path(name).name != name:
            raise ValueError(f"Unexpected nested raw path: {relative}")
        if name in hashes:
            raise ValueError(f"Duplicate historical hash entry: {name}")
        hashes[name] = digest.lower()
    if not hashes:
        raise ValueError(f"No historical hashes found for {scene}")
    return hashes


def extract_raw_camera(archive: Path, scene: str, destination: Path) -> str:
    member_name = (
        f"{CAMERA_ARCHIVE_PREFIX}/{scene}/RGB/sparse_aligned/0/cameras.txt"
    )
    with tarfile.open(archive, "r:gz") as handle:
        member = handle.getmember(member_name)
        source = handle.extractfile(member)
        if source is None:
            raise FileNotFoundError(member_name)
        payload = source.read()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return member_name


def inspect_raw(item: tuple[Path, str]) -> dict[str, object]:
    path, expected_digest = item
    with PILImage.open(path) as image:
        width, height = image.size
        mode = image.mode
    digest = sha256_file(path)
    if digest.lower() != expected_digest:
        raise ValueError(
            f"Historical SHA256 mismatch for {path.name}: "
            f"{digest} != {expected_digest}"
        )
    return {
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": digest,
        "width": width,
        "height": height,
        "mode": mode,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--reference-model", required=True, type=Path)
    parser.add_argument("--historical-sha256", required=True, type=Path)
    parser.add_argument("--compact-evidence", required=True, type=Path)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--read-write-model", required=True, type=Path)
    parser.add_argument("--hash-workers", type=int, default=4)
    args = parser.parse_args()

    if not SCENE_PATTERN.fullmatch(args.scene):
        raise ValueError(f"Invalid scene name: {args.scene}")
    for path in (
        args.raw_root,
        args.reference_model,
        args.historical_sha256,
        args.compact_evidence,
        args.read_write_model,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.hash_workers < 1 or args.hash_workers > 16:
        raise ValueError("--hash-workers must be between 1 and 16")

    evidence_root = args.candidate_root / "evidence"
    input_root = evidence_root / "pose_only_input"
    sparse_root = input_root / "sparse" / "0"
    text_root = input_root / "model_text" / "0"
    for path in (evidence_root, sparse_root, text_root):
        path.mkdir(parents=True, exist_ok=True)
    protected_outputs = [
        sparse_root / "cameras.bin",
        sparse_root / "images.bin",
        sparse_root / "points3D.bin",
        evidence_root / "PREPARATION.json",
    ]
    if any(path.exists() for path in protected_outputs):
        raise FileExistsError("Refusing to overwrite an existing prepared scene")

    raw_camera_path = evidence_root / "archived_raw_cameras.txt"
    archive_member = extract_raw_camera(
        args.compact_evidence, args.scene, raw_camera_path
    )
    colmap = load_colmap_module(args.read_write_model)
    raw_cameras = colmap.read_cameras_text(str(raw_camera_path))
    frozen_cameras = colmap.read_cameras_binary(
        str(args.reference_model / "cameras.bin")
    )
    frozen_images = colmap.read_images_binary(
        str(args.reference_model / "images.bin")
    )
    if len(raw_cameras) != 1 or len(frozen_cameras) != 1:
        raise ValueError("Expected exactly one raw and one frozen camera")
    raw_camera = next(iter(raw_cameras.values()))
    frozen_camera = next(iter(frozen_cameras.values()))
    if (raw_camera.model, raw_camera.width, raw_camera.height) != (
        "SIMPLE_RADIAL",
        5280,
        3956,
    ):
        raise ValueError(f"Unexpected archived raw camera: {raw_camera}")
    if frozen_camera.model != "PINHOLE":
        raise ValueError(f"Unexpected frozen camera model: {frozen_camera}")
    if set(raw_cameras) != set(frozen_cameras):
        raise ValueError("Raw and frozen camera IDs differ")

    expected_hashes = load_scene_hashes(args.historical_sha256, args.scene)
    frozen_names = [image.name for image in frozen_images.values()]
    if len(frozen_names) != len(set(frozen_names)):
        raise ValueError("Frozen image names are not unique")
    if set(frozen_names) != set(expected_hashes):
        raise ValueError(
            "Frozen image names and historical raw hash entries differ: "
            f"frozen={len(frozen_names)}, hashes={len(expected_hashes)}"
        )
    disk_names = {
        path.name
        for path in args.raw_root.iterdir()
        if path.is_file() and path.name.endswith("_D.JPG")
    }
    if disk_names != set(frozen_names):
        raise ValueError(
            "Raw image files and frozen model names differ: "
            f"raw={len(disk_names)}, frozen={len(frozen_names)}"
        )

    print(
        json.dumps(
            {
                "status": "hashing_raw_images",
                "scene": args.scene,
                "image_count": len(frozen_names),
            },
            separators=(",", ":"),
        ),
        flush=True,
    )
    work = [
        (args.raw_root / name, expected_hashes[name])
        for name in sorted(frozen_names)
    ]
    records: list[dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.hash_workers
    ) as pool:
        for index, record in enumerate(pool.map(inspect_raw, work), start=1):
            records.append(record)
            if index % 100 == 0 or index == len(work):
                print(
                    json.dumps(
                        {
                            "status": "hashing_raw_images",
                            "scene": args.scene,
                            "done": index,
                            "total": len(work),
                        },
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
    dimensions = {(record["width"], record["height"]) for record in records}
    modes = {record["mode"] for record in records}
    if dimensions != {(raw_camera.width, raw_camera.height)} or modes != {"RGB"}:
        raise ValueError(
            f"Unexpected raw image metadata: dimensions={dimensions}, modes={modes}"
        )

    pose_only_images = {}
    max_qnorm_error = 0.0
    for image_id in sorted(frozen_images):
        image = frozen_images[image_id]
        if image.camera_id not in raw_cameras:
            raise ValueError(f"Image {image.name} references absent raw camera")
        qnorm_error = abs(float(np.linalg.norm(image.qvec)) - 1.0)
        max_qnorm_error = max(max_qnorm_error, qnorm_error)
        if not np.isfinite(np.concatenate((image.qvec, image.tvec))).all():
            raise ValueError(f"Non-finite pose for {image.name}")
        pose_only_images[image_id] = colmap.Image(
            id=image.id,
            qvec=np.asarray(image.qvec, dtype=np.float64),
            tvec=np.asarray(image.tvec, dtype=np.float64),
            camera_id=image.camera_id,
            name=image.name,
            xys=np.empty((0, 2), dtype=np.float64),
            point3D_ids=np.empty((0,), dtype=np.int64),
        )
    if max_qnorm_error > 1e-6:
        raise ValueError(f"Non-unit quaternion; max norm error={max_qnorm_error}")

    colmap.write_cameras_binary(raw_cameras, str(sparse_root / "cameras.bin"))
    colmap.write_images_binary(pose_only_images, str(sparse_root / "images.bin"))
    colmap.write_points3D_binary({}, str(sparse_root / "points3D.bin"))
    colmap.write_cameras_text(raw_cameras, str(text_root / "cameras.txt"))
    colmap.write_images_text(pose_only_images, str(text_root / "images.txt"))
    colmap.write_points3D_text({}, str(text_root / "points3D.txt"))

    check_cameras = colmap.read_cameras_binary(str(sparse_root / "cameras.bin"))
    check_images = colmap.read_images_binary(str(sparse_root / "images.bin"))
    check_points = colmap.read_points3D_binary(str(sparse_root / "points3D.bin"))
    if check_points or set(check_cameras) != set(raw_cameras):
        raise ValueError("Pose-only binary round trip changed camera/point identities")
    if set(check_images) != set(frozen_images):
        raise ValueError("Pose-only binary round trip changed image IDs")
    max_qvec_error = 0.0
    max_tvec_error = 0.0
    for image_id, checked in check_images.items():
        reference = frozen_images[image_id]
        if checked.name != reference.name or checked.camera_id != reference.camera_id:
            raise ValueError(f"Image identity changed for ID {image_id}")
        max_qvec_error = max(
            max_qvec_error,
            float(np.max(np.abs(checked.qvec - reference.qvec))),
        )
        max_tvec_error = max(
            max_tvec_error,
            float(np.max(np.abs(checked.tvec - reference.tvec))),
        )
        if len(checked.xys) or len(checked.point3D_ids):
            raise ValueError(f"Image {image_id} retained feature observations")
    if max_qvec_error != 0.0 or max_tvec_error != 0.0:
        raise ValueError("Pose changed during pose-only binary round trip")

    sha_manifest = input_root / "raw_images.sha256"
    with sha_manifest.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(f"{record['sha256']}  {record['name']}\n")
    bytes_total = sum(int(record["bytes"]) for record in records)
    estimated_seconds = len(records) * 946.092 / 94.0
    report = {
        "schema": "gs-gcp-colmap-native-quarter-preparation-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scene": args.scene,
        "status": "pass",
        "gpu_used": False,
        "training_started": False,
        "raw_images": {
            "root": str(args.raw_root.resolve()),
            "count": len(records),
            "bytes": bytes_total,
            "dimensions": [raw_camera.width, raw_camera.height],
            "mode": "RGB",
            "historical_sha256_mismatch_count": 0,
        },
        "archived_raw_camera": {
            "archive_member": archive_member,
            "id": raw_camera.id,
            "model": raw_camera.model,
            "width": raw_camera.width,
            "height": raw_camera.height,
            "params": raw_camera.params.tolist(),
        },
        "frozen_reference": {
            "model_root": str(args.reference_model.resolve()),
            "camera": {
                "id": frozen_camera.id,
                "model": frozen_camera.model,
                "width": frozen_camera.width,
                "height": frozen_camera.height,
                "params": frozen_camera.params.tolist(),
            },
            "image_count": len(frozen_images),
        },
        "pose_only_model": {
            "root": str(sparse_root.resolve()),
            "camera_count": len(check_cameras),
            "image_count": len(check_images),
            "point_count": len(check_points),
            "max_qvec_abs_error": max_qvec_error,
            "max_tvec_abs_error": max_tvec_error,
            "max_quaternion_norm_error": max_qnorm_error,
            "files": {
                name: file_record(sparse_root / name)
                for name in ("cameras.bin", "images.bin", "points3D.bin")
            },
        },
        "raw_sha256_manifest": file_record(sha_manifest),
        "runtime_gate": {
            "basis_scene": "gcp_3000_20260602",
            "basis_seconds": 946.092,
            "basis_images": 94,
            "estimated_seconds": estimated_seconds,
            "estimated_hours": estimated_seconds / 3600.0,
            "under_10_hours": estimated_seconds < 10 * 3600,
        },
        "sources": {
            "compact_evidence": file_record(args.compact_evidence),
            "historical_sha256": file_record(args.historical_sha256),
            "read_write_model": file_record(args.read_write_model),
            "preparation_script": file_record(Path(__file__)),
        },
    }
    report_path = evidence_root / "PREPARATION.json"
    with report_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "status": "pass",
                "scene": args.scene,
                "image_count": len(records),
                "bytes": bytes_total,
                "estimated_hours": estimated_seconds / 3600.0,
                "report": str(report_path.resolve()),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
