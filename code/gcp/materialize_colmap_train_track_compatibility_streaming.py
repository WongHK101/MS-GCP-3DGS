#!/usr/bin/env python3
"""Select frozen train image records from an all-image COLMAP model by streaming.

The all-image points3D.bin is retained byte-identically because CityGaussianV2
and CityGS-X use this view only for their qualified scale/neighbor preparation,
which explicitly selects training-image observations.  This is not a reciprocal
standalone reconstruction; consumers that require one use the separate track
closure materializer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
from pathlib import Path
from typing import BinaryIO


U64 = struct.Struct("<Q")
IMAGE_FIXED = struct.Struct("<i7di")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_exact(handle: BinaryIO, size: int) -> bytes:
    value = handle.read(size)
    if len(value) != size:
        raise EOFError(f"expected {size} bytes, received {len(value)}")
    return value


def read_c_string(handle: BinaryIO) -> bytes:
    value = bytearray()
    while True:
        byte = read_exact(handle, 1)
        value.extend(byte)
        if byte == b"\0":
            return bytes(value)


def require_sha(path: Path, expected: str, label: str) -> str:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"{label} SHA mismatch: expected {expected}, got {actual}")
    return actual


def formal_headers(path: Path) -> dict[str, bytes]:
    records: dict[str, bytes] = {}
    with path.open("rb") as handle:
        total = U64.unpack(read_exact(handle, U64.size))[0]
        for _ in range(total):
            fixed = read_exact(handle, IMAGE_FIXED.size)
            name_raw = read_c_string(handle)
            name = name_raw[:-1].decode("utf-8")
            count = U64.unpack(read_exact(handle, U64.size))[0]
            if count:
                raise RuntimeError("formal train images.bin unexpectedly contains 2D records")
            if name in records:
                raise RuntimeError(f"duplicate formal image name: {name}")
            records[name] = fixed
        if handle.read(1):
            raise RuntimeError("trailing bytes in formal images.bin")
    return records


def select_train_images(
    source: Path, target: Path, train_headers: dict[str, bytes], full_names: set[str]
) -> tuple[int, int, int]:
    selected: set[str] = set()
    keypoints = 0
    linked = 0
    seen: set[str] = set()
    with source.open("rb") as src, target.open("wb") as dst:
        total = U64.unpack(read_exact(src, U64.size))[0]
        dst.write(U64.pack(len(train_headers)))
        for _ in range(total):
            fixed = read_exact(src, IMAGE_FIXED.size)
            name_raw = read_c_string(src)
            name = name_raw[:-1].decode("utf-8")
            if name in seen:
                raise RuntimeError(f"duplicate all-image record: {name}")
            seen.add(name)
            count_raw = read_exact(src, U64.size)
            count = U64.unpack(count_raw)[0]
            points = read_exact(src, count * 24)
            if name in train_headers:
                if fixed != train_headers[name]:
                    raise RuntimeError(f"image identity or pose differs from formal train model: {name}")
                dst.write(fixed)
                dst.write(name_raw)
                dst.write(count_raw)
                dst.write(points)
                selected.add(name)
                keypoints += count
                for offset in range(0, len(points), 24):
                    if struct.unpack_from("<q", points, offset + 16)[0] >= 0:
                        linked += 1
        if src.read(1):
            raise RuntimeError("trailing bytes in all-image images.bin")
    if seen != full_names:
        raise RuntimeError("all-image COLMAP inventory differs from the frozen split manifest")
    if selected != set(train_headers):
        raise RuntimeError("selected train image inventory is incomplete")
    return total, keypoints, linked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--formal-train-model", type=Path, required=True)
    parser.add_argument("--formal-manifest", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--expected-source-cameras-sha256", required=True)
    parser.add_argument("--expected-source-images-sha256", required=True)
    parser.add_argument("--expected-source-points3d-sha256", required=True)
    parser.add_argument("--expected-formal-cameras-sha256", required=True)
    parser.add_argument("--expected-formal-images-sha256", required=True)
    parser.add_argument("--expected-train-count", type=int, required=True)
    parser.add_argument("--expected-test-count", type=int, required=True)
    args = parser.parse_args()

    source = args.source_model.resolve()
    formal = args.formal_train_model.resolve()
    output = args.output_model.resolve()
    evidence_path = args.output_manifest.resolve()
    if output.exists() or evidence_path.exists():
        raise FileExistsError("output already exists; overwrite/resume is forbidden")
    source_hashes = {
        "cameras.bin": require_sha(source / "cameras.bin", args.expected_source_cameras_sha256, "source cameras"),
        "images.bin": require_sha(source / "images.bin", args.expected_source_images_sha256, "source images"),
        "points3D.bin": require_sha(source / "points3D.bin", args.expected_source_points3d_sha256, "source points"),
    }
    formal_hashes = {
        "cameras.bin": require_sha(formal / "cameras.bin", args.expected_formal_cameras_sha256, "formal cameras"),
        "images.bin": require_sha(formal / "images.bin", args.expected_formal_images_sha256, "formal images"),
    }
    if source_hashes["cameras.bin"] != formal_hashes["cameras.bin"]:
        raise RuntimeError("all-image and formal-train native-quarter cameras are not identical")
    manifest = json.loads(args.formal_manifest.read_text(encoding="utf-8"))
    train_names = {row["image_name"] for row in manifest["images"] if row["role"] == "train"}
    test_names = {row["image_name"] for row in manifest["images"] if row["role"] == "test"}
    if len(train_names) != args.expected_train_count or len(test_names) != args.expected_test_count:
        raise RuntimeError("frozen split count mismatch")
    headers = formal_headers(formal / "images.bin")
    if set(headers) != train_names:
        raise RuntimeError("formal train images.bin inventory mismatch")

    output.mkdir(parents=True)
    try:
        os.link(source / "cameras.bin", output / "cameras.bin")
        os.link(source / "points3D.bin", output / "points3D.bin")
        source_count, keypoints, linked = select_train_images(
            source / "images.bin", output / "images.bin", headers, train_names | test_names
        )
        derived_hashes = {name: sha256(output / name) for name in source_hashes}
        evidence = {
            "schema": "m3m_gcp_native_quarter_city_track_compatibility_streaming_v1",
            "scene": manifest["scene"],
            "status": "PASS",
            "passed": True,
            "role": "non-authoritative train-image compatibility view derived after shared all-image SfM",
            "source_model": {
                "path": str(source),
                "image_count": source_count,
                "sha256": source_hashes,
            },
            "formal_train_model": {
                "path": str(formal),
                "image_count": len(headers),
                "sha256": formal_hashes,
            },
            "derived_model": {
                "path": str(output),
                "image_count": len(headers),
                "heldout_image_record_count": 0,
                "keypoint_count": keypoints,
                "linked_observation_count": linked,
                "sha256": derived_hashes,
            },
            "validation": {
                "all_image_sfm_precedes_split": True,
                "selected_image_records_byte_preserved": True,
                "selected_pose_and_identity_byte_equal_to_formal_train": True,
                "source_points3d_bin_byte_identical": derived_hashes["points3D.bin"] == source_hashes["points3D.bin"],
                "heldout_image_records_absent": True,
                "pixel_resampling_or_reencoding": False,
            },
            "access_boundary": {
                "heldout_rgb_opened": 0,
                "gcp_opened": 0,
                "lidar_opened": 0,
                "formal_training_started": False,
                "note": "points3D.bin retains shared all-image tracks; qualified consumers select only training-image observations",
            },
            "materializer": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
        }
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(evidence, indent=2, sort_keys=True))
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
