#!/usr/bin/env python3
"""Stream-filter a COLMAP binary model to a frozen training-view track closure.

The implementation intentionally never decodes image pixels and never holds the
large images.bin or points3D.bin payload in memory.  Camera poses, keypoints and
point geometry are copied byte-for-byte; only held-out image records and their
point-track elements are removed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import tempfile
from pathlib import Path
from typing import BinaryIO


U64 = struct.Struct("<Q")
IMAGE_FIXED = struct.Struct("<i7di")
POINT_FIXED = struct.Struct("<Q3d3BdQ")
TRACK = struct.Struct("<ii")


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
    chunks = bytearray()
    while True:
        value = read_exact(handle, 1)
        chunks.extend(value)
        if value == b"\0":
            return bytes(chunks)


def require_sha(path: Path, expected: str) -> str:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"SHA256 mismatch for {path}: expected {expected}, got {actual}")
    return actual


def filter_images(
    source: Path, target: Path, train_names: set[str]
) -> tuple[set[int], int, int, int]:
    selected_ids: set[int] = set()
    selected_names: set[str] = set()
    source_selected_points2d = 0
    selected_positive = 0
    with source.open("rb") as src, target.open("wb") as dst:
        total = U64.unpack(read_exact(src, U64.size))[0]
        dst.write(U64.pack(len(train_names)))
        for _ in range(total):
            fixed = read_exact(src, IMAGE_FIXED.size)
            image_id = IMAGE_FIXED.unpack(fixed)[0]
            name_raw = read_c_string(src)
            name = name_raw[:-1].decode("utf-8")
            count_raw = read_exact(src, U64.size)
            count = U64.unpack(count_raw)[0]
            point_bytes = count * 24
            if name in train_names:
                if name in selected_names or image_id in selected_ids:
                    raise RuntimeError(f"duplicate selected COLMAP image: {name}/{image_id}")
                selected_names.add(name)
                selected_ids.add(image_id)
                dst.write(fixed)
                dst.write(name_raw)
                dst.write(count_raw)
                block = read_exact(src, point_bytes)
                dst.write(block)
                for offset in range(0, len(block), 24):
                    if struct.unpack_from("<q", block, offset + 16)[0] >= 0:
                        selected_positive += 1
                source_selected_points2d += count
            else:
                src.seek(point_bytes, 1)
        if src.tell() != source.stat().st_size or src.read(1):
            raise RuntimeError("trailing bytes in source images.bin")
    if selected_names != train_names:
        missing = sorted(train_names - selected_names)[:5]
        raise RuntimeError(f"source model is missing frozen training images: {missing}")
    return selected_ids, total, source_selected_points2d, selected_positive


def filter_points(
    source: Path,
    target: Path,
    selected_image_ids: set[int],
) -> tuple[int, int, int, int, int]:
    kept_points = 0
    kept_tracks = 0
    removed_tracks = 0
    removed_points = 0
    descriptor, temporary_name = tempfile.mkstemp(dir=target.parent)
    os.close(descriptor)
    body_path = Path(temporary_name)
    try:
        with body_path.open("wb") as body, source.open("rb") as src:
            total = U64.unpack(read_exact(src, U64.size))[0]
            for _ in range(total):
                fixed = read_exact(src, POINT_FIXED.size)
                track_count = POINT_FIXED.unpack(fixed)[-1]
                kept = bytearray()
                for _ in range(track_count):
                    pair = read_exact(src, TRACK.size)
                    image_id, _point2d_idx = TRACK.unpack(pair)
                    if image_id in selected_image_ids:
                        kept.extend(pair)
                    else:
                        removed_tracks += 1
                new_count = len(kept) // TRACK.size
                if new_count:
                    body.write(fixed[:-U64.size])
                    body.write(U64.pack(new_count))
                    body.write(kept)
                    kept_points += 1
                    kept_tracks += new_count
                else:
                    removed_points += 1
            if src.read(1):
                raise RuntimeError("trailing bytes in source points3D.bin")
        with target.open("wb") as dst, body_path.open("rb") as src_body:
            dst.write(U64.pack(kept_points))
            shutil.copyfileobj(src_body, dst, length=4 * 1024 * 1024)
    finally:
        body_path.unlink(missing_ok=True)
    return total, kept_points, kept_tracks, removed_tracks, removed_points


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_model", type=Path, required=True)
    parser.add_argument("--formal_manifest", type=Path, required=True)
    parser.add_argument("--output_model", type=Path, required=True)
    parser.add_argument("--output_manifest", type=Path, required=True)
    parser.add_argument("--expected_cameras_sha256", required=True)
    parser.add_argument("--expected_images_sha256", required=True)
    parser.add_argument("--expected_points3d_sha256", required=True)
    parser.add_argument("--expected_train_count", type=int, required=True)
    parser.add_argument("--expected_test_count", type=int, required=True)
    args = parser.parse_args()

    source = args.source_model.resolve()
    output = args.output_model.resolve()
    output_manifest = args.output_manifest.resolve()
    if output.exists() or output_manifest.exists():
        raise FileExistsError("output already exists; overwrite/resume is forbidden")
    source_hashes = {
        "cameras.bin": require_sha(source / "cameras.bin", args.expected_cameras_sha256),
        "images.bin": require_sha(source / "images.bin", args.expected_images_sha256),
        "points3D.bin": require_sha(source / "points3D.bin", args.expected_points3d_sha256),
    }
    formal = json.loads(args.formal_manifest.read_text(encoding="utf-8"))
    train_names = {row["image_name"] for row in formal["images"] if row["role"] == "train"}
    test_names = {row["image_name"] for row in formal["images"] if row["role"] == "test"}
    if len(train_names) != args.expected_train_count or len(test_names) != args.expected_test_count:
        raise RuntimeError("formal split counts differ from the exact bound counts")
    if train_names & test_names:
        raise RuntimeError("formal train and test name sets overlap")

    output.mkdir(parents=True)
    try:
        shutil.copyfile(source / "cameras.bin", output / "cameras.bin")
        selected_ids, source_images, source_point2d_count, positive_observations = filter_images(
            source / "images.bin", output / "images.bin", train_names
        )
        source_points, kept_points, kept_tracks, removed_tracks, removed_points = filter_points(
            source / "points3D.bin",
            output / "points3D.bin",
            selected_ids,
        )
        if kept_tracks != positive_observations:
            raise RuntimeError(
                "closed point tracks differ from selected-image positive observations: "
                f"{kept_tracks} != {positive_observations}"
            )
        derived_hashes = {name: sha256(output / name) for name in source_hashes}
        evidence = {
            "schema": "m3m_gcp_colmap_streaming_frozen_train_track_closure_v1",
            "scene": formal["scene"],
            "status": "PASS",
            "passed": True,
            "source_model": {"path": str(source), "sha256": source_hashes},
            "formal_manifest": {
                "path": str(args.formal_manifest.resolve()),
                "train_count": len(train_names),
                "test_count": len(test_names),
            },
            "derived_model": {
                "path": str(output),
                "source_image_count": source_images,
                "image_count": len(selected_ids),
                "source_point_count": source_points,
                "point_count": kept_points,
                "source_selected_point2d_count": source_point2d_count,
                "point2d_count": positive_observations,
                "track_element_count": kept_tracks,
                "sha256": derived_hashes,
            },
            "track_closure": {
                "removed_test_image_records": len(test_names),
                "retained_untriangulated_point2d_count": source_point2d_count
                - positive_observations,
                "removed_track_element_count": removed_tracks,
                "removed_unobserved_point_count": removed_points,
                "all_kept_tracks_reference_frozen_training_images": True,
                "image_track_reciprocity_count_equal": True,
            },
            "byte_semantics": {
                "camera_records_copied_without_decoding": True,
                "selected_image_records_including_untriangulated_points_copied_without_numeric_reencoding": True,
                "point_xyz_rgb_error_fields_copied_without_numeric_reencoding": True,
                "heldout_image_records_and_their_point_track_elements_removed": True,
                "retained_point_track_indices_unchanged": True,
            },
            "access_boundary": {
                "image_pixels_opened": 0,
                "test_rgb_opened": 0,
                "gcp_opened": 0,
                "lidar_opened": 0,
                "formal_training_started": False,
            },
            "materializer": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
        }
        output_manifest.parent.mkdir(parents=True, exist_ok=True)
        output_manifest.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(evidence, indent=2, sort_keys=True))
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
