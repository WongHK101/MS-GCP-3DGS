#!/usr/bin/env python3
"""Create the exact 211-camera GCP loader root with no real RGB pixels."""

from __future__ import annotations

import argparse
import json
import os
import struct
import zlib
from collections import Counter
from pathlib import Path
from typing import Any

from materialize_m3m_native_quarter_evaluation_subset import materialize_subset
from m3m_gcp_lidar_artifacts import canonical_sha256, sha256_file


SCENE = "gcp_100000_20260610"
FORMAL_INPUT_SHA = "c2cf9e951d95fee12a28d942e95c5c420df55bc364738b3f8737fed1c78bef3d"
FORMAL_INPUT_CANONICAL_SHA = "5b4fe34743310bd2225feb2dd236200606be933002fec19d2c9ecb9f3ba6769d"
PROTOCOL_OBSERVATIONS_SHA = "4332c503b35a51b36d0dc679b5d318c936219df3abb7dc9ac8115593e3a5ae52"
PROTOCOL_RELEASE_SHA = "21fbac75d66433169535ea7440c31393f7a5ecdb4ed94fcefd31d1780c28bea4"


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def black_png(width: int, height: int) -> bytes:
    signature = b"\x89PNG\r\n\x1a\n"
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    scanline = b"\x00" + (b"\x00" * (width * 3))
    pixels = zlib.compress(scanline * height, level=9)
    return signature + png_chunk(b"IHDR", header) + png_chunk(b"IDAT", pixels) + png_chunk(b"IEND", b"")


def identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(
            descriptor,
            (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
    finally:
        os.close(descriptor)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-input-manifest", type=Path, required=True)
    parser.add_argument("--protocol-observations", type=Path, required=True)
    parser.add_argument("--protocol-release", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()

    formal_path = args.formal_input_manifest.resolve()
    observations_path = args.protocol_observations.resolve()
    release_path = args.protocol_release.resolve()
    output_root = args.output_root.resolve()
    evidence = args.evidence.resolve()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(output_root)
    if evidence.exists() or evidence.is_symlink():
        raise FileExistsError(evidence)
    if sha256_file(formal_path) != FORMAL_INPUT_SHA:
        raise RuntimeError("formal input manifest SHA mismatch")
    formal = json.loads(formal_path.read_text(encoding="utf-8"))
    if formal.get("manifest_sha256") != FORMAL_INPUT_CANONICAL_SHA:
        raise RuntimeError("formal input manifest canonical identity mismatch")
    if sha256_file(observations_path) != PROTOCOL_OBSERVATIONS_SHA:
        raise RuntimeError("100K protocol observation table SHA mismatch")
    if sha256_file(release_path) != PROTOCOL_RELEASE_SHA:
        raise RuntimeError("GCP protocol release SHA mismatch")

    subset = materialize_subset(
        scene=SCENE,
        formal_input_manifest_path=formal_path,
        protocol_observations_path=observations_path,
        output_root=output_root,
        file_mode="hardlink",
    )
    if (
        subset.get("status") != "PASS"
        or subset.get("camera_view_count") != 211
        or subset.get("observation_count") != 256
        or subset.get("protocol_observations_file_sha256") != PROTOCOL_OBSERVATIONS_SHA
    ):
        raise RuntimeError("GCP camera subset materialization mismatch")
    roles = Counter(str(row.get("formal_role")) for row in subset["images"])
    if roles != Counter({"train": 187, "test": 24}):
        raise RuntimeError(f"GCP observation-camera role counts changed: {dict(roles)}")

    source_manifest_path = output_root / "EVALUATION_CAMERA_SUBSET_MANIFEST.json"
    source_manifest_sha = sha256_file(source_manifest_path)
    source_manifest_canonical = subset["manifest_sha256"]
    source_manifest_path.unlink()

    images_root = output_root / "images"
    placeholder = output_root / "BLACK_CAMERA_LOADER_PLACEHOLDER.png"
    placeholder.write_bytes(black_png(1414, 1024))
    for path in sorted(images_root.iterdir(), key=lambda item: item.name):
        if path.is_file() or path.is_symlink():
            path.unlink()
    linked: list[str] = []
    for row in subset["images"]:
        target = images_root / str(row["image_name"])
        os.link(placeholder, target)
        linked.append(target.name)
    if len(linked) != 211 or len(set(linked)) != 211:
        raise RuntimeError("placeholder image inventory mismatch")
    if any(sha256_file(images_root / name) != sha256_file(placeholder) for name in linked):
        raise RuntimeError("placeholder image byte identity mismatch")

    sparse_root = output_root / "sparse" / "0"
    sparse = {
        name: identity(sparse_root / name)
        for name in ("cameras.bin", "images.bin", "points3D.bin", "points3D.ply")
    }
    payload: dict[str, Any] = {
        "schema": "m3m_gcp_100k_gcp_evaluation_camera_root_v1",
        "status": "PASS_GCP_EVALUATION_CAMERA_ROOT_NO_RGB_PIXELS",
        "scene": SCENE,
        "protocol_id": "m3m_gcp_native_quarter_geometry_v2",
        "purpose": "post-freeze GCP geometry evaluation only; never training, prior, RGB metric, checkpoint or seed selection",
        "formal_input_manifest": {
            "path": str(formal_path),
            "sha256": FORMAL_INPUT_SHA,
            "canonical_sha256": FORMAL_INPUT_CANONICAL_SHA,
        },
        "protocol_release": {
            "path": str(release_path),
            "sha256": PROTOCOL_RELEASE_SHA,
        },
        "protocol_observations": {
            "path": str(observations_path),
            "sha256": PROTOCOL_OBSERVATIONS_SHA,
            "observation_count": 256,
            "unique_camera_count": 211,
            "formal_role_counts": {"train": 187, "test": 24},
        },
        "source_subset_receipt": {
            "former_path": str(source_manifest_path),
            "file_sha256_before_placeholder_replacement": source_manifest_sha,
            "canonical_sha256": source_manifest_canonical,
            "camera_pose_and_intrinsics_preserved": True,
        },
        "output": {
            "root": str(output_root),
            "camera_view_count": 211,
            "sparse_files": sparse,
            "image_names": sorted(linked),
            "placeholder": identity(placeholder),
            "all_named_loader_images_are_hardlinks_to_placeholder": True,
        },
        "rgb_truth_boundary": {
            "real_rgb_pixels_present": False,
            "placeholder_format": "deterministic black RGB PNG byte stream under frozen COLMAP image names",
            "placeholder_dimensions": [1414, 1024],
            "test_role_means_rgb_loss_holdout_only": True,
            "test_camera_pose_and_external_gcp_annotation_use_after_model_freeze": True,
        },
        "materializer": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    payload["canonical_sha256"] = canonical_sha256(payload)
    manifest_path = output_root / "GCP_EVALUATION_CAMERA_ROOT_MANIFEST.json"
    write_exclusive(manifest_path, payload)
    write_exclusive(evidence, payload)
    if sha256_file(manifest_path) != sha256_file(evidence):
        raise RuntimeError("GCP camera-root manifest/evidence bytes differ")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
