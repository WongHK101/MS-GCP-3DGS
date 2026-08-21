#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "filter_colmap_model_to_frozen_train_streaming.py"
U64 = struct.Struct("<Q")
IMAGE_FIXED = struct.Struct("<i7di")
POINT_FIXED = struct.Struct("<Q3d3BdQ")
TRACK = struct.Struct("<ii")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_record(image_id: int, name: str, points: list[tuple[float, float, int]]) -> bytes:
    payload = bytearray(IMAGE_FIXED.pack(image_id, 1, 0, 0, 0, 0, 0, 0, 1))
    payload.extend(name.encode() + b"\0")
    payload.extend(U64.pack(len(points)))
    for x, y, point_id in points:
        payload.extend(struct.pack("<ddq", x, y, point_id))
    return bytes(payload)


def point_record(point_id: int, tracks: list[tuple[int, int]]) -> bytes:
    payload = bytearray(POINT_FIXED.pack(point_id, 1, 2, 3, 4, 5, 6, 0.1, len(tracks)))
    for image_id, point2d_idx in tracks:
        payload.extend(TRACK.pack(image_id, point2d_idx))
    return bytes(payload)


class StreamingFilterTest(unittest.TestCase):
    def test_exact_train_track_closure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            (source / "cameras.bin").write_bytes(b"camera-bytes")
            (source / "images.bin").write_bytes(
                U64.pack(3)
                + image_record(1, "train.jpg", [(30, 40, -1), (10, 20, 10)])
                + image_record(2, "test-a.jpg", [(11, 21, 10), (31, 41, 11)])
                + image_record(3, "test-b.jpg", [(12, 22, 11)])
            )
            (source / "points3D.bin").write_bytes(
                U64.pack(2)
                + point_record(10, [(1, 1), (2, 0)])
                + point_record(11, [(2, 1), (3, 0)])
            )
            formal = root / "formal.json"
            formal.write_text(
                json.dumps(
                    {
                        "scene": "synthetic",
                        "images": [
                            {"image_name": "train.jpg", "role": "train"},
                            {"image_name": "test-a.jpg", "role": "test"},
                            {"image_name": "test-b.jpg", "role": "test"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "output"
            evidence = root / "evidence.json"
            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(SCRIPT),
                    "--source_model",
                    str(source),
                    "--formal_manifest",
                    str(formal),
                    "--output_model",
                    str(output),
                    "--output_manifest",
                    str(evidence),
                    "--expected_cameras_sha256",
                    sha(source / "cameras.bin"),
                    "--expected_images_sha256",
                    sha(source / "images.bin"),
                    "--expected_points3d_sha256",
                    sha(source / "points3D.bin"),
                    "--expected_train_count",
                    "1",
                    "--expected_test_count",
                    "2",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(evidence.read_text(encoding="utf-8"))
            self.assertTrue(report["passed"])
            self.assertEqual(report["derived_model"]["image_count"], 1)
            self.assertEqual(report["derived_model"]["point_count"], 1)
            self.assertEqual(report["derived_model"]["track_element_count"], 1)
            self.assertEqual(report["track_closure"]["removed_test_image_records"], 2)
            self.assertEqual(report["track_closure"]["retained_untriangulated_point2d_count"], 1)
            self.assertEqual((output / "cameras.bin").read_bytes(), b"camera-bytes")
            self.assertEqual(U64.unpack((output / "images.bin").read_bytes()[:8])[0], 1)
            self.assertEqual(U64.unpack((output / "points3D.bin").read_bytes()[:8])[0], 1)
            point_bytes = (output / "points3D.bin").read_bytes()
            self.assertEqual(TRACK.unpack_from(point_bytes, 8 + POINT_FIXED.size), (1, 1))


if __name__ == "__main__":
    unittest.main()
