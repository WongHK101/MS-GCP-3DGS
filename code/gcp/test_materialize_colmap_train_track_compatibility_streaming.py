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
SCRIPT = HERE / "materialize_colmap_train_track_compatibility_streaming.py"
U64 = struct.Struct("<Q")
FIXED = struct.Struct("<i7di")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image(image_id: int, name: str, points: list[tuple[float, float, int]]) -> bytes:
    data = bytearray(FIXED.pack(image_id, 1, 0, 0, 0, 1, 2, 3, 1))
    data.extend(name.encode() + b"\0")
    data.extend(U64.pack(len(points)))
    for x, y, point_id in points:
        data.extend(struct.pack("<ddq", x, y, point_id))
    return bytes(data)


class CompatibilityStreamingTest(unittest.TestCase):
    def test_selects_train_records_and_preserves_full_points(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            formal = root / "formal"
            source.mkdir(); formal.mkdir()
            (source / "cameras.bin").write_bytes(b"camera")
            (formal / "cameras.bin").write_bytes(b"camera")
            (source / "images.bin").write_bytes(
                U64.pack(2)
                + image(1, "train.jpg", [(1, 2, -1), (3, 4, 9)])
                + image(2, "test.jpg", [(5, 6, 9)])
            )
            (formal / "images.bin").write_bytes(U64.pack(1) + image(1, "train.jpg", []))
            (source / "points3D.bin").write_bytes(b"full-points-with-test-tracks")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"scene":"s","images":[
                {"image_name":"train.jpg","role":"train"},
                {"image_name":"test.jpg","role":"test"}]}), encoding="utf-8")
            output = root / "output"; evidence = root / "evidence.json"
            cmd = [sys.executable,"-B",str(SCRIPT),
                "--source-model",str(source),"--formal-train-model",str(formal),
                "--formal-manifest",str(manifest),"--output-model",str(output),
                "--output-manifest",str(evidence),
                "--expected-source-cameras-sha256",sha(source/"cameras.bin"),
                "--expected-source-images-sha256",sha(source/"images.bin"),
                "--expected-source-points3d-sha256",sha(source/"points3D.bin"),
                "--expected-formal-cameras-sha256",sha(formal/"cameras.bin"),
                "--expected-formal-images-sha256",sha(formal/"images.bin"),
                "--expected-train-count","1","--expected-test-count","1"]
            result=subprocess.run(cmd,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            report=json.loads(evidence.read_text())
            self.assertTrue(report["passed"])
            self.assertEqual(report["derived_model"]["image_count"],1)
            self.assertEqual(report["derived_model"]["keypoint_count"],2)
            self.assertEqual((output/"points3D.bin").read_bytes(),(source/"points3D.bin").read_bytes())
            self.assertEqual(U64.unpack((output/"images.bin").read_bytes()[:8])[0],1)


if __name__ == "__main__":
    unittest.main()
