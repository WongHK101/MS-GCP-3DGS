#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "materialize_m3m_gcp_100k_method_inputs.py"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module():
    spec = importlib.util.spec_from_file_location("materializer", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


class MethodInputMaterializerTest(unittest.TestCase):
    def test_per_method_semantics(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            formal = root / "formal"
            formal_sparse = formal / "train" / "sparse" / "0"
            formal_images = formal / "train" / "images"
            formal_sparse.mkdir(parents=True); formal_images.mkdir(parents=True)
            (formal_images / "train.jpg").write_bytes(b"rgb")
            for name, data in {"cameras.bin":b"cam","images.bin":b"pose","points3D.ply":b"ply"}.items():
                (formal_sparse/name).write_bytes(data)
            manifest={"manifest_sha256":"synthetic-canonical","images":[
                {"image_name":"train.jpg","role":"train"},
                {"image_name":"test.jpg","role":"test"}]}
            manifest_path=formal/"NATIVE_QUARTER_INPUT_MANIFEST.json"
            manifest_path.write_text(json.dumps(manifest),encoding="utf-8")

            full=root/"full"; city=root/"city"; metro=root/"metro"
            for directory in (full,city,metro): directory.mkdir()
            for name,data in {"cameras.bin":b"cam","images.bin":b"full-images","points3D.bin":b"full-points","points3D.ply":b"ply","frames.bin":b"frames","rigs.bin":b"rigs"}.items():
                (full/name).write_bytes(data)
            for name,data in {"cameras.bin":b"cam","images.bin":b"train-tracks","points3D.bin":b"full-points"}.items():
                (city/name).write_bytes(data)
            for name,data in {"cameras.bin":b"cam","images.bin":b"train-tracks","points3D.bin":b"closed-points"}.items():
                (metro/name).write_bytes(data)
            audit=root/"audit.json"; audit.write_text(json.dumps({"status":"pass","counts":{"images":2510}}))
            city_ev=root/"city.json"; metro_ev=root/"metro.json"
            city_hashes={name:sha(city/name) for name in ("cameras.bin","images.bin","points3D.bin")}
            metro_hashes={name:sha(metro/name) for name in ("cameras.bin","images.bin","points3D.bin")}
            city_ev.write_text(json.dumps({"schema":"m3m_gcp_native_quarter_city_track_compatibility_streaming_v1","status":"PASS","passed":True,"derived_model":{"sha256":city_hashes}}))
            metro_ev.write_text(json.dumps({"schema":"m3m_gcp_colmap_streaming_frozen_train_track_closure_v1","status":"PASS","passed":True,"derived_model":{"sha256":metro_hashes}}))

            module.FORMAL_FILE_SHA=sha(manifest_path); module.FORMAL_CANONICAL_SHA="synthetic-canonical"
            module.TRAIN_COUNT=1; module.TEST_COUNT=1
            module.FORMAL_CAMERA_SHA=sha(formal_sparse/"cameras.bin"); module.FORMAL_IMAGES_SHA=sha(formal_sparse/"images.bin")
            module.INITIAL_PLY_SHA=sha(formal_sparse/"points3D.ply")
            module.FULL_HASHES={name:sha(full/name) for name in ("cameras.bin","images.bin","points3D.bin","points3D.ply","frames.bin","rigs.bin")}
            module.FULL_AUDIT_SHA=sha(audit); module.CITY_HASHES=city_hashes; module.METRO_HASHES=metro_hashes

            outputs={name:root/name for name in ("common","qgs","city-out","cityx-out","metro-out")}
            evidence=root/"result.json"
            argv=[str(SCRIPT),"--formal-scene-root",str(formal),"--full-model-root",str(full),
                "--full-package-audit",str(audit),"--city-track-model",str(city),
                "--city-track-evidence",str(city_ev),"--expected-city-track-evidence-sha256",sha(city_ev),
                "--metro-track-model",str(metro),"--metro-track-evidence",str(metro_ev),
                "--expected-metro-track-evidence-sha256",sha(metro_ev),"--common-root",str(outputs["common"]),
                "--qgs-root",str(outputs["qgs"]),"--citygaussian-root",str(outputs["city-out"]),
                "--citygs-root",str(outputs["cityx-out"]),"--metrogs-root",str(outputs["metro-out"]),
                "--evidence-output",str(evidence)]
            with mock.patch.object(sys,"argv",argv), mock.patch.object(
                module, "link_dir", lambda target, link: shutil.copytree(target, link)
            ):
                module.main()
            report=json.loads(evidence.read_text())
            self.assertEqual(report["status"],"PASS_PER_METHOD_INPUT_PREPARATION_NO_TRAINING_NO_PRIOR")
            self.assertTrue(report["access_boundary"]["all_images_participated_in_sfm"])
            self.assertEqual((outputs["city-out"]/"sparse/0/points3D.bin").read_bytes(),b"full-points")
            self.assertEqual((outputs["metro-out"]/"sparse/0/points3D.bin").read_bytes(),b"closed-points")
            self.assertTrue((outputs["qgs"]/"images_undistorted_1.0"/"train.jpg").is_file())


if __name__ == "__main__":
    unittest.main()
