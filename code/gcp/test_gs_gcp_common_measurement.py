#!/usr/bin/env python3
"""CPU-side tests for common measurement contracts and aggregation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from gs_gcp_common_measurement import _load_cfg_args, _percentile, inspect_original_3dgs_representation, prepare_original_3dgs_evaluation_model, validate_rgb_image_set
from gs_gcp_stage0_5 import write_cameras_binary, write_images_binary
from read_write_model import Camera, Image


def test_cfg_args_namespace_parser() -> None:
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "cfg_args"
        path.write_text("Namespace(sh_degree=3, white_background=False, source_path='/x')", encoding="utf-8")
        assert _load_cfg_args(path)["sh_degree"] == 3


def test_evaluation_model_is_read_only_adapter() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        trained = root / "trained"
        ply = trained / "point_cloud" / "iteration_30000" / "point_cloud.ply"
        ply.parent.mkdir(parents=True)
        ply.write_bytes(b"checkpoint")
        (trained / "cfg_args").write_text("Namespace(sh_degree=3, resolution=4)", encoding="utf-8")
        source = root / "source"
        sparse = source / "sparse" / "0"
        sparse.mkdir(parents=True)
        write_cameras_binary(
            {1: Camera(id=1, model="PINHOLE", width=100, height=80, params=np.asarray([50.0, 51.0, 50.0, 40.0]))},
            sparse / "cameras.bin",
        )
        write_images_binary(
            {7: Image(id=7, qvec=np.asarray([1.0, 0.0, 0.0, 0.0]), tvec=np.asarray([1.0, 2.0, 3.0]), camera_id=1, name="image.JPG", xys=np.empty((0, 2)), point3D_ids=np.empty((0,), dtype=np.int64))},
            sparse / "images.bin",
        )
        evaluation = root / "evaluation"
        result = prepare_original_3dgs_evaluation_model(trained, evaluation, source)
        assert result["checkpoint_mutated"] is False
        assert (evaluation / "point_cloud" / "iteration_30000" / "point_cloud.ply").read_bytes() == b"checkpoint"
        assert _load_cfg_args(evaluation / "cfg_args")["source_path"] == str(source.resolve())
        cameras = json.loads((evaluation / "cameras.json").read_text(encoding="utf-8"))
        assert cameras == [{
            "id": 0, "img_name": "image", "width": 100, "height": 80,
            "position": [-1.0, -2.0, -3.0],
            "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "fy": 51.0, "fx": 50.0,
        }]
        assert result["cameras_json_count"] == 1


def test_percentile() -> None:
    assert _percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert abs(_percentile([1.0, 2.0, 3.0, 4.0], 0.9) - 3.7) < 1e-12


def test_failed_render_accounting() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        render = root / "render"
        gt = root / "gt"
        render.mkdir()
        (gt / "gt").mkdir(parents=True)
        (render / "a.png").write_bytes(b"render")
        (gt / "gt" / "a.png").write_bytes(b"gt")
        (gt / "gt" / "b.png").write_bytes(b"other")
        from gs_gcp_common_measurement import sha256_file
        manifest = {
            "images": [
                {"image_name": "a.JPG", "gt_relative_path": "gt/a.png", "gt_png_sha256": sha256_file(gt / "gt" / "a.png")},
                {"image_name": "b.JPG", "gt_relative_path": "gt/b.png", "gt_png_sha256": sha256_file(gt / "gt" / "b.png")},
            ]
        }
        result = validate_rgb_image_set(render, manifest, gt)
        assert not result["passed"]
        assert result["missing_render_names"] == ["b"]


def test_representation_count_and_bytes() -> None:
    with tempfile.TemporaryDirectory() as temp:
        model = Path(temp)
        ply = model / "point_cloud" / "iteration_30000" / "point_cloud.ply"
        ply.parent.mkdir(parents=True)
        ply.write_bytes(
            b"ply\nformat binary_little_endian 1.0\nelement vertex 2\n"
            b"property float x\nproperty float y\nend_header\n" + b"\x00" * 16
        )
        report = inspect_original_3dgs_representation(model, 30000)
        assert report["gaussian_count"] == 2
        assert report["serialized_scalar_count"] == 4
        assert report["deployable_model_bytes"] == ply.stat().st_size


def test_contract_supports_two_render_modes() -> None:
    root = Path(__file__).resolve().parents[2]
    contract = json.loads((root / "configs" / "gs_gcp_common_measurement_suite_v1.json").read_text(encoding="utf-8"))
    assert contract["single_gpu_reference_render"]["primary_comparable_column"] is True
    assert contract["method_native_render"]["role"] == "secondary"
    assert contract["complete_scene_required_for_primary_rgb_mean"] is True


TESTS = [test_cfg_args_namespace_parser, test_evaluation_model_is_read_only_adapter, test_percentile, test_failed_render_accounting, test_representation_count_and_bytes, test_contract_supports_two_render_modes]


def main() -> int:
    for test in TESTS:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS {len(TESTS)}/{len(TESTS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
