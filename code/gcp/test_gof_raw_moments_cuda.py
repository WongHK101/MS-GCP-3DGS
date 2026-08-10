#!/usr/bin/env python3
"""Target-GPU conformance test for the evaluation-only GOF A/M1/M2/H output."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def settings(torch: Any, image_size: int = 9) -> Any:
    from diff_gaussian_rasterization import GaussianRasterizationSettings
    from utils.graphics_utils import getProjectionMatrix, getWorld2View2

    fov = math.radians(60.0)
    world_view = torch.tensor(
        getWorld2View2(np.eye(3), np.zeros(3)),
        dtype=torch.float32,
        device="cuda",
    ).transpose(0, 1)
    projection = getProjectionMatrix(0.01, 100.0, fov, fov).transpose(0, 1).to(device="cuda", dtype=torch.float32)
    full_projection = world_view.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0)
    return GaussianRasterizationSettings(
        image_height=image_size,
        image_width=image_size,
        tanfovx=math.tan(fov * 0.5),
        tanfovy=math.tan(fov * 0.5),
        kernel_size=0.0,
        subpixel_offset=torch.zeros((image_size, image_size, 2), dtype=torch.float32, device="cuda"),
        bg=torch.zeros(3, dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=world_view,
        projmatrix=full_projection,
        sh_degree=0,
        campos=torch.zeros(3, dtype=torch.float32, device="cuda"),
        prefiltered=False,
        debug=False,
    )


def rasterize(torch: Any, z_values: list[float], opacity_values: list[float]) -> Any:
    from diff_gaussian_rasterization import GaussianRasterizer

    count = len(z_values)
    means3d = torch.tensor([[0.0, 0.0, z] for z in z_values], dtype=torch.float32, device="cuda")
    means2d = torch.zeros((count, 3), dtype=torch.float32, device="cuda")
    colors = torch.full((count, 3), 0.5, dtype=torch.float32, device="cuda")
    opacities = torch.tensor([[value] for value in opacity_values], dtype=torch.float32, device="cuda")
    scales = torch.full((count, 3), 3.0, dtype=torch.float32, device="cuda")
    rotations = torch.zeros((count, 4), dtype=torch.float32, device="cuda")
    rotations[:, 0] = 1.0
    rasterizer = GaussianRasterizer(raster_settings=settings(torch))
    return rasterizer(
        means3D=means3d,
        means2D=means2d,
        colors_precomp=colors,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
    )


def raw_peak(rendered_image: Any) -> dict[str, float | int]:
    value = rendered_image.detach().cpu().numpy().astype(np.float32)
    if value.shape[0] != 12:
        raise AssertionError(f"patched GOF rendered image must have 12 planes, got {value.shape}")
    y, x = np.unravel_index(int(np.argmax(value[7])), value[7].shape)
    return {
        "x": int(x),
        "y": int(y),
        "A": float(value[7, y, x]),
        "M1": float(value[9, y, x]),
        "M2": float(value[10, y, x]),
        "H": float(value[11, y, x]),
        "native_median_or_max_depth_diagnostic": float(value[6, y, x]),
    }


def assert_close(name: str, actual: float, expected: float, *, atol: float = 5e-5, rtol: float = 2e-5) -> dict[str, Any]:
    absolute = abs(actual - expected)
    allowed = atol + rtol * abs(expected)
    if absolute > allowed:
        raise AssertionError(f"{name}: actual={actual}, expected={expected}, abs={absolute}, allowed={allowed}")
    return {
        "name": name,
        "actual": actual,
        "expected": expected,
        "absolute_error": absolute,
        "allowed_error": allowed,
        "passed": True,
    }


def run(eval_repo: Path) -> dict[str, Any]:
    sys.path.insert(0, str(eval_repo))
    import torch
    import diff_gaussian_rasterization

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    extension = next(Path(diff_gaussian_rasterization.__file__).resolve().parent.glob("_C*.so"))

    single_outputs = rasterize(torch, [20.0], [0.25])
    if len(single_outputs) != 2:
        raise AssertionError(f"unexpected GOF rasterizer output count: {len(single_outputs)}")
    single = raw_peak(single_outputs[0])
    single_checks = [
        assert_close("single_A", float(single["A"]), 0.25),
        assert_close("single_M1", float(single["M1"]), 5.0),
        assert_close("single_M2", float(single["M2"]), 100.0),
        assert_close("single_H", float(single["H"]), 0.0125),
        assert_close("single_expected_z", float(single["M1"]) / float(single["A"]), 20.0),
        assert_close("single_harmonic_z", float(single["A"]) / float(single["H"]), 20.0),
    ]

    two_outputs = rasterize(torch, [10.0, 30.0], [0.4, 0.5])
    two = raw_peak(two_outputs[0])
    two_checks = [
        assert_close("two_A", float(two["A"]), 0.7),
        assert_close("two_M1", float(two["M1"]), 13.0),
        assert_close("two_M2", float(two["M2"]), 310.0),
        assert_close("two_H", float(two["H"]), 0.05),
        assert_close("two_expected_z", float(two["M1"]) / float(two["A"]), 13.0 / 0.7),
        assert_close("two_harmonic_z", float(two["A"]) / float(two["H"]), 14.0),
    ]
    variance = float(two["M2"]) / float(two["A"]) - (float(two["M1"]) / float(two["A"])) ** 2
    variance_check = assert_close("two_variance", variance, 4800.0 / 49.0, atol=3e-4, rtol=2e-5)

    return {
        "schema": "m3m_gcp_native_quarter_gof_raw_moment_cuda_conformance_v1",
        "protocol_id": "m3m_gcp_native_quarter_geometry_v2",
        "method_id": "gof",
        "status": "PASS",
        "passed": True,
        "training_started": False,
        "common_primary_planes": ["A", "M1"],
        "diagnostic_evaluation_only_planes": ["M2", "H"],
        "native_depth_channel_used_as_common_primary": False,
        "physical_surface_claim": False,
        "single_layer": {"raw": single, "checks": single_checks},
        "two_layer": {"raw": two, "checks": two_checks + [variance_check]},
        "runtime": {
            "python": sys.version,
            "torch": str(torch.__version__),
            "torch_cuda": str(torch.version.cuda),
            "gpu": torch.cuda.get_device_name(0),
            "package": str(Path(diff_gaussian_rasterization.__file__).resolve()),
            "extension": str(extension),
            "extension_sha256": sha256(extension),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval_repo", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.eval_repo.resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
