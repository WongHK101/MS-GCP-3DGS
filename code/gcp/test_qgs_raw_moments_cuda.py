#!/usr/bin/env python3
"""Target-GPU conformance test for QGS evaluation-only A/M1/M2/H."""

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
    from diff_quadratic_rasterization import GaussianRasterizationSettings
    from utils.graphics_utils import getProjectionMatrix, getWorld2View2

    fov = math.radians(60.0)
    focal = image_size / (2.0 * math.tan(fov * 0.5))
    world_view = torch.tensor(
        getWorld2View2(np.eye(3), np.zeros(3)),
        dtype=torch.float32,
        device="cuda",
    ).transpose(0, 1)
    projection = getProjectionMatrix(0.2, 100.0, fov, fov).transpose(0, 1).to(device="cuda", dtype=torch.float32)
    full_projection = world_view.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0)
    cam_intr = torch.tensor(
        [focal, focal, image_size / 2.0, image_size / 2.0],
        dtype=torch.float32,
    )
    return GaussianRasterizationSettings(
        image_height=image_size,
        image_width=image_size,
        tanfovx=math.tan(fov * 0.5),
        tanfovy=math.tan(fov * 0.5),
        kernel_size=0.0,
        subpixel_offset=torch.zeros((image_size, image_size, 2), dtype=torch.float32, device="cuda"),
        bg=torch.zeros(3, dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        sigma=3.0,
        viewmatrix=world_view,
        projmatrix=full_projection,
        sh_degree=0,
        campos=torch.zeros(3, dtype=torch.float32, device="cuda"),
        prefiltered=False,
        debug=False,
        cam_intr=cam_intr,
        stop_z_gradient=False,
        reciprocal_z=False,
        return_depth=True,
        return_normal=True,
    )


def rasterize(torch: Any, z_values: list[float], opacity_values: list[float]) -> Any:
    from diff_quadratic_rasterization import GaussianRasterizer

    count = len(z_values)
    means3d = torch.tensor([[0.0, 0.0, z] for z in z_values], dtype=torch.float32, device="cuda")
    means2d = torch.zeros((count, 3), dtype=torch.float32, device="cuda")
    colors = torch.full((count, 3), 0.5, dtype=torch.float32, device="cuda")
    opacities = torch.tensor([[value] for value in opacity_values], dtype=torch.float32, device="cuda")
    scales = torch.tensor([[3.0, 3.0, 0.1] for _ in z_values], dtype=torch.float32, device="cuda")
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


def raw_peak(outputs: Any) -> dict[str, float | int]:
    rendered = outputs[0].detach().cpu().numpy().astype(np.float32)
    if rendered.shape[0] != 13:
        raise AssertionError(f"unexpected patched QGS output shape: {rendered.shape}")
    alpha = rendered[7]
    y, x = np.unravel_index(int(np.argmax(alpha)), alpha.shape)
    return {
        "x": int(x),
        "y": int(y),
        "A": float(rendered[7, y, x]),
        "M1": float(rendered[6, y, x]),
        "M2": float(rendered[8, y, x]),
        "H": float(rendered[9, y, x]),
    }


def assert_close(name: str, actual: float, expected: float, *, atol: float = 5e-5, rtol: float = 2e-5) -> dict[str, Any]:
    absolute = abs(actual - expected)
    allowed = atol + rtol * abs(expected)
    if absolute > allowed:
        raise AssertionError(f"{name}: actual={actual}, expected={expected}, abs={absolute}, allowed={allowed}")
    return {"name": name, "actual": actual, "expected": expected, "absolute_error": absolute, "allowed_error": allowed, "passed": True}


def run(eval_repo: Path) -> dict[str, Any]:
    sys.path.insert(0, str(eval_repo))
    import torch
    import diff_quadratic_rasterization

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    extension = next(Path(diff_quadratic_rasterization.__file__).resolve().parent.glob("_C*.so"))

    single = raw_peak(rasterize(torch, [20.0], [0.25]))
    if float(single["A"]) <= 0.0:
        raise AssertionError(f"single QGS primitive did not contribute: {single}")
    single_checks = [
        assert_close("single_expected_z", float(single["M1"]) / float(single["A"]), 20.0),
        assert_close("single_second_moment", float(single["M2"]) / float(single["A"]), 400.0),
        assert_close("single_inverse_z", float(single["H"]) / float(single["A"]), 0.05),
        assert_close("single_harmonic_z", float(single["A"]) / float(single["H"]), 20.0),
    ]

    z1, z2 = 10.0, 30.0
    two = raw_peak(rasterize(torch, [z1, z2], [0.25, 0.35]))
    if float(two["A"]) <= 0.0:
        raise AssertionError(f"two QGS primitives did not contribute: {two}")
    expected_m2 = (z1 + z2) * float(two["M1"]) - z1 * z2 * float(two["A"])
    expected_h = ((z1 + z2) * float(two["A"]) - float(two["M1"])) / (z1 * z2)
    expected_z = float(two["M1"]) / float(two["A"])
    variance = float(two["M2"]) / float(two["A"]) - expected_z**2
    if not (z1 < expected_z < z2):
        raise AssertionError(f"two-layer expected z does not contain both centers: {two}")
    if variance <= 1.0:
        raise AssertionError(f"two-layer variance is too small to prove both centers contributed: {variance}")
    two_checks = [
        assert_close("two_second_moment_identity", float(two["M2"]), expected_m2, atol=3e-4),
        assert_close("two_inverse_moment_identity", float(two["H"]), expected_h, atol=3e-5),
    ]

    return {
        "schema": "m3m_gcp_native_quarter_qgs_raw_moment_cuda_conformance_v1",
        "protocol_id": "m3m_gcp_native_quarter_geometry_v2",
        "method_id": "qgs",
        "status": "PASS",
        "passed": True,
        "method_training_started": False,
        "pixel_resorting_path": True,
        "common_primary_planes": ["A", "M1"],
        "diagnostic_evaluation_only_planes": ["M2", "H"],
        "native_quadric_intersection_depth_used_as_common_primary": False,
        "single_layer": {"raw": single, "checks": single_checks},
        "two_layer": {"raw": two, "checks": two_checks, "variance": variance},
        "runtime": {
            "python": sys.version,
            "torch": str(torch.__version__),
            "torch_cuda": str(torch.version.cuda),
            "gpu": torch.cuda.get_device_name(0),
            "package": str(Path(diff_quadratic_rasterization.__file__).resolve()),
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
