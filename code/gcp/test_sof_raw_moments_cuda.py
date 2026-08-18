#!/usr/bin/env python3
"""Target-GPU conformance for SOF's native hierarchical compositor."""

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


def raster_settings(torch: Any, eval_repo: Path, image_size: int = 9) -> Any:
    from diff_gaussian_rasterization import (
        DebugVisualization,
        ExtendedSettings,
        GaussianRasterizationSettings,
        GlobalSortOrder,
        SortMode,
    )
    from utils.graphics_utils import getProjectionMatrix, getWorld2View2

    fov = math.radians(60.0)
    world_view = torch.tensor(
        getWorld2View2(np.eye(3), np.zeros(3)),
        dtype=torch.float32,
        device="cuda",
    ).transpose(0, 1)
    projection = getProjectionMatrix(0.01, 100.0, fov, fov).transpose(0, 1).to(
        device="cuda", dtype=torch.float32
    )
    full_projection = world_view.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0)
    settings = ExtendedSettings.from_json(str(eval_repo / "configs" / "hierarchical.json"))
    settings.detach_alpha = False
    settings.far_plane = 100.0
    if settings.sort_settings.sort_mode != SortMode.HIER:
        raise AssertionError(f"unexpected SOF sort mode: {settings.sort_settings.sort_mode}")
    if settings.sort_settings.sort_order != GlobalSortOrder.PTD_MAX:
        raise AssertionError(f"unexpected SOF sort order: {settings.sort_settings.sort_order}")
    return GaussianRasterizationSettings(
        image_height=image_size,
        image_width=image_size,
        tanfovx=math.tan(fov * 0.5),
        tanfovy=math.tan(fov * 0.5),
        bg=torch.zeros(3, dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=world_view,
        projmatrix=full_projection,
        inv_viewprojmatrix=full_projection.inverse(),
        sh_degree=0,
        campos=torch.zeros(3, dtype=torch.float32, device="cuda"),
        prefiltered=False,
        settings=settings,
        debug_data=DebugVisualization(),
        debug=False,
    )


def rasterize_raw(
    torch: Any,
    eval_repo: Path,
    z_values: list[float],
    opacity_values: list[float],
) -> Any:
    from diff_gaussian_rasterization import GaussianRasterizer

    count = len(z_values)
    means3d = torch.tensor([[0.0, 0.0, z] for z in z_values], dtype=torch.float32, device="cuda")
    means2d = torch.zeros((count, 3), dtype=torch.float32, device="cuda")
    opacities = torch.tensor([[value] for value in opacity_values], dtype=torch.float32, device="cuda")
    scales = torch.full((count, 3), 0.35, dtype=torch.float32, device="cuda")
    rotations = torch.zeros((count, 4), dtype=torch.float32, device="cuda")
    rotations[:, 0] = 1.0
    z = torch.tensor(z_values, dtype=torch.float32, device="cuda")
    ones = torch.ones_like(z)
    zeros = torch.zeros_like(z)
    primary = torch.stack((ones, z, z.square()), dim=1)
    harmonic = torch.stack((z.reciprocal(), zeros, zeros), dim=1)
    rasterizer = GaussianRasterizer(raster_settings=raster_settings(torch, eval_repo))
    common = {
        "means3D": means3d,
        "means2D": means2d,
        "opacities": opacities,
        "scales": scales,
        "rotations": rotations,
    }
    primary_map, _ = rasterizer(shs=None, colors_precomp=primary, **common)
    harmonic_map, _ = rasterizer(shs=None, colors_precomp=harmonic, **common)
    return torch.cat((primary_map[0:3], harmonic_map[0:1]), dim=0)


def raw_peak(raw_map: Any) -> dict[str, float | int]:
    value = raw_map.detach().cpu().numpy().astype(np.float32)
    if value.shape[0] != 4:
        raise AssertionError(f"SOF raw packet must have four planes, got {value.shape}")
    y, x = np.unravel_index(int(np.argmax(value[0])), value[0].shape)
    return {
        "x": int(x),
        "y": int(y),
        "A": float(value[0, y, x]),
        "M1": float(value[1, y, x]),
        "M2": float(value[2, y, x]),
        "H": float(value[3, y, x]),
    }


def assert_close(
    name: str,
    actual: float,
    expected: float,
    *,
    atol: float = 3e-5,
    rtol: float = 3e-5,
) -> dict[str, Any]:
    absolute = abs(actual - expected)
    allowed = atol + rtol * abs(expected)
    if absolute > allowed:
        raise AssertionError(
            f"{name}: actual={actual}, expected={expected}, abs={absolute}, allowed={allowed}"
        )
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

    def isolated_checks(label: str, raw: dict[str, float | int], z: float, opacity: float) -> list[dict[str, Any]]:
        alpha = float(raw["A"])
        return [
            assert_close(
                f"{label}_A_input_response",
                alpha,
                opacity,
                rtol=5e-4,
            ),
            assert_close(f"{label}_M1_identity", float(raw["M1"]), alpha * z),
            assert_close(f"{label}_M2_identity", float(raw["M2"]), alpha * z * z),
            assert_close(f"{label}_H_identity", float(raw["H"]), alpha / z),
        ]

    single = raw_peak(rasterize_raw(torch, eval_repo, [20.0], [0.25]))
    single_checks = [
        *isolated_checks("single", single, 20.0, 0.25),
    ]

    front = raw_peak(rasterize_raw(torch, eval_repo, [10.0], [0.4]))
    back = raw_peak(rasterize_raw(torch, eval_repo, [30.0], [0.5]))
    front_alpha = float(front["A"])
    back_alpha = float(back["A"])
    front_weight = front_alpha
    back_weight = (1.0 - front_alpha) * back_alpha
    expected_a = front_weight + back_weight
    expected_m1 = front_weight * 10.0 + back_weight * 30.0
    expected_m2 = front_weight * 100.0 + back_weight * 900.0
    expected_h = front_weight / 10.0 + back_weight / 30.0
    two = raw_peak(rasterize_raw(torch, eval_repo, [10.0, 30.0], [0.4, 0.5]))
    two_checks = [
        *isolated_checks("front", front, 10.0, 0.4),
        *isolated_checks("back", back, 30.0, 0.5),
        assert_close("two_A_compositing", float(two["A"]), expected_a),
        assert_close("two_M1_compositing", float(two["M1"]), expected_m1),
        assert_close("two_M2_compositing", float(two["M2"]), expected_m2),
        assert_close("two_H_compositing", float(two["H"]), expected_h),
        assert_close(
            "two_expected_z",
            float(two["M1"]) / float(two["A"]),
            expected_m1 / expected_a,
        ),
        assert_close(
            "two_harmonic_z",
            float(two["A"]) / float(two["H"]),
            expected_a / expected_h,
        ),
    ]
    variance = float(two["M2"]) / float(two["A"]) - (
        float(two["M1"]) / float(two["A"])
    ) ** 2
    expected_variance = expected_m2 / expected_a - (expected_m1 / expected_a) ** 2
    variance_check = assert_close("two_variance", variance, expected_variance, atol=3e-4)

    return {
        "schema": "m3m_gcp_native_quarter_sof_raw_moment_cuda_conformance_v1",
        "protocol_id": "m3m_gcp_native_quarter_geometry_v2",
        "method_id": "sof",
        "status": "PASS",
        "passed": True,
        "training_started": False,
        "native_rasterizer_modified": False,
        "native_sort_mode": "HIER",
        "native_sort_order": "PTD_MAX",
        "adapter_strategy": "two auxiliary native three-channel compositor passes with zero background",
        "primary_common_planes": ["A", "M1"],
        "single_layer": {"raw": single, "checks": single_checks},
        "two_layer": {
            "isolated_front_raw": front,
            "isolated_back_raw": back,
            "expected_from_isolated_native_alpha": {
                "A": expected_a,
                "M1": expected_m1,
                "M2": expected_m2,
                "H": expected_h,
            },
            "raw": two,
            "checks": two_checks + [variance_check],
        },
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
