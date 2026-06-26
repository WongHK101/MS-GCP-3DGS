from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metric_depth_packet import (  # noqa: E402
    DEFAULT_NORMALIZATION_EPSILON,
    DEFAULT_NUMERICAL_SUPPORT_FLOOR,
    DEFAULT_VARIANCE_CLAMP_TOLERANCE,
    HISTORICAL_INVALID_TENSOR,
    METRIC_PACKET_SCHEMA,
    METRIC_PACKET_TENSOR_NAMES,
    PRIMARY_DEPTH_TENSOR,
    file_sha256,
    recompute_and_compare_packet,
    write_json,
)


def max_abs_rel(actual: Any, expected: Any) -> tuple[float, float]:
    a = np.asarray(actual, dtype=np.float64)
    e = np.asarray(expected, dtype=np.float64)
    diff = np.nan_to_num(a - e, nan=0.0)
    abs_err = float(np.max(np.abs(diff))) if diff.size else 0.0
    denom = np.maximum(np.abs(np.nan_to_num(e, nan=0.0)), 1e-12)
    rel_err = float(np.max(np.abs(diff) / denom)) if diff.size else 0.0
    return abs_err, rel_err


def make_settings(torch: Any, train_repo: Path, return_packet: bool, image_size: int = 9) -> Any:
    from diff_gaussian_rasterization import GaussianRasterizationSettings  # noqa: WPS433
    from utils.graphics_utils import getProjectionMatrix, getWorld2View2  # noqa: WPS433

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
        bg=torch.zeros(3, dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=world_view,
        projmatrix=full_projection,
        sh_degree=0,
        campos=torch.zeros(3, dtype=torch.float32, device="cuda"),
        prefiltered=False,
        debug=False,
        antialiasing=False,
        return_metric_depth_packet=return_packet,
        numerical_support_floor=DEFAULT_NUMERICAL_SUPPORT_FLOOR,
        normalization_epsilon=DEFAULT_NORMALIZATION_EPSILON,
        variance_clamp_tolerance=DEFAULT_VARIANCE_CLAMP_TOLERANCE,
    )


def rasterize_case(torch: Any, train_repo: Path, z_values: list[float], opacities: list[float], return_packet: bool):
    from diff_gaussian_rasterization import GaussianRasterizer  # noqa: WPS433

    settings = make_settings(torch, train_repo, return_packet=return_packet)
    rasterizer = GaussianRasterizer(settings)
    n = len(z_values)
    means3d = torch.tensor([[0.0, 0.0, z] for z in z_values], dtype=torch.float32, device="cuda")
    means2d = torch.zeros((n, 3), dtype=torch.float32, device="cuda", requires_grad=True)
    colors = torch.ones((n, 3), dtype=torch.float32, device="cuda") * 0.5
    opacity = torch.tensor([[value] for value in opacities], dtype=torch.float32, device="cuda")
    scales = torch.ones((n, 3), dtype=torch.float32, device="cuda") * 0.35
    rotations = torch.zeros((n, 4), dtype=torch.float32, device="cuda")
    rotations[:, 0] = 1.0
    return rasterizer(
        means3D=means3d,
        means2D=means2d,
        colors_precomp=colors,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
    )


def extract_peak_packet(packet: Any) -> dict[str, Any]:
    packet_np = packet.detach().cpu().numpy().astype(np.float32)
    alpha = packet_np[0]
    y, x = np.unravel_index(int(np.nanargmax(alpha)), alpha.shape)
    pixel = {name: float(packet_np[i, y, x]) for i, name in enumerate(METRIC_PACKET_TENSOR_NAMES[:-1])}
    pixel["metric_depth_valid_mask"] = bool(packet_np[-1, y, x] > 0.5)
    return {"x": int(x), "y": int(y), "packet_np": packet_np, "pixel": pixel}


def assert_metric_close(name: str, actual: float, expected: float, atol: float = 1e-5, rtol: float = 1e-5) -> dict[str, Any]:
    abs_err = abs(float(actual) - float(expected))
    rel_err = abs_err / max(abs(float(expected)), 1e-12)
    passed = abs_err <= (atol + rtol * abs(float(expected)))
    if not passed:
        raise AssertionError(f"{name}: actual={actual}, expected={expected}, abs={abs_err}, rel={rel_err}")
    return {
        "name": name,
        "actual": float(actual),
        "expected": float(expected),
        "max_abs_error": abs_err,
        "max_rel_error": rel_err,
        "atol": atol,
        "rtol": rtol,
        "passed": True,
    }


def run_cuda_test(train_repo: Path, out_dir: Path) -> dict[str, Any]:
    sys.path.insert(0, str(train_repo))
    import torch  # noqa: WPS433

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    torch.cuda.reset_peak_memory_stats()

    disabled_single = rasterize_case(torch, train_repo, [20.0], [0.25], return_packet=False)
    enabled_single = rasterize_case(torch, train_repo, [20.0], [0.25], return_packet=True)
    if len(disabled_single) != 3 or len(enabled_single) != 4:
        raise AssertionError(f"Unexpected rasterizer output lengths: {len(disabled_single)}, {len(enabled_single)}")
    rgb_disabled, _radii_disabled, old_depth_disabled = disabled_single
    rgb_enabled, _radii_enabled, old_depth_enabled, packet_single = enabled_single
    rgb_bitwise = bool(torch.equal(rgb_disabled, rgb_enabled))
    old_depth_bitwise = bool(torch.equal(old_depth_disabled, old_depth_enabled))
    rgb_abs = float((rgb_disabled - rgb_enabled).abs().max().item())
    old_abs = float((old_depth_disabled - old_depth_enabled).abs().max().item())
    if not rgb_bitwise or not old_depth_bitwise:
        raise AssertionError(f"Legacy output changed when enabling packet: rgb_abs={rgb_abs}, old_depth_abs={old_abs}")

    peak_single = extract_peak_packet(packet_single)
    single_cases = [
        assert_metric_close("single_expected_camera_z", peak_single["pixel"][PRIMARY_DEPTH_TENSOR], 20.0),
        assert_metric_close("single_harmonic_camera_z", peak_single["pixel"]["harmonic_camera_z"], 20.0),
        assert_metric_close("single_variance", peak_single["pixel"]["camera_z_variance"], 0.0, atol=1e-4, rtol=1e-4),
    ]

    _rgb2, _radii2, old_depth_two, packet_two = rasterize_case(torch, train_repo, [10.0, 30.0], [0.4, 0.5], return_packet=True)
    peak_two = extract_peak_packet(packet_two)
    pixel = peak_two["pixel"]
    a = pixel["accumulated_alpha"]
    m1 = pixel["weighted_camera_z_sum"]
    m2 = pixel["weighted_camera_z_second_moment"]
    h = pixel["weighted_inverse_camera_z_sum"]
    expected_z = m1 / a
    expected_inverse_z = h / a
    harmonic_z = a / h
    variance = m2 / a - expected_z * expected_z
    two_cases = [
        assert_metric_close("two_expected_z_from_raw", pixel[PRIMARY_DEPTH_TENSOR], expected_z),
        assert_metric_close("two_expected_inverse_z_from_raw", pixel["alpha_normalized_expected_inverse_camera_z"], expected_inverse_z),
        assert_metric_close("two_harmonic_z_from_raw", pixel["harmonic_camera_z"], harmonic_z),
        assert_metric_close("two_variance_from_raw", pixel["camera_z_variance"], max(0.0, variance), atol=1e-4, rtol=1e-4),
    ]
    old_depth_peak = float(old_depth_two.detach().cpu().numpy().squeeze()[peak_two["y"], peak_two["x"]])
    old_depth_expected = h
    old_case = assert_metric_close("historical_invalid_inverse_depth_equals_H", old_depth_peak, old_depth_expected, atol=1e-5, rtol=1e-5)

    packet_payload = {
        name: peak_two["packet_np"][i].astype(np.float32)
        for i, name in enumerate(METRIC_PACKET_TENSOR_NAMES)
    }
    packet_payload["metric_depth_valid_mask"] = packet_payload["metric_depth_valid_mask"] > 0.5
    recompute = recompute_and_compare_packet(packet_payload, atol=1e-4, rtol=1e-4)
    if not recompute["passed"]:
        raise AssertionError(recompute)

    zero_enabled = rasterize_case(torch, train_repo, [20.0], [0.0], return_packet=True)
    zero_peak = extract_peak_packet(zero_enabled[3])
    zero_alpha = zero_peak["pixel"]["accumulated_alpha"]
    zero_mask = zero_peak["pixel"]["metric_depth_valid_mask"]
    zero_expected = zero_peak["pixel"][PRIMARY_DEPTH_TENSOR]
    zero_passed = (not zero_mask) and zero_alpha <= DEFAULT_NUMERICAL_SUPPORT_FLOOR and math.isnan(zero_expected)
    if not zero_passed:
        raise AssertionError(f"zero alpha invalid policy failed: {zero_peak['pixel']}")

    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "tiny_metric_depth_packet_two_layer.npz"
    np.savez_compressed(
        npz_path,
        **packet_payload,
        **{HISTORICAL_INVALID_TENSOR: old_depth_two.detach().cpu().numpy().squeeze().astype(np.float32)},
    )
    manifest = {
        "schema": "metric_depth_packet_tiny_cuda_test_v1",
        "packet_schema": METRIC_PACKET_SCHEMA,
        "train_repo": str(train_repo),
        "packet_path": str(npz_path),
        "packet_sha256": file_sha256(npz_path),
        "torch": str(torch.__version__),
        "torch_cuda": str(getattr(torch.version, "cuda", "")),
        "cuda_device": torch.cuda.get_device_name(0),
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }
    manifest_path = out_dir / "tiny_metric_depth_packet_cuda_manifest.json"
    write_json(manifest_path, manifest)
    return {
        "schema": "metric_depth_packet_cuda_test_matrix_v1",
        "status": "PASS",
        "packet_schema": METRIC_PACKET_SCHEMA,
        "tests": [
            {
                "test": "eval_disabled_backward_compatibility_proxy",
                "rgb_bitwise_equal": rgb_bitwise,
                "old_depth_bitwise_equal": old_depth_bitwise,
                "rgb_max_abs_error": rgb_abs,
                "old_depth_max_abs_error": old_abs,
            },
            {"test": "single_plane", "peak_pixel": {"x": peak_single["x"], "y": peak_single["y"]}, "cases": single_cases},
            {"test": "two_layer_raw_vs_derived", "peak_pixel": {"x": peak_two["x"], "y": peak_two["y"]}, "cases": two_cases + [old_case]},
            {"test": "derived_tensor_recomputation", **recompute},
            {
                "test": "zero_alpha_invalid",
                "passed": zero_passed,
                "accumulated_alpha": zero_alpha,
                "valid_mask": zero_mask,
                "expected_camera_z_is_nan": math.isnan(zero_expected),
            },
        ],
        "artifacts": {
            "npz": str(npz_path),
            "npz_sha256": file_sha256(npz_path),
            "manifest": str(manifest_path),
        },
        "runtime": {
            "torch": str(torch.__version__),
            "torch_cuda": str(getattr(torch.version, "cuda", "")),
            "device": torch.cuda.get_device_name(0),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny synthetic CUDA test for metric depth packet rasterizer output.")
    parser.add_argument("--train_repo", default=r"E:\Multispectral")
    parser.add_argument("--out_dir", default=r"E:\M3M-GCP-3DGS\outputs\metric_depth_packet_20260626\cuda_tiny")
    args = parser.parse_args()
    result = run_cuda_test(Path(args.train_repo).resolve(), Path(args.out_dir).resolve())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
