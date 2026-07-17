#!/usr/bin/env python3
"""GS-GCP common training-resolution contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CONTRACT_SCHEMA = "gs_gcp_training_resolution_contract_v1"
RULE_ID = "graphdeco_rminus1_1600_width_cap_v1"


def graphdeco_rminus1_dimensions(
    original_width: int,
    original_height: int,
    *,
    max_width: int = 1600,
) -> tuple[int, int]:
    """Reproduce the original 3DGS ``--resolution -1`` dimension rule."""
    if not isinstance(original_width, int) or not isinstance(original_height, int):
        raise TypeError("image dimensions must be integers")
    if original_width <= 0 or original_height <= 0:
        raise ValueError("image dimensions must be positive")
    if not isinstance(max_width, int) or max_width <= 0:
        raise ValueError("max_width must be a positive integer")
    if original_width <= max_width:
        return original_width, original_height
    scale = float(original_width) / float(max_width)
    return int(original_width / scale), int(original_height / scale)


def resolution_record(original_width: int, original_height: int, *, max_width: int = 1600) -> dict[str, Any]:
    loaded_width, loaded_height = graphdeco_rminus1_dimensions(
        original_width,
        original_height,
        max_width=max_width,
    )
    return {
        "rule_id": RULE_ID,
        "original_width": original_width,
        "original_height": original_height,
        "loaded_width": loaded_width,
        "loaded_height": loaded_height,
        "scale_x": loaded_width / original_width,
        "scale_y": loaded_height / original_height,
        "upscaled": loaded_width > original_width or loaded_height > original_height,
    }


def validate_contract(contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = {
        "schema": CONTRACT_SCHEMA,
        "rule_id": RULE_ID,
        "reference_method_argument": -1,
        "max_width": 1600,
        "small_image_policy": "identity_no_upscale",
        "large_image_rounding": "python_int_truncation_after_float64_division",
        "dimension_source": "decoded_benchmark_undistorted_image_matrix",
        "aspect_policy": "preserve_by_common_width_derived_scale",
    }
    for key, expected in required.items():
        if contract.get(key) != expected:
            errors.append(f"{key} must equal {expected!r}")
    if contract.get("crop_policy") != "none" or contract.get("pad_policy") != "none":
        errors.append("crop and pad policies must both be none")
    if contract.get("pixel_convention") != "zero_based_pixel_centers":
        errors.append("pixel convention mismatch")
    if contract.get("per_view_loaded_dimension_manifest_required") is not True:
        errors.append("per-view loaded dimension manifest must be required")
    if contract.get("loaded_tensor_hash_probe_required") is not True:
        errors.append("loaded tensor hash probe must be required")
    if contract.get("method_cli_resolution_alias_is_sufficient_evidence") is not False:
        errors.append("method CLI aliases must not be accepted as resolution evidence")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "configs" / "gs_gcp_training_resolution_v1.json",
    )
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    errors = validate_contract(contract)
    result: dict[str, Any] = {
        "schema": "gs_gcp_training_resolution_validation_v1",
        "status": "pass" if not errors else "fail",
        "errors": errors,
    }
    if (args.width is None) != (args.height is None):
        raise SystemExit("--width and --height must be provided together")
    if args.width is not None and args.height is not None:
        result["record"] = resolution_record(args.width, args.height, max_width=int(contract["max_width"]))
    print(json.dumps(result, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
