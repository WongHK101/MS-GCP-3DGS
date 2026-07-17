#!/usr/bin/env python3
"""Tests for the GS-GCP common resolution contract."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from gs_gcp_resolution import graphdeco_rminus1_dimensions, resolution_record, validate_contract


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = json.loads((ROOT / "configs" / "gs_gcp_training_resolution_v1.json").read_text(encoding="utf-8"))


def test_contract_passes() -> None:
    assert validate_contract(copy.deepcopy(CONTRACT)) == []


def test_real_3k_case() -> None:
    assert graphdeco_rminus1_dimensions(5654, 4098) == (1600, 1159)


def test_small_image_is_not_upscaled() -> None:
    assert graphdeco_rminus1_dimensions(1599, 913) == (1599, 913)
    assert resolution_record(1599, 913)["upscaled"] is False


def test_dimension_half_is_truncated_not_rounded() -> None:
    assert 2001 / (3200 / 1600) == 1000.5
    assert graphdeco_rminus1_dimensions(3200, 2001) == (1600, 1000)


def test_width_and_height_use_same_scale() -> None:
    width, height = graphdeco_rminus1_dimensions(3201, 2001)
    assert width == 1600
    assert height == int(2001 / (3201 / 1600))


def test_rejects_r8_or_cli_alias_evidence() -> None:
    bad = copy.deepcopy(CONTRACT)
    bad["reference_method_argument"] = 8
    bad["method_cli_resolution_alias_is_sufficient_evidence"] = True
    errors = validate_contract(bad)
    assert any("reference_method_argument" in item for item in errors)
    assert any("CLI aliases" in item for item in errors)


def main() -> int:
    tests = [
        test_contract_passes,
        test_real_3k_case,
        test_small_image_is_not_upscaled,
        test_dimension_half_is_truncated_not_rounded,
        test_width_and_height_use_same_scale,
        test_rejects_r8_or_cli_alias_evidence,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS {len(tests)}/{len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
