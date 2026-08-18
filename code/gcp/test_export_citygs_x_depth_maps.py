#!/usr/bin/env python3
"""CPU/static checks for the CityGS-X packet exporter."""

from __future__ import annotations

import argparse
from pathlib import Path

from export_citygs_x_depth_maps import build_official_defaults, build_parser


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--citygs_x_repo", type=Path)
    parser.add_argument("--pytorch3d_compat", type=Path)
    args = parser.parse_args()

    export_parser = build_parser()
    assert export_parser.get_default("image_domain") == "colmap_4_0_4_image_undistorter_pinhole_max_1414"
    assert export_parser.get_default("raw_camera_z_to_protocol_scale") == 1.0
    assert export_parser.get_default("camera_z_unit_contract") == "frozen_colmap_model_camera_z_units"

    if args.citygs_x_repo is not None:
        if args.pytorch3d_compat is None:
            raise ValueError("--pytorch3d_compat is required with --citygs_x_repo")
        defaults, groups = build_official_defaults(
            args.citygs_x_repo.resolve(),
            args.pytorch3d_compat.resolve(),
        )
        assert defaults.resolution == 1
        assert defaults.appearance_dim == 0
        assert defaults.bsz == 1
        assert set(groups) == {"model", "pipeline"}

    print("citygs_x_export_cpu_test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
