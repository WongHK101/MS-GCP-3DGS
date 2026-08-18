#!/usr/bin/env python3
"""CPU checks for the CityGaussianV2 frozen-camera packet exporter."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np

from export_citygaussian_v2_depth_maps import (
    build_parser,
    read_allowlist,
    resolve_sparse_model,
    save_packet,
)


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        sparse = root / "scene" / "sparse" / "0"
        sparse.mkdir(parents=True)
        for name in ("cameras.bin", "images.bin", "points3D.bin"):
            (sparse / name).write_bytes(name.encode("ascii"))
        assert resolve_sparse_model(root / "scene") == sparse.resolve()

        allowlist = root / "allowlist.csv"
        with allowlist.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["image_name", "status"])
            writer.writeheader()
            writer.writerow({"image_name": "folder/a.JPG", "status": "active"})
            writer.writerow({"image_name": "b.JPG", "status": "inactive"})
        assert read_allowlist(
            allowlist,
            image_name_column="image_name",
            status_column="status",
            status_values="active",
        ) == ["a.JPG"]

        packet_dir = root / "packets"
        packet_dir.mkdir()
        raw = np.zeros((4, 3, 4), dtype=np.float32)
        raw[0] = 0.5
        raw[1] = 2.0
        raw[2] = 8.0
        raw[3] = 0.125
        row = save_packet(
            raw,
            image_name="a.JPG",
            index=0,
            out_dir=packet_dir,
            numerical_support_floor=1.0e-6,
            variance_clamp_tolerance=1.0e-5,
        )
        assert row["packet_recompute_passed"] is True
        assert row["width"] == 4 and row["height"] == 3
        assert Path(row["packet_path"]).is_file()
        with np.load(row["packet_path"], allow_pickle=False) as packet:
            np.testing.assert_allclose(packet["alpha_normalized_expected_camera_z"], 4.0)
            assert packet["metric_depth_valid_mask"].dtype == np.bool_

        parser = build_parser()
        assert parser.get_default("image_domain") == "colmap_4_0_4_image_undistorter_pinhole_max_1414"
        assert parser.get_default("raw_camera_z_to_protocol_scale") == 1.0

    print("citygaussian_v2_export_cpu_test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
