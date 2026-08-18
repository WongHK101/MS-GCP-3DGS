#!/usr/bin/env python3
"""CPU contract checks for the MetroGS frozen-camera packet exporter."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np

from export_metrogs_depth_maps import (
    EXPECTED_RENDERER_CLASS,
    METHOD_ID,
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
        with np.load(row["packet_path"], allow_pickle=False) as packet:
            np.testing.assert_allclose(packet["alpha_normalized_expected_camera_z"], 4.0)
            assert packet["metric_depth_valid_mask"].dtype == np.bool_

        parser = build_parser()
        assert parser.get_default("raw_camera_z_to_protocol_scale") == 1.0
        assert METHOD_ID == "metrogs"
        assert EXPECTED_RENDERER_CLASS == "DistributedRendererImpl"
    print("metrogs_export_cpu_test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
