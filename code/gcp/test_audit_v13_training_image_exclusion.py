#!/usr/bin/env python3
"""Unit checks for the v1.3 image-level exclusion audit helpers."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from audit_v13_training_image_exclusion import (
    frozen_release_references,
    rotation_angle_deg,
)


def main() -> int:
    assert rotation_angle_deg(np.eye(3)) == 0.0
    half_turn = np.diag([-1.0, -1.0, 1.0])
    assert abs(rotation_angle_deg(half_turn) - 180.0) < 1e-12

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "a.json").write_text('{"image":"target.JPG"}', encoding="utf-8")
        (root / "b.csv").write_text("image\r\ntarget.JPG\r\n", encoding="utf-8")
        (root / "ignored.bin").write_bytes(b"target.JPG")
        assert frozen_release_references(root, "target.JPG") == ["a.json", "b.csv"]
        assert frozen_release_references(root, "missing.JPG") == []

    print("image exclusion helper tests: 4/4 PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
