#!/usr/bin/env python3
"""Require tensor-bitwise parity for the RGB-free geometry camera loader."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def packet_paths(root: Path) -> list[Path]:
    return sorted(root.glob("*_metric_depth_packet.npz"))


def compare(baseline: Path, candidate: Path) -> dict[str, Any]:
    baseline_packets = packet_paths(baseline)
    candidate_packets = packet_paths(candidate)
    errors: list[str] = []
    if [path.name for path in baseline_packets] != [path.name for path in candidate_packets]:
        errors.append("packet filename inventory differs")
    rows: list[dict[str, Any]] = []
    for baseline_path, candidate_path in zip(baseline_packets, candidate_packets):
        packet_errors: list[str] = []
        tensors: list[dict[str, Any]] = []
        with np.load(baseline_path, allow_pickle=False) as left, np.load(
            candidate_path, allow_pickle=False
        ) as right:
            if left.files != right.files:
                packet_errors.append("tensor key order differs")
            for name in left.files:
                if name not in right.files:
                    continue
                a = left[name]
                b = right[name]
                bitwise = (
                    a.shape == b.shape
                    and a.dtype == b.dtype
                    and a.tobytes(order="C") == b.tobytes(order="C")
                )
                if not bitwise:
                    packet_errors.append(f"tensor differs: {name}")
                tensors.append(
                    {
                        "name": name,
                        "shape": list(a.shape),
                        "dtype": str(a.dtype),
                        "bitwise_equal": bitwise,
                    }
                )
        if packet_errors:
            errors.extend(f"{baseline_path.name}: {item}" for item in packet_errors)
        rows.append(
            {
                "packet": baseline_path.name,
                "baseline_sha256": sha256(baseline_path),
                "candidate_sha256": sha256(candidate_path),
                "tensor_bitwise_equal": not packet_errors,
                "tensors": tensors,
            }
        )
    baseline_manifest = json.loads(
        (baseline / "depth_export_manifest.json").read_text(encoding="utf-8")
    )
    candidate_manifest = json.loads(
        (candidate / "depth_export_manifest.json").read_text(encoding="utf-8")
    )
    if baseline_manifest.get("rendered_view_count") != candidate_manifest.get(
        "rendered_view_count"
    ):
        errors.append("rendered view count differs")
    if candidate_manifest.get("geometry_camera_loader", {}).get("applied") is not True:
        errors.append("candidate geometry camera-only loader was not applied")
    return {
        "schema": "m3m_gcp_geometry_camera_only_parity_v1",
        "status": "PASS" if not errors else "FAIL",
        "baseline": str(baseline.resolve()),
        "candidate": str(candidate.resolve()),
        "packet_count": len(rows),
        "tensor_value_contract": "shape, dtype, and C-order bytes must match exactly",
        "rows": rows,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.baseline.resolve(), args.candidate.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
