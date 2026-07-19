#!/usr/bin/env python3
"""Compare direct and externally probed Original 3DGS micro runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ply_summary(path: Path) -> dict[str, Any]:
    from plyfile import PlyData
    ply = PlyData.read(path)
    vertex = ply["vertex"].data
    properties = [(name, str(vertex.dtype.fields[name][0])) for name in vertex.dtype.names]
    return {
        "count": len(vertex),
        "properties": properties,
        "array": vertex,
        "sha256": sha256_file(path),
    }


def _normalized_argv(argv: list[str], allowed_value_options: set[str]) -> list[str]:
    result = list(argv)
    for index, value in enumerate(result[:-1]):
        if value in allowed_value_options:
            result[index + 1] = f"<{value.lstrip('-')}_compatibility_value>"
    return result


def compare(
    direct_root: Path,
    probed_root: Path,
    direct_trace: Path,
    probed_trace: Path,
    allowed_argv_value_options: set[str] | None = None,
) -> dict[str, Any]:
    direct_payload = json.loads(direct_trace.read_text(encoding="utf-8"))
    probed_payload = json.loads(probed_trace.read_text(encoding="utf-8"))
    checks = []

    def check(name: str, passed: bool, evidence: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "evidence": evidence})

    check("trace_status", direct_payload.get("status") == probed_payload.get("status") == "PASS", [direct_payload.get("status"), probed_payload.get("status")])
    allowed = allowed_argv_value_options or set()
    direct_argv = _normalized_argv(direct_payload.get("official_child_argv", []), allowed)
    probed_argv = _normalized_argv(probed_payload.get("official_child_argv", []), allowed)
    check("child_argv", direct_argv == probed_argv, {"allowed_value_options": sorted(allowed)})
    check("train_image_order", direct_payload.get("train_image_order") == probed_payload.get("train_image_order"), {
        "direct_length": len(direct_payload.get("train_image_order", [])),
        "probed_length": len(probed_payload.get("train_image_order", [])),
    })
    check("loss_trace", direct_payload.get("iterations") == probed_payload.get("iterations"), {
        "direct_length": len(direct_payload.get("iterations", [])),
        "probed_length": len(probed_payload.get("iterations", [])),
    })
    direct_ply = _ply_summary(direct_root / "point_cloud" / "iteration_100" / "point_cloud.ply")
    probed_ply = _ply_summary(probed_root / "point_cloud" / "iteration_100" / "point_cloud.ply")
    check("gaussian_count", direct_ply["count"] == probed_ply["count"], [direct_ply["count"], probed_ply["count"]])
    check("property_schema", direct_ply["properties"] == probed_ply["properties"], None)
    bitwise = direct_ply["sha256"] == probed_ply["sha256"]
    check("point_cloud_bitwise_identity", bitwise, {"direct_sha256": direct_ply["sha256"], "probed_sha256": probed_ply["sha256"]})
    check("gaussian_order_and_values", bool((direct_ply["array"] == probed_ply["array"]).all()), None)
    finite = True
    for array in (direct_ply["array"], probed_ply["array"]):
        for name in array.dtype.names:
            if array.dtype.fields[name][0].kind == "f" and not bool(__import__("numpy").isfinite(array[name]).all()):
                finite = False
    check("point_cloud_finite", finite, None)
    passed = all(row["passed"] for row in checks)
    return {
        "schema": "gs_gcp_original_3dgs_micro_probe_equivalence_v1",
        "status": "PASS" if passed else "BLOCKER",
        "comparison_policy": "strict_bitwise_primary; tolerance requires separate pre-existing CUDA nondeterminism proof",
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct_model", type=Path, required=True)
    parser.add_argument("--probed_model", type=Path, required=True)
    parser.add_argument("--direct_trace", type=Path, required=True)
    parser.add_argument("--probed_trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allowed_argv_value_option", action="append", default=[])
    args = parser.parse_args()
    result = compare(
        args.direct_model, args.probed_model, args.direct_trace, args.probed_trace,
        set(args.allowed_argv_value_option),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
