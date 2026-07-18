#!/usr/bin/env python3
"""Select the frozen Original 3DGS camera residency contract from resource evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def decide(a50: str, a100: str | None, b50: str | None, b100: str | None) -> dict[str, Any]:
    a_rows = [status for status in (a50, a100) if status is not None]
    if any(status not in {"PASS", "GPU_MEMORY_BLOCKED"} for status in a_rows):
        return {"status": "BLOCKER", "selected_contract": None, "reason": "candidate_a_non_gpu_failure"}
    if a50 == "PASS" and a100 == "PASS":
        return {"status": "SELECTED_A", "selected_contract": "path_backed_cuda_resident", "reason": "candidate_a_50k_100k_pass"}
    if a50 == "PASS" and a100 is None:
        return {"status": "NEEDS_A100", "selected_contract": None, "reason": "candidate_a_50k_pass"}
    if all(status in {"PASS", "GPU_MEMORY_BLOCKED"} for status in a_rows) and "GPU_MEMORY_BLOCKED" in a_rows:
        if b50 is None:
            return {"status": "B_ELIGIBLE", "selected_contract": None, "reason": "candidate_a_gpu_only_failure"}
        if b50 != "PASS" or (b100 is not None and b100 != "PASS"):
            return {"status": "BLOCKER", "selected_contract": None, "reason": "candidate_b_failure"}
        if b100 is None:
            return {"status": "NEEDS_B100", "selected_contract": None, "reason": "candidate_b_50k_pass"}
        return {"status": "SELECTED_B", "selected_contract": "path_backed_cpu_backed_official", "reason": "candidate_b_50k_100k_pass"}
    return {"status": "BLOCKER", "selected_contract": None, "reason": "incomplete_or_invalid_candidate_evidence"}


def status_from_path(path: Path | None) -> str | None:
    if path is None:
        return None
    return str(json.loads(path.read_text(encoding="utf-8"))["status"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a50", type=Path, required=True)
    parser.add_argument("--a100", type=Path)
    parser.add_argument("--b50", type=Path)
    parser.add_argument("--b100", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = decide(
        status_from_path(args.a50), status_from_path(args.a100),
        status_from_path(args.b50), status_from_path(args.b100),
    )
    result.update({
        "schema": "gs_gcp_original_3dgs_camera_contract_selection_v1",
        "selection_order": "candidate_a_then_candidate_b_only_for_a_gpu_only_failure",
        "original_3dgs_data_residency": (
            "cuda_resident_official" if result["status"] == "SELECTED_A"
            else "cpu_backed_official" if result["status"] == "SELECTED_B"
            else None
        ),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] in {"SELECTED_A", "SELECTED_B", "NEEDS_A100", "B_ELIGIBLE", "NEEDS_B100"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
