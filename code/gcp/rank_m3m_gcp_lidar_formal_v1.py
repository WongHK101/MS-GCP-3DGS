#!/usr/bin/env python3
"""Build the frozen six-scene, input-class LiDAR formal-v1 ranking."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from verify_m3m_gcp_lidar_formal_v1 import (
    METRIC_FIELDS,
    OVERALL_RANK_KEYS,
    canonical_sha256,
    competition_rank_rows,
    sha256_file,
)


PROTOCOL_ID = "m3m_gcp_lidar_rendered_surface_v1"
SCENES = (
    "gcp_3000_20260602",
    "gcp_5000_20260602",
    "gcp_20000_20260602",
    "gcp_10000_20260610",
    "gcp_50000_20260610",
    "gcp_100000_20260610",
)
COMPLETE = "COMPLETE_RANKED"
ALLOWED_STATUSES = {COMPLETE, "OOM_UNRANKED", "FAILED_UNRANKED", "INCOMPLETE_UNRANKED"}


def macro_mean(rows: list[dict[str, Any]], field: str) -> float:
    return sum(float(row["metrics"][field]) for row in rows) / len(rows)


def build_ranking(
    manifest: dict[str, Any], *, contract_sha256: str, activation_sha256: str
) -> dict[str, Any]:
    if manifest.get("schema") != "m3m_gcp_lidar_six_scene_results_manifest_v1":
        raise ValueError("six-scene results manifest schema mismatch")
    if manifest.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("six-scene results manifest protocol mismatch")
    if manifest.get("canonical_sha256") != canonical_sha256(manifest):
        raise ValueError("six-scene results manifest canonical SHA mismatch")
    output_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for method in manifest.get("methods", []):
        method_id = str(method["method_id"])
        if method_id in seen:
            raise ValueError(f"duplicate method: {method_id}")
        seen.add(method_id)
        scene_entries = method.get("scenes", [])
        if {entry.get("scene") for entry in scene_entries} != set(SCENES):
            raise ValueError(f"{method_id}: scene inventory differs from frozen six scenes")
        complete_results: list[dict[str, Any]] = []
        statuses: dict[str, str] = {}
        for entry in scene_entries:
            scene = str(entry["scene"])
            status = str(entry["status"])
            if status not in ALLOWED_STATUSES:
                raise ValueError(f"{method_id}/{scene}: unknown status {status}")
            statuses[scene] = status
            if status != COMPLETE:
                if entry.get("method_result_path") or entry.get("method_result_sha256"):
                    raise ValueError(f"{method_id}/{scene}: failed status cannot carry fabricated result")
                continue
            result_path = Path(str(entry["method_result_path"]))
            if not result_path.is_file() or sha256_file(result_path) != entry.get("method_result_sha256"):
                raise ValueError(f"{method_id}/{scene}: result path/SHA mismatch")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("schema") != "m3m_gcp_lidar_method_result_v1":
                raise ValueError(f"{method_id}/{scene}: method result schema mismatch")
            if result.get("canonical_sha256") != canonical_sha256(result):
                raise ValueError(f"{method_id}/{scene}: result canonical SHA mismatch")
            if result.get("contract_file_sha256") != contract_sha256:
                raise ValueError(f"{method_id}/{scene}: contract SHA mismatch")
            if result.get("activation_manifest_sha256") != activation_sha256:
                raise ValueError(f"{method_id}/{scene}: activation SHA mismatch")
            if result.get("scene") != scene or result.get("method_id") != method_id:
                raise ValueError(f"{method_id}/{scene}: result identity mismatch")
            if result.get("input_class") != method.get("input_class"):
                raise ValueError(f"{method_id}/{scene}: input class mismatch")
            if set(result.get("metrics", {})) != set(METRIC_FIELDS):
                raise ValueError(f"{method_id}/{scene}: metric inventory mismatch")
            complete_results.append(result)

        completed = len(complete_results)
        row: dict[str, Any] = {
            "method_id": method_id,
            "method_name": method["method_name"],
            "input_class": method["input_class"],
            "completed_scene_count": completed,
            "scene_statuses": statuses,
            "overall_status": COMPLETE if completed == len(SCENES) else "INCOMPLETE_UNRANKED",
            "ranking_eligible": completed == len(SCENES),
        }
        if completed:
            row["partial_macro_diagnostic"] = {
                field: macro_mean(complete_results, field) for field in METRIC_FIELDS
            }
        if completed == len(SCENES):
            row.update(
                {
                    "macro_fscore_10cm": macro_mean(complete_results, "fscore_10cm"),
                    "macro_chamfer_l1_mean_m": macro_mean(complete_results, "chamfer_l1_mean_m"),
                    "macro_precision_10cm": macro_mean(complete_results, "precision_10cm"),
                    "macro_recall_10cm": macro_mean(complete_results, "recall_10cm"),
                }
            )
        output_rows.append(row)

    eligible = [row for row in output_rows if row["ranking_eligible"]]
    for input_class in sorted({row["input_class"] for row in eligible}):
        ranked = competition_rank_rows(
            [row for row in eligible if row["input_class"] == input_class],
            OVERALL_RANK_KEYS,
        )
        ranks = {row["method_id"]: row["rank"] for row in ranked}
        for row in output_rows:
            if row["ranking_eligible"] and row["input_class"] == input_class:
                row["official_input_class_rank"] = ranks[row["method_id"]]
    descriptive = competition_rank_rows(eligible, OVERALL_RANK_KEYS)
    positions = {row["method_id"]: index for index, row in enumerate(descriptive, 1)}
    for row in output_rows:
        if row["ranking_eligible"]:
            row["combined_descriptive_order_not_official_rank"] = positions[row["method_id"]]
    output = {
        "schema": "m3m_gcp_lidar_six_scene_ranking_v1",
        "protocol_id": PROTOCOL_ID,
        "contract_file_sha256": contract_sha256,
        "activation_manifest_sha256": activation_sha256,
        "scene_order": list(SCENES),
        "aggregation": "unweighted_arithmetic_macro_average",
        "micro_pooling": "FORBIDDEN",
        "official_ranking_scope": "within_input_class_only",
        "complete_scene_count_required_for_rank": 6,
        "methods": output_rows,
    }
    output["canonical_sha256"] = canonical_sha256(output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite six-scene ranking")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    result = build_ranking(
        manifest,
        contract_sha256=sha256_file(args.contract),
        activation_sha256=sha256_file(args.activation),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
