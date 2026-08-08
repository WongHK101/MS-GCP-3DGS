#!/usr/bin/env python3
"""Validate the frozen M3M-GCP native-quarter protocol overlay and its source release."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from evaluate_m3m_native_quarter_geometry import verify_data_release, verify_protocol_release
from m3m_native_quarter_protocol import PROTOCOL_ID, coverage_gate, sha256_file


EXPECTED_QUARANTINE = {
    ("gcp_100000_20260610", "dxl3"),
    ("gcp_100000_20260610", "dyl2"),
    ("gcp_100000_20260610", "wy3_1"),
    ("gcp_100000_20260610", "wy3_2"),
    ("gcp_20000_20260602", "dyl2"),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def validate(
    data_root: Path,
    protocol_root: Path,
    pin_path: Path,
) -> dict[str, Any]:
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    try:
        manifest = verify_protocol_release(protocol_root)
        verify_data_release(data_root, manifest)
    except Exception as exc:
        return {
            "schema": "m3m_gcp_native_quarter_protocol_release_validation_v2",
            "passed": False,
            "errors": [f"cryptographic/source verification failed: {exc}"],
        }
    pin = json.loads(pin_path.read_text(encoding="utf-8"))
    require(pin.get("schema") == "m3m_gcp_native_quarter_protocol_release_pin_v2", "pin schema mismatch")
    require(pin.get("protocol_id") == PROTOCOL_ID, "pin protocol id mismatch")
    require(protocol_root.name == pin.get("release_directory_name"), "protocol directory name mismatch")
    require(
        sha256_file(protocol_root / "protocol_release_manifest.json")
        == pin.get("protocol_release_manifest_sha256"),
        "protocol manifest SHA differs from repo pin",
    )
    require(
        sha256_file(protocol_root / "SHA256SUMS.txt") == pin.get("sha256sums_sha256"),
        "SHA256SUMS SHA differs from repo pin",
    )
    require(
        manifest.get("payload_manifest_canonical_sha256")
        == pin.get("payload_manifest_canonical_sha256"),
        "payload canonical digest differs from repo pin",
    )
    require(data_root.name == pin.get("source_data_directory_name"), "source data directory name mismatch")
    pin_coverage = pin.get("coverage_contract", {})
    require(pin_coverage.get("minimum_valid_oblique_azimuth_bins") == 2, "pin oblique-bin gate mismatch")
    require(pin_coverage.get("minimum_oblique_azimuth_circular_bin_separation") == 2, "pin azimuth-separation gate mismatch")
    require(pin_coverage.get("azimuth_bin_count") == 8, "pin azimuth-bin count mismatch")
    require(pin_coverage.get("actual_angle_at_least_90_degrees_claimed") is False, "pin incorrectly claims a continuous 90-degree gate")
    pin_ranking = pin.get("ranking_contract", {})
    require(pin_ranking.get("complete") == "COMPLETE_RANKED", "pin complete status mismatch")
    require(pin_ranking.get("incomplete") == "INCOMPLETE_UNRANKED", "pin incomplete status mismatch")
    require(pin_ranking.get("all_formal_checkpoints_required") is True, "pin all-checkpoint ranking gate missing")
    aggregation = manifest.get("aggregation_contract", {})
    require(aggregation.get("minimum_valid_oblique_azimuth_bins") == 2, "manifest oblique-bin gate mismatch")
    require(aggregation.get("minimum_oblique_azimuth_circular_bin_separation") == 2, "manifest azimuth-separation gate mismatch")
    require("not a claim" in str(aggregation.get("separation_interpretation", "")), "manifest continuous-angle disclaimer missing")
    ranking = manifest.get("ranking_contract", {})
    require(ranking.get("ranked_status") == "COMPLETE_RANKED", "manifest complete status mismatch")
    require(ranking.get("incomplete_status") == "INCOMPLETE_UNRANKED", "manifest incomplete status mismatch")

    sum_errors = []
    sum_count = 0
    for line in (protocol_root / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            sum_errors.append(f"unparseable SHA256SUMS row: {line}")
            continue
        expected, relative = parts
        path = protocol_root / relative
        if not path.is_file() or sha256_file(path) != expected:
            sum_errors.append(relative)
        sum_count += 1
    require(not sum_errors, f"SHA256SUMS payload failures: {sum_errors}")

    dispositions = read_csv(protocol_root / "point_instance_disposition.csv")
    active = [row for row in dispositions if truthy(row["active_formal_eligible"])]
    quarantined = {
        (row["scene"], row["point_name"])
        for row in dispositions
        if row["audit_disposition"] == "quarantined_anchor_binding"
    }
    roles = Counter(row["active_role"] for row in active)
    require(len(dispositions) == 87, "source disposition count is not 87")
    require(len(active) == 82, "active disposition count is not 82")
    require(roles == Counter({"control": 45, "checkpoint": 37}), f"active role counts differ: {roles}")
    require(quarantined == EXPECTED_QUARANTINE, f"quarantine differs: {sorted(quarantined ^ EXPECTED_QUARANTINE)}")
    require(not any(row["surface_level"] == "roof" for row in active), "active roof instance exists")

    observations = read_csv(protocol_root / "observation_semantics.csv")
    active_observations = [row for row in observations if truthy(row["active_formal_eligible"])]
    require(len(active_observations) == 1018, "active observation count is not 1018")
    require(all(truthy(row["safe_bilinear_stencil"]) for row in active_observations), "unsafe bilinear stencil exists")
    require({row["view_class"] for row in active_observations} == {"nadir", "oblique"}, "view classes incomplete")
    observations_by_instance: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in active_observations:
        observations_by_instance[(row["scene"], row["point_name"])].append(row)
    source_coverage_rows = []
    for (scene, point_name), rows in sorted(observations_by_instance.items()):
        gate = coverage_gate(
            expected_observation_count=len(rows),
            valid_view_classes=[row["view_class"] for row in rows],
            valid_azimuth_bins_45deg=[int(row["azimuth_bin_45deg"]) for row in rows],
        )
        require(gate["passed"], f"{scene}/{point_name}: source observations fail v2 coverage: {gate['failure_reasons']}")
        source_coverage_rows.append(gate)
    require(len(source_coverage_rows) == 82, "source coverage instance count is not 82")

    scene_rows = []
    for scene_summary in manifest.get("scene_summaries", []):
        scene = scene_summary["scene"]
        transform_path = protocol_root / "scenes" / scene / "common_sim3.json"
        transform = json.loads(transform_path.read_text(encoding="utf-8"))
        require(transform.get("method_result_refit_forbidden") is True, f"{scene}: method refit not forbidden")
        require(transform.get("control_count") >= 5, f"{scene}: too few controls")
        require(transform.get("checkpoint_count") >= 4, f"{scene}: too few checkpoints")
        scene_rows.append(
            {
                "scene": scene,
                "control_count": transform["control_count"],
                "checkpoint_count": transform["checkpoint_count"],
                "checkpoint_rmse_h_m": transform["baseline_residual_statistics"]["checkpoint"]["rmse_h_m"],
                "checkpoint_rmse_z_m": transform["baseline_residual_statistics"]["checkpoint"]["rmse_z_m"],
                "loo_max_omitted_prediction_z_m": transform["leave_one_out"]["max_omitted_prediction_error_z_m"],
                "loo_max_camera_center_shift_z_m": transform["leave_one_out"]["max_camera_center_shift_z_m"],
                "transform_canonical_sha256": transform["transform_canonical_sha256"],
            }
        )
    require(len(scene_rows) == 6, "scene transform count is not 6")
    require(manifest.get("training_allowed_globally") is False, "protocol overlay unlocked global training")
    require(pin.get("training_allowed_globally") is False, "repo pin unlocked global training")
    return {
        "schema": "m3m_gcp_native_quarter_protocol_release_validation_v2",
        "protocol_id": PROTOCOL_ID,
        "passed": not errors,
        "protocol_release_manifest_sha256": sha256_file(
            protocol_root / "protocol_release_manifest.json"
        ),
        "sha256sums_sha256": sha256_file(protocol_root / "SHA256SUMS.txt"),
        "verified_sha256sum_entries": sum_count,
        "counts": manifest["counts"],
        "source_observation_coverage": {
            "instance_count": len(source_coverage_rows),
            "minimum_oblique_azimuth_bin_count": min(
                row["valid_oblique_azimuth_bin_count"] for row in source_coverage_rows
            ),
            "minimum_max_oblique_circular_bin_separation": min(
                row["max_oblique_azimuth_circular_bin_separation"] for row in source_coverage_rows
            ),
        },
        "scene_evidence": scene_rows,
        "training_allowed_globally": False,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True, type=Path)
    parser.add_argument("--protocol_root", required=True, type=Path)
    parser.add_argument("--pin", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = validate(
        args.data_root.resolve(), args.protocol_root.resolve(), args.pin.resolve()
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
