#!/usr/bin/env python3
"""Evaluate one method with the active M3M-GCP native-quarter protocol.

The evaluator consumes raw A/M1 metric-depth packets, samples floating pixels,
uses the frozen per-scene common Sim(3), and never estimates a method-specific
registration transform.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from evaluate_gaussian_gcp_geometry import (
    backproject_world,
    load_depth_index,
    load_depth_manifest,
    validate_metric_packet_npz,
)
from m3m_native_quarter_protocol import (
    DEFAULT_SUPPORT_FLOOR,
    PIXEL_CONVENTION,
    PIXEL_DOMAIN,
    PROTOCOL_ID,
    PROTOCOL_RELEASE_SCHEMA,
    aggregate_view_groups,
    coverage_gate,
    half_pixel_sensitivity,
    residual_statistics,
    sample_raw_moment_camera_z,
    sha256_file,
    sim3_from_mapping,
)
from metric_depth_packet import METRIC_PACKET_MANIFEST_SCHEMA
from triangulate_gcp_points import read_model


TARGET_FIELDS = (
    "cgcs2000_gk_cm108_e_m",
    "cgcs2000_gk_cm108_n_m",
    "cgcs2000_normal_height_m",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def verify_protocol_release(protocol_root: Path) -> dict[str, Any]:
    manifest_path = protocol_root / "protocol_release_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != PROTOCOL_RELEASE_SCHEMA or manifest.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("not an active M3M-GCP native-quarter protocol release")
    for entry in manifest.get("payload_files", []):
        path = protocol_root / entry["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"protocol payload SHA-256 mismatch: {path}")
    if manifest.get("method_result_sim3_refit_allowed") is not False:
        raise ValueError("protocol release does not explicitly forbid method-specific Sim(3) refit")
    return manifest


def verify_data_release(data_root: Path, protocol_manifest: dict[str, Any]) -> dict[str, Any]:
    contract_path = data_root / "DATA_CONTRACT_DRAFT.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    expected = protocol_manifest["source_data"]
    if contract.get("release_root_digest_sha256") != expected["release_root_digest_sha256"]:
        raise ValueError("source data release root digest differs from protocol release")
    if sha256_file(contract_path) != expected["data_contract_sha256"]:
        raise ValueError("source data contract SHA-256 differs from protocol release")
    for key_path, key_sha in (
        ("source_split_relative_path", "source_split_sha256"),
        ("source_points_relative_path", "source_points_sha256"),
    ):
        path = data_root / expected[key_path]
        if sha256_file(path) != expected[key_sha]:
            raise ValueError(f"source data payload SHA-256 mismatch: {path}")
    return contract


def load_targets(path: Path) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for row in read_csv(path):
        output[row["point_name"]] = np.asarray(
            [float(row[field]) for field in TARGET_FIELDS], dtype=np.float64
        )
    return output


def validate_packet_contract(
    manifest: dict[str, Any],
    scene: str,
    protocol_manifest: dict[str, Any],
) -> None:
    if manifest.get("schema") != METRIC_PACKET_MANIFEST_SCHEMA:
        raise ValueError("native-quarter formal evaluation requires a v2 metric-depth packet manifest")
    requirements = {
        "protocol_id": PROTOCOL_ID,
        "scene": scene,
        "image_domain": PIXEL_DOMAIN,
        "pixel_coordinate_convention": PIXEL_CONVENTION,
        "camera_z_unit_contract": "frozen_colmap_model_camera_z_units",
        "adapter_conformance_status": "PASS",
        "source_data_release_root_digest_sha256": protocol_manifest["source_data"][
            "release_root_digest_sha256"
        ],
    }
    mismatches = {
        key: {"expected": expected, "actual": manifest.get(key)}
        for key, expected in requirements.items()
        if manifest.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"metric packet native-quarter contract mismatch: {mismatches}")
    if not math.isclose(
        float(manifest["numerical_support_floor"]),
        DEFAULT_SUPPORT_FLOOR,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("packet numerical support floor differs from active protocol")
    conformance_path = Path(str(manifest.get("adapter_conformance_report", "")))
    if not conformance_path.is_absolute():
        conformance_path = Path(str(manifest.get("_manifest_path", ""))).parent / conformance_path
    expected_sha = str(manifest.get("adapter_conformance_report_sha256", ""))
    if not conformance_path.is_file() or not expected_sha or sha256_file(conformance_path) != expected_sha:
        raise ValueError("adapter conformance report is absent or has the wrong SHA-256")


def evaluate(
    data_root: Path,
    protocol_root: Path,
    scene: str,
    method_id: str,
    packet_manifest_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    data_root = data_root.resolve()
    protocol_root = protocol_root.resolve()
    packet_manifest_path = packet_manifest_path.resolve()
    out_dir = out_dir.resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty evaluator output: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    protocol_manifest = verify_protocol_release(protocol_root)
    verify_data_release(data_root, protocol_manifest)
    valid_scenes = {row["scene"] for row in protocol_manifest["scene_summaries"]}
    if scene not in valid_scenes:
        raise ValueError(f"scene is not in protocol release: {scene}")

    packet_manifest = load_depth_manifest(packet_manifest_path)
    packet_manifest["_manifest_path"] = str(packet_manifest_path)
    validate_packet_contract(packet_manifest, scene, protocol_manifest)
    packet_index = load_depth_index(packet_manifest_path, packet_manifest)

    transform_path = protocol_root / "scenes" / scene / "common_sim3.json"
    transform_payload = json.loads(transform_path.read_text(encoding="utf-8"))
    if transform_payload.get("protocol_id") != PROTOCOL_ID or transform_payload.get("scene") != scene:
        raise ValueError("common Sim(3) identity mismatch")
    sim3 = sim3_from_mapping(transform_payload)

    observations = [
        row
        for row in read_csv(protocol_root / "observation_semantics.csv")
        if row["scene"] == scene and truthy(row["active_formal_eligible"])
    ]
    dispositions = {
        row["point_name"]: row
        for row in read_csv(protocol_root / "point_instance_disposition.csv")
        if row["scene"] == scene and truthy(row["active_formal_eligible"])
    }
    if not observations or not dispositions:
        raise ValueError(f"protocol release contains no active observations for {scene}")
    source_points_path = data_root / protocol_manifest["source_data"]["source_points_relative_path"]
    targets = load_targets(source_points_path)

    cameras, images, _points = read_model(data_root / scene / "sparse/0")
    images_by_name = {image.name: image for image in images.values()}
    packet_cache: dict[str, dict[str, np.ndarray]] = {}
    observation_rows: list[dict[str, Any]] = []
    valid_by_point: dict[str, list[dict[str, Any]]] = defaultdict(list)
    failure_counts: Counter[str] = Counter()

    for row in sorted(observations, key=lambda value: (value["point_name"], value["image_name"])):
        image_name = row["image_name"]
        failure_reason = ""
        sample: dict[str, Any]
        sensitivity: dict[str, Any] = {
            "max_abs_camera_z_delta_model_units": None,
            "median_abs_camera_z_delta_model_units": None,
        }
        model_xyz = np.full(3, np.nan, dtype=np.float64)
        entry = packet_index.get(image_name) or packet_index.get(Path(image_name).name)
        if entry is None:
            sample = {
                "valid": False,
                "failure_reason": "missing_metric_packet",
                "accumulated_alpha_interp": math.nan,
                "weighted_camera_z_sum_interp": math.nan,
                "camera_z": math.nan,
            }
        elif image_name not in images_by_name:
            sample = {
                "valid": False,
                "failure_reason": "image_absent_from_frozen_colmap_model",
                "accumulated_alpha_interp": math.nan,
                "weighted_camera_z_sum_interp": math.nan,
                "camera_z": math.nan,
            }
        else:
            if int(entry["width"]) != int(row["target_width"]) or int(entry["height"]) != int(
                row["target_height"]
            ):
                raise ValueError(f"packet/annotation shape mismatch for {image_name}")
            cache_key = str(entry["packet_path"])
            if cache_key not in packet_cache:
                packet_cache[cache_key] = validate_metric_packet_npz(
                    Path(cache_key),
                    entry,
                    numerical_support_floor=float(packet_manifest["numerical_support_floor"]),
                    variance_clamp_tolerance=float(packet_manifest["variance_clamp_tolerance"]),
                    variance_validation_policy=str(packet_manifest["variance_validation_policy"]),
                    variance_validation_abs_floor=float(packet_manifest["variance_validation_abs_floor"]),
                    variance_validation_ulp_factor=float(packet_manifest["variance_validation_ulp_factor"]),
                    variance_validation_dtype=str(packet_manifest["variance_validation_dtype"]),
                    variance_validation_rtol=float(packet_manifest["variance_validation_rtol"]),
                    variance_nonnegativity_policy=str(packet_manifest["variance_nonnegativity_policy"]),
                    variance_negative_handling=str(packet_manifest["variance_negative_handling"]),
                    variance_raw_packet_modified=bool(packet_manifest["variance_raw_packet_modified"]),
                )
            packet = packet_cache[cache_key]
            u = float(row["u_px"])
            v = float(row["v_px"])
            sample = sample_raw_moment_camera_z(
                packet["accumulated_alpha"],
                packet["weighted_camera_z_sum"],
                u,
                v,
                support_floor=DEFAULT_SUPPORT_FLOOR,
            )
            sensitivity = half_pixel_sensitivity(
                packet["accumulated_alpha"],
                packet["weighted_camera_z_sum"],
                u,
                v,
                support_floor=DEFAULT_SUPPORT_FLOOR,
            )
            if sample["valid"]:
                image = images_by_name[image_name]
                model_xyz = backproject_world(
                    cameras[image.camera_id], image, u, v, float(sample["camera_z"])
                )
                valid_by_point[row["point_name"]].append(
                    {
                        "model_xyz": model_xyz,
                        "view_class": row["view_class"],
                        "azimuth_bin_45deg": int(row["azimuth_bin_45deg"]),
                        "image_name": image_name,
                        "observation_id": row["observation_id"],
                    }
                )
        failure_reason = str(sample["failure_reason"])
        if failure_reason:
            failure_counts[failure_reason] += 1
        observation_rows.append(
            {
                "observation_id": row["observation_id"],
                "scene": scene,
                "point_name": row["point_name"],
                "role": row["active_role"],
                "image_name": image_name,
                "u_px": row["u_px"],
                "v_px": row["v_px"],
                "view_class": row["view_class"],
                "azimuth_bin_45deg": row["azimuth_bin_45deg"],
                "valid": str(bool(sample["valid"])).lower(),
                "failure_reason": failure_reason,
                "accumulated_alpha_interp": sample["accumulated_alpha_interp"],
                "weighted_camera_z_sum_interp": sample["weighted_camera_z_sum_interp"],
                "camera_z": sample["camera_z"],
                "model_x": model_xyz[0] if np.isfinite(model_xyz[0]) else "",
                "model_y": model_xyz[1] if np.isfinite(model_xyz[1]) else "",
                "model_z": model_xyz[2] if np.isfinite(model_xyz[2]) else "",
                "half_pixel_max_abs_camera_z_delta_model_units": sensitivity.get(
                    "max_abs_camera_z_delta_model_units"
                ),
                "half_pixel_median_abs_camera_z_delta_model_units": sensitivity.get(
                    "median_abs_camera_z_delta_model_units"
                ),
                "half_pixel_max_abs_camera_z_delta_target_m": (
                    float(sim3.scale) * float(sensitivity["max_abs_camera_z_delta_model_units"])
                    if sensitivity.get("max_abs_camera_z_delta_model_units") is not None
                    else None
                ),
            }
        )

    point_rows: list[dict[str, Any]] = []
    residual_vectors: dict[str, list[np.ndarray]] = {"control": [], "checkpoint": [], "all": []}
    for point_name in sorted(dispositions):
        disposition = dispositions[point_name]
        rows_for_point = [row for row in observations if row["point_name"] == point_name]
        valid = valid_by_point.get(point_name, [])
        gate = coverage_gate(
            expected_observation_count=len(rows_for_point),
            valid_view_classes=[row["view_class"] for row in valid],
        )
        role = disposition["active_role"]
        base: dict[str, Any] = {
            "scene": scene,
            "point_name": point_name,
            "role": role,
            "surface_level": disposition["surface_level"],
            **gate,
        }
        base["failure_reasons"] = ";".join(gate["failure_reasons"])
        if not gate["passed"]:
            point_rows.append(base)
            continue
        aggregate, diagnostics = aggregate_view_groups(valid)
        predicted = sim3.apply(aggregate)
        target = targets[point_name]
        residual = predicted - target
        residual_vectors[role].append(residual)
        residual_vectors["all"].append(residual)
        base.update(
            {
                "aggregation_group_count": diagnostics["group_count"],
                "model_x": aggregate[0],
                "model_y": aggregate[1],
                "model_z": aggregate[2],
                "predicted_e_m": predicted[0],
                "predicted_n_m": predicted[1],
                "predicted_z_m": predicted[2],
                "target_e_m": target[0],
                "target_n_m": target[1],
                "target_z_m": target[2],
                "residual_e_m": residual[0],
                "residual_n_m": residual[1],
                "residual_z_m": residual[2],
                "error_h_m": float(np.linalg.norm(residual[:2])),
                "error_z_m": float(abs(residual[2])),
                "error_3d_m": float(np.linalg.norm(residual)),
                "multiview_scatter_median_m": diagnostics["scatter_median_m"],
                "multiview_scatter_p90_m": diagnostics["scatter_p90_m"],
                "multiview_scatter_max_m": diagnostics["scatter_max_m"],
            }
        )
        point_rows.append(base)

    role_totals = Counter(row["active_role"] for row in dispositions.values())
    role_passed = Counter(row["role"] for row in point_rows if truthy(row.get("passed")))
    stats = {
        role: residual_statistics(residual_vectors[role])
        for role in ("control", "checkpoint", "all")
    }
    summary = {
        "schema": "m3m_gcp_native_quarter_method_evaluation_v1",
        "protocol_id": PROTOCOL_ID,
        "scene": scene,
        "method_id": method_id,
        "status": "completed_with_coverage_report",
        "common_primary_semantics": "render-support expected camera-z coordinate sampled as bilinear(M1)/bilinear(A)",
        "physical_surface_claim": False,
        "method_specific_sim3_fitted": False,
        "common_sim3_path": str(transform_path),
        "common_sim3_sha256": sha256_file(transform_path),
        "packet_manifest": str(packet_manifest_path),
        "packet_manifest_sha256": sha256_file(packet_manifest_path),
        "point_counts": {
            "control_total": role_totals["control"],
            "control_passed": role_passed["control"],
            "checkpoint_total": role_totals["checkpoint"],
            "checkpoint_passed": role_passed["checkpoint"],
        },
        "checkpoint_coverage_rate": (
            role_passed["checkpoint"] / role_totals["checkpoint"]
            if role_totals["checkpoint"]
            else None
        ),
        "residual_statistics": stats,
        "observation_failure_counts": dict(sorted(failure_counts.items())),
        "ranking_policy": "checkpoint errors and checkpoint coverage are reported together; missing surface support is never silently deleted",
    }

    write_csv(
        out_dir / "observation_samples.csv",
        observation_rows,
        [
            "observation_id",
            "scene",
            "point_name",
            "role",
            "image_name",
            "u_px",
            "v_px",
            "view_class",
            "azimuth_bin_45deg",
            "valid",
            "failure_reason",
            "accumulated_alpha_interp",
            "weighted_camera_z_sum_interp",
            "camera_z",
            "model_x",
            "model_y",
            "model_z",
            "half_pixel_max_abs_camera_z_delta_model_units",
            "half_pixel_median_abs_camera_z_delta_model_units",
            "half_pixel_max_abs_camera_z_delta_target_m",
        ],
    )
    write_csv(
        out_dir / "point_results.csv",
        point_rows,
        [
            "scene",
            "point_name",
            "role",
            "surface_level",
            "passed",
            "failure_reasons",
            "expected_observation_count",
            "required_valid_observation_count",
            "valid_observation_count",
            "valid_nadir_count",
            "valid_oblique_count",
            "aggregation_group_count",
            "model_x",
            "model_y",
            "model_z",
            "predicted_e_m",
            "predicted_n_m",
            "predicted_z_m",
            "target_e_m",
            "target_n_m",
            "target_z_m",
            "residual_e_m",
            "residual_n_m",
            "residual_z_m",
            "error_h_m",
            "error_z_m",
            "error_3d_m",
            "multiview_scatter_median_m",
            "multiview_scatter_p90_m",
            "multiview_scatter_max_m",
        ],
    )
    write_json(out_dir / "evaluation_summary.json", summary)
    evaluator_manifest = {
        "schema": "m3m_gcp_native_quarter_evaluator_run_manifest_v1",
        "protocol_release_manifest": str(protocol_root / "protocol_release_manifest.json"),
        "protocol_release_manifest_sha256": sha256_file(
            protocol_root / "protocol_release_manifest.json"
        ),
        "source_data_contract": str(data_root / "DATA_CONTRACT_DRAFT.json"),
        "source_data_contract_sha256": sha256_file(data_root / "DATA_CONTRACT_DRAFT.json"),
        "packet_manifest": str(packet_manifest_path),
        "packet_manifest_sha256": sha256_file(packet_manifest_path),
        "operator": "bilinear_raw_moment_ratio_v1",
        "aggregation": "view_class_azimuth_group_geometric_median_v1",
        "sim3_policy": "frozen_common_transform_no_method_refit",
        "outputs": {
            name: sha256_file(out_dir / name)
            for name in ("observation_samples.csv", "point_results.csv", "evaluation_summary.json")
        },
    }
    write_json(out_dir / "evaluator_manifest.json", evaluator_manifest)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True, type=Path)
    parser.add_argument("--protocol_release", required=True, type=Path)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--method_id", required=True)
    parser.add_argument("--metric_packet_manifest", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    args = parser.parse_args()
    summary = evaluate(
        args.data_root,
        args.protocol_release,
        args.scene,
        args.method_id,
        args.metric_packet_manifest,
        args.out_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
