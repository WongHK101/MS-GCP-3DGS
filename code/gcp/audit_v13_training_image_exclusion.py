#!/usr/bin/env python3
"""Freeze evidence for image-level training exclusion candidates in v1.3."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

COLMAP_CODE = Path(__file__).resolve().parents[1] / "colmap"
if str(COLMAP_CODE) not in sys.path:
    sys.path.insert(0, str(COLMAP_CODE))

from prepare_scene_colmap import _estimate_similarity_umeyama, _lla_to_local_enu

from audit_v13_candidate_recall_and_gps import (
    ANNOTATION_RELATIVE_PATHS,
    camera_center,
    camera_indices,
    image_metadata,
    sha256_file,
    write_json,
)


EXPECTED_SCENES = {
    "gcp_3000_20260602",
    "gcp_5000_20260602",
    "gcp_10000_20260610",
    "gcp_20000_20260602",
    "gcp_50000_20260610",
    "gcp_100000_20260610",
}
EXPECTED_OUTLIER = ("gcp_50000_20260610", "DJI_20260610161948_0002_D.JPG")
TEXT_RELEASE_SUFFIXES = {".csv", ".json", ".md", ".txt"}


def rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def frozen_release_references(release_dir: Path, image_name: str) -> list[str]:
    references: list[str] = []
    for path in sorted(release_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_RELEASE_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            continue
        if image_name in text:
            references.append(path.relative_to(release_dir).as_posix())
    return references


def verify_remote_hash_manifest(root: Path) -> dict[str, Any]:
    lines = (root / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
    verified = 0
    for line in lines:
        expected, remote_path = line.split("  ", 1)
        marker = "/six_scene_image_pose_qc_20260716/"
        if marker not in remote_path:
            raise RuntimeError(f"Unexpected remote hash path: {remote_path}")
        relative = remote_path.split(marker, 1)[1]
        local_path = root / Path(relative)
        if not local_path.is_file() or sha256_file(local_path) != expected:
            raise RuntimeError(f"Downloaded image-pose QC hash mismatch: {relative}")
        verified += 1
    return {"registered_file_count": len(lines), "verified_file_count": verified}


def sim3_exclusion_sensitivity(
    scene: str,
    excluded_image: str,
    scene_record: dict[str, Any],
    metadata_by_name: dict[str, dict[str, Any]],
    origin: list[float],
) -> dict[str, Any]:
    _, images = camera_indices(scene_record)
    names: list[str] = []
    source: list[np.ndarray] = []
    target: list[np.ndarray] = []
    for image_name, image in images.items():
        metadata = metadata_by_name[image_name]
        names.append(image_name)
        source.append(camera_center(image))
        target.append(
            _lla_to_local_enu(
                float(metadata["lat"]),
                float(metadata["lon"]),
                float(metadata["ellipsoid_alt_m"]),
                *origin,
            )
        )
    name_array = np.asarray(names)
    source_array = np.asarray(source, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    keep = name_array != excluded_image
    scale, rotation, translation = _estimate_similarity_umeyama(
        source_array[keep], target_array[keep]
    )
    transformed = scale * (rotation @ source_array.T).T + translation
    shifts = np.linalg.norm(transformed - source_array, axis=1)
    return {
        "matched_image_count": int(len(names)),
        "retained_image_count": int(np.count_nonzero(keep)),
        "excluded_image_count": int(np.count_nonzero(~keep)),
        "refit_scale": float(scale),
        "refit_rotation_deg": rotation_angle_deg(rotation),
        "refit_translation_m": translation.tolist(),
        "retained_camera_shift_median_m": float(np.median(shifts[keep])),
        "retained_camera_shift_p95_m": float(np.percentile(shifts[keep], 95)),
        "retained_camera_shift_max_m": float(np.max(shifts[keep])),
        "excluded_camera_shift_m": float(shifts[~keep][0]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--image_pose_qc_root",
        type=Path,
        default=Path(r"E:\M3M-GCP-3DGS\outputs\six_scene_image_pose_qc_20260716"),
    )
    parser.add_argument(
        "--remote_manifest",
        type=Path,
        default=Path(
            r"E:\M3M-GCP-3DGS\outputs\gcp_6scene_annotation_domain_inputs_20260628"
            r"\gcp_6scene_annotation_domain_jsonlight_20260628\remote_light_manifest.json"
        ),
    )
    parser.add_argument(
        "--candidate_root",
        type=Path,
        default=Path(r"E:\M3M-GCP-3DGS\outputs\gcp_annotation_candidates_20260617_all"),
    )
    parser.add_argument(
        "--training_cameras_json",
        type=Path,
        default=Path(
            r"E:\M3M-GCP-3DGS\outputs\gcp_metric_depth_regression_6scene_v1_2_2_consolidated_20260630_032936"
            r"\remote_model_provenance_901\gcp_50000_20260610\Model_RGB\cameras.json"
        ),
    )
    parser.add_argument(
        "--frozen_release_dir",
        type=Path,
        default=Path(r"E:\datasets\M3M-GCP\scenes\gcp_manual_annotations_v1_2_2"),
    )
    parser.add_argument("--output_root", type=Path)
    parser.add_argument("--stamp", default=time.strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    output_root = args.output_root or (
        args.repo / "outputs" / f"gcp_v13_training_image_exclusion_audit_{args.stamp}"
    )
    output_root.mkdir(parents=True, exist_ok=False)
    hash_verification = verify_remote_hash_manifest(args.image_pose_qc_root)
    six_scene = json.loads(
        (args.image_pose_qc_root / "six_scene_summary.json").read_text(encoding="utf-8")
    )
    if {row["scene"] for row in six_scene["scenes"]} != EXPECTED_SCENES:
        raise RuntimeError("Six-scene image-pose QC scene set mismatch")
    candidates = [
        candidate
        for scene in six_scene["scenes"]
        for candidate in scene["feature_pose_qc_candidates"]
    ]
    keys = {(row["scene"], row["image_name"]) for row in candidates}
    if keys != {EXPECTED_OUTLIER}:
        raise RuntimeError(f"Unexpected six-scene image-pose candidate set: {sorted(keys)}")

    scene, image_name = EXPECTED_OUTLIER
    manifest = json.loads(args.remote_manifest.read_text(encoding="utf-8"))
    alignment_path = (
        args.remote_manifest.parent
        / "models"
        / scene
        / "raw_model"
        / "georegistration_alignment_summary.json"
    )
    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    sensitivity = sim3_exclusion_sensitivity(
        scene,
        image_name,
        manifest["scenes"][scene],
        image_metadata(args.candidate_root, scene),
        alignment["enu_origin_lat_lon_alt"],
    )
    training_cameras = json.loads(args.training_cameras_json.read_text(encoding="utf-8"))
    training_matches = [row for row in training_cameras if row["img_name"] == image_name]
    if len(training_matches) != 1:
        raise RuntimeError("Outlier image is not uniquely present in training cameras.json")

    working_path = args.repo / ANNOTATION_RELATIVE_PATHS[scene]
    working = pd.read_csv(working_path, dtype=str, keep_default_na=False)
    affected_points = sorted(
        working.loc[working["image_name"].eq(image_name), "point_name"].unique()
    )
    image_path = Path(r"E:\datasets\M3M-GCP\scenes") / scene / image_name
    release_references = frozen_release_references(args.frozen_release_dir, image_name)
    if not release_references:
        raise RuntimeError("Expected v1.2.2 frozen-release references were not found")

    decision = {
        "scene": scene,
        "image_name": image_name,
        "raw_image_path": str(image_path),
        "raw_image_sha256": sha256_file(image_path),
        "affected_working_annotation_points": affected_points,
        "frozen_v1_2_2_reference_count": len(release_references),
        "frozen_v1_2_2_references": release_references,
        "raw_acquisition_action": "retain_byte_identical_because_frozen_release_references_it",
        "future_v1_3_formal_annotation_action": "exclude_all_observations_from_image",
        "future_canonical_training_view_action": "exclude_image_from_all_method_training_sets",
        "future_clean_source_view_action": "omit_from_manifest_driven_v1_3_source_view",
        "future_colmap_action": "create_and_freeze_canonical_source_model_without_image",
        "physical_file_deletion": False,
        "physical_deletion_reason": "would_break_frozen_v1_2_2_source_hash_and_camera_provenance",
        "selection_basis": "predeclared_image_level_feature_pose_qc_not_gcp_or_3dgs_residual",
        "training_camera_present_in_temporary_model": True,
        "training_camera_count": len(training_cameras),
        "excluded_fraction_of_50k_training_views": 1.0 / len(training_cameras),
        "gps_interpretation": (
            "gps_is_post_sfm_global_sim3_alignment_input; the image has millimetre-level global "
            "refit impact, while weak visual pose can still degrade local Gaussian supervision"
        ),
    }
    write_json(output_root / "training_image_exclusion_decision.json", decision)
    write_json(output_root / "gps_global_sim3_exclusion_sensitivity.json", sensitivity)
    pd.DataFrame(
        [
            {
                "scene": scene,
                "image_name": image_name,
                "raw_image_sha256": decision["raw_image_sha256"],
                "formal_v1_3_include": False,
                "training_v1_3_include": False,
                "raw_acquisition_delete": False,
                "reason": decision["selection_basis"],
            }
        ]
    ).to_csv(output_root / "v1_3_image_exclusion_manifest.csv", index=False)
    write_json(
        output_root / "audit_summary.json",
        {
            "schema": "ms_gcp_v13_training_image_exclusion_audit_v2",
            "status": "single_predeclared_image_exclusion_ready_for_v1_3",
            "six_scene_registered_image_count": six_scene["image_count"],
            "six_scene_feature_pose_qc_candidate_count": len(candidates),
            "candidate": decision,
            "gps_global_sim3_sensitivity": sensitivity,
            "downloaded_hash_verification": hash_verification,
            "gpu_used": False,
            "model_or_dataset_modified": False,
        },
    )
    print(output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
