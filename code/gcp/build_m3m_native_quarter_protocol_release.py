#!/usr/bin/env python3
"""Build the immutable overlay for the M3M-GCP native-quarter protocol.

The source image/camera release is read-only.  This program writes a sibling
protocol release containing the 82-instance disposition, observation semantics,
one common Sim(3) per scene, and leave-one-control-out evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from fit_gcp_sim3 import fit_similarity_umeyama
from m3m_native_quarter_protocol import (
    PIXEL_CONVENTION,
    PIXEL_DOMAIN,
    PROTOCOL_ID,
    PROTOCOL_RELEASE_SCHEMA,
    Sim3,
    canonical_json_sha256,
    residual_statistics,
    sha256_file,
)
from triangulate_gcp_points import (
    normalize_observation,
    observation_is_usable,
    project_point,
    qvec2rotmat,
    read_model,
    triangulate_point,
)


SOURCE_RELEASE_REL = Path("benchmark/source_release_v1_3_0")
ANNOTATION_RELEASE_REL = Path("benchmark/native_quarter_annotations_v1")
SPLIT_NAME = "gcp_control_checkpoint_split_v1_3_0.csv"
POINTS_NAME = "gcp_points_cgcs2000_cm108_v1_3_0.csv"
QC_COVERAGE_NAME = "annotation_qc_provenance/post_supplement_point_coverage.csv"
TARGET_FIELDS = (
    "cgcs2000_gk_cm108_e_m",
    "cgcs2000_gk_cm108_n_m",
    "cgcs2000_normal_height_m",
)
QUARANTINED_INSTANCES = {
    ("gcp_100000_20260610", "dxl3"),
    ("gcp_100000_20260610", "dyl2"),
    ("gcp_100000_20260610", "wy3_1"),
    ("gcp_100000_20260610", "wy3_2"),
    ("gcp_20000_20260602", "dyl2"),
}
ROOF_POINTS = {"dxl3", "dyl2", "wy3_1", "wy3_2"}


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


def target_map(rows: Sequence[dict[str, str]]) -> dict[str, np.ndarray]:
    return {
        row["point_name"]: np.asarray([float(row[field]) for field in TARGET_FIELDS], dtype=np.float64)
        for row in rows
    }


def transform_payload(sim3: Sim3) -> dict[str, Any]:
    return {
        "scale": float(sim3.scale),
        "rotation": sim3.rotation.tolist(),
        "translation": sim3.translation.tolist(),
        "definition": "target_xyz = scale * rotation @ frozen_colmap_model_xyz + translation",
    }


def camera_geometry(image: Any, sim3: Sim3) -> tuple[np.ndarray, np.ndarray, float]:
    rotation_world_to_camera = qvec2rotmat(image.qvec)
    centre_model = -rotation_world_to_camera.T @ image.tvec
    centre_target = sim3.apply(centre_model)
    optical_axis_model = rotation_world_to_camera.T @ np.asarray([0.0, 0.0, 1.0])
    optical_axis_target = sim3.rotate_direction(optical_axis_model)
    optical_axis_target /= np.linalg.norm(optical_axis_target)
    cosine = float(np.clip(np.dot(optical_axis_target, np.asarray([0.0, 0.0, -1.0])), -1.0, 1.0))
    off_nadir = math.degrees(math.acos(cosine))
    return centre_model, centre_target, off_nadir


def azimuth_and_bin(camera_target: np.ndarray, point_target: np.ndarray) -> tuple[float, int]:
    azimuth = (
        math.degrees(
            math.atan2(
                float(camera_target[0] - point_target[0]),
                float(camera_target[1] - point_target[1]),
            )
        )
        + 360.0
    ) % 360.0
    return azimuth, int(((azimuth + 22.5) % 360.0) // 45.0)


def triangulate_scene(
    scene: str,
    annotation_rows: Sequence[dict[str, str]],
    cameras: dict[int, Any],
    images_by_name: dict[str, Any],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows_by_point: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in annotation_rows:
        if not truthy(row.get("formal_eligible")):
            continue
        if not observation_is_usable(row, min_confidence=1.0):
            continue
        if row.get("image_name") not in images_by_name:
            raise RuntimeError(f"{scene}: annotation image absent from frozen model: {row.get('image_name')}")
        grouped[row["point_name"]].append(normalize_observation(row))
        rows_by_point[row["point_name"]].append(row)
    points: dict[str, np.ndarray] = {}
    point_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    for point_name in sorted(grouped):
        observations = sorted(grouped[point_name], key=lambda row: row["image_name"])
        if len(observations) < 2:
            raise RuntimeError(f"{scene}/{point_name}: fewer than two formal observations")
        xyz = triangulate_point(observations, cameras, images_by_name)
        points[point_name] = xyz
        errors: list[float] = []
        for observation in observations:
            image = images_by_name[observation["image_name"]]
            projected = project_point(cameras[image.camera_id], image, xyz)
            if projected is None:
                raise RuntimeError(f"{scene}/{point_name}: triangulated point projects behind camera")
            error = math.hypot(
                float(projected[0]) - float(observation["u_px"]),
                float(projected[1]) - float(observation["v_px"]),
            )
            errors.append(error)
            residual_rows.append(
                {
                    "scene": scene,
                    "point_name": point_name,
                    "image_name": observation["image_name"],
                    "u_px": observation["u_px"],
                    "v_px": observation["v_px"],
                    "reprojected_u_px": projected[0],
                    "reprojected_v_px": projected[1],
                    "reprojection_error_px": error,
                }
            )
        point_rows.append(
            {
                "scene": scene,
                "point_name": point_name,
                "model_x": xyz[0],
                "model_y": xyz[1],
                "model_z": xyz[2],
                "observation_count": len(observations),
                "mean_reprojection_error_px": float(np.mean(errors)),
                "median_reprojection_error_px": float(np.median(errors)),
                "max_reprojection_error_px": float(np.max(errors)),
                "used_image_names": ";".join(row["image_name"] for row in observations),
            }
        )
    return points, point_rows, residual_rows


def fit_scene_transform(
    scene: str,
    points: dict[str, np.ndarray],
    roles: dict[str, str],
    targets: dict[str, np.ndarray],
) -> tuple[Sim3, list[str], list[str]]:
    controls = sorted(name for name, role in roles.items() if role == "control")
    checkpoints = sorted(name for name, role in roles.items() if role == "checkpoint")
    missing = sorted(set(controls + checkpoints) - set(points))
    if missing:
        raise RuntimeError(f"{scene}: active split points did not triangulate: {missing}")
    if len(controls) < 4:
        raise RuntimeError(f"{scene}: fewer than four active controls")
    scale, rotation, translation = fit_similarity_umeyama(
        np.vstack([points[name] for name in controls]),
        np.vstack([targets[name] for name in controls]),
        estimate_scale=True,
    )
    return Sim3(scale, rotation, translation), controls, checkpoints


def sim3_residual_rows(
    scene: str,
    points: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    roles: dict[str, str],
    sim3: Sim3,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    residuals: dict[str, list[np.ndarray]] = {"control": [], "checkpoint": [], "all": []}
    for name in sorted(roles):
        predicted = sim3.apply(points[name])
        residual = predicted - targets[name]
        role = roles[name]
        residuals[role].append(residual)
        residuals["all"].append(residual)
        rows.append(
            {
                "scene": scene,
                "point_name": name,
                "role": role,
                "model_x": points[name][0],
                "model_y": points[name][1],
                "model_z": points[name][2],
                "target_e_m": targets[name][0],
                "target_n_m": targets[name][1],
                "target_z_m": targets[name][2],
                "predicted_e_m": predicted[0],
                "predicted_n_m": predicted[1],
                "predicted_z_m": predicted[2],
                "residual_e_m": residual[0],
                "residual_n_m": residual[1],
                "residual_z_m": residual[2],
                "error_h_m": float(np.linalg.norm(residual[:2])),
                "error_z_m": float(abs(residual[2])),
                "error_3d_m": float(np.linalg.norm(residual)),
            }
        )
    return rows, {role: residual_statistics(values) for role, values in residuals.items()}


def jackknife_rows(
    scene: str,
    points: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    controls: Sequence[str],
    full_sim3: Sim3,
    images_by_name: dict[str, Any],
) -> list[dict[str, Any]]:
    camera_centres = []
    for image in images_by_name.values():
        rotation = qvec2rotmat(image.qvec)
        camera_centres.append(-rotation.T @ image.tvec)
    centres = np.vstack(camera_centres)
    full_centres = full_sim3.apply(centres)
    rows: list[dict[str, Any]] = []
    for omitted in sorted(controls):
        retained = [name for name in controls if name != omitted]
        scale, rotation, translation = fit_similarity_umeyama(
            np.vstack([points[name] for name in retained]),
            np.vstack([targets[name] for name in retained]),
            estimate_scale=True,
        )
        loo = Sim3(scale, rotation, translation)
        prediction = loo.apply(points[omitted])
        residual = prediction - targets[omitted]
        centre_shift = loo.apply(centres) - full_centres
        horizontal = np.linalg.norm(centre_shift[:, :2], axis=1)
        vertical = np.abs(centre_shift[:, 2])
        distance = np.linalg.norm(centre_shift, axis=1)
        rows.append(
            {
                "scene": scene,
                "omitted_control": omitted,
                "retained_control_count": len(retained),
                "omitted_prediction_error_h_m": float(np.linalg.norm(residual[:2])),
                "omitted_prediction_error_z_m": float(abs(residual[2])),
                "omitted_prediction_error_3d_m": float(np.linalg.norm(residual)),
                "camera_center_shift_max_h_m": float(np.max(horizontal)),
                "camera_center_shift_max_z_m": float(np.max(vertical)),
                "camera_center_shift_max_3d_m": float(np.max(distance)),
                "camera_center_shift_p95_3d_m": float(np.percentile(distance, 95)),
                "loo_scale": float(scale),
                "loo_rotation_json": json.dumps(rotation.tolist(), separators=(",", ":")),
                "loo_translation_json": json.dumps(translation.tolist(), separators=(",", ":")),
            }
        )
    return rows


def disposition_rows(split_rows: Sequence[dict[str, str]], split_sha: str) -> list[dict[str, Any]]:
    output = []
    for row in sorted(split_rows, key=lambda value: (value["scene"], value["point_name"])):
        key = (row["scene"], row["point_name"])
        quarantined = key in QUARANTINED_INSTANCES
        point_name = row["point_name"]
        if point_name == "dxl3":
            reason = "roof anchor identity is not independently verifiable; observed vertical discrepancy is about 0.219 m"
        elif quarantined:
            reason = "multiview image anchor is internally consistent but disagrees with the surveyed roof coordinate by about 3.5-4.1 m"
        else:
            reason = "retained after native-quarter multiview geometry audit"
        output.append(
            {
                "scene": row["scene"],
                "point_name": point_name,
                "source_role": row["role"],
                "active_role": "diagnostic_only" if quarantined else row["role"],
                "active_formal_eligible": str(not quarantined).lower(),
                "audit_disposition": "quarantined_anchor_binding" if quarantined else "formal_primary",
                "surface_level": "roof" if point_name in ROOF_POINTS else "ground",
                "surface_semantic": (
                    "roof_texture_anchor_identity_unverified"
                    if point_name in ROOF_POINTS
                    else "surveyed_ground_anchor"
                ),
                "anchor_definition": (
                    "manual image anchor cannot currently be proven identical to surveyed roof target"
                    if point_name in ROOF_POINTS
                    else "survey coordinate bound to the manually annotated ground target"
                ),
                "surface_boundary_class": "not_machine_verified",
                "occlusion_class": "formal_good_visible_annotation_no_additional_occlusion_label",
                "audit_reason": reason,
                "source_split_sha256": split_sha,
            }
        )
    return output


def validate_resumable_incomplete_output(out_dir: Path) -> None:
    if (out_dir / "protocol_release_manifest.json").exists() or (out_dir / "SHA256SUMS.txt").exists():
        raise FileExistsError("refusing to resume an output that contains completion artifacts")
    allowed_root_files = {"point_instance_disposition.csv", "observation_semantics.csv", "README.md"}
    allowed_scene_files = {
        "triangulated_model_points.csv",
        "triangulation_observation_residuals.csv",
        "common_sim3_residuals.csv",
        "control_leave_one_out.csv",
        "common_sim3.json",
    }
    unexpected = []
    for path in out_dir.rglob("*"):
        if path.is_dir():
            continue
        relative = path.relative_to(out_dir)
        valid = (
            len(relative.parts) == 1 and relative.name in allowed_root_files
        ) or (
            len(relative.parts) == 3
            and relative.parts[0] == "scenes"
            and relative.name in allowed_scene_files
        )
        if not valid:
            unexpected.append(relative.as_posix())
    if unexpected:
        raise FileExistsError(f"incomplete output contains unexpected files: {unexpected}")


def build_release(
    data_root: Path,
    out_dir: Path,
    release_date: str,
    resume_incomplete: bool = False,
) -> dict[str, Any]:
    data_root = data_root.resolve()
    out_dir = out_dir.resolve()
    if out_dir.exists():
        if not resume_incomplete:
            raise FileExistsError(f"refusing to overwrite existing protocol release: {out_dir}")
        validate_resumable_incomplete_output(out_dir)
    data_contract_path = data_root / "DATA_CONTRACT_DRAFT.json"
    source_root = data_root / SOURCE_RELEASE_REL
    annotation_root = data_root / ANNOTATION_RELEASE_REL
    split_path = source_root / SPLIT_NAME
    points_path = source_root / POINTS_NAME
    qc_path = source_root / QC_COVERAGE_NAME
    for required in (data_contract_path, split_path, points_path, qc_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    data_contract = json.loads(data_contract_path.read_text(encoding="utf-8"))
    if data_contract.get("contract_id") != "m3m_gcp_colmap_native_quarter_max1414_v1":
        raise RuntimeError("unexpected native-quarter data contract")
    split_rows = read_csv(split_path)
    if len(split_rows) != 87:
        raise RuntimeError(f"expected 87 source split instances, got {len(split_rows)}")
    if {row["role"] for row in split_rows} != {"control", "checkpoint"}:
        raise RuntimeError("unexpected split roles")
    points_rows = read_csv(points_path)
    targets = target_map(points_rows)
    qc = {(row["scene"], row["point_name"]): row for row in read_csv(qc_path)}
    split_sha = sha256_file(split_path)
    dispositions = disposition_rows(split_rows, split_sha)
    active_dispositions = [row for row in dispositions if truthy(row["active_formal_eligible"])]
    if len(active_dispositions) != 82:
        raise RuntimeError(f"active disposition count is not 82: {len(active_dispositions)}")
    active_role_counts = {
        role: sum(row["active_role"] == role for row in active_dispositions)
        for role in ("control", "checkpoint")
    }
    if active_role_counts != {"control": 45, "checkpoint": 37}:
        raise RuntimeError(f"unexpected active role counts: {active_role_counts}")

    out_dir.mkdir(parents=True, exist_ok=resume_incomplete)
    write_csv(
        out_dir / "point_instance_disposition.csv",
        dispositions,
        [
            "scene",
            "point_name",
            "source_role",
            "active_role",
            "active_formal_eligible",
            "audit_disposition",
            "surface_level",
            "surface_semantic",
            "anchor_definition",
            "surface_boundary_class",
            "occlusion_class",
            "audit_reason",
            "source_split_sha256",
        ],
    )

    observation_semantics: list[dict[str, Any]] = []
    scene_summaries = []
    scenes = sorted({row["scene"] for row in split_rows})
    for scene in scenes:
        annotation_path = annotation_root / f"{scene}_gcp_annotations_native_quarter_v1.csv"
        annotation_sha = sha256_file(annotation_path)
        annotation_rows = read_csv(annotation_path)
        cameras, images, _points3d = read_model(data_root / scene / "sparse/0")
        images_by_name = {image.name: image for image in images.values()}
        model_points, triangulated_rows, triangulation_residual_rows = triangulate_scene(
            scene, annotation_rows, cameras, images_by_name
        )
        active_scene = [row for row in active_dispositions if row["scene"] == scene]
        roles = {row["point_name"]: row["active_role"] for row in active_scene}
        full_sim3, controls, checkpoints = fit_scene_transform(scene, model_points, roles, targets)
        residual_rows, stats = sim3_residual_rows(scene, model_points, targets, roles, full_sim3)
        loo_rows = jackknife_rows(scene, model_points, targets, controls, full_sim3, images_by_name)
        scene_dir = out_dir / "scenes" / scene
        write_csv(
            scene_dir / "triangulated_model_points.csv",
            triangulated_rows,
            [
                "scene",
                "point_name",
                "model_x",
                "model_y",
                "model_z",
                "observation_count",
                "mean_reprojection_error_px",
                "median_reprojection_error_px",
                "max_reprojection_error_px",
                "used_image_names",
            ],
        )
        write_csv(
            scene_dir / "triangulation_observation_residuals.csv",
            triangulation_residual_rows,
            [
                "scene",
                "point_name",
                "image_name",
                "u_px",
                "v_px",
                "reprojected_u_px",
                "reprojected_v_px",
                "reprojection_error_px",
            ],
        )
        write_csv(
            scene_dir / "common_sim3_residuals.csv",
            residual_rows,
            [
                "scene",
                "point_name",
                "role",
                "model_x",
                "model_y",
                "model_z",
                "target_e_m",
                "target_n_m",
                "target_z_m",
                "predicted_e_m",
                "predicted_n_m",
                "predicted_z_m",
                "residual_e_m",
                "residual_n_m",
                "residual_z_m",
                "error_h_m",
                "error_z_m",
                "error_3d_m",
            ],
        )
        write_csv(
            scene_dir / "control_leave_one_out.csv",
            loo_rows,
            [
                "scene",
                "omitted_control",
                "retained_control_count",
                "omitted_prediction_error_h_m",
                "omitted_prediction_error_z_m",
                "omitted_prediction_error_3d_m",
                "camera_center_shift_max_h_m",
                "camera_center_shift_max_z_m",
                "camera_center_shift_max_3d_m",
                "camera_center_shift_p95_3d_m",
                "loo_scale",
                "loo_rotation_json",
                "loo_translation_json",
            ],
        )

        transform = {
            "schema": "m3m_gcp_native_quarter_common_sim3_v1",
            "protocol_id": PROTOCOL_ID,
            "scene": scene,
            "source_data_release_root_digest_sha256": data_contract["release_root_digest_sha256"],
            "annotation_csv_relative_path": annotation_path.relative_to(data_root).as_posix(),
            "annotation_csv_sha256": annotation_sha,
            "control_points": controls,
            "checkpoint_points": checkpoints,
            "control_count": len(controls),
            "checkpoint_count": len(checkpoints),
            "transform": transform_payload(full_sim3),
            "baseline_residual_statistics": stats,
            "leave_one_out": {
                "definition_omitted_control_prediction": "error at the omitted surveyed control after refitting on all other controls",
                "definition_camera_center_shift": "difference between leave-one-out and full transforms evaluated on every frozen COLMAP camera centre",
                "max_omitted_prediction_error_h_m": max(row["omitted_prediction_error_h_m"] for row in loo_rows),
                "max_omitted_prediction_error_z_m": max(row["omitted_prediction_error_z_m"] for row in loo_rows),
                "max_camera_center_shift_h_m": max(row["camera_center_shift_max_h_m"] for row in loo_rows),
                "max_camera_center_shift_z_m": max(row["camera_center_shift_max_z_m"] for row in loo_rows),
                "rows_relative_path": "control_leave_one_out.csv",
            },
            "method_result_refit_forbidden": True,
        }
        transform["transform_canonical_sha256"] = canonical_json_sha256(transform["transform"])
        write_json(scene_dir / "common_sim3.json", transform)

        point_expected_counts: dict[str, int] = defaultdict(int)
        scene_observations: list[dict[str, Any]] = []
        dispositions_by_point = {
            row["point_name"]: row for row in dispositions if row["scene"] == scene
        }
        for row in annotation_rows:
            point_name = row.get("point_name", "")
            if point_name not in dispositions_by_point or not truthy(row.get("formal_eligible")):
                continue
            if not observation_is_usable(row, min_confidence=1.0):
                continue
            image = images_by_name[row["image_name"]]
            centre_model, _centre_target, off_nadir = camera_geometry(image, full_sim3)
            view_class = "nadir" if off_nadir <= 5.0 else "oblique"
            # Match the frozen v1.3 QC definition exactly: camera position and
            # triangulated point are both evaluated in the COLMAP model frame.
            # A similarity transform preserves the angle, while substituting a
            # surveyed target can move a value across a 45-degree bin boundary.
            azimuth, azimuth_bin = azimuth_and_bin(centre_model, model_points[point_name])
            u = float(row["u_px"])
            v = float(row["v_px"])
            width = int(row["target_width"])
            height = int(row["target_height"])
            safe_stencil = 0.0 <= u < width - 1 and 0.0 <= v < height - 1
            if not safe_stencil:
                raise RuntimeError(f"{scene}/{point_name}/{row['image_name']}: unsafe bilinear stencil")
            disposition = dispositions_by_point[point_name]
            point_expected_counts[point_name] += 1
            scene_observations.append(
                {
                    "observation_id": row["observation_id"],
                    "scene": scene,
                    "point_name": point_name,
                    "image_name": row["image_name"],
                    "u_px": row["u_px"],
                    "v_px": row["v_px"],
                    "target_width": width,
                    "target_height": height,
                    "pixel_domain": row["pixel_domain"],
                    "pixel_convention": row["pixel_convention"],
                    "source_role": disposition["source_role"],
                    "active_role": disposition["active_role"],
                    "active_formal_eligible": disposition["active_formal_eligible"],
                    "audit_disposition": disposition["audit_disposition"],
                    "surface_level": disposition["surface_level"],
                    "surface_semantic": disposition["surface_semantic"],
                    "surface_boundary_class": disposition["surface_boundary_class"],
                    "occlusion_class": disposition["occlusion_class"],
                    "view_class": view_class,
                    "off_nadir_deg": off_nadir,
                    "camera_position_azimuth_deg": azimuth,
                    "azimuth_bin_45deg": azimuth_bin,
                    "safe_bilinear_stencil": "true",
                    "native_image_sha256": row["native_image_sha256"],
                    "source_annotation_csv_sha256": annotation_sha,
                }
            )
        for item in scene_observations:
            item["expected_formal_observation_count"] = point_expected_counts[item["point_name"]]
        observation_semantics.extend(scene_observations)

        for point_name in sorted(roles):
            rows_for_point = [
                row for row in scene_observations
                if row["point_name"] == point_name and truthy(row["active_formal_eligible"])
            ]
            view_counts = {
                "total": len(rows_for_point),
                "nadir": sum(row["view_class"] == "nadir" for row in rows_for_point),
                "oblique": sum(row["view_class"] == "oblique" for row in rows_for_point),
                "azimuth_bins": len({row["azimuth_bin_45deg"] for row in rows_for_point}),
            }
            expected = qc[(scene, point_name)]
            expected_counts = {
                "total": int(expected["future_formal_good_count"]),
                "nadir": int(expected["future_formal_nadir_good_count"]),
                "oblique": int(expected["future_formal_oblique_good_count"]),
                "azimuth_bins": int(expected["future_formal_azimuth_bin_count"]),
            }
            if view_counts != expected_counts:
                raise RuntimeError(
                    f"{scene}/{point_name}: derived view semantics disagree with frozen QC: "
                    f"{view_counts} versus {expected_counts}"
                )

        scene_summaries.append(
            {
                "scene": scene,
                "source_instances": sum(row["scene"] == scene for row in dispositions),
                "active_instances": len(active_scene),
                "active_controls": len(controls),
                "active_checkpoints": len(checkpoints),
                "active_observations": sum(
                    truthy(row["active_formal_eligible"]) for row in scene_observations
                ),
                "registered_images": len(images_by_name),
                "common_sim3_canonical_sha256": transform["transform_canonical_sha256"],
                "checkpoint_baseline": stats["checkpoint"],
                "loo_max_omitted_prediction_z_m": transform["leave_one_out"]["max_omitted_prediction_error_z_m"],
                "loo_max_camera_center_shift_z_m": transform["leave_one_out"]["max_camera_center_shift_z_m"],
            }
        )

    write_csv(
        out_dir / "observation_semantics.csv",
        observation_semantics,
        [
            "observation_id",
            "scene",
            "point_name",
            "image_name",
            "u_px",
            "v_px",
            "target_width",
            "target_height",
            "pixel_domain",
            "pixel_convention",
            "source_role",
            "active_role",
            "active_formal_eligible",
            "audit_disposition",
            "surface_level",
            "surface_semantic",
            "surface_boundary_class",
            "occlusion_class",
            "view_class",
            "off_nadir_deg",
            "camera_position_azimuth_deg",
            "azimuth_bin_45deg",
            "safe_bilinear_stencil",
            "expected_formal_observation_count",
            "native_image_sha256",
            "source_annotation_csv_sha256",
        ],
    )
    active_observations = [row for row in observation_semantics if truthy(row["active_formal_eligible"])]
    if any(row["pixel_domain"] != PIXEL_DOMAIN for row in active_observations):
        raise RuntimeError("active observation pixel-domain mismatch")
    if any(row["pixel_convention"] != PIXEL_CONVENTION for row in active_observations):
        raise RuntimeError("active observation pixel-convention mismatch")

    readme = "\n".join(
        [
            "# M3M-GCP native-quarter protocol overlay v1",
            "",
            "This directory is an immutable evaluation overlay for the sibling",
            "`M3M-GCP-colmap-native-quarter-v1` image/camera release. It does not",
            "duplicate or modify images, cameras, survey coordinates, or v1.3.0 files.",
            "",
            "- Active formal set: 82 scene-point instances (45 controls, 37 checkpoints).",
            "- Quarantine: five roof scene-point instances are diagnostic only.",
            "- Registration: one frozen, ground-control Sim(3) per scene; method-specific refit is forbidden.",
            "- Common primary sample: `bilinear(M1) / bilinear(A)` at floating zero-based pixel centres.",
            "- Training remains locked until an individual method recipe and adapter pass qualification.",
            "",
        ]
    )
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    payload_paths = sorted(
        path for path in out_dir.rglob("*")
        if path.is_file() and path.name not in {"protocol_release_manifest.json", "SHA256SUMS.txt"}
    )
    payload_files = [
        {
            "path": path.relative_to(out_dir).as_posix(),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in payload_paths
    ]
    payload_digest = canonical_json_sha256(payload_files)
    manifest = {
        "schema": PROTOCOL_RELEASE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "release_date": release_date,
        "status": "active_evaluation_contract_training_hold_pending_per_method_qualification",
        "source_data": {
            "directory_name": data_root.name,
            "contract_id": data_contract["contract_id"],
            "release_root_digest_sha256": data_contract["release_root_digest_sha256"],
            "data_contract_sha256": sha256_file(data_contract_path),
            "source_split_relative_path": (SOURCE_RELEASE_REL / SPLIT_NAME).as_posix(),
            "source_split_sha256": split_sha,
            "source_points_relative_path": (SOURCE_RELEASE_REL / POINTS_NAME).as_posix(),
            "source_points_sha256": sha256_file(points_path),
        },
        "pixel_contract": {
            "pixel_domain": PIXEL_DOMAIN,
            "pixel_convention": PIXEL_CONVENTION,
            "subpixel_operator": "bilinear(weighted_camera_z_sum) / bilinear(accumulated_alpha)",
            "support_gate": "bilinear(accumulated_alpha) > 1e-6",
            "normalization_epsilon_used": False,
            "out_of_bounds_policy": "invalid_no_padding_no_clamping",
        },
        "aggregation_contract": {
            "operator": "geometric median within each (view_class, 45-degree azimuth bin), then geometric median across groups",
            "minimum_valid": "max(4, ceil(0.5 * expected_formal_observation_count))",
            "minimum_valid_nadir": 2,
            "minimum_valid_oblique": 2,
        },
        "surface_tracks": {
            "common_primary": "render-support expected camera-z coordinate (M1/A); cross-method formal ranking",
            "native_surface_secondary": "method-family native surface; separately reported after adapter freeze",
            "z50_and_mesh": "diagnostic until contribution-order or meshing parity is demonstrated",
            "formal_roof_claim_available": False,
        },
        "counts": {
            "scene_count": len(scenes),
            "source_scene_point_instances": len(dispositions),
            "active_scene_point_instances": len(active_dispositions),
            "active_controls": active_role_counts["control"],
            "active_checkpoints": active_role_counts["checkpoint"],
            "quarantined_scene_point_instances": len(QUARANTINED_INSTANCES),
            "active_observations": len(active_observations),
        },
        "scene_summaries": scene_summaries,
        "method_result_sim3_refit_allowed": False,
        "training_allowed_globally": False,
        "training_unlock_policy": "per-method 3K qualification only after frozen recipe, raw-moment adapter, synthetic conformance, and CPU preflight pass",
        "payload_files": payload_files,
        "payload_manifest_canonical_sha256": payload_digest,
    }
    write_json(out_dir / "protocol_release_manifest.json", manifest)
    sum_paths = sorted(path for path in out_dir.rglob("*") if path.is_file() and path.name != "SHA256SUMS.txt")
    sums = "".join(
        f"{sha256_file(path)}  {path.relative_to(out_dir).as_posix()}\n" for path in sum_paths
    )
    (out_dir / "SHA256SUMS.txt").write_text(sums, encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--release_date", default="2026-08-07")
    parser.add_argument(
        "--resume_incomplete",
        action="store_true",
        help="Resume only a manifest-free partial directory containing known generator outputs.",
    )
    args = parser.parse_args()
    manifest = build_release(
        args.data_root,
        args.out_dir,
        args.release_date,
        resume_incomplete=bool(args.resume_incomplete),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
