"""Generate the transactional MS-GCP v1.3.0 control-heavy release.

This program only reads source annotations, source images, camera manifests,
and audited RTK data.  All derived files are written to a unique staging
directory and the formal release appears only through a final atomic rename.
"""

from __future__ import annotations

import argparse
import json
import math
import stat
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gcp_pixel_domain_v1_2 import (  # noqa: E402
    CACHED_TARGET_TOL_PX,
    ORIENTATION_POLICY,
    PIXEL_CONVENTION,
    ROUNDTRIP_TOL_PX,
    SCENES,
    SOURCE_PIXEL_DOMAIN,
    TARGET_PIXEL_DOMAIN,
    TRANSFORM_VERSION,
    camera_record_hash,
    canonical_record_sha256,
    canonical_records_root_sha256,
    file_sha256,
    fmt_float,
    generator_provenance,
    image_pose_record_hash,
    load_manifest_model,
    load_raw_image_orientation_record,
    payload_manifest_entries,
    payload_root_digest,
    pose_equivalence,
    raw_to_target_projection,
    read_csv,
    verify_payload_integrity,
    write_csv_deterministic,
    write_json_deterministic,
)
from gcp_pixel_domain_v1_3 import (  # noqa: E402
    PROJECTION_STATUS_DIAGNOSTIC_OOB,
    PROJECTION_STATUS_NO_CLICK,
    PROJECTION_STATUS_VALID,
    RELEASE_V130_ID,
    RELEASE_V130_SCHEMA,
    RELEASE_V130_TOKEN,
    observation_id_from_fields_v13,
)
from generate_gcp_release_v1_2 import (  # noqa: E402
    compact_model,
    compare_dirs,
    make_unique_staging,
    mapping_primary_key,
    mapping_record,
    model_file,
    percentile,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = Path(r"E:\datasets\M3M-GCP")
DEFAULT_PROJECT_ROOT = Path(r"E:\M3M-GCP-3DGS")
DEFAULT_INPUT_MANIFEST = REPO_ROOT / "configs" / "gcp_v13_release_inputs_v1.json"
DEFAULT_FINAL_DIR = DEFAULT_DATASET_ROOT / "scenes" / "gcp_manual_annotations_v1_3_0"

EXPECTED_COUNTS = {
    "row_count": 1383,
    "annotation_good_count": 1155,
    "formal_eligible_count": 1069,
    "coordinate_row_count": 1213,
    "no_coordinate_row_count": 170,
    "formal_split_scene_point_count": 87,
    "formal_split_unique_point_count": 50,
    "all_annotation_unique_point_count": 53,
    "annotated_image_count": 951,
    "training_view_count": 6187,
    "v1_2_2_preserved_observation_count": 611,
    "diagnostic_projection_out_of_bounds_count": 2,
}

OBS_FIELDS = [
    "observation_id",
    "scene",
    "point_name",
    "raw_image_name",
    "raw_manual_x",
    "raw_manual_y",
    "annotation_visible",
    "annotation_quality",
    "annotation_confidence",
    "annotation_annotator",
    "annotation_note",
    "annotation_updated_at",
    "annotation_good",
    "formal_eligible",
    "formal_role",
    "projection_status",
    "raw_coordinate_in_bounds",
    "target_in_bounds",
    "source_annotation_schema",
    "source_annotation_file_sha256",
    "source_annotation_row_number",
    "source_pixel_domain",
    "source_pixel_convention",
    "source_image_width",
    "source_image_height",
    "source_image_sha256",
    "source_exif_orientation_raw_value",
    "source_orientation_policy",
    "source_rgb_pixel_matrix_sha256",
    "source_camera_id",
    "source_camera_model",
    "source_camera_width",
    "source_camera_height",
    "source_camera_params",
    "source_camera_record_sha256",
    "source_pose_record_sha256",
    "source_cameras_bin_sha256",
    "source_images_bin_sha256",
    "normalized_x",
    "normalized_y",
    "normalized_unit_ray_x",
    "normalized_unit_ray_y",
    "normalized_unit_ray_z",
    "target_image_name",
    "target_pixel_domain",
    "target_pixel_convention",
    "target_image_width",
    "target_image_height",
    "target_image_sha256",
    "target_camera_id",
    "target_camera_model",
    "target_camera_width",
    "target_camera_height",
    "target_camera_params",
    "target_camera_record_sha256",
    "target_pose_record_sha256",
    "target_cameras_bin_sha256",
    "target_images_bin_sha256",
    "target_x",
    "target_y",
    "mapping_type",
    "transform_version",
    "source_target_mapping_record_sha256",
    "roundtrip_raw_x",
    "roundtrip_raw_y",
    "roundtrip_error_px",
]

POINT_EXTRA_FIELDS = [
    "formal_primary_eligible_v1_3_0",
    "v1_3_coordinate_audit_status",
    "rtk_authoritative_source_sha256",
    "rtk_quality_record_sha256",
]


def load_input_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "ms_gcp_v1_3_0_release_input_manifest_v1":
        raise ValueError(f"unexpected input manifest schema: {payload.get('schema')}")
    if set(payload.get("working_annotations", {})) != set(SCENES):
        raise ValueError("input manifest must provide exactly six working annotation files")
    return payload


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")
    return path


def require_dir(path: Path, label: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} missing: {path}")
    return path


def copy_file_verified(source: Path, destination: Path) -> dict[str, Any]:
    require_file(source, "byte-copy source")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    source_sha = file_sha256(source)
    target_sha = file_sha256(destination)
    if source_sha != target_sha or source.stat().st_size != destination.stat().st_size:
        raise ValueError(f"byte-copy verification failed: {source} -> {destination}")
    return {
        "source_path": str(source),
        "release_relative_path": destination.as_posix(),
        "bytes": source.stat().st_size,
        "sha256": source_sha,
        "byte_equal": True,
    }


def remove_tree_readonly(path: Path) -> None:
    """Remove a generated staging tree even when copied evidence is read-only."""

    if not path.exists():
        return
    for item in sorted(path.rglob("*"), key=lambda candidate: len(candidate.parts), reverse=True):
        try:
            item.chmod(stat.S_IREAD | stat.S_IWRITE)
        except OSError:
            pass
    path.chmod(stat.S_IREAD | stat.S_IWRITE)
    shutil.rmtree(path)


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def canonical_quality(row: dict[str, str]) -> tuple[bool, str]:
    quality = str(row.get("quality", "")).strip().lower()
    visible_text = str(row.get("visible", "")).strip().lower()
    if quality not in {"good", "ambiguous", "not_visible"}:
        raise ValueError(f"unsupported annotation quality: {quality!r}")
    if visible_text not in {"0", "1", "false", "true"}:
        raise ValueError(f"unsupported annotation visibility: {visible_text!r}")
    visible = visible_text in {"1", "true"}
    if quality == "not_visible" and visible:
        raise ValueError("not_visible row cannot be marked visible")
    if quality in {"good", "ambiguous"} and not visible:
        raise ValueError(f"{quality} row must be marked visible")
    return visible, quality


def load_split_rows(path: Path) -> tuple[list[dict[str, str]], dict[tuple[str, str], str]]:
    rows = read_csv(require_file(path, "geometry-only split candidate"))
    if len(rows) != EXPECTED_COUNTS["formal_split_scene_point_count"]:
        raise ValueError(f"unexpected split row count: {len(rows)}")
    roles: dict[tuple[str, str], str] = {}
    for row in rows:
        key = (row["scene"], row["point_name"])
        if key in roles:
            raise ValueError(f"duplicate split row: {key}")
        role = row["role"].strip().lower()
        if role not in {"control", "checkpoint"}:
            raise ValueError(f"invalid split role: {key}={role}")
        roles[key] = role
    if len({point for _, point in roles}) != EXPECTED_COUNTS["formal_split_unique_point_count"]:
        raise ValueError("unexpected unique formal point count")
    return rows, roles


def write_frozen_split(staging: Path, source_path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "scene",
        "point_name",
        "role",
        "cgcs2000_gk_cm108_e_m",
        "cgcs2000_gk_cm108_n_m",
        "cgcs2000_normal_height_m",
        "good_view_count",
        "coordinate_status",
        "control_eligible",
        "split_status",
        "role_reason",
    ]
    out = []
    for row in rows:
        item = {field: row.get(field, "") for field in fields}
        item["split_status"] = "frozen_release_v1_3_0"
        out.append(item)
    write_csv_deterministic(staging / "gcp_control_checkpoint_split_v1_3_0.csv", out, fields)
    write_json_deterministic(
        staging / "split_selection_provenance_v1_3_0.json",
        {
            "schema": "ms_gcp_geometry_only_split_provenance_v1_3_0",
            "source_candidate_path": str(source_path),
            "source_candidate_sha256": file_sha256(source_path),
            "scene_point_count": len(rows),
            "unique_point_count": len({row["point_name"] for row in rows}),
            "selection_inputs_allowed": [
                "surveyed XYZ",
                "annotation-side usable count",
                "view diversity",
                "visibility/QC",
                "scene boundary",
            ],
            "selection_inputs_forbidden": [
                "residual",
                "RMSE",
                "depth",
                "alpha",
                "variance",
                "multiview scatter",
                "model-error classification",
            ],
            "image_exclusion_count": 0,
        },
    )


def build_point_table(
    *,
    release_v122: Path,
    split_rows: list[dict[str, str]],
    annotation_points: set[str],
    review_table_path: Path,
    quality_summary_path: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    old_path = release_v122 / "gcp_points_primary_usable_cgcs2000_cm108_v1.csv"
    old_rows = read_csv(require_file(old_path, "v1.2.2 point table"))
    old_by_name = {row["point_name"]: row for row in old_rows}
    review_by_name = {
        row["point_name"]: row for row in read_csv(require_file(review_table_path, "review-only coordinate table"))
    }
    quality_by_name = {
        row["point_name"]: row for row in read_csv(require_file(quality_summary_path, "RTK quality summary"))
    }
    split_coords: dict[str, tuple[str, str, str]] = {}
    for row in split_rows:
        value = (
            row["cgcs2000_gk_cm108_e_m"],
            row["cgcs2000_gk_cm108_n_m"],
            row["cgcs2000_normal_height_m"],
        )
        previous = split_coords.setdefault(row["point_name"], value)
        if previous != value:
            raise ValueError(f"conflicting split coordinates for {row['point_name']}")
    fieldnames = list(old_rows[0].keys()) + [field for field in POINT_EXTRA_FIELDS if field not in old_rows[0]]
    formal_points = set(split_coords)
    if annotation_points - (set(old_by_name) | set(review_by_name) | formal_points):
        raise ValueError(f"missing surveyed coordinates for points: {sorted(annotation_points - (set(old_by_name) | set(review_by_name) | formal_points))}")

    result: list[dict[str, Any]] = []
    for point in sorted(annotation_points):
        base = dict(old_by_name.get(point, review_by_name.get(point, {})))
        base["point_name"] = point
        if point in split_coords:
            e, n, h = split_coords[point]
            base["cgcs2000_gk_cm108_e_m"] = e
            base["cgcs2000_gk_cm108_n_m"] = n
            base["cgcs2000_normal_height_m"] = h
        base.setdefault("horizontal_crs_epsg", "4545")
        if point in {"G07", "G09", "G39"}:
            quality = quality_by_name.get(point)
            if quality is None or int(float(quality["fixed_solution_count"])) != 27:
                raise ValueError(f"authoritative 27-fixed-epoch RTK evidence missing for {point}")
            base.update(
                {
                    "open_dataset_use": "primary_usable_v1_3_0",
                    "point_category": "rtk_surveyed_gcp",
                    "quality_evaluation": "27_fixed_epoch_rtk_observation_audited",
                    "observation_count": quality["observation_count"],
                    "plane_dispersion_range_m": quality["horizontal_pairwise_range_m"],
                    "height_dispersion_range_m": quality["height_range_m"],
                    "mean_pdop": quality["pdop_median"],
                    "min_satellites": quality["satellite_min"],
                    "max_satellites": quality["satellite_max"],
                    "coordinate_source": "corrected_RTK_authoritative_package",
                    "rtk_authoritative_source_sha256": file_sha256(quality_summary_path),
                    "rtk_quality_record_sha256": canonical_record_sha256(quality),
                }
            )
        if point not in formal_points:
            base["open_dataset_use"] = "diagnostic_only_v1_3_0"
        base["formal_primary_eligible_v1_3_0"] = bool_text(point in formal_points)
        base["v1_3_coordinate_audit_status"] = (
            "authoritative_corrected_rtk_revalidated" if point in {"G07", "G09", "G39"} else "preserved_from_accepted_v1_2_2_point_table"
        )
        result.append({field: base.get(field, "") for field in fieldnames})
    return result, fieldnames


def full_camera_model_payload(model: dict[str, Any]) -> dict[str, Any]:
    names = {Path(str(row["image_name"])).name for row in model.get("images", [])}
    payload = compact_model(model, names)
    payload["all_model_file_records"] = [
        {key: row.get(key, "") for key in ["name", "path", "bytes", "sha256"]}
        for row in model.get("files", [])
    ]
    return payload


def source_image_path(dataset_root: Path, scene: str, image_name: str) -> Path:
    return dataset_root / "scenes" / scene / image_name


def write_protocol_doc(path: Path) -> None:
    path.write_text(
        """# MS-GCP v1.3.0 Multi-View Control-Heavy Protocol

Release v1.3.0 preserves all 1,383 reviewed annotation rows. Annotation quality
and formal eligibility are separate: only Good observations whose scene/point
belongs to the frozen 87-row geometry-only split enter formal evaluation.

Manual coordinates are always stored in the decoded raw DJI image domain with
zero-based pixel centers. Cached benchmark coordinates are derived, never an
independent truth, and release-mode evaluation recomputes them from raw pixels
and frozen camera provenance.

Rows reviewed as Not visible without a click remain in the release with empty
projection fields and can never enter formal evaluation. Ambiguous rows and
Good rows outside the frozen split are diagnostic only.

Two non-formal reviewed rows project outside the raw/benchmark image bounds
(one Ambiguous and one Not visible stale click). Their original audit values
are preserved with an explicit diagnostic out-of-bounds status. Any formal
observation outside either image domain remains a hard failure.

The complete benchmark training-view list is frozen independently from the
annotation subset. No source image is physically deleted due to annotation QC;
in particular, reviewed blurry 0002 observations are excluded through row
quality, not by altering training imagery.

v1.2.2 remains an immutable sparse-control diagnostic release. v1.3.0 is the
control-heavy multi-view primary release and does not modify v1.2.2.

Algorithm implementations must use method-isolated clean worktrees, isolated
environments, isolated CUDA/build caches, and unique non-overwriting run roots.
Raw scene directories and release payloads are read-only inputs.
""",
        encoding="utf-8",
        newline="\n",
    )


def generate_payload(staging: Path, args: argparse.Namespace, command_manifest: dict[str, Any]) -> dict[str, Any]:
    dataset_root = Path(args.dataset_root)
    release_v122 = require_dir(Path(args.release_v122), "v1.2.2 release")
    input_manifest_path = require_file(Path(args.input_manifest), "v1.3 input manifest")
    inputs = load_input_manifest(input_manifest_path)
    remote_path = require_file(Path(inputs["remote_camera_manifest"]), "remote camera manifest")
    remote = json.loads(remote_path.read_text(encoding="utf-8"))
    split_path = Path(inputs["geometry_split_candidate"])
    split_rows, split_roles = load_split_rows(split_path)

    working_paths = {scene: require_file(Path(inputs["working_annotations"][scene]), f"{scene} working annotations") for scene in SCENES}
    working_rows = {scene: read_csv(path) for scene, path in working_paths.items()}
    all_source_rows = [row for scene in SCENES for row in working_rows[scene]]
    if len(all_source_rows) != EXPECTED_COUNTS["row_count"]:
        raise ValueError(f"unexpected working annotation row count: {len(all_source_rows)}")
    annotation_points = {row["point_name"] for row in all_source_rows}
    if len(annotation_points) != EXPECTED_COUNTS["all_annotation_unique_point_count"]:
        raise ValueError(f"unexpected annotation point count: {len(annotation_points)}")

    point_rows, point_fields = build_point_table(
        release_v122=release_v122,
        split_rows=split_rows,
        annotation_points=annotation_points,
        review_table_path=Path(inputs["review_only_coordinate_table"]),
        quality_summary_path=Path(inputs["rtk_quality_summary"]),
    )
    write_csv_deterministic(staging / "gcp_points_cgcs2000_cm108_v1_3_0.csv", point_rows, point_fields)
    write_frozen_split(staging, split_path, split_rows)
    copy_file_verified(
        release_v122 / "scene_metadata_gcp_benchmark_v1_1.csv",
        staging / "scene_metadata_gcp_benchmark_v1_3_0.csv",
    )

    lineage_dir = staging / "legacy_release_provenance" / "v1_2_2"
    lineage_records = []
    for name in [
        "gcp_benchmark_release_v1_2_2.json",
        "v1_2_2_release_file_manifest.json",
        "v1_2_2_release_root_digest.json",
    ]:
        source = release_v122 / name
        destination = lineage_dir / name
        record = copy_file_verified(source, destination)
        record["release_relative_path"] = destination.relative_to(staging).as_posix()
        lineage_records.append(record)
    write_json_deterministic(
        staging / "release_lineage_v1_3_0.json",
        {
            "schema": "ms_gcp_release_lineage_v1_3_0",
            "current_release_id": RELEASE_V130_ID,
            "preserved_sparse_control_diagnostic_release_id": "gcp_benchmark_release_v1_2_2_pixel_domain_20260628",
            "v1_2_2_modified": False,
            "lineage_files": lineage_records,
        },
    )

    rtk_source = require_dir(Path(inputs["rtk_authoritative_dir"]), "authoritative RTK package")
    rtk_copy_rows = []
    for source in sorted(path for path in rtk_source.rglob("*") if path.is_file()):
        destination = staging / "rtk_authoritative" / source.relative_to(rtk_source)
        record = copy_file_verified(source, destination)
        record["release_relative_path"] = destination.relative_to(staging).as_posix()
        rtk_copy_rows.append(record)
    write_json_deterministic(
        staging / "rtk_authoritative_copy_manifest_v1_3_0.json",
        {"schema": "ms_gcp_rtk_authoritative_copy_manifest_v1_3_0", "files": rtk_copy_rows},
    )

    qc_source = require_dir(Path(inputs["final_observation_qc"]), "final observation QC evidence")
    qc_copy_rows = []
    for source in sorted(path for path in qc_source.rglob("*") if path.is_file()):
        destination = staging / "annotation_qc_provenance" / source.relative_to(qc_source)
        record = copy_file_verified(source, destination)
        record["release_relative_path"] = destination.relative_to(staging).as_posix()
        qc_copy_rows.append(record)
    write_json_deterministic(
        staging / "annotation_qc_provenance_manifest_v1_3_0.json",
        {
            "schema": "ms_gcp_annotation_qc_provenance_manifest_v1_3_0",
            "source_directory": str(qc_source),
            "files": qc_copy_rows,
            "release_policy": "provenance_only; formal eligibility is recomputed from canonical rows and frozen split",
        },
    )

    v122_rows: dict[tuple[str, str, str], dict[str, str]] = {}
    for scene in SCENES:
        for row in read_csv(release_v122 / f"{scene}_gcp_annotations_pixel_domain_v1_2_2.csv"):
            key = (scene, row["point_name"], Path(row["raw_image_name"]).name)
            v122_rows[key] = row

    all_obs_rows: list[dict[str, Any]] = []
    annotation_provenance: list[dict[str, Any]] = []
    orientation_by_image: dict[tuple[str, str], dict[str, Any]] = {}
    mapping_by_key: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    camera_manifest: dict[str, Any] = {
        "schema": "ms_gcp_camera_provenance_v1_3_0",
        "release_id": RELEASE_V130_ID,
        "canonical_record_hash_policy": {
            "serialization": "UTF-8 no BOM; ensure_ascii=False; sort_keys=True; separators=(',', ':')",
            "camera_record_fields": ["camera_id", "model", "width", "height", "params"],
            "pose_record_fields": ["image_id", "image_name", "camera_id", "qvec", "tvec"],
            "float_format": ".17g",
        },
        "scenes": {},
    }
    training_views: list[dict[str, Any]] = []
    scene_projection_summary: dict[str, Any] = {}
    source_keys: set[tuple[str, str, str]] = set()
    preserved_v122_ids = 0

    for scene in SCENES:
        scene_entry = remote["scenes"][scene]
        src_cams, src_imgs, src_model = load_manifest_model(scene_entry, "raw_model")
        tgt_cams, tgt_imgs, tgt_model = load_manifest_model(scene_entry, "target_model")
        target_hashes = {Path(row["image_name"]).name: row for row in scene_entry.get("target_image_hashes", [])}
        if set(tgt_imgs) != set(target_hashes):
            raise ValueError(f"target model/image hash inventory mismatch for {scene}")
        camera_manifest["scenes"][scene] = {
            "source_model": full_camera_model_payload(src_model),
            "target_model": full_camera_model_payload(tgt_model),
        }
        annotated_names = {Path(row["image_name"]).name for row in working_rows[scene]}
        formal_names = {
            Path(row["image_name"]).name
            for row in working_rows[scene]
            if str(row.get("quality", "")).lower() == "good" and (scene, row["point_name"]) in split_roles
        }
        for image_name in sorted(tgt_imgs):
            image = tgt_imgs[image_name]
            camera = tgt_cams[image.camera_id]
            image_hash = target_hashes[image_name]
            training_views.append(
                {
                    "scene": scene,
                    "image_name": image_name,
                    "image_id": image.image_id,
                    "camera_id": image.camera_id,
                    "image_width": camera.width,
                    "image_height": camera.height,
                    "target_image_bytes": image_hash["bytes"],
                    "target_image_sha256": image_hash["sha256"],
                    "target_camera_record_sha256": camera_record_hash(camera),
                    "target_pose_record_sha256": image_pose_record_hash(image),
                    "target_cameras_bin_sha256": model_file(tgt_model, "cameras.bin"),
                    "target_images_bin_sha256": model_file(tgt_model, "images.bin"),
                    "target_points3d_bin_sha256": model_file(tgt_model, "points3D.bin"),
                    "target_pixel_domain": TARGET_PIXEL_DOMAIN,
                    "target_pixel_convention": PIXEL_CONVENTION,
                    "training_view_included": "true",
                    "any_annotation_view": bool_text(image_name in annotated_names),
                    "formal_annotation_view": bool_text(image_name in formal_names),
                }
            )

        raw_displacements: list[float] = []
        roundtrip_errors: list[float] = []
        clicked_count = 0
        no_click_count = 0
        in_bounds_count = 0
        source_sha = file_sha256(working_paths[scene])
        for row_number, row in enumerate(working_rows[scene], start=2):
            if row.get("scene") != scene:
                raise ValueError(f"working annotation scene mismatch: {scene} row {row_number}")
            image_name = Path(row["image_name"]).name
            key = (scene, row["point_name"], image_name)
            if key in source_keys:
                raise ValueError(f"duplicate working annotation key: {key}")
            source_keys.add(key)
            raw_path = require_file(source_image_path(dataset_root, scene, image_name), "raw source image")
            orient_key = (scene, image_name)
            if orient_key not in orientation_by_image:
                orientation_by_image[orient_key] = {"scene": scene, "image_name": image_name, **load_raw_image_orientation_record(raw_path)}
            orient = orientation_by_image[orient_key]
            src_img = src_imgs.get(image_name)
            tgt_img = tgt_imgs.get(image_name)
            if src_img is None or tgt_img is None:
                raise ValueError(f"source/target camera mapping missing for {key}")
            src_cam = src_cams[src_img.camera_id]
            tgt_cam = tgt_cams[tgt_img.camera_id]
            if int(orient["decoded_width"]) != src_cam.width or int(orient["decoded_height"]) != src_cam.height:
                raise ValueError(f"raw image dimensions mismatch source camera for {key}")
            pose = pose_equivalence(src_img, tgt_img)
            if not pose["pose_equivalent"]:
                raise ValueError(f"source/target pose mismatch for {key}: {pose}")
            target_hash = target_hashes.get(image_name)
            if target_hash is None:
                raise ValueError(f"target image hash missing for {key}")

            map_row = mapping_record(
                scene,
                image_name,
                src_img,
                src_cam,
                tgt_img,
                tgt_cam,
                str(orient["raw_image_sha256"]),
                str(target_hash["sha256"]),
            )
            map_hash = canonical_record_sha256(map_row)
            map_row["source_target_mapping_record_sha256"] = map_hash
            map_key = mapping_primary_key(map_row)
            if map_key in mapping_by_key and mapping_by_key[map_key] != map_row:
                raise ValueError(f"conflicting image-level mapping record for {map_key}")
            mapping_by_key[map_key] = map_row

            visible, quality = canonical_quality(row)
            raw_x = str(row.get("manual_x", "")).strip()
            raw_y = str(row.get("manual_y", "")).strip()
            if bool(raw_x) != bool(raw_y):
                raise ValueError(f"half-populated raw coordinate for {key}")
            has_click = bool(raw_x)
            annotation_good = visible and quality == "good" and has_click
            formal_role = split_roles.get((scene, row["point_name"]))
            formal_eligible = annotation_good and formal_role in {"control", "checkpoint"}
            observation_id = observation_id_from_fields_v13(
                scene,
                row["point_name"],
                image_name,
                str(orient["raw_image_sha256"]),
                raw_x,
                raw_y,
            )
            projection: dict[str, Any] = {}
            raw_in_bounds: bool | None = None
            target_in_bounds: bool | None = None
            if has_click:
                if not all(math.isfinite(float(value)) for value in [raw_x, raw_y]):
                    raise ValueError(f"non-finite raw coordinates for {key}")
                raw_in_bounds = 0.0 <= float(raw_x) < src_cam.width and 0.0 <= float(raw_y) < src_cam.height
                if formal_eligible and not raw_in_bounds:
                    raise ValueError(f"formal raw coordinates out of bounds for {key}")
                projection = raw_to_target_projection(src_cam, tgt_cam, float(raw_x), float(raw_y))
                if projection["roundtrip_error_px"] > ROUNDTRIP_TOL_PX:
                    raise ValueError(f"roundtrip error exceeds tolerance for {key}")
                target_in_bounds = 0.0 <= projection["target_x"] < tgt_cam.width and 0.0 <= projection["target_y"] < tgt_cam.height
                if formal_eligible and not target_in_bounds:
                    raise ValueError(f"formal target coordinates out of bounds for {key}")
                clicked_count += 1
                in_bounds_count += bool(target_in_bounds)
                raw_displacements.append(math.hypot(projection["target_x"] - float(raw_x), projection["target_y"] - float(raw_y)))
                roundtrip_errors.append(float(projection["roundtrip_error_px"]))
            else:
                if quality != "not_visible" or visible:
                    raise ValueError(f"no-click row must be not_visible for {key}")
                no_click_count += 1

            obs = {
                "observation_id": observation_id,
                "scene": scene,
                "point_name": row["point_name"],
                "raw_image_name": image_name,
                "raw_manual_x": raw_x,
                "raw_manual_y": raw_y,
                "annotation_visible": bool_text(visible),
                "annotation_quality": quality,
                "annotation_confidence": row.get("confidence", ""),
                "annotation_annotator": row.get("annotator", ""),
                "annotation_note": row.get("note", ""),
                "annotation_updated_at": row.get("updated_at", ""),
                "annotation_good": bool_text(annotation_good),
                "formal_eligible": bool_text(formal_eligible),
                "formal_role": formal_role if formal_eligible else "not_formal",
                "projection_status": (
                    PROJECTION_STATUS_VALID
                    if has_click and raw_in_bounds and target_in_bounds
                    else PROJECTION_STATUS_DIAGNOSTIC_OOB
                    if has_click
                    else PROJECTION_STATUS_NO_CLICK
                ),
                "raw_coordinate_in_bounds": bool_text(bool(raw_in_bounds)) if has_click else "",
                "target_in_bounds": bool_text(bool(target_in_bounds)) if has_click else "",
                "source_annotation_schema": row.get("schema", ""),
                "source_annotation_file_sha256": source_sha,
                "source_annotation_row_number": row_number,
                "source_pixel_domain": SOURCE_PIXEL_DOMAIN,
                "source_pixel_convention": PIXEL_CONVENTION,
                "source_image_width": orient["decoded_width"],
                "source_image_height": orient["decoded_height"],
                "source_image_sha256": orient["raw_image_sha256"],
                "source_exif_orientation_raw_value": orient["exif_orientation_raw_value"],
                "source_orientation_policy": ORIENTATION_POLICY,
                "source_rgb_pixel_matrix_sha256": orient["rgb_pixel_matrix_sha256"],
                "source_camera_id": src_cam.camera_id,
                "source_camera_model": src_cam.model,
                "source_camera_width": src_cam.width,
                "source_camera_height": src_cam.height,
                "source_camera_params": ";".join(fmt_float(x) for x in src_cam.params),
                "source_camera_record_sha256": camera_record_hash(src_cam),
                "source_pose_record_sha256": image_pose_record_hash(src_img),
                "source_cameras_bin_sha256": model_file(src_model, "cameras.bin"),
                "source_images_bin_sha256": model_file(src_model, "images.bin"),
                "normalized_x": fmt_float(projection["normalized_x"]) if has_click else "",
                "normalized_y": fmt_float(projection["normalized_y"]) if has_click else "",
                "normalized_unit_ray_x": fmt_float(projection["normalized_unit_ray_x"]) if has_click else "",
                "normalized_unit_ray_y": fmt_float(projection["normalized_unit_ray_y"]) if has_click else "",
                "normalized_unit_ray_z": fmt_float(projection["normalized_unit_ray_z"]) if has_click else "",
                "target_image_name": image_name,
                "target_pixel_domain": TARGET_PIXEL_DOMAIN,
                "target_pixel_convention": PIXEL_CONVENTION,
                "target_image_width": tgt_cam.width,
                "target_image_height": tgt_cam.height,
                "target_image_sha256": target_hash["sha256"],
                "target_camera_id": tgt_cam.camera_id,
                "target_camera_model": tgt_cam.model,
                "target_camera_width": tgt_cam.width,
                "target_camera_height": tgt_cam.height,
                "target_camera_params": ";".join(fmt_float(x) for x in tgt_cam.params),
                "target_camera_record_sha256": camera_record_hash(tgt_cam),
                "target_pose_record_sha256": image_pose_record_hash(tgt_img),
                "target_cameras_bin_sha256": model_file(tgt_model, "cameras.bin"),
                "target_images_bin_sha256": model_file(tgt_model, "images.bin"),
                "target_x": fmt_float(projection["target_x"]) if has_click else "",
                "target_y": fmt_float(projection["target_y"]) if has_click else "",
                "mapping_type": map_row["mapping_type"],
                "transform_version": TRANSFORM_VERSION,
                "source_target_mapping_record_sha256": map_hash,
                "roundtrip_raw_x": fmt_float(projection["roundtrip_raw_x"]) if has_click else "",
                "roundtrip_raw_y": fmt_float(projection["roundtrip_raw_y"]) if has_click else "",
                "roundtrip_error_px": fmt_float(projection["roundtrip_error_px"]) if has_click else "",
            }
            old = v122_rows.get(key)
            if old is not None:
                if old["raw_manual_x"] != raw_x or old["raw_manual_y"] != raw_y:
                    raise ValueError(f"v1.2.2 canonical raw coordinate changed for {key}")
                if old["observation_id"] != observation_id:
                    raise ValueError(f"v1.2.2 observation ID changed for {key}")
                preserved_v122_ids += 1
            all_obs_rows.append(obs)
            annotation_provenance.append(
                {
                    "observation_id": observation_id,
                    "scene": scene,
                    "point_name": row["point_name"],
                    "raw_image_name": image_name,
                    "source_annotation_path": str(working_paths[scene]),
                    "source_annotation_sha256": source_sha,
                    "source_annotation_row_number": row_number,
                    "annotation_quality": quality,
                    "annotation_good": bool_text(annotation_good),
                    "formal_eligible": bool_text(formal_eligible),
                    "disposition": "formal_primary" if formal_eligible else "diagnostic_or_qc_only",
                }
            )
        scene_projection_summary[scene] = {
            "annotation_row_count": len(working_rows[scene]),
            "coordinate_row_count": clicked_count,
            "no_coordinate_row_count": no_click_count,
            "target_in_bounds_count": in_bounds_count,
            "raw_to_target_displacement_px": {
                "median": fmt_float(percentile(raw_displacements, 0.5)),
                "p95": fmt_float(percentile(raw_displacements, 0.95)),
                "max": fmt_float(max(raw_displacements)),
            },
            "roundtrip_error_px": {
                "median": fmt_float(percentile(roundtrip_errors, 0.5)),
                "p95": fmt_float(percentile(roundtrip_errors, 0.95)),
                "max": fmt_float(max(roundtrip_errors)),
            },
        }

    if set(v122_rows) - source_keys:
        raise ValueError(f"v1.2.2 observations missing from v1.3 row spine: {len(set(v122_rows) - source_keys)}")
    if preserved_v122_ids != EXPECTED_COUNTS["v1_2_2_preserved_observation_count"]:
        raise ValueError(f"unexpected preserved v1.2.2 ID count: {preserved_v122_ids}")
    if len({row["observation_id"] for row in all_obs_rows}) != len(all_obs_rows):
        raise ValueError("duplicate v1.3 observation IDs")

    counts = Counter()
    for row in all_obs_rows:
        counts["row_count"] += 1
        counts["annotation_good_count"] += row["annotation_good"] == "true"
        counts["formal_eligible_count"] += row["formal_eligible"] == "true"
        counts["coordinate_row_count"] += bool(row["raw_manual_x"])
        counts["no_coordinate_row_count"] += not bool(row["raw_manual_x"])
        counts["diagnostic_projection_out_of_bounds_count"] += row["projection_status"] == PROJECTION_STATUS_DIAGNOSTIC_OOB
    for key in [
        "row_count",
        "annotation_good_count",
        "formal_eligible_count",
        "coordinate_row_count",
        "no_coordinate_row_count",
        "diagnostic_projection_out_of_bounds_count",
    ]:
        if int(counts[key]) != EXPECTED_COUNTS[key]:
            raise ValueError(f"frozen count mismatch {key}: {counts[key]} vs {EXPECTED_COUNTS[key]}")
    if len(orientation_by_image) != EXPECTED_COUNTS["annotated_image_count"]:
        raise ValueError(f"annotated image count mismatch: {len(orientation_by_image)}")
    if len(training_views) != EXPECTED_COUNTS["training_view_count"]:
        raise ValueError(f"training view count mismatch: {len(training_views)}")

    mapping_rows = sorted(mapping_by_key.values(), key=mapping_primary_key)
    if len(mapping_rows) != EXPECTED_COUNTS["annotated_image_count"]:
        raise ValueError(f"mapping record count mismatch: {len(mapping_rows)}")
    mapping_hashes = {row["source_target_mapping_record_sha256"] for row in mapping_rows}
    used_mapping_hashes = {row["source_target_mapping_record_sha256"] for row in all_obs_rows}
    if mapping_hashes != used_mapping_hashes:
        raise ValueError("mapping references are not a bijection over annotated images")

    for scene in SCENES:
        scene_rows = [row for row in all_obs_rows if row["scene"] == scene]
        write_csv_deterministic(
            staging / f"{scene}_gcp_annotations_pixel_domain_v1_3_0.csv",
            scene_rows,
            OBS_FIELDS,
        )
    write_csv_deterministic(
        staging / "annotation_inclusion_provenance_v1_3_0.csv",
        annotation_provenance,
        list(annotation_provenance[0].keys()),
    )
    orientation_rows = [orientation_by_image[key] for key in sorted(orientation_by_image)]
    write_json_deterministic(staging / "raw_image_orientation_manifest_v1_3_0.json", orientation_rows)
    write_json_deterministic(staging / "source_target_mapping_manifest_v1_3_0.json", mapping_rows)
    write_csv_deterministic(
        staging / "source_target_mapping_manifest_v1_3_0.csv",
        mapping_rows,
        list(mapping_rows[0].keys()),
    )
    write_json_deterministic(staging / "camera_provenance_manifest_v1_3_0.json", camera_manifest)
    training_fields = list(training_views[0].keys())
    write_csv_deterministic(staging / "benchmark_training_view_manifest_v1_3_0.csv", training_views, training_fields)
    write_json_deterministic(
        staging / "benchmark_training_view_manifest_v1_3_0.json",
        {
            "schema": "ms_gcp_benchmark_training_view_manifest_v1_3_0",
            "release_id": RELEASE_V130_ID,
            "training_view_count": len(training_views),
            "source_directories_read_only": True,
            "no_physical_image_exclusions": True,
            "views": training_views,
        },
    )

    point_dispositions = []
    formal_points = {point for _, point in split_roles}
    for point in sorted(annotation_points):
        point_dispositions.append(
            {
                "point_name": point,
                "formal_primary_eligible": bool_text(point in formal_points),
                "disposition": "formal_control_or_checkpoint" if point in formal_points else "diagnostic_only_not_in_v1_3_split",
                "reason": "geometry_only_split" if point in formal_points else "not_selected_by_frozen_geometry_only_split",
            }
        )
    write_csv_deterministic(
        staging / "point_disposition_v1_3_0.csv",
        point_dispositions,
        list(point_dispositions[0].keys()),
    )

    mapping_json_sha = file_sha256(staging / "source_target_mapping_manifest_v1_3_0.json")
    mapping_csv_sha = file_sha256(staging / "source_target_mapping_manifest_v1_3_0.csv")
    mapping_root = canonical_records_root_sha256(
        mapping_rows,
        ["scene", "source_image_id", "source_image_name", "target_image_id", "target_image_name"],
    )
    write_json_deterministic(
        staging / "projection_manifest_v1_3_0.json",
        {
            "schema": "ms_gcp_pixel_domain_projection_manifest_v1_3_0",
            "release_id": RELEASE_V130_ID,
            "source_pixel_domain": SOURCE_PIXEL_DOMAIN,
            "target_pixel_domain": TARGET_PIXEL_DOMAIN,
            "pixel_convention": PIXEL_CONVENTION,
            "transform_version": TRANSFORM_VERSION,
            "float_dtype": "float64",
            "cached_target_tolerance_px_per_coordinate": fmt_float(CACHED_TARGET_TOL_PX),
            "roundtrip_tolerance_px_euclidean": fmt_float(ROUNDTRIP_TOL_PX),
            "annotation_row_count": len(all_obs_rows),
            "coordinate_row_count": int(counts["coordinate_row_count"]),
            "no_coordinate_row_count": int(counts["no_coordinate_row_count"]),
            "mapping_json_file_sha256": mapping_json_sha,
            "mapping_csv_file_sha256": mapping_csv_sha,
            "mapping_records_root_sha256": mapping_root,
            "scene_projection_validation_summary": scene_projection_summary,
        },
    )

    write_protocol_doc(staging / "V1_3_0_MULTIVIEW_CONTROL_HEAVY_PROTOCOL.md")
    generator = generator_provenance(REPO_ROOT, "code/gcp/generate_gcp_release_v1_3.py", command_manifest)
    if not generator["generator_worktree_clean"]:
        raise ValueError(f"generator worktree must be clean: {generator['generator_worktree_status_porcelain']!r}")
    config = {
        "schema": RELEASE_V130_SCHEMA,
        "release_id": RELEASE_V130_ID,
        "release_role": "control_heavy_multi_view_primary",
        "preserved_diagnostic_release_id": "gcp_benchmark_release_v1_2_2_pixel_domain_20260628",
        "annotation_csv_pattern": "gcp_*_gcp_annotations_pixel_domain_v1_3_0.csv",
        "gcp_csv": "gcp_points_cgcs2000_cm108_v1_3_0.csv",
        "split_csv": "gcp_control_checkpoint_split_v1_3_0.csv",
        "scene_metadata_csv": "scene_metadata_gcp_benchmark_v1_3_0.csv",
        "inclusion_provenance_csv": "annotation_inclusion_provenance_v1_3_0.csv",
        "point_disposition_csv": "point_disposition_v1_3_0.csv",
        "projection_manifest": "projection_manifest_v1_3_0.json",
        "camera_provenance_manifest": "camera_provenance_manifest_v1_3_0.json",
        "orientation_manifest": "raw_image_orientation_manifest_v1_3_0.json",
        "mapping_manifest_csv": "source_target_mapping_manifest_v1_3_0.csv",
        "mapping_manifest_json": "source_target_mapping_manifest_v1_3_0.json",
        "training_view_manifest_csv": "benchmark_training_view_manifest_v1_3_0.csv",
        "training_view_manifest_json": "benchmark_training_view_manifest_v1_3_0.json",
        "payload_manifest": "v1_3_0_release_file_manifest.json",
        "root_digest_record": "v1_3_0_release_root_digest.json",
        "mapping_json_file_sha256": mapping_json_sha,
        "mapping_csv_file_sha256": mapping_csv_sha,
        "mapping_records_root_sha256": mapping_root,
        "formal_annotation_policy": "annotation_good AND scene_point_in_frozen_split",
        "formal_annotation_quality": "good_only",
        "formal_min_valid_observations": 1,
        "control_policy": "require_all",
        "image_exclusion_count": 0,
        "source_scene_directories_read_only": True,
        "frozen_counts": {
            **EXPECTED_COUNTS,
            "scene_rows": {scene: sum(row["scene"] == scene for row in all_obs_rows) for scene in SCENES},
            "scene_annotation_good": {scene: sum(row["scene"] == scene and row["annotation_good"] == "true" for row in all_obs_rows) for scene in SCENES},
            "scene_formal_eligible": {scene: sum(row["scene"] == scene and row["formal_eligible"] == "true" for row in all_obs_rows) for scene in SCENES},
            "scene_training_views": {scene: sum(row["scene"] == scene for row in training_views) for scene in SCENES},
        },
        "input_manifest_path": str(input_manifest_path),
        "input_manifest_sha256": file_sha256(input_manifest_path),
        "remote_camera_manifest_sha256": file_sha256(remote_path),
        "generator_provenance": generator,
    }
    write_json_deterministic(staging / "gcp_benchmark_release_v1_3_0.json", config)

    entries = payload_manifest_entries(
        staging,
        exclude={"v1_3_0_release_file_manifest.json", "v1_3_0_release_root_digest.json"},
    )
    payload_manifest = {
        "schema": "ms_gcp_release_payload_manifest_v1",
        "release_id": RELEASE_V130_ID,
        "path_sort": "UTF-8 byte order over POSIX relative paths",
        "excluded_files": ["v1_3_0_release_file_manifest.json", "v1_3_0_release_root_digest.json"],
        "files": entries,
    }
    write_json_deterministic(staging / "v1_3_0_release_file_manifest.json", payload_manifest)
    manifest_sha = file_sha256(staging / "v1_3_0_release_file_manifest.json")
    root_record = {
        "schema": "ms_gcp_release_root_digest_v1",
        "release_id": RELEASE_V130_ID,
        "payload_file_count": len(entries),
        "payload_manifest_path": "v1_3_0_release_file_manifest.json",
        "payload_manifest_sha256": manifest_sha,
        "payload_root_digest_algorithm": "sha256(sorted manifest entries; JSON ensure_ascii=False sort_keys=True separators=(',', ':'); UTF-8 no BOM)",
        "payload_root_digest_sha256": payload_root_digest(entries),
        "generator_provenance": generator,
    }
    write_json_deterministic(staging / "v1_3_0_release_root_digest.json", root_record)
    integrity = verify_payload_integrity(
        staging,
        staging / "v1_3_0_release_file_manifest.json",
        staging / "v1_3_0_release_root_digest.json",
    )
    if not integrity["passed"]:
        raise ValueError(f"release integrity failed: {integrity}")
    return {
        "release_id": RELEASE_V130_ID,
        "counts": dict(counts),
        "mapping_record_count": len(mapping_rows),
        "annotated_image_count": len(orientation_rows),
        "training_view_count": len(training_views),
        "preserved_v1_2_2_observation_ids": preserved_v122_ids,
        "integrity": integrity,
        "generator_provenance": generator,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the transactional MS-GCP v1.3.0 release")
    parser.add_argument("--dataset_root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--project_root", default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("--release_v122", default=str(DEFAULT_DATASET_ROOT / "scenes" / "gcp_manual_annotations_v1_2_2"))
    parser.add_argument("--input_manifest", default=str(DEFAULT_INPUT_MANIFEST))
    parser.add_argument("--final_dir", default=str(DEFAULT_FINAL_DIR))
    parser.add_argument("--summary_out", default="")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()

    final_dir = Path(args.final_dir)
    if final_dir.exists():
        raise SystemExit(f"formal v1.3.0 release directory already exists; refusing overwrite: {final_dir}")
    staging = make_unique_staging(final_dir)
    compare = make_unique_staging(final_dir.with_name(final_dir.name + ".compare"))
    command_manifest = {
        "script": "code/gcp/generate_gcp_release_v1_3.py",
        "dataset_root": str(Path(args.dataset_root)),
        "release_v122": str(Path(args.release_v122)),
        "input_manifest": str(Path(args.input_manifest)),
        "input_manifest_sha256": file_sha256(Path(args.input_manifest)),
        "publish": bool(args.publish),
    }
    try:
        first = generate_payload(staging, args, command_manifest)
        second = generate_payload(compare, args, command_manifest)
        differences = compare_dirs(staging, compare)
        if differences:
            raise ValueError(f"byte-identical regeneration failed: {differences[:20]}")
        if args.publish:
            # All validation and cleanup must finish before the formal path appears.
            remove_tree_readonly(compare)
            staging.rename(final_dir)
            release_dir = final_dir
        else:
            release_dir = staging
        summary = {
            "status": "PASS",
            "release_dir": str(release_dir),
            "published": bool(args.publish),
            "byte_identical_regeneration": True,
            **first,
        }
        if args.summary_out:
            write_json_deterministic(Path(args.summary_out), summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except Exception as exc:  # noqa: BLE001
        blocker = {
            "status": "BLOCKER",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "formal_release_created": final_dir.exists(),
            "staging_dir": str(staging) if staging.exists() else "",
        }
        if staging.exists():
            write_json_deterministic(staging / "BLOCKER.json", blocker)
        if args.summary_out:
            write_json_deterministic(Path(args.summary_out), blocker)
        print(json.dumps(blocker, ensure_ascii=False, indent=2))
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
