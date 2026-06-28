from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gcp_pixel_domain_v1_2 import (  # noqa: E402
    ARCHIVED_UNDISTORTED_TOL_PX,
    CACHED_TARGET_TOL_PX,
    ORIENTATION_POLICY,
    PIXEL_CONVENTION,
    RELEASE_V12_ID,
    RELEASE_V12_SCHEMA,
    RGB_MATRIX_HASH_MAGIC,
    ROUNDTRIP_TOL_PX,
    SCENES,
    SOURCE_PIXEL_DOMAIN,
    TARGET_PIXEL_DOMAIN,
    TRANSFORM_VERSION,
    CameraRecord,
    ImageRecord,
    canonical_record_sha256,
    camera_canonical_record,
    camera_record_hash,
    file_sha256,
    fmt_float,
    generator_provenance,
    image_pose_canonical_record,
    image_pose_record_hash,
    load_manifest_model,
    load_raw_image_orientation_record,
    model_file_record,
    observation_id_from_fields,
    observation_id_from_payload,
    observation_id_payload,
    payload_manifest_entries,
    payload_root_digest,
    pose_equivalence,
    raw_to_target_projection,
    read_csv,
    relative_posix,
    rgb_pixel_matrix_sha256,
    serialize_observation_id_payload,
    serialize_rgb_pixel_matrix,
    sha256_bytes,
    verify_payload_integrity,
    write_csv_deterministic,
    write_json_deterministic,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = Path(r"E:\datasets\M3M-GCP")
DEFAULT_RELEASE_V11 = DEFAULT_DATASET_ROOT / "scenes" / "gcp_manual_annotations"
DEFAULT_FINAL_DIR = DEFAULT_DATASET_ROOT / "scenes" / "gcp_manual_annotations_v1_2"
DEFAULT_PROJECT_ROOT = Path(r"E:\M3M-GCP-3DGS")
DEFAULT_REMOTE_MANIFEST = (
    DEFAULT_PROJECT_ROOT
    / "outputs"
    / "gcp_6scene_annotation_domain_inputs_20260628"
    / "gcp_6scene_annotation_domain_jsonlight_20260628"
    / "remote_light_manifest.json"
)
DEFAULT_STAGE1_DIR = DEFAULT_PROJECT_ROOT / "outputs" / "gcp_6scene_annotation_domain_audit_20260628_20260628_060551"

COPY_METADATA_FILES = [
    "gcp_points_primary_usable_cgcs2000_cm108_v1.csv",
    "gcp_control_checkpoint_splits_v1.csv",
    "scene_metadata_gcp_benchmark_v1_1.csv",
    "final_annotation_inclusion_provenance.csv",
    "final_pointset_file_manifest.json",
]

OBS_FIELDS = [
    "observation_id",
    "scene",
    "point_name",
    "raw_image_name",
    "raw_manual_x",
    "raw_manual_y",
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


def read_stage1_observation_ids(stage1_dir: Path) -> dict[tuple[str, str, str, str, str], str]:
    path = stage1_dir / "per_observation_domain_audit.csv"
    rows = read_csv(path)
    return {
        (
            row["scene"],
            row["point_name"],
            row["raw_image_name"],
            row["raw_manual_x_text"],
            row["raw_manual_y_text"],
        ): row["observation_id"]
        for row in rows
    }


def make_unique_staging(final_dir: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = final_dir.with_name(f"{final_dir.name}.staging_{stamp}")
    if not base.exists():
        base.mkdir(parents=True)
        return base
    for i in range(1, 1000):
        candidate = final_dir.with_name(f"{final_dir.name}.staging_{stamp}_{i:02d}")
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
    raise RuntimeError("could not allocate unique staging dir")


def source_image_path(dataset_root: Path, scene: str, image_name: str) -> Path:
    return dataset_root / "scenes" / scene / image_name


def load_archived_undistorted(project_root: Path, scene: str) -> dict[tuple[str, str, str], tuple[float, float]]:
    candidates = {
        "gcp_3000_20260602": [
            project_root / "outputs" / "gaussian_gcp_eval_20260618" / "annotations_undistorted" / "gcp_image_observations_undistorted_for_evaluation.csv",
        ],
        "gcp_5000_20260602": [
            project_root
            / "outputs"
            / "remote_sync"
            / "three_scene_diagnostics_light_20260624"
            / "gaussian-gcp-eval-official-3scenes-20260624"
            / "gcp_5000_20260602"
            / "annotations_undistorted"
            / "gcp_image_observations_undistorted_for_evaluation.csv",
        ],
        "gcp_100000_20260610": [
            project_root
            / "outputs"
            / "gcp_eval_official_2scenes_20260623"
            / "gcp_100000_20260610"
            / "annotations_undistorted"
            / "gcp_image_observations_undistorted_for_evaluation.csv",
        ],
    }
    for path in candidates.get(scene, []):
        if path.exists():
            result = {}
            for row in read_csv(path):
                if row.get("scene") != scene:
                    continue
                x = row.get("undistorted_x") or row.get("undistorted_u") or row.get("u_px") or row.get("target_x") or row.get("manual_x")
                y = row.get("undistorted_y") or row.get("undistorted_v") or row.get("v_px") or row.get("target_y") or row.get("manual_y")
                if x and y:
                    result[(scene, row["point_name"], Path(row["image_name"]).name)] = (float(x), float(y))
            return result
    return {}


def copy_metadata(release_v11: Path, staging: Path) -> list[dict[str, Any]]:
    rows = []
    for name in COPY_METADATA_FILES:
        src = release_v11 / name
        if not src.exists():
            raise FileNotFoundError(f"required byte-copy metadata file missing: {src}")
        dst = staging / name
        shutil.copy2(src, dst)
        src_hash = file_sha256(src)
        dst_hash = file_sha256(dst)
        rows.append(
            {
                "file_name": name,
                "v1_1_source_path": str(src),
                "v1_1_sha256": src_hash,
                "v1_2_copied_path": str(dst),
                "v1_2_sha256": dst_hash,
                "byte_equal": src_hash == dst_hash and src.stat().st_size == dst.stat().st_size,
            }
        )
        if not rows[-1]["byte_equal"]:
            raise ValueError(f"byte-copy verification failed for {name}")
    return rows


def mapping_record(scene: str, image_name: str, src_img: ImageRecord, src_cam: CameraRecord, tgt_img: ImageRecord, tgt_cam: CameraRecord) -> dict[str, Any]:
    pose = pose_equivalence(src_img, tgt_img)
    return {
        "mapping_type": "pose_equivalent_colmap_undistortion_intrinsics_remap",
        "scene": scene,
        "source_camera_id": src_cam.camera_id,
        "source_camera_record_sha256": camera_record_hash(src_cam),
        "source_image_id": src_img.image_id,
        "source_image_name": image_name,
        "source_pose_record_sha256": image_pose_record_hash(src_img),
        "target_camera_id": tgt_cam.camera_id,
        "target_camera_record_sha256": camera_record_hash(tgt_cam),
        "target_image_id": tgt_img.image_id,
        "target_image_name": image_name,
        "target_pose_record_sha256": image_pose_record_hash(tgt_img),
        "transform_version": TRANSFORM_VERSION,
        **pose,
    }


def generate_payload(staging: Path, args: argparse.Namespace, command_manifest: dict[str, Any]) -> dict[str, Any]:
    release_v11 = Path(args.release_v11)
    dataset_root = Path(args.dataset_root)
    project_root = Path(args.project_root)
    stage1_dir = Path(args.stage1_dir)
    remote_manifest_path = Path(args.remote_light_manifest)
    remote_manifest = json.loads(remote_manifest_path.read_text(encoding="utf-8"))
    stage1_ids = read_stage1_observation_ids(stage1_dir)
    metadata_copy = copy_metadata(release_v11, staging)

    copied_files = {row["file_name"]: row for row in metadata_copy}
    all_obs_rows = []
    all_mapping_rows = []
    orientation_by_image: dict[tuple[str, str], dict[str, Any]] = {}
    camera_manifest: dict[str, Any] = {"schema": "ms_gcp_camera_provenance_v1_2", "scenes": {}}
    projection_summary_rows = []
    archived_summary_rows = []

    for scene in SCENES:
        release_rows = read_csv(release_v11 / f"{scene}_gcp_annotations_final_good_nadir_v1.csv")
        scene_entry = remote_manifest["scenes"][scene]
        src_cams, src_imgs, src_model = load_manifest_model(scene_entry, "raw_model")
        tgt_cams, tgt_imgs, tgt_model = load_manifest_model(scene_entry, "target_model")
        target_hashes = {Path(row["image_name"]).name: row for row in scene_entry.get("target_image_hashes", [])}
        archived = load_archived_undistorted(project_root, scene)
        scene_obs = []
        archived_errors = []
        for row in release_rows:
            image_name = Path(row["image_name"]).name
            point_name = row["point_name"]
            raw_x_text = row["manual_x"]
            raw_y_text = row["manual_y"]
            raw_path = source_image_path(dataset_root, scene, image_name)
            if not raw_path.exists():
                raise FileNotFoundError(f"raw source image missing: {raw_path}")
            orient_key = (scene, image_name)
            if orient_key not in orientation_by_image:
                orientation_by_image[orient_key] = {
                    "scene": scene,
                    "image_name": image_name,
                    **load_raw_image_orientation_record(raw_path),
                }
            orient = orientation_by_image[orient_key]
            src_img = src_imgs.get(image_name)
            tgt_img = tgt_imgs.get(image_name)
            if src_img is None or tgt_img is None:
                raise ValueError(f"missing source/target image mapping for {scene} {image_name}")
            src_cam = src_cams[src_img.camera_id]
            tgt_cam = tgt_cams[tgt_img.camera_id]
            if int(orient["decoded_width"]) != int(src_cam.width) or int(orient["decoded_height"]) != int(src_cam.height):
                raise ValueError(f"decoded dimensions do not match source camera for {scene} {image_name}")
            pose = pose_equivalence(src_img, tgt_img)
            if not pose["pose_equivalent"]:
                raise ValueError(f"source/target pose mismatch for {scene} {image_name}: {pose}")
            projection = raw_to_target_projection(src_cam, tgt_cam, float(raw_x_text), float(raw_y_text))
            if projection["roundtrip_error_px"] > ROUNDTRIP_TOL_PX:
                raise ValueError(f"roundtrip error too high for {scene} {image_name}: {projection['roundtrip_error_px']}")
            if not (0 <= projection["target_x"] < tgt_cam.width and 0 <= projection["target_y"] < tgt_cam.height):
                raise ValueError(f"target projection out of bounds for {scene} {image_name}")
            archived_xy = archived.get((scene, point_name, image_name))
            archived_error = ""
            if archived_xy:
                archived_error_value = math.hypot(projection["target_x"] - archived_xy[0], projection["target_y"] - archived_xy[1])
                archived_error = fmt_float(archived_error_value)
                archived_errors.append(archived_error_value)
                if archived_error_value > ARCHIVED_UNDISTORTED_TOL_PX:
                    raise ValueError(f"archived undistorted mismatch for {scene} {point_name} {image_name}: {archived_error_value}")
            obs_id = observation_id_from_fields(
                scene,
                point_name,
                image_name,
                orient["raw_image_sha256"],
                raw_x_text,
                raw_y_text,
            )
            stage1_key = (scene, point_name, image_name, raw_x_text, raw_y_text)
            if stage1_ids.get(stage1_key) != obs_id:
                raise ValueError(f"Stage-1 observation_id mismatch for {stage1_key}: {obs_id} != {stage1_ids.get(stage1_key)}")
            mrec = mapping_record(scene, image_name, src_img, src_cam, tgt_img, tgt_cam)
            mrec_sha = canonical_record_sha256(mrec)
            out = {
                "observation_id": obs_id,
                "scene": scene,
                "point_name": point_name,
                "raw_image_name": image_name,
                "raw_manual_x": raw_x_text,
                "raw_manual_y": raw_y_text,
                "source_pixel_domain": SOURCE_PIXEL_DOMAIN,
                "source_pixel_convention": PIXEL_CONVENTION,
                "source_image_width": src_cam.width,
                "source_image_height": src_cam.height,
                "source_image_sha256": orient["raw_image_sha256"],
                "source_exif_orientation_raw_value": orient["exif_orientation_raw_value"],
                "source_orientation_policy": ORIENTATION_POLICY,
                "source_rgb_pixel_matrix_sha256": orient["rgb_pixel_matrix_sha256"],
                "source_camera_id": src_cam.camera_id,
                "source_camera_model": src_cam.model,
                "source_camera_width": src_cam.width,
                "source_camera_height": src_cam.height,
                "source_camera_params": ";".join(fmt_float(v) for v in src_cam.params),
                "source_camera_record_sha256": camera_record_hash(src_cam),
                "source_pose_record_sha256": image_pose_record_hash(src_img),
                "source_cameras_bin_sha256": model_file(src_model, "cameras.bin"),
                "source_images_bin_sha256": model_file(src_model, "images.bin"),
                "normalized_x": fmt_float(projection["normalized_x"]),
                "normalized_y": fmt_float(projection["normalized_y"]),
                "normalized_unit_ray_x": fmt_float(projection["normalized_unit_ray_x"]),
                "normalized_unit_ray_y": fmt_float(projection["normalized_unit_ray_y"]),
                "normalized_unit_ray_z": fmt_float(projection["normalized_unit_ray_z"]),
                "target_image_name": image_name,
                "target_pixel_domain": TARGET_PIXEL_DOMAIN,
                "target_pixel_convention": PIXEL_CONVENTION,
                "target_image_width": tgt_cam.width,
                "target_image_height": tgt_cam.height,
                "target_image_sha256": target_hashes.get(image_name, {}).get("sha256", ""),
                "target_camera_id": tgt_cam.camera_id,
                "target_camera_model": tgt_cam.model,
                "target_camera_width": tgt_cam.width,
                "target_camera_height": tgt_cam.height,
                "target_camera_params": ";".join(fmt_float(v) for v in tgt_cam.params),
                "target_camera_record_sha256": camera_record_hash(tgt_cam),
                "target_pose_record_sha256": image_pose_record_hash(tgt_img),
                "target_cameras_bin_sha256": model_file(tgt_model, "cameras.bin"),
                "target_images_bin_sha256": model_file(tgt_model, "images.bin"),
                "target_x": fmt_float(projection["target_x"]),
                "target_y": fmt_float(projection["target_y"]),
                "mapping_type": mrec["mapping_type"],
                "transform_version": TRANSFORM_VERSION,
                "source_target_mapping_record_sha256": mrec_sha,
                "roundtrip_raw_x": fmt_float(projection["roundtrip_raw_x"]),
                "roundtrip_raw_y": fmt_float(projection["roundtrip_raw_y"]),
                "roundtrip_error_px": fmt_float(projection["roundtrip_error_px"]),
            }
            if not out["target_image_sha256"]:
                raise ValueError(f"missing target image SHA-256 in remote manifest for {scene} {image_name}")
            scene_obs.append(out)
            all_obs_rows.append(out)
            all_mapping_rows.append({**mrec, "source_target_mapping_record_sha256": mrec_sha})
        write_csv_deterministic(staging / f"{scene}_gcp_annotations_pixel_domain_v1_2.csv", scene_obs, OBS_FIELDS)
        projection_summary_rows.append(
            {
                "scene": scene,
                "observation_count": len(scene_obs),
                "median_raw_to_target_displacement_px": "",  # retained in Stage-1 package
                "max_roundtrip_error_px": max(float(r["roundtrip_error_px"]) for r in scene_obs),
                "archived_undistorted_rows": len(archived_errors),
                "archived_undistorted_max_error_px": max(archived_errors) if archived_errors else "",
            }
        )
        camera_manifest["scenes"][scene] = {
            "source_model": compact_model(src_model),
            "target_model": compact_model(tgt_model),
        }
        if archived_errors:
            archived_summary_rows.append(
                {
                    "scene": scene,
                    "count": len(archived_errors),
                    "max_error_px": fmt_float(max(archived_errors)),
                    "tolerance_px": fmt_float(ARCHIVED_UNDISTORTED_TOL_PX),
                }
            )

    obs_ids = [row["observation_id"] for row in all_obs_rows]
    if len(obs_ids) != 611 or len(set(obs_ids)) != 611:
        raise ValueError(f"observation id count/collision failure: rows={len(obs_ids)} unique={len(set(obs_ids))}")

    write_csv_deterministic(staging / "source_target_mapping_manifest_v1_2.csv", all_mapping_rows, sorted({k for row in all_mapping_rows for k in row}))
    write_json_deterministic(staging / "source_target_mapping_manifest_v1_2.json", all_mapping_rows)
    write_json_deterministic(staging / "camera_provenance_manifest_v1_2.json", camera_manifest)
    write_json_deterministic(staging / "raw_image_orientation_manifest_v1_2.json", sorted(orientation_by_image.values(), key=lambda r: (r["scene"], r["image_name"])))
    write_csv_deterministic(staging / "projection_validation_summary_v1_2.csv", projection_summary_rows, sorted({k for row in projection_summary_rows for k in row}))
    write_csv_deterministic(staging / "archived_undistorted_agreement_v1_2.csv", archived_summary_rows, ["scene", "count", "max_error_px", "tolerance_px"])
    write_protocol_doc(staging / "V1_2_PIXEL_DOMAIN_PROTOCOL.md")

    generator = generator_provenance(
        REPO_ROOT,
        "code/gcp/generate_gcp_release_v1_2.py",
        command_manifest,
    )
    if not generator["generator_worktree_clean"]:
        raise ValueError(
            "Generator worktree must be clean before freezing release v1.2; "
            f"dirty status: {generator['generator_worktree_status_porcelain']!r}"
        )
    config = {
        "schema": RELEASE_V12_SCHEMA,
        "release_id": RELEASE_V12_ID,
        "annotation_csv_pattern": "gcp_*_gcp_annotations_pixel_domain_v1_2.csv",
        "gcp_csv": "gcp_points_primary_usable_cgcs2000_cm108_v1.csv",
        "split_csv": "gcp_control_checkpoint_splits_v1.csv",
        "scene_metadata_csv": "scene_metadata_gcp_benchmark_v1_1.csv",
        "inclusion_provenance_csv": "final_annotation_inclusion_provenance.csv",
        "projection_manifest": "projection_manifest_v1_2.json",
        "camera_provenance_manifest": "camera_provenance_manifest_v1_2.json",
        "orientation_manifest": "raw_image_orientation_manifest_v1_2.json",
        "mapping_manifest_csv": "source_target_mapping_manifest_v1_2.csv",
        "mapping_manifest_json": "source_target_mapping_manifest_v1_2.json",
        "payload_manifest": "v1_2_release_file_manifest.json",
        "root_digest_record": "v1_2_release_root_digest.json",
        "formal_run_requirements": [
            "recompute target_x/target_y from canonical raw pixel and camera provenance",
            "fail on source/target image or camera hash mismatch",
            "fail on orientation policy/hash mismatch",
            "fail on cached target projection mismatch",
            "do not use v1.1 raw manual_x/manual_y directly against undistorted packets",
        ],
        "frozen_counts": {
            "final_annotation_observations": 611,
            "scene_final_observations": {scene: sum(1 for row in all_obs_rows if row["scene"] == scene) for scene in SCENES},
        },
        "generator_provenance": generator,
    }
    write_json_deterministic(staging / "projection_manifest_v1_2.json", build_projection_manifest(all_obs_rows))
    write_json_deterministic(staging / "gcp_benchmark_release_v1_2.json", config)

    entries = payload_manifest_entries(
        staging,
        exclude={"v1_2_release_file_manifest.json", "v1_2_release_root_digest.json"},
    )
    manifest = {
        "schema": "ms_gcp_release_payload_manifest_v1",
        "release_id": RELEASE_V12_ID,
        "path_sort": "UTF-8 byte order over POSIX relative paths",
        "excluded_files": ["v1_2_release_file_manifest.json", "v1_2_release_root_digest.json"],
        "files": entries,
    }
    write_json_deterministic(staging / "v1_2_release_file_manifest.json", manifest)
    manifest_sha = file_sha256(staging / "v1_2_release_file_manifest.json")
    root_digest = payload_root_digest(entries)
    root_record = {
        "schema": "ms_gcp_release_root_digest_v1",
        "release_id": RELEASE_V12_ID,
        "payload_file_count": len(entries),
        "payload_manifest_path": "v1_2_release_file_manifest.json",
        "payload_manifest_sha256": manifest_sha,
        "payload_root_digest_algorithm": "sha256(sorted manifest entries; JSON ensure_ascii=False sort_keys=True separators=( ',', ':' ); UTF-8 no BOM)",
        "payload_root_digest_sha256": root_digest,
        "generator_provenance": generator,
    }
    write_json_deterministic(staging / "v1_2_release_root_digest.json", root_record)
    integrity = verify_payload_integrity(staging, staging / "v1_2_release_file_manifest.json", staging / "v1_2_release_root_digest.json")
    if not integrity["passed"]:
        raise ValueError(f"release payload integrity failed: {integrity}")

    return {
        "observation_count": len(all_obs_rows),
        "unique_observation_ids": len(set(obs_ids)),
        "metadata_copy": metadata_copy,
        "integrity": integrity,
        "generator_provenance": generator,
        "orientation_image_count": len(orientation_by_image),
    }


def model_file(model: dict[str, Any], name: str) -> str:
    for row in model.get("files", []):
        if Path(str(row.get("name", ""))).name == name:
            return str(row.get("sha256", ""))
    return ""


def compact_model(model: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": model.get("path", ""),
        "files": [
            {k: row.get(k, "") for k in ["name", "path", "bytes", "sha256"]}
            for row in model.get("files", [])
            if Path(str(row.get("name", ""))).name in {"cameras.bin", "images.bin", "cameras.txt"}
        ],
        "cameras": model.get("cameras", []),
        "images": model.get("images", []),
    }


def build_projection_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": "ms_gcp_pixel_domain_projection_manifest_v1_2",
        "release_id": RELEASE_V12_ID,
        "source_pixel_domain": "raw_dji_decoded_pixel_matrix_ignore_exif_orientation",
        "target_pixel_domain": "benchmark_colmap_undistorted_pinhole_pixel_domain",
        "pixel_convention": "zero_based_pixel_centers",
        "transform_version": TRANSFORM_VERSION,
        "projection_numeric_policy": {
            "float_dtype": "float64",
            "simple_radial_max_iterations": 20,
            "simple_radial_convergence_abs": "1e-12",
            "simple_radial_convergence_rel": "1e-12",
            "cached_target_tolerance_px_per_coordinate": "1e-9",
            "roundtrip_tolerance_px_euclidean": "1e-6",
        },
        "observation_count": len(rows),
        "mapping_manifest_sha256": canonical_record_sha256(
            [
                {
                    "observation_id": row["observation_id"],
                    "target_x": row["target_x"],
                    "target_y": row["target_y"],
                    "mapping_hash": row["source_target_mapping_record_sha256"],
                }
                for row in rows
            ]
        ),
    }


def write_protocol_doc(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# MS-GCP Release v1.2 Pixel-Domain Protocol",
                "",
                "Release v1.2 preserves the v1.1 raw manual annotations but formal evaluation uses verified cached projections into the benchmark undistorted COLMAP camera domain.",
                "",
                "Cached target coordinates are not authoritative by themselves. A release-mode evaluator must recompute target coordinates from canonical raw pixels, source camera intrinsics, and mapping records before using them.",
                "",
                "Raw JPEG coordinates are defined in the pixel matrix produced by `PIL.Image.open(path).convert(\"RGB\")` without EXIF orientation transpose.",
                "",
                "The benchmark camera track requires methods to use the benchmark undistorted images and cameras. Method-specific camera evaluation requires pose-equivalent cameras or an independently verified explicit pixel remap.",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def compare_dirs(a: Path, b: Path) -> list[str]:
    problems = []
    files_a = sorted([p.relative_to(a).as_posix() for p in a.rglob("*") if p.is_file()])
    files_b = sorted([p.relative_to(b).as_posix() for p in b.rglob("*") if p.is_file()])
    if files_a != files_b:
        return ["file_list_mismatch"]
    for rel in files_a:
        if file_sha256(a / rel) != file_sha256(b / rel):
            problems.append(rel)
    return problems


def package_dir(source_dir: Path, package_path: Path) -> tuple[Path, Path]:
    package_path.parent.mkdir(parents=True, exist_ok=True)
    if package_path.exists():
        stamp = datetime.now().strftime("%H%M%S")
        package_path = package_path.with_name(f"{package_path.stem}_{stamp}_01{package_path.suffix}")
    with zipfile.ZipFile(package_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(source_dir).as_posix())
    sha_path = package_path.with_suffix(package_path.suffix + ".sha256")
    sha_path.write_text(f"{file_sha256(package_path)}  {package_path.name}\n", encoding="utf-8")
    return package_path, sha_path


def make_review_package(out_dir: Path, release_dir: Path, package_dir_path: Path, blocker: bool = False) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "release_path.txt").write_text(str(release_dir) + "\n", encoding="utf-8")
    (out_dir / "git_commit.txt").write_text(
        __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip() + "\n",
        encoding="utf-8",
    )
    (out_dir / "git_status_porcelain.txt").write_text(
        __import__("subprocess").check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True) + "\n",
        encoding="utf-8",
    )
    name = "GPT_GCP_POINTSET_RELEASE_V1_2_PIXEL_DOMAIN_BLOCKER_20260628.zip" if blocker else "GPT_GCP_POINTSET_RELEASE_V1_2_PIXEL_DOMAIN_REVIEW_20260628.zip"
    return package_dir(out_dir, package_dir_path / name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and freeze MS-GCP v1.2 pixel-domain release.")
    parser.add_argument("--dataset_root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--release_v11", default=str(DEFAULT_RELEASE_V11))
    parser.add_argument("--final_dir", default=str(DEFAULT_DATASET_ROOT / "scenes" / "gcp_manual_annotations_v1_2"))
    parser.add_argument("--project_root", default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("--remote_light_manifest", default=str(DEFAULT_REMOTE_MANIFEST))
    parser.add_argument("--stage1_dir", default=str(DEFAULT_STAGE1_DIR))
    parser.add_argument("--review_out", default=str(DEFAULT_PROJECT_ROOT / "outputs" / "gcp_release_v1_2_pixel_domain_20260628"))
    parser.add_argument("--package_dir", default=str(DEFAULT_PROJECT_ROOT / "outputs" / "gpt_review_packages"))
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()

    final_dir = Path(args.final_dir)
    review_out = Path(args.review_out)
    package_dir_path = Path(args.package_dir)
    command_manifest = {
        "dataset_root": args.dataset_root,
        "release_v11": args.release_v11,
        "remote_light_manifest_sha256": file_sha256(Path(args.remote_light_manifest)),
        "stage1_dir": args.stage1_dir,
        "script": "code/gcp/generate_gcp_release_v1_2.py",
    }
    if final_dir.exists():
        blocker = review_out / f"blocker_existing_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        blocker.mkdir(parents=True, exist_ok=True)
        (blocker / "BLOCKER.md").write_text(f"Final v1.2 directory already exists and will not be overwritten: {final_dir}\n", encoding="utf-8")
        package_path, sha_path = make_review_package(blocker, final_dir, package_dir_path, blocker=True)
        print(json.dumps({"status": "BLOCKER", "package": str(package_path), "sha256": file_sha256(package_path)}, indent=2))
        raise SystemExit(2)

    staging = make_unique_staging(final_dir)
    compare = make_unique_staging(final_dir.with_name(final_dir.name + ".compare"))
    review_run = review_out / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    review_run.mkdir(parents=True, exist_ok=True)
    try:
        first = generate_payload(staging, args, command_manifest)
        second = generate_payload(compare, args, command_manifest)
        diff = compare_dirs(staging, compare)
        if diff:
            raise ValueError(f"byte-identical regeneration failed: {diff[:10]}")
        root_sha = file_sha256(staging / "v1_2_release_root_digest.json")
        (review_run / "v1_2_release_root_digest.json.sha256").write_text(
            f"{root_sha}  v1_2_release_root_digest.json\n",
            encoding="utf-8",
        )
        shutil.copytree(staging, review_run / "release_snapshot")
        write_json_deterministic(review_run / "generation_summary.json", {"status": "PASS", "release_dir": str(final_dir), **first})
        if args.publish:
            staging.rename(final_dir)
            shutil.rmtree(compare)
            release_dir = final_dir
        else:
            release_dir = staging
        package_path, sha_path = make_review_package(review_run, release_dir, package_dir_path, blocker=False)
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "release_dir": str(release_dir),
                    "published": bool(args.publish),
                    "package": str(package_path),
                    "package_sha256": file_sha256(package_path),
                    "root_record_sha256": root_sha,
                    "observation_count": first["observation_count"],
                    "unique_observation_ids": first["unique_observation_ids"],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    except Exception as exc:  # noqa: BLE001
        blocker = review_run / "BLOCKER"
        blocker.mkdir(parents=True, exist_ok=True)
        (blocker / "BLOCKER.md").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        if staging.exists():
            shutil.copytree(staging, blocker / "staging_snapshot", dirs_exist_ok=True)
        package_path, sha_path = make_review_package(blocker, staging, package_dir_path, blocker=True)
        print(json.dumps({"status": "BLOCKER", "error": repr(exc), "package": str(package_path), "package_sha256": file_sha256(package_path)}, indent=2))
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
