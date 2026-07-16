"""MS-GCP v1.3.0 pixel-domain release contract and evaluator gates.

The v1.3 release preserves every reviewed annotation row.  Rows without a
manual click remain auditable release records, but can never enter formal
geometry evaluation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

from gcp_pixel_domain_v1_2 import (
    CACHED_TARGET_TOL_PX,
    ORIENTATION_POLICY,
    PIXEL_CONVENTION,
    ROUNDTRIP_TOL_PX,
    SOURCE_PIXEL_DOMAIN,
    TARGET_PIXEL_DOMAIN,
    TRANSFORM_VERSION,
    CameraRecord,
    camera_provenance_lookup,
    camera_record_hash,
    canonical_record_sha256,
    colmap_camera_to_record,
    colmap_image_to_record,
    image_pose_record_hash,
    load_release_v12_sidecars,
    observation_id_from_payload,
    raw_to_target_projection,
    read_csv,
)


RELEASE_V130_SCHEMA = "ms_gcp_3dgs_benchmark_release_config_v1_3_0"
RELEASE_V130_ID = "gcp_benchmark_release_v1_3_0_multiview_control_heavy_20260717"
RELEASE_V130_TOKEN = "v1_3_0"
OBSERVATION_ID_V130_SCHEMA = "ms_gcp_observation_id_v1_3_0"

ANNOTATION_QUALITY_VALUES = {"good", "ambiguous", "not_visible"}
PROJECTION_STATUS_VALID = "valid_raw_to_benchmark_projection"
PROJECTION_STATUS_NO_CLICK = "not_applicable_no_raw_click"


def observation_id_payload_v13(
    scene: str,
    point_name: str,
    raw_image_name: str,
    raw_image_sha256: str,
    raw_manual_x_text: str,
    raw_manual_y_text: str,
) -> list[str]:
    """Return the frozen observation-ID payload.

    Clicked rows use exactly the Stage-1/v1.2 serialization.  v1.3 extends the
    same payload to reviewed no-click rows by allowing both coordinate strings
    to be empty.  A half-populated coordinate pair is always invalid.
    """

    required = [scene, point_name, raw_image_name, raw_image_sha256]
    if not all(str(value).strip() for value in required):
        raise ValueError("observation_id identity fields must be non-empty")
    x_text = str(raw_manual_x_text).strip()
    y_text = str(raw_manual_y_text).strip()
    if bool(x_text) != bool(y_text):
        raise ValueError("raw manual coordinates must be both present or both empty")
    image_sha = str(raw_image_sha256).strip().lower()
    if len(image_sha) != 64 or any(ch not in "0123456789abcdef" for ch in image_sha):
        raise ValueError("raw image SHA-256 must be lowercase hexadecimal")
    return [
        str(scene).strip(),
        str(point_name).strip(),
        Path(str(raw_image_name)).name,
        image_sha,
        x_text,
        y_text,
    ]


def observation_id_from_fields_v13(
    scene: str,
    point_name: str,
    raw_image_name: str,
    raw_image_sha256: str,
    raw_manual_x_text: str,
    raw_manual_y_text: str,
) -> str:
    return observation_id_from_payload(
        observation_id_payload_v13(
            scene,
            point_name,
            raw_image_name,
            raw_image_sha256,
            raw_manual_x_text,
            raw_manual_y_text,
        )
    )


def parse_release_bool(value: Any, field: str) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true"}:
        return True
    if text in {"0", "false"}:
        return False
    raise ValueError(f"{field} must be true/false, got {value!r}")


def _rows_by_key(rows: Sequence[dict[str, Any]], keys: Sequence[str]) -> dict[tuple[str, ...], dict[str, Any]]:
    result: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(str(row.get(field, "")) for field in keys)
        if key in result:
            raise ValueError(f"duplicate v1.3 sidecar key: {keys}={key}")
        result[key] = dict(row)
    return result


def load_release_v13_sidecars(release_base: Path) -> dict[str, Any]:
    manifest = release_base / "v1_3_0_release_file_manifest.json"
    root = release_base / "v1_3_0_release_root_digest.json"
    if not manifest.exists() or not root.exists():
        raise FileNotFoundError(f"v1.3.0 release integrity records are missing in {release_base}")
    # The v1.2 loader is token-generic once the v1.3 token is discoverable.
    sidecars = load_release_v12_sidecars(release_base)
    if sidecars.get("release_token") != RELEASE_V130_TOKEN:
        raise ValueError(f"unexpected release token: {sidecars.get('release_token')}")
    return sidecars


def load_v13_split_roles(release_base: Path) -> dict[tuple[str, str], str]:
    config_path = release_base / "gcp_benchmark_release_v1_3_0.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != RELEASE_V130_SCHEMA or config.get("release_id") != RELEASE_V130_ID:
        raise ValueError("v1.3.0 release config schema/release_id mismatch")
    split_path = release_base / str(config["split_csv"])
    roles: dict[tuple[str, str], str] = {}
    for row in read_csv(split_path):
        key = (str(row["scene"]), str(row["point_name"]))
        role = str(row["role"]).strip().lower()
        if role not in {"control", "checkpoint"}:
            raise ValueError(f"unsupported formal role for {key}: {role}")
        if key in roles:
            raise ValueError(f"duplicate split row: {key}")
        roles[key] = role
    return roles


def _verify_common_row_provenance(
    *,
    row: dict[str, str],
    scene: str,
    orientation: dict[tuple[str, ...], dict[str, Any]],
    mapping: dict[tuple[str, ...], dict[str, Any]],
    camera_lookup: dict[tuple[str, str, int], dict[str, Any]],
    pose_lookup: dict[tuple[str, str, str], dict[str, Any]],
    cameras: dict[int, CameraRecord],
    images: dict[str, Any],
    depth_manifest: dict[str, Any] | None,
) -> tuple[str, CameraRecord, CameraRecord]:
    image_name = Path(str(row["raw_image_name"])).name
    orient = orientation.get((scene, image_name))
    if orient is None:
        raise ValueError(f"missing orientation record for {scene} {image_name}")
    if str(orient.get("raw_image_sha256", "")).lower() != str(row.get("source_image_sha256", "")).lower():
        raise ValueError(f"source image SHA mismatch for {scene} {image_name}")
    if orient.get("rgb_pixel_matrix_sha256") != row.get("source_rgb_pixel_matrix_sha256"):
        raise ValueError(f"source RGB pixel-matrix hash mismatch for {scene} {image_name}")
    if orient.get("applied_orientation_policy") != ORIENTATION_POLICY or row.get("source_orientation_policy") != ORIENTATION_POLICY:
        raise ValueError(f"source orientation policy mismatch for {scene} {image_name}")

    target_image = images.get(image_name)
    if target_image is None:
        raise ValueError(f"target image missing in evaluator COLMAP model: {scene} {image_name}")
    target_camera = cameras.get(int(target_image.camera_id))
    if target_camera is None:
        raise ValueError(f"target camera missing in evaluator COLMAP model: {scene} {image_name}")
    if camera_record_hash(target_camera) != row.get("target_camera_record_sha256"):
        raise ValueError(f"target camera record hash mismatch for {scene} {image_name}")
    if image_pose_record_hash(target_image) != row.get("target_pose_record_sha256"):
        raise ValueError(f"target pose record hash mismatch for {scene} {image_name}")
    if int(row["target_image_width"]) != target_camera.width or int(row["target_image_height"]) != target_camera.height:
        raise ValueError(f"target dimensions mismatch for {scene} {image_name}")

    source_camera = CameraRecord(
        camera_id=int(row["source_camera_id"]),
        model=str(row["source_camera_model"]),
        width=int(row["source_camera_width"]),
        height=int(row["source_camera_height"]),
        params=tuple(float(x) for x in str(row["source_camera_params"]).split(";")),
    )
    if camera_record_hash(source_camera) != row.get("source_camera_record_sha256"):
        raise ValueError(f"source camera record hash mismatch for {scene} {image_name}")

    source_camera_record = camera_lookup.get((scene, "source", source_camera.camera_id))
    target_camera_record = camera_lookup.get((scene, "target", target_camera.camera_id))
    source_pose_record = pose_lookup.get((scene, "source", image_name))
    target_pose_record = pose_lookup.get((scene, "target", image_name))
    if any(record is None for record in [source_camera_record, target_camera_record, source_pose_record, target_pose_record]):
        raise ValueError(f"camera/pose provenance record missing for {scene} {image_name}")
    provenance_checks = [
        (source_camera_record, "source_camera_record_sha256"),
        (target_camera_record, "target_camera_record_sha256"),
        (source_pose_record, "source_pose_record_sha256"),
        (target_pose_record, "target_pose_record_sha256"),
    ]
    for record, field in provenance_checks:
        assert record is not None
        if str(record.get("record_sha256", "")) != str(row.get(field, "")):
            raise ValueError(f"{field} provenance mismatch for {scene} {image_name}")

    m = mapping.get((scene, image_name, image_name))
    if m is None:
        raise ValueError(f"missing source-target mapping record for {scene} {image_name}")
    unhashed = dict(m)
    claimed_hash = str(unhashed.pop("source_target_mapping_record_sha256", ""))
    if canonical_record_sha256(unhashed) != claimed_hash:
        raise ValueError(f"mapping record self-hash mismatch for {scene} {image_name}")
    if claimed_hash != row.get("source_target_mapping_record_sha256"):
        raise ValueError(f"source-target mapping hash mismatch for {scene} {image_name}")
    for field in [
        "source_image_sha256",
        "target_image_sha256",
        "source_camera_record_sha256",
        "target_camera_record_sha256",
        "source_pose_record_sha256",
        "target_pose_record_sha256",
        "mapping_type",
        "transform_version",
    ]:
        if str(m.get(field, "")) != str(row.get(field, "")):
            raise ValueError(f"mapping {field} mismatch for {scene} {image_name}")
    if not bool(m.get("pose_equivalent", False)):
        raise ValueError(f"source-target pose is not equivalent for {scene} {image_name}")
    if row.get("transform_version") != TRANSFORM_VERSION:
        raise ValueError(f"transform version mismatch for {scene} {image_name}")
    if row.get("source_pixel_domain") != SOURCE_PIXEL_DOMAIN or row.get("target_pixel_domain") != TARGET_PIXEL_DOMAIN:
        raise ValueError(f"pixel-domain mismatch for {scene} {image_name}")
    if row.get("source_pixel_convention") != PIXEL_CONVENTION or row.get("target_pixel_convention") != PIXEL_CONVENTION:
        raise ValueError(f"pixel convention mismatch for {scene} {image_name}")
    if depth_manifest is not None:
        if str(depth_manifest.get("target_cameras_bin_sha256", "")) != row.get("target_cameras_bin_sha256"):
            raise ValueError(f"depth manifest target cameras hash mismatch for {scene} {image_name}")
        if str(depth_manifest.get("target_images_bin_sha256", "")) != row.get("target_images_bin_sha256"):
            raise ValueError(f"depth manifest target images hash mismatch for {scene} {image_name}")
        if str(depth_manifest.get("pixel_coordinate_convention", "")) != PIXEL_CONVENTION:
            raise ValueError(f"depth manifest pixel convention mismatch for {scene} {image_name}")
    return image_name, source_camera, target_camera


def validate_release_v13_rows_for_evaluator(
    *,
    release_base: Path,
    scene: str,
    rows: Sequence[dict[str, str]],
    colmap_cameras: dict[int, Any],
    colmap_images: dict[int, Any],
    depth_manifest: dict[str, Any] | None = None,
    return_all_rows: bool = False,
) -> list[dict[str, str]]:
    """Validate every v1.3 row and return only formal rows by default."""

    sidecars = load_release_v13_sidecars(release_base)
    orientation = _rows_by_key(sidecars["orientation"], ["scene", "image_name"])
    mapping = _rows_by_key(sidecars["mapping"], ["scene", "source_image_name", "target_image_name"])
    camera_lookup, pose_lookup = camera_provenance_lookup(sidecars["camera"])
    cameras = {int(cid): colmap_camera_to_record(int(cid), cam) for cid, cam in colmap_cameras.items()}
    images = {Path(str(img.name)).name: colmap_image_to_record(int(iid), img) for iid, img in colmap_images.items()}
    split_roles = load_v13_split_roles(release_base)
    if depth_manifest is not None:
        for field in ["target_cameras_bin_sha256", "target_images_bin_sha256", "pixel_coordinate_convention"]:
            if field not in depth_manifest:
                raise ValueError(f"v1.3 release mode requires depth manifest field {field}")

    validated_all: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    seen_keys: set[tuple[str, str, str]] = set()
    for source_row in rows:
        row = dict(source_row)
        if row.get("scene") != scene:
            raise ValueError(f"annotation scene mismatch: expected {scene}, got {row.get('scene')}")
        image_name = Path(str(row["raw_image_name"])).name
        key = (scene, str(row["point_name"]), image_name)
        if key in seen_keys:
            raise ValueError(f"duplicate v1.3 observation key: {key}")
        seen_keys.add(key)
        observation_id = observation_id_from_fields_v13(
            scene,
            str(row["point_name"]),
            image_name,
            str(row["source_image_sha256"]),
            str(row.get("raw_manual_x", "")),
            str(row.get("raw_manual_y", "")),
        )
        if observation_id != row.get("observation_id"):
            raise ValueError(f"observation_id mismatch for {key}")
        if observation_id in seen_ids:
            raise ValueError(f"duplicate v1.3 observation_id: {observation_id}")
        seen_ids.add(observation_id)

        image_name, source_camera, target_camera = _verify_common_row_provenance(
            row=row,
            scene=scene,
            orientation=orientation,
            mapping=mapping,
            camera_lookup=camera_lookup,
            pose_lookup=pose_lookup,
            cameras=cameras,
            images=images,
            depth_manifest=depth_manifest,
        )
        quality = str(row.get("annotation_quality", "")).strip().lower()
        if quality not in ANNOTATION_QUALITY_VALUES:
            raise ValueError(f"unsupported annotation quality for {key}: {quality}")
        visible = parse_release_bool(row.get("annotation_visible"), "annotation_visible")
        x_text = str(row.get("raw_manual_x", "")).strip()
        y_text = str(row.get("raw_manual_y", "")).strip()
        has_click = bool(x_text) and bool(y_text)
        expected_good = quality == "good" and visible and has_click
        annotation_good = parse_release_bool(row.get("annotation_good"), "annotation_good")
        if annotation_good != expected_good:
            raise ValueError(f"annotation_good mismatch for {key}")
        split_role = split_roles.get((scene, str(row["point_name"])))
        expected_formal = expected_good and split_role in {"control", "checkpoint"}
        formal_eligible = parse_release_bool(row.get("formal_eligible"), "formal_eligible")
        if formal_eligible != expected_formal:
            raise ValueError(f"formal_eligible mismatch for {key}")
        expected_role = split_role if expected_formal else "not_formal"
        if str(row.get("formal_role", "")) != expected_role:
            raise ValueError(f"formal_role mismatch for {key}")

        out = dict(row)
        if has_click:
            for value, field in [(x_text, "raw_manual_x"), (y_text, "raw_manual_y")]:
                if not math.isfinite(float(value)):
                    raise ValueError(f"non-finite {field} for {key}")
            projection = raw_to_target_projection(source_camera, target_camera, float(x_text), float(y_text))
            if row.get("projection_status") != PROJECTION_STATUS_VALID:
                raise ValueError(f"projection_status mismatch for clicked row {key}")
            dx = abs(projection["target_x"] - float(row["target_x"]))
            dy = abs(projection["target_y"] - float(row["target_y"]))
            if dx > CACHED_TARGET_TOL_PX or dy > CACHED_TARGET_TOL_PX:
                raise ValueError(f"cached target projection mismatch for {key}: dx={dx} dy={dy}")
            if projection["roundtrip_error_px"] > ROUNDTRIP_TOL_PX:
                raise ValueError(f"roundtrip error exceeds tolerance for {key}")
            if not (0.0 <= float(row["target_x"]) < target_camera.width and 0.0 <= float(row["target_y"]) < target_camera.height):
                raise ValueError(f"target coordinates out of bounds for {key}")
            if formal_eligible:
                out["u_px"] = row["target_x"]
                out["v_px"] = row["target_y"]
                out["manual_x"] = row["target_x"]
                out["manual_y"] = row["target_y"]
                out["image_name"] = row.get("target_image_name", image_name)
        else:
            if quality != "not_visible" or visible or annotation_good or formal_eligible:
                raise ValueError(f"no-click row has invalid status for {key}")
            if row.get("projection_status") != PROJECTION_STATUS_NO_CLICK:
                raise ValueError(f"projection_status mismatch for no-click row {key}")
            for field in [
                "normalized_x",
                "normalized_y",
                "normalized_unit_ray_x",
                "normalized_unit_ray_y",
                "normalized_unit_ray_z",
                "target_x",
                "target_y",
                "roundtrip_raw_x",
                "roundtrip_raw_y",
                "roundtrip_error_px",
            ]:
                if str(row.get(field, "")).strip():
                    raise ValueError(f"no-click row must leave {field} empty for {key}")
        validated_all.append(out)

    if return_all_rows:
        return validated_all
    return [row for row in validated_all if parse_release_bool(row["formal_eligible"], "formal_eligible")]
