from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gcp_pixel_domain_v1_2 import (  # noqa: E402
    CACHED_TARGET_TOL_PX,
    ORIENTATION_POLICY,
    PIXEL_CONVENTION,
    RELEASE_V122_ID,
    ROUNDTRIP_TOL_PX,
    SCENES,
    TARGET_PIXEL_DOMAIN,
    TRANSFORM_VERSION,
    CameraRecord,
    ImageRecord,
    camera_canonical_record,
    camera_record_hash,
    canonical_record_sha256,
    image_pose_canonical_record,
    image_pose_record_hash,
    load_manifest_model,
    load_release_v12_sidecars,
    observation_id_from_fields,
    payload_manifest_entries,
    payload_root_digest,
    raw_to_target_projection,
    read_csv,
    relative_posix,
    release_sidecar_name,
    serialize_observation_id_payload,
    serialize_rgb_pixel_matrix,
    sha256_bytes,
    validate_release_v12_rows_for_evaluator,
    verify_payload_integrity,
    write_json_deterministic,
)
from evaluate_gaussian_gcp_geometry import (  # noqa: E402
    load_release_config,
    pixel_domain_release_layout,
    release_annotation_name_for_scene,
    release_file_registry,
    require_release_registry_file,
    verify_release_files,
)

RELEASE_TOKEN = "v1_2_2"
DEFAULT_RELEASE_V12 = Path(r"E:\datasets\M3M-GCP\scenes\gcp_manual_annotations_v1_2")
DEFAULT_RELEASE_V122 = Path(r"E:\datasets\M3M-GCP\scenes\gcp_manual_annotations_v1_2_2")
DEFAULT_RELEASE_V121 = Path(r"E:\datasets\M3M-GCP\scenes\gcp_manual_annotations_v1_2_1")
DEFAULT_REMOTE_MANIFEST = Path(
    r"E:\M3M-GCP-3DGS\outputs\gcp_6scene_annotation_domain_inputs_20260628\gcp_6scene_annotation_domain_jsonlight_20260628\remote_light_manifest.json"
)


def _write_payload_manifests(root: Path) -> None:
    entries = payload_manifest_entries(
        root,
        exclude={f"{RELEASE_TOKEN}_release_file_manifest.json", f"{RELEASE_TOKEN}_release_root_digest.json"},
    )
    write_json_deterministic(
        root / f"{RELEASE_TOKEN}_release_file_manifest.json",
        {
            "schema": "ms_gcp_release_payload_manifest_v1",
            "release_id": RELEASE_V122_ID,
            "files": entries,
        },
    )
    manifest_sha = _sha(root / f"{RELEASE_TOKEN}_release_file_manifest.json")
    write_json_deterministic(
        root / f"{RELEASE_TOKEN}_release_root_digest.json",
        {
            "schema": "ms_gcp_release_root_digest_v1",
            "release_id": RELEASE_V122_ID,
            "payload_file_count": len(entries),
            "payload_manifest_path": f"{RELEASE_TOKEN}_release_file_manifest.json",
            "payload_manifest_sha256": manifest_sha,
            "payload_root_digest_sha256": payload_root_digest(entries),
        },
    )


def _sha(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _make_release_fixture(root: Path) -> tuple[list[dict[str, str]], dict[int, Any], dict[int, Any], dict[str, Any]]:
    scene = "scene_a"
    image_name = "DJI_001.JPG"
    raw_sha = "a" * 64
    rgb_sha = "b" * 64
    target_sha = "c" * 64
    source_cameras_sha = "1" * 64
    source_images_sha = "2" * 64
    target_cameras_sha = "3" * 64
    target_images_sha = "4" * 64
    source_cam = CameraRecord(1, "SIMPLE_RADIAL", 1000, 800, (500.0, 500.0, 400.0, 0.01))
    target_cam = CameraRecord(2, "PINHOLE", 1000, 800, (500.0, 500.0, 500.0, 400.0))
    source_img = ImageRecord(1, image_name, 1, (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    target_img = ImageRecord(1, image_name, 2, (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    projection = raw_to_target_projection(source_cam, target_cam, 520.0, 410.0)
    mapping_payload = {
        "scene": scene,
        "source_camera_id": 1,
        "source_camera_record_sha256": camera_record_hash(source_cam),
        "source_image_id": 1,
        "source_image_name": image_name,
        "source_image_sha256": raw_sha,
        "source_pose_record_sha256": image_pose_record_hash(source_img),
        "target_camera_id": 2,
        "target_camera_record_sha256": camera_record_hash(target_cam),
        "target_image_id": 1,
        "target_image_name": image_name,
        "target_image_sha256": target_sha,
        "target_pose_record_sha256": image_pose_record_hash(target_img),
        "transform_version": TRANSFORM_VERSION,
        "mapping_type": "pose_equivalent_intrinsics_projection",
        "pose_equivalent": True,
    }
    mapping_hash = canonical_record_sha256(mapping_payload)
    obs_id = observation_id_from_fields(
        scene=scene,
        point_name="P1",
        raw_image_name=image_name,
        raw_image_sha256=raw_sha,
        raw_manual_x_text="520.0",
        raw_manual_y_text="410.0",
    )
    row = {
        "observation_id": obs_id,
        "scene": scene,
        "point_name": "P1",
        "raw_image_name": image_name,
        "raw_manual_x": "520.0",
        "raw_manual_y": "410.0",
        "source_image_sha256": raw_sha,
        "source_orientation_policy": ORIENTATION_POLICY,
        "source_rgb_pixel_matrix_sha256": rgb_sha,
        "source_camera_id": "1",
        "source_camera_model": source_cam.model,
        "source_camera_width": str(source_cam.width),
        "source_camera_height": str(source_cam.height),
        "source_camera_params": ";".join(str(x) for x in source_cam.params),
        "source_camera_record_sha256": camera_record_hash(source_cam),
        "source_pose_record_sha256": image_pose_record_hash(source_img),
        "source_cameras_bin_sha256": source_cameras_sha,
        "source_images_bin_sha256": source_images_sha,
        "target_image_name": image_name,
        "target_pixel_domain": TARGET_PIXEL_DOMAIN,
        "target_pixel_convention": PIXEL_CONVENTION,
        "target_image_width": str(target_cam.width),
        "target_image_height": str(target_cam.height),
        "target_image_sha256": target_sha,
        "target_camera_id": "2",
        "target_camera_model": target_cam.model,
        "target_camera_width": str(target_cam.width),
        "target_camera_height": str(target_cam.height),
        "target_camera_params": ";".join(str(x) for x in target_cam.params),
        "target_camera_record_sha256": camera_record_hash(target_cam),
        "target_pose_record_sha256": image_pose_record_hash(target_img),
        "target_cameras_bin_sha256": target_cameras_sha,
        "target_images_bin_sha256": target_images_sha,
        "target_x": f"{projection['target_x']:.17g}",
        "target_y": f"{projection['target_y']:.17g}",
        "mapping_type": "pose_equivalent_intrinsics_projection",
        "transform_version": TRANSFORM_VERSION,
        "source_target_mapping_record_sha256": mapping_hash,
        "roundtrip_error_px": f"{projection['roundtrip_error_px']:.17g}",
    }
    write_json_deterministic(
        root / f"raw_image_orientation_manifest_{RELEASE_TOKEN}.json",
        [
            {
                "scene": scene,
                "image_name": image_name,
                "raw_image_sha256": raw_sha,
                "rgb_pixel_matrix_sha256": rgb_sha,
                "applied_orientation_policy": ORIENTATION_POLICY,
            }
        ],
    )
    write_json_deterministic(
        root / f"source_target_mapping_manifest_{RELEASE_TOKEN}.json",
        [
            {
                **mapping_payload,
                "source_target_mapping_record_sha256": mapping_hash,
            }
        ],
    )
    write_json_deterministic(root / f"projection_manifest_{RELEASE_TOKEN}.json", {"schema": "fixture"})
    write_json_deterministic(
        root / f"camera_provenance_manifest_{RELEASE_TOKEN}.json",
        {
            "schema": "fixture",
            "scenes": {
                scene: {
                    "source_model": {
                        "cameras": [{**camera_canonical_record(source_cam), "record_sha256": camera_record_hash(source_cam)}],
                        "images": [{**image_pose_canonical_record(source_img), "record_sha256": image_pose_record_hash(source_img)}],
                    },
                    "target_model": {
                        "cameras": [{**camera_canonical_record(target_cam), "record_sha256": camera_record_hash(target_cam)}],
                        "images": [{**image_pose_canonical_record(target_img), "record_sha256": image_pose_record_hash(target_img)}],
                    },
                }
            },
        },
    )
    _write_payload_manifests(root)
    cameras = {
        2: SimpleNamespace(model="PINHOLE", width=1000, height=800, params=np.asarray(target_cam.params)),
    }
    images = {
        1: SimpleNamespace(name=image_name, camera_id=2, qvec=np.asarray(target_img.qvec), tvec=np.asarray(target_img.tvec)),
    }
    depth_manifest = {
        "target_cameras_bin_sha256": target_cameras_sha,
        "target_images_bin_sha256": target_images_sha,
        "pixel_coordinate_convention": PIXEL_CONVENTION,
    }
    return [row], cameras, images, depth_manifest


def _expect_fail(fn: Callable[[], Any]) -> str:
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        return type(exc).__name__
    raise AssertionError("expected failure")


def test_golden_serialization() -> dict[str, Any]:
    fields = ["scene_a", "P1", "im.jpg", "a" * 64, "123.450", "67.890"]
    got = serialize_observation_id_payload(fields)
    expected_hex = (
        "5b227363656e655f61222c225031222c22696d2e6a7067222c22"
        + ("61" * 64)
        + "222c223132332e343530222c2236372e383930225d"
    )
    if got.hex() != expected_hex:
        raise AssertionError(got.hex())
    oid = sha256_bytes(got)
    expected_oid = "8ba0c679ad61822732a0c27e1370a97d978a28b19769b26fad1662f4be4bcda4"
    if oid != expected_oid:
        raise AssertionError(oid)
    return {"serialized_hex": got.hex(), "observation_id": oid}


def test_golden_pixel_matrix_hash() -> dict[str, Any]:
    rgb = bytes([255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255])
    serialized = serialize_rgb_pixel_matrix("RGB", 2, 2, rgb)
    expected_hex = "4d535f4743505f5247425f4d41545249585f484153485f56310003005247420200000002000000ff000000ff000000ffffffff"
    if serialized.hex() != expected_hex:
        raise AssertionError(serialized.hex())
    expected_sha = "f8aae1bde814079b0e942b7b67068b4b1e93c8dc56482cfd9dcaad297f97dada"
    got_sha = sha256_bytes(serialized)
    if got_sha != expected_sha:
        raise AssertionError(got_sha)
    return {"serialized_hex": serialized.hex(), "sha256": got_sha}


def test_projection_roundtrip() -> dict[str, Any]:
    source = CameraRecord(1, "SIMPLE_RADIAL", 1000, 800, (500.0, 500.0, 400.0, 0.01))
    target = CameraRecord(2, "PINHOLE", 1000, 800, (500.0, 500.0, 500.0, 400.0))
    p = raw_to_target_projection(source, target, 520.0, 410.0)
    if p["roundtrip_error_px"] > ROUNDTRIP_TOL_PX:
        raise AssertionError(p)
    bad = CameraRecord(3, "OPENCV", 1000, 800, (1.0, 2.0, 3.0, 4.0))
    rejected = _expect_fail(lambda: raw_to_target_projection(bad, target, 520.0, 410.0))
    return {"target_x": p["target_x"], "target_y": p["target_y"], "unsupported_model_rejected": rejected}


def test_integrity_manifest_rejections() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "payload.txt").write_text("ok\n", encoding="utf-8")
        _write_payload_manifests(root)
        ok = verify_payload_integrity(root, root / f"{RELEASE_TOKEN}_release_file_manifest.json", root / f"{RELEASE_TOKEN}_release_root_digest.json")
        if not ok["passed"]:
            raise AssertionError(ok)
        (root / "payload.txt").write_text("changed\n", encoding="utf-8")
        modified = verify_payload_integrity(root, root / f"{RELEASE_TOKEN}_release_file_manifest.json", root / f"{RELEASE_TOKEN}_release_root_digest.json")
        (root / "payload.txt").write_text("ok\n", encoding="utf-8")
        (root / "unregistered.txt").write_text("x\n", encoding="utf-8")
        unregistered = verify_payload_integrity(root, root / f"{RELEASE_TOKEN}_release_file_manifest.json", root / f"{RELEASE_TOKEN}_release_root_digest.json")
    if modified["passed"] or unregistered["passed"]:
        raise AssertionError({"modified": modified, "unregistered": unregistered})
    return {"modified_problem_count": modified["problem_count"], "unregistered_problem_count": unregistered["problem_count"]}


def test_evaluator_v12_hard_gates() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        rows, cameras, images, depth_manifest = _make_release_fixture(root)
        validated = validate_release_v12_rows_for_evaluator(
            release_base=root,
            scene="scene_a",
            rows=rows,
            colmap_cameras=cameras,
            colmap_images=images,
            depth_manifest=depth_manifest,
        )
        if validated[0]["manual_x"] != rows[0]["target_x"] or validated[0]["image_name"] != rows[0]["target_image_name"]:
            raise AssertionError(validated[0])
        failures: dict[str, str] = {}
        for name, mutate in {
            "target_x_tamper": lambda r: r.__setitem__("target_x", str(float(r["target_x"]) + CACHED_TARGET_TOL_PX * 10.0)),
            "target_y_tamper": lambda r: r.__setitem__("target_y", str(float(r["target_y"]) + CACHED_TARGET_TOL_PX * 10.0)),
            "raw_pixel_modified": lambda r: r.__setitem__("raw_manual_x", "521.0"),
            "mapping_hash_modified": lambda r: r.__setitem__("source_target_mapping_record_sha256", "e" * 64),
            "wrong_target_camera_hash": lambda r: r.__setitem__("target_camera_record_sha256", "f" * 64),
        }.items():
            bad = [dict(rows[0])]
            mutate(bad[0])
            failures[name] = _expect_fail(
                lambda bad=bad: validate_release_v12_rows_for_evaluator(
                    release_base=root,
                    scene="scene_a",
                    rows=bad,
                    colmap_cameras=cameras,
                    colmap_images=images,
                    depth_manifest=depth_manifest,
                )
            )
        bad_manifest = dict(depth_manifest)
        bad_manifest["target_cameras_bin_sha256"] = "9" * 64
        failures["packet_camera_hash_modified"] = _expect_fail(
            lambda: validate_release_v12_rows_for_evaluator(
                release_base=root,
                scene="scene_a",
                rows=rows,
                colmap_cameras=cameras,
                colmap_images=images,
                depth_manifest=bad_manifest,
            )
        )
    return {"valid_row_count": len(validated), "negative_failures": failures}


def _validate_fixture(
    root: Path,
    rows: list[dict[str, str]],
    cameras: dict[int, Any],
    images: dict[int, Any],
    depth_manifest: dict[str, Any],
) -> list[dict[str, str]]:
    return validate_release_v12_rows_for_evaluator(
        release_base=root,
        scene="scene_a",
        rows=rows,
        colmap_cameras=cameras,
        colmap_images=images,
        depth_manifest=depth_manifest,
    )


def hard_gate_case(
    name: str,
    *,
    mutate_row: Callable[[dict[str, str]], None] | None = None,
    mutate_depth_manifest: Callable[[dict[str, Any]], None] | None = None,
    mutate_cameras: Callable[[dict[int, Any]], None] | None = None,
    mutate_images: Callable[[dict[int, Any]], None] | None = None,
    mutate_sidecars: Callable[[Path], None] | None = None,
    refresh_integrity_after_sidecar_mutation: bool = False,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        rows, cameras, images, depth_manifest = _make_release_fixture(root)
        if mutate_row:
            mutate_row(rows[0])
        if mutate_depth_manifest:
            mutate_depth_manifest(depth_manifest)
        if mutate_cameras:
            mutate_cameras(cameras)
        if mutate_images:
            mutate_images(images)
        if mutate_sidecars:
            mutate_sidecars(root)
            if refresh_integrity_after_sidecar_mutation:
                _write_payload_manifests(root)
        error_type = _expect_fail(lambda: _validate_fixture(root, rows, cameras, images, depth_manifest))
    return {"case": name, "rejected_by": error_type}


def hard_gate_cases() -> list[tuple[str, Callable[[], dict[str, Any]]]]:
    return [
        ("target_x_tamper", lambda: hard_gate_case("target_x_tamper", mutate_row=lambda r: r.__setitem__("target_x", str(float(r["target_x"]) + CACHED_TARGET_TOL_PX * 10)))),
        ("target_y_tamper", lambda: hard_gate_case("target_y_tamper", mutate_row=lambda r: r.__setitem__("target_y", str(float(r["target_y"]) + CACHED_TARGET_TOL_PX * 10)))),
        ("coordinates_from_another_image", lambda: hard_gate_case("coordinates_from_another_image", mutate_row=lambda r: (r.__setitem__("target_x", str(float(r["target_x"]) + 2.0)), r.__setitem__("target_y", str(float(r["target_y"]) + 2.0))))),
        ("canonical_raw_pixel_tamper", lambda: hard_gate_case("canonical_raw_pixel_tamper", mutate_row=lambda r: r.__setitem__("raw_manual_x", "521.0"))),
        ("observation_id_tamper", lambda: hard_gate_case("observation_id_tamper", mutate_row=lambda r: r.__setitem__("observation_id", "0" * 64))),
        ("source_image_hash_tamper", lambda: hard_gate_case("source_image_hash_tamper", mutate_row=lambda r: r.__setitem__("source_image_sha256", "1" * 64))),
        ("source_rgb_matrix_hash_tamper", lambda: hard_gate_case("source_rgb_matrix_hash_tamper", mutate_row=lambda r: r.__setitem__("source_rgb_pixel_matrix_sha256", "1" * 64))),
        ("source_camera_record_hash_tamper", lambda: hard_gate_case("source_camera_record_hash_tamper", mutate_row=lambda r: r.__setitem__("source_camera_record_sha256", "1" * 64))),
        ("source_pose_record_hash_tamper", lambda: hard_gate_case("source_pose_record_hash_tamper", mutate_row=lambda r: r.__setitem__("source_pose_record_sha256", "1" * 64))),
        ("target_image_hash_tamper", lambda: hard_gate_case("target_image_hash_tamper", mutate_row=lambda r: r.__setitem__("target_image_sha256", "1" * 64))),
        ("target_camera_record_hash_tamper", lambda: hard_gate_case("target_camera_record_hash_tamper", mutate_row=lambda r: r.__setitem__("target_camera_record_sha256", "1" * 64))),
        ("target_pose_record_hash_tamper", lambda: hard_gate_case("target_pose_record_hash_tamper", mutate_row=lambda r: r.__setitem__("target_pose_record_sha256", "1" * 64))),
        ("mapping_record_hash_tamper", lambda: hard_gate_case("mapping_record_hash_tamper", mutate_row=lambda r: r.__setitem__("source_target_mapping_record_sha256", "1" * 64))),
        ("pose_mismatch", lambda: hard_gate_case("pose_mismatch", mutate_images=lambda images: setattr(images[1], "qvec", np.asarray([0.999, 0.001, 0.0, 0.0])))),
        ("zero_one_based_offset", lambda: hard_gate_case("zero_one_based_offset", mutate_row=lambda r: (r.__setitem__("target_x", str(float(r["target_x"]) + 1.0)), r.__setitem__("target_y", str(float(r["target_y"]) + 1.0))))),
        ("xy_swap", lambda: hard_gate_case("xy_swap", mutate_row=lambda r: (r.__setitem__("target_x", r["target_y"]), r.__setitem__("target_y", r["target_x"])))),
        ("resize_dimension_mismatch", lambda: hard_gate_case("resize_dimension_mismatch", mutate_row=lambda r: r.__setitem__("target_image_width", "999"))),
        ("target_out_of_bounds", lambda: hard_gate_case("target_out_of_bounds", mutate_row=lambda r: (r.__setitem__("target_x", "1001"), r.__setitem__("target_y", "801")))),
        ("missing_mapping", lambda: hard_gate_case("missing_mapping", mutate_sidecars=lambda root: (root / f"source_target_mapping_manifest_{RELEASE_TOKEN}.json").write_text("[]\n", encoding="utf-8"), refresh_integrity_after_sidecar_mutation=True)),
        ("unknown_transform_version", lambda: hard_gate_case("unknown_transform_version", mutate_row=lambda r: r.__setitem__("transform_version", "unknown_transform"))),
        ("packet_camera_hash_mismatch", lambda: hard_gate_case("packet_camera_hash_mismatch", mutate_depth_manifest=lambda m: m.__setitem__("target_cameras_bin_sha256", "9" * 64))),
        ("packet_pixel_convention_mismatch", lambda: hard_gate_case("packet_pixel_convention_mismatch", mutate_depth_manifest=lambda m: m.__setitem__("pixel_coordinate_convention", "one_based_pixels"))),
        ("exif_orientation_policy_mismatch", lambda: hard_gate_case("exif_orientation_policy_mismatch", mutate_row=lambda r: r.__setitem__("source_orientation_policy", "apply_exif_transpose"))),
        (
            "camera_provenance_vs_csv_hash_namespace_mismatch",
            lambda: hard_gate_case(
                "camera_provenance_vs_csv_hash_namespace_mismatch",
                mutate_sidecars=lambda root: _tamper_camera_provenance(root, "source_model", "cameras", "record_sha256", "2" * 64),
                refresh_integrity_after_sidecar_mutation=True,
            ),
        ),
        (
            "camera_provenance_vs_mapping_hash_namespace_mismatch",
            lambda: hard_gate_case(
                "camera_provenance_vs_mapping_hash_namespace_mismatch",
                mutate_sidecars=lambda root: _tamper_mapping(root, "source_camera_record_sha256", "2" * 64),
                refresh_integrity_after_sidecar_mutation=True,
            ),
        ),
    ]


def _tamper_camera_provenance(root: Path, model_key: str, record_group: str, field: str, value: str) -> None:
    path = root / f"camera_provenance_manifest_{RELEASE_TOKEN}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["scenes"]["scene_a"][model_key][record_group][0][field] = value
    write_json_deterministic(path, payload)


def _tamper_mapping(root: Path, field: str, value: str) -> None:
    path = root / f"source_target_mapping_manifest_{RELEASE_TOKEN}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[0][field] = value
    # Make the mapping self-hash valid so the namespace mismatch is specifically
    # between mapping and camera provenance, not a stale mapping hash.
    record = dict(payload[0])
    record.pop("source_target_mapping_record_sha256", None)
    payload[0]["source_target_mapping_record_sha256"] = canonical_record_sha256(record)
    write_json_deterministic(path, payload)


def test_mapping_unique_key_rejections() -> dict[str, Any]:
    results = {}
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        rows, cameras, images, depth_manifest = _make_release_fixture(root)
        mapping_path = root / f"source_target_mapping_manifest_{RELEASE_TOKEN}.json"
        mapping_rows = json.loads(mapping_path.read_text(encoding="utf-8"))
        mapping_rows.append(dict(mapping_rows[0]))
        write_json_deterministic(mapping_path, mapping_rows)
        _write_payload_manifests(root)
        results["identical_duplicate_mapping_injection"] = _expect_fail(
            lambda: validate_release_v12_rows_for_evaluator(
                release_base=root,
                scene="scene_a",
                rows=rows,
                colmap_cameras=cameras,
                colmap_images=images,
                depth_manifest=depth_manifest,
            )
        )
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        rows, cameras, images, depth_manifest = _make_release_fixture(root)
        mapping_path = root / f"source_target_mapping_manifest_{RELEASE_TOKEN}.json"
        mapping_rows = json.loads(mapping_path.read_text(encoding="utf-8"))
        bad = dict(mapping_rows[0])
        bad["target_image_sha256"] = "d" * 64
        payload = dict(bad)
        payload.pop("source_target_mapping_record_sha256", None)
        bad["source_target_mapping_record_sha256"] = canonical_record_sha256(payload)
        mapping_rows.append(bad)
        write_json_deterministic(mapping_path, mapping_rows)
        _write_payload_manifests(root)
        results["conflicting_duplicate_mapping_injection"] = _expect_fail(
            lambda: validate_release_v12_rows_for_evaluator(
                release_base=root,
                scene="scene_a",
                rows=rows,
                colmap_cameras=cameras,
                colmap_images=images,
                depth_manifest=depth_manifest,
            )
        )
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        rows, cameras, images, depth_manifest = _make_release_fixture(root)
        rows[0]["source_target_mapping_record_sha256"] = "9" * 64
        results["observation_wrong_mapping_reference"] = _expect_fail(
            lambda: validate_release_v12_rows_for_evaluator(
                release_base=root,
                scene="scene_a",
                rows=rows,
                colmap_cameras=cameras,
                colmap_images=images,
                depth_manifest=depth_manifest,
            )
        )
    return results


def _colmap_records_for_validator(cameras: dict[int, CameraRecord], images: dict[str, ImageRecord]) -> tuple[dict[int, Any], dict[int, Any]]:
    camera_objects = {
        int(cid): SimpleNamespace(model=cam.model, width=int(cam.width), height=int(cam.height), params=np.asarray(cam.params))
        for cid, cam in cameras.items()
    }
    image_objects = {
        int(img.image_id): SimpleNamespace(
            name=img.image_name,
            camera_id=int(img.camera_id),
            qvec=np.asarray(img.qvec),
            tvec=np.asarray(img.tvec),
        )
        for img in images.values()
    }
    return camera_objects, image_objects


def test_actual_release_smoke(release_dir: Path, v121_dir: Path, remote_manifest_path: Path) -> dict[str, Any]:
    if not release_dir.exists():
        raise FileNotFoundError(f"real release directory missing: {release_dir}")
    if not v121_dir.exists():
        raise FileNotFoundError(f"v1.2.1 comparison directory missing: {v121_dir}")
    remote_manifest = json.loads(remote_manifest_path.read_text(encoding="utf-8"))
    sidecars = load_release_v12_sidecars(release_dir)
    token = str(sidecars["release_token"])
    mapping_rows = list(sidecars["mapping"])
    primary_keys = [
        (
            str(row["scene"]),
            str(row["source_image_id"]),
            str(row["source_image_name"]),
            str(row["target_image_id"]),
            str(row["target_image_name"]),
        )
        for row in mapping_rows
    ]
    evaluator_keys = [(str(row["scene"]), str(row["source_image_name"]), str(row["target_image_name"])) for row in mapping_rows]
    if len(primary_keys) != len(set(primary_keys)):
        raise AssertionError("actual release has duplicate mapping primary keys")
    if len(evaluator_keys) != len(set(evaluator_keys)):
        raise AssertionError("actual release has duplicate evaluator mapping keys")
    if len(mapping_rows) != 420:
        raise AssertionError(f"unexpected mapping record count: {len(mapping_rows)}")
    mapping_hashes = {str(row["source_target_mapping_record_sha256"]) for row in mapping_rows}
    scene_counts = {}
    validated_total = 0
    observation_reference_total = 0
    target_coordinate_max_error = 0.0
    observation_id_mismatch_count = 0
    for scene in SCENES:
        rows = read_csv(release_dir / f"{scene}_gcp_annotations_pixel_domain_{token}.csv")
        rows121 = read_csv(v121_dir / f"{scene}_gcp_annotations_pixel_domain_v1_2_1.csv")
        by_oid121 = {row["observation_id"]: row for row in rows121}
        for row in rows:
            observation_reference_total += 1
            if row["source_target_mapping_record_sha256"] not in mapping_hashes:
                raise AssertionError(f"observation mapping reference not found: {scene} {row['observation_id']}")
            old = by_oid121.get(row["observation_id"])
            if old is None:
                observation_id_mismatch_count += 1
                continue
            target_coordinate_max_error = max(
                target_coordinate_max_error,
                abs(float(row["target_x"]) - float(old["target_x"])),
                abs(float(row["target_y"]) - float(old["target_y"])),
            )
        target_cameras, target_images, _target_model = load_manifest_model(remote_manifest["scenes"][scene], "target_model")
        camera_objects, image_objects = _colmap_records_for_validator(target_cameras, target_images)
        validated = validate_release_v12_rows_for_evaluator(
            release_base=release_dir,
            scene=scene,
            rows=rows,
            colmap_cameras=camera_objects,
            colmap_images=image_objects,
            depth_manifest=None,
        )
        if len(validated) != len(rows):
            raise AssertionError(f"validated row count mismatch for {scene}: {len(validated)} != {len(rows)}")
        scene_counts[scene] = len(validated)
        validated_total += len(validated)
    if observation_id_mismatch_count:
        raise AssertionError(f"observation IDs missing from v1.2.1 comparison: {observation_id_mismatch_count}")
    if target_coordinate_max_error > CACHED_TARGET_TOL_PX:
        raise AssertionError(f"v1.2.2 target coordinates differ from v1.2.1: {target_coordinate_max_error}")
    if validated_total != 611 or observation_reference_total != 611:
        raise AssertionError({"validated_total": validated_total, "observation_reference_total": observation_reference_total})
    return {
        "mapping_record_count": len(mapping_rows),
        "observation_reference_count": observation_reference_total,
        "validated_total": validated_total,
        "scene_counts": scene_counts,
        "target_coordinate_max_error_px": target_coordinate_max_error,
        "observation_id_mismatch_count": observation_id_mismatch_count,
    }


def test_formal_loader_layouts(
    release_dir: Path,
    v12_dir: Path = DEFAULT_RELEASE_V12,
    v121_dir: Path = DEFAULT_RELEASE_V121,
) -> dict[str, Any]:
    expected = {
        v12_dir: ("v1_2", "pixel_domain_v1_2.csv", "gcp_benchmark_release_v1_2.json"),
        v121_dir: ("v1_2_1", "pixel_domain_v1_2_1.csv", "gcp_benchmark_release_v1_2_1.json"),
        release_dir: ("v1_2_2", "pixel_domain_v1_2_2.csv", "gcp_benchmark_release_v1_2_2.json"),
    }
    layout_results = {}
    for root, (token, suffix, config_name) in expected.items():
        if not root.exists():
            raise FileNotFoundError(f"release directory missing for loader layout test: {root}")
        config_path = root / config_name
        config = load_release_config(config_path)
        layout = pixel_domain_release_layout(config)
        if layout["token"] != token:
            raise AssertionError(f"{root.name} token mismatch: {layout['token']} != {token}")
        if layout["annotation_suffix"] != suffix:
            raise AssertionError(f"{root.name} annotation suffix mismatch: {layout['annotation_suffix']} != {suffix}")
        verified = verify_release_files(config_path, config)
        registry = release_file_registry(verified)
        manifest_name = f"{token}_release_file_manifest.json"
        root_name = f"{token}_release_root_digest.json"
        if manifest_name not in registry:
            raise AssertionError(f"{root.name} payload manifest not registered under actual name: {manifest_name}")
        if root_name not in registry:
            raise AssertionError(f"{root.name} root digest not registered under actual name: {root_name}")
        annotation_name = release_annotation_name_for_scene(config, "gcp_3000_20260602")
        if annotation_name.endswith("final_good_nadir_v1.csv"):
            raise AssertionError("pixel-domain release loader fell back to raw v1 annotation")
        if annotation_name not in registry:
            raise AssertionError(f"{root.name} annotation not in registry: {annotation_name}")
        layout_results[root.name] = {
            "token": token,
            "payload_manifest": manifest_name,
            "root_digest": root_name,
            "annotation_name": annotation_name,
            "verified_file_count": len(verified),
        }

    config = load_release_config(release_dir / "gcp_benchmark_release_v1_2_2.json")
    verified = verify_release_files(release_dir / "gcp_benchmark_release_v1_2_2.json", config)
    registry = release_file_registry(verified)
    annotation_name = release_annotation_name_for_scene(config, "gcp_3000_20260602")
    missing_registry = dict(registry)
    missing_registry.pop(annotation_name, None)
    missing_registry.pop(str((release_dir / annotation_name).resolve()), None)
    try:
        require_release_registry_file(missing_registry, release_dir, annotation_name, "Frozen annotation file")
    except ValueError:
        missing_annotation_hard_fail = True
    else:
        raise AssertionError("missing v1.2.2 pixel-domain annotation did not hard fail")
    try:
        pixel_domain_release_layout({"schema": "ms_gcp_3dgs_benchmark_release_config_v1_2_999"})
    except ValueError:
        unknown_schema_hard_fail = True
    else:
        raise AssertionError("unknown pixel-domain schema did not hard fail")
    return {
        "layout_results": layout_results,
        "missing_annotation_hard_fail": missing_annotation_hard_fail,
        "unknown_schema_hard_fail": unknown_schema_hard_fail,
        "no_depth_tensor_read": True,
        "formal_metric_computation": False,
    }


def run_tests(real_release_dir: Path | None = None, v121_dir: Path = DEFAULT_RELEASE_V121, remote_manifest_path: Path = DEFAULT_REMOTE_MANIFEST) -> list[dict[str, Any]]:
    tests: list[tuple[str, Callable[[], dict[str, Any]]]] = [
        ("golden_observation_id_serialization", test_golden_serialization),
        ("golden_rgb_pixel_matrix_hash", test_golden_pixel_matrix_hash),
        ("projection_roundtrip_and_model_rejection", test_projection_roundtrip),
        ("release_integrity_rejections", test_integrity_manifest_rejections),
        ("evaluator_v12_positive_control", test_evaluator_v12_hard_gates),
        ("mapping_unique_key_rejections", test_mapping_unique_key_rejections),
        *[(f"hard_gate_{name}", fn) for name, fn in hard_gate_cases()],
    ]
    if real_release_dir is not None:
        tests.append(
            (
                "actual_v1_2_2_release_interface_smoke",
                lambda: test_actual_release_smoke(real_release_dir, v121_dir, remote_manifest_path),
            )
        )
        tests.append(
            (
                "actual_v1_2_2_formal_loader_layout_smoke",
                lambda: test_formal_loader_layouts(real_release_dir, DEFAULT_RELEASE_V12, v121_dir),
            )
        )
    rows = []
    for name, fn in tests:
        try:
            rows.append({"test": name, "status": "PASS", **fn()})
        except Exception as exc:  # noqa: BLE001
            rows.append({"test": name, "status": "FAIL", "error": repr(exc)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GS-GCP pixel-domain release protocol tests.")
    parser.add_argument("--real_release_dir", default="", help="Optional v1.2.2 release directory for the real 611-row release-interface smoke test.")
    parser.add_argument("--v121_dir", default=str(DEFAULT_RELEASE_V121))
    parser.add_argument("--remote_manifest", default=str(DEFAULT_REMOTE_MANIFEST))
    args = parser.parse_args()
    real_release_dir = Path(args.real_release_dir) if args.real_release_dir else None
    rows = run_tests(
        real_release_dir=real_release_dir,
        v121_dir=Path(args.v121_dir),
        remote_manifest_path=Path(args.remote_manifest),
    )
    payload = {
        "schema": "ms_gcp_release_v1_2_2_test_matrix_v1",
        "test_count": len(rows),
        "passed": sum(1 for r in rows if r["status"] == "PASS"),
        "failed": sum(1 for r in rows if r["status"] != "PASS"),
        "real_release_dir": str(real_release_dir) if real_release_dir else "",
        "results": rows,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if payload["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
