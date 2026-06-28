from __future__ import annotations

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
    RELEASE_V12_ID,
    ROUNDTRIP_TOL_PX,
    TARGET_PIXEL_DOMAIN,
    TRANSFORM_VERSION,
    CameraRecord,
    ImageRecord,
    camera_record_hash,
    image_pose_record_hash,
    observation_id_from_fields,
    payload_manifest_entries,
    payload_root_digest,
    raw_to_target_projection,
    relative_posix,
    serialize_observation_id_payload,
    serialize_rgb_pixel_matrix,
    sha256_bytes,
    validate_release_v12_rows_for_evaluator,
    verify_payload_integrity,
    write_json_deterministic,
)


def _write_payload_manifests(root: Path) -> None:
    entries = payload_manifest_entries(
        root,
        exclude={"v1_2_release_file_manifest.json", "v1_2_release_root_digest.json"},
    )
    write_json_deterministic(
        root / "v1_2_release_file_manifest.json",
        {
            "schema": "ms_gcp_release_payload_manifest_v1",
            "release_id": RELEASE_V12_ID,
            "files": entries,
        },
    )
    manifest_sha = _sha(root / "v1_2_release_file_manifest.json")
    write_json_deterministic(
        root / "v1_2_release_root_digest.json",
        {
            "schema": "ms_gcp_release_root_digest_v1",
            "release_id": RELEASE_V12_ID,
            "payload_file_count": len(entries),
            "payload_manifest_path": "v1_2_release_file_manifest.json",
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
    mapping_hash = "d" * 64
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
        root / "raw_image_orientation_manifest_v1_2.json",
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
        root / "source_target_mapping_manifest_v1_2.json",
        [
            {
                "scene": scene,
                "source_image_name": image_name,
                "target_image_name": image_name,
                "mapping_type": row["mapping_type"],
                "transform_version": TRANSFORM_VERSION,
                "pose_equivalent": True,
                "source_target_mapping_record_sha256": mapping_hash,
            }
        ],
    )
    write_json_deterministic(root / "projection_manifest_v1_2.json", {"schema": "fixture"})
    write_json_deterministic(root / "camera_provenance_manifest_v1_2.json", {"schema": "fixture"})
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
        ok = verify_payload_integrity(root, root / "v1_2_release_file_manifest.json", root / "v1_2_release_root_digest.json")
        if not ok["passed"]:
            raise AssertionError(ok)
        (root / "payload.txt").write_text("changed\n", encoding="utf-8")
        modified = verify_payload_integrity(root, root / "v1_2_release_file_manifest.json", root / "v1_2_release_root_digest.json")
        (root / "payload.txt").write_text("ok\n", encoding="utf-8")
        (root / "unregistered.txt").write_text("x\n", encoding="utf-8")
        unregistered = verify_payload_integrity(root, root / "v1_2_release_file_manifest.json", root / "v1_2_release_root_digest.json")
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


def run_tests() -> list[dict[str, Any]]:
    tests: list[tuple[str, Callable[[], dict[str, Any]]]] = [
        ("golden_observation_id_serialization", test_golden_serialization),
        ("golden_rgb_pixel_matrix_hash", test_golden_pixel_matrix_hash),
        ("projection_roundtrip_and_model_rejection", test_projection_roundtrip),
        ("release_integrity_rejections", test_integrity_manifest_rejections),
        ("evaluator_v12_hard_gates", test_evaluator_v12_hard_gates),
    ]
    rows = []
    for name, fn in tests:
        try:
            rows.append({"test": name, "status": "PASS", **fn()})
        except Exception as exc:  # noqa: BLE001
            rows.append({"test": name, "status": "FAIL", "error": repr(exc)})
    return rows


def main() -> None:
    rows = run_tests()
    payload = {
        "schema": "ms_gcp_release_v1_2_test_matrix_v1",
        "test_count": len(rows),
        "passed": sum(1 for r in rows if r["status"] == "PASS"),
        "failed": sum(1 for r in rows if r["status"] != "PASS"),
        "results": rows,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if payload["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
