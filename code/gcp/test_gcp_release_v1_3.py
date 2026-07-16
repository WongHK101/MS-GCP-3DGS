from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gcp_pixel_domain_v1_2 import SCENES, load_manifest_model, read_csv, verify_payload_integrity
from gcp_pixel_domain_v1_3 import (
    PROJECTION_STATUS_DIAGNOSTIC_OOB,
    RELEASE_V130_ID,
    RELEASE_V130_SCHEMA,
    validate_release_v13_rows_for_evaluator,
)
from generate_gcp_release_v1_3 import EXPECTED_COUNTS
from evaluate_gaussian_gcp_geometry import (
    load_release_config,
    release_annotation_name_for_scene,
    release_file_registry,
    verify_release_files,
)


def check(name: str, fn) -> dict[str, object]:
    try:
        detail = fn()
        return {"test": name, "status": "PASS", "detail": detail}
    except Exception as exc:  # noqa: BLE001
        return {"test": name, "status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}


def expect_rejected(fn) -> str:
    try:
        fn()
    except ValueError as exc:
        return type(exc).__name__
    raise AssertionError("expected ValueError")


def load_release_camera_provenance(release: Path) -> dict:
    config = json.loads((release / "gcp_benchmark_release_v1_3_0.json").read_text(encoding="utf-8"))
    relative = Path(str(config["camera_provenance_manifest"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"invalid release camera provenance path: {relative}")
    return json.loads((release / relative).read_text(encoding="utf-8"))


def model_for_scene(remote: dict, scene: str):
    cameras, images_by_name, _ = load_manifest_model(remote["scenes"][scene], "target_model")
    images = {
        image.image_id: SimpleNamespace(
            name=image.image_name,
            camera_id=image.camera_id,
            qvec=image.qvec,
            tvec=image.tvec,
        )
        for image in images_by_name.values()
    }
    return cameras, images


def test_integrity(release: Path) -> dict[str, object]:
    result = verify_payload_integrity(
        release,
        release / "v1_3_0_release_file_manifest.json",
        release / "v1_3_0_release_root_digest.json",
    )
    assert result["passed"], result
    config = json.loads((release / "gcp_benchmark_release_v1_3_0.json").read_text(encoding="utf-8"))
    assert config["schema"] == RELEASE_V130_SCHEMA
    assert config["release_id"] == RELEASE_V130_ID
    assert config["frozen_counts"]["row_count"] == EXPECTED_COUNTS["row_count"]
    return result


def test_real_release_interface(release: Path) -> dict[str, object]:
    remote = load_release_camera_provenance(release)
    total_all = 0
    total_formal = 0
    scene_all = {}
    scene_formal = {}
    observation_ids = set()
    oob = []
    no_click = 0
    formal_control = 0
    formal_checkpoint = 0
    for scene in SCENES:
        rows = read_csv(release / f"{scene}_gcp_annotations_pixel_domain_v1_3_0.csv")
        cameras, images = model_for_scene(remote, scene)
        validated_all = validate_release_v13_rows_for_evaluator(
            release_base=release,
            scene=scene,
            rows=rows,
            colmap_cameras=cameras,
            colmap_images=images,
            return_all_rows=True,
        )
        formal = validate_release_v13_rows_for_evaluator(
            release_base=release,
            scene=scene,
            rows=rows,
            colmap_cameras=cameras,
            colmap_images=images,
        )
        assert len(validated_all) == len(rows)
        assert all(row.get("u_px") and row.get("v_px") for row in formal)
        scene_all[scene] = len(validated_all)
        scene_formal[scene] = len(formal)
        total_all += len(validated_all)
        total_formal += len(formal)
        for row in validated_all:
            assert row["observation_id"] not in observation_ids
            observation_ids.add(row["observation_id"])
            if row["projection_status"] == PROJECTION_STATUS_DIAGNOSTIC_OOB:
                assert row["formal_eligible"] == "false"
                oob.append((scene, row["point_name"], row["raw_image_name"]))
            if not row["raw_manual_x"]:
                no_click += 1
        formal_control += sum(row["formal_role"] == "control" for row in formal)
        formal_checkpoint += sum(row["formal_role"] == "checkpoint" for row in formal)
    assert total_all == EXPECTED_COUNTS["row_count"]
    assert total_formal == EXPECTED_COUNTS["formal_eligible_count"]
    assert len(observation_ids) == EXPECTED_COUNTS["row_count"]
    assert no_click == EXPECTED_COUNTS["no_coordinate_row_count"]
    assert len(oob) == EXPECTED_COUNTS["diagnostic_projection_out_of_bounds_count"]
    return {
        "validated_all_rows": total_all,
        "formal_rows": total_formal,
        "scene_all_rows": scene_all,
        "scene_formal_rows": scene_formal,
        "formal_control_observations": formal_control,
        "formal_checkpoint_observations": formal_checkpoint,
        "diagnostic_out_of_bounds_rows": oob,
        "no_click_rows": no_click,
        "depth_tensor_values_read": False,
        "formal_metric_computed": False,
    }


def test_negative_row_gates(release: Path) -> dict[str, object]:
    remote = load_release_camera_provenance(release)
    scene = "gcp_100000_20260610"
    cameras, images = model_for_scene(remote, scene)
    source_rows = read_csv(release / f"{scene}_gcp_annotations_pixel_domain_v1_3_0.csv")
    formal_index = next(i for i, row in enumerate(source_rows) if row["formal_eligible"] == "true")
    no_click_index = next(i for i, row in enumerate(source_rows) if not row["raw_manual_x"])

    target_tamper = copy.deepcopy(source_rows)
    target_tamper[formal_index]["target_x"] = str(float(target_tamper[formal_index]["target_x"]) + 1.0)
    target_rejected = expect_rejected(
        lambda: validate_release_v13_rows_for_evaluator(
            release_base=release,
            scene=scene,
            rows=target_tamper,
            colmap_cameras=cameras,
            colmap_images=images,
            return_all_rows=True,
        )
    )

    eligibility_tamper = copy.deepcopy(source_rows)
    eligibility_tamper[formal_index]["formal_eligible"] = "false"
    eligibility_rejected = expect_rejected(
        lambda: validate_release_v13_rows_for_evaluator(
            release_base=release,
            scene=scene,
            rows=eligibility_tamper,
            colmap_cameras=cameras,
            colmap_images=images,
            return_all_rows=True,
        )
    )

    no_click_tamper = copy.deepcopy(source_rows)
    no_click_tamper[no_click_index]["target_x"] = "1"
    no_click_rejected = expect_rejected(
        lambda: validate_release_v13_rows_for_evaluator(
            release_base=release,
            scene=scene,
            rows=no_click_tamper,
            colmap_cameras=cameras,
            colmap_images=images,
            return_all_rows=True,
        )
    )
    return {
        "cached_target_tamper": target_rejected,
        "formal_eligibility_tamper": eligibility_rejected,
        "no_click_projection_tamper": no_click_rejected,
    }


def test_formal_loader_layout(release: Path) -> dict[str, object]:
    config_path = release / "gcp_benchmark_release_v1_3_0.json"
    config = load_release_config(config_path)
    verified = verify_release_files(config_path, config)
    registry = release_file_registry(verified)
    names = {}
    for scene in SCENES:
        name = release_annotation_name_for_scene(config, scene)
        assert name == f"{scene}_gcp_annotations_pixel_domain_v1_3_0.csv"
        assert name in registry
        names[scene] = name
    return {
        "verified_file_count": len(verified),
        "annotation_names": names,
        "depth_tensor_values_read": False,
        "formal_metric_computed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_release_dir", required=True)
    args = parser.parse_args()
    release = Path(args.real_release_dir)
    tests = [
        ("v1_3_payload_integrity", lambda: test_integrity(release)),
        ("v1_3_real_release_interface_1383_rows", lambda: test_real_release_interface(release)),
        ("v1_3_negative_row_gates", lambda: test_negative_row_gates(release)),
        ("v1_3_formal_loader_layout_no_depth", lambda: test_formal_loader_layout(release)),
    ]
    results = [check(name, fn) for name, fn in tests]
    payload = {
        "schema": "ms_gcp_release_v1_3_0_test_matrix_v1",
        "test_count": len(results),
        "passed": sum(row["status"] == "PASS" for row in results),
        "failed": sum(row["status"] != "PASS" for row in results),
        "results": results,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if payload["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
