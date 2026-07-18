#!/usr/bin/env python3
"""Tests for deterministic Original 3DGS camera compatibility evidence."""

from __future__ import annotations

from original_3dgs_camera_compatibility import compare_reports, freeze_samples


def assignment(scene: str, index: int, role: str, stratum: str, width: int = 100) -> dict:
    return {
        "scene": scene,
        "image_id": index + 1,
        "camera_id": 1,
        "image_name": f"DJI_20260719000000_{index + 1:04d}_D.JPG",
        "capture_timestamp": "20260719000000",
        "capture_sequence": index + 1,
        "split_role": role,
        "stratum_id": stratum,
        "decoded_width": width,
        "decoded_height": 80,
        "image_sha256": f"{index + 1:064x}",
    }


def test_freeze_samples_is_deterministic_and_preserves_all_3k() -> None:
    three_k = [assignment("gcp_3000_20260602", index, "train" if index < 8 else "test", "strip_0") for index in range(10)]
    five_k = [assignment("gcp_5000_20260602", index, "test" if index % 8 == 0 else "train", f"strip_{index // 8}") for index in range(24)]
    split = {"manifest_sha256": "abc", "scenes": [
        {"scene": "gcp_5000_20260602", "assignments": five_k},
        {"scene": "gcp_3000_20260602", "assignments": three_k},
    ]}
    first = freeze_samples(split)
    second = freeze_samples(split)
    assert first == second
    by_scene = {row["scene"]: row for row in first["scenes"]}
    assert by_scene["gcp_3000_20260602"]["selected_image_count"] == 10
    assert by_scene["gcp_5000_20260602"]["selected_image_count"] >= 16


def test_nonmodal_dimensions_are_always_selected() -> None:
    rows = [assignment("gcp_5000_20260602", index, "test" if index == 0 else "train", "strip_0") for index in range(20)]
    rows[-1]["decoded_width"] = 101
    result = freeze_samples({"manifest_sha256": "abc", "scenes": [{"scene": "gcp_5000_20260602", "assignments": rows}]})
    names = {row["image_name"] for row in result["scenes"][0]["images"]}
    assert rows[-1]["image_name"] in names


def test_report_comparison_rejects_tensor_or_order_changes() -> None:
    row = {
        "image_id": 1,
        "image_name": "a.JPG",
        "camera_id": 1,
        "loaded_width": 10,
        "loaded_height": 8,
        "channels": 3,
        "dtype": "torch.float32",
        "tensor_sha256": "a",
        "R": ["1"],
        "T": ["0"],
        "FoVx": "1",
        "FoVy": "1",
        "world_view_transform_sha256": "b",
        "projection_matrix_sha256": "c",
        "full_proj_transform_sha256": "d",
        "camera_center_sha256": "e",
    }
    reference = {"camera_records": [row], "max_normalized_ray_coordinate_error": 0}
    assert compare_reports(reference, reference)["status"] == "PASS"
    changed = {"camera_records": [{**row, "tensor_sha256": "x"}], "max_normalized_ray_coordinate_error": 0}
    assert compare_reports(reference, changed)["status"] == "BLOCKER"


TESTS = [
    test_freeze_samples_is_deterministic_and_preserves_all_3k,
    test_nonmodal_dimensions_are_always_selected,
    test_report_comparison_rejects_tensor_or_order_changes,
]


def main() -> int:
    for test in TESTS:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS {len(TESTS)}/{len(TESTS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
