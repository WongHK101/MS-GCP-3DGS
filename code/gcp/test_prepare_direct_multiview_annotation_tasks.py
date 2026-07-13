from __future__ import annotations

from manual_gcp_annotator import candidate_crop_size, candidate_search_radius
from prepare_direct_multiview_annotation_tasks import select_primary_candidates


def candidate(image: str, view_type: str, azimuth_bin: int, score: float = 0.5) -> dict:
    return {
        "scene": "scene",
        "point_name": "P1",
        "image_name": image,
        "candidate_source": "coarse_exif_gimbal_all_orientations",
        "view_type": view_type,
        "azimuth_bin_45deg": azimuth_bin,
        "center_score": score,
        "edge_margin_px": 100.0,
    }


def test_primary_mix_and_diversity() -> None:
    rows = [candidate(f"n{i}.jpg", "nadir", i % 8, 1.0 - i / 100) for i in range(12)]
    rows += [candidate(f"o{i}.jpg", "oblique", i % 8, 1.0 - i / 100) for i in range(12)]
    selected = select_primary_candidates(rows, per_point=16)
    assert len(selected) == 16
    assert sum(row["view_type"] == "nadir" for row in selected) == 8
    assert sum(row["view_type"] == "oblique" for row in selected) == 8
    assert len({row["azimuth_bin_45deg"] for row in selected}) == 8
    assert [row["rank_for_gcp"] for row in selected] == list(range(1, 17))


def test_primary_fill_when_only_one_view_type_exists() -> None:
    rows = [candidate(f"n{i}.jpg", "nadir", i % 4) for i in range(7)]
    selected = select_primary_candidates(rows, per_point=16)
    assert len(selected) == 7
    assert all(row["view_type"] == "nadir" for row in selected)


def test_dynamic_search_crop() -> None:
    assert candidate_search_radius({"projection_uncertainty_px": "800"}) == 800.0
    assert candidate_crop_size({"projection_uncertainty_px": "800"}, 720) == 1600
    assert candidate_crop_size({"projection_uncertainty_px": "2000"}, 720) == 2400
    assert candidate_crop_size({}, 720) == 720


def main() -> int:
    tests = [
        test_primary_mix_and_diversity,
        test_primary_fill_when_only_one_view_type_exists,
        test_dynamic_search_crop,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
