#!/usr/bin/env python3

from __future__ import annotations

from prepare_v13_uniform_fixed_candidate_supplement import (
    robust_discovery_frame,
    select_uniform_supplement,
)


def candidate(name: str, view_type: str, bin_id: int) -> dict:
    return {
        "image_name": name,
        "view_type": view_type,
        "azimuth_bin_45deg": bin_id,
        "already_attempted": False,
        "edge_margin_px": 1000.0,
        "center_score": 0.9,
    }


def test_only_deficient_view_class_is_selected() -> None:
    summary = {
        "good_view_count": 8,
        "good_nadir_count": 8,
        "good_oblique_count": 0,
        "good_azimuth_bin_count": 2,
        "good_azimuth_bins": [0, 4],
    }
    pool = [candidate(f"o{i}", "oblique", i) for i in range(8)] + [candidate("n", "nadir", 2)]
    selected, plan = select_uniform_supplement(pool, summary)
    assert len(selected) == 6
    assert all(row["view_type"] == "oblique" for row in selected)
    assert plan["selected_oblique_count"] == 6


def test_complete_point_gets_no_candidate() -> None:
    summary = {
        "good_view_count": 10,
        "good_nadir_count": 5,
        "good_oblique_count": 5,
        "good_azimuth_bin_count": 5,
        "good_azimuth_bins": [0, 1, 2, 3, 4],
    }
    selected, plan = select_uniform_supplement([candidate("x", "nadir", 6)], summary)
    assert selected == []
    assert plan["selected_candidate_count"] == 0


def main() -> int:
    tests = [test_only_deficient_view_class_is_selected, test_complete_point_gets_no_candidate]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)}/{len(tests)} PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
