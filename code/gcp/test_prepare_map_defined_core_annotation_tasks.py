from __future__ import annotations

from prepare_map_defined_core_annotation_tasks import (
    MAP_DEFINED_POINTS,
    classify_coverage,
)


def test_map_defined_scene_counts() -> None:
    assert len(MAP_DEFINED_POINTS["gcp_5000_20260602"]) == 10
    assert len(MAP_DEFINED_POINTS["gcp_3000_20260602"]) == 9
    assert len(MAP_DEFINED_POINTS["gcp_10000_20260610"]) == 10
    assert len(MAP_DEFINED_POINTS["gcp_20000_20260602"]) == 10
    assert "G34" not in MAP_DEFINED_POINTS["gcp_20000_20260602"]
    assert "G39" not in MAP_DEFINED_POINTS["gcp_20000_20260602"]


def test_coverage_classification() -> None:
    assert classify_coverage(8, 0, 0) == "complete_usable_annotations"
    assert classify_coverage(2, 0, 0) == "partial_good_annotations"
    assert classify_coverage(0, 8, 0) == "review_required_no_good"
    assert classify_coverage(0, 0, 4) == "no_usable_annotation_all_not_visible"
    assert classify_coverage(0, 0, 0) == "unannotated"


def main() -> int:
    tests = [test_map_defined_scene_counts, test_coverage_classification]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
