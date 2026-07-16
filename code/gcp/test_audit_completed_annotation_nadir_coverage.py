from __future__ import annotations

import math

from audit_completed_annotation_nadir_coverage import (
    classify_future_formal_candidate,
    visible_good,
)
from audit_multiview_annotation_expansion import project_simple_radial


def test_excludes_oblique_rich_point_without_nadir_overlap() -> None:
    status, reason = classify_future_formal_candidate(17, 1, 4)
    assert status == "exclude_from_future_v1_3_formal_primary_draft"
    assert "fewer_than_3_human_verified_near_nadir_observations" in reason


def test_exact_minimum_is_flagged_for_independent_review() -> None:
    status, reason = classify_future_formal_candidate(11, 3, 6)
    assert status == "provisionally_eligible_at_minimum_nadir_overlap_requires_independent_review"
    assert reason == "passes_draft_minimum_exactly"


def test_total_good_count_is_independent_gate() -> None:
    status, reason = classify_future_formal_candidate(3, 3, 2)
    assert status == "exclude_from_future_v1_3_formal_primary_draft"
    assert "fewer_than_4_human_verified_good_observations" in reason


def test_not_visible_never_counts_as_good() -> None:
    assert visible_good({"quality": "good", "visible": "1"})
    assert not visible_good({"quality": "not_visible", "visible": "0"})
    assert not visible_good({"quality": "ambiguous", "visible": "1"})


def test_candidate_projector_rejects_nonmonotonic_simple_radial_fold() -> None:
    u, v, z = project_simple_radial(
        xyz_world=[0.6212255966217382, 2.946798890661975, 1.0],
        qvec=[1.0, 0.0, 0.0, 0.0],
        tvec=[0.0, 0.0, 0.0],
        params=[3703.4967225010305, 2640.0, 1978.0, -0.11493612085570779],
    )
    assert z > 0.0
    assert math.isnan(u) and math.isnan(v)


def main() -> int:
    tests = [
        test_excludes_oblique_rich_point_without_nadir_overlap,
        test_exact_minimum_is_flagged_for_independent_review,
        test_total_good_count_is_independent_gate,
        test_not_visible_never_counts_as_good,
        test_candidate_projector_rejects_nonmonotonic_simple_radial_fold,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
