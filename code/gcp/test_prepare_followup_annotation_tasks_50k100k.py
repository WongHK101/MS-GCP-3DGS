from __future__ import annotations

from prepare_followup_annotation_tasks_50k100k import (
    FIFTY_K_TASK_POINTS,
    HUNDRED_K_NEW_POINTS,
    HUNDRED_K_SUPPLEMENT_POINTS,
    qvec_to_rotation,
    rank_per_point,
    visible_good,
)


def test_visible_good() -> None:
    assert visible_good({"quality": "good", "visible": "1"})
    assert not visible_good({"quality": "ambiguous", "visible": "1"})
    assert not visible_good({"quality": "good", "visible": "0"})


def test_identity_quaternion() -> None:
    rotation = qvec_to_rotation([1.0, 0.0, 0.0, 0.0])
    assert abs(float(rotation[0, 0]) - 1.0) < 1e-15
    assert abs(float(rotation[1, 1]) - 1.0) < 1e-15
    assert abs(float(rotation[2, 2]) - 1.0) < 1e-15


def test_rank_order_and_uniqueness() -> None:
    points = [*FIFTY_K_TASK_POINTS, *HUNDRED_K_NEW_POINTS, *HUNDRED_K_SUPPLEMENT_POINTS]
    rows = []
    for point in reversed(points):
        rows.extend(
            [
                {"point_name": point, "image_name": f"{point}_a.jpg"},
                {"point_name": point, "image_name": f"{point}_b.jpg"},
            ]
        )
    ranked = rank_per_point(rows)
    assert len(ranked) == len(points) * 2
    for point in points:
        ranks = [row["rank_for_gcp"] for row in ranked if row["point_name"] == point]
        assert ranks == [1, 2]


def main() -> int:
    tests = [test_visible_good, test_identity_quaternion, test_rank_order_and_uniqueness]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
