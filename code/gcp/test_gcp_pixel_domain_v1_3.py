from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_gaussian_gcp_geometry import PIXEL_DOMAIN_RELEASE_LAYOUTS
from gcp_pixel_domain_v1_2 import observation_id_from_fields
from gcp_pixel_domain_v1_3 import (
    RELEASE_V130_SCHEMA,
    observation_id_from_fields_v13,
    observation_id_payload_v13,
)


IMAGE_SHA = "8d8d79b8c5f7d7f07e2ffa1d5d75c81046e379c5d091f5eabe2d392e6ffeacbd"


def check(name: str, fn) -> dict[str, object]:
    try:
        fn()
        return {"name": name, "passed": True}
    except Exception as exc:  # noqa: BLE001
        return {"name": name, "passed": False, "error": f"{type(exc).__name__}: {exc}"}


def expect_raises(fn) -> None:
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_clicked_id_is_v12_compatible() -> None:
    args = (
        "gcp_3000_20260602",
        "G11",
        "DJI_20260602165038_0001_D.JPG",
        IMAGE_SHA,
        "3152.583",
        "1750.957",
    )
    assert observation_id_from_fields_v13(*args) == observation_id_from_fields(*args)
    assert observation_id_from_fields_v13(*args) == "17b0e32696d50b47741ec8c5e78c40e8cc1005294621785d44c9b3061781ed37"


def test_no_click_golden_id() -> None:
    actual = observation_id_from_fields_v13(
        "gcp_3000_20260602",
        "G11",
        "DJI_20260602165038_0001_D.JPG",
        IMAGE_SHA,
        "",
        "",
    )
    assert actual == "736990b8fb22cb546345ed45305188a838ea17a1b4b01104c8fa4f2925c9af9c"


def test_half_click_rejected() -> None:
    expect_raises(
        lambda: observation_id_payload_v13(
            "gcp_3000_20260602",
            "G11",
            "DJI_20260602165038_0001_D.JPG",
            IMAGE_SHA,
            "1.0",
            "",
        )
    )


def test_invalid_image_hash_rejected() -> None:
    expect_raises(
        lambda: observation_id_payload_v13(
            "gcp_3000_20260602",
            "G11",
            "DJI_20260602165038_0001_D.JPG",
            "not-a-sha",
            "",
            "",
        )
    )


def test_evaluator_layout_is_explicit() -> None:
    assert PIXEL_DOMAIN_RELEASE_LAYOUTS[RELEASE_V130_SCHEMA] == {
        "token": "v1_3_0",
        "annotation_suffix": "pixel_domain_v1_3_0.csv",
        "payload_manifest": "v1_3_0_release_file_manifest.json",
        "root_digest_record": "v1_3_0_release_root_digest.json",
    }


def main() -> None:
    tests = [
        ("clicked_id_is_v12_compatible", test_clicked_id_is_v12_compatible),
        ("no_click_golden_id", test_no_click_golden_id),
        ("half_click_rejected", test_half_click_rejected),
        ("invalid_image_hash_rejected", test_invalid_image_hash_rejected),
        ("evaluator_layout_is_explicit", test_evaluator_layout_is_explicit),
    ]
    rows = [check(name, fn) for name, fn in tests]
    print(json.dumps({"passed": all(row["passed"] for row in rows), "tests": rows}, indent=2))
    if not all(row["passed"] for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

