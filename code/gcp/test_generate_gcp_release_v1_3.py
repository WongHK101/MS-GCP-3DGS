from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gcp_pixel_domain_v1_2 import SCENES, read_csv
from generate_gcp_release_v1_3 import (
    EXPECTED_COUNTS,
    canonical_quality,
    load_input_manifest,
    load_split_rows,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_MANIFEST = REPO_ROOT / "configs" / "gcp_v13_release_inputs_v1.json"


def check(name: str, fn) -> dict[str, object]:
    try:
        detail = fn()
        return {"name": name, "passed": True, "detail": detail}
    except Exception as exc:  # noqa: BLE001
        return {"name": name, "passed": False, "error": f"{type(exc).__name__}: {exc}"}


def test_frozen_source_counts() -> dict[str, object]:
    inputs = load_input_manifest(INPUT_MANIFEST)
    split_rows, roles = load_split_rows(Path(inputs["geometry_split_candidate"]))
    counts: Counter[str] = Counter()
    keys: set[tuple[str, str, str]] = set()
    points: set[str] = set()
    missing_images = []
    for scene in SCENES:
        rows = read_csv(Path(inputs["working_annotations"][scene]))
        for row in rows:
            if row["scene"] != scene:
                raise AssertionError(f"scene mismatch: {scene} {row['scene']}")
            key = (scene, row["point_name"], Path(row["image_name"]).name)
            if key in keys:
                raise AssertionError(f"duplicate observation key: {key}")
            keys.add(key)
            points.add(row["point_name"])
            visible, quality = canonical_quality(row)
            x = str(row.get("manual_x", "")).strip()
            y = str(row.get("manual_y", "")).strip()
            if bool(x) != bool(y):
                raise AssertionError(f"half-populated coordinate: {key}")
            has_click = bool(x)
            annotation_good = visible and quality == "good" and has_click
            counts["row_count"] += 1
            counts["annotation_good_count"] += annotation_good
            counts["formal_eligible_count"] += annotation_good and (scene, row["point_name"]) in roles
            counts["coordinate_row_count"] += has_click
            counts["no_coordinate_row_count"] += not has_click
            image = Path(r"E:\datasets\M3M-GCP\scenes") / scene / Path(row["image_name"]).name
            if not image.is_file():
                missing_images.append(str(image))
    for key in ["row_count", "annotation_good_count", "formal_eligible_count", "coordinate_row_count", "no_coordinate_row_count"]:
        assert counts[key] == EXPECTED_COUNTS[key], (key, counts[key], EXPECTED_COUNTS[key])
    assert len(points) == EXPECTED_COUNTS["all_annotation_unique_point_count"]
    assert len(split_rows) == EXPECTED_COUNTS["formal_split_scene_point_count"]
    assert len({point for _, point in roles}) == EXPECTED_COUNTS["formal_split_unique_point_count"]
    assert not missing_images, missing_images[:5]
    return {"counts": dict(counts), "unique_points": len(points), "split_rows": len(split_rows)}


def test_source_directories_are_not_output_roots() -> dict[str, object]:
    inputs = load_input_manifest(INPUT_MANIFEST)
    source_dirs = {
        (Path(r"E:\datasets\M3M-GCP\scenes") / scene).resolve()
        for scene in SCENES
    }
    planned_release = Path(r"E:\datasets\M3M-GCP\scenes\gcp_manual_annotations_v1_3_0").resolve()
    assert planned_release not in source_dirs
    for source in source_dirs:
        assert planned_release.parent == source.parent
        assert planned_release != source
    for path in inputs["working_annotations"].values():
        assert Path(path).is_file()
    return {"raw_scene_directory_count": len(source_dirs), "planned_release": str(planned_release)}


def test_input_manifest_is_complete() -> dict[str, object]:
    inputs = load_input_manifest(INPUT_MANIFEST)
    assert set(inputs["working_annotations"]) == set(SCENES)
    for field in [
        "geometry_split_candidate",
        "remote_camera_manifest",
        "release_v1_2_2",
        "rtk_authoritative_dir",
        "rtk_quality_summary",
        "review_only_coordinate_table",
    ]:
        assert Path(inputs[field]).exists(), field
    return {"scene_count": len(inputs["working_annotations"])}


def main() -> None:
    tests = [
        ("frozen_source_counts", test_frozen_source_counts),
        ("source_directories_are_not_output_roots", test_source_directories_are_not_output_roots),
        ("input_manifest_is_complete", test_input_manifest_is_complete),
    ]
    rows = [check(name, fn) for name, fn in tests]
    payload = {"passed": all(row["passed"] for row in rows), "tests": rows}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

