from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from manual_gcp_annotator import Annotator
from prepare_tgs_gcp_rgb_outsourcing import (
    Camera,
    NADIR_TARGET_PER_POINT,
    OBLIQUE_TARGET_PER_POINT,
    Point,
    SCENES,
    compact_hash,
    distance_to_image_bounds_px,
    project,
    select_diverse_class,
    select_stratified_candidates,
)
from tgs_gcp_outsourcing_runtime import validate_result


def camera(
    name: str,
    order: int,
    strip: str,
    e: float = 0.0,
    n: float = 0.0,
    pitch: float = -90.0,
) -> Camera:
    return Camera(
        scene="GCP-3K",
        dataset_scene_dir="GCP-3K",
        image_path=Path(name),
        image_name=name,
        capture_order=order,
        width=4032,
        height=3024,
        orientation=1,
        e=e,
        n=n,
        h_ellipsoid=30.0,
        flight_yaw_deg=0.0,
        gimbal_yaw_deg=0.0,
        gimbal_pitch_deg=pitch,
        gimbal_roll_deg=180.0,
        focal_length_mm=6.72,
        focal_35mm_mm=24.0,
        focal_px=2688.0,
        strip_id=strip,
    )


class TgsOutsourcingTests(unittest.TestCase):
    def test_roll_180_projection_orientation(self) -> None:
        cam = camera("0001.jpg", 1, "strip_001")
        point = Point(
            "P",
            e=1.0,
            n=2.0,
            h_ellipsoid=0.0,
            h_ellipsoid_source="test",
            role="control",
        )
        result = project(cam, point)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertLess(float(result["u"]), cam.width / 2.0)
        self.assertGreater(float(result["v"]), cam.height / 2.0)

    def test_stratified_selection_freezes_nadir_oblique_quotas(self) -> None:
        pool = []
        for index in range(40):
            pitch = -90.0 if index < 20 else -45.0
            cam = camera(f"{index + 1:04d}.jpg", index + 1, f"strip_{index % 10:03d}", pitch=pitch)
            pool.append(
                {
                    "camera": cam,
                    "inside_image": True,
                    "center_score": 1.0 - index / 100.0,
                    "edge_margin_px": 500.0,
                    "ground_distance_m": float(index),
                    "azimuth_bin_45deg": index % 4,
                }
            )
        selected = select_stratified_candidates(pool, "0007.jpg")
        self.assertEqual(len(selected), NADIR_TARGET_PER_POINT + OBLIQUE_TARGET_PER_POINT)
        self.assertEqual(sum(row["camera"].gimbal_pitch_deg == -90.0 for row in selected), NADIR_TARGET_PER_POINT)
        self.assertEqual(sum(row["camera"].gimbal_pitch_deg != -90.0 for row in selected), OBLIQUE_TARGET_PER_POINT)
        self.assertEqual(len({row["azimuth_bin_45deg"] for row in selected[:NADIR_TARGET_PER_POINT]}), 4)
        self.assertEqual(len({row["azimuth_bin_45deg"] for row in selected[NADIR_TARGET_PER_POINT:]}), 4)
        self.assertIn("0007.jpg", {row["camera"].image_name for row in selected})

    def test_center_priority_applies_within_same_strip_azimuth_group(self) -> None:
        pool = []
        for index, score in enumerate([0.2, 0.9, 0.5], 1):
            pool.append(
                {
                    "camera": camera(f"{index:04d}.jpg", index, "strip_001"),
                    "inside_image": True,
                    "center_score": score,
                    "edge_margin_px": 500.0,
                    "ground_distance_m": 10.0,
                    "azimuth_bin_45deg": 0,
                }
            )
        selected = select_diverse_class(pool, 1, None)
        self.assertEqual(selected[0]["camera"].image_name, "0002.jpg")

    def test_insufficient_view_class_is_rejected(self) -> None:
        pool = []
        for index in range(19):
            pitch = -90.0 if index < 12 else -45.0
            pool.append(
                {
                    "camera": camera(f"{index + 1:04d}.jpg", index + 1, f"strip_{index:03d}", pitch=pitch),
                    "inside_image": True,
                    "center_score": 1.0,
                    "edge_margin_px": 500.0,
                    "ground_distance_m": 10.0,
                    "azimuth_bin_45deg": index % 8,
                }
            )
        with self.assertRaisesRegex(RuntimeError, "class quota"):
            select_stratified_candidates(pool, None)

    def test_uncertainty_disk_intersection_is_not_rectangular_margin(self) -> None:
        self.assertGreater(distance_to_image_bounds_px(-700.0, -700.0, 4032, 3024), 900.0)
        self.assertLess(distance_to_image_bounds_px(-600.0, -600.0, 4032, 3024), 900.0)
        self.assertEqual(distance_to_image_bounds_px(100.0, 200.0, 4032, 3024), 0.0)

    def test_public_scene_names_match_dataset_directories(self) -> None:
        self.assertEqual(
            set(SCENES.values()),
            {"GCP-3K", "GCP-5K", "GCP-10K", "GCP-20K", "GCP-50K", "GCP-100K"},
        )

    def test_task_hash_is_deterministic(self) -> None:
        value = ["schema", "scene", "point", "image", "a" * 64]
        self.assertEqual(compact_hash(value), compact_hash(value))
        self.assertNotEqual(compact_hash(value), compact_hash(value[:-1] + ["b" * 64]))

    def test_portable_image_root_resolves_nested_relative_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "GCP-3K" / "rgb" / "0001.jpg"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"jpeg-placeholder")
            annotator = Annotator.__new__(Annotator)
            annotator.image_root = root
            resolved = annotator.resolve_image_path(
                {"image_path": "GCP-3K/rgb/0001.jpg", "image_name": "0001.jpg", "scene": "GCP-3K"}
            )
            self.assertEqual(resolved, image)

    def test_minimal_return_requires_click_for_good(self) -> None:
        candidate = {
            "scene": "GCP-3K",
            "point_name": "G11",
            "image_name": "0001.jpg",
            "task_id": "task",
            "source_image_sha256": "a" * 64,
            "source_image_width": "4032",
            "source_image_height": "3024",
        }
        result = {
            **candidate,
            "schema": "gs_gcp_tgs_rgb_manual_image_observation_v1",
            "visible": "1",
            "quality": "good",
            "manual_x": "",
            "manual_y": "",
        }
        with self.assertRaisesRegex(RuntimeError, "requires a manual coordinate"):
            validate_result([candidate], [result])

    def test_not_visible_minimal_return(self) -> None:
        candidate = {
            "scene": "GCP-3K",
            "point_name": "G11",
            "image_name": "0001.jpg",
            "task_id": "task",
            "source_image_sha256": "a" * 64,
            "source_image_width": "4032",
            "source_image_height": "3024",
        }
        result = {
            **candidate,
            "schema": "gs_gcp_tgs_rgb_manual_image_observation_v1",
            "visible": "0",
            "quality": "not_visible",
            "manual_x": "",
            "manual_y": "",
        }
        minimal = validate_result([candidate], [result])
        self.assertEqual(len(minimal), 1)
        self.assertEqual(minimal[0]["quality"], "not_visible")


if __name__ == "__main__":
    unittest.main()
