from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from manual_gcp_annotator import Annotator
from prepare_tgs_gcp_rgb_outsourcing import (
    Camera,
    Point,
    SCENES,
    compact_hash,
    distance_to_image_bounds_px,
    order_all_candidates,
    project,
)
from tgs_gcp_outsourcing_runtime import validate_result


def camera(name: str, order: int, strip: str, e: float = 0.0, n: float = 0.0) -> Camera:
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
        gimbal_pitch_deg=-90.0,
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

    def test_all_candidates_are_retained_and_known_anchor_is_recalled(self) -> None:
        pool = []
        for index in range(12):
            cam = camera(f"{index + 1:04d}.jpg", index + 1, f"strip_{index % 3:03d}")
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
        included = order_all_candidates(pool, "0007.jpg")
        self.assertEqual(len(included), len(pool))
        self.assertEqual({row["camera"].image_name for row in included}, {row["camera"].image_name for row in pool})

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
