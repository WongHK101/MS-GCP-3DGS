from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from compare_geometry_camera_only_packets import compare


def write_fixture(
    root: Path,
    values: np.ndarray,
    *,
    applied: bool,
    trace_token: str = "same",
) -> None:
    root.mkdir()
    np.savez_compressed(
        root / "view_metric_depth_packet.npz",
        accumulated_alpha=values,
        metric_depth_valid_mask=values > 0,
    )
    (root / "depth_export_manifest.json").write_text(
        json.dumps(
            {
                "rendered_view_count": 1,
                "geometry_camera_loader": {"applied": applied},
                "camera_state_trace": [
                    {
                        "image_name": "view.JPG",
                        "world_view_transform": {"bytes_sha256": trace_token},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_equal_tensor_bytes_pass() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        values = np.asarray([[0.0, 1.0]], dtype=np.float32)
        write_fixture(root / "baseline", values, applied=False)
        write_fixture(root / "candidate", values.copy(), applied=True)
        assert compare(root / "baseline", root / "candidate")["status"] == "PASS"


def test_one_bit_difference_fails() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        baseline = np.asarray([[1.0]], dtype=np.float32)
        candidate = baseline.copy()
        candidate.view(np.uint32)[0, 0] += 1
        write_fixture(root / "baseline", baseline, applied=False)
        write_fixture(root / "candidate", candidate, applied=True)
        report = compare(root / "baseline", root / "candidate")
        assert report["status"] == "FAIL"
        assert any("tensor differs" in item for item in report["errors"])


def test_camera_state_difference_fails() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        values = np.asarray([[1.0]], dtype=np.float32)
        write_fixture(root / "baseline", values, applied=False, trace_token="a")
        write_fixture(root / "candidate", values, applied=True, trace_token="b")
        report = compare(root / "baseline", root / "candidate")
        assert report["status"] == "FAIL"
        assert any("camera state" in item for item in report["errors"])


def main() -> int:
    tests = [
        test_equal_tensor_bytes_pass,
        test_one_bit_difference_fails,
        test_camera_state_difference_fails,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS {len(tests)}/{len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
