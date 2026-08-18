from __future__ import annotations

import argparse
import csv
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from export_gaussian_depth_maps import (
    collect_views,
    convert_raw_camera_z_units,
    derive_packet_from_raw_accumulators,
    parse_train_repo,
    read_allowlist,
    resolve_rasterizer_repo,
)


def write_list(path: Path, names: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["target_image_name", "formal_eligible"])
        writer.writeheader()
        for name in names:
            writer.writerow({"target_image_name": name, "formal_eligible": "true"})


def args_for(path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        image_list_csv=str(path),
        image_list_status_values="true",
        image_list_status_column="formal_eligible",
        image_name_column="target_image_name",
    )


def test_extensionless_runtime_names_resolve_to_release_names() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "views.csv"
        write_list(path, ["DJI_0001.JPG", "DJI_0002.JPG"])
        allowlist = read_allowlist(args_for(path))
        scene = SimpleNamespace(
            getTrainCameras=lambda: [SimpleNamespace(image_name="DJI_0001"), SimpleNamespace(image_name="DJI_0002")],
            getTestCameras=lambda: [],
        )
        views = collect_views(scene, "all", allowlist)
        assert [row[2] for row in views] == ["DJI_0001.JPG", "DJI_0002.JPG"]


def test_missing_requested_view_hard_fails() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "views.csv"
        write_list(path, ["DJI_0001.JPG", "DJI_0002.JPG"])
        allowlist = read_allowlist(args_for(path))
        scene = SimpleNamespace(
            getTrainCameras=lambda: [SimpleNamespace(image_name="DJI_0001")],
            getTestCameras=lambda: [],
        )
        try:
            collect_views(scene, "all", allowlist)
        except ValueError as exc:
            assert "missing=" in str(exc)
        else:
            raise AssertionError("missing requested view did not hard fail")


def test_train_repository_must_be_explicit() -> None:
    try:
        parse_train_repo([])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("missing --train_repo did not fail")


def test_explicit_train_repository_is_resolved() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        assert parse_train_repo(["--train_repo", tmp]) == Path(tmp).resolve()


def test_raw_renderer_accumulators_are_derived_on_cpu() -> None:
    raw = np.asarray([[[0.8]], [[12.0]], [[200.0]], [[0.08]]], dtype=np.float32)
    packet = derive_packet_from_raw_accumulators(
        raw,
        numerical_support_floor=1.0e-6,
        variance_clamp_tolerance=1.0e-6,
    )
    assert np.isclose(packet["alpha_normalized_expected_camera_z"][0, 0], 15.0)
    assert np.isclose(packet["alpha_normalized_expected_inverse_camera_z"][0, 0], 0.1)
    assert bool(packet["metric_depth_valid_mask"][0, 0])


def test_raw_renderer_accumulators_are_reversibly_scaled_to_protocol_units() -> None:
    normalized = np.asarray([[[0.7]], [[1.3]], [[3.1]], [[0.5]]], dtype=np.float32)
    converted = convert_raw_camera_z_units(
        normalized,
        camera_z_to_protocol_scale=10.0,
    )
    assert np.allclose(converted[:, 0, 0], [0.7, 13.0, 310.0, 0.05])
    packet = derive_packet_from_raw_accumulators(
        converted,
        numerical_support_floor=1.0e-6,
        variance_clamp_tolerance=1.0e-6,
    )
    assert np.isclose(packet["alpha_normalized_expected_camera_z"][0, 0], 13.0 / 0.7)
    assert np.isclose(packet["harmonic_camera_z"][0, 0], 14.0)
    try:
        convert_raw_camera_z_units(normalized, camera_z_to_protocol_scale=0.0)
    except ValueError as exc:
        assert "finite and positive" in str(exc)
    else:
        raise AssertionError("zero camera-z conversion scale was accepted")


def test_rasterizer_repository_infers_3dgs_or_2dgs_layout() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        surfel = root / "submodules" / "diff-surfel-rasterization"
        surfel.mkdir(parents=True)
        assert resolve_rasterizer_repo(root) == surfel.resolve()
        gaussian = root / "submodules" / "diff-gaussian-rasterization"
        gaussian.mkdir()
        try:
            resolve_rasterizer_repo(root)
        except RuntimeError as exc:
            assert "pass --rasterizer_repo explicitly" in str(exc)
        else:
            raise AssertionError("ambiguous rasterizer layout did not fail closed")
        assert resolve_rasterizer_repo(root, "submodules/diff-surfel-rasterization") == surfel.resolve()


def test_rasterizer_repository_cannot_escape_train_repo() -> None:
    with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside:
        try:
            resolve_rasterizer_repo(Path(tmp), outside)
        except ValueError as exc:
            assert "must be inside train_repo" in str(exc)
        else:
            raise AssertionError("out-of-tree rasterizer repository was accepted")


def main() -> int:
    tests = [
        test_extensionless_runtime_names_resolve_to_release_names,
        test_missing_requested_view_hard_fails,
        test_train_repository_must_be_explicit,
        test_explicit_train_repository_is_resolved,
        test_raw_renderer_accumulators_are_derived_on_cpu,
        test_raw_renderer_accumulators_are_reversibly_scaled_to_protocol_units,
        test_rasterizer_repository_infers_3dgs_or_2dgs_layout,
        test_rasterizer_repository_cannot_escape_train_repo,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS {len(tests)}/{len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
