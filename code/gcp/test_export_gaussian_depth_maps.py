from __future__ import annotations

import argparse
import csv
import tempfile
from pathlib import Path
from types import SimpleNamespace

from export_gaussian_depth_maps import collect_views, parse_train_repo, read_allowlist


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


def main() -> int:
    tests = [
        test_extensionless_runtime_names_resolve_to_release_names,
        test_missing_requested_view_hard_fails,
        test_train_repository_must_be_explicit,
        test_explicit_train_repository_is_resolved,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS {len(tests)}/{len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
