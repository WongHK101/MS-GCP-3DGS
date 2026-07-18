#!/usr/bin/env python3

from pathlib import Path

from validate_stage0_5_file_access import validate_access


def test_train_only_passes() -> None:
    trace = 'openat(AT_FDCWD, "/run/train/images/a.JPG", O_RDONLY) = 3\n'
    assert validate_access(trace, Path("/run/train"), [Path("/run/test"), Path("/run/full")])["status"] == "PASS"


def test_test_rgb_rejected() -> None:
    trace = 'openat(AT_FDCWD, "/run/test/images/a.JPG", O_RDONLY) = 3\n'
    assert validate_access(trace, Path("/run/train"), [Path("/run/test")])["status"] == "BLOCKER"


def test_full_tracks_rejected() -> None:
    trace = 'openat(AT_FDCWD, "/run/train/sparse/0/points3D.bin", O_RDONLY) = 3\n'
    assert validate_access(trace, Path("/run/train"), [])["status"] == "BLOCKER"


TESTS = [test_train_only_passes, test_test_rgb_rejected, test_full_tracks_rejected]


def main() -> int:
    for test in TESTS:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS {len(TESTS)}/{len(TESTS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
