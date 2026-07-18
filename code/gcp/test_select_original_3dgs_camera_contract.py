#!/usr/bin/env python3

from select_original_3dgs_camera_contract import decide


def test_a_selected_without_b():
    result = decide("PASS", "PASS", None, None)
    assert result["status"] == "SELECTED_A"


def test_a_host_failure_blocks_b():
    result = decide("HOST_RAM_BLOCKED", None, "PASS", "PASS")
    assert result["status"] == "BLOCKER"


def test_a_gpu_failure_allows_b():
    result = decide("GPU_MEMORY_BLOCKED", None, None, None)
    assert result["status"] == "B_ELIGIBLE"


def test_b_requires_both_large_scenes():
    assert decide("GPU_MEMORY_BLOCKED", None, "PASS", None)["status"] == "NEEDS_B100"
    assert decide("GPU_MEMORY_BLOCKED", None, "PASS", "PASS")["status"] == "SELECTED_B"


def test_b_failure_is_blocker():
    assert decide("GPU_MEMORY_BLOCKED", None, "HOST_RAM_BLOCKED", None)["status"] == "BLOCKER"


TESTS = [test_a_selected_without_b, test_a_host_failure_blocks_b, test_a_gpu_failure_allows_b, test_b_requires_both_large_scenes, test_b_failure_is_blocker]


def main() -> int:
    for test in TESTS:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS {len(TESTS)}/{len(TESTS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
