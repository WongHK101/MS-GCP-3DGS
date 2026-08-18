#!/usr/bin/env python3
"""CPU-only command-contract checks for the CityGS-X training wrapper."""

from __future__ import annotations

from pathlib import Path

from run_citygs_x_training import build_command


def values(command: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for index, token in enumerate(command[:-1]):
        if token.startswith("--") or token == "-s":
            result[token] = command[index + 1]
    return result


def main() -> int:
    common = {
        "python": Path("/env/bin/python"),
        "repo": Path("/repo"),
        "dataset": Path("/data/MatrixCity-scene/train/block_all"),
        "model_path": Path("/run/model"),
    }
    formal = build_command(mode="formal", iterations=100_000, **common)
    formal_values = values(formal)
    assert formal_values["--iterations"] == "100000"
    assert formal_values["--resolution"] == "1"
    assert formal_values["--images"] == "images"
    assert formal_values["--single_view_weight_from_iter"] == "10000"
    assert formal_values["--dpt_loss_from_iter"] == "10000"
    assert formal_values["--multi_view_weight_from_iter"] == "30000"
    assert formal_values["--multi_view_num"] == "8"
    assert formal_values["--multi_view_max_angle"] == "15"
    assert formal_values["--multi_view_min_dis"] == "0.01"
    assert formal_values["--multi_view_max_dis"] == "25"
    assert formal_values["--dpt_end_iter"] == "30000"
    assert formal_values["--default_voxel_size"] == "0.001"
    assert formal_values["--save_iterations"] == "100000"

    qualification = build_command(mode="qualification", iterations=4, **common)
    qualification_values = values(qualification)
    assert qualification_values["--iterations"] == "4"
    assert qualification_values["--single_view_weight_from_iter"] == "0"
    assert qualification_values["--dpt_loss_from_iter"] == "0"
    assert qualification_values["--multi_view_weight_from_iter"] == "0"
    assert qualification_values["--dpt_end_iter"] == "4"

    try:
        build_command(mode="formal", iterations=99_999, **common)
    except ValueError:
        pass
    else:
        raise AssertionError("non-100K formal budget was accepted")
    print("citygs_x_training_command_test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
