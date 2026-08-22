#!/usr/bin/env python3
"""CPU-only lifecycle checks for the CityGS-X 100K wrapper."""

from __future__ import annotations

import unittest
from pathlib import Path

from run_citygs_x_100k_training import build_command


def option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


class CityGSX100KTrainingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.common = {
            "python": Path("/env/bin/python"),
            "repo": Path("/repo"),
            "dataset": Path("/dataset"),
            "model_path": Path("/run/model"),
            "mode": "formal",
            "iterations": 100_000,
        }

    def test_default_preserves_legacy_in_training_evaluation(self) -> None:
        command = build_command(**self.common)
        self.assertEqual(option(command, "--test_iterations"), "100000")
        self.assertEqual(option(command, "--save_iterations"), "100000")

    def test_qualification_recipe_can_save_before_offline_evaluation(self) -> None:
        command = build_command(**self.common, defer_evaluation=True)
        self.assertEqual(option(command, "--test_iterations"), "100001")
        self.assertEqual(option(command, "--save_iterations"), "100000")


if __name__ == "__main__":
    unittest.main()
