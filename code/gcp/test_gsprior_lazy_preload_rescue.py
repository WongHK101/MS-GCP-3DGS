#!/usr/bin/env python3
"""CPU-only tests for the bounded GSPrior lazy-residency retry."""

from __future__ import annotations

import unittest

from run_gsprior_lazy_preload_rescue import (
    LAZY_PRELOAD_PARSE_BLOCK,
    ORIGINAL_PARSE_BLOCK,
    patch_training_source,
    validate_formal_argv,
)


class GSPriorLazyPreloadRescueTest(unittest.TestCase):
    def test_patch_changes_only_the_unique_argument_parse_block(self) -> None:
        source = "prefix\n" + ORIGINAL_PARSE_BLOCK + "suffix\n"
        patched = patch_training_source(source)
        self.assertEqual(
            patched, "prefix\n" + LAZY_PRELOAD_PARSE_BLOCK + "suffix\n"
        )

    def test_formal_budget_is_required(self) -> None:
        validate_formal_argv(
            [
                "-s", "/dataset", "-m", "/attempt/model", "--resolution", "1",
                "--iterations", "40000", "--test_iterations", "20000", "30000", "40000",
                "--save_iterations", "20000", "30000", "40000",
                "--checkpoint_iterations", "20000", "30000", "40000", "--quiet",
            ]
        )

    def test_changed_budget_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_formal_argv(
                [
                    "--resolution", "1", "--iterations", "39999",
                    "--test_iterations", "20000", "30000", "40000",
                    "--save_iterations", "20000", "30000", "40000",
                    "--checkpoint_iterations", "20000", "30000", "40000",
                ]
            )


if __name__ == "__main__":
    unittest.main()
