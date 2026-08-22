#!/usr/bin/env python3
"""CPU-only tests for corrected 100K recipe generation."""

from __future__ import annotations

import unittest

from build_m3m_gcp_100k_execution_recipes import METHODS
from build_m3m_gcp_100k_qualification_recipes import (
    GNU_TIME,
    METHOD_INPUT_EVIDENCE_SHA,
    build_recipe,
    corrected_command,
    training_environment,
)


def option_values(command: list[str], option: str) -> list[str]:
    index = command.index(option) + 1
    result = []
    while index < len(command) and not command[index].startswith("-"):
        result.append(command[index])
        index += 1
    return result


class QualificationRecipeTest(unittest.TestCase):
    def test_lifecycle_corrections_do_not_change_save_budget(self) -> None:
        for method in ("2dgs", "rade_gs"):
            command = corrected_command(method, METHODS[method])
            self.assertEqual(option_values(command, "--test_iterations"), ["30001"])
            self.assertEqual(
                option_values(command, "--save_iterations"), ["7000", "30000"]
            )

    def test_import_compatibility_is_explicit(self) -> None:
        self.assertIn("compat/pgsr", training_environment("pgsr")["PYTHONPATH"])
        gsprior = training_environment("gsprior")["PYTHONPATH"]
        self.assertIn("compat/gsprior", gsprior)
        self.assertIn("code/gcp", gsprior)

    def test_3dgs_is_reused_without_training(self) -> None:
        recipe = build_recipe("3dgs_original")
        self.assertIsNone(recipe["training"])
        self.assertFalse(recipe["reuse_model"]["retrain_allowed"])

    def test_known_wrapper_corrections_are_present(self) -> None:
        self.assertIn(
            "--defer_evaluation", build_recipe("citygs_x")["training"]["command"]
        )
        city = build_recipe("citygaussian_v2")["training"]["command"]
        self.assertIn("--sequential_blocks", city)
        self.assertIn("--resume_from", city)
        self.assertIn("--resume_manifest", city)
        self.assertIn(
            "--formal_input_manifest",
            build_recipe("metrogs")["training"]["command"],
        )

    def test_frozen_runtime_tool_and_method_input_lineage_are_bound(self) -> None:
        expected_profiles = {
            "citygaussian_v2": "city_train_records_with_full_all_image_sfm_points",
            "citygs_x": "city_train_records_with_full_all_image_sfm_points",
            "metrogs": "metrogs_reciprocal_train_track_closure_after_all_image_sfm",
        }
        for method in METHODS:
            recipe = build_recipe(method)
            self.assertEqual(
                recipe["training"]["resource_probe"]["time_binary"], GNU_TIME
            )
            self.assertEqual(
                recipe["prepared_method_input_binding"]["evidence_sha256"],
                METHOD_INPUT_EVIDENCE_SHA,
            )
            self.assertTrue(recipe["benchmark_required_files_sha256"])
        for method, profile in expected_profiles.items():
            self.assertEqual(
                build_recipe(method)["prepared_method_input_binding"]["input_profile"],
                profile,
            )
        gsprior = build_recipe("gsprior")
        self.assertIn("prior", gsprior["phase_commands"])
        self.assertEqual(
            gsprior["prepared_method_input_binding"]["input_profile"],
            "exact_formal_train_view_from_shared_all_image_sfm",
        )


if __name__ == "__main__":
    unittest.main()
