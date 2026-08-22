#!/usr/bin/env python3
"""CPU-only checks for the streamlined 100K qualification launcher."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from run_m3m_gcp_100k_qualification import validate_bound_training_input


class QualificationLauncherTest(unittest.TestCase):
    @patch("run_m3m_gcp_100k_qualification.validate_prepared_method_input")
    @patch("run_m3m_gcp_100k_qualification.validate_frozen_training_images")
    def test_launch_preflight_runs_both_live_input_identity_checks(
        self, frozen_images, prepared_input
    ) -> None:
        recipe = {"method_id": "citygaussian_v2"}
        dataset = Path("/dataset")
        validate_bound_training_input(recipe, dataset)
        frozen_images.assert_called_once_with(recipe, dataset)
        prepared_input.assert_called_once_with(recipe, dataset)


if __name__ == "__main__":
    unittest.main()
