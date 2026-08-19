#!/usr/bin/env python3
"""CPU checks for MetroGS's upstream depth-prior filtering semantics."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from prepare_metrogs_training_priors import validate_scales


def main() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        path = root / "estimated_mask_depth_scales.json"
        values = {
            "accepted_a.JPG": {"scale": 1.0, "offset": 0.1},
            "accepted_b.JPG": {"scale": 1.2, "offset": -0.2},
            "rejected_low.JPG": {"scale": 0.1, "offset": 0.0},
            "rejected_high.JPG": {"scale": 8.0, "offset": 0.0},
        }
        path.write_text(json.dumps(values), encoding="utf-8")
        rows, median, accepted, rejected = validate_scales(path, set(values))

        assert median == 1.1
        assert accepted == ["accepted_a.JPG", "accepted_b.JPG"]
        assert rejected == ["rejected_high.JPG", "rejected_low.JPG"]
        by_name = {row["image_name"]: row for row in rows}
        assert by_name["accepted_a.JPG"]["official_depth_prior_accepted"] is True
        assert by_name["rejected_low.JPG"]["official_depth_prior_accepted"] is False

        missing = root / "missing.json"
        missing.write_text(
            json.dumps({"accepted_a.JPG": values["accepted_a.JPG"]}), encoding="utf-8"
        )
        try:
            validate_scales(missing, set(values))
        except RuntimeError as exc:
            assert "inventory mismatch" in str(exc)
        else:
            raise AssertionError("missing scale entry was not rejected")

    print("metrogs_prior_filter_test: PASS")


if __name__ == "__main__":
    main()
