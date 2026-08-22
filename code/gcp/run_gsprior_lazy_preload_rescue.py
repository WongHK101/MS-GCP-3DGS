#!/usr/bin/env python3
"""Run the approved GSPrior retry with lazy RGB residency.

The method source remains untouched.  This wrapper verifies the exact formal
``train.py``, writes one run-local copy, and forces only ``preload_img=False``
after the official argument parser.  Training, TSDF, losses, schedules, seed,
and view order are otherwise unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ORIGINAL_TRAIN_SHA256 = (
    "44e434741b68075d7e3f92f2c71f471cab2c66e4d3bd129261e7701da03274f1"
)

ORIGINAL_PARSE_BLOCK = '''    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
'''

LAZY_PRELOAD_PARSE_BLOCK = '''    args = parser.parse_args(sys.argv[1:])
    # M3M-GCP bounded 100K resource rescue: change image residency only.
    args.preload_img = False
    args.save_iterations.append(args.iterations)
'''


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def option_value(argv: list[str], *names: str) -> str:
    indices = [index for index, value in enumerate(argv) if value in names]
    if len(indices) != 1 or indices[0] + 1 >= len(argv):
        raise ValueError(f"expected exactly one option from {names}")
    return argv[indices[0] + 1]


def option_values(argv: list[str], name: str) -> list[str]:
    indices = [index for index, value in enumerate(argv) if value == name]
    if len(indices) != 1:
        raise ValueError(f"expected exactly one {name}")
    start = indices[0] + 1
    end = start
    while end < len(argv) and not argv[end].startswith("-"):
        end += 1
    return argv[start:end]


def patch_training_source(source: str) -> str:
    if source.count(ORIGINAL_PARSE_BLOCK) != 1:
        raise RuntimeError("official GSPrior argument-parse block identity/count mismatch")
    patched = source.replace(ORIGINAL_PARSE_BLOCK, LAZY_PRELOAD_PARSE_BLOCK, 1)
    if patched.count("args.preload_img = False") != 1:
        raise RuntimeError("GSPrior lazy-residency injection count mismatch")
    return patched


def validate_formal_argv(argv: list[str]) -> None:
    if option_value(argv, "--iterations") != "40000":
        raise ValueError("GSPrior rescue requires the unchanged 40000 iterations")
    if option_value(argv, "--resolution") != "1":
        raise ValueError("GSPrior rescue requires the unchanged resolution=1")
    frozen = ["20000", "30000", "40000"]
    for option in (
        "--test_iterations",
        "--save_iterations",
        "--checkpoint_iterations",
    ):
        if option_values(argv, option) != frozen:
            raise ValueError(f"GSPrior rescue requires unchanged {option}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-train-sha256", default=ORIGINAL_TRAIN_SHA256)
    parser.add_argument("training_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    training_args = list(args.training_args)
    if training_args and training_args[0] == "--":
        training_args.pop(0)
    if not training_args:
        raise ValueError("missing forwarded GSPrior training arguments")
    validate_formal_argv(training_args)

    source_root = args.source_root.resolve()
    source_path = source_root / "train.py"
    actual_source_sha = sha256(source_path)
    if actual_source_sha != args.expected_train_sha256:
        raise RuntimeError("official GSPrior train.py hash mismatch")
    patched = patch_training_source(source_path.read_text(encoding="utf-8"))

    model_root = Path(option_value(training_args, "-m", "--model_path")).resolve()
    attempt_root = model_root.parent
    generated_root = attempt_root / "materialized_sources"
    generated_root.mkdir(parents=True, exist_ok=False)
    generated_path = generated_root / "gsprior_train_lazy_preload_rescue.py"
    generated_path.write_text(patched, encoding="utf-8", newline="\n")

    receipt = {
        "schema": "m3m_gcp_gsprior_lazy_preload_materialization_v1",
        "status": "COMPLETE",
        "official_source_path": str(source_path),
        "official_source_sha256": actual_source_sha,
        "materialized_source_path": str(generated_path),
        "materialized_source_sha256": sha256(generated_path),
        "parse_block_before_sha256": hashlib.sha256(
            ORIGINAL_PARSE_BLOCK.encode("utf-8")
        ).hexdigest(),
        "parse_block_after_sha256": hashlib.sha256(
            LAZY_PRELOAD_PARSE_BLOCK.encode("utf-8")
        ).hexdigest(),
        "forced_argument": {"preload_img": False},
        "scientific_contract_changes": [],
        "data_residency_only": True,
        "created_utc": utc_now(),
    }
    write_json(attempt_root / "gsprior_lazy_preload_materialization.json", receipt)

    os.chdir(source_root)
    sys.path.insert(0, str(source_root))
    sys.argv = [str(source_path), *training_args]
    namespace = {
        "__name__": "__main__",
        "__file__": str(source_path),
        "__package__": None,
        "__cached__": None,
    }
    exec(compile(patched, str(source_path), "exec"), namespace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
