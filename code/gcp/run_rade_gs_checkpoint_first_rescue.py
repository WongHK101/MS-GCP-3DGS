#!/usr/bin/env python3
"""Run RaDe-GS with a bounded final-serialization rescue.

The official training source is left untouched.  This wrapper verifies its exact
hash, materializes one run-local copy, and changes only the iteration-30,000
serialization order:

1. save the native ``gaussians.capture()`` checkpoint;
2. save the exact ``filter_3D`` tensor as a sidecar;
3. call the original ``scene.save()`` implementation.

No optimization, filtering, or evaluation happens between these operations.
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


ORIGINAL_TRAIN_SHA256 = "10dd1be2b912091db9f7d15afcf3ee088d9e550d6149cc536d7292a595da5328"

ORIGINAL_SAVE_BLOCK = '''            if iteration in saving_iterations:
                print("\\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
'''

CHECKPOINT_FIRST_SAVE_BLOCK = '''            if iteration in checkpoint_iterations and iteration == opt.iterations:
                _m3m_save_final_serialization_state(gaussians, scene.model_path, iteration)

            if iteration in saving_iterations:
                print("\\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
'''

ORIGINAL_CHECKPOINT_CONDITION = "            if iteration in checkpoint_iterations:\n"
NONFINAL_CHECKPOINT_CONDITION = (
    "            if iteration in checkpoint_iterations and iteration != opt.iterations:\n"
)


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
    if source.count(ORIGINAL_SAVE_BLOCK) != 1:
        raise RuntimeError("official RaDe-GS save block identity/count mismatch")
    if source.count(ORIGINAL_CHECKPOINT_CONDITION) != 1:
        raise RuntimeError("official RaDe-GS checkpoint block identity/count mismatch")
    patched = source.replace(
        ORIGINAL_CHECKPOINT_CONDITION, NONFINAL_CHECKPOINT_CONDITION, 1
    )
    patched = patched.replace(ORIGINAL_SAVE_BLOCK, CHECKPOINT_FIRST_SAVE_BLOCK, 1)
    if patched.count("_m3m_save_final_serialization_state(") != 1:
        raise RuntimeError("RaDe-GS rescue injection count mismatch")
    return patched


_PATCH_CONTEXT: dict[str, Any] = {}


def _m3m_save_final_serialization_state(
    gaussians: Any, model_path: str, iteration: int
) -> None:
    """Atomically seal the exact final state before the original PLY writer."""
    import torch

    if iteration != 30_000:
        raise RuntimeError("serialization rescue is authorized only at iteration 30000")

    model_root = Path(model_path).resolve()
    checkpoint = model_root / f"chkpnt{iteration}.pth"
    sidecar = model_root / f"chkpnt{iteration}.filter_3D.pt"
    receipt = model_root / f"chkpnt{iteration}.serialization_state.json"
    for path in (checkpoint, sidecar, receipt):
        if path.exists():
            raise FileExistsError(path)

    checkpoint_tmp = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
    sidecar_tmp = sidecar.with_suffix(sidecar.suffix + ".tmp")
    try:
        print()
        print(f"[ITER {iteration}] Saving Checkpoint Before Final PLY")
        torch.save((gaussians.capture(), iteration), checkpoint_tmp)
        os.replace(checkpoint_tmp, checkpoint)

        filter_cpu = gaussians.filter_3D.detach().cpu().contiguous()
        if filter_cpu.ndim != 2 or filter_cpu.shape[1] != 1:
            raise RuntimeError(f"unexpected filter_3D shape: {tuple(filter_cpu.shape)}")
        if int(filter_cpu.shape[0]) != int(gaussians.get_xyz.shape[0]):
            raise RuntimeError("filter_3D and Gaussian point counts differ")
        finite = bool(torch.isfinite(filter_cpu).all().item())
        if not finite:
            raise RuntimeError("filter_3D contains non-finite values")

        print(f"[ITER {iteration}] Saving Exact filter_3D Sidecar")
        torch.save(filter_cpu, sidecar_tmp)
        os.replace(sidecar_tmp, sidecar)
        payload = {
            "schema": "m3m_gcp_rade_gs_final_serialization_state_v1",
            "status": "COMPLETE",
            "iteration": iteration,
            "checkpoint": {
                "path": str(checkpoint),
                "bytes": checkpoint.stat().st_size,
                "sha256_deferred_until_training_process_exit": True,
                "format": "native_torch_save_gaussians_capture_and_iteration",
            },
            "filter_3D_sidecar": {
                "path": str(sidecar),
                "bytes": sidecar.stat().st_size,
                "sha256": sha256(sidecar),
                "dtype": str(filter_cpu.dtype),
                "shape": list(filter_cpu.shape),
                "finite": finite,
            },
            "gaussian_count": int(gaussians.get_xyz.shape[0]),
            "ordering": [
                "native_checkpoint",
                "exact_filter_3D_sidecar",
                "original_scene_save_follows",
            ],
            "no_optimization_or_filter_recompute_between_artifacts": True,
            "source_materialization": _PATCH_CONTEXT,
            "finished_utc": utc_now(),
        }
        write_json(receipt, payload)
    finally:
        checkpoint_tmp.unlink(missing_ok=True)
        sidecar_tmp.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--expected-train-sha256", default=ORIGINAL_TRAIN_SHA256
    )
    parser.add_argument("training_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    training_args = list(args.training_args)
    if training_args and training_args[0] == "--":
        training_args.pop(0)
    if not training_args:
        raise ValueError("missing forwarded RaDe-GS training arguments")
    if option_value(training_args, "--iterations") != "30000":
        raise ValueError("rescue wrapper requires the unchanged 30000-iteration budget")
    if option_values(training_args, "--checkpoint_iterations") != ["15000", "30000"]:
        raise ValueError("rescue wrapper requires checkpoints at 15000 and 30000")
    if option_values(training_args, "--save_iterations") != ["7000", "30000"]:
        raise ValueError("rescue wrapper requires unchanged save iterations")

    source_root = args.source_root.resolve()
    source_path = source_root / "train.py"
    if sha256(source_path) != args.expected_train_sha256:
        raise RuntimeError("official RaDe-GS train.py hash mismatch")
    source = source_path.read_text(encoding="utf-8")
    patched = patch_training_source(source)

    model_root = Path(option_value(training_args, "-m", "--model_path")).resolve()
    attempt_root = model_root.parent
    generated_root = attempt_root / "materialized_sources"
    generated_root.mkdir(parents=True, exist_ok=False)
    generated_path = generated_root / "rade_gs_train_checkpoint_first_rescue.py"
    generated_path.write_text(patched, encoding="utf-8", newline="\n")

    global _PATCH_CONTEXT
    _PATCH_CONTEXT = {
        "official_source_path": str(source_path),
        "official_source_sha256": args.expected_train_sha256,
        "materialized_source_path": str(generated_path),
        "materialized_source_sha256": sha256(generated_path),
        "save_block_before_sha256": hashlib.sha256(
            ORIGINAL_SAVE_BLOCK.encode("utf-8")
        ).hexdigest(),
        "save_block_after_sha256": hashlib.sha256(
            CHECKPOINT_FIRST_SAVE_BLOCK.encode("utf-8")
        ).hexdigest(),
        "scientific_contract_changes": [],
        "serialization_only": True,
    }
    write_json(
        attempt_root / "rade_gs_checkpoint_first_materialization.json",
        {
            "schema": "m3m_gcp_rade_gs_checkpoint_first_materialization_v1",
            "status": "COMPLETE",
            **_PATCH_CONTEXT,
            "created_utc": utc_now(),
        },
    )

    os.chdir(source_root)
    sys.path.insert(0, str(source_root))
    sys.argv = [str(source_path), *training_args]
    namespace = {
        "__name__": "__main__",
        "__file__": str(source_path),
        "__package__": None,
        "__cached__": None,
        "_m3m_save_final_serialization_state": _m3m_save_final_serialization_state,
    }
    exec(compile(patched, str(source_path), "exec"), namespace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
