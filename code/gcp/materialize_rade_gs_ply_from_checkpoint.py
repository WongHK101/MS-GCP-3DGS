#!/usr/bin/env python3
"""Materialize one RaDe-GS PLY from an exact 30K checkpoint and filter sidecar."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def ply_vertex_count(path: Path) -> int:
    with path.open("rb") as handle:
        for _ in range(256):
            raw = handle.readline()
            if not raw:
                break
            line = raw.decode("ascii", errors="strict").strip()
            if line.startswith("element vertex "):
                return int(line.split()[-1])
            if line == "end_header":
                break
    raise RuntimeError("PLY vertex count not found")


def git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True
    ).rstrip("\r\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--expected-train-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--filter-sidecar", type=Path, required=True)
    parser.add_argument("--filter-sidecar-sha256", required=True)
    parser.add_argument("--serialization-state", type=Path, required=True)
    parser.add_argument("--output-ply", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    checkpoint = args.checkpoint.resolve()
    sidecar_path = args.filter_sidecar.resolve()
    state_path = args.serialization_state.resolve()
    output = args.output_ply.resolve()
    receipt = args.receipt.resolve()
    for path in (source_root, checkpoint, sidecar_path, state_path):
        if not path.exists():
            raise FileNotFoundError(path)
    for path in (output, receipt):
        if path.exists():
            raise FileExistsError(path)
    if git_output(source_root, "rev-parse", "HEAD") != args.source_commit:
        raise RuntimeError("RaDe-GS source commit mismatch")
    if git_output(source_root, "rev-parse", "HEAD^{tree}") != args.source_tree:
        raise RuntimeError("RaDe-GS source tree mismatch")
    if git_output(source_root, "status", "--porcelain=v1"):
        raise RuntimeError("RaDe-GS source worktree is dirty")
    if sha256(source_root / "train.py") != args.expected_train_sha256:
        raise RuntimeError("RaDe-GS train.py hash mismatch")
    if sha256(checkpoint) != args.checkpoint_sha256:
        raise RuntimeError("30K checkpoint hash mismatch")
    if sha256(sidecar_path) != args.filter_sidecar_sha256:
        raise RuntimeError("filter_3D sidecar hash mismatch")

    state = read_json(state_path)
    if (
        state.get("schema") != "m3m_gcp_rade_gs_final_serialization_state_v1"
        or state.get("status") != "COMPLETE"
        or state.get("iteration") != 30_000
        or Path(state.get("checkpoint", {}).get("path", "")).resolve() != checkpoint
        or state.get("checkpoint", {}).get("bytes") != checkpoint.stat().st_size
        or Path(
            state.get("filter_3D_sidecar", {}).get("path", "")
        ).resolve()
        != sidecar_path
        or state.get("filter_3D_sidecar", {}).get("bytes")
        != sidecar_path.stat().st_size
        or state.get("filter_3D_sidecar", {}).get("sha256")
        != args.filter_sidecar_sha256
        or not state.get("no_optimization_or_filter_recompute_between_artifacts")
    ):
        raise RuntimeError("final serialization state receipt mismatch")

    sys.path.insert(0, str(source_root))
    os.chdir(source_root)
    import torch
    from arguments import OptimizationParams
    from scene import GaussianModel

    opt_parser = ArgumentParser(add_help=False)
    op = OptimizationParams(opt_parser)
    opt_args = opt_parser.parse_args(["--iterations", "30000"])
    opt = op.extract(opt_args)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, tuple) or len(payload) != 2 or payload[1] != 30_000:
        raise RuntimeError("native RaDe-GS checkpoint identity mismatch")
    model_params, iteration = payload
    if not isinstance(model_params, tuple) or len(model_params) != 15:
        raise RuntimeError("native gaussians.capture() payload shape mismatch")

    gaussians = GaussianModel(3)
    gaussians.app_model = model_params[12]
    gaussians._appearance_embeddings = model_params[14]
    gaussians.restore(model_params, opt)

    sidecar = torch.load(sidecar_path, map_location="cpu", weights_only=False)
    expected_shape = state["filter_3D_sidecar"]["shape"]
    if list(sidecar.shape) != expected_shape:
        raise RuntimeError("filter_3D sidecar shape differs from receipt")
    if str(sidecar.dtype) != state["filter_3D_sidecar"]["dtype"]:
        raise RuntimeError("filter_3D sidecar dtype differs from receipt")
    if sidecar.ndim != 2 or sidecar.shape[1] != 1:
        raise RuntimeError("invalid filter_3D sidecar shape")
    if int(sidecar.shape[0]) != int(gaussians.get_xyz.shape[0]):
        raise RuntimeError("filter_3D and Gaussian point counts differ")
    if int(sidecar.shape[0]) != int(state.get("gaussian_count", -1)):
        raise RuntimeError("filter_3D and serialization receipt counts differ")
    if not bool(torch.isfinite(sidecar).all().item()):
        raise RuntimeError("filter_3D sidecar contains non-finite values")
    gaussians.filter_3D = sidecar.to(
        device=gaussians.get_xyz.device, dtype=gaussians.get_xyz.dtype
    )

    # Optimizer state is not part of PLY serialization.  Release it before the
    # official writer builds its large host-side structured arrays.
    gaussians.optimizer = None
    del payload, model_params, sidecar
    gc.collect()

    output.parent.mkdir(parents=True, exist_ok=True)
    gaussians.save_ply(str(output))
    vertices = ply_vertex_count(output)
    if vertices != int(gaussians.get_xyz.shape[0]):
        raise RuntimeError("materialized PLY vertex count mismatch")

    result = {
        "schema": "m3m_gcp_rade_gs_checkpoint_ply_materialization_v1",
        "status": "PASS",
        "iteration": iteration,
        "source": {
            "root": str(source_root),
            "commit": args.source_commit,
            "tree": args.source_tree,
            "train_sha256": args.expected_train_sha256,
        },
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": args.checkpoint_sha256,
        },
        "filter_3D_sidecar": {
            "path": str(sidecar_path),
            "sha256": args.filter_sidecar_sha256,
            "shape": expected_shape,
            "dtype": state["filter_3D_sidecar"]["dtype"],
        },
        "output": {
            "path": str(output),
            "bytes": output.stat().st_size,
            "sha256": sha256(output),
            "vertices": vertices,
        },
        "reads_training_rgb": False,
        "recomputes_filter_3D": False,
        "optimizer_steps": 0,
        "finished_utc": utc_now(),
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    write_json(receipt, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
