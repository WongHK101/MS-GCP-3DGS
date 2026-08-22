#!/usr/bin/env python3
"""CPU-only equivalence smoke test for RaDe-GS checkpoint+sidecar PLY recovery."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from argparse import ArgumentParser
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    work_dir = args.work_dir.resolve()
    if work_dir.exists():
        raise FileExistsError(work_dir)
    work_dir.mkdir(parents=True)
    sys.path.insert(0, str(source_root))
    os.chdir(source_root)

    import torch
    from torch import nn
    from arguments import OptimizationParams
    from scene import GaussianModel

    opt_parser = ArgumentParser(add_help=False)
    op = OptimizationParams(opt_parser)
    opt = op.extract(opt_parser.parse_args(["--iterations", "30000"]))

    torch.manual_seed(20260823)
    count = 17
    original = GaussianModel(3)
    original._xyz = nn.Parameter(torch.randn(count, 3))
    original._features_dc = nn.Parameter(torch.randn(count, 1, 3))
    original._features_rest = nn.Parameter(torch.randn(count, 15, 3))
    original._scaling = nn.Parameter(torch.randn(count, 3))
    original._rotation = nn.Parameter(torch.randn(count, 4))
    original._opacity = nn.Parameter(torch.randn(count, 1))
    original.max_radii2D = torch.zeros(count)
    original.spatial_lr_scale = 1.0
    original.app_model = original.App_model.NO
    original._appearance_embeddings = None
    original.filter_3D = torch.rand(count, 1)
    original.training_setup(opt)

    direct = work_dir / "direct.ply"
    checkpoint = work_dir / "chkpnt30000.pth"
    sidecar_path = work_dir / "chkpnt30000.filter_3D.pt"
    recovered = work_dir / "recovered.ply"
    original.save_ply(str(direct))
    torch.save((original.capture(), 30_000), checkpoint)
    torch.save(original.filter_3D.detach().cpu().contiguous(), sidecar_path)

    model_params, iteration = torch.load(
        checkpoint, map_location="cpu", weights_only=False
    )
    restored = GaussianModel(3)
    restored.app_model = model_params[12]
    restored._appearance_embeddings = model_params[14]
    restored.restore(model_params, opt)
    sidecar = torch.load(sidecar_path, map_location="cpu", weights_only=False)
    restored.filter_3D = sidecar.to(
        device=restored.get_xyz.device, dtype=restored.get_xyz.dtype
    )
    restored.optimizer = None
    restored.save_ply(str(recovered))

    direct_sha = sha256(direct)
    recovered_sha = sha256(recovered)
    result = {
        "schema": "m3m_gcp_rade_gs_serialization_rescue_smoke_v1",
        "status": "PASS" if direct_sha == recovered_sha else "FAIL",
        "iteration": iteration,
        "gaussian_count": count,
        "direct_ply_sha256": direct_sha,
        "recovered_ply_sha256": recovered_sha,
        "byte_identical": direct_sha == recovered_sha,
        "device": "cpu",
    }
    (work_dir / "smoke_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["byte_identical"]:
        raise RuntimeError("checkpoint+sidecar PLY differs from direct PLY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
