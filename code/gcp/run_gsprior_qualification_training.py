#!/usr/bin/env python3
"""Exercise one real GSPrior TSDF refresh and a post-TSDF optimizer step."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gsprior_repo", type=Path, required=True)
    parser.add_argument("--source_path", type=Path, required=True)
    parser.add_argument("--model_path", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    cli = parse_args()
    repo = cli.gsprior_repo.resolve()
    source = cli.source_path.resolve()
    model = cli.model_path.resolve()
    if model.exists():
        raise FileExistsError(f"qualification model must be fresh: {model}")
    if not (source / "sparse" / "images.bin").is_file():
        raise FileNotFoundError(f"GSPrior flat sparse model missing: {source}")

    sys.path.insert(0, str(repo))
    old_cwd = Path.cwd()
    os.chdir(repo)
    try:
        from argparse import ArgumentParser

        import torch
        from arguments import ModelParams, OptimizationParams, PipelineParams
        from train import training
        from utils.general_utils import safe_state

        parser = ArgumentParser()
        lp = ModelParams(parser)
        op = OptimizationParams(parser)
        pp = PipelineParams(parser)
        args = parser.parse_args(
            [
                "--source_path",
                str(source),
                "--model_path",
                str(model),
                "--resolution",
                "1",
                "--iterations",
                "4",
            ]
        )
        safe_state(True)
        training(
            lp.extract(args),
            op.extract(args),
            pp.extract(args),
            testing_iterations=[],
            saving_iterations=[2, 4],
            checkpoint_iterations=[],
            checkpoint="",
            debug_from=-100,
            exp_name=model.name,
            tsdf_iter=[2],
            tsdf_trunc=[96],
        )
        torch.cuda.synchronize()
    finally:
        os.chdir(old_cwd)

    required = [
        model / "point_cloud" / "iteration_2" / "point_cloud.ply",
        model / "point_cloud" / "iteration_4" / "point_cloud.ply",
        model / f"tsdf_{model.name}_2.npy",
        model / f"vol_origin_{model.name}_2.npy",
        model / f"gsprior_{model.name}_2.ply",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"GSPrior qualification outputs missing: {missing}")
    payload = {
        "schema": "m3m_gsprior_tsdf_branch_qualification_v1",
        "status": "PASS",
        "iterations": 4,
        "tsdf_iterations": [2],
        "tsdf_truncation_values": [96],
        "post_tsdf_optimizer_step_executed": True,
        "formal_training_started": False,
        "model_path": str(model),
        "outputs": [
            {"path": str(path), "bytes": path.stat().st_size}
            for path in required
        ],
    }
    (model / "qualification_validation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
