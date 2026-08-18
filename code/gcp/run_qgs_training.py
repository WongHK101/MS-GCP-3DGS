#!/usr/bin/env python3
"""Run the frozen QGS training function without modifying upstream sources."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qgs_repo", type=Path, required=True)
    parser.add_argument("--conf_path", type=Path, required=True)
    parser.add_argument("--test_iterations", nargs="*", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, required=True)
    parser.add_argument("--checkpoint_iterations", nargs="*", type=int, default=[])
    parser.add_argument("--start_checkpoint", default=None)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    repo = args.qgs_repo.expanduser().resolve()
    conf_path = args.conf_path.expanduser().resolve()
    sys.path.insert(0, str(repo))
    old_cwd = Path.cwd()
    os.chdir(repo)
    try:
        from train import training
        from utils.general_utils import safe_state

        config = OmegaConf.merge(
            OmegaConf.load(repo / "config/base.yaml"),
            OmegaConf.load(conf_path),
        )
        OmegaConf.resolve(config)
        config.model_path = str(config.model_path).replace(" ", "").replace("\n", "")

        safe_state(args.quiet)
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.set_device(torch.device("cuda:0"))
        torch.autograd.set_detect_anomaly(False)
        training(
            config,
            args.test_iterations,
            args.save_iterations,
            args.checkpoint_iterations,
            args.start_checkpoint,
            args.debug_from,
        )
    finally:
        os.chdir(old_cwd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
