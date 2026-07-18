#!/usr/bin/env python3
"""Run the official 3DGS training loop with identical micro-run audit traces."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def main() -> int:
    outer = argparse.ArgumentParser(description=__doc__)
    outer.add_argument("--method_root", type=Path, required=True)
    outer.add_argument("--trace_path", type=Path, required=True)
    outer.add_argument("official_args", nargs=argparse.REMAINDER)
    parsed = outer.parse_args()
    official_args = list(parsed.official_args)
    if official_args and official_args[0] == "--":
        official_args = official_args[1:]
    method_root = parsed.method_root.resolve()
    sys.path.insert(0, str(method_root))
    import torch
    import train as official
    from argparse import ArgumentParser
    from arguments import ModelParams, OptimizationParams, PipelineParams

    trace = {
        "schema": "gs_gcp_original_3dgs_micro_trace_v1",
        "status": "RUNNING",
        "official_child_argv": official_args,
        "train_image_order": [],
        "iterations": [],
    }
    original_render = official.render
    original_report = official.training_report

    def traced_render(viewpoint, *args, **kwargs):
        trace["train_image_order"].append(str(viewpoint.image_name))
        return original_render(viewpoint, *args, **kwargs)

    def traced_report(tb_writer, iteration, ll1, loss, l1_loss, elapsed, testing_iterations, scene, render_func, render_args):
        original_report(
            tb_writer, iteration, ll1, loss, l1_loss, elapsed,
            testing_iterations, scene, render_func, render_args,
        )
        ll1_value = float(ll1.detach().item())
        loss_value = float(loss.detach().item())
        if not math.isfinite(ll1_value) or not math.isfinite(loss_value):
            raise ValueError("micro loss trace contains NaN/Inf")
        trace["iterations"].append({
            "iteration": int(iteration),
            "l1": format(ll1_value, ".17g"),
            "loss": format(loss_value, ".17g"),
            "gaussian_count_before_densification": int(scene.gaussians.get_xyz.shape[0]),
        })

    official.render = traced_render
    official.training_report = traced_report
    parser = ArgumentParser(description="Official 3DGS micro-run child")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(official_args)
    args.save_iterations.append(args.iterations)
    try:
        official.safe_state(args.quiet)
        official.network_gui.init(args.ip, args.port)
        torch.autograd.set_detect_anomaly(args.detect_anomaly)
        official.training(
            lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations,
            args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from,
        )
        trace["status"] = "PASS"
    except Exception as exc:
        trace["status"] = "FAIL"
        trace["exception"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        parsed.trace_path.parent.mkdir(parents=True, exist_ok=True)
        parsed.trace_path.write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
