#!/usr/bin/env python3
"""Render frozen heldout RGB views from a formal QGS checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_gaussian_rgb import export_rgb  # noqa: E402
from export_qgs_depth_maps import load_config, load_qgs_runtime  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_repo", required=True)
    parser.add_argument("--qgs_config_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--camera_source_path", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--camera_sets", choices=["all", "train", "test"], default="train")
    parser.add_argument("--rgb_contract", required=True)
    parser.add_argument("--input_manifest", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--method_id", default="qgs")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--manifest_path", default="")
    parser.add_argument("--appearance_policy", default="none")
    parser.add_argument("--benchmark_repo", required=True)
    parser.add_argument("--benchmark_commit", required=True)
    parser.add_argument("--benchmark_tree", required=True)
    parser.add_argument("--runtime_pythonpath", action="append", default=[])
    parser.add_argument("--allow_review_candidate", action="store_true")
    parser.add_argument("--technical_smoke_root", default="")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo = Path(args.train_repo).expanduser().resolve()
    config = load_config(repo, Path(args.qgs_config_path).expanduser().resolve())
    config.model_path = str(Path(args.model_path).expanduser().resolve())
    camera_source = Path(args.camera_source_path).expanduser().resolve()
    if not (camera_source / "sparse" / "0").is_dir():
        raise FileNotFoundError(camera_source / "sparse" / "0")
    config.root_dir = str(camera_source)
    config.gs_model.root_dir = str(camera_source)
    runtime = load_qgs_runtime(repo, config)
    runtime["safe_state"](args.quiet)
    args.adapter_kind = "qgs_rgb_v1"
    args.adapter_path = str(Path(__file__).resolve())
    dataset = SimpleNamespace(
        sh_degree=int(config.gs_model.sh_degree),
        white_background=bool(config.dataset.white_background),
        model_path=str(config.model_path),
        source_path=str(config.root_dir),
        train_test_exp=False,
    )
    old_cwd = Path.cwd()
    os.chdir(repo)
    try:
        manifest = export_rgb(args, dataset, config.pipeline, runtime)
    finally:
        os.chdir(old_cwd)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
