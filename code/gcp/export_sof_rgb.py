#!/usr/bin/env python3
"""Render frozen heldout RGB views from a formal Sorted Opacity Fields model."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_gaussian_depth_maps import (  # noqa: E402
    load_gaussian_runtime,
    parse_train_repo,
)
from export_gaussian_rgb import build_parser, export_rgb  # noqa: E402


def extract_sof_rgb(payload: dict[str, Any]) -> Any:
    """Select the documented RGB prefix from SOF's 10-channel render packet."""

    rendered = payload.get("render")
    shape = getattr(rendered, "shape", ())
    if len(shape) != 3 or int(shape[0]) != 10:
        raise ValueError(
            "SOF render packet must have frozen shape [10,H,W] before RGB extraction, "
            f"got {tuple(shape)}"
        )
    return rendered[:3, ...]


def main() -> int:
    train_repo = parse_train_repo(sys.argv[1:])
    runtime = load_gaussian_runtime(train_repo)
    runtime["extract_rgb"] = extract_sof_rgb
    runtime["rgb_extraction_policy"] = (
        "sof_render_packet_first_three_rgb_channels_of_exact_ten_v1"
    )
    parser, model_group, pipeline_group = build_parser(runtime)
    args = runtime["get_combined_args"](parser)
    runtime["safe_state"](args.quiet)
    dataset = model_group.extract(args)
    pipeline = pipeline_group.extract(args)
    args.adapter_kind = "sof_rgb_v1"
    args.adapter_path = str(Path(__file__).resolve())
    manifest = export_rgb(args, dataset, pipeline, runtime)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
