#!/usr/bin/env python3
"""Validate strace file access against Stage 0.5 train-time shared-SfM rules."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


OPEN_PATH = re.compile(r'(?:open|openat|openat2)\([^\"]*\"([^\"]+)\"')


def validate_access(trace: str, train_root: Path, forbidden_roots: list[Path]) -> dict:
    opened = []
    for line in trace.splitlines():
        match = OPEN_PATH.search(line)
        if match:
            opened.append(Path(match.group(1)).resolve())
    forbidden_targets = {
        item.resolve(): str(root)
        for root in forbidden_roots
        if root.exists()
        for item in root.rglob("*")
        if item.is_file()
    }
    forbidden = []
    for path in opened:
        if path in forbidden_targets:
            forbidden.append({"path": str(path), "forbidden_root": forbidden_targets[path], "relative_path": "resolved_file_identity"})
            continue
        for root in forbidden_roots:
            try:
                relative = path.relative_to(root.resolve())
            except ValueError:
                continue
            forbidden.append({"path": str(path), "forbidden_root": str(root), "relative_path": relative.as_posix()})
    full_tracks = [str(path) for path in opened if path.name == "points3D.bin"]
    train_images = (train_root / "images").resolve()
    allowed_train_images = {
        item.resolve() for item in train_images.iterdir()
        if item.is_file() and item.suffix.lower() in (".jpg", ".jpeg")
    } if train_images.is_dir() else set()
    train_image_reads = [
        str(path) for path in opened
        if path in allowed_train_images
        or (path.suffix.lower() in (".jpg", ".jpeg") and train_images in path.parents)
    ]
    passed = not forbidden and not full_tracks and bool(train_image_reads)
    return {
        "schema": "gs_gcp_stage0_5_training_file_access_validation_v1",
        "status": "PASS" if passed else "BLOCKER",
        "opened_path_count": len(opened),
        "train_rgb_read_count": len(set(train_image_reads)),
        "full_points3d_track_reads": sorted(set(full_tracks)),
        "forbidden_accesses": forbidden,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--train_root", type=Path, required=True)
    parser.add_argument("--forbidden_root", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate_access(
        args.trace.read_text(encoding="utf-8", errors="replace"),
        args.train_root,
        args.forbidden_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
