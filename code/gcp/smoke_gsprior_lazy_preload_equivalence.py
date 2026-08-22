#!/usr/bin/env python3
"""Compare GSPrior preload and lazy image paths on formal RGB inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=3)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    image_root = args.image_root.resolve()
    work_dir = args.work_dir.resolve()
    if work_dir.exists():
        raise FileExistsError(work_dir)
    work_dir.mkdir(parents=True)
    images = sorted(
        path for path in image_root.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )[: args.count]
    if len(images) != args.count:
        raise RuntimeError(f"expected {args.count} formal images, found {len(images)}")

    sys.path.insert(0, str(source_root))
    os.chdir(source_root)
    import numpy as np
    import torch
    from PIL import Image
    from scene.cameras import Camera, process_image

    original_cuda = torch.Tensor.cuda
    torch.Tensor.cuda = lambda self, *pos, **kw: self
    rows = []
    try:
        for index, image_path in enumerate(images):
            with Image.open(image_path) as pil_image:
                resolution = tuple(pil_image.size)
                mode = pil_image.mode
            common = {
                "colmap_id": index,
                "R": np.eye(3),
                "T": np.zeros(3),
                "FoVx": 1.0,
                "FoVy": 1.0,
                "image_width": resolution[0],
                "image_height": resolution[1],
                "image_path": str(image_path),
                "image_name": image_path.stem,
                "uid": index,
                "ncc_scale": 1.0,
                "data_device": "cpu",
            }
            preloaded = Camera(**common, preload_img=True)
            lazy = Camera(**common, preload_img=False)
            preload_rgb, preload_gray = preloaded.get_image()
            lazy_rgb, lazy_gray = lazy.get_image()
            direct_rgb, direct_gray, direct_mask = process_image(
                str(image_path), resolution, 1.0
            )
            checks = {
                "rgb_exact": bool(torch.equal(preload_rgb, lazy_rgb)),
                "gray_exact": bool(torch.equal(preload_gray, lazy_gray)),
                "rgb_direct_exact": bool(torch.equal(lazy_rgb, direct_rgb)),
                "gray_direct_exact": bool(torch.equal(lazy_gray, direct_gray)),
                "rgb_shape_equal": preload_rgb.shape == lazy_rgb.shape,
                "gray_shape_equal": preload_gray.shape == lazy_gray.shape,
                "rgb_dtype_equal": preload_rgb.dtype == lazy_rgb.dtype,
                "gray_dtype_equal": preload_gray.dtype == lazy_gray.dtype,
                "formal_mask_none_both_paths": (
                    preloaded.mask is None and direct_mask is None
                ),
            }
            if not all(checks.values()):
                raise RuntimeError(f"preload/lazy mismatch for {image_path.name}: {checks}")
            rows.append(
                {
                    "image": image_path.name,
                    "image_sha256": sha256(image_path),
                    "mode": mode,
                    "resolution": list(resolution),
                    "rgb_shape": list(preload_rgb.shape),
                    "gray_shape": list(preload_gray.shape),
                    "rgb_dtype": str(preload_rgb.dtype),
                    "gray_dtype": str(preload_gray.dtype),
                    "checks": checks,
                }
            )
    finally:
        torch.Tensor.cuda = original_cuda

    result = {
        "schema": "m3m_gcp_gsprior_lazy_preload_equivalence_smoke_v1",
        "status": "PASS",
        "device": "cpu_with_tensor_cuda_identity_for_camera_metadata",
        "formal_image_root": str(image_root),
        "image_count": len(rows),
        "rows": rows,
        "conclusion": "preload and lazy paths produce exact formal RGB/gray tensors",
    }
    result_path = work_dir / "smoke_result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
