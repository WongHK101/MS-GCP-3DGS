from __future__ import annotations

import argparse
import hashlib
import runpy
import sys
from pathlib import Path

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_script", type=Path, required=True)
    parser.add_argument("--expected_sha256", required=True)
    parser.add_argument("--model_path", type=Path, required=True)
    args = parser.parse_args()

    script = args.metrics_script.resolve()
    if script.name != "metrics.py" or not script.is_file():
        raise ValueError("official metrics.py was not found")
    actual_sha256 = sha256_file(script)
    if actual_sha256 != args.expected_sha256:
        raise ValueError(
            f"official metrics.py SHA-256 mismatch: {actual_sha256} != {args.expected_sha256}"
        )
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"metrics model path not found: {args.model_path}")

    torch.set_grad_enabled(False)
    sys.path.insert(0, str(script.parent))
    sys.argv = [str(script), "-m", str(args.model_path.resolve())]
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
