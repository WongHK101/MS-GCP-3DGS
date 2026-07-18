from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any


METRICS = ("SSIM", "PSNR", "LPIPS")
OFFICIAL_CHUNK_AGGREGATE_ABS_TOLERANCE = 1e-5


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _png_names(path: Path) -> list[str]:
    if not path.is_dir():
        raise FileNotFoundError(f"image directory not found: {path}")
    names = sorted(item.name for item in path.iterdir() if item.is_file() and item.suffix.lower() == ".png")
    if not names:
        raise ValueError(f"no PNG images found: {path}")
    return names


def prepare_chunks(
    renders_dir: Path,
    gt_dir: Path,
    output_root: Path,
    expected_view_count: int,
    chunk_size: int,
    method_name: str,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"chunk output already exists: {output_root}")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    render_names = _png_names(renders_dir)
    gt_names = _png_names(gt_dir)
    if render_names != gt_names:
        raise ValueError("render and ground-truth PNG names differ")
    if len(render_names) != expected_view_count:
        raise ValueError(
            f"rendered view count {len(render_names)} does not match expected {expected_view_count}"
        )

    output_root.mkdir(parents=True)
    chunks: list[dict[str, Any]] = []
    for start in range(0, len(render_names), chunk_size):
        names = render_names[start : start + chunk_size]
        chunk_id = f"chunk_{start // chunk_size:04d}"
        model_root = output_root / chunk_id
        method_root = model_root / "test" / method_name
        chunk_renders = method_root / "renders"
        chunk_gt = method_root / "gt"
        chunk_renders.mkdir(parents=True)
        chunk_gt.mkdir(parents=True)
        for name in names:
            os.symlink((renders_dir / name).resolve(), chunk_renders / name)
            os.symlink((gt_dir / name).resolve(), chunk_gt / name)
        chunks.append(
            {
                "chunk_id": chunk_id,
                "model_path": str(model_root),
                "view_count": len(names),
                "image_names": names,
            }
        )

    manifest = {
        "schema": "gs_gcp_official_metrics_chunk_manifest_v1",
        "source_split": "train",
        "official_metrics_input_alias": "test",
        "method_name": method_name,
        "expected_view_count": expected_view_count,
        "actual_view_count": len(render_names),
        "chunk_size": chunk_size,
        "chunk_count": len(chunks),
        "renders_dir": str(renders_dir.resolve()),
        "gt_dir": str(gt_dir.resolve()),
        "chunks": chunks,
    }
    _write_json(output_root / "chunk_manifest.json", manifest)
    return manifest


def _extract_official_payload(path: Path, method_name: str) -> dict[str, Any]:
    payload = _load_json(path)
    if not isinstance(payload, dict) or set(payload) != {method_name}:
        raise ValueError(f"unexpected method keys in {path}: {sorted(payload)}")
    return payload[method_name]


def merge_chunks(
    chunks_root: Path,
    output_dir: Path,
    expected_view_count: int,
    method_name: str,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"merged output already exists: {output_dir}")
    manifest = _load_json(chunks_root / "chunk_manifest.json")
    if int(manifest["expected_view_count"]) != expected_view_count:
        raise ValueError("chunk manifest expected-view count mismatch")
    if manifest["method_name"] != method_name:
        raise ValueError("chunk manifest method mismatch")

    merged: dict[str, dict[str, float]] = {metric: {} for metric in METRICS}
    chunk_checks: list[dict[str, Any]] = []
    for chunk in manifest["chunks"]:
        model_path = Path(chunk["model_path"])
        results_path = model_path / "results.json"
        per_view_path = model_path / "per_view.json"
        if not results_path.is_file() or not per_view_path.is_file():
            raise FileNotFoundError(f"official metrics output missing for {chunk['chunk_id']}")
        aggregate = _extract_official_payload(results_path, method_name)
        per_view = _extract_official_payload(per_view_path, method_name)
        if set(aggregate) != set(METRICS) or set(per_view) != set(METRICS):
            raise ValueError(f"metric key mismatch in {chunk['chunk_id']}")
        expected_names = set(chunk["image_names"])
        max_chunk_aggregate_error = 0.0
        for metric in METRICS:
            values = per_view[metric]
            if set(values) != expected_names:
                raise ValueError(f"per-view image set mismatch for {chunk['chunk_id']} {metric}")
            for name, value in values.items():
                numeric = float(value)
                if not math.isfinite(numeric):
                    raise ValueError(f"non-finite {metric} for {name}")
                if name in merged[metric]:
                    raise ValueError(f"duplicate per-view metric for {name}")
                merged[metric][name] = numeric
            recomputed = math.fsum(float(value) for value in values.values()) / len(values)
            error = abs(recomputed - float(aggregate[metric]))
            max_chunk_aggregate_error = max(max_chunk_aggregate_error, error)
            if error > OFFICIAL_CHUNK_AGGREGATE_ABS_TOLERANCE:
                raise ValueError(
                    f"official chunk aggregate mismatch for {chunk['chunk_id']} {metric}: {error}"
                )
        chunk_checks.append(
            {
                "chunk_id": chunk["chunk_id"],
                "view_count": len(expected_names),
                "max_official_aggregate_abs_error": max_chunk_aggregate_error,
            }
        )

    for metric in METRICS:
        if len(merged[metric]) != expected_view_count:
            raise ValueError(
                f"merged {metric} count {len(merged[metric])} does not match {expected_view_count}"
            )
    image_sets = [set(merged[metric]) for metric in METRICS]
    if any(names != image_sets[0] for names in image_sets[1:]):
        raise ValueError("merged metric image sets differ")

    aggregates = {
        metric: math.fsum(merged[metric].values()) / expected_view_count for metric in METRICS
    }
    ordered_per_view = {
        metric: {name: merged[metric][name] for name in sorted(merged[metric])} for metric in METRICS
    }
    output_dir.mkdir(parents=True)
    _write_json(output_dir / "results.json", {method_name: aggregates})
    _write_json(output_dir / "per_view.json", {method_name: ordered_per_view})
    summary = {
        "schema": "gs_gcp_reconstruction_fidelity_summary_v1",
        "status": "pass",
        "protocol_id": "graphdeco_train_view_psnr_ssim_lpips_v1",
        "metric_role": "secondary_train_view_reconstruction_fidelity",
        "held_out_novel_view_metric": False,
        "source_split": "train",
        "official_metrics_input_alias": "test",
        "method_name": method_name,
        "view_count": expected_view_count,
        "metrics": aggregates,
        "chunk_checks": chunk_checks,
        "max_chunk_aggregate_abs_error": max(
            check["max_official_aggregate_abs_error"] for check in chunk_checks
        ),
        "official_chunk_aggregate_recompute_abs_tolerance": (
            OFFICIAL_CHUNK_AGGREGATE_ABS_TOLERANCE
        ),
    }
    _write_json(output_dir / "reconstruction_fidelity_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--renders_dir", type=Path, required=True)
    prepare.add_argument("--gt_dir", type=Path, required=True)
    prepare.add_argument("--output_root", type=Path, required=True)
    prepare.add_argument("--expected_view_count", type=int, required=True)
    prepare.add_argument("--chunk_size", type=int, default=64)
    prepare.add_argument("--method_name", default="ours_30000")

    merge = subparsers.add_parser("merge")
    merge.add_argument("--chunks_root", type=Path, required=True)
    merge.add_argument("--output_dir", type=Path, required=True)
    merge.add_argument("--expected_view_count", type=int, required=True)
    merge.add_argument("--method_name", default="ours_30000")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "prepare":
        payload = prepare_chunks(
            args.renders_dir,
            args.gt_dir,
            args.output_root,
            args.expected_view_count,
            args.chunk_size,
            args.method_name,
        )
    else:
        payload = merge_chunks(
            args.chunks_root,
            args.output_dir,
            args.expected_view_count,
            args.method_name,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
