#!/usr/bin/env python3
"""Run the approved 3K ten-method held-out LiDAR candidate validation.

This is deliberately a candidate-validation entrypoint, not a formal protocol
promotion.  It replays each method's already successful 66-view exporter
command while changing only the frozen image allowlist and three packet output
paths.  Each 12-view packet set is evaluated with the LiDAR-v1 numeric core and
removed only after a complete, hash-bound result has been written.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import evaluate_m3m_gcp_lidar_formal_v1 as core
from m3m_gcp_lidar_artifacts import canonical_sha256, sha256_file
from rgb_quality_contract import validate_benchmark_checkout
from verify_m3m_gcp_lidar_formal_v1 import METRIC_FIELDS


SCENE = "gcp_3000_20260602"
PROTOCOL_ID = "m3m_gcp_lidar_heldout_visible_surface_candidate_v1"
EXPECTED_METHOD_IDS = (
    "3dgs_original",
    "2dgs",
    "pgsr",
    "rade_gs",
    "qgs",
    "gsprior",
    "sof",
    "citygaussian_v2",
    "citygs_x",
    "metrogs",
)
PACKET_DIR_NAME = "lidar_packets_3k_heldout_candidate_v1"
RESULT_DIR_NAME = "lidar_geometry_3k_heldout_candidate_v1"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def identity(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def replace_unique_option(argv: list[str], option: str, value: Path) -> None:
    indices = [index for index, token in enumerate(argv) if token == option]
    if len(indices) != 1 or indices[0] + 1 >= len(argv):
        raise ValueError(f"expected exactly one value-bearing {option}")
    argv[indices[0] + 1] = str(value.resolve())


def candidate_export_command(
    original: dict[str, Any],
    *,
    allowlist: Path,
    packet_root: Path,
    environment_override: dict[str, str] | None = None,
) -> tuple[list[str], Path, dict[str, str]]:
    argv = [str(token) for token in original.get("argv", [])]
    if not argv:
        raise ValueError("original exporter command has no argv")
    replace_unique_option(argv, "--image_list_csv", allowlist)
    replace_unique_option(argv, "--depth_output_dir", packet_root)
    replace_unique_option(argv, "--manifest_path", packet_root / "depth_export_manifest.json")
    replace_unique_option(argv, "--mapping_csv", packet_root / "depth_map_index.csv")
    if "--camera_sets" in argv:
        camera_sets_index = argv.index("--camera_sets")
        if argv[camera_sets_index + 1] != "train":
            raise ValueError("original exporter is not bound to the full evaluation-camera set")
    working_directory = Path(str(original["working_directory"])).resolve()
    if not working_directory.is_dir():
        raise FileNotFoundError(working_directory)
    environment = dict(os.environ)
    for key, value in original.get("runtime_environment", {}).items():
        if value is None:
            environment.pop(str(key), None)
        else:
            environment[str(key)] = str(value)
    environment.update(
        {str(key): str(value) for key, value in (environment_override or {}).items()}
    )
    return argv, working_directory, environment


def expected_heldout_names(split_path: Path) -> tuple[str, ...]:
    split = read_json(split_path)
    scenes = [row for row in split.get("scenes", []) if row.get("scene") == SCENE]
    if len(scenes) != 1:
        raise ValueError("RGB split does not contain one exact 3K scene")
    names = tuple(sorted(str(name) for name in scenes[0]["test_image_names"]))
    if len(names) != 12 or len(set(names)) != 12:
        raise ValueError("3K held-out candidate requires exactly 12 unique views")
    return names


def method_metadata(registry_path: Path) -> dict[str, dict[str, Any]]:
    registry = read_json(registry_path)
    if registry.get("schema") != "m3m_gcp_native_quarter_method_registry_v3":
        raise ValueError("unexpected method metadata registry")
    selected: dict[str, dict[str, Any]] = {}
    for row in registry.get("methods", []):
        method_id = str(row.get("method_id"))
        if method_id not in EXPECTED_METHOD_IDS:
            continue
        formal = row.get("formal_3k_result", {})
        if formal.get("status") != "COMPLETE_RANKED":
            raise ValueError(f"{method_id}: no complete frozen 3K result")
        selected[method_id] = {
            "display_name": str(row["display_name"]),
            "input_class": str(row["input_class"]),
            "model_checkpoint_sha256": str(formal["final_checkpoint_sha256"]),
        }
    if tuple(method_id for method_id in EXPECTED_METHOD_IDS if method_id in selected) != EXPECTED_METHOD_IDS:
        raise ValueError("method metadata registry does not bind all ten methods")
    return selected


def runtime_methods(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    if payload.get("scene") != SCENE:
        raise ValueError("runtime registry scene mismatch")
    rows = payload.get("methods", [])
    ids = tuple(str(row.get("method_id")) for row in rows)
    if ids != EXPECTED_METHOD_IDS:
        raise ValueError(f"runtime method order/set mismatch: {ids}")
    for row in rows:
        run_root = Path(str(row["run_root"])).resolve()
        if not run_root.is_dir():
            raise FileNotFoundError(run_root)
    return rows


def runtime_environment_overrides(path: Path) -> dict[str, dict[str, str]]:
    payload = read_json(path)
    if (
        payload.get("schema")
        != "m3m_gcp_3k_heldout_candidate_environment_overrides_901_v1"
        or payload.get("status") != "ACTIVE_REUSE_PROVEN_EVALUATION_ENVIRONMENTS"
        or payload.get("scene") != SCENE
    ):
        raise ValueError("unexpected 3K candidate environment-override registry")
    environments = payload.get("environments", {})
    if set(environments) != set(EXPECTED_METHOD_IDS):
        raise ValueError("environment overrides do not bind all ten methods")
    output: dict[str, dict[str, str]] = {}
    for method_id, values in environments.items():
        if not isinstance(values, dict):
            raise TypeError(f"{method_id}: environment override must be an object")
        output[str(method_id)] = {str(key): str(value) for key, value in values.items()}
    return output


def load_reference(
    reference_path: Path, expected_origin: np.ndarray, pilot_batch: dict[str, Any]
) -> np.ndarray:
    with np.load(reference_path, allow_pickle=False) as payload:
        reference = payload["points_local_m"]
        origin = payload["local_origin_utm49n_normal_height_m"]
    expected_points = {int(row["reference_points"]) for row in pilot_batch["results"]}
    if (
        reference.dtype != np.float64
        or not np.array_equal(origin, expected_origin)
        or expected_points != {len(reference)}
    ):
        raise ValueError("pilot reference cache numeric frame/count mismatch")
    return reference


def rank_positions(rows: list[dict[str, Any]]) -> dict[str, int]:
    ordered = sorted(
        rows,
        key=lambda row: (
            -float(row["fscore_10cm"]),
            float(row["chamfer_l1_mean_m"]),
            -float(row["precision_10cm"]),
            str(row["method_id"]),
        ),
    )
    return {str(row["method_id"]): index for index, row in enumerate(ordered, 1)}


def ranking_diagnostics(
    candidate_rows: list[dict[str, Any]], pilot_batch: dict[str, Any]
) -> dict[str, Any]:
    pilot_rows = [
        row for row in pilot_batch["results"] if row["method_id"] in EXPECTED_METHOD_IDS
    ]
    if len(pilot_rows) != len(EXPECTED_METHOD_IDS):
        raise ValueError("pilot comparison does not contain all ten methods")
    candidate_rank = rank_positions(candidate_rows)
    pilot_rank = rank_positions(pilot_rows)
    n = len(EXPECTED_METHOD_IDS)
    squared_rank_delta = sum(
        (candidate_rank[method_id] - pilot_rank[method_id]) ** 2
        for method_id in EXPECTED_METHOD_IDS
    )
    spearman = 1.0 - (6.0 * squared_rank_delta) / (n * (n * n - 1))
    concordant = 0
    total = 0
    for left_index, left in enumerate(EXPECTED_METHOD_IDS):
        for right in EXPECTED_METHOD_IDS[left_index + 1 :]:
            total += 1
            pilot_order = pilot_rank[left] < pilot_rank[right]
            candidate_order = candidate_rank[left] < candidate_rank[right]
            concordant += int(pilot_order == candidate_order)
    pilot_by_id = {str(row["method_id"]): row for row in pilot_rows}
    comparisons: list[dict[str, Any]] = []
    for row in candidate_rows:
        method_id = str(row["method_id"])
        baseline = pilot_by_id[method_id]
        comparisons.append(
            {
                "method_id": method_id,
                "pilot_66_rank": pilot_rank[method_id],
                "heldout_12_rank": candidate_rank[method_id],
                "rank_shift": candidate_rank[method_id] - pilot_rank[method_id],
                "surface_voxel_ratio_heldout12_over_pilot66": float(row["reconstruction_points"])
                / float(baseline["reconstruction_points"]),
                "precision_10cm_delta": float(row["precision_10cm"])
                - float(baseline["precision_10cm"]),
                "recall_10cm_delta": float(row["recall_10cm"])
                - float(baseline["recall_10cm"]),
                "fscore_10cm_delta": float(row["fscore_10cm"])
                - float(baseline["fscore_10cm"]),
                "chamfer_l1_delta_m": float(row["chamfer_l1_mean_m"])
                - float(baseline["chamfer_l1_mean_m"]),
            }
        )
    ratios = np.asarray(
        [row["surface_voxel_ratio_heldout12_over_pilot66"] for row in comparisons],
        dtype=np.float64,
    )
    return {
        "interpretation_scope": (
            "candidate coverage/ranking-bias evidence only; no automatic protocol promotion"
        ),
        "pilot_view_count": 66,
        "candidate_view_count": 12,
        "spearman_rank_correlation_fscore_10cm": spearman,
        "pairwise_order_agreement_fraction": concordant / total,
        "pairwise_order_agreement_count": concordant,
        "pairwise_comparison_count": total,
        "same_top_method": min(candidate_rank, key=candidate_rank.get)
        == min(pilot_rank, key=pilot_rank.get),
        "surface_voxel_ratio_mean": float(np.mean(ratios)),
        "surface_voxel_ratio_min": float(np.min(ratios)),
        "surface_voxel_ratio_max": float(np.max(ratios)),
        "surface_voxel_ratio_coefficient_of_variation": float(np.std(ratios) / np.mean(ratios)),
        "methods": comparisons,
    }


def cleanup_packets(packet_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    removed: list[dict[str, Any]] = []
    expected = []
    for row in manifest["depth_index"]:
        packet = Path(str(row["packet_path"])).resolve()
        if packet.parent != packet_root.resolve() or packet.suffix != ".npz":
            raise ValueError(f"refusing packet cleanup outside candidate root: {packet}")
        expected.append(packet)
    for packet in expected:
        item = identity(packet)
        packet.unlink()
        removed.append(item)
    return {
        "schema": "m3m_gcp_3k_heldout_candidate_packet_cleanup_v1",
        "status": "PACKETS_CLEANED_AFTER_COMPLETE_RESULT",
        "created_at": now(),
        "packet_root": str(packet_root.resolve()),
        "removed_count": len(removed),
        "removed": removed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--benchmark-commit", required=True)
    parser.add_argument("--benchmark-tree", required=True)
    parser.add_argument("--runtime-methods", type=Path, required=True)
    parser.add_argument("--environment-overrides", type=Path, required=True)
    parser.add_argument("--method-registry", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--colmap-model", type=Path, required=True)
    parser.add_argument("--sim3-json", type=Path, required=True)
    parser.add_argument("--gcp-csv", type=Path, required=True)
    parser.add_argument("--reference-npz", type=Path, required=True)
    parser.add_argument("--pilot-batch-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--roi-buffer-m", type=float, default=8.0)
    parser.add_argument("--normal-minus-ellipsoid-m", type=float, default=23.980600991639484)
    parser.add_argument("--alpha-min", type=float, default=0.5)
    parser.add_argument("--pixel-stride", type=int, default=4)
    parser.add_argument("--reconstruction-voxel-m", type=float, default=0.05)
    parser.add_argument("--query-chunk-points", type=int, default=250_000)
    parser.add_argument("--thresholds-m", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    parser.add_argument("--threshold-epsilon-m", type=float, default=core.THRESHOLD_EPSILON_M)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for name in (
        "repo",
        "runtime_methods",
        "environment_overrides",
        "method_registry",
        "split",
        "colmap_model",
        "sim3_json",
        "gcp_csv",
        "reference_npz",
        "pilot_batch_result",
        "output_root",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(args.output_root)
    benchmark = validate_benchmark_checkout(
        benchmark_repo=args.repo,
        expected_commit=args.benchmark_commit,
        expected_tree=args.benchmark_tree,
        entrypoint=Path(__file__).resolve(),
    )
    names = expected_heldout_names(args.split)
    metadata = method_metadata(args.method_registry)
    methods = runtime_methods(args.runtime_methods)
    environment_overrides = runtime_environment_overrides(args.environment_overrides)
    pilot_batch = read_json(args.pilot_batch_result)
    sim3 = read_json(args.sim3_json)
    if (
        sim3.get("protocol_id") != core.SOURCE_PROTOCOL_ID
        or sim3.get("scene") != SCENE
        or sim3.get("method_result_refit_forbidden") is not True
    ):
        raise ValueError("common Sim(3) identity/policy mismatch")
    roi, _ = core.build_roi(args.gcp_csv, sim3, args.roi_buffer_m)
    origin = core.freeze_local_origin(roi, args.reconstruction_voxel_m)
    reference = load_reference(args.reference_npz, origin, pilot_batch)
    cameras, images = core.read_colmap_model(args.colmap_model)

    args.output_root.mkdir(parents=True)
    allowlist = args.output_root / "heldout12_allowlist.csv"
    with allowlist.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_name"])
        writer.writerows((name,) for name in names)
    plan_methods: list[dict[str, Any]] = []
    for method in methods:
        run_root = Path(str(method["run_root"])).resolve()
        source_command = run_root / "formal_evaluation/export_resource_probe/command.json"
        packet_root = run_root / "formal_evaluation" / PACKET_DIR_NAME
        result_root = run_root / "formal_evaluation" / RESULT_DIR_NAME
        if packet_root.exists() or result_root.exists():
            raise FileExistsError(f"candidate output already exists: {method['method_id']}")
        plan_methods.append(
            {
                "method_id": method["method_id"],
                "run_root": str(run_root),
                "source_export_command": identity(source_command),
                "packet_root": str(packet_root),
                "result_root": str(result_root),
            }
        )
    plan = {
        "schema": "m3m_gcp_3k_heldout_candidate_validation_plan_v1",
        "protocol_id": PROTOCOL_ID,
        "status": "CANDIDATE_VALIDATION_NOT_FORMAL_PROTOCOL",
        "created_at": now(),
        "scene": SCENE,
        "benchmark": benchmark,
        "heldout_image_names": list(names),
        "allowlist": identity(allowlist),
        "inputs": {
            "runtime_methods": identity(args.runtime_methods),
            "environment_overrides": identity(args.environment_overrides),
            "method_registry": identity(args.method_registry),
            "split": identity(args.split),
            "colmap_cameras": identity(args.colmap_model / "cameras.bin"),
            "colmap_images": identity(args.colmap_model / "images.bin"),
            "sim3": identity(args.sim3_json),
            "gcp_csv": identity(args.gcp_csv),
            "reference": identity(args.reference_npz),
            "pilot_batch_result": identity(args.pilot_batch_result),
        },
        "methods": plan_methods,
    }
    plan["canonical_sha256"] = canonical_sha256(plan)
    write_json(args.output_root / "plan.json", plan)

    rows: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    for index, (method, plan_method) in enumerate(zip(methods, plan_methods), 1):
        method_id = str(method["method_id"])
        run_root = Path(str(method["run_root"])).resolve()
        packet_root = Path(plan_method["packet_root"])
        result_root = Path(plan_method["result_root"])
        receipt: dict[str, Any] = {
            "method_id": method_id,
            "started_at": now(),
            "status": "RUNNING",
            "packet_root": str(packet_root),
            "result_root": str(result_root),
        }
        receipts.append(receipt)
        write_json(args.output_root / "receipt.json", {"status": "RUNNING", "jobs": receipts})
        print(f"[{index}/{len(methods)}] {method_id}: exporting 12 held-out views", flush=True)
        try:
            original_path = run_root / "formal_evaluation/export_resource_probe/command.json"
            original = read_json(original_path)
            packet_root.mkdir(parents=True)
            argv, cwd, env = candidate_export_command(
                original,
                allowlist=allowlist,
                packet_root=packet_root,
                environment_override=environment_overrides[method_id],
            )
            command_record = {
                "schema": "m3m_gcp_3k_heldout_candidate_export_command_v1",
                "source_command": identity(original_path),
                "allowed_changes": [
                    "--image_list_csv",
                    "--depth_output_dir",
                    "--manifest_path",
                    "--mapping_csv",
                ],
                "argv": argv,
                "working_directory": str(cwd),
                "runtime_environment": original.get("runtime_environment", {}),
                "candidate_environment_override": environment_overrides[method_id],
            }
            command_record["canonical_sha256"] = canonical_sha256(command_record)
            write_json(packet_root / "candidate_command.json", command_record)
            export_log = packet_root / "export.log"
            export_started = time.monotonic()
            with export_log.open("wb") as handle:
                completed = subprocess.run(
                    argv,
                    cwd=cwd,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            receipt["export_seconds"] = time.monotonic() - export_started
            receipt["export_returncode"] = completed.returncode
            receipt["export_log"] = identity(export_log)
            if completed.returncode != 0:
                raise RuntimeError(f"exporter returned {completed.returncode}")
            manifest_path = packet_root / "depth_export_manifest.json"
            manifest = read_json(manifest_path)
            core.validate_packet_manifest(manifest, scene=SCENE, expected_image_names=names)
            receipt["packet_manifest"] = identity(manifest_path)

            print(f"[{index}/{len(methods)}] {method_id}: building surface and querying LiDAR", flush=True)
            result_root.mkdir(parents=True)
            evaluation_started = time.monotonic()
            reconstruction, surface_audit = core.build_reconstruction(
                run_root,
                cameras,
                images,
                sim3,
                roi,
                args.alpha_min,
                args.pixel_stride,
                args.reconstruction_voxel_m,
                origin,
                SCENE,
                names,
                manifest_path,
            )
            surface_path = result_root / "surface_voxel_centres_local_metric.npz"
            np.savez_compressed(
                surface_path,
                points_local_m=reconstruction,
                local_origin_utm49n_normal_height_m=origin,
            )
            metrics, recon_to_ref, ref_to_recon = core.summarize_distances(
                reconstruction,
                reference,
                list(args.thresholds_m),
                args.query_chunk_points,
                args.threshold_epsilon_m,
            )
            distances_path = result_root / "nearest_neighbor_distances.npz"
            np.savez_compressed(
                distances_path,
                reconstruction_to_lidar_m=recon_to_ref,
                lidar_to_reconstruction_m=ref_to_recon,
            )
            formal_metrics = {field: metrics[field] for field in METRIC_FIELDS}
            row = {
                "method_id": method_id,
                "method": metadata[method_id]["display_name"],
                "input_class": metadata[method_id]["input_class"],
                "status": "COMPLETE_CANDIDATE_EVIDENCE",
                **formal_metrics,
                "total_seconds": time.monotonic() - evaluation_started,
                "nearest_neighbor_seconds": metrics["nearest_neighbor_seconds"],
                "peak_rss_gib": core.peak_rss_gib(),
                "oom": 0,
            }
            result = {
                "schema": "m3m_gcp_3k_heldout_candidate_method_result_v1",
                "protocol_id": PROTOCOL_ID,
                "status": "COMPLETE_CANDIDATE_EVIDENCE",
                "created_at": now(),
                "scene": SCENE,
                "method_id": method_id,
                "model_checkpoint_sha256": metadata[method_id]["model_checkpoint_sha256"],
                "heldout_image_names": list(names),
                "packet_manifest": identity(manifest_path),
                "surface": identity(surface_path),
                "distances": identity(distances_path),
                "reference": identity(args.reference_npz),
                "surface_audit": surface_audit,
                "metrics": formal_metrics,
                "summary_row": row,
            }
            result["canonical_sha256"] = canonical_sha256(result)
            metrics_path = result_root / "metrics.json"
            write_json(metrics_path, result)
            core.write_csv(result_root / "view_surface_counts.csv", surface_audit["view_rows"])
            rows.append(row)
            write_csv(args.output_root / "lidar_metrics_partial.csv", rows)
            cleanup = cleanup_packets(packet_root, manifest)
            cleanup["canonical_sha256"] = canonical_sha256(cleanup)
            write_json(packet_root / "packet_cleanup_receipt.json", cleanup)
            receipt.update(
                {
                    "finished_at": now(),
                    "status": "COMPLETE_CANDIDATE_EVIDENCE_PACKETS_CLEANED",
                    "metrics": identity(metrics_path),
                    "packet_cleanup": identity(packet_root / "packet_cleanup_receipt.json"),
                    "reconstruction_points": len(reconstruction),
                    "fscore_10cm": formal_metrics["fscore_10cm"],
                }
            )
            print(
                f"[{index}/{len(methods)}] {method_id}: COMPLETE; "
                f"voxels={len(reconstruction):,}, F@10={formal_metrics['fscore_10cm']:.6f}",
                flush=True,
            )
        except Exception as error:  # continue: failures are valid benchmark evidence
            receipt.update(
                {
                    "finished_at": now(),
                    "status": "FAILED_RECORDED_NO_RESCUE",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"[{index}/{len(methods)}] {method_id}: FAILED: {error}", flush=True)
        write_json(args.output_root / "receipt.json", {"status": "RUNNING", "jobs": receipts})

    complete_ids = {str(row["method_id"]) for row in rows}
    all_complete = complete_ids == set(EXPECTED_METHOD_IDS)
    diagnostics = ranking_diagnostics(rows, pilot_batch) if all_complete else None
    batch = {
        "schema": "m3m_gcp_3k_heldout_candidate_batch_result_v1",
        "protocol_id": PROTOCOL_ID,
        "status": (
            "COMPLETE_EVIDENCE_REQUIRES_SCIENTIFIC_JUDGMENT"
            if all_complete
            else "COMPLETE_WITH_RECORDED_FAILURES_NO_PROTOCOL_PROMOTION"
        ),
        "created_at": now(),
        "scene": SCENE,
        "method_count": len(EXPECTED_METHOD_IDS),
        "completed_method_count": len(rows),
        "failed_method_count": len(EXPECTED_METHOD_IDS) - len(rows),
        "protocol_promotion": "NOT_AUTHORIZED_BY_THIS_RUN",
        "plan": identity(args.output_root / "plan.json"),
        "results": rows,
        "ranking_and_coverage_diagnostics": diagnostics,
        "jobs": receipts,
    }
    batch["canonical_sha256"] = canonical_sha256(batch)
    write_json(args.output_root / "batch_result.json", batch)
    write_csv(args.output_root / "lidar_metrics.csv", rows)
    final_receipt = {
        "schema": "m3m_gcp_3k_heldout_candidate_runner_receipt_v1",
        "status": batch["status"],
        "finished_at": now(),
        "batch_result": identity(args.output_root / "batch_result.json"),
        "jobs": receipts,
    }
    final_receipt["canonical_sha256"] = canonical_sha256(final_receipt)
    write_json(args.output_root / "receipt.json", final_receipt)
    print(json.dumps(final_receipt, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if all_complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
