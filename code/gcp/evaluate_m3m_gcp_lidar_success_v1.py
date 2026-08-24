#!/usr/bin/env python3
"""Evaluate one promoted 100K model with the unchanged LiDAR-v1 numeric core.

This entrypoint removes the retired activation-v4 lifecycle gate while keeping
the scientific contract, exact train-view packet, common Sim(3), ROI, surface
construction, thresholds, and distance implementation unchanged.  A shared
scene reference cache is built once and byte-bound for all successful methods.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import evaluate_m3m_gcp_lidar_formal_v1 as core
from m3m_gcp_100k_geometry_paths import (
    formal_input_manifest_canonical_sha256,
    lidar_full_train_packet_manifest,
    lidar_heldout_candidate_packet_manifest,
)
from m3m_gcp_lidar_artifacts import canonical_sha256, sha256_file
from rgb_quality_contract import validate_benchmark_checkout
from verify_m3m_gcp_lidar_formal_v1 import METRIC_FIELDS


SCENE = "gcp_100000_20260610"
RUNTIME_REGISTRY_SCHEMA = (
    "m3m_gcp_native_quarter_rgb_quality_100k_success_registry_v1"
)
PROTOCOL_ID = core.PROTOCOL_ID
SOURCE_PROTOCOL_ID = core.SOURCE_PROTOCOL_ID
HELDOUT_CANDIDATE_PROTOCOL_ID = (
    "m3m_gcp_lidar_heldout_visible_surface_candidate_v1"
)
SURFACE_SAMPLING_TRACKS = {
    "full_train": {
        "split_role": "train",
        "expected_views": 2196,
        "protocol_id": PROTOCOL_ID,
        "protocol_status": (
            "SCIENTIFIC_CONTRACT_UNCHANGED_LIFECYCLE_GATE_REPLACED"
        ),
        "packet_view_split": "exactly all 2196 frozen train views",
        "result_schema": "m3m_gcp_lidar_100k_success_method_result_v1",
        "protocol_schema": "m3m_gcp_lidar_100k_success_method_protocol_v1",
    },
    "heldout_candidate": {
        "split_role": "test",
        "expected_views": 314,
        "protocol_id": HELDOUT_CANDIDATE_PROTOCOL_ID,
        "protocol_status": (
            "HELDOUT_VISIBLE_SURFACE_CANDIDATE_VALIDATION_NOT_FORMAL"
        ),
        "packet_view_split": "exactly all 314 frozen held-out camera poses",
        "result_schema": (
            "m3m_gcp_lidar_100k_heldout_candidate_method_result_v1"
        ),
        "protocol_schema": (
            "m3m_gcp_lidar_100k_heldout_candidate_method_protocol_v1"
        ),
    },
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def validate_query_workers(workers: int) -> int:
    workers = int(workers)
    if workers != -1 and workers < 1:
        raise ValueError("nearest-neighbour query workers must be -1 or positive")
    return workers


def exact_query_distances(
    tree: Any,
    points: np.ndarray,
    chunk: int,
    workers: int,
) -> np.ndarray:
    """Run the same exact cKDTree k=1 query with configurable parallelism."""

    if points.dtype != np.float64:
        raise TypeError("nearest-neighbour queries require float64 local coordinates")
    workers = validate_query_workers(workers)
    output = np.empty(len(points), dtype=np.float64)
    for start in range(0, len(points), chunk):
        stop = min(start + chunk, len(points))
        distance, _ = tree.query(points[start:stop], k=1, workers=workers)
        output[start:stop] = distance
    return output


def summarize_distances_exact(
    reconstruction: np.ndarray,
    reference: np.ndarray,
    thresholds_m: list[float],
    query_chunk: int,
    threshold_epsilon_m: float,
    query_workers: int = 1,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Reuse the frozen metric core while varying only exact-query workers."""

    query_workers = validate_query_workers(query_workers)
    frozen_query = core.query_distances
    if query_workers == 1:
        result = core.summarize_distances(
            reconstruction,
            reference,
            thresholds_m,
            query_chunk,
            threshold_epsilon_m,
        )
    else:
        def parallel_query(tree: Any, points: np.ndarray, chunk: int) -> np.ndarray:
            return exact_query_distances(tree, points, chunk, query_workers)

        core.query_distances = parallel_query
        try:
            result = core.summarize_distances(
                reconstruction,
                reference,
                thresholds_m,
                query_chunk,
                threshold_epsilon_m,
            )
        finally:
            core.query_distances = frozen_query
    metrics, recon_to_ref, ref_to_recon = result
    metrics["nearest_neighbor_query_workers"] = query_workers
    return metrics, recon_to_ref, ref_to_recon


def identity(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(
            descriptor,
            (
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8"),
        )
    finally:
        os.close(descriptor)


def validate_contract(contract: dict[str, Any], args: argparse.Namespace) -> None:
    if (
        contract.get("schema") != "m3m_gcp_lidar_rendered_surface_contract_v1"
        or contract.get("protocol_id") != PROTOCOL_ID
        or contract.get("source_geometry_protocol_id") != SOURCE_PROTOCOL_ID
    ):
        raise ValueError("not the frozen LiDAR-v1 scientific contract")
    surface = contract["reconstruction_surface"]
    reference = contract["reference_surface"]
    metrics = contract["metrics"]
    lidar = contract["lidar_source"]
    expected = {
        "roi_buffer_m": reference["roi_buffer_m"],
        "normal_minus_ellipsoid_m": lidar["normal_minus_ellipsoid_m"],
        "alpha_min": surface["alpha_min_inclusive"],
        "pixel_stride": surface["pixel_stride"],
        "reconstruction_voxel_m": surface["reconstruction_voxel_m"],
        "reference_voxel_m": reference["reference_voxel_m"],
        "threshold_epsilon_m": surface["threshold_comparison_epsilon_m"],
        "thresholds_m": metrics["thresholds_m"],
    }
    actual = {
        "roi_buffer_m": args.roi_buffer_m,
        "normal_minus_ellipsoid_m": args.normal_minus_ellipsoid_m,
        "alpha_min": args.alpha_min,
        "pixel_stride": args.pixel_stride,
        "reconstruction_voxel_m": args.reconstruction_voxel_m,
        "reference_voxel_m": args.reference_voxel_m,
        "threshold_epsilon_m": args.threshold_epsilon_m,
        "thresholds_m": list(args.thresholds_m),
    }
    changed = {
        key: {"actual": actual[key], "frozen": expected[key]}
        for key in expected
        if actual[key] != expected[key]
    }
    if changed:
        raise ValueError(f"runtime values differ from LiDAR-v1 contract: {changed}")
    scene = next((row for row in contract["scenes"] if row["scene"] == SCENE), None)
    if (
        scene is None
        or int(scene["train_views"]) != 2196
        or int(scene["test_views"]) != 314
    ):
        raise ValueError("LiDAR-v1 contract has no exact 100K scene binding")


def validate_source_bindings(
    contract: dict[str, Any], args: argparse.Namespace
) -> dict[str, dict[str, Any]]:
    geometry = contract["source_geometry_binding"]
    geometry_manifest = args.geometry_release_root / geometry[
        "release_manifest_relative_path"
    ]
    formal_manifest = args.formal_input_root / "NATIVE_QUARTER_INPUT_MANIFEST.json"
    formal_expected = contract["formal_input_binding"]["scene_manifests"][SCENE]
    small_files = (
        (geometry_manifest, geometry["release_manifest_sha256"], "geometry release"),
        (args.gcp_csv, geometry["gcp_points_sha256"], "GCP coordinate table"),
        (
            args.sim3_json,
            geometry["scene_common_sim3_sha256"][SCENE],
            "common Sim(3)",
        ),
        (
            args.split,
            contract["source_data_release"]["split_manifest_file_sha256"],
            "RGB split",
        ),
        (formal_manifest, formal_expected["file_sha256"], "formal input manifest"),
        (
            args.lidar_inventory,
            contract["lidar_source"]["payload_sha256_inventory_file_sha256"],
            "LiDAR inventory",
        ),
    )
    for path, expected_sha, label in small_files:
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise ValueError(f"{label} identity mismatch: {path}")
    formal_payload = read_json(formal_manifest)
    if (
        formal_input_manifest_canonical_sha256(formal_payload)
        != formal_expected["canonical_sha256"]
    ):
        raise ValueError("formal input manifest canonical identity mismatch")

    verify_full_hashes = not args.reference_cache_root.exists()
    laz_files: dict[str, dict[str, Any]] = {}
    for relative, expected in sorted(contract["lidar_source"]["laz_files_exact"].items()):
        path = args.lidar_root / relative
        if not path.is_file() or path.stat().st_size != int(expected["bytes"]):
            raise ValueError(f"LiDAR source size mismatch: {path}")
        if verify_full_hashes and sha256_file(path) != expected["sha256"]:
            raise ValueError(f"LiDAR source SHA mismatch: {path}")
        laz_files[relative] = {
            "path": str(path.resolve()),
            "bytes": int(expected["bytes"]),
            "sha256": str(expected["sha256"]),
            "identity_source": "frozen_contract_exact_and_verified_before_cache_creation",
        }
    return laz_files


def validate_runtime(
    registry_path: Path,
    method_id: str,
    packet_manifest: Path,
    surface_sampling_track: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    registry = read_json(registry_path)
    if (
        registry.get("schema") != RUNTIME_REGISTRY_SCHEMA
        or registry.get("status") != "ACTIVE_FROZEN"
        or registry.get("scene") != SCENE
        or registry.get("canonical_sha256") != canonical_sha256(registry)
    ):
        raise ValueError("success-subset registry identity/status mismatch")
    methods = [row for row in registry["methods"] if row["method_id"] == method_id]
    if len(methods) != 1 or method_id not in registry.get("ready_method_ids", []):
        raise ValueError("method is not a unique promoted success")
    method = methods[0]
    run_root = Path(str(method["run_root"])).resolve()
    expected_packet = (
        lidar_full_train_packet_manifest(run_root)
        if surface_sampling_track == "full_train"
        else lidar_heldout_candidate_packet_manifest(run_root)
    )
    if packet_manifest.resolve() != expected_packet:
        raise ValueError("LiDAR packet is outside the promoted run's fixed packet root")
    return registry, method


def expected_packet_names(
    split_path: Path, surface_sampling_track: str
) -> tuple[str, ...]:
    split = read_json(split_path)
    scene = next(row for row in split["scenes"] if row["scene"] == SCENE)
    track = SURFACE_SAMPLING_TRACKS[surface_sampling_track]
    split_role = str(track["split_role"])
    expected_views = int(track["expected_views"])
    names = tuple(
        sorted(str(name) for name in scene[f"{split_role}_image_names"])
    )
    if len(names) != expected_views or len(set(names)) != expected_views:
        raise ValueError(
            f"split does not contain exactly {expected_views} unique 100K "
            f"{split_role} views"
        )
    return names


def materialize_heldout_candidate_manifest_alias(
    source_path: Path,
    packet: dict[str, Any],
    *,
    surface_sampling_track: str,
    expected_image_names: tuple[str, ...],
) -> tuple[Path, dict[str, Any], dict[str, Any] | None]:
    """Adapt one dedicated-exporter label without modifying formal-v1 code.

    Generic exporters describe an explicitly supplied held-out allowlist as
    ``train`` because the renderer receives it through its train-camera slot.
    Dedicated exporters instead retain the more literal
    ``frozen_evaluation_allowlist`` label.  Both carry the same exact frozen
    image names.  Only the held-out candidate track may materialize a
    byte-bound copy whose sole semantic-field change is that label; the formal
    full-train path remains strict and unchanged.
    """

    source_path = source_path.resolve()
    source_camera_sets = packet.get("camera_sets")
    if surface_sampling_track != "heldout_candidate" or source_camera_sets == "train":
        core.validate_packet_manifest(
            packet, scene=SCENE, expected_image_names=expected_image_names
        )
        return source_path, packet, None
    if source_camera_sets != "frozen_evaluation_allowlist":
        raise ValueError(
            f"unsupported held-out candidate camera_sets: {source_camera_sets!r}"
        )

    normalized = dict(packet)
    normalized["camera_sets"] = "train"
    if "canonical_sha256" in normalized:
        normalized["canonical_sha256"] = canonical_sha256(normalized)
    core.validate_packet_manifest(
        normalized, scene=SCENE, expected_image_names=expected_image_names
    )
    alias_path = source_path.parent / "depth_export_manifest_heldout_train_alias.json"
    if alias_path.exists():
        if read_json(alias_path) != normalized:
            raise ValueError(f"stale held-out candidate manifest alias: {alias_path}")
    else:
        write_json(alias_path, normalized)

    receipt_path = source_path.parent / "heldout_manifest_camera_set_alias_receipt.json"
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if (
            receipt.get("canonical_sha256") != canonical_sha256(receipt)
            or receipt.get("status")
            != "REPRESENTATION_ALIAS_ONLY_EXACT_IMAGE_LIST_UNCHANGED"
            or receipt.get("source_manifest") != identity(source_path)
            or receipt.get("core_compatible_manifest") != identity(alias_path)
        ):
            raise ValueError(f"stale held-out candidate alias receipt: {receipt_path}")
    else:
        receipt = {
            "schema": "m3m_gcp_100k_heldout_manifest_camera_set_alias_v1",
            "protocol_id": HELDOUT_CANDIDATE_PROTOCOL_ID,
            "status": "REPRESENTATION_ALIAS_ONLY_EXACT_IMAGE_LIST_UNCHANGED",
            "created_at": now(),
            "source_manifest": identity(source_path),
            "core_compatible_manifest": identity(alias_path),
            "changed_field": "camera_sets",
            "source_value": source_camera_sets,
            "core_compatible_value": "train",
            "heldout_image_names": list(expected_image_names),
            "rendered_view_count": len(expected_image_names),
            "formal_evaluator_code_modified": False,
        }
        receipt["canonical_sha256"] = canonical_sha256(receipt)
        write_json(receipt_path, receipt)
    return alias_path, normalized, identity(receipt_path)


def load_or_build_reference(
    *,
    cache_root: Path,
    binding: dict[str, Any],
    laz_dir: Path,
    roi: Any,
    normal_minus_ellipsoid_m: float,
    voxel_m: float,
    chunk_points: int,
    origin: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any], str]:
    cache_root = cache_root.resolve()
    points_path = cache_root / "reference_voxel_centres_local_metric.npz"
    audit_path = cache_root / "reference_audit.json"
    manifest_path = cache_root / "reference_cache_manifest.json"
    if cache_root.exists():
        manifest = read_json(manifest_path)
        cached_binding = manifest.get("binding")
        binding_mode = reference_cache_binding_mode(cached_binding, binding)
        if (
            manifest.get("schema") != "m3m_gcp_lidar_100k_reference_cache_v1"
            or manifest.get("status") != "PASS_REFERENCE_CACHE"
            or binding_mode is None
            or manifest.get("canonical_sha256") != canonical_sha256(manifest)
            or manifest.get("points", {}).get("sha256") != sha256_file(points_path)
            or manifest.get("audit", {}).get("sha256") != sha256_file(audit_path)
        ):
            raise ValueError("shared LiDAR reference cache identity mismatch")
        with np.load(points_path, allow_pickle=False) as payload:
            reference = payload["points_local_m"]
            cached_origin = payload["local_origin_utm49n_normal_height_m"]
        if reference.dtype != np.float64 or not np.array_equal(cached_origin, origin):
            raise ValueError("shared LiDAR reference cache numeric frame mismatch")
        return reference, read_json(audit_path), manifest, binding_mode

    cache_root.mkdir(parents=True)
    reference, audit = core.build_reference(
        laz_dir,
        roi,
        normal_minus_ellipsoid_m,
        voxel_m,
        chunk_points,
        origin,
    )
    np.savez_compressed(
        points_path,
        points_local_m=reference,
        local_origin_utm49n_normal_height_m=origin,
    )
    audit["reference_points_file_sha256"] = sha256_file(points_path)
    write_json(audit_path, audit)
    manifest = {
        "schema": "m3m_gcp_lidar_100k_reference_cache_v1",
        "status": "PASS_REFERENCE_CACHE",
        "created_at": now(),
        "binding": binding,
        "reference_point_count": len(reference),
        "points": identity(points_path),
        "audit": identity(audit_path),
    }
    manifest["canonical_sha256"] = canonical_sha256(manifest)
    write_json(manifest_path, manifest)
    return reference, audit, manifest, "EXACT_BINDING"


def reference_cache_binding_mode(
    cached: Any, expected: dict[str, Any]
) -> str | None:
    """Allow reuse when only the enclosing contract file identity changed.

    Every reference-defining field is duplicated explicitly beside ``contract``
    in the cache binding.  Ignoring only that implementation/lifecycle identity
    keeps the cached point bytes reusable across evaluator-only revisions while
    still rejecting any source, ROI, vertical-frame, voxel, or origin change.
    """
    if not isinstance(cached, dict):
        return None
    if cached.get("canonical_sha256") != canonical_sha256(cached) or expected.get(
        "canonical_sha256"
    ) != canonical_sha256(expected):
        return None
    if cached == expected:
        return "EXACT_BINDING"
    if set(cached) != set(expected):
        return None
    lifecycle_keys = {"contract", "canonical_sha256"}
    cached_scientific = {
        key: value for key, value in cached.items() if key not in lifecycle_keys
    }
    expected_scientific = {
        key: value for key, value in expected.items() if key not in lifecycle_keys
    }
    if cached_scientific != expected_scientific:
        return None
    if not isinstance(cached.get("contract"), dict) or not isinstance(
        expected.get("contract"), dict
    ):
        return None
    return "SCIENTIFIC_BINDING_EQUAL_CONTRACT_FILE_IDENTITY_CHANGED"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--benchmark-commit", required=True)
    parser.add_argument("--benchmark-tree", required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--artifact-schema", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--geometry-release-root", type=Path, required=True)
    parser.add_argument("--formal-input-root", type=Path, required=True)
    parser.add_argument("--lidar-inventory", type=Path, required=True)
    parser.add_argument("--lidar-root", type=Path, required=True)
    parser.add_argument("--colmap-model", type=Path, required=True)
    parser.add_argument("--gcp-csv", type=Path, required=True)
    parser.add_argument("--sim3-json", type=Path, required=True)
    parser.add_argument("--packet-manifest", type=Path, required=True)
    parser.add_argument("--reference-cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--method-id", required=True)
    parser.add_argument(
        "--surface-sampling-track",
        choices=tuple(SURFACE_SAMPLING_TRACKS),
        default="full_train",
    )
    parser.add_argument("--roi-buffer-m", type=float, default=8.0)
    parser.add_argument(
        "--normal-minus-ellipsoid-m", type=float, default=23.980600991639484
    )
    parser.add_argument("--alpha-min", type=float, default=0.5)
    parser.add_argument("--pixel-stride", type=int, default=4)
    parser.add_argument("--reconstruction-voxel-m", type=float, default=0.05)
    parser.add_argument("--reference-voxel-m", type=float, default=0.05)
    parser.add_argument("--laz-chunk-points", type=int, default=1_000_000)
    parser.add_argument("--query-chunk-points", type=int, default=250_000)
    parser.add_argument("--query-workers", type=int, default=1)
    parser.add_argument(
        "--thresholds-m", type=float, nargs="+", default=[0.05, 0.10, 0.20]
    )
    parser.add_argument(
        "--threshold-epsilon-m", type=float, default=core.THRESHOLD_EPSILON_M
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for name in (
        "repo",
        "registry",
        "contract",
        "artifact_schema",
        "split",
        "geometry_release_root",
        "formal_input_root",
        "lidar_inventory",
        "lidar_root",
        "colmap_model",
        "gcp_csv",
        "sim3_json",
        "packet_manifest",
        "reference_cache_root",
        "output_root",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(args.output_root)

    contract = read_json(args.contract)
    benchmark_identity = validate_benchmark_checkout(
        benchmark_repo=args.repo,
        expected_commit=args.benchmark_commit,
        expected_tree=args.benchmark_tree,
        entrypoint=Path(__file__).resolve(),
    )
    validate_contract(contract, args)
    laz_files = validate_source_bindings(contract, args)
    track = SURFACE_SAMPLING_TRACKS[args.surface_sampling_track]
    registry, method = validate_runtime(
        args.registry,
        args.method_id,
        args.packet_manifest,
        args.surface_sampling_track,
    )
    names = expected_packet_names(args.split, args.surface_sampling_track)
    packet = read_json(args.packet_manifest)
    effective_packet_manifest, packet, packet_manifest_alias = (
        materialize_heldout_candidate_manifest_alias(
            args.packet_manifest,
            packet,
            surface_sampling_track=args.surface_sampling_track,
            expected_image_names=names,
        )
    )
    sim3 = read_json(args.sim3_json)
    if (
        sim3.get("protocol_id") != SOURCE_PROTOCOL_ID
        or sim3.get("scene") != SCENE
        or sim3.get("method_result_refit_forbidden") is not True
    ):
        raise ValueError("frozen common Sim(3) identity/policy mismatch")

    roi, gcp_rows = core.build_roi(args.gcp_csv, sim3, args.roi_buffer_m)
    origin = core.freeze_local_origin(roi, args.reference_voxel_m)
    laz_dir = args.lidar_root / "lidars/terra_laz_1_4"
    reference_binding = {
        "scene": SCENE,
        "contract": identity(args.contract),
        "geometry_release_manifest": identity(
            args.geometry_release_root / "protocol_release_manifest.json"
        ),
        "formal_input_manifest": identity(
            args.formal_input_root / "NATIVE_QUARTER_INPUT_MANIFEST.json"
        ),
        "lidar_inventory": identity(args.lidar_inventory),
        "gcp_csv": identity(args.gcp_csv),
        "sim3": identity(args.sim3_json),
        "roi_buffer_m": args.roi_buffer_m,
        "normal_minus_ellipsoid_m": args.normal_minus_ellipsoid_m,
        "reference_voxel_m": args.reference_voxel_m,
        "local_origin_utm49n_normal_height_m": origin.tolist(),
        "laz_files": laz_files,
    }
    reference_binding["canonical_sha256"] = canonical_sha256(reference_binding)
    numeric_self_test = core.run_numeric_self_tests(args.reference_voxel_m)
    (
        reference,
        reference_audit,
        reference_manifest,
        reference_cache_binding_mode_value,
    ) = load_or_build_reference(
        cache_root=args.reference_cache_root,
        binding=reference_binding,
        laz_dir=laz_dir,
        roi=roi,
        normal_minus_ellipsoid_m=args.normal_minus_ellipsoid_m,
        voxel_m=args.reference_voxel_m,
        chunk_points=args.laz_chunk_points,
        origin=origin,
    )

    args.output_root.mkdir(parents=True)
    started = time.monotonic()
    reconstruction, surface_audit = core.build_reconstruction(
        Path(str(method["run_root"])).resolve(),
        *core.read_colmap_model(args.colmap_model),
        sim3,
        roi,
        args.alpha_min,
        args.pixel_stride,
        args.reconstruction_voxel_m,
        origin,
        SCENE,
        names,
        effective_packet_manifest,
    )
    surface_path = args.output_root / "surface_voxel_centres_local_metric.npz"
    np.savez_compressed(
        surface_path,
        points_local_m=reconstruction,
        local_origin_utm49n_normal_height_m=origin,
    )
    metrics, recon_to_ref, ref_to_recon = summarize_distances_exact(
        reconstruction,
        reference,
        list(args.thresholds_m),
        args.query_chunk_points,
        args.threshold_epsilon_m,
        args.query_workers,
    )
    distance_path = args.output_root / "nearest_neighbor_distances.npz"
    np.savez_compressed(
        distance_path,
        reconstruction_to_lidar_m=recon_to_ref,
        lidar_to_reconstruction_m=ref_to_recon,
    )
    formal_metrics = {field: metrics[field] for field in METRIC_FIELDS}
    summary = {
        "method_id": args.method_id,
        "method": method["display_name"],
        "input_class": method["input_class"],
        "surface_sampling_track": args.surface_sampling_track,
        "packet_view_count": len(names),
        "status": "COMPLETE_RANKED",
        **formal_metrics,
        "total_seconds": time.monotonic() - started,
        "nearest_neighbor_seconds": metrics["nearest_neighbor_seconds"],
        "nearest_neighbor_query_workers": metrics[
            "nearest_neighbor_query_workers"
        ],
        "peak_rss_gib": core.peak_rss_gib(),
        "oom": 0,
    }
    protocol = {
        "schema": track["protocol_schema"],
        "protocol_id": track["protocol_id"],
        "source_lidar_numeric_protocol_id": PROTOCOL_ID,
        "source_geometry_protocol_id": SOURCE_PROTOCOL_ID,
        "status": track["protocol_status"],
        "created_at": now(),
        "scene": SCENE,
        "method_id": args.method_id,
        "surface_representation": "deterministic_voxel_centres_of_backprojected_alpha_normalized_expected_camera_z",
        "method_specific_registration": "forbidden",
        "icp": "forbidden",
        "lidar_training_access": "forbidden; evaluation only",
        "surface_sampling_track": args.surface_sampling_track,
        "packet_view_split": track["packet_view_split"],
        "packet_view_count": len(names),
        "evaluator_heldout_rgb_pixels_read": False,
        "packet_renderer_heldout_rgb_pixels_used": False,
        "alpha_min": args.alpha_min,
        "pixel_stride": args.pixel_stride,
        "reconstruction_voxel_m": args.reconstruction_voxel_m,
        "reference_voxel_m": args.reference_voxel_m,
        "thresholds_m": list(args.thresholds_m),
        "threshold_comparison_epsilon_m": args.threshold_epsilon_m,
        "nearest_neighbor_query_workers": args.query_workers,
        "roi_buffer_m": args.roi_buffer_m,
        "roi_area_m2": float(roi.area),
        "roi_bounds_utm49n": list(map(float, roi.bounds)),
        "normal_minus_ellipsoid_m": args.normal_minus_ellipsoid_m,
        "local_origin_utm49n_normal_height_m": origin.tolist(),
        "numeric_self_test": numeric_self_test,
        "reference_cache": identity(
            args.reference_cache_root / "reference_cache_manifest.json"
        ),
        "reference_cache_binding_mode": reference_cache_binding_mode_value,
        "inputs": {
            "benchmark_repository": benchmark_identity,
            "registry": identity(args.registry),
            "contract": identity(args.contract),
            "artifact_schema": identity(args.artifact_schema),
            "split": identity(args.split),
            "source_packet_manifest": identity(args.packet_manifest),
            "packet_manifest": identity(effective_packet_manifest),
            "packet_manifest_camera_set_alias": packet_manifest_alias,
            "colmap_cameras": identity(args.colmap_model / "cameras.bin"),
            "colmap_images": identity(args.colmap_model / "images.bin"),
            "evaluator": identity(Path(__file__).resolve()),
        },
        "gcp_rows": gcp_rows,
        "reference_audit_sha256": canonical_sha256(reference_audit),
        "reference_cache_canonical_sha256": reference_manifest[
            "canonical_sha256"
        ],
    }
    protocol["canonical_sha256"] = canonical_sha256(protocol)
    protocol_path = args.output_root / "protocol_manifest.json"
    write_json(protocol_path, protocol)
    result = {
        "schema": track["result_schema"],
        "protocol_id": track["protocol_id"],
        "source_lidar_numeric_protocol_id": PROTOCOL_ID,
        "status": "COMPLETE_RANKED",
        "scene": SCENE,
        "method_id": args.method_id,
        "surface_sampling_track": args.surface_sampling_track,
        "packet_view_count": len(names),
        "model_checkpoint_sha256": method["formal_model_sha256"],
        "source_packet_manifest_sha256": sha256_file(args.packet_manifest),
        "packet_manifest_sha256": sha256_file(effective_packet_manifest),
        "packet_manifest_camera_set_alias": packet_manifest_alias,
        "protocol_manifest": identity(protocol_path),
        "surface": identity(surface_path),
        "distances": identity(distance_path),
        "reference": identity(
            args.reference_cache_root / "reference_voxel_centres_local_metric.npz"
        ),
        "surface_audit": surface_audit,
        "metrics": formal_metrics,
        "summary_row": summary,
    }
    result["canonical_sha256"] = canonical_sha256(result)
    write_json(args.output_root / "metrics.json", result)
    core.write_csv(args.output_root / "lidar_metrics.csv", [summary])
    core.write_csv(
        args.output_root / "view_surface_counts.csv", surface_audit["view_rows"]
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
