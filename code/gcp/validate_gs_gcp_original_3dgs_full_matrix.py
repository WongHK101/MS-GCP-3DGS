#!/usr/bin/env python3
"""Validate the approved original-3DGS v1.3.0 full-matrix execution plan."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from gs_gcp_resolution import RULE_ID, graphdeco_rminus1_dimensions
from validate_gs_gcp_method_registry import validate_registry


SCHEMA = "gs_gcp_v13_original_3dgs_full_matrix_plan_v1"
RELEASE_DIGEST = "513f8999fe4b110f15bcbecad7932895781cee755ee9ccd7a14ff10298546d75"
EXPECTED_SCENES = {
    "gcp_3000_20260602",
    "gcp_5000_20260602",
    "gcp_10000_20260610",
    "gcp_20000_20260602",
    "gcp_50000_20260610",
    "gcp_100000_20260610",
}
EXECUTION_SCENES = EXPECTED_SCENES - {"gcp_3000_20260602"}
SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def validate_plan(
    plan: dict[str, Any],
    registry: dict[str, Any],
    repo_root: Path,
    *,
    scene: str | None = None,
    scene_root: Path | None = None,
    release_root: Path | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    _require(plan.get("schema") == SCHEMA, "unknown full-matrix plan schema", errors)
    _require(plan.get("method_id") == "3dgs_original", "method must be original 3DGS", errors)
    _require(
        plan.get("release", {}).get("payload_root_digest_sha256") == RELEASE_DIGEST,
        "release root digest mismatch",
        errors,
    )

    registry_result = validate_registry(registry, repo_root)
    _require(registry_result["passed"], "method registry validation failed", errors)
    _require(
        registry_result["full_scene_matrix_eligible"] == ["3dgs_original"],
        "only original 3DGS may be full-matrix eligible",
        errors,
    )

    review = plan.get("qualification_review", {})
    _require(review.get("status") == "PASS", "qualification review is not PASS", errors)
    _require(bool(SHA256_RE.fullmatch(str(review.get("review_package_sha256", "")))), "invalid review package SHA", errors)
    evidence = (repo_root / str(review.get("evidence_path", ""))).resolve()
    _require(evidence.is_relative_to(repo_root.resolve()), "review evidence escapes repository", errors)
    _require(evidence.is_file(), "review evidence is missing", errors)
    if evidence.is_file():
        _require(sha256_file(evidence) == review.get("evidence_sha256"), "review evidence SHA mismatch", errors)

    identity = plan.get("frozen_method_identity", {})
    expected_identity = {
        "training_source_commit": "2eee0e26d2d5fd00ec462df47752223952f6bf4e",
        "training_source_tree": "5eee127dfc0942bf83d9fdd72328e03ec0cbf6c4",
        "training_iterations": 30000,
        "seed": 0,
        "formal_model": "point_cloud/iteration_30000/point_cloud.ply",
        "resolution_rule_id": RULE_ID,
        "metric_adapter_commit": "69842bcbcf1d3a159d08256a8cac557261234d36",
        "metric_rasterizer_commit": "c7c8ec385986ea5230dcdd517b8f6cc06db0049d",
        "packet_schema": "ms_gcp_metric_depth_packet_v2",
        "formal_tensor": "alpha_normalized_expected_camera_z",
        "formal_formula": "M1/A",
        "formal_semantics": "camera_z",
        "patch_protocol": "native_packet_pixel_patch_v1",
        "patch_size": 7,
        "patch_radius": 3,
        "aggregation": "robust_multiview_median",
        "control_policy": "require_all",
        "min_valid_observations": 1,
    }
    for key, expected in expected_identity.items():
        _require(identity.get(key) == expected, f"frozen identity mismatch: {key}", errors)
    _require(bool(SHA1_RE.fullmatch(str(identity.get("training_source_commit", "")))), "invalid training commit", errors)
    _require(bool(SHA1_RE.fullmatch(str(identity.get("metric_adapter_commit", "")))), "invalid adapter commit", errors)

    scenes = plan.get("scenes", [])
    scene_ids = [row.get("scene") for row in scenes if isinstance(row, dict)]
    _require(len(scene_ids) == len(set(scene_ids)), "duplicate scene rows", errors)
    _require(set(scene_ids) == EXPECTED_SCENES, "scene set mismatch", errors)
    _require(set(plan.get("execution_order", [])) == EXECUTION_SCENES, "execution-order scene set mismatch", errors)
    _require(len(plan.get("execution_order", [])) == len(EXECUTION_SCENES), "execution order contains duplicates", errors)
    _require(plan.get("execution", {}).get("other_methods_authorized") is False, "other methods must remain unauthorized", errors)

    by_scene = {row["scene"]: row for row in scenes if isinstance(row, dict) and row.get("scene")}
    for scene_id, row in by_scene.items():
        expected_status = "qualified_pass_frozen_reference" if scene_id == "gcp_3000_20260602" else "approved_pending_execution"
        _require(row.get("status") == expected_status, f"{scene_id}: status mismatch", errors)
        for field in (
            "source_manifest_sha256",
            "cameras_bin_sha256",
            "images_bin_sha256",
            "points3D_bin_sha256",
        ):
            _require(bool(SHA256_RE.fullmatch(str(row.get(field, "")))), f"{scene_id}: invalid {field}", errors)
        for field in (
            "training_image_count",
            "formal_observation_count",
            "formal_target_view_count",
            "control_count",
            "checkpoint_count",
        ):
            _require(isinstance(row.get(field), int) and row[field] > 0, f"{scene_id}: invalid {field}", errors)
        _require(row.get("formal_target_view_count", 0) <= row.get("training_image_count", 0), f"{scene_id}: too many target views", errors)
        try:
            expected_dims = graphdeco_rminus1_dimensions(row["original_width"], row["original_height"])
        except Exception as exc:
            errors.append(f"{scene_id}: invalid source dimensions: {exc}")
        else:
            _require(expected_dims == (row.get("loaded_width"), row.get("loaded_height")), f"{scene_id}: loaded dimensions mismatch", errors)

    runtime: dict[str, Any] | None = None
    if scene is not None:
        _require(scene in EXECUTION_SCENES, "runtime scene is not approved for remaining-five execution", errors)
        row = by_scene.get(scene)
        _require(row is not None, "runtime scene has no plan row", errors)
        _require(scene_root is not None and scene_root.is_dir(), "runtime scene root is missing", errors)
        _require(release_root is not None and release_root.is_dir(), "runtime release root is missing", errors)
        if row is not None and scene_root is not None and scene_root.is_dir():
            manifest = scene_root / "SOURCE_MANIFEST.json"
            _require(manifest.is_file(), "runtime source manifest is missing", errors)
            if manifest.is_file():
                _require(sha256_file(manifest) == row["source_manifest_sha256"], "runtime source manifest SHA mismatch", errors)
            image_paths = sorted(path for path in (scene_root / "images").iterdir() if path.is_file())
            _require(len(image_paths) == row["training_image_count"], "runtime image count mismatch", errors)
            for name, field in (
                ("cameras.bin", "cameras_bin_sha256"),
                ("images.bin", "images_bin_sha256"),
                ("points3D.bin", "points3D_bin_sha256"),
            ):
                path = scene_root / "sparse" / "0" / name
                _require(path.is_file(), f"runtime {name} is missing", errors)
                if path.is_file():
                    _require(sha256_file(path) == row[field], f"runtime {name} SHA mismatch", errors)
            runtime = {
                "scene": scene,
                "scene_root": str(scene_root),
                "training_image_count": len(image_paths),
                "source_manifest_sha256": sha256_file(manifest) if manifest.is_file() else None,
            }
        if row is not None and release_root is not None and release_root.is_dir():
            annotation = release_root / f"{scene}_gcp_annotations_pixel_domain_v1_3_0.csv"
            _require(annotation.is_file(), "runtime annotation CSV is missing", errors)
            if annotation.is_file():
                rows = list(csv.DictReader(annotation.open("r", encoding="utf-8-sig", newline="")))
                formal = [r for r in rows if r.get("formal_eligible", "").strip().lower() in {"1", "true", "yes"}]
                target_field = next((key for key in ("target_image_name", "image_name", "raw_image_name") if rows and key in rows[0]), None)
                targets = {r[target_field] for r in formal} if target_field else set()
                _require(len(formal) == row["formal_observation_count"], "runtime formal observation count mismatch", errors)
                _require(len(targets) == row["formal_target_view_count"], "runtime target-view count mismatch", errors)
                if runtime is not None:
                    runtime["formal_observation_count"] = len(formal)
                    runtime["formal_target_view_count"] = len(targets)

    return {
        "schema": "gs_gcp_v13_original_3dgs_full_matrix_validation_v1",
        "passed": not errors,
        "method_id": plan.get("method_id"),
        "scene_count": len(scene_ids),
        "execution_order": plan.get("execution_order", []),
        "runtime": runtime,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--repo_root", type=Path, default=root)
    parser.add_argument("--plan", type=Path, default=root / "configs/gs_gcp_v13_original_3dgs_full_matrix_v1.json")
    parser.add_argument("--registry", type=Path, default=root / "configs/gs_gcp_method_registry_v1.json")
    parser.add_argument("--scene")
    parser.add_argument("--scene_root", type=Path)
    parser.add_argument("--release_root", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = validate_plan(
        json.loads(args.plan.read_text(encoding="utf-8")),
        json.loads(args.registry.read_text(encoding="utf-8")),
        args.repo_root.resolve(),
        scene=args.scene,
        scene_root=args.scene_root.resolve() if args.scene_root else None,
        release_root=args.release_root.resolve() if args.release_root else None,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
