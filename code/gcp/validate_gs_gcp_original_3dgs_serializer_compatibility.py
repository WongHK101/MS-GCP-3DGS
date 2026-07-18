#!/usr/bin/env python3
"""Validate the approved original-3DGS memory-safe PLY serializer patch."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


SCHEMA = "gs_gcp_original_3dgs_serializer_compatibility_v1"
UPSTREAM_COMMIT = "2eee0e26d2d5fd00ec462df47752223952f6bf4e"
UPSTREAM_TREE = "5eee127dfc0942bf83d9fdd72328e03ec0cbf6c4"
PATCH_COMMIT = "db8deebca67e8d5e1507e67c98de603eca0dfd85"
PATCH_TREE = "bcb9df570c43755ed4cd43b51bafcc3cf180a466"
SOURCE_SHA256 = "d6f148e7b3f08d55925125bc94c90fd240ce18fcccb5d70f878bab6a7d416970"
DIFF_SHA256 = "b6ac837fbd1b75d8777f719a193782609c8d917975ff79ab64407479c97b49b3"
FIXTURE_SHA256 = "88df8eb33ccf22d37381ca7200bac8c7eb8225616f319eaac5fdfd8e538024d0"
ALLOWED_FILES = {
    "scene/gaussian_model.py",
    "tests/compare_ply_parity.py",
    "tests/save_ply_rss_probe.py",
    "tests/test_save_ply_memory_safe.py",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(path: Path, *args: str, text: bool = True):
    output = subprocess.check_output(
        ["git", "-c", f"safe.directory={path.as_posix()}", "-C", str(path), *args],
        text=text,
    )
    return output.strip() if text else output


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def validate_serializer_compatibility(
    config: dict[str, Any],
    *,
    repo_root: Path,
    upstream_source: Path | None = None,
    patched_source: Path | None = None,
    parity_evidence_root: Path | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    _require(config.get("schema") == SCHEMA, "unknown serializer compatibility schema", errors)
    _require(
        config.get("status") == "approved_verified_for_remaining_original_3dgs_matrix",
        "serializer compatibility is not approved and verified",
        errors,
    )
    upstream = config.get("upstream", {})
    _require(upstream.get("commit") == UPSTREAM_COMMIT, "upstream commit mismatch", errors)
    _require(upstream.get("tree") == UPSTREAM_TREE, "upstream tree mismatch", errors)
    patch = config.get("patch", {})
    _require(patch.get("commit") == PATCH_COMMIT, "serializer patch commit mismatch", errors)
    _require(patch.get("tree") == PATCH_TREE, "serializer patch tree mismatch", errors)
    _require(patch.get("parent_commit") == UPSTREAM_COMMIT, "serializer patch parent mismatch", errors)
    _require(patch.get("diff_sha256") == DIFF_SHA256, "serializer diff SHA mismatch", errors)
    _require(patch.get("patched_source_sha256") == SOURCE_SHA256, "patched source SHA mismatch", errors)
    _require(set(patch.get("allowed_changed_files", [])) == ALLOWED_FILES, "changed-file allowlist mismatch", errors)
    _require(patch.get("training_math_changed") is False, "training math must remain unchanged", errors)
    _require(patch.get("training_recipe_changed") is False, "training recipe must remain unchanged", errors)

    approval = config.get("approval", {})
    evidence_path = (repo_root / str(approval.get("evidence_path", ""))).resolve()
    _require(evidence_path.is_relative_to(repo_root.resolve()), "approval evidence escapes repository", errors)
    _require(evidence_path.is_file(), "approval evidence is missing", errors)
    if evidence_path.is_file():
        _require(sha256_file(evidence_path) == approval.get("evidence_sha256"), "approval evidence SHA mismatch", errors)
    _require(
        bool(re.fullmatch(r"[0-9a-f]{64}", str(approval.get("blocker_review_package_sha256", "")))),
        "blocker review package SHA is invalid",
        errors,
    )

    synthetic = config.get("synthetic_parity", {})
    for field in (
        "vertex_count_equal",
        "format_and_endianness_equal",
        "property_schema_equal",
        "float32_fields_bitwise_equal",
        "loaded_gaussian_tensors_bitwise_equal",
        "noncontiguous_tensor_input_covered",
        "atomic_failure_cleanup_passed",
    ):
        _require(synthetic.get(field) is True, f"synthetic parity gate failed: {field}", errors)
    _require(synthetic.get("status") == "PASS", "synthetic parity status is not PASS", errors)

    real = config.get("real_5k_parity", {})
    _require(real.get("status") == "PASS", "real 5K parity status is not PASS", errors)
    _require(real.get("fixture_sha256") == FIXTURE_SHA256, "5K fixture SHA mismatch", errors)
    for field in ("patched_output_byte_identical", "fields_bitwise_equal", "loaded_gaussian_tensors_bitwise_equal"):
        _require(real.get(field) is True, f"real 5K parity gate failed: {field}", errors)
    official_rss = real.get("official_maximum_rss_kib")
    patched_rss = real.get("patched_maximum_rss_kib")
    _require(isinstance(official_rss, int) and isinstance(patched_rss, int), "RSS evidence is malformed", errors)
    if isinstance(official_rss, int) and isinstance(patched_rss, int):
        _require(0 < patched_rss < official_rss, "patched writer does not reduce maximum RSS", errors)

    invariants = config.get("formal_invariants", {})
    expected_invariants = {
        "release_root_digest": "513f8999fe4b110f15bcbecad7932895781cee755ee9ccd7a14ff10298546d75",
        "seed": 0,
        "iterations": 30000,
        "resolution_rule": "graphdeco_rminus1_1600_width_cap_v1",
        "metric_packet_schema": "ms_gcp_metric_depth_packet_v2",
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
    for field, expected in expected_invariants.items():
        _require(invariants.get(field) == expected, f"formal invariant mismatch: {field}", errors)

    if upstream_source is not None:
        try:
            _require(_git(upstream_source, "rev-parse", "HEAD") == UPSTREAM_COMMIT, "upstream source HEAD mismatch", errors)
            _require(_git(upstream_source, "rev-parse", "HEAD^{tree}") == UPSTREAM_TREE, "upstream source tree mismatch", errors)
            _require(not _git(upstream_source, "status", "--porcelain=v1"), "upstream source is dirty", errors)
        except (OSError, subprocess.CalledProcessError) as exc:
            errors.append(f"upstream source verification failed: {exc}")

    if patched_source is not None:
        try:
            _require(_git(patched_source, "rev-parse", "HEAD") == PATCH_COMMIT, "patched source HEAD mismatch", errors)
            _require(_git(patched_source, "rev-parse", "HEAD^{tree}") == PATCH_TREE, "patched source tree mismatch", errors)
            _require(_git(patched_source, "rev-parse", "HEAD^") == UPSTREAM_COMMIT, "patched source parent mismatch", errors)
            _require(not _git(patched_source, "status", "--porcelain=v1"), "patched source is dirty", errors)
            changed = set(_git(patched_source, "diff", "--name-only", f"{UPSTREAM_COMMIT}..{PATCH_COMMIT}").splitlines())
            _require(changed == ALLOWED_FILES, "runtime changed-file set mismatch", errors)
            diff_bytes = _git(
                patched_source,
                "diff",
                "--binary",
                f"{UPSTREAM_COMMIT}..{PATCH_COMMIT}",
                text=False,
            )
            _require(hashlib.sha256(diff_bytes).hexdigest() == DIFF_SHA256, "runtime diff SHA mismatch", errors)
            _require(sha256_file(patched_source / "scene" / "gaussian_model.py") == SOURCE_SHA256, "runtime patched source SHA mismatch", errors)
            _require(
                _git(patched_source / "submodules" / "diff-gaussian-rasterization", "rev-parse", "HEAD")
                == upstream.get("diff_gaussian_rasterization_commit"),
                "runtime rasterizer commit mismatch",
                errors,
            )
            _require(
                _git(patched_source / "submodules" / "simple-knn", "rev-parse", "HEAD")
                == upstream.get("simple_knn_commit"),
                "runtime simple-knn commit mismatch",
                errors,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            errors.append(f"patched source verification failed: {exc}")

    if parity_evidence_root is not None:
        summary_path = parity_evidence_root / str(real.get("summary_file", ""))
        _require(summary_path.is_file(), "real parity summary is missing", errors)
        if summary_path.is_file():
            _require(sha256_file(summary_path) == real.get("summary_sha256"), "real parity summary SHA mismatch", errors)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            _require(summary.get("passed") is True, "real parity summary is not PASS", errors)
            _require(summary.get("fixture_sha256") == FIXTURE_SHA256, "runtime parity fixture SHA mismatch", errors)
            _require(summary.get("official_maximum_rss_kib") == official_rss, "official RSS evidence mismatch", errors)
            _require(summary.get("patched_maximum_rss_kib") == patched_rss, "patched RSS evidence mismatch", errors)
            for comparison in ("fixture_vs_official", "fixture_vs_patched", "official_vs_patched"):
                _require(summary.get(comparison, {}).get("passed") is True, f"runtime parity comparison failed: {comparison}", errors)
                _require(summary.get(comparison, {}).get("byte_identical") is True, f"runtime PLY bytes differ: {comparison}", errors)

    return {
        "schema": "gs_gcp_original_3dgs_serializer_compatibility_validation_v1",
        "passed": not errors,
        "upstream_commit": UPSTREAM_COMMIT,
        "patch_commit": PATCH_COMMIT,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--config", type=Path, default=root / "configs/gs_gcp_original_3dgs_serializer_compatibility_v1.json")
    parser.add_argument("--upstream_source", type=Path)
    parser.add_argument("--patched_source", type=Path)
    parser.add_argument("--parity_evidence_root", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = validate_serializer_compatibility(
        json.loads(args.config.read_text(encoding="utf-8")),
        repo_root=root,
        upstream_source=args.upstream_source,
        patched_source=args.patched_source,
        parity_evidence_root=args.parity_evidence_root,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
