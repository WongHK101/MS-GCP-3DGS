#!/usr/bin/env python3
"""Validate the frozen GS-GCP method admission registry."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ROLES = {"formal_core", "scalability_extension", "conditional"}
FROZEN_SOURCE_STATES = {
    "pre_registered_for_3k_qualification",
    "qualified_for_full_scene_matrix",
    "frozen_source_pending_recipe",
    "frozen_source_pending_license_review",
}
BLOCKED_SOURCE_STATES = {"blocked_no_official_public_implementation"}


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_registry(data: dict[str, Any], repo_root: Path | None = None) -> dict[str, Any]:
    errors: list[str] = []
    _require(data.get("schema") == "gs_gcp_method_registry_v1", "unknown registry schema", errors)
    release = data.get("release", {})
    _require(
        bool(SHA256_RE.fullmatch(str(release.get("payload_root_digest_sha256", "")))),
        "release root digest must be lowercase SHA-256",
        errors,
    )
    methods = data.get("methods")
    _require(isinstance(methods, list) and bool(methods), "methods must be a non-empty list", errors)
    methods = methods if isinstance(methods, list) else []

    ids = [str(method.get("method_id", "")) for method in methods]
    duplicate_ids = sorted(method_id for method_id, count in Counter(ids).items() if count > 1)
    _require(not duplicate_ids, f"duplicate method ids: {duplicate_ids}", errors)

    role_counts = Counter(str(method.get("role", "")) for method in methods)
    _require(set(role_counts).issubset(ROLES), f"unknown roles: {sorted(set(role_counts) - ROLES)}", errors)
    _require(role_counts == Counter(data.get("role_counts", {})), "declared role counts do not match methods", errors)

    for method in methods:
        method_id = str(method.get("method_id", "<missing>"))
        publication = method.get("publication", {})
        source = method.get("source", {})
        status = method.get("source_status")
        full_scene_eligible = method.get("full_scene_matrix_eligible") is True
        three_k_status = method.get("three_k_qualification_status", "not_run")
        external_review_status = method.get("external_review_status", "not_recorded")
        prefix = f"{method_id}: "
        _require(publication.get("gate") == "PASS", prefix + "publication gate is not PASS", errors)
        _require(bool(publication.get("venue")), prefix + "publication venue is missing", errors)
        _require(str(publication.get("record", "")).startswith("https://"), prefix + "publication record is missing", errors)
        _require(status in FROZEN_SOURCE_STATES | BLOCKED_SOURCE_STATES, prefix + "unknown source status", errors)

        if full_scene_eligible:
            evidence = method.get("qualification_evidence", {})
            _require(status == "qualified_for_full_scene_matrix", prefix + "full matrix source status mismatch", errors)
            _require(three_k_status == "PASS", prefix + "full matrix eligibility requires 3K PASS", errors)
            _require(external_review_status == "PASS", prefix + "full matrix eligibility requires external PASS", errors)
            _require(method.get("three_k_qualification_allowed") is True, prefix + "full matrix method must retain 3K qualification admission", errors)
            _require(bool(SHA256_RE.fullmatch(str(evidence.get("review_package_sha256", "")))), prefix + "review package SHA is invalid", errors)
            _require(bool(SHA256_RE.fullmatch(str(evidence.get("external_review_evidence_sha256", "")))), prefix + "external review evidence SHA is invalid", errors)
            _require(bool(evidence.get("review_package")), prefix + "review package identity is missing", errors)
            evidence_rel = evidence.get("external_review_evidence_path")
            _require(isinstance(evidence_rel, str) and bool(evidence_rel), prefix + "external review evidence path is missing", errors)
            if repo_root is not None and isinstance(evidence_rel, str) and evidence_rel:
                resolved_root = repo_root.resolve()
                evidence_path = (resolved_root / evidence_rel).resolve()
                evidence_in_root = evidence_path.is_relative_to(resolved_root)
                _require(evidence_in_root, prefix + "external review evidence escapes repository", errors)
                _require(evidence_path.is_file(), prefix + "external review evidence file is missing", errors)
                if evidence_in_root and evidence_path.is_file():
                    _require(
                        _sha256_file(evidence_path) == evidence.get("external_review_evidence_sha256"),
                        prefix + "external review evidence SHA mismatch",
                        errors,
                    )
        else:
            _require(status != "qualified_for_full_scene_matrix", prefix + "qualified source must be full matrix eligible", errors)
            _require(three_k_status != "PASS", prefix + "3K PASS must enable full matrix eligibility", errors)
            _require(external_review_status != "PASS", prefix + "external PASS must enable full matrix eligibility", errors)

        if status in FROZEN_SOURCE_STATES:
            _require(str(source.get("official_repository", "")).startswith("https://"), prefix + "official repository missing", errors)
            _require(bool(SHA1_RE.fullmatch(str(source.get("commit", "")))), prefix + "commit is not a full SHA-1", errors)
            _require(bool(SHA1_RE.fullmatch(str(source.get("tree", "")))), prefix + "tree is not a full SHA-1", errors)
            for path, commit in source.get("gitlinks", {}).items():
                _require(bool(path), prefix + "empty gitlink path", errors)
                _require(bool(SHA1_RE.fullmatch(str(commit))), prefix + f"invalid gitlink commit for {path}", errors)
        else:
            _require(source.get("official_repository") is None, prefix + "blocked implementation must not invent repository", errors)
            _require(source.get("commit") is None and source.get("tree") is None, prefix + "blocked implementation must not invent source hashes", errors)
            _require(method.get("three_k_qualification_allowed") is False, prefix + "blocked method cannot qualify", errors)

        if method.get("three_k_qualification_allowed"):
            _require(bool(method.get("recipe")), prefix + "qualification requires a frozen recipe", errors)
            _require(source.get("license_status") == "present_at_frozen_commit", prefix + "qualification requires license evidence", errors)
            _require(status in {"pre_registered_for_3k_qualification", "qualified_for_full_scene_matrix"}, prefix + "qualification state mismatch", errors)
        elif status == "pre_registered_for_3k_qualification":
            errors.append(prefix + "pre-registered method must be qualification allowed")

    expected_core = {
        "3dgs_original",
        "2dgs",
        "pgsr",
        "rade_gs",
        "gof",
        "citygaussian_v2",
    }
    actual_core = {method["method_id"] for method in methods if method.get("role") == "formal_core"}
    _require(actual_core == expected_core, "formal core set differs from frozen Stage 0 set", errors)
    _require(
        next((method for method in methods if method.get("method_id") == "qgs"), {}).get("source_status")
        == "blocked_no_official_public_implementation",
        "QGS must remain blocked until an official public implementation is recoverable",
        errors,
    )

    return {
        "schema": "gs_gcp_method_registry_validation_v1",
        "passed": not errors,
        "method_count": len(methods),
        "role_counts": dict(sorted(role_counts.items())),
        "qualification_allowed": sorted(
            method["method_id"] for method in methods if method.get("three_k_qualification_allowed")
        ),
        "full_scene_matrix_eligible": sorted(
            method["method_id"] for method in methods if method.get("full_scene_matrix_eligible")
        ),
        "blocked_methods": sorted(
            method["method_id"] for method in methods if method.get("source_status") in BLOCKED_SOURCE_STATES
        ),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--repo_root", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve() if args.repo_root else args.registry.resolve().parents[1]
    result = validate_registry(json.loads(args.registry.read_text(encoding="utf-8")), repo_root)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
