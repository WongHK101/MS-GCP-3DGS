#!/usr/bin/env python3
"""Validate the native-quarter nine-method candidate registry."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


SHA1 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_METHODS = {
    "3dgs_original",
    "2dgs",
    "pgsr",
    "rade_gs",
    "gof",
    "qgs",
    "citygaussian_v2",
    "citygs_x",
    "metrogs",
}
EXTERNAL_PRIOR_METHODS = {"citygs_x", "metrogs"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_registry(value: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(value.get("schema") == "m3m_gcp_native_quarter_method_registry_v1", "unknown schema")
    require(value.get("protocol_id") == "m3m_gcp_native_quarter_geometry_v1", "protocol mismatch")
    require(value.get("method_count") == 9, "method_count must be 9")
    require(
        bool(SHA256.fullmatch(str(value.get("source_data_release", {}).get("release_root_digest_sha256", "")))),
        "source data release digest is invalid",
    )
    methods = value.get("methods", [])
    ids = [str(method.get("method_id", "")) for method in methods]
    require(len(ids) == len(set(ids)), "duplicate method ids")
    require(set(ids) == EXPECTED_METHODS, f"method set mismatch: {sorted(set(ids) ^ EXPECTED_METHODS)}")
    require(value.get("method_ids") == ids, "method_ids order differs from methods")

    for method in methods:
        method_id = str(method.get("method_id", "<missing>"))
        prefix = method_id + ": "
        source = method.get("source", {})
        require(str(source.get("official_repository", "")).startswith("https://github.com/"), prefix + "official repository missing")
        require(bool(SHA1.fullmatch(str(source.get("commit", "")))), prefix + "commit is not full SHA-1")
        require(bool(SHA1.fullmatch(str(source.get("tree", "")))), prefix + "tree is not full SHA-1")
        require(str(method.get("publication", {}).get("record", "")).startswith("https://"), prefix + "publication record missing")
        input_class = method.get("input_class")
        priors = method.get("external_priors", [])
        if method_id in EXTERNAL_PRIOR_METHODS:
            require(input_class == "rgb_colmap_external_geometry_prior", prefix + "external-prior input class missing")
            require(isinstance(priors, list) and bool(priors), prefix + "external prior inventory missing")
            require(
                all(str(prior.get("freeze_status", "")).startswith("PENDING_") for prior in priors),
                prefix + "current external priors must remain explicitly pending",
            )
        else:
            require(input_class == "rgb_colmap_only", prefix + "unexpected input class")
            require(priors == [], prefix + "RGB+COLMAP method must not carry undeclared priors")
        require(method.get("three_k_training_allowed") is False, prefix + "training must remain locked")
        require(method.get("full_scene_matrix_eligible") is False, prefix + "full matrix must remain locked")
        require(method.get("three_k_qualification_status") == "NOT_RUN", prefix + "qualification status mismatch")

    qgs = next((method for method in methods if method.get("method_id") == "qgs"), {})
    require(qgs.get("source", {}).get("official_repository") == "https://github.com/will-zzy/QGS", "QGS official repository not recorded")
    require(qgs.get("source", {}).get("license_status") == "present_at_frozen_commit", "QGS license evidence missing")
    require(qgs.get("source", {}).get("license_git_blob") == "c869e695fa63bfde6f887d63a24a2a71f03480ac", "QGS license blob mismatch")

    three_dgs = next((method for method in methods if method.get("method_id") == "3dgs_original"), {})
    adapter = three_dgs.get("common_adapter", {})
    report_rel = adapter.get("report")
    require(adapter.get("status") == "CPU_OPERATOR_PREFLIGHT_PASS_RENDERER_INTEGRATION_PENDING", "3DGS adapter status mismatch")
    require(isinstance(report_rel, str) and bool(report_rel), "3DGS CPU preflight report path missing")
    require(bool(SHA256.fullmatch(str(adapter.get("report_sha256", "")))), "3DGS CPU preflight SHA invalid")
    if isinstance(report_rel, str) and report_rel:
        report_path = (repo_root / report_rel).resolve()
        require(report_path.is_relative_to(repo_root.resolve()), "3DGS CPU preflight escapes repo")
        require(report_path.is_file(), "3DGS CPU preflight report missing")
        if report_path.is_file():
            require(file_sha256(report_path) == adapter.get("report_sha256"), "3DGS CPU preflight SHA mismatch")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            require(report.get("status") == "PASS", "3DGS CPU preflight did not pass")
    smoke_rel = adapter.get("end_to_end_cpu_smoke")
    smoke_sha = str(adapter.get("end_to_end_cpu_smoke_sha256", ""))
    require(isinstance(smoke_rel, str) and bool(smoke_rel), "3DGS evaluator smoke path missing")
    require(bool(SHA256.fullmatch(smoke_sha)), "3DGS evaluator smoke SHA invalid")
    if isinstance(smoke_rel, str) and smoke_rel:
        smoke_path = (repo_root / smoke_rel).resolve()
        require(smoke_path.is_relative_to(repo_root.resolve()), "3DGS evaluator smoke escapes repo")
        require(smoke_path.is_file(), "3DGS evaluator smoke report missing")
        if smoke_path.is_file():
            require(file_sha256(smoke_path) == smoke_sha, "3DGS evaluator smoke SHA mismatch")
            smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
            require(smoke.get("status") == "PASS", "3DGS evaluator smoke did not pass")
            require(
                smoke.get("assertions", {}).get("method_specific_sim3_fitted") is False,
                "3DGS evaluator smoke fitted a method-specific Sim3",
            )

    city = next((method for method in methods if method.get("method_id") == "citygs_x"), {})
    require("redistribution_blocked" in str(city.get("source", {}).get("license_status", "")), "CityGS-X redistribution risk missing")
    metro = next((method for method in methods if method.get("method_id") == "metrogs"), {})
    prior_names = {str(prior.get("name")) for prior in metro.get("external_priors", [])}
    require(prior_names == {"pointmap dense initialization", "MoGe-2"}, "MetroGS prior inventory incomplete")
    require(value.get("global_training_allowed") is False, "global training lock missing")
    return {
        "schema": "m3m_gcp_native_quarter_method_registry_validation_v1",
        "passed": not errors,
        "method_count": len(methods),
        "method_ids": ids,
        "external_prior_methods": sorted(EXTERNAL_PRIOR_METHODS),
        "training_allowed_methods": sorted(
            method["method_id"] for method in methods if method.get("three_k_training_allowed") is True
        ),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--repo_root", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    repo_root = (args.repo_root or args.registry.resolve().parents[1]).resolve()
    result = validate_registry(json.loads(args.registry.read_text(encoding="utf-8")), repo_root)
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
