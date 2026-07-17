#!/usr/bin/env python3
"""Validate GS-GCP Stage 0 contracts and training readiness."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from gcp_pixel_domain_v1_2 import verify_payload_integrity
from run_with_resource_probe import validate_contract as validate_resource_contract
from validate_gs_gcp_method_registry import validate_registry


RELEASE_DIGEST = "513f8999fe4b110f15bcbecad7932895781cee755ee9ccd7a14ff10298546d75"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _component(passed: bool, **details: Any) -> dict[str, Any]:
    return {"passed": bool(passed), **details}


def validate_stage0(repo_root: Path, release_root: Path | None, method_id: str | None) -> dict[str, Any]:
    configs = repo_root / "configs"
    registry_path = configs / "gs_gcp_method_registry_v1.json"
    resource_path = configs / "gs_gcp_resource_probe_contract_v1.json"
    resolution_path = configs / "gs_gcp_training_resolution_v1.json"
    review_path = configs / "gs_gcp_v13_release_review_status_v1.json"
    mirror_path = configs / "gs_gcp_v13_data_mirror_v1.json"
    promotion_path = configs / "gs_gcp_repository_promotion_status_v1.json"
    runtime_path = configs / "gs_gcp_autodl740_runtime_status_v1.json"

    registry_data = _load(registry_path)
    registry_result = validate_registry(registry_data)
    components: dict[str, dict[str, Any]] = {
        "method_registry": _component(
            registry_result["passed"],
            path=str(registry_path),
            sha256=sha256_file(registry_path),
            validation=registry_result,
        )
    }

    try:
        resource_data = _load(resource_path)
        validate_resource_contract(resource_data)
        resource_error = None
    except Exception as exc:
        resource_error = f"{type(exc).__name__}: {exc}"
    components["resource_probe_contract"] = _component(
        resource_error is None,
        path=str(resource_path),
        sha256=sha256_file(resource_path),
        error=resource_error,
    )

    resolution = _load(resolution_path)
    resolution_passed = (
        resolution.get("schema") == "gs_gcp_training_resolution_contract_v1"
        and resolution.get("rule_id") == "graphdeco_rminus1_1600_width_cap_v1"
        and resolution.get("reference_method_argument") == -1
        and resolution.get("max_width") == 1600
    )
    components["training_resolution"] = _component(
        resolution_passed,
        path=str(resolution_path),
        sha256=sha256_file(resolution_path),
        rule_id=resolution.get("rule_id"),
    )

    review = _load(review_path)
    review_contract_valid = (
        review.get("schema") == "gs_gcp_release_review_status_v1"
        and review.get("payload_root_digest_sha256") == RELEASE_DIGEST
        and review.get("external_review_status") in {"not_recorded", "PASS", "BLOCKED"}
        and (review.get("external_review_status") == "PASS") == bool(review.get("training_authorized"))
    )
    components["release_review"] = _component(
        review_contract_valid,
        path=str(review_path),
        sha256=sha256_file(review_path),
        external_review_status=review.get("external_review_status"),
        training_authorized=bool(review.get("training_authorized")),
        blocking_reason=review.get("blocking_reason"),
    )

    mirror = _load(mirror_path)
    expected_scenes = {
        "gcp_3000_20260602",
        "gcp_5000_20260602",
        "gcp_10000_20260610",
        "gcp_20000_20260602",
        "gcp_50000_20260610",
        "gcp_100000_20260610",
    }
    mirror_contract_valid = (
        mirror.get("schema") == "gs_gcp_v13_data_mirror_status_v1"
        and mirror.get("release_root_digest_sha256") == RELEASE_DIGEST
        and set(mirror.get("scene_source_manifest_sha256", {})) == expected_scenes
        and mirror.get("target_overwrite_allowed") is False
        and mirror.get("atomic_publish_required") is True
    )
    components["data_mirror"] = _component(
        mirror_contract_valid,
        path=str(mirror_path),
        sha256=sha256_file(mirror_path),
        verification_status=mirror.get("verification_status"),
        target_root=mirror.get("target_root"),
    )

    promotion = _load(promotion_path)
    promotion_contract_valid = (
        promotion.get("schema") == "gs_gcp_repository_promotion_status_v1"
        and promotion.get("umgs_training_code_included") is False
    )
    components["repository_promotion"] = _component(
        promotion_contract_valid,
        path=str(promotion_path),
        sha256=sha256_file(promotion_path),
        promotion_status=promotion.get("promotion_status"),
        publication_blocking=promotion.get("publication_blocking"),
    )

    runtime = _load(runtime_path)
    host_tool = resource_data.get("host_tool", {}) if resource_error is None else {}
    runtime_contract_valid = (
        runtime.get("schema") == "gs_gcp_autodl_runtime_status_v1"
        and runtime.get("server") == "AutoDL-740"
        and runtime.get("resource_probe_tool", {}).get("status") == "verified_isolated_install"
        and runtime.get("resource_probe_tool", {}).get("binary_sha256") == host_tool.get("binary_sha256")
        and runtime.get("dataset_mirror_status") == "verified_complete_read_only"
    )
    components["autodl_runtime"] = _component(
        runtime_contract_valid,
        path=str(runtime_path),
        sha256=sha256_file(runtime_path),
        gpu_count=runtime.get("gpu_count"),
        orchestrator_deployment_status=runtime.get("orchestrator_deployment_status"),
        original_3dgs_environment_status=runtime.get("original_3dgs_environment_status"),
    )

    if release_root is None:
        components["release_integrity"] = _component(False, status="not_checked")
    else:
        manifest = release_root / "v1_3_0_release_file_manifest.json"
        root_record = release_root / "v1_3_0_release_root_digest.json"
        try:
            integrity = verify_payload_integrity(release_root, manifest, root_record)
            integrity_passed = bool(integrity["passed"])
            integrity_error = None
        except Exception as exc:
            integrity = None
            integrity_passed = False
            integrity_error = f"{type(exc).__name__}: {exc}"
        components["release_integrity"] = _component(
            integrity_passed,
            release_root=str(release_root),
            result=integrity,
            error=integrity_error,
        )

    method_gate = True
    method_details: dict[str, Any] = {"method_id": method_id}
    if method_id:
        matched = [method for method in registry_data["methods"] if method["method_id"] == method_id]
        method_gate = len(matched) == 1 and bool(matched[0].get("three_k_qualification_allowed"))
        method_details["matched_count"] = len(matched)
        method_details["three_k_qualification_allowed"] = bool(matched and matched[0].get("three_k_qualification_allowed"))
    components["method_qualification"] = _component(method_gate, **method_details)

    contract_components = (
        "method_registry",
        "resource_probe_contract",
        "training_resolution",
        "release_review",
        "data_mirror",
        "repository_promotion",
        "autodl_runtime",
    )
    contracts_valid = all(components[name]["passed"] for name in contract_components)
    blockers: list[str] = []
    if not review.get("training_authorized"):
        blockers.append("v1.3.0_external_release_review_pass_not_recorded")
    if mirror.get("verification_status") != "verified_complete_read_only":
        blockers.append("autodl_740_v1.3_data_mirror_not_verified_complete")
    if not components["release_integrity"]["passed"]:
        blockers.append("v1.3.0_release_integrity_not_verified_at_runtime")
    if runtime.get("orchestrator_deployment_status") != "deployed_clean_fixed_commit":
        blockers.append("autodl_740_orchestrator_not_deployed_at_fixed_commit")
    if runtime.get("original_3dgs_environment_status") != "verified_frozen_environment":
        blockers.append("autodl_740_original_3dgs_environment_not_verified")
    if not method_gate:
        blockers.append(f"method_not_pre_registered_for_3k_qualification:{method_id}")
    if not contracts_valid:
        blockers.append("stage0_contract_validation_failed")

    training_ready = not blockers
    return {
        "schema": "gs_gcp_stage0_readiness_report_v1",
        "contracts_valid": contracts_valid,
        "training_ready": training_ready,
        "publication_repository_ready": promotion.get("promotion_status") == "complete",
        "components": components,
        "blockers": blockers,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--release_root", type=Path)
    parser.add_argument("--method_id")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--require_training_ready", action="store_true")
    args = parser.parse_args()
    result = validate_stage0(args.repo_root.resolve(), args.release_root.resolve() if args.release_root else None, args.method_id)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not result["contracts_valid"]:
        return 1
    if args.require_training_ready and not result["training_ready"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
