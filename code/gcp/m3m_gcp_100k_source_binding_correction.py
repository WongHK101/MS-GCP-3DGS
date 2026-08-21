#!/usr/bin/env python3
"""Validate the isolated 3DGS Linux source-identity metadata correction."""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

from m3m_gcp_lidar_artifacts import canonical_sha256, sha256_file


SCENE = "gcp_100000_20260610"
METHOD_ORDER = [
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
]
RECEIPT_SCHEMA = "m3m_gcp_100k_source_binding_correction_v1"
RECEIPT_STATUS = "SEALED_LINUX_IDENTITY_METADATA_CORRECTION"
WINDOWS_HEADER_SHA = "7fdf17df5880f2819551e70a162937abff526b7e6b0337ccb8d6fe184f18c3f2"
LINUX_HEADER_SHA = "c4f5f2df458e75290bdaff7510b87395d5b8ca47ef07b51f47c9b5cb7e580629"
REPOSITORY_ROLES = {
    "recipe_manifest_v2",
    "3dgs_recipe_v2",
    "recipe_manifest_v3",
    "3dgs_recipe_v3",
    "linux_identity_proof",
    "renderer_patch",
    "rasterizer_patch",
    "trigger_audit_local",
}


def _bound_path(repo: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else repo / path


def _role_map(rows: object) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list):
        raise RuntimeError("source-binding correction artifact inventory is not a list")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("role"), str):
            raise RuntimeError("source-binding correction artifact row is invalid")
        role = str(row["role"])
        if role in result:
            raise RuntimeError(f"duplicate source-binding correction role: {role}")
        result[role] = row
    if set(result) != REPOSITORY_ROLES:
        raise RuntimeError("source-binding correction artifact role mismatch")
    return result


def _validate_file(path: Path, row: dict[str, Any], label: str) -> None:
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != int(row.get("bytes", -1))
        or sha256_file(path) != row.get("sha256")
    ):
        raise RuntimeError(f"{label} changed or disappeared: {path}")
    expected_canonical = row.get("canonical_sha256")
    if expected_canonical is not None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("canonical_sha256") != expected_canonical
            or canonical_sha256(payload) != expected_canonical
        ):
            raise RuntimeError(f"{label} canonical SHA mismatch")


def _git_value(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, encoding="utf-8"
    ).strip()


def _git_porcelain_status(repo: Path) -> str:
    payload = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain=v1", "--untracked-files=no"]
    )
    return payload.rstrip(b"\r\n").decode("utf-8")


def _linux_files(proof: dict[str, Any]) -> dict[str, str]:
    return {
        **{
            relative: digest
            for relative, digest in proof.get("renderer_files", {}).items()
        },
        **{
            f"submodules/diff-gaussian-rasterization/{relative}": digest
            for relative, digest in proof.get("rasterizer_files", {}).items()
        },
    }


def _expected_recipe_correction(
    *, proof_path: Path, proof: dict[str, Any], renderer_patch: Path,
    rasterizer_patch: Path,
) -> dict[str, Any]:
    proof_relative = (
        "docs/protocol_evidence/"
        "3dgs_native_quarter_adapter_linux_identity_proof_v1.json"
    )
    return {
        "type": "LINUX_IDENTITY_METADATA_CORRECTION_ONLY",
        "phase": "packet",
        "source_modified": False,
        "child_started": False,
        "attempt_consumed": False,
        "dual_hash_tolerance": False,
        "superseded_windows_header_sha256": WINDOWS_HEADER_SHA,
        "formal_linux_header_sha256": LINUX_HEADER_SHA,
        "linux_identity_proof": {
            "path": proof_relative,
            "sha256": sha256_file(proof_path),
            "status_required": "PASS",
            "patched_file_count": 8,
        },
        "frozen_patches": [
            {
                "path": (
                    "patches/3dgs_original/"
                    "native_quarter_raw_moment_renderer_2eee0e26_v1.patch"
                ),
                "sha256": sha256_file(renderer_patch),
            },
            {
                "path": (
                    "patches/3dgs_original/"
                    "native_quarter_raw_moment_rasterizer_59f5f77_v1.patch"
                ),
                "sha256": sha256_file(rasterizer_patch),
            },
        ],
    }


def _validate_live_sources(
    *, repo: Path, manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest_row in manifest.get("recipes", []):
        recipe_path = _bound_path(repo, manifest_row.get("path", "")).resolve()
        if not recipe_path.is_file() or sha256_file(recipe_path) != manifest_row.get("sha256"):
            raise RuntimeError("live source preflight recipe identity mismatch")
        recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
        if recipe.get("canonical_sha256") != canonical_sha256(recipe):
            raise RuntimeError("live source preflight recipe canonical mismatch")
        for phase, binding in recipe.get("source_bindings", {}).items():
            root = Path(str(binding.get("root", "")))
            if (
                not root.is_absolute()
                or _git_value(root, "rev-parse", "HEAD") != binding.get("commit")
                or _git_value(root, "rev-parse", "HEAD^{tree}") != binding.get("tree")
                or _git_porcelain_status(root) != binding.get("required_status", "")
            ):
                raise RuntimeError(
                    f"live source binding mismatch: {recipe.get('method_id')}:{phase}"
                )
            for relative, expected_sha in binding.get(
                "required_files_sha256", {}
            ).items():
                path = root / relative
                if not path.is_file() or sha256_file(path) != expected_sha:
                    raise RuntimeError(
                        "live source file identity mismatch: "
                        f"{recipe.get('method_id')}:{phase}:{relative}"
                    )
            rows.append({"method_id": recipe.get("method_id"), "phase": phase})
    if len(rows) != 23 or {row["method_id"] for row in rows} != set(METHOD_ORDER):
        raise RuntimeError("live source phase-binding cardinality mismatch")
    return rows


def validate_source_binding_correction(
    *, repo: Path, plan: dict[str, Any], require_live_sources: bool = False
) -> dict[str, Any]:
    """Validate old/new manifests, Linux proof, trigger audit, and live sources."""

    repo = repo.resolve()
    binding = plan.get("source_binding_correction", {})
    receipt_row = binding.get("receipt", {})
    receipt_path = _bound_path(repo, receipt_row.get("path", "")).resolve()
    if (
        not receipt_path.is_file()
        or receipt_path.is_symlink()
        or sha256_file(receipt_path) != receipt_row.get("sha256")
    ):
        raise RuntimeError("source-binding correction receipt identity mismatch")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    classification = receipt.get("classification", {})
    if (
        receipt.get("schema") != RECEIPT_SCHEMA
        or receipt.get("status") != RECEIPT_STATUS
        or receipt.get("status") != binding.get("status_required")
        or receipt.get("scene") != SCENE
        or receipt.get("method_id") != "3dgs_original"
        or receipt.get("phase") != "packet"
        or receipt.get("canonical_sha256") != canonical_sha256(receipt)
        or classification.get("type")
        != "LINUX_IDENTITY_METADATA_CORRECTION_ONLY"
        or classification.get("source_modified") is not False
        or classification.get("child_started") is not False
        or classification.get("attempt_consumed") is not False
        or classification.get("activation_v3_generated") is not False
        or classification.get("dual_hash_tolerance") is not False
        or classification.get("algorithm_or_renderer_semantics_changed") is not False
        or classification.get("budget_command_input_or_output_path_changed") is not False
        or binding.get("type") != "LINUX_IDENTITY_METADATA_CORRECTION_ONLY"
        or binding.get("source_modified") is not False
        or binding.get("child_started") is not False
        or binding.get("attempt_consumed") is not False
        or binding.get("dual_hash_tolerance") is not False
    ):
        raise RuntimeError("source-binding correction classification mismatch")

    hash_correction = receipt.get("hash_correction", {})
    if hash_correction != {
        "path": "submodules/diff-gaussian-rasterization/rasterize_points.h",
        "superseded_windows_checkout_sha256": WINDOWS_HEADER_SHA,
        "formal_linux_sha256": LINUX_HEADER_SHA,
        "formal_hash_count": 1,
    }:
        raise RuntimeError("source-binding hash correction is not singular")

    artifact_rows = _role_map(receipt.get("repository_artifacts"))
    artifact_paths: dict[str, Path] = {}
    for role, row in artifact_rows.items():
        relative = Path(str(row.get("path", "")))
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise RuntimeError(f"unsafe source-binding correction path: {relative}")
        path = (repo / relative).resolve()
        _validate_file(path, row, f"source-binding correction artifact {role}")
        artifact_paths[role] = path

    audit = receipt.get("trigger_audit", {})
    remote_audit_value = str(audit.get("remote_path", ""))
    remote_audit = Path(remote_audit_value)
    local_audit = artifact_paths["trigger_audit_local"]
    if (
        not PurePosixPath(remote_audit_value).is_absolute()
        or sha256_file(local_audit) != audit.get("sha256")
        or audit.get("phase_binding_count") != 23
        or audit.get("failed_binding_count") != 1
    ):
        raise RuntimeError("source-binding trigger audit identity mismatch")
    if require_live_sources and (
        not remote_audit.is_file()
        or remote_audit.is_symlink()
        or remote_audit.stat().st_size != int(audit.get("bytes", -1))
        or sha256_file(remote_audit) != audit.get("sha256")
    ):
        raise RuntimeError("remote source-binding trigger audit identity mismatch")
    audit_payload = json.loads(local_audit.read_text(encoding="utf-8"))
    failures = audit_payload.get("failed_bindings")
    if (
        audit_payload.get("schema")
        != "m3m_gcp_100k_all_source_binding_preflight_v1"
        or audit_payload.get("status") != "FAIL"
        or audit_payload.get("phase_binding_count") != 23
        or audit_payload.get("failed_binding_count") != 1
        or not isinstance(failures, list)
        or len(failures) != 1
    ):
        raise RuntimeError("source-binding trigger audit classification mismatch")
    failure = failures[0]
    mismatches = failure.get("mismatches")
    if (
        failure.get("method_id") != "3dgs_original"
        or failure.get("phase") != "packet"
        or not isinstance(mismatches, list)
        or len(mismatches) != 1
        or mismatches[0]
        != {
            "kind": "file",
            "path": "submodules/diff-gaussian-rasterization/rasterize_points.h",
            "expected": WINDOWS_HEADER_SHA,
            "actual": LINUX_HEADER_SHA,
        }
    ):
        raise RuntimeError("source-binding trigger audit mismatch is not isolated")

    old_manifest = json.loads(
        artifact_paths["recipe_manifest_v2"].read_text(encoding="utf-8")
    )
    old_recipe = json.loads(
        artifact_paths["3dgs_recipe_v2"].read_text(encoding="utf-8")
    )
    manifest = json.loads(
        artifact_paths["recipe_manifest_v3"].read_text(encoding="utf-8")
    )
    recipe = json.loads(
        artifact_paths["3dgs_recipe_v3"].read_text(encoding="utf-8")
    )
    proof_path = artifact_paths["linux_identity_proof"]
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    renderer_patch = artifact_paths["renderer_patch"]
    rasterizer_patch = artifact_paths["rasterizer_patch"]
    linux_files = _linux_files(proof)
    old_packet_binding = old_recipe.get("source_bindings", {}).get("packet", {})
    source_runtime = receipt.get("source_runtime", {})
    if (
        proof.get("schema") != "m3m_3dgs_eval_adapter_linux_identity_proof_v1"
        or proof.get("status") != "PASS"
        or proof.get("passed") is not True
        or proof.get("renderer_patch_sha256") != sha256_file(renderer_patch)
        or proof.get("rasterizer_patch_sha256") != sha256_file(rasterizer_patch)
        or len(linux_files) != 8
        or linux_files.get(
            "submodules/diff-gaussian-rasterization/rasterize_points.h"
        )
        != LINUX_HEADER_SHA
        or linux_files != receipt.get("formal_linux_patched_files_sha256")
        or source_runtime.get("root") != old_packet_binding.get("root")
        or source_runtime.get("repository_commit") != old_packet_binding.get("commit")
        or source_runtime.get("repository_tree") != old_packet_binding.get("tree")
        or source_runtime.get("required_status")
        != old_packet_binding.get("required_status")
    ):
        raise RuntimeError("formal Linux identity proof mismatch")

    if (
        old_manifest.get("schema")
        != "m3m_gcp_native_quarter_100k_recipe_manifest_v2"
        or manifest.get("schema")
        != "m3m_gcp_native_quarter_100k_recipe_manifest_v3"
        or manifest.get("method_order") != METHOD_ORDER
        or [row.get("method_id") for row in manifest.get("recipes", [])]
        != METHOD_ORDER
        or manifest.get("previous_manifest", {}).get("sha256")
        != sha256_file(artifact_paths["recipe_manifest_v2"])
        or manifest.get("previous_manifest", {}).get("canonical_sha256")
        != old_manifest.get("canonical_sha256")
        or manifest.get("correction_scope", {}).get("changed_method_ids")
        != ["3dgs_original"]
        or manifest.get("correction_scope", {}).get("unchanged_v2_recipe_rows")
        != 9
        or manifest.get("correction_scope", {}).get("source_modified") is not False
    ):
        raise RuntimeError("recipe manifest v3 correction scope mismatch")
    if manifest["recipes"][1:] != old_manifest.get("recipes", [])[1:]:
        raise RuntimeError("manifest v3 changed a non-3DGS recipe row")
    first = manifest["recipes"][0]
    if (
        first.get("method_id") != "3dgs_original"
        or _bound_path(repo, first.get("path", "")).resolve()
        != artifact_paths["3dgs_recipe_v3"]
        or first.get("sha256") != sha256_file(artifact_paths["3dgs_recipe_v3"])
        or first.get("canonical_sha256") != recipe.get("canonical_sha256")
    ):
        raise RuntimeError("manifest v3 3DGS recipe row mismatch")

    expected_recipe = copy.deepcopy(old_recipe)
    expected_recipe["schema"] = "m3m_gcp_native_quarter_100k_execution_recipe_v3"
    expected_recipe["source_bindings"]["packet"]["required_files_sha256"] = linux_files
    proof_relative = artifact_rows["linux_identity_proof"]["path"]
    expected_recipe["benchmark_required_files_sha256"][proof_relative] = sha256_file(
        proof_path
    )
    expected_recipe["source_identity_correction"] = _expected_recipe_correction(
        proof_path=proof_path,
        proof=proof,
        renderer_patch=renderer_patch,
        rasterizer_patch=rasterizer_patch,
    )
    expected_recipe["canonical_sha256"] = canonical_sha256(expected_recipe)
    if recipe != expected_recipe:
        raise RuntimeError("3DGS recipe v3 is not the single exact metadata correction")
    header_value = recipe["source_bindings"]["packet"][
        "required_files_sha256"
    ]["submodules/diff-gaussian-rasterization/rasterize_points.h"]
    if not isinstance(header_value, str) or header_value != LINUX_HEADER_SHA:
        raise RuntimeError("formal 3DGS header binding is not the single Linux hash")

    manifest_binding = binding.get("recipe_manifest", {})
    if (
        _bound_path(repo, manifest_binding.get("path", "")).resolve()
        != artifact_paths["recipe_manifest_v3"]
        or manifest_binding.get("sha256")
        != sha256_file(artifact_paths["recipe_manifest_v3"])
        or manifest_binding.get("canonical_sha256")
        != manifest.get("canonical_sha256")
    ):
        raise RuntimeError("plan source-binding correction manifest mismatch")
    if require_live_sources:
        _validate_live_sources(repo=repo, manifest=manifest)
    return receipt
