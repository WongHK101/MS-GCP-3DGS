#!/usr/bin/env python3
"""Build the exact ten-method pre-evaluation attempt manifest for 100K."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from m3m_gcp_lidar_artifacts import (
    METHOD_IDS,
    PROTOCOL_ID,
    canonical_sha256,
    sha256_file,
    validate_failure_evidence_file,
)


SCENE = "gcp_100000_20260610"
PLAIN_METHODS = {"3dgs_original", "2dgs", "pgsr", "rade_gs", "gsprior", "sof"}


def write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(
            descriptor,
            (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
    finally:
        os.close(descriptor)


def require_file(path: Path, expected_sha: str | None = None) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if expected_sha is not None and sha256_file(path) != expected_sha:
        raise RuntimeError(f"file SHA mismatch: {path}")
    return path


def inventory_row(path: Path) -> dict[str, Any]:
    path = require_file(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def read_summary(path: Path, *, method_id: str, required_status: str) -> dict[str, Any]:
    payload = json.loads(require_file(path).read_text(encoding="utf-8"))
    if payload.get("method_id") != method_id or payload.get("status") != required_status:
        raise RuntimeError(f"training summary identity/status mismatch: {path}")
    if payload.get("mode") != "formal":
        raise RuntimeError(f"training summary is not formal: {path}")
    return payload


def validate_frozen_attempt_paths(
    *, repo: Path, plan_path: Path, recipe_manifest_path: Path,
    registry_path: Path, identity_root: Path, output: Path,
) -> dict[str, Any]:
    plan_path = plan_path.resolve()
    plan = json.loads(require_file(plan_path).read_text(encoding="utf-8"))
    if (
        plan.get("schema")
        != "m3m_gcp_native_quarter_100k_ten_method_execution_plan_v1"
        or plan.get("scene") != SCENE
        or plan.get("canonical_sha256") != canonical_sha256(plan)
        or plan.get("execution_authorized") is not False
    ):
        raise RuntimeError("100K execution plan identity mismatch")
    freeze = plan.get("attempt_freeze", {})
    expected_plan = (repo / str(freeze.get("execution_plan_path", ""))).resolve()
    expected_recipe = (repo / str(freeze.get("recipe_manifest_path", ""))).resolve()
    expected_registry = (repo / str(freeze.get("method_registry_path", ""))).resolve()
    expected_identity = Path(str(freeze.get("model_identity_root", ""))).resolve()
    expected_output = Path(str(freeze.get("attempt_manifest_path", ""))).resolve()
    if plan_path != expected_plan:
        raise RuntimeError("attempt builder plan path differs from the frozen plan")
    if recipe_manifest_path.resolve() != expected_recipe:
        raise RuntimeError("attempt builder recipe manifest path differs from the frozen plan")
    if registry_path.resolve() != expected_registry:
        raise RuntimeError("attempt builder method registry path differs from the frozen plan")
    if identity_root.resolve() != expected_identity:
        raise RuntimeError("attempt builder model-identity root differs from the frozen plan")
    if output.resolve() != expected_output:
        raise RuntimeError("attempt builder output path differs from the frozen plan")
    return plan


def phase_success_inventory(
    path: Path, *, method_id: str, phase: str, recipe_sha256: str
) -> dict[str, Any]:
    payload = json.loads(require_file(path).read_text(encoding="utf-8"))
    if set(payload) != {
        "schema",
        "status",
        "scene",
        "method_id",
        "phase",
        "recipe_sha256",
        "command_sha256",
        "ended_at_utc",
        "canonical_sha256",
    }:
        raise RuntimeError(f"phase success field inventory mismatch: {path}")
    if (
        payload.get("schema") != "m3m_gcp_100k_phase_success_v1"
        or payload.get("status") != "PASS"
        or payload.get("scene") != SCENE
        or payload.get("method_id") != method_id
        or payload.get("phase") != phase
        or payload.get("recipe_sha256") != recipe_sha256
        or payload.get("canonical_sha256") != canonical_sha256(payload)
    ):
        raise RuntimeError(f"phase success identity mismatch: {path}")
    return inventory_row(path)


def success_inventory(method_id: str, recipe: dict[str, Any]) -> list[dict[str, Any]]:
    run_root = Path(str(recipe["authorized_run_root"])).resolve()
    budget = recipe.get("budget", {})
    rows: list[dict[str, Any]] = []
    if method_id in PLAIN_METHODS:
        iteration = int(budget.get("value", -1))
        if method_id == "3dgs_original":
            reuse = recipe.get("reuse_model_binding", {})
            iteration = 30000
            point_cloud = require_file(
                run_root / str(reuse.get("point_cloud_relative_path", "")),
                str(reuse.get("point_cloud_sha256", "")),
            )
        else:
            point_cloud = require_file(
                run_root / "model" / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
            )
        rows.append(inventory_row(point_cloud))
        cfg_args = run_root / "model" / "cfg_args"
        if cfg_args.is_file():
            rows.append(inventory_row(cfg_args))
        if method_id == "gsprior":
            rows.append(
                inventory_row(
                    Path(str(recipe["phase_roots"]["training"]["prior_root"]))
                    / "normalization_manifest.json"
                )
            )
    elif method_id == "qgs":
        iteration = int(budget.get("value", -1))
        rows.extend([
            inventory_row(run_root / "formal_training_config.yaml"),
            inventory_row(run_root / "model" / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"),
        ])
    elif method_id == "citygaussian_v2":
        summary_path = run_root / "pipeline" / "pipeline_summary.json"
        summary = read_summary(summary_path, method_id=method_id, required_status="PIPELINE_PASS")
        if int(summary.get("coarse_steps", -1)) != 30000 or int(summary.get("fine_steps", -1)) != 60000:
            raise RuntimeError("CityGaussianV2 summary budget mismatch")
        merged = summary.get("merged_checkpoint", {})
        config = summary.get("resolved_fine_config", {})
        cleanup = summary.get("transient_checkpoint_cleanup", {})
        rows.extend([
            inventory_row(summary_path),
            inventory_row(require_file(Path(str(merged.get("path", ""))), str(merged.get("sha256", "")))),
            inventory_row(require_file(Path(str(config.get("path", ""))), str(config.get("sha256", "")))),
            inventory_row(
                require_file(
                    Path(str(cleanup.get("inventory_path", ""))),
                    str(cleanup.get("inventory_sha256", "")),
                )
            ),
            inventory_row(
                Path(str(recipe["phase_roots"]["prior"]["prior_root"]))
                / "depth_prior_v1.json"
            ),
        ])
    elif method_id == "citygs_x":
        summary_path = run_root / "model" / "training_wrapper_summary.json"
        summary = read_summary(summary_path, method_id=method_id, required_status="TRAINING_PASS")
        if int(summary.get("iterations", -1)) != 100000:
            raise RuntimeError("CityGS-X summary budget mismatch")
        checkpoint = Path(str(summary.get("checkpoint", {}).get("path", ""))).resolve()
        point_cloud = require_file(
            checkpoint / str(summary.get("checkpoint", {}).get("point_cloud_file", "")),
            str(summary.get("checkpoint", {}).get("point_cloud_sha256", "")),
        )
        rows.extend([
            inventory_row(summary_path), inventory_row(point_cloud),
            inventory_row(checkpoint / "additional_attributes.npz"),
            inventory_row(checkpoint / "checkpoints.pth"),
            inventory_row(
                Path(str(recipe["phase_roots"]["prior"]["prior_root"]))
                / "depth_and_multiview_prior_v1.json"
            ),
        ])
    elif method_id == "metrogs":
        summary_path = run_root / "model" / "training_wrapper_summary.json"
        summary = read_summary(summary_path, method_id=method_id, required_status="TRAINING_PASS")
        if int(summary.get("effective_iterations", -1)) != 150000:
            raise RuntimeError("MetroGS summary budget mismatch")
        checkpoint = summary.get("checkpoint", {})
        cleanup = summary.get("rank_checkpoint_cleanup", {})
        rows.extend([
            inventory_row(summary_path),
            inventory_row(require_file(Path(str(checkpoint.get("merged_path", ""))), str(checkpoint.get("merged_sha256", "")))),
            inventory_row(require_file(Path(str(checkpoint.get("point_cloud_path", ""))), str(checkpoint.get("point_cloud_sha256", "")))),
            inventory_row(
                require_file(
                    Path(str(cleanup.get("inventory_path", ""))),
                    str(cleanup.get("inventory_sha256", "")),
                )
            ),
            inventory_row(
                Path(str(recipe["phase_roots"]["prior"]["prior_root"]))
                / "training_priors.json"
            ),
            inventory_row(
                Path(str(recipe["phase_roots"]["prior"]["prior_root"]))
                / "TRAINING_PRIORS_PASS"
            ),
        ])
    else:  # pragma: no cover - exact METHOD_IDS make this unreachable
        raise RuntimeError(f"unsupported method: {method_id}")
    if len({row["path"] for row in rows}) != len(rows):
        raise RuntimeError(f"{method_id}: duplicate model-identity inventory path")
    return sorted(rows, key=lambda row: row["path"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--recipe-manifest", type=Path, required=True)
    parser.add_argument("--method-registry", type=Path, required=True)
    parser.add_argument("--model-identity-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    plan_path = args.plan.resolve()
    recipe_manifest_path = args.recipe_manifest.resolve()
    registry_path = args.method_registry.resolve()
    identity_root = args.model_identity_root.resolve()
    output = args.output.resolve()
    validate_frozen_attempt_paths(
        repo=repo,
        plan_path=plan_path,
        recipe_manifest_path=recipe_manifest_path,
        registry_path=registry_path,
        identity_root=identity_root,
        output=output,
    )
    if output.exists() or identity_root.exists():
        raise FileExistsError("attempt/model-identity output already exists; overwrite is forbidden")
    manifest = json.loads(recipe_manifest_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if manifest.get("method_order") != list(METHOD_IDS) or len(manifest.get("recipes", [])) != 10:
        raise RuntimeError("recipe manifest is not the exact ordered ten-method pool")
    registry_rows = {row["method_id"]: row for row in registry.get("methods", [])}
    rows = []
    created_identities: list[Path] = []
    try:
        identity_root.mkdir(parents=True)
        for manifest_row in manifest["recipes"]:
            method_id = str(manifest_row["method_id"])
            recipe_path = (repo / str(manifest_row["path"])).resolve()
            require_file(recipe_path, str(manifest_row["sha256"]))
            recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
            if recipe.get("method_id") != method_id or recipe.get("scene") != SCENE:
                raise RuntimeError(f"{method_id}: recipe identity mismatch")
            adapter_path = (repo / str(recipe.get("renderer_adapter_path", ""))).resolve()
            require_file(adapter_path, str(recipe.get("renderer_adapter_sha256", "")))
            evidence_root = Path(str(recipe.get("authorized_evidence_root", ""))).resolve()
            failures = [
                path for path in (
                    evidence_root / "prior" / "failure.json",
                    evidence_root / "training" / "failure.json",
                ) if path.is_file()
            ]
            common = {
                "method_id": method_id,
                "method_name": registry_rows[method_id]["display_name"],
                "input_class": recipe["input_class"],
                "run_root": str(Path(str(recipe["authorized_run_root"])).resolve()),
                "recipe_path": str(recipe_path), "recipe_sha256": sha256_file(recipe_path),
                "renderer_adapter_path": str(adapter_path),
                "renderer_adapter_sha256": sha256_file(adapter_path),
            }
            if failures:
                if len(failures) != 1:
                    raise RuntimeError(f"{method_id}: multiple pre-freeze failures exist")
                failure = json.loads(failures[0].read_text(encoding="utf-8"))
                errors = validate_failure_evidence_file(
                    failures[0], expected_sha256=sha256_file(failures[0]),
                    expected_scene=SCENE, expected_method_id=method_id,
                    expected_status=str(failure.get("status", "")),
                )
                if errors or failure.get("status") not in {"OOM_UNRANKED", "FAILED_UNRANKED"}:
                    raise RuntimeError(f"{method_id}: invalid pre-freeze failure: {'; '.join(errors)}")
                if failure.get("run_root") != common["run_root"]:
                    raise RuntimeError(f"{method_id}: failure run root differs from recipe")
                rows.append({
                    **common, "attempt_status": failure["status"],
                    "model_checkpoint_path": None, "model_checkpoint_sha256": None,
                    "failure_evidence_path": str(failures[0]),
                    "failure_evidence_sha256": sha256_file(failures[0]),
                })
                continue
            inventory = success_inventory(method_id, recipe)
            for phase in ("prior", "training"):
                if phase in recipe.get("phase_commands", {}):
                    inventory.append(
                        phase_success_inventory(
                            evidence_root / phase / "phase_success.json",
                            method_id=method_id,
                            phase=phase,
                            recipe_sha256=sha256_file(recipe_path),
                        )
                    )
            inventory = sorted(inventory, key=lambda row: row["path"])
            identity = {
                "schema": "m3m_gcp_100k_model_identity_v1", "protocol_id": PROTOCOL_ID,
                "scene": SCENE, "method_id": method_id, "run_root": common["run_root"],
                "inventory": inventory,
            }
            identity["canonical_sha256"] = canonical_sha256(identity)
            identity_path = identity_root / f"{method_id}.json"
            write_exclusive(identity_path, identity)
            created_identities.append(identity_path)
            rows.append({
                **common, "attempt_status": "READY_FOR_EVALUATION",
                "model_checkpoint_path": str(identity_path),
                "model_checkpoint_sha256": sha256_file(identity_path),
                "failure_evidence_path": None, "failure_evidence_sha256": None,
            })
        payload = {
            "schema": "m3m_gcp_lidar_formal_methods_v1", "protocol_id": PROTOCOL_ID,
            "scene": SCENE, "methods": rows,
        }
        payload["canonical_sha256"] = canonical_sha256(payload)
        write_exclusive(output, payload)
    except Exception:
        for path in created_identities:
            path.unlink(missing_ok=True)
        try:
            identity_root.rmdir()
        except OSError:
            pass
        raise
    print(json.dumps({"status": "PASS_100K_ATTEMPT_MANIFEST_CREATED", "path": str(output), "sha256": sha256_file(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
