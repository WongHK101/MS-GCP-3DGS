#!/usr/bin/env python3
"""Build the exact ten-method pre-evaluation attempt manifest for 100K."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

from m3m_gcp_lidar_artifacts import (
    METHOD_IDS,
    PROTOCOL_ID,
    canonical_sha256,
    command_sha256,
    sha256_file,
    validate_failure_evidence_file,
)
from m3m_gcp_100k_phase_products import (
    phase_product_row,
    revalidate_phase_product_row,
    validate_gaussian_ply,
    validate_npz,
    validate_torch_checkpoint,
)
from m3m_gcp_100k_activation_v4_continuity import validate_continuity_for_plan
from m3m_gcp_100k_source_binding_correction import (
    validate_source_binding_correction,
)


SCENE = "gcp_100000_20260610"
PLAIN_METHODS = {"3dgs_original", "2dgs", "pgsr", "rade_gs", "gsprior", "sof"}
REQUIRED_NOFILE_SOFT = 65536


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


def inventory_row(
    path: Path, *, validate_model_container: bool = False,
    method_id: str | None = None,
) -> dict[str, Any]:
    path = require_file(path)
    if validate_model_container:
        if path.suffix.lower() == ".ply":
            if not method_id:
                raise RuntimeError("Gaussian model inventory validation requires a method ID")
            validate_gaussian_ply(path, method_id=method_id)
        elif path.suffix.lower() in {".ckpt", ".pth"}:
            validate_torch_checkpoint(path)
        elif path.suffix.lower() == ".npz":
            validate_npz(path)
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
        plan.get("schema") not in {
            "m3m_gcp_native_quarter_100k_ten_method_execution_plan_v3",
            "m3m_gcp_native_quarter_100k_ten_method_execution_plan_v4",
        }
        or plan.get("scene") != SCENE
        or plan.get("canonical_sha256") != canonical_sha256(plan)
        or plan.get("execution_authorized") is not False
    ):
        raise RuntimeError("100K execution plan identity mismatch")
    validate_continuity_for_plan(repo=repo, plan=plan)
    validate_source_binding_correction(repo=repo, plan=plan)
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
    path: Path, *, method_id: str, phase: str, recipe_sha256: str,
    expected_command_sha256: str,
    frozen_budget: dict[str, Any] | None = None,
    expected_product_paths: list[Path] | None = None,
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
        "frozen_budget",
        "environment_manifest_path",
        "environment_manifest_sha256",
        "completion_evidence",
        "products",
        "ended_at_utc",
        "canonical_sha256",
    }:
        raise RuntimeError(f"phase success field inventory mismatch: {path}")
    if (
        payload.get("schema") != "m3m_gcp_100k_phase_success_v2"
        or payload.get("status") != "PASS"
        or payload.get("scene") != SCENE
        or payload.get("method_id") != method_id
        or payload.get("phase") != phase
        or payload.get("recipe_sha256") != recipe_sha256
        or payload.get("command_sha256") != expected_command_sha256
        or payload.get("frozen_budget") != (frozen_budget or {})
        or payload.get("canonical_sha256") != canonical_sha256(payload)
    ):
        raise RuntimeError(f"phase success identity mismatch: {path}")
    environment_path = require_file(
        Path(str(payload.get("environment_manifest_path", ""))),
        str(payload.get("environment_manifest_sha256", "")),
    )
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    limits = environment.get("resource_limits", {})
    parent_after = limits.get("parent_after", {})
    child = limits.get("child_actual", {})

    def hard_ok(value: object) -> bool:
        return value == "unlimited" or (
            isinstance(value, int) and value >= REQUIRED_NOFILE_SOFT
        )

    if (
        environment.get("schema") != "m3m_gcp_100k_execution_environment_v2"
        or environment.get("scene") != SCENE
        or environment.get("method_id") != method_id
        or environment.get("phase") != phase
        or environment.get("canonical_sha256") != canonical_sha256(environment)
        or limits.get("required_soft") != REQUIRED_NOFILE_SOFT
        or limits.get("hard_minimum") != REQUIRED_NOFILE_SOFT
        or parent_after.get("soft") != REQUIRED_NOFILE_SOFT
        or not hard_ok(parent_after.get("hard"))
        or child.get("soft") != REQUIRED_NOFILE_SOFT
        or not hard_ok(child.get("hard"))
    ):
        raise RuntimeError(f"phase success environment evidence mismatch: {path}")
    completion = payload.get("completion_evidence")
    if (
        not isinstance(completion, dict)
        or completion.get("required_product_postvalidation_passed") is not True
        or not isinstance(completion.get("progress_unit"), str)
        or not isinstance(completion.get("last_valid_progress"), (int, float))
        or not math.isfinite(float(completion.get("last_valid_progress", float("nan"))))
    ):
        raise RuntimeError(f"phase success completion evidence mismatch: {path}")
    products = payload.get("products")
    if not isinstance(products, list) or not products:
        raise RuntimeError(f"phase success product inventory is empty: {path}")
    product_paths = [revalidate_phase_product_row(row) for row in products]
    if product_paths != sorted(product_paths, key=str) or len(product_paths) != len(set(product_paths)):
        raise RuntimeError(f"phase success product order/cardinality mismatch: {path}")
    if expected_product_paths is not None:
        expected_rows = [
            phase_product_row(
                item,
                validate_model_container=phase == "training",
                method_id=method_id if phase == "training" else None,
            )
            for item in sorted((item.resolve() for item in expected_product_paths), key=str)
        ]
        if products != expected_rows:
            raise RuntimeError(f"phase success products differ from final model: {path}")
    return inventory_row(path)


def frozen_phase_command_sha256(
    *, recipe: dict[str, Any], phase: str, repo: Path
) -> str:
    roots = recipe.get("phase_roots", {}).get(phase, {})
    source = recipe.get("source_bindings", {}).get(phase, {})
    template = recipe.get("phase_commands", {}).get(phase)
    if not isinstance(template, list) or not template:
        raise RuntimeError(f"recipe has no frozen {phase} command")
    replacements = {
        "repo": str(repo.resolve()),
        "dataset_root": str(Path(str(roots.get("dataset_root", ""))).resolve()),
        "source_root": str(Path(str(source.get("root", ""))).resolve()),
        "prior_root": str(Path(str(roots.get("prior_root", ""))).resolve()),
        "run_root": str(Path(str(recipe.get("authorized_run_root", ""))).resolve()),
        "packet_set_root": str(
            Path(str(recipe.get("authorized_packet_set_root", ""))).resolve()
        ),
    }
    return command_sha256([str(item).format(**replacements) for item in template])


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
        rows.append(inventory_row(
            point_cloud, validate_model_container=True, method_id=method_id
        ))
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
            inventory_row(
                run_root / "model" / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply",
                validate_model_container=True,
                method_id=method_id,
            ),
        ])
    elif method_id == "citygaussian_v2":
        summary_path = run_root / "pipeline" / "pipeline_summary.json"
        summary = read_summary(summary_path, method_id=method_id, required_status="PIPELINE_PASS")
        if (
            summary.get("scene") != SCENE
            or summary.get("protocol_id") != "m3m_gcp_native_quarter_geometry_v2"
            or summary.get("formal_result") is not True
            or int(summary.get("coarse_steps", -1)) != 30000
            or int(summary.get("fine_steps", -1)) != 60000
        ):
            raise RuntimeError("CityGaussianV2 summary budget mismatch")
        merged = summary.get("merged_checkpoint", {})
        config = summary.get("resolved_fine_config", {})
        cleanup = summary.get("transient_checkpoint_cleanup", {})
        rows.extend([
            inventory_row(summary_path),
            inventory_row(
                require_file(Path(str(merged.get("path", ""))), str(merged.get("sha256", ""))),
                validate_model_container=True,
                method_id=method_id,
            ),
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
        if (
            summary.get("scene") != SCENE
            or summary.get("protocol_id") != "m3m_gcp_native_quarter_geometry_v2"
            or summary.get("formal_result") is not True
            or int(summary.get("iterations", -1)) != 100000
        ):
            raise RuntimeError("CityGS-X summary budget mismatch")
        checkpoint = Path(str(summary.get("checkpoint", {}).get("path", ""))).resolve()
        point_cloud = require_file(
            checkpoint / str(summary.get("checkpoint", {}).get("point_cloud_file", "")),
            str(summary.get("checkpoint", {}).get("point_cloud_sha256", "")),
        )
        checkpoint_row = summary.get("checkpoint", {})
        attributes = checkpoint_row.get("additional_attributes", {})
        optimizer = checkpoint_row.get("optimizer_checkpoint", {})
        if (
            point_cloud.stat().st_size != checkpoint_row.get("point_cloud_bytes")
            or Path(str(attributes.get("path", ""))).resolve()
            != checkpoint / "additional_attributes.npz"
            or require_file(
                checkpoint / "additional_attributes.npz", str(attributes.get("sha256", ""))
            ).stat().st_size != attributes.get("bytes")
            or Path(str(optimizer.get("path", ""))).resolve() != checkpoint / "checkpoints.pth"
            or require_file(
                checkpoint / "checkpoints.pth", str(optimizer.get("sha256", ""))
            ).stat().st_size != optimizer.get("bytes")
        ):
            raise RuntimeError("CityGS-X companion checkpoint identity mismatch")
        rows.extend([
            inventory_row(summary_path),
            inventory_row(point_cloud, validate_model_container=True, method_id=method_id),
            inventory_row(
                checkpoint / "additional_attributes.npz",
                validate_model_container=True,
                method_id=method_id,
            ),
            inventory_row(
                checkpoint / "checkpoints.pth",
                validate_model_container=True,
                method_id=method_id,
            ),
            inventory_row(
                Path(str(recipe["phase_roots"]["prior"]["prior_root"]))
                / "depth_and_multiview_prior_v1.json"
            ),
        ])
    elif method_id == "metrogs":
        summary_path = run_root / "model" / "training_wrapper_summary.json"
        summary = read_summary(summary_path, method_id=method_id, required_status="TRAINING_PASS")
        if (
            summary.get("scene") != SCENE
            or summary.get("protocol_id") != "m3m_gcp_native_quarter_geometry_v2"
            or summary.get("formal_result") is not True
            or int(summary.get("effective_iterations", -1)) != 150000
            or int(summary.get("optimizer_steps", -1)) != 37500
        ):
            raise RuntimeError("MetroGS summary budget mismatch")
        checkpoint = summary.get("checkpoint", {})
        cleanup = summary.get("rank_checkpoint_cleanup", {})
        rows.extend([
            inventory_row(summary_path),
            inventory_row(
                require_file(Path(str(checkpoint.get("merged_path", ""))), str(checkpoint.get("merged_sha256", ""))),
                validate_model_container=True,
                method_id=method_id,
            ),
            inventory_row(
                require_file(Path(str(checkpoint.get("point_cloud_path", ""))), str(checkpoint.get("point_cloud_sha256", ""))),
                validate_model_container=True,
                method_id=method_id,
            ),
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


def required_training_product_paths(method_id: str, recipe: dict[str, Any]) -> list[Path]:
    run_root = Path(str(recipe["authorized_run_root"])).resolve()
    budget = recipe.get("budget", {})
    if method_id in PLAIN_METHODS:
        if method_id == "3dgs_original":
            return [run_root / str(recipe["reuse_model_binding"]["point_cloud_relative_path"])]
        return [
            run_root / "model" / "point_cloud"
            / f"iteration_{int(budget.get('value', -1))}" / "point_cloud.ply"
        ]
    if method_id == "qgs":
        return [
            run_root / "model" / "point_cloud"
            / f"iteration_{int(budget.get('value', -1))}" / "point_cloud.ply"
        ]
    if method_id == "citygaussian_v2":
        summary_path = run_root / "pipeline" / "pipeline_summary.json"
        summary = json.loads(require_file(summary_path).read_text(encoding="utf-8"))
        return [summary_path, Path(str(summary["merged_checkpoint"]["path"])).resolve()]
    summary_path = run_root / "model" / "training_wrapper_summary.json"
    summary = json.loads(require_file(summary_path).read_text(encoding="utf-8"))
    checkpoint = summary["checkpoint"]
    if method_id == "citygs_x":
        checkpoint_root = Path(str(checkpoint["path"])).resolve()
        return [
            summary_path,
            checkpoint_root / str(checkpoint["point_cloud_file"]),
            checkpoint_root / "additional_attributes.npz",
            checkpoint_root / "checkpoints.pth",
        ]
    if method_id == "metrogs":
        cleanup = summary["rank_checkpoint_cleanup"]
        return [
            summary_path,
            Path(str(checkpoint["merged_path"])).resolve(),
            Path(str(checkpoint["point_cloud_path"])).resolve(),
            Path(str(cleanup["inventory_path"])).resolve(),
        ]
    raise RuntimeError(f"unsupported training product method: {method_id}")


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
                            expected_command_sha256=frozen_phase_command_sha256(
                                recipe=recipe, phase=phase, repo=repo
                            ),
                            frozen_budget=recipe.get("budget", {}),
                            expected_product_paths=(
                                required_training_product_paths(method_id, recipe)
                                if phase == "training" else None
                            ),
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
