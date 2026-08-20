#!/usr/bin/env python3
"""Build the frozen, independently verified six-scene LiDAR formal-v1 ranking."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from verify_m3m_gcp_lidar_formal_v1 import (
    METRIC_FIELDS,
    OVERALL_RANK_KEYS,
    canonical_sha256,
    competition_rank_rows,
    sha256_file,
)
from m3m_gcp_lidar_artifacts import (
    validate_failure_evidence_file,
    validate_scene_attempt_freeze,
)


PROTOCOL_ID = "m3m_gcp_lidar_rendered_surface_v1"
SCENES = (
    "gcp_3000_20260602",
    "gcp_5000_20260602",
    "gcp_20000_20260602",
    "gcp_10000_20260610",
    "gcp_50000_20260610",
    "gcp_100000_20260610",
)
COMPLETE = "COMPLETE_RANKED"
ALLOWED_STATUSES = {COMPLETE, "OOM_UNRANKED", "FAILED_UNRANKED", "INCOMPLETE_UNRANKED"}


def macro_mean(rows: list[dict[str, Any]], field: str) -> float:
    return sum(float(row["metrics"][field]) for row in rows) / len(rows)


def _validate_complete_result(
    *, entry: dict[str, Any], scene: str, method_id: str, input_class: str,
    contract: dict[str, Any], contract_sha256: str, activation_sha256: str,
    schema: dict[str, Any], schema_sha256: str, scene_attempt_freeze_sha256: str,
) -> dict[str, Any]:
    result_path = Path(str(entry["method_result_path"]))
    if not result_path.is_file() or sha256_file(result_path) != entry.get("method_result_sha256"):
        raise ValueError(f"{method_id}/{scene}: result path/SHA mismatch")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result_schema = schema["method_result_json"]
    if set(result) != set(result_schema["top_level_fields_exact"]):
        raise ValueError(f"{method_id}/{scene}: method result field inventory mismatch")
    if result.get("schema") != result_schema["schema"] or result.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"{method_id}/{scene}: method result schema/protocol mismatch")
    if result.get("canonical_sha256") != canonical_sha256(result):
        raise ValueError(f"{method_id}/{scene}: result canonical SHA mismatch")
    for field in result_schema["required_identity_fields"]:
        if not result.get(field):
            raise ValueError(f"{method_id}/{scene}: missing result identity {field}")
    for field in result_schema["required_count_fields"]:
        if not isinstance(result.get(field), int) or int(result[field]) < 1:
            raise ValueError(f"{method_id}/{scene}: invalid result count {field}")
    if result.get("contract_file_sha256") != contract_sha256:
        raise ValueError(f"{method_id}/{scene}: contract SHA mismatch")
    if result.get("activation_manifest_sha256") != activation_sha256:
        raise ValueError(f"{method_id}/{scene}: activation SHA mismatch")
    if result.get("artifact_schema_sha256") != schema_sha256:
        raise ValueError(f"{method_id}/{scene}: artifact-schema SHA mismatch")
    if result.get("scene_attempt_freeze_sha256") != scene_attempt_freeze_sha256:
        raise ValueError(f"{method_id}/{scene}: scene-attempt-freeze SHA mismatch")
    implementation = contract["implementation"]
    for field, expected in (
        ("evaluator_sha256", implementation["evaluator_sha256"]),
        ("verifier_sha256", implementation["verifier_sha256"]),
        ("artifact_schema_sha256", implementation["artifact_schema_sha256"]),
    ):
        if result.get(field) != expected:
            raise ValueError(f"{method_id}/{scene}: {field} differs from contract")
    if result.get("scene") != scene or result.get("method_id") != method_id:
        raise ValueError(f"{method_id}/{scene}: result identity mismatch")
    if result.get("input_class") != input_class:
        raise ValueError(f"{method_id}/{scene}: input class mismatch")
    if set(result.get("metrics", {})) != set(METRIC_FIELDS):
        raise ValueError(f"{method_id}/{scene}: metric inventory mismatch")
    if any(not math.isfinite(float(value)) for value in result["metrics"].values()):
        raise ValueError(f"{method_id}/{scene}: nonfinite metric")

    report_path = Path(str(entry["verification_report_path"]))
    if not report_path.is_file() or sha256_file(report_path) != entry.get("verification_report_sha256"):
        raise ValueError(f"{method_id}/{scene}: verification report path/SHA mismatch")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report_schema = schema["method_verification_report_json"]
    if set(report) != set(report_schema["required_fields_exact"]):
        raise ValueError(f"{method_id}/{scene}: verification report field inventory mismatch")
    if report.get("schema") != report_schema["schema"] or report.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"{method_id}/{scene}: verification report schema/protocol mismatch")
    if report.get("status") != report_schema["required_status"]:
        raise ValueError(f"{method_id}/{scene}: verification report is not PASS")
    if report.get("canonical_sha256") != canonical_sha256(report):
        raise ValueError(f"{method_id}/{scene}: verification report canonical SHA mismatch")
    expected_report = {
        "method_id": method_id, "scene": scene,
        "method_result_sha256": entry["method_result_sha256"],
        "contract_file_sha256": contract_sha256,
        "activation_manifest_sha256": activation_sha256,
        "scene_execution_authorization_sha256": result["scene_execution_authorization_sha256"],
        "scene_attempt_freeze_sha256": scene_attempt_freeze_sha256,
        "formal_methods_manifest_sha256": result["formal_methods_manifest_sha256"],
        "artifact_schema_sha256": schema_sha256,
        "evaluator_sha256": result["evaluator_sha256"],
        "verifier_sha256": result["verifier_sha256"],
        "surface_npz_sha256": result["surface_npz_sha256"],
        "distance_npz_sha256": result["distance_npz_sha256"],
        "reference_npz_sha256": result["reference_npz_sha256"],
        "reconstruction_to_lidar_distance_count": result["reconstruction_to_lidar_distance_count"],
        "lidar_to_reconstruction_distance_count": result["lidar_to_reconstruction_distance_count"],
    }
    for field, expected in expected_report.items():
        if report.get(field) != expected:
            raise ValueError(f"{method_id}/{scene}: verification report mismatch: {field}")
    if report.get("errors") != [] or set(report.get("recomputed_metrics", {})) != set(METRIC_FIELDS):
        raise ValueError(f"{method_id}/{scene}: invalid verifier payload")
    for field in METRIC_FIELDS:
        if not math.isclose(float(report["recomputed_metrics"][field]), float(result["metrics"][field]), rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"{method_id}/{scene}: verifier metric mismatch: {field}")
    return result


def build_ranking(
    manifest: dict[str, Any], *, contract: dict[str, Any], contract_sha256: str,
    activation: dict[str, Any], activation_sha256: str, schema: dict[str, Any],
    schema_sha256: str, registry: dict[str, Any], registry_sha256: str,
) -> dict[str, Any]:
    if manifest.get("schema") != "m3m_gcp_lidar_six_scene_results_manifest_v1":
        raise ValueError("six-scene results manifest schema mismatch")
    if manifest.get("protocol_id") != PROTOCOL_ID or manifest.get("canonical_sha256") != canonical_sha256(manifest):
        raise ValueError("six-scene results manifest protocol/canonical mismatch")
    if activation.get("canonical_sha256") != canonical_sha256(activation):
        raise ValueError("activation canonical SHA mismatch")
    if activation.get("contract_file_sha256") != contract_sha256:
        raise ValueError("activation contract SHA mismatch")
    results_schema = schema["six_scene_results_manifest"]
    if set(manifest) != set(results_schema["top_level_fields_exact"]):
        raise ValueError("six-scene results manifest field inventory mismatch")
    freeze_rows = manifest.get("scene_attempt_freezes", [])
    if [row.get("scene") for row in freeze_rows] != list(SCENES):
        raise ValueError("scene attempt freeze order differs from frozen six scenes")
    freeze_fields = set(results_schema["scene_attempt_freeze_fields_exact"])
    scene_freezes: dict[str, tuple[str, dict[str, Any]]] = {}
    for row in freeze_rows:
        scene = str(row.get("scene"))
        if set(row) != freeze_fields:
            raise ValueError(f"{scene}: scene attempt freeze entry fields mismatch")
        freeze_path = Path(str(row.get("path", "")))
        if not freeze_path.is_file() or sha256_file(freeze_path) != row.get("sha256"):
            raise ValueError(f"{scene}: scene attempt freeze path/SHA mismatch")
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        freeze_errors, frozen_methods = validate_scene_attempt_freeze(
            freeze, freeze_path=freeze_path, expected_scene=scene
        )
        if freeze_errors or frozen_methods is None:
            raise ValueError(f"{scene}: invalid scene attempt freeze: {freeze_errors}")
        scene_freezes[scene] = (str(row["sha256"]), frozen_methods)
    binding = contract["method_registry_binding"]
    if registry_sha256 != binding["file_sha256"]:
        raise ValueError("method registry SHA mismatch")
    method_ids = list(binding["active_method_ids_in_order"])
    method_classes = dict(binding["active_method_input_classes"])
    registry_rows = {row["method_id"]: row for row in registry.get("methods", [])}
    if registry.get("active_benchmark_method_ids") != method_ids:
        raise ValueError("method registry active order mismatch")
    methods = manifest.get("methods", [])
    if [row.get("method_id") for row in methods] != method_ids:
        raise ValueError("results manifest is not the exact ordered ten-method pool")
    method_fields = set(results_schema["method_fields_exact"])
    scene_fields = set(results_schema["scene_entry_fields_exact"])

    output_rows: list[dict[str, Any]] = []
    for method in methods:
        method_id = str(method["method_id"])
        if set(method) != method_fields:
            raise ValueError(f"{method_id}: method fields differ from artifact schema")
        if method.get("input_class") != method_classes[method_id]:
            raise ValueError(f"{method_id}: frozen input class mismatch")
        if method.get("method_name") != registry_rows[method_id].get("display_name"):
            raise ValueError(f"{method_id}: frozen method name mismatch")
        scene_entries = method.get("scenes", [])
        if [entry.get("scene") for entry in scene_entries] != list(SCENES):
            raise ValueError(f"{method_id}: scene order differs from frozen six scenes")
        complete_results: list[dict[str, Any]] = []
        statuses: dict[str, str] = {}
        for entry in scene_entries:
            scene = str(entry["scene"])
            if set(entry) != scene_fields:
                raise ValueError(f"{method_id}/{scene}: scene-entry field inventory mismatch")
            status = str(entry["status"])
            if status not in ALLOWED_STATUSES:
                raise ValueError(f"{method_id}/{scene}: unknown status {status}")
            statuses[scene] = status
            evidence_fields = ("method_result_path", "method_result_sha256", "verification_report_path", "verification_report_sha256")
            failure_fields = ("failure_evidence_path", "failure_evidence_sha256")
            freeze_sha, frozen_methods = scene_freezes[scene]
            frozen_method = next(
                (row for row in frozen_methods.get("methods", []) if row.get("method_id") == method_id),
                None,
            )
            if frozen_method is None:
                raise ValueError(f"{method_id}/{scene}: absent from scene attempt freeze")
            if status != COMPLETE:
                if any(entry.get(field) is not None for field in evidence_fields):
                    raise ValueError(f"{method_id}/{scene}: failed status cannot carry fabricated result")
                if any(not entry.get(field) for field in failure_fields):
                    raise ValueError(f"{method_id}/{scene}: failed status lacks immutable failure evidence")
                if frozen_method.get("attempt_status") != status:
                    raise ValueError(f"{method_id}/{scene}: final failure status differs from scene attempt freeze")
                if frozen_method.get("failure_evidence_path") != entry.get("failure_evidence_path") or frozen_method.get("failure_evidence_sha256") != entry.get("failure_evidence_sha256"):
                    raise ValueError(f"{method_id}/{scene}: failure evidence differs from scene attempt freeze")
                failure_errors = validate_failure_evidence_file(
                    Path(str(entry["failure_evidence_path"])),
                    expected_sha256=str(entry["failure_evidence_sha256"]),
                    expected_scene=scene,
                    expected_method_id=method_id,
                    expected_status=status,
                )
                if failure_errors:
                    raise ValueError(f"{method_id}/{scene}: invalid failure evidence: {failure_errors}")
                continue
            if any(not entry.get(field) for field in evidence_fields):
                raise ValueError(f"{method_id}/{scene}: complete result lacks verifier evidence")
            if any(entry.get(field) is not None for field in failure_fields):
                raise ValueError(f"{method_id}/{scene}: complete result carries failure evidence")
            if frozen_method.get("attempt_status") != "READY_FOR_EVALUATION":
                raise ValueError(f"{method_id}/{scene}: complete result is not READY in scene attempt freeze")
            complete_results.append(_validate_complete_result(
                entry=entry, scene=scene, method_id=method_id, input_class=method["input_class"],
                contract=contract, contract_sha256=contract_sha256,
                activation_sha256=activation_sha256, schema=schema, schema_sha256=schema_sha256,
                scene_attempt_freeze_sha256=freeze_sha,
            ))

        completed = len(complete_results)
        row: dict[str, Any] = {
            "method_id": method_id, "method_name": method["method_name"],
            "input_class": method["input_class"], "completed_scene_count": completed,
            "scene_statuses": statuses,
            "overall_status": COMPLETE if completed == len(SCENES) else "INCOMPLETE_UNRANKED",
            "ranking_eligible": completed == len(SCENES),
        }
        if completed:
            row["partial_macro_diagnostic"] = {field: macro_mean(complete_results, field) for field in METRIC_FIELDS}
        if completed == len(SCENES):
            row.update({
                "macro_fscore_10cm": macro_mean(complete_results, "fscore_10cm"),
                "macro_chamfer_l1_mean_m": macro_mean(complete_results, "chamfer_l1_mean_m"),
                "macro_precision_10cm": macro_mean(complete_results, "precision_10cm"),
                "macro_recall_10cm": macro_mean(complete_results, "recall_10cm"),
            })
        output_rows.append(row)

    eligible = [row for row in output_rows if row["ranking_eligible"]]
    for input_class in sorted({row["input_class"] for row in eligible}):
        ranked = competition_rank_rows([row for row in eligible if row["input_class"] == input_class], OVERALL_RANK_KEYS)
        ranks = {row["method_id"]: row["rank"] for row in ranked}
        for row in output_rows:
            if row["ranking_eligible"] and row["input_class"] == input_class:
                row["official_input_class_rank"] = ranks[row["method_id"]]
    descriptive = competition_rank_rows(eligible, OVERALL_RANK_KEYS)
    positions = {row["method_id"]: index for index, row in enumerate(descriptive, 1)}
    for row in output_rows:
        if row["ranking_eligible"]:
            row["combined_descriptive_order_not_official_rank"] = positions[row["method_id"]]
    output = {
        "schema": "m3m_gcp_lidar_six_scene_ranking_v1", "protocol_id": PROTOCOL_ID,
        "contract_file_sha256": contract_sha256, "activation_manifest_sha256": activation_sha256,
        "artifact_schema_sha256": schema_sha256, "method_registry_sha256": registry_sha256,
        "scene_order": list(SCENES), "aggregation": "unweighted_arithmetic_macro_average",
        "micro_pooling": "FORBIDDEN", "official_ranking_scope": "within_input_class_only",
        "complete_scene_count_required_for_rank": 6, "methods": output_rows,
    }
    output["canonical_sha256"] = canonical_sha256(output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--artifact-schema", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite six-scene ranking")
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    activation = json.loads(args.activation.read_text(encoding="utf-8"))
    schema = json.loads(args.artifact_schema.read_text(encoding="utf-8"))
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    result = build_ranking(
        json.loads(args.manifest.read_text(encoding="utf-8")),
        contract=contract, contract_sha256=sha256_file(args.contract),
        activation=activation, activation_sha256=sha256_file(args.activation),
        schema=schema, schema_sha256=sha256_file(args.artifact_schema),
        registry=registry, registry_sha256=sha256_file(args.registry),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
