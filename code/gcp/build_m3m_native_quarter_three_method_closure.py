#!/usr/bin/env python3
"""Build the closed three-family preliminary-evidence package.

The builder consumes only frozen formal reports and their locally mirrored small
artifacts.  It does not launch training, render packets, or alter protocol
semantics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


METHOD_SPECS = (
    {
        "method_id": "3dgs_original",
        "display_name": "3DGS",
        "archive_directory": "3DGS_seed0",
        "family_id": "volumetric_3d_gaussian_ellipsoid",
        "family_label_zh": "三维体高斯/椭球类",
        "surface_semantics_zh": "三维协方差体高斯；表面位置由统一渲染支持取点算子解释。",
    },
    {
        "method_id": "2dgs",
        "display_name": "2DGS",
        "archive_directory": "2DGS_seed0",
        "family_id": "explicit_gaussian_surface_element",
        "family_label_zh": "显式高斯面元类",
        "surface_semantics_zh": "高斯基元自身对应局部平面面元。",
    },
    {
        "method_id": "gof",
        "display_name": "GOF",
        "archive_directory": "GOF_seed0",
        "family_id": "implicit_gaussian_opacity_field_isosurface",
        "family_label_zh": "隐式高斯场/等值面类",
        "surface_semantics_zh": "由连续高斯不透明度场的等值面定义方法原生表面。",
    },
)

POINT_CSV_RELATIVE = Path("formal_evaluation/evaluator/point_results.csv")
EVALUATOR_PATH = "code/gcp/evaluate_m3m_native_quarter_geometry.py"
PROTOCOL_CORE_PATH = "code/gcp/m3m_native_quarter_protocol.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def git_blob(repo_root: Path, commit: str, path: str) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", f"{commit}:{path}"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def source_commit(report: dict[str, Any]) -> str:
    return str(
        report.get("source", {}).get("official_repository_commit")
        or report.get("training", {}).get("source_commit")
        or ""
    )


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def format_mm(value_m: float) -> str:
    return f"{value_m * 1000.0:.2f}"


def build_markdown(
    methods: list[dict[str, Any]],
    combined_rows: list[dict[str, Any]],
    deferred_display_names: list[str],
    fairness: dict[str, Any],
) -> str:
    lines = [
        "# M3M-GCP 3K 前期三类代表方法收口",
        "",
        "> 状态：前期实验范围已锁定为三类表面表示、每类一个代表方法。QGS 等其余候选保持锁定；本文件不解锁新训练，也不改变 `m3m_gcp_native_quarter_geometry_v2` 协议语义。",
        "",
        "## 方法分类与代表方法",
        "",
        "| 一级类别 | 表面语义 | 本轮代表方法 | 正式状态 |",
        "|---|---|---|---|",
    ]
    for method in methods:
        lines.append(
            f"| {method['family_label_zh']} | {method['surface_semantics_zh']} | "
            f"{method['display_name']} | 3K、seed 0、30K，COMPLETE_RANKED |"
        )

    lines.extend(
        [
            "",
            "## 统一实验条件",
            "",
            "- 场景：`gcp_3000_20260602`；82 张训练影像，原生 1/4 分辨率 1414×1025。",
            "- 三种方法均为 RGB + 冻结 COLMAP 输入，不使用外部深度或几何先验。",
            "- 评测使用同一 66 相机名单、同一公共 Sim(3) 和同一 A/M1 期望相机 z 取点口径。",
            f"- 公共评测程序 Git blob：`{fairness['evaluator_core_git_blob']}`；公共协议核心 Git blob：`{fairness['protocol_core_git_blob']}`。",
            "- 每种方法仅运行一个 seed；以下数值是前期可行性证据，不作统计显著性声明。",
            "",
            "## 正式检查点结果与成本",
            "",
            "| 方法 | RMSE-3D (mm) | 水平 RMSE (mm) | 高程 RMSE (mm) | 训练时间 (min) | 峰值显存 (MiB) | 高斯数 (M) | 最终 PLY (MB) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in methods:
        checkpoint = method["residual_statistics_m"]["checkpoint"]
        training = method["training"]
        lines.append(
            f"| {method['display_name']} | {format_mm(checkpoint['rmse_3d_m'])} | "
            f"{format_mm(checkpoint['rmse_h_m'])} | {format_mm(checkpoint['rmse_z_m'])} | "
            f"{training['wall_seconds'] / 60.0:.2f} | {training['peak_gpu_memory_mib']:.0f} | "
            f"{training['gaussian_vertex_count'] / 1_000_000.0:.3f} | "
            f"{training['final_ply_bytes'] / 1_000_000.0:.2f} |"
        )

    by_point: dict[str, dict[str, dict[str, Any]]] = {}
    for row in combined_rows:
        by_point.setdefault(row["point_name"], {})[row["method_id"]] = row
    ordered_points = sorted(
        by_point,
        key=lambda name: (0, int(name[1:])) if name.startswith("G") and name[1:].isdigit() else (1, name),
    )
    lines.extend(
        [
            "",
            "## 逐点三维误差",
            "",
            "| 点名 | 角色 | 表面类型 | 3DGS (mm) | 2DGS (mm) | GOF (mm) |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for point_name in ordered_points:
        rows = by_point[point_name]
        exemplar = next(iter(rows.values()))
        lines.append(
            f"| {point_name} | {exemplar['role']} | {exemplar['surface_level']} | "
            f"{format_mm(rows['3dgs_original']['error_3d_m'])} | "
            f"{format_mm(rows['2dgs']['error_3d_m'])} | "
            f"{format_mm(rows['gof']['error_3d_m'])} |"
        )

    lines.extend(
        [
            "",
            "完整的逐点残差分量、观测覆盖与多视图离散度见 `protocol_evidence/m3m_native_quarter_three_method_point_results_v1.csv`。",
            "",
            "## 适用边界",
            "",
            "- 当前只有一个 3K 场景、一个 seed；不能据此宣称跨场景稳定性或统计优越性。",
            "- 4 个检查点和 5 个控制点的 `surface_level` 均为 `ground`，尚未验证屋顶等其他表面语义。",
            "- 主排名采用公共 A/M1 期望深度，不等同于各方法原生物理表面；GOF 等值面仅属于方法族定义，不混入公共主轨。",
            "- 5 个控制点用于冻结的公共 Sim(3) 对齐；正式精度比较以 4 个独立检查点为主。",
            f"- 延后候选：{', '.join(deferred_display_names)}。除非用户明确重开范围，否则不再做资格测试、正式训练、多 seed 或六场景扩展。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--created-utc", required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()

    require(
        re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", args.created_utc) is not None,
        "created-utc must use canonical YYYY-MM-DDTHH:MM:SSZ form",
    )

    repo_root = args.repo_root.resolve()
    archive_root = args.archive_root.resolve()
    registry_path = repo_root / "configs/m3m_gcp_native_quarter_method_registry_v2.json"
    registry = load_json(registry_path)
    require(registry["protocol_id"] == "m3m_gcp_native_quarter_geometry_v2", "protocol mismatch")
    require(registry["global_training_allowed"] is False, "global training is not locked")
    require(registry["per_method_training_allowed_methods"] == [], "method training allowlist is not empty")

    registry_methods = {item["method_id"]: item for item in registry["methods"]}
    selected_ids = [spec["method_id"] for spec in METHOD_SPECS]
    deferred_ids = [method_id for method_id in registry["method_ids"] if method_id not in selected_ids]
    deferred_display_names = [registry_methods[method_id]["display_name"] for method_id in deferred_ids]

    methods: list[dict[str, Any]] = []
    combined_rows: list[dict[str, Any]] = []
    shared_values: dict[str, Any] | None = None
    point_identity: list[tuple[str, str, str]] | None = None

    csv_fields = [
        "method_id", "display_name", "family_id", "family_label_zh", "scene", "seed", "iterations",
        "point_name", "role", "surface_level", "passed", "expected_observation_count",
        "required_valid_observation_count", "valid_observation_count", "valid_nadir_count",
        "valid_oblique_count", "residual_e_m", "residual_n_m", "residual_z_m", "error_h_m",
        "error_z_m", "error_3d_m", "multiview_scatter_median_m", "multiview_scatter_p90_m",
        "multiview_scatter_max_m",
    ]

    for spec in METHOD_SPECS:
        method_id = spec["method_id"]
        registry_method = registry_methods[method_id]
        formal_pointer = registry_method["formal_3k_result"]
        report_path = repo_root / formal_pointer["report"]
        require(sha256_file(report_path) == formal_pointer["report_sha256"], f"{method_id} formal report hash mismatch")
        report = load_json(report_path)
        require(report["method_id"] == method_id, f"{method_id} report identity mismatch")
        require(report["passed"] is True and report["evaluation"]["status"] == "COMPLETE_RANKED", f"{method_id} is not complete-ranked")
        require(report["evaluation"]["ranking_eligible"] is True, f"{method_id} is not ranking eligible")
        require(report["evaluation"]["method_specific_sim3_fitted"] is False, f"{method_id} fitted a method-specific Sim(3)")
        require(report["evaluation"]["physical_surface_claim"] is False, f"{method_id} makes a physical-surface claim")
        require(report["packet_export"]["packet_count"] == 66, f"{method_id} packet count mismatch")
        require(report["packet_export"]["packet_recomputation_all_passed"] is True, f"{method_id} packet recomputation failed")
        require(report["packet_export"]["variance_validation_fail_pixel_total"] == 0, f"{method_id} variance validation failed")
        require(report["training"]["memory_events_delta"]["oom"] == 0, f"{method_id} recorded OOM")
        require(registry_method["three_k_training_allowed"] is False, f"{method_id} remains launchable")
        require(registry_method["full_scene_matrix_eligible"] is False, f"{method_id} full matrix is unlocked")

        evaluator_commit = report["evaluation"]["evaluator_commit"]
        evaluator_blob = git_blob(repo_root, evaluator_commit, EVALUATOR_PATH)
        protocol_blob = git_blob(repo_root, evaluator_commit, PROTOCOL_CORE_PATH)
        input_block = report["input"]
        packet_block = report["packet_export"]
        protocol_block = report["protocol"]
        current_shared = {
            "protocol_id": report["protocol_id"],
            "scene": report["scene"],
            "seed": report["seed"],
            "iterations": report["iterations"],
            "release_root_digest_sha256": input_block["release_root_digest_sha256"],
            "formal_input_manifest_sha256": input_block["formal_input_manifest_sha256"],
            "train_view_count": input_block["train_view_count"],
            "heldout_test_view_count": input_block["heldout_test_view_count"],
            "width": input_block["width"],
            "height": input_block["height"],
            "common_sim3_sha256": report["evaluation"]["common_sim3_sha256"],
            "allowlist_sha256": packet_block["allowlist_sha256"],
            "release_manifest_sha256": protocol_block["release_manifest_sha256"],
            "sha256sums_sha256": protocol_block["sha256sums_sha256"],
            "evaluator_core_git_blob": evaluator_blob,
            "protocol_core_git_blob": protocol_blob,
        }
        if shared_values is None:
            shared_values = current_shared
        else:
            require(current_shared == shared_values, f"{method_id} shared fairness identity differs")

        point_path = archive_root / spec["archive_directory"] / POINT_CSV_RELATIVE
        expected_point_sha = report["evaluation"]["outputs"]["point_results.csv"]["sha256"]
        require(point_path.is_file(), f"{method_id} local point results missing")
        require(sha256_file(point_path) == expected_point_sha, f"{method_id} local point results hash mismatch")
        with point_path.open("r", encoding="utf-8-sig", newline="") as handle:
            source_rows = list(csv.DictReader(handle))
        require(len(source_rows) == 9, f"{method_id} point count mismatch")
        require(all(row["passed"].lower() == "true" for row in source_rows), f"{method_id} has failed points")
        identities = [(row["point_name"], row["role"], row["surface_level"]) for row in source_rows]
        if point_identity is None:
            point_identity = identities
        else:
            require(identities == point_identity, f"{method_id} point identity/order mismatch")

        for source_row in source_rows:
            row: dict[str, Any] = {
                "method_id": method_id,
                "display_name": spec["display_name"],
                "family_id": spec["family_id"],
                "family_label_zh": spec["family_label_zh"],
                "scene": report["scene"],
                "seed": report["seed"],
                "iterations": report["iterations"],
            }
            for field in csv_fields[7:]:
                if field in {"passed"}:
                    row[field] = source_row[field].lower() == "true"
                elif field in {
                    "expected_observation_count", "required_valid_observation_count", "valid_observation_count",
                    "valid_nadir_count", "valid_oblique_count",
                }:
                    row[field] = int(source_row[field])
                elif field in {
                    "residual_e_m", "residual_n_m", "residual_z_m", "error_h_m", "error_z_m", "error_3d_m",
                    "multiview_scatter_median_m", "multiview_scatter_p90_m", "multiview_scatter_max_m",
                }:
                    row[field] = as_float(source_row, field)
                else:
                    row[field] = source_row[field]
            combined_rows.append(row)

        source_status = (
            report.get("source", {}).get("official_training_source_status_porcelain_after")
            if "source" in report
            else report["training"].get("source_status_porcelain_after")
        )
        require(source_status == "", f"{method_id} official training source is dirty")
        methods.append(
            {
                **spec,
                "formal_report": formal_pointer["report"],
                "formal_report_sha256": formal_pointer["report_sha256"],
                "source_commit": source_commit(report),
                "evaluator_commit": evaluator_commit,
                "local_point_results": str(Path("3K") / spec["archive_directory"] / POINT_CSV_RELATIVE).replace("\\", "/"),
                "local_point_results_sha256": expected_point_sha,
                "training": {
                    "wall_seconds": report["training"]["wall_seconds"],
                    "gpu_hours": report["training"]["gpu_hours"],
                    "peak_gpu_memory_mib": report["training"]["peak_gpu_memory_mib"],
                    "gaussian_vertex_count": report["training"]["gaussian_vertex_count"],
                    "final_ply_bytes": report["training"]["final_ply_bytes"],
                    "final_ply_sha256": report["training"]["final_ply_sha256"],
                },
                "residual_statistics_m": report["evaluation"]["residual_statistics"],
                "packet_count": report["packet_export"]["packet_count"],
                "variance_validation_fail_pixel_total": report["packet_export"]["variance_validation_fail_pixel_total"],
            }
        )

    require(shared_values is not None and point_identity is not None, "no methods were processed")
    require({surface for _, _, surface in point_identity} == {"ground"}, "non-ground point unexpectedly present")
    require(sum(role == "checkpoint" for _, role, _ in point_identity) == 4, "checkpoint count mismatch")
    require(sum(role == "control" for _, role, _ in point_identity) == 5, "control count mismatch")

    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(combined_rows)

    markdown = build_markdown(methods, combined_rows, deferred_display_names, shared_values)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(markdown, encoding="utf-8", newline="\n")

    script_path = Path(__file__).resolve()
    report = {
        "schema": "m3m_gcp_native_quarter_three_method_closure_v1",
        "created_utc": args.created_utc,
        "protocol_id": registry["protocol_id"],
        "protocol_semantics_changed": False,
        "status": "PASS_PRELIMINARY_SCOPE_CLOSED",
        "passed": True,
        "generated_by": {
            "script": str(script_path.relative_to(repo_root)).replace("\\", "/"),
            "script_sha256": sha256_file(script_path),
        },
        "scope": {
            "scene": shared_values["scene"],
            "nominal_scene_scale": "3K",
            "seed": shared_values["seed"],
            "iterations": shared_values["iterations"],
            "single_scene_only": True,
            "single_seed_only": True,
            "statistical_significance_claim": False,
            "checkpoint_count": 4,
            "control_point_count": 5,
            "surface_levels": ["ground"],
            "selected_method_ids": selected_ids,
            "deferred_candidate_method_ids": deferred_ids,
            "candidate_expansion_status": "LOCKED_UNLESS_EXPLICITLY_REOPENED",
            "six_scene_matrix_status": "LOCKED",
            "multi_seed_status": "NOT_AUTHORIZED",
            "new_training_started_by_closure": False,
        },
        "classification": [
            {
                "family_id": method["family_id"],
                "family_label_zh": method["family_label_zh"],
                "surface_semantics_zh": method["surface_semantics_zh"],
                "representative_method_id": method["method_id"],
                "representative_display_name": method["display_name"],
            }
            for method in methods
        ],
        "fairness_identity": {
            **shared_values,
            "input_class": "rgb_colmap_only",
            "external_prior_used": False,
            "formal_common_primary": "render_support_expected_camera_z_A_over_M1",
            "method_specific_sim3_fitted": False,
            "physical_surface_claim": False,
            "all_packet_recomputation_passed": True,
            "all_variance_validation_fail_pixel_total": 0,
            "all_methods_complete_ranked": True,
        },
        "methods": methods,
        "derived_artifacts": {
            "point_results_csv": str(args.csv_output.resolve().relative_to(repo_root)).replace("\\", "/"),
            "point_results_csv_sha256": sha256_file(args.csv_output),
            "human_summary_markdown": str(args.markdown_output.resolve().relative_to(repo_root)).replace("\\", "/"),
            "human_summary_markdown_sha256": sha256_file(args.markdown_output),
        },
        "boundaries": [
            "The package is preliminary feasibility evidence, not a comprehensive algorithm benchmark.",
            "Only one 3K scene and seed 0 were run; no statistical significance is claimed.",
            "All four checkpoints and five control points are ground-level points.",
            "The common A/M1 primary is not asserted to be any method's native physical surface.",
            "Other candidate methods, multi-seed runs, and the six-scene matrix remain locked.",
            "M3M-GCP and TGS-GCP remain separate; this package does not read or modify TGS-GCP.",
        ],
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": "PASS", "methods": selected_ids, "point_rows": len(combined_rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
