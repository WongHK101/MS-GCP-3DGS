#!/usr/bin/env python
"""Build a Chinese visual-review HTML entrypoint for the multiview audit.

The source audit package intentionally contains many candidate projections.
This helper creates a clearer review interface that separates:

- existing human annotations;
- candidate images to label;
- candidate images to review;
- rejected candidates.

It only reads the previous audit outputs and writes a new review helper folder.
It does not modify any release, split, metric packet, or evaluator output.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
import time
import zipfile
from pathlib import Path

import pandas as pd


DEFAULT_AUDIT_ROOT = Path(
    r"E:\M3M-GCP-3DGS\outputs\gcp_multiview_annotation_expansion_control_heavy_audit_20260702_160926"
)
DEFAULT_OUTPUT_PARENT = Path(r"E:\M3M-GCP-3DGS\outputs")


SOURCE_CN = {
    "current_annotation": "已标注：你之前保存的正式标注",
    "triangulated_annotation_rays": "候选：由已有标注视线三角化后预测",
    "coarse_exif_gimbal_seed": "候选：EXIF/Gimbal 粗投影，仅作搜索线索",
}

ACTION_CN = {
    "keep_existing": "已标注，核查即可",
    "label": "建议补标：若真实可见，请后续打开标注工具人工点",
    "review": "仅复核：判断是否可见/是否值得补标",
    "reject": "暂拒绝：大概率不可用或超出搜索范围",
}


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def esc(value: object) -> str:
    return html.escape(str(value))


def file_uri(path: object) -> str:
    if not path:
        return ""
    try:
        return Path(str(path)).resolve().as_uri()
    except Exception:
        return ""


def table_html(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    use = df[columns].copy()
    if max_rows is not None:
        use = use.head(max_rows)
    lines = ["<table><thead><tr>"]
    for column in columns:
        lines.append(f"<th>{esc(column)}</th>")
    lines.append("</tr></thead><tbody>")
    for row in use.itertuples(index=False):
        lines.append("<tr>")
        for value in row:
            lines.append(f"<td>{esc(value)}</td>")
        lines.append("</tr>")
    lines.append("</tbody></table>")
    return "\n".join(lines)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit_root", type=Path, default=DEFAULT_AUDIT_ROOT)
    parser.add_argument("--output_parent", type=Path, default=DEFAULT_OUTPUT_PARENT)
    parser.add_argument("--stamp", default=time.strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    audit_root = args.audit_root
    out_root = args.output_parent / f"gcp_multiview_visual_review_zh_{args.stamp}"
    out_root.mkdir(parents=True, exist_ok=False)

    matrix = pd.read_csv(audit_root / "point_image_candidate_coverage_matrix.csv")
    worklist = pd.read_csv(audit_root / "manual_supplemental_annotation_worklist.csv")
    low_view = pd.read_csv(audit_root / "low_view_cause_classification.csv")
    contacts = pd.read_csv(audit_root / "contact_sheet_manifest.csv")
    summary = pd.read_csv(audit_root / "scene_audit_summary.csv")
    control = pd.read_csv(audit_root / "control_heavy_scene_design_options.csv")

    for df in [matrix, worklist, low_view, contacts]:
        for column in df.columns:
            if df[column].dtype == object:
                df[column] = df[column].fillna("")

    contact_map = {(r.scene, r.point_name): r.contact_sheet for r in contacts.itertuples(index=False)}

    low_current = low_view[
        low_view["has_current_annotation"].map(parse_bool)
        & (pd.to_numeric(low_view["annotation_side_usable_count"], errors="coerce") <= 3)
    ].copy()
    low_current["contact_sheet"] = [
        contact_map.get((r.scene, r.point_name), "") for r in low_current.itertuples(index=False)
    ]
    contact_local_dir = out_root / "contact_sheets_priority"
    contact_local_dir.mkdir(parents=True, exist_ok=True)
    local_contact_sheets: list[str] = []
    for row in low_current.itertuples(index=False):
        source = Path(str(row.contact_sheet))
        if source.exists():
            target = contact_local_dir / source.name
            shutil.copy2(source, target)
            local_contact_sheets.append(target.relative_to(out_root).as_posix())
        else:
            local_contact_sheets.append("")
    low_current["local_contact_sheet"] = local_contact_sheets

    current_annotations = matrix[matrix["candidate_source"].eq("current_annotation")].copy()
    current_annotations = current_annotations[
        [
            "scene",
            "point_name",
            "image_name",
            "candidate_x",
            "candidate_y",
            "recommended_action",
            "candidate_source",
            "already_annotated",
            "visibility_classification",
        ]
    ].rename(columns={"candidate_x": "raw_x", "candidate_y": "raw_y"})

    worklist_cn = worklist.copy()
    worklist_cn["candidate_source_explained_zh"] = worklist_cn["candidate_source"].map(SOURCE_CN).fillna(
        worklist_cn["candidate_source"]
    )
    worklist_cn["recommended_action_explained_zh"] = worklist_cn["recommended_action"].map(ACTION_CN).fillna(
        worklist_cn["recommended_action"]
    )

    full_matrix_cn = matrix.copy()
    full_matrix_cn["candidate_source_explained_zh"] = full_matrix_cn["candidate_source"].map(SOURCE_CN).fillna(
        full_matrix_cn["candidate_source"]
    )
    full_matrix_cn["recommended_action_explained_zh"] = full_matrix_cn["recommended_action"].map(ACTION_CN).fillna(
        full_matrix_cn["recommended_action"]
    )
    full_matrix_cn["is_previous_human_annotation"] = full_matrix_cn["candidate_source"].eq("current_annotation")

    write_csv(low_current, out_root / "low_view_priority_points.csv")
    write_csv(current_annotations, out_root / "current_annotations_previous_labels.csv")
    write_csv(worklist_cn, out_root / "candidates_to_label_or_review.csv")
    write_csv(full_matrix_cn, out_root / "all_candidates_with_previous_labels.csv")

    point_template = low_current[
        [
            "scene",
            "point_name",
            "current_v1_2_2_role",
            "annotation_side_usable_count",
            "potential_usable_or_review_views",
            "low_view_cause_primary",
            "future_formal_primary_disposition",
        ]
    ].copy()
    point_template.insert(0, "feedback_level", "point")
    point_template["your_decision"] = ""
    point_template["reason"] = ""
    point_template["need_codex_open_annotation_tool"] = ""
    point_template["notes"] = ""
    write_csv(point_template, out_root / "feedback_template_point_level.csv")

    image_template = worklist_cn[
        [
            "scene",
            "point_name",
            "image_name",
            "candidate_source",
            "recommended_action",
            "raw_candidate_x",
            "raw_candidate_y",
            "projection_uncertainty_px",
            "current_annotation_count",
            "image_path",
        ]
    ].copy()
    image_template.insert(0, "feedback_level", "image")
    image_template["your_decision"] = ""
    image_template["visibility"] = ""
    image_template["approx_correct_x_optional"] = ""
    image_template["approx_correct_y_optional"] = ""
    image_template["quality"] = ""
    image_template["notes"] = ""
    write_csv(image_template, out_root / "feedback_template_image_level_all.csv")

    focus_keys = set(zip(low_current["scene"], low_current["point_name"]))
    focus_mask = [(scene, point) in focus_keys for scene, point in zip(image_template["scene"], image_template["point_name"])]
    write_csv(image_template[focus_mask].copy(), out_root / "feedback_template_image_level_low_view_first.csv")

    cards = []
    for row in low_current.sort_values(["scene", "point_name"]).itertuples(index=False):
        sheet = getattr(row, "local_contact_sheet", "") or getattr(row, "contact_sheet", "")
        rows = matrix[(matrix["scene"].eq(row.scene)) & (matrix["point_name"].eq(row.point_name))].copy()
        rows["类别"] = rows["candidate_source"].map(SOURCE_CN).fillna(rows["candidate_source"])
        rows["动作"] = rows["recommended_action"].map(ACTION_CN).fillna(rows["recommended_action"])
        old_count = int((rows["candidate_source"] == "current_annotation").sum())
        label_count = int((rows["recommended_action"] == "label").sum())
        review_count = int((rows["recommended_action"] == "review").sum())
        uri = esc(sheet) if str(sheet).startswith("contact_sheets_priority/") else file_uri(sheet)
        cards.append(
            f"""
<section class="card" id="{esc(row.scene)}_{esc(row.point_name)}">
  <h3>{esc(row.scene)} / {esc(row.point_name)}</h3>
  <p><b>当前角色：</b>{esc(row.current_v1_2_2_role)}；
  <b>当前可用视图：</b>{esc(row.annotation_side_usable_count)}；
  <b>潜在可补标/复核视图：</b>{esc(row.potential_usable_or_review_views)}；
  <b>初步原因：</b>{esc(row.low_view_cause_primary)}</p>
  <p><b>读图提醒：</b><code>source=current_annotation</code> 是旧标注；
  <code>action=label/review</code> 是候选预测，准心不一定在真实像控点上。</p>
  <p>旧标注 {old_count} 张；建议补标 {label_count} 张；仅复核 {review_count} 张。</p>
  <p><a href="{uri}" target="_blank">打开该点 contact sheet 原图</a></p>
  <img src="{uri}" loading="lazy" alt="{esc(row.scene)} {esc(row.point_name)} contact sheet">
  <details><summary>展开该点逐图候选明细</summary>
  {table_html(rows[["image_name", "类别", "动作", "candidate_x", "candidate_y", "projection_uncertainty_px", "visibility_classification", "reject_reason"]].rename(columns={
      "image_name": "图像",
      "candidate_x": "预测raw_x",
      "candidate_y": "预测raw_y",
      "projection_uncertainty_px": "不确定性px",
      "visibility_classification": "可见性初判",
      "reject_reason": "拒绝原因",
  }), ["图像", "类别", "动作", "预测raw_x", "预测raw_y", "不确定性px", "可见性初判", "拒绝原因"])}
  </details>
</section>
"""
        )

    counts_by_action = worklist_cn.groupby(["scene", "recommended_action"]).size().reset_index(name="count")
    counts_by_action["recommended_action_explained_zh"] = counts_by_action["recommended_action"].map(ACTION_CN).fillna(
        counts_by_action["recommended_action"]
    )

    nav = "".join(
        f'<a href="#{esc(r.scene)}_{esc(r.point_name)}">{esc(r.scene)} / {esc(r.point_name)}</a> '
        for r in low_current.sort_values(["scene", "point_name"]).itertuples(index=False)
    )

    html_text = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>GS-GCP 多视角补标目视核查入口</title>
<style>
body {{ font-family: "Microsoft YaHei", Arial, sans-serif; line-height:1.56; margin:24px; color:#222; }}
h1,h2,h3 {{ color:#12355b; }}
.notice {{ background:#fff8db; border-left:6px solid #e0a800; padding:12px 16px; margin:16px 0; }}
.ok {{ background:#eaf7ed; border-left:6px solid #2e7d32; padding:12px 16px; margin:16px 0; }}
.warn {{ background:#fdecea; border-left:6px solid #c62828; padding:12px 16px; margin:16px 0; }}
.card {{ border:1px solid #ddd; border-radius:8px; padding:16px; margin:24px 0; box-shadow:0 1px 3px rgba(0,0,0,.08); }}
.card img {{ max-width:100%; border:1px solid #ccc; }}
table {{ border-collapse:collapse; font-size:13px; width:100%; margin:12px 0; }}
th,td {{ border:1px solid #ddd; padding:5px 7px; vertical-align:top; }}
th {{ background:#f3f5f7; }}
code {{ background:#f1f1f1; padding:1px 4px; border-radius:3px; }}
.small {{ color:#666; font-size:13px; }}
.nav a {{ display:inline-block; margin:4px 8px 4px 0; }}
</style></head><body>
<h1>GS-GCP 多视角补标目视核查入口（中文）</h1>
<div class="warn"><b>先看这里：</b>这个 HTML 不是标注工具。图上的准心对 <code>label/review</code> 候选只是预测/搜索位置，不是像控点真值。只有 <code>source=current_annotation</code> + <code>action=keep_existing</code> 是你之前已经标过的正式标注。</div>
<div class="notice"><b>你要做的判断：</b>对每个低视图点，判断原始图像里是否还有清晰、可唯一定位、具有不同观察方向的视图可以补标。如果候选准心不在点上但图里能看见点，也可以反馈“可补标，但需人工重新点”。</div>

<h2>1. 四类图像/点的含义</h2>
<table><tr><th>页面/字段</th><th>含义</th><th>你应该怎么处理</th></tr>
<tr><td><code>source=current_annotation</code></td><td>你之前已经标过的点</td><td>只核查是否明显标错；若无问题不用重标</td></tr>
<tr><td><code>action=label</code></td><td>建议补标候选</td><td>如果图中像控点真实清晰可见，反馈“可补标”</td></tr>
<tr><td><code>action=review</code></td><td>仅复核候选</td><td>判断是否值得补标；不确定就写“不确定”</td></tr>
<tr><td><code>action=reject</code></td><td>脚本认为大概率不可用</td><td>一般不用看，除非你怀疑漏掉了清晰视图</td></tr>
</table>

<h2>2. 建议检查顺序</h2>
<ol>
<li>先看下面“低视图重点点”。这是当前最影响 control-heavy 设计的点。</li>
<li>每个点先看 <code>current_annotation</code> 的旧标注是否正常。</li>
<li>再看 <code>label/review</code> 候选：不要相信准心，重点看 crop 和全局图中是否能找到像控点。</li>
<li>如果候选图里点清晰可见，记录 scene、point_name、image_name 和“可补标”。</li>
<li>如果看不见、被遮挡、模糊或无法唯一定位，记录原因。</li>
</ol>

<h2>3. 反馈格式</h2>
<div class="ok">最简单：你可以直接把下面这种文本发给我。也可以填写本目录下的 CSV 模板。</div>
<pre>点位反馈：
scene,point_name,decision,reason
例如：gcp_5000_20260602,G18,可补标,0001/0014/0015清晰，0002被树挡住
例如：gcp_5000_20260602,G13,排除formal primary,只有1张清晰图，其余都看不见或无法唯一定位

逐图反馈：
scene,point_name,image_name,decision,visibility,quality,notes
例如：gcp_5000_20260602,G18,DJI_20260602172627_0001_D.JPG,可补标,清晰可见,good,准心偏到树上但点在路面接缝处
例如：gcp_20000_20260602,wy3_1,DJI_20260602180616_0062_D.JPG,不可用,边缘/不可见,bad,点位超出画面或无法确认</pre>
<p>可填写的模板：</p>
<ul>
<li><a href="{(out_root / 'feedback_template_point_level.csv').resolve().as_uri()}">feedback_template_point_level.csv</a></li>
<li><a href="{(out_root / 'feedback_template_image_level_low_view_first.csv').resolve().as_uri()}">feedback_template_image_level_low_view_first.csv</a></li>
<li><a href="{(out_root / 'feedback_template_image_level_all.csv').resolve().as_uri()}">feedback_template_image_level_all.csv（完整候选）</a></li>
</ul>

<h2>4. 场景汇总</h2>
{table_html(summary, list(summary.columns))}

<h2>5. 低视图重点点（优先看）</h2>
<div class="nav">{nav}</div>
{''.join(cards)}

<h2>6. 补标/复核数量概览</h2>
{table_html(counts_by_action, ["scene", "recommended_action", "recommended_action_explained_zh", "count"])}

<h2>7. control-heavy 设计概览</h2>
{table_html(control, list(control.columns))}
<p class="small">生成时间：{args.stamp}。本页只用于目视核查和补标计划，不修改 v1.2.2 release，不产生正式评价结果。</p>
</body></html>"""

    (out_root / "index_zh.html").write_text(html_text, encoding="utf-8")
    (out_root / "README_review_zh.md").write_text(
        """# GS-GCP 多视角补标目视核查说明

打开 `index_zh.html`。

重点：
- `source=current_annotation` 是你之前标过的正式标注。
- `action=label/review` 是候选预测，不是已经标注，也不是精确真值。
- 准心偏离像控点时，不代表旧标注错；请判断图中是否能清晰看见像控点。

反馈可直接发文字，也可填写 CSV 模板：

点位反馈：
scene,point_name,decision,reason

逐图反馈：
scene,point_name,image_name,decision,visibility,quality,notes
""",
        encoding="utf-8",
    )

    manifest = []
    for path in sorted(out_root.rglob("*")):
        if path.is_file():
            manifest.append(
                {
                    "path": path.relative_to(out_root).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    (out_root / "manifest_sha256.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    zip_path = out_root.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(out_root.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(out_root).as_posix())
    sha = sha256_file(zip_path)
    zip_path.with_suffix(zip_path.suffix + ".sha256").write_text(f"{sha}  {zip_path.name}\n", encoding="ascii")

    print(
        json.dumps(
            {
                "output_root": str(out_root),
                "html": str(out_root / "index_zh.html"),
                "zip": str(zip_path),
                "sha256": sha,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
