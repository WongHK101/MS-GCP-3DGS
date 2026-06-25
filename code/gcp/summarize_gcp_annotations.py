from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


EVALUATION_FIELDS = [
    "scene",
    "point_name",
    "image_name",
    "image_path",
    "u_px",
    "v_px",
    "projected_x",
    "projected_y",
    "confidence",
    "quality",
    "annotator",
    "source_annotation_csv",
    "updated_at",
    "note",
]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def usable_observation(row: Dict[str, str]) -> bool:
    if str(row.get("visible", "")).strip() not in {"1", "true", "True", "yes", "Y"}:
        return False
    quality = str(row.get("quality", "")).strip()
    if quality in {"not_visible", "reject", "rejected"}:
        return False
    if quality and quality != "good":
        return False
    return bool(str(row.get("manual_x", "")).strip() and str(row.get("manual_y", "")).strip())


def normalize_eval_row(row: Dict[str, str], source: Path) -> Dict[str, str]:
    return {
        "scene": row.get("scene", ""),
        "point_name": row.get("point_name", ""),
        "image_name": row.get("image_name", ""),
        "image_path": row.get("image_path", ""),
        "u_px": row.get("manual_x", ""),
        "v_px": row.get("manual_y", ""),
        "projected_x": row.get("projected_x", ""),
        "projected_y": row.get("projected_y", ""),
        "confidence": row.get("confidence", ""),
        "quality": row.get("quality", ""),
        "annotator": row.get("annotator", ""),
        "source_annotation_csv": str(source),
        "updated_at": row.get("updated_at", ""),
        "note": row.get("note", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize manual GCP image annotations and export evaluation-ready 2D observations."
    )
    parser.add_argument("--annotation_csv", action="append", required=True, help="Manual annotation CSV. Repeatable.")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    annotation_paths = [Path(p) for p in args.annotation_csv]
    out_dir = Path(args.out_dir)
    all_rows: List[Dict[str, str]] = []
    eval_rows: List[Dict[str, str]] = []
    for path in annotation_paths:
        rows = read_csv(path)
        for row in rows:
            row["_source_annotation_csv"] = str(path)
        all_rows.extend(rows)
        eval_rows.extend(normalize_eval_row(row, path) for row in rows if usable_observation(row))

    per_scene: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "scene": "",
            "annotation_rows": 0,
            "usable_observations": 0,
            "points_with_usable_observations": set(),
            "images_with_usable_observations": set(),
            "good": 0,
            "ambiguous": 0,
            "not_visible": 0,
        }
    )
    for row in all_rows:
        scene = row.get("scene", "")
        item = per_scene[scene]
        item["scene"] = scene
        item["annotation_rows"] += 1
        quality = row.get("quality", "")
        if quality == "good":
            item["good"] += 1
        elif quality == "ambiguous":
            item["ambiguous"] += 1
        elif quality == "not_visible":
            item["not_visible"] += 1
        if usable_observation(row):
            item["usable_observations"] += 1
            item["points_with_usable_observations"].add(row.get("point_name", ""))
            item["images_with_usable_observations"].add(row.get("image_name", ""))

    summary_rows: List[Dict[str, Any]] = []
    for scene in sorted(per_scene):
        item = per_scene[scene]
        summary_rows.append(
            {
                "scene": scene,
                "annotation_rows": item["annotation_rows"],
                "usable_observations": item["usable_observations"],
                "points_with_usable_observations": len(item["points_with_usable_observations"]),
                "images_with_usable_observations": len(item["images_with_usable_observations"]),
                "good": item["good"],
                "ambiguous": item["ambiguous"],
                "not_visible": item["not_visible"],
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "gcp_image_observations_for_evaluation.csv", eval_rows, EVALUATION_FIELDS)
    write_csv(
        out_dir / "gcp_annotation_summary_by_scene.csv",
        summary_rows,
        [
            "scene",
            "annotation_rows",
            "usable_observations",
            "points_with_usable_observations",
            "images_with_usable_observations",
            "good",
            "ambiguous",
            "not_visible",
        ],
    )
    manifest = {
        "schema": "m3m_gcp_annotation_summary_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "annotation_csv": [str(p) for p in annotation_paths],
        "annotation_rows": len(all_rows),
        "usable_observations": len(eval_rows),
        "scene_count": len(summary_rows),
        "outputs": {
            "evaluation_observations": str(out_dir / "gcp_image_observations_for_evaluation.csv"),
            "scene_summary": str(out_dir / "gcp_annotation_summary_by_scene.csv"),
        },
        "notes": [
            "This file only packages manual 2D GCP observations.",
            "It does not perform 3D reconstruction, bundle adjustment, or GCP residual evaluation.",
            "Rows with visible=1, manual_x/manual_y, and quality=good are exported as evaluation-ready observations.",
        ],
    }
    (out_dir / "gcp_annotation_summary_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    md = [
        "# GCP Annotation Summary",
        "",
        f"- Annotation CSV files: {len(annotation_paths)}",
        f"- Total annotation rows: {len(all_rows)}",
        f"- Evaluation-ready observations: {len(eval_rows)}",
        "",
        "## Per-scene summary",
        "",
        "| Scene | Rows | Usable observations | Points | Images | Good | Ambiguous | Not visible |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        md.append(
            "| {scene} | {annotation_rows} | {usable_observations} | "
            "{points_with_usable_observations} | {images_with_usable_observations} | "
            "{good} | {ambiguous} | {not_visible} |".format(**row)
        )
    (out_dir / "gcp_annotation_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
