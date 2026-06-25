from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]

ANNOTATION_FIELDS = [
    "schema",
    "scene",
    "point_name",
    "image_name",
    "image_path",
    "rank_for_gcp",
    "candidate_score",
    "projected_x",
    "projected_y",
    "manual_x",
    "manual_y",
    "visible",
    "quality",
    "confidence",
    "annotator",
    "note",
    "updated_at",
]


DEFAULT_SCENES = [
    "gcp_3000_20260602",
    "gcp_5000_20260602",
    "gcp_10000_20260610",
    "gcp_20000_20260602",
    "gcp_50000_20260610",
    "gcp_100000_20260610",
]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_float(value: object, default: float | None = None) -> float | None:
    try:
        text = str(value).strip()
        if text == "":
            return default
        return float(text)
    except Exception:
        return default


def row_visible(row: Dict[str, str]) -> bool:
    return str(row.get("visible", "1")).strip() in {"", "1", "true", "True", "yes", "Y"}


def row_has_pixel(row: Dict[str, str]) -> bool:
    return str(row.get("manual_x", "")).strip() != "" and str(row.get("manual_y", "")).strip() != ""


def row_quality(row: Dict[str, str]) -> str:
    return str(row.get("quality", "")).strip() or "unknown"


def row_is_good(row: Dict[str, str]) -> bool:
    return row_visible(row) and row_has_pixel(row) and row_quality(row) == "good"


def resolve_annotation_csv(scene: str, official_dir: Path, rework_dir: Path) -> Path:
    official = official_dir / f"{scene}_manual_annotations.csv"
    if official.exists():
        return official
    rework = rework_dir / f"{scene}_manual_annotations.csv"
    if rework.exists():
        return rework
    raise FileNotFoundError(f"No annotation CSV for {scene}: {official} or {rework}")


def image_exists(row: Dict[str, str], dataset_root: Path) -> bool:
    candidates = []
    raw = str(row.get("image_path", "")).strip()
    if raw:
        candidates.append(Path(raw))
        candidates.append(dataset_root / raw)
    name = str(row.get("image_name", "")).strip()
    scene = str(row.get("scene", "")).strip()
    if name and scene:
        candidates.append(dataset_root / "scenes" / scene / Path(name).name)
    return any(p.exists() for p in candidates)


def load_off_nadir(scene: str, metadata_root: Path) -> Dict[str, float]:
    path = metadata_root / scene / "image_metadata.csv"
    out: Dict[str, float] = {}
    if not path.exists():
        return out
    for row in read_csv(path):
        name = Path(str(row.get("image_name", row.get("SourceFile", ""))).strip()).name
        if not name:
            continue
        off = parse_float(row.get("off_nadir_deg"))
        if off is None:
            pitch = parse_float(row.get("pitch_deg", row.get("GimbalPitchDegree")))
            if pitch is not None:
                off = abs(pitch + 90.0)
        if off is not None:
            out[name] = float(off)
    return out


def classify_point(
    point_rows: Sequence[Dict[str, str]],
    off_nadir_by_image: Dict[str, float],
    dataset_root: Path,
    nadir_threshold_deg: float,
    min_good_observations: int,
    min_good_nadir_observations: int,
) -> Dict[str, object]:
    quality_counts = Counter(row_quality(r) for r in point_rows)
    visible_rows = [r for r in point_rows if row_visible(r)]
    good_rows = [r for r in point_rows if row_is_good(r)]
    existing_good_rows = [r for r in good_rows if image_exists(r, dataset_root)]
    missing_image_rows = [r for r in good_rows if not image_exists(r, dataset_root)]
    good_nadir_rows = []
    good_oblique_rows = []
    unknown_pose_good_rows = []
    for row in existing_good_rows:
        name = Path(row.get("image_name", "")).name
        off = off_nadir_by_image.get(name)
        if off is None:
            unknown_pose_good_rows.append(row)
        elif off <= nadir_threshold_deg:
            good_nadir_rows.append(row)
        else:
            good_oblique_rows.append(row)

    reasons: List[str] = []
    if not good_rows:
        reasons.append("no_good_observation")
    if not existing_good_rows:
        reasons.append("no_existing_good_observation")
    if len(good_nadir_rows) < min_good_nadir_observations:
        reasons.append("no_good_nadir_coverage")
    if len(existing_good_rows) < min_good_observations:
        reasons.append("too_few_good_observations")

    include = not reasons
    return {
        "include_for_eval": include,
        "exclude_reason": "pass" if include else ";".join(reasons),
        "row_count": len(point_rows),
        "visible_rows": len(visible_rows),
        "good_rows": len(good_rows),
        "ambiguous_rows": int(quality_counts.get("ambiguous", 0)),
        "not_visible_rows": int(quality_counts.get("not_visible", 0)),
        "other_quality_rows": sum(v for k, v in quality_counts.items() if k not in {"good", "ambiguous", "not_visible"}),
        "existing_good_rows": len(existing_good_rows),
        "missing_good_image_rows": len(missing_image_rows),
        "good_nadir_rows": len(good_nadir_rows),
        "good_oblique_rows": len(good_oblique_rows),
        "unknown_pose_good_rows": len(unknown_pose_good_rows),
    }


def build_audit(args: argparse.Namespace) -> None:
    repo = Path(args.repo_root)
    official_dir = Path(args.official_dir)
    rework_dir = Path(args.rework_dir)
    dataset_root = Path(args.dataset_root)
    metadata_root = Path(args.metadata_root)
    out_root = Path(args.out_root)
    filtered_dir = out_root / "filtered_annotations"
    out_root.mkdir(parents=True, exist_ok=True)

    point_decisions: List[Dict[str, object]] = []
    scene_summary: List[Dict[str, object]] = []
    excluded_rows: List[Dict[str, object]] = []
    missing_rows: List[Dict[str, object]] = []
    manifest = {
        "schema": "ms_gcp_3dgs_annotation_inclusion_audit_v1",
        "nadir_threshold_deg": float(args.nadir_threshold_deg),
        "min_good_observations": int(args.min_good_observations),
        "min_good_nadir_observations": int(args.min_good_nadir_observations),
        "observation_policy": (
            "Only visible quality=good rows are exported. Point-level inclusion additionally "
            "requires at least one good near-nadir observation and enough existing good observations."
        ),
        "scenes": {},
    }

    for scene in args.scenes:
        ann_path = resolve_annotation_csv(scene, official_dir, rework_dir)
        rows = read_csv(ann_path)
        off_nadir_by_image = load_off_nadir(scene, metadata_root)
        by_point: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        for row in rows:
            by_point[row.get("point_name", "")].append(row)

        included_points = set()
        scene_point_decisions = []
        for point, point_rows in sorted(by_point.items()):
            decision = classify_point(
                point_rows,
                off_nadir_by_image=off_nadir_by_image,
                dataset_root=dataset_root,
                nadir_threshold_deg=float(args.nadir_threshold_deg),
                min_good_observations=int(args.min_good_observations),
                min_good_nadir_observations=int(args.min_good_nadir_observations),
            )
            decision_row = {
                "scene": scene,
                "point_name": point,
                "source_annotation_csv": str(ann_path),
                **decision,
            }
            point_decisions.append(decision_row)
            scene_point_decisions.append(decision_row)
            if decision["include_for_eval"]:
                included_points.add(point)

        filtered_rows = []
        for row in rows:
            point = row.get("point_name", "")
            base = {
                "scene": scene,
                "point_name": point,
                "image_name": row.get("image_name", ""),
                "quality": row.get("quality", ""),
                "visible": row.get("visible", ""),
                "confidence": row.get("confidence", ""),
            }
            if not image_exists(row, dataset_root):
                missing_rows.append({**base, "image_path": row.get("image_path", ""), "source_annotation_csv": str(ann_path)})
            if point in included_points and row_is_good(row) and image_exists(row, dataset_root):
                filtered_rows.append(row)
            else:
                reason = "point_excluded"
                if point in included_points and not row_is_good(row):
                    reason = "observation_not_good"
                elif point in included_points and not image_exists(row, dataset_root):
                    reason = "image_missing"
                excluded_rows.append({**base, "exclude_row_reason": reason, "source_annotation_csv": str(ann_path)})

        out_csv = filtered_dir / f"{scene}_manual_annotations_eval_strict_good_nadir.csv"
        fields = list(rows[0].keys()) if rows else ANNOTATION_FIELDS
        write_csv(out_csv, filtered_rows, fields)

        total_points = len(by_point)
        included_point_count = len(included_points)
        good_points_no_nadir = [
            d["point_name"]
            for d in scene_point_decisions
            if "no_good_nadir_coverage" in str(d["exclude_reason"]) and int(d["good_rows"]) > 0
        ]
        no_good_points = [
            d["point_name"]
            for d in scene_point_decisions
            if "no_good_observation" in str(d["exclude_reason"])
        ]
        summary = {
            "scene": scene,
            "source_annotation_csv": str(ann_path),
            "input_rows": len(rows),
            "input_points": total_points,
            "included_points": included_point_count,
            "excluded_points": total_points - included_point_count,
            "filtered_rows": len(filtered_rows),
            "points_good_but_no_nadir": ",".join(good_points_no_nadir),
            "points_no_good_observation": ",".join(no_good_points),
            "missing_image_rows": sum(1 for r in rows if not image_exists(r, dataset_root)),
            "filtered_annotation_csv": str(out_csv),
        }
        scene_summary.append(summary)
        manifest["scenes"][scene] = summary

    point_fields = [
        "scene",
        "point_name",
        "include_for_eval",
        "exclude_reason",
        "row_count",
        "visible_rows",
        "good_rows",
        "ambiguous_rows",
        "not_visible_rows",
        "other_quality_rows",
        "existing_good_rows",
        "missing_good_image_rows",
        "good_nadir_rows",
        "good_oblique_rows",
        "unknown_pose_good_rows",
        "source_annotation_csv",
    ]
    summary_fields = [
        "scene",
        "source_annotation_csv",
        "input_rows",
        "input_points",
        "included_points",
        "excluded_points",
        "filtered_rows",
        "points_good_but_no_nadir",
        "points_no_good_observation",
        "missing_image_rows",
        "filtered_annotation_csv",
    ]
    write_csv(out_root / "point_inclusion_decisions.csv", point_decisions, point_fields)
    write_csv(out_root / "scene_inclusion_summary.csv", scene_summary, summary_fields)
    write_csv(
        out_root / "excluded_annotation_rows.csv",
        excluded_rows,
        ["scene", "point_name", "image_name", "quality", "visible", "confidence", "exclude_row_reason", "source_annotation_csv"],
    )
    write_csv(
        out_root / "missing_image_rows.csv",
        missing_rows,
        ["scene", "point_name", "image_name", "image_path", "quality", "visible", "confidence", "source_annotation_csv"],
    )
    (out_root / "annotation_inclusion_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# GCP Annotation Inclusion Audit",
        "",
        "This audit builds strict evaluation-ready annotation CSV files without modifying the original manual annotation tables.",
        "",
        f"- Near-nadir threshold: off-nadir <= {float(args.nadir_threshold_deg):.1f} deg.",
        f"- Point inclusion: >= {int(args.min_good_observations)} existing `quality=good` observations and >= {int(args.min_good_nadir_observations)} existing good near-nadir observation.",
        "- Exported observations: only visible rows with `quality=good`, valid manual pixels, existing source image, and included point.",
        "- `ambiguous` rows are excluded from evaluation exports.",
        "",
        "## Scene Summary",
        "",
        "| Scene | Input points | Included | Excluded | Filtered obs. | Good but no nadir | No good obs. | Missing-image rows |",
        "|---|---:|---:|---:|---:|---|---|---:|",
    ]
    for row in scene_summary:
        lines.append(
            "| {scene} | {input_points} | {included_points} | {excluded_points} | {filtered_rows} | {points_good_but_no_nadir} | {points_no_good_observation} | {missing_image_rows} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `point_inclusion_decisions.csv`: point-level include/exclude decisions.",
            "- `scene_inclusion_summary.csv`: scene-level counts.",
            "- `filtered_annotations/*_manual_annotations_eval_strict_good_nadir.csv`: evaluation-ready annotation tables.",
            "- `missing_image_rows.csv`: annotation rows whose image file is absent locally.",
        ]
    )
    (out_root / "ANNOTATION_INCLUSION_AUDIT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit GCP manual annotation inclusion for strict evaluation.")
    parser.add_argument("--repo_root", default=str(REPO_ROOT))
    parser.add_argument("--dataset_root", default=r"E:\datasets\M3M-GCP")
    parser.add_argument(
        "--official_dir",
        default=str(REPO_ROOT / "gcp_manual_annotations_official_3scenes_20260624"),
    )
    parser.add_argument(
        "--rework_dir",
        default=str(REPO_ROOT / "gcp_manual_annotations_all6_provisional_pending_outsource_rework_20260623"),
    )
    parser.add_argument(
        "--metadata_root",
        default=str(REPO_ROOT / "outputs" / "gcp_annotation_candidates_20260617_all"),
    )
    parser.add_argument("--out_root", default=str(REPO_ROOT / "outputs" / "gcp_annotation_inclusion_audit_20260625"))
    parser.add_argument("--scenes", nargs="*", default=DEFAULT_SCENES)
    parser.add_argument("--nadir_threshold_deg", type=float, default=5.0)
    parser.add_argument("--min_good_observations", type=int, default=3)
    parser.add_argument("--min_good_nadir_observations", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    build_audit(parse_args())
