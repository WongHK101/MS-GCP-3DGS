#!/usr/bin/env python3
"""Portable launcher, data verifier, and minimal-return packer for TGS-GCP."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_DIR = PACKAGE_ROOT / "candidate_lists"
RESULT_DIR = PACKAGE_ROOT / "results"
RETURN_DIR = PACKAGE_ROOT / "return"
SETTINGS_PATH = PACKAGE_ROOT / ".local_settings.json"

MINIMAL_FIELDS = [
    "schema",
    "task_id",
    "scene",
    "point_name",
    "image_name",
    "manual_x",
    "manual_y",
    "visible",
    "quality",
    "confidence",
    "annotator",
    "note",
    "updated_at",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])


def candidate_files() -> list[Path]:
    paths = sorted(CANDIDATE_DIR.glob("*_rgb_candidates.csv"), key=lambda path: path.name)
    if len(paths) != 6:
        raise RuntimeError(f"Expected six candidate files, found {len(paths)}")
    return paths


def load_settings() -> dict[str, str]:
    if SETTINGS_PATH.exists():
        try:
            return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def save_settings(values: dict[str, str]) -> None:
    SETTINGS_PATH.write_text(json.dumps(values, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def choose_image_root(initial: str = "") -> Path | None:
    root = tk.Tk()
    root.withdraw()
    selected = filedialog.askdirectory(title="选择 TGS-GCP 根目录", initialdir=initial or str(Path.home()))
    root.destroy()
    return Path(selected) if selected else None


def validate_image_root(image_root: Path, check_hashes: bool = True) -> dict[str, Any]:
    tasks = [row for path in candidate_files() for row in read_csv(path)]
    unique: dict[str, dict[str, str]] = {}
    for row in tasks:
        key = row["image_path"].replace("\\", "/")
        previous = unique.get(key)
        if previous and previous["source_image_sha256"] != row["source_image_sha256"]:
            raise RuntimeError(f"Conflicting candidate image identity: {key}")
        unique[key] = row
    failures: list[str] = []
    checked = 0
    for relative, row in sorted(unique.items()):
        path = image_root.joinpath(*relative.split("/"))
        if not path.exists():
            failures.append(f"missing: {relative}")
            continue
        try:
            with Image.open(path) as image:
                size = image.size
        except OSError as exc:
            failures.append(f"decode_failed: {relative}: {exc}")
            continue
        expected = (int(row["source_image_width"]), int(row["source_image_height"]))
        if size != expected:
            failures.append(f"dimension_mismatch: {relative}: {size} != {expected}")
            continue
        if check_hashes and sha256_file(path).lower() != row["source_image_sha256"].lower():
            failures.append(f"sha256_mismatch: {relative}")
            continue
        checked += 1
    report = {
        "schema": "gs_gcp_tgs_rgb_local_data_validation_v1",
        "image_root": str(image_root),
        "candidate_task_count": len(tasks),
        "unique_candidate_image_count": len(unique),
        "checked_pass_count": checked,
        "failure_count": len(failures),
        "failures": failures,
        "hashes_checked": check_hashes,
        "status": "pass" if not failures else "fail",
    }
    return report


def verify_command(args: argparse.Namespace) -> int:
    settings = load_settings()
    image_root = Path(args.image_root) if args.image_root else choose_image_root(settings.get("image_root", ""))
    if image_root is None:
        return 1
    report = validate_image_root(image_root, check_hashes=not args.skip_hashes)
    report_path = PACKAGE_ROOT / "local_data_validation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if report["status"] == "pass":
        save_settings({**settings, "image_root": str(image_root)})
        messagebox.showinfo(
            "校验通过",
            f"已校验 {report['checked_pass_count']} 张任务 RGB 图像。\n报告：{report_path}",
        )
        return 0
    messagebox.showerror(
        "校验失败",
        f"发现 {report['failure_count']} 个问题。请查看：\n{report_path}",
    )
    return 2


class Launcher:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("TGS-GCP RGB 标注任务")
        settings = load_settings()
        self.image_root = tk.StringVar(value=settings.get("image_root", ""))
        self.annotator = tk.StringVar(value=settings.get("annotator", "outsourcer"))
        self.build()

    def build(self) -> None:
        frame = ttk.Frame(self.root, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frame, text="TGS-GCP 根目录").grid(row=0, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.image_root, width=70).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(frame, text="选择...", command=self.browse).grid(row=0, column=2)
        ttk.Label(frame, text="标注人员代号").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(frame, textvariable=self.annotator, width=24).grid(row=1, column=1, sticky="w", padx=6, pady=(8, 0))
        ttk.Label(
            frame,
            text="只标 RGB。黄色准心是粗搜索提示；必须核对地面手写点名。",
            foreground="#8b0000",
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=12)
        for index, candidate_path in enumerate(candidate_files(), start=3):
            scene = candidate_path.name.removesuffix("_rgb_candidates.csv")
            count = len(read_csv(candidate_path))
            ttk.Button(
                frame,
                text=f"打开 {scene}（{count} 条）",
                command=lambda path=candidate_path, scene_name=scene: self.open_scene(path, scene_name),
                width=42,
            ).grid(row=index, column=0, columnspan=3, sticky="ew", pady=3)
        ttk.Button(frame, text="关闭", command=self.root.destroy).grid(row=10, column=2, sticky="e", pady=(12, 0))
        frame.columnconfigure(1, weight=1)

    def browse(self) -> None:
        selected = filedialog.askdirectory(title="选择 TGS-GCP 根目录", initialdir=self.image_root.get() or str(Path.home()))
        if selected:
            self.image_root.set(selected)

    def open_scene(self, candidate_path: Path, scene: str) -> None:
        image_root = Path(self.image_root.get().strip())
        annotator = self.annotator.get().strip()
        if not image_root.exists():
            messagebox.showerror("目录不存在", "请先选择正确的 TGS-GCP 根目录。")
            return
        if not annotator:
            messagebox.showerror("缺少代号", "请填写标注人员代号。")
            return
        settings = {"image_root": str(image_root), "annotator": annotator}
        save_settings(settings)
        RESULT_DIR.mkdir(exist_ok=True)
        output_path = RESULT_DIR / f"{scene}_manual_annotations.csv"
        command = [
            sys.executable,
            str(Path(__file__).with_name("manual_gcp_annotator.py")),
            "--candidates_csv",
            str(candidate_path),
            "--out_csv",
            str(output_path),
            "--image_root",
            str(image_root),
            "--annotator",
            annotator,
            "--history_hint_mode",
            "off",
            "--crop_size",
            "900",
            "--display_size",
            "900",
        ]
        subprocess.Popen(command, cwd=PACKAGE_ROOT)

    def run(self) -> int:
        self.root.mainloop()
        return 0


def validate_result(candidate_rows: list[dict[str, str]], result_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    expected = {(row["scene"], row["point_name"], row["image_name"]): row for row in candidate_rows}
    actual: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in result_rows:
        key = (row.get("scene", ""), row.get("point_name", ""), row.get("image_name", ""))
        if key in actual:
            raise RuntimeError(f"Duplicate annotation row: {key}")
        actual[key] = row
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise RuntimeError(f"Result key mismatch: missing={missing[:5]} ({len(missing)}), extra={extra[:5]} ({len(extra)})")
    minimal: list[dict[str, str]] = []
    for key in sorted(expected):
        candidate = expected[key]
        row = actual[key]
        for field in ["task_id", "source_image_sha256", "source_image_width", "source_image_height"]:
            if row.get(field, "").lower() != candidate.get(field, "").lower():
                raise RuntimeError(f"{key}: {field} differs from the frozen task")
        visible = row.get("visible", "")
        quality = row.get("quality", "")
        x_text, y_text = row.get("manual_x", ""), row.get("manual_y", "")
        if visible == "1" and quality in {"good", "ambiguous"}:
            if not x_text or not y_text:
                raise RuntimeError(f"{key}: {quality} requires a manual coordinate")
            x, y = float(x_text), float(y_text)
            if not (0.0 <= x < float(candidate["source_image_width"]) and 0.0 <= y < float(candidate["source_image_height"])):
                raise RuntimeError(f"{key}: manual coordinate is out of bounds")
        elif visible == "0" and quality == "not_visible":
            if x_text or y_text:
                raise RuntimeError(f"{key}: not_visible must not contain a manual coordinate")
        else:
            raise RuntimeError(f"{key}: unselected or invalid status visible={visible!r}, quality={quality!r}")
        minimal.append({field: row.get(field, "") for field in MINIMAL_FIELDS})
    return minimal


def pack_command(args: argparse.Namespace) -> int:
    all_minimal: list[dict[str, str]] = []
    per_scene: dict[str, dict[str, int]] = {}
    for candidate_path in candidate_files():
        scene = candidate_path.name.removesuffix("_rgb_candidates.csv")
        result_path = RESULT_DIR / f"{scene}_manual_annotations.csv"
        if not result_path.exists():
            raise RuntimeError(f"Missing result file: {result_path}")
        minimal = validate_result(read_csv(candidate_path), read_csv(result_path))
        all_minimal.extend(minimal)
        per_scene[scene] = {
            "rows": len(minimal),
            "good": sum(row["quality"] == "good" for row in minimal),
            "ambiguous": sum(row["quality"] == "ambiguous" for row in minimal),
            "not_visible": sum(row["quality"] == "not_visible" for row in minimal),
        }
    RETURN_DIR.mkdir(exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    staging = RETURN_DIR / f"minimal_return_{stamp}"
    staging.mkdir()
    annotation_csv = staging / "tgs_gcp_rgb_manual_annotations_minimal.csv"
    write_csv(annotation_csv, all_minimal, MINIMAL_FIELDS)
    task_manifest = PACKAGE_ROOT / "task_manifest.json"
    manifest = {
        "schema": "gs_gcp_tgs_rgb_minimal_annotation_return_v1",
        "task_manifest_sha256": sha256_file(task_manifest),
        "total_rows": len(all_minimal),
        "unique_task_ids": len({row["task_id"] for row in all_minimal}),
        "scene_summary": per_scene,
        "annotation_csv": annotation_csv.name,
        "annotation_csv_sha256": sha256_file(annotation_csv),
        "contains_images": False,
        "contains_thermal": False,
    }
    manifest_path = staging / "return_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksums = [
        {"path": annotation_csv.name, "size": annotation_csv.stat().st_size, "sha256": sha256_file(annotation_csv)},
        {"path": manifest_path.name, "size": manifest_path.stat().st_size, "sha256": sha256_file(manifest_path)},
    ]
    checksum_path = staging / "RETURN_SHA256SUMS.csv"
    write_csv(checksum_path, checksums, ["path", "size", "sha256"])
    zip_path = RETURN_DIR / f"TGS_GCP_RGB_ANNOTATIONS_MINIMAL_{stamp}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(staging.iterdir(), key=lambda item: item.name.encode("utf-8")):
            archive.write(path, path.name)
    print(f"Minimal return package created: {zip_path}")
    messagebox.showinfo("回传包已生成", f"请只回传这个文件：\n{zip_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--image_root", default="")
    verify.add_argument("--skip_hashes", action="store_true")
    subparsers.add_parser("launch")
    subparsers.add_parser("pack")
    args = parser.parse_args()
    try:
        if args.command == "verify":
            return verify_command(args)
        if args.command == "launch":
            return Launcher().run()
        if args.command == "pack":
            return pack_command(args)
    except Exception as exc:
        messagebox.showerror("操作失败", str(exc))
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    raise AssertionError(args.command)


if __name__ == "__main__":
    sys.exit(main())
