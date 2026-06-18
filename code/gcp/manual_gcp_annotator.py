from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import re
import statistics
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageTk

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


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def sort_candidates(candidates: List[Dict[str, str]]) -> List[Dict[str, str]]:
    def key(row: Dict[str, str]) -> tuple[str, str, int, str]:
        try:
            rank = int(float(row.get("rank_for_gcp") or 9999))
        except ValueError:
            rank = 9999
        return (row.get("scene", ""), row.get("point_name", ""), rank, row.get("image_name", ""))

    return sorted(candidates, key=key)


class Annotator:
    def __init__(
        self,
        root: tk.Tk,
        candidates: List[Dict[str, str]],
        out_csv: Path,
        crop_size: int,
        display_size: int,
        annotator: str,
        candidates_csv: Optional[Path] = None,
        image_root: Optional[Path] = None,
        point_name_filter: Optional[str] = None,
        max_rows: int = 0,
    ):
        self.root = root
        self.candidates = candidates
        self.candidates_csv = candidates_csv
        self.out_csv = out_csv
        self.default_out_dir = out_csv.parent
        self.image_root = image_root
        self.point_name_filter = point_name_filter
        self.max_rows = int(max_rows)
        self.crop_size = int(crop_size)
        self.default_canvas_size = int(display_size)
        self.display_size = self.default_canvas_size
        self.view_zoom = 1.0
        self.min_view_zoom = 1.0
        self.max_view_zoom = 12.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.drag_last = None
        self.current_crop = None
        self.current_candidate_xy = None
        self.current_corrected_xy = None
        self.current_correction_info = ""
        self.current_manual_crop_xy = None
        self.render_scale = self.default_canvas_size / self.crop_size
        self.annotator = annotator
        self.idx = 0
        self.annotations: Dict[tuple[str, str, str], Dict[str, str]] = {}
        self.root.title("GCP manual annotator")
        self.candidates_var = tk.StringVar(value=str(candidates_csv or ""))
        self.out_csv_var = tk.StringVar(value=str(out_csv))
        self.image_root_var = tk.StringVar(value=str(image_root or ""))
        self.zoom_var = tk.StringVar(value=f"{self.view_zoom:.2f}")
        self.updating_listbox = False
        self.build_path_controls(root)
        self.load_annotations()
        self.info = ttk.Label(root, text="", font=("Arial", 11))
        self.info.pack(fill=tk.X, padx=8, pady=4)
        btns = ttk.Frame(root)
        btns.pack(fill=tk.X, padx=8, pady=4)
        for text, cmd in [
            ("1 Good", lambda: self.mark("1", "good", "1.0")),
            ("2 Ambiguous", lambda: self.mark("1", "ambiguous", "0.5")),
            ("3 Not visible", lambda: self.mark("0", "not_visible", "0.0")),
            ("4 Prev", self.prev_item),
            ("5 Next", self.next_item),
            ("6 Save", self.save),
        ]:
            ttk.Button(btns, text=text, command=cmd).pack(side=tk.LEFT, padx=3)
        ttk.Label(btns, text="Zoom").pack(side=tk.LEFT, padx=(12, 3))
        zoom_entry = ttk.Entry(btns, textvariable=self.zoom_var, width=7)
        zoom_entry.pack(side=tk.LEFT, padx=3)
        zoom_entry.bind("<Return>", lambda e: self.apply_zoom_entry())
        zoom_entry.bind("<FocusOut>", lambda e: self.apply_zoom_entry())
        self.note_var = tk.StringVar()
        ttk.Entry(root, textvariable=self.note_var).pack(fill=tk.X, padx=8, pady=4)
        self.status = ttk.Label(root, text="", font=("Arial", 10))
        self.status.pack(fill=tk.X, padx=8, pady=4)
        body = ttk.Panedwindow(root, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        canvas_frame = ttk.Frame(body)
        list_frame = ttk.Frame(body)
        body.add(canvas_frame, weight=5)
        body.add(list_frame, weight=2)
        self.canvas = tk.Canvas(canvas_frame, width=self.default_canvas_size, height=self.default_canvas_size, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=(0, 6), pady=4)
        ttk.Label(list_frame, text="Candidate list").pack(anchor=tk.W)
        list_container = ttk.Frame(list_frame)
        list_container.pack(fill=tk.BOTH, expand=True)
        self.listbox = tk.Listbox(list_container, width=48, height=34, exportselection=False)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        list_scroll = ttk.Scrollbar(list_container, orient=tk.VERTICAL, command=self.listbox.yview)
        list_scroll.pack(side=tk.LEFT, fill=tk.Y)
        self.listbox.configure(yscrollcommand=list_scroll.set)
        self.listbox.bind("<<ListboxSelect>>", self.on_listbox_select)
        self.listbox.bind("<Left>", lambda e: self.run_shortcut(e, lambda: self.nudge_manual(-0.1, 0.0)))
        self.listbox.bind("<Right>", lambda e: self.run_shortcut(e, lambda: self.nudge_manual(0.1, 0.0)))
        self.listbox.bind("<Up>", lambda e: self.run_shortcut(e, lambda: self.nudge_manual(0.0, -0.1)))
        self.listbox.bind("<Down>", lambda e: self.run_shortcut(e, lambda: self.nudge_manual(0.0, 0.1)))
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<ButtonPress-3>", self.on_right_press)
        self.canvas.bind("<B3-Motion>", self.on_right_drag)
        self.canvas.bind("<ButtonRelease-3>", self.on_right_release)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        root.bind("1", lambda e: self.run_shortcut(e, lambda: self.mark("1", "good", "1.0")))
        root.bind("2", lambda e: self.run_shortcut(e, lambda: self.mark("1", "ambiguous", "0.5")))
        root.bind("3", lambda e: self.run_shortcut(e, lambda: self.mark("0", "not_visible", "0.0")))
        root.bind("4", lambda e: self.run_shortcut(e, self.prev_item))
        root.bind("5", lambda e: self.run_shortcut(e, self.next_item))
        root.bind("6", lambda e: self.run_shortcut(e, self.save))
        root.bind("<Left>", lambda e: self.run_shortcut(e, lambda: self.nudge_manual(-0.1, 0.0)))
        root.bind("<Right>", lambda e: self.run_shortcut(e, lambda: self.nudge_manual(0.1, 0.0)))
        root.bind("<Up>", lambda e: self.run_shortcut(e, lambda: self.nudge_manual(0.0, -0.1)))
        root.bind("<Down>", lambda e: self.run_shortcut(e, lambda: self.nudge_manual(0.0, 0.1)))
        root.bind("q", lambda e: self.quit())
        root.bind("u", lambda e: self.run_shortcut(e, self.clear_current_status))
        root.bind("+", lambda e: self.zoom_in())
        root.bind("=", lambda e: self.zoom_in())
        root.bind("-", lambda e: self.zoom_out())
        root.bind("0", lambda e: self.zoom_reset())
        self.photo = None
        self.crop_origin = (0, 0)
        self.scale = 1.0
        self.current_manual = None
        self.show_item()

    def run_shortcut(self, event, action):
        widget_class = event.widget.winfo_class()
        if widget_class in {"Entry", "TEntry", "Text"}:
            return None
        action()
        return "break"

    def build_path_controls(self, root: tk.Tk) -> None:
        frame = ttk.LabelFrame(root, text="Scene / file switching")
        frame.pack(fill=tk.X, padx=8, pady=6)

        ttk.Label(frame, text="Candidates").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(frame, textvariable=self.candidates_var).grid(row=0, column=1, sticky="ew", padx=4, pady=2)
        ttk.Button(frame, text="Browse...", command=self.browse_candidates).grid(row=0, column=2, padx=4, pady=2)

        ttk.Label(frame, text="Output CSV").grid(row=1, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(frame, textvariable=self.out_csv_var).grid(row=1, column=1, sticky="ew", padx=4, pady=2)
        ttk.Button(frame, text="Set...", command=self.browse_output_csv).grid(row=1, column=2, padx=4, pady=2)

        ttk.Label(frame, text="Image root").grid(row=2, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(frame, textvariable=self.image_root_var).grid(row=2, column=1, sticky="ew", padx=4, pady=2)
        ttk.Button(frame, text="Set...", command=self.browse_image_root).grid(row=2, column=2, padx=4, pady=2)

        ttk.Button(frame, text="Reload current", command=self.reload_from_entries).grid(row=3, column=1, sticky="e", padx=4, pady=3)
        frame.columnconfigure(1, weight=1)

    def key(self, cand: Dict[str, str]) -> tuple[str, str, str]:
        return (cand["scene"], cand["point_name"], cand["image_name"])

    def load_annotations(self) -> None:
        self.annotations = {}
        if self.out_csv.exists():
            for row in read_csv(self.out_csv):
                self.annotations[(row["scene"], row["point_name"], row["image_name"])] = row

    def load_candidates_file(self, path: Path) -> List[Dict[str, str]]:
        candidates = read_csv(path)
        if self.point_name_filter:
            candidates = [r for r in candidates if r.get("point_name") == self.point_name_filter]
        candidates = sort_candidates(candidates)
        if self.max_rows and self.max_rows > 0:
            candidates = candidates[: self.max_rows]
        return candidates

    def infer_default_output_csv(self, candidates: List[Dict[str, str]]) -> Path:
        scenes = sorted({r.get("scene", "") for r in candidates if r.get("scene")})
        if len(scenes) == 1:
            return self.default_out_dir / f"{scenes[0]}_manual_annotations.csv"
        return self.out_csv

    def browse_candidates(self) -> None:
        initial_dir = str((self.candidates_csv or REPO_ROOT).parent if self.candidates_csv else REPO_ROOT / "outputs")
        selected = filedialog.askopenfilename(
            title="Select GCP candidate CSV",
            initialdir=initial_dir,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if selected:
            self.switch_candidates(Path(selected), update_output=True)

    def browse_output_csv(self) -> None:
        selected = filedialog.asksaveasfilename(
            title="Select annotation output CSV",
            initialdir=str(self.out_csv.parent),
            initialfile=self.out_csv.name,
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if selected:
            self.save()
            self.out_csv = Path(selected)
            self.out_csv_var.set(str(self.out_csv))
            self.default_out_dir = self.out_csv.parent
            self.load_annotations()
            self.show_item()

    def browse_image_root(self) -> None:
        initial_dir = str(self.image_root or REPO_ROOT)
        selected = filedialog.askdirectory(title="Select image root override", initialdir=initial_dir)
        if selected:
            self.image_root = Path(selected)
            self.image_root_var.set(str(self.image_root))
            self.reset_view()
            self.show_item()

    def reload_from_entries(self) -> None:
        candidates_path = Path(self.candidates_var.get())
        if not candidates_path.exists():
            messagebox.showerror("Missing candidates", f"Candidate CSV does not exist:\n{candidates_path}")
            return
        image_root_text = self.image_root_var.get().strip()
        self.image_root = Path(image_root_text) if image_root_text else None
        out_csv_text = self.out_csv_var.get().strip()
        if out_csv_text:
            self.out_csv = Path(out_csv_text)
            self.default_out_dir = self.out_csv.parent
        self.switch_candidates(candidates_path, update_output=False)

    def switch_candidates(self, path: Path, update_output: bool) -> None:
        self.save()
        try:
            candidates = self.load_candidates_file(path)
        except Exception as exc:  # pragma: no cover - GUI feedback path
            messagebox.showerror("Load failed", f"Could not load candidates:\n{path}\n\n{exc}")
            return
        if not candidates:
            messagebox.showwarning("No candidates", f"No candidates loaded from:\n{path}")
            return
        self.candidates = candidates
        self.candidates_csv = path
        self.candidates_var.set(str(path))
        if update_output:
            self.out_csv = self.infer_default_output_csv(candidates)
            self.out_csv_var.set(str(self.out_csv))
        self.load_annotations()
        self.idx = 0
        self.reset_view()
        self.show_item()
        scenes = ", ".join(sorted({r.get("scene", "") for r in candidates if r.get("scene")}))
        self.status.configure(text=f"Loaded {len(candidates)} candidates for {scenes}; output={self.out_csv}")

    def resolve_image_path(self, cand: Dict[str, str]) -> Path:
        raw = Path(cand["image_path"])
        if raw.exists():
            return raw
        image_name = cand.get("image_name") or raw.name
        if self.image_root:
            candidates = [
                self.image_root / image_name,
                self.image_root / raw.name,
                self.image_root / cand.get("scene", "") / image_name,
            ]
            for p in candidates:
                if p.exists():
                    return p
        return raw

    def saved_residuals(
        self,
        scene: str,
        image_name: Optional[str] = None,
        point_name: Optional[str] = None,
        exclude_key: Optional[tuple[str, str, str]] = None,
    ) -> List[Tuple[float, float]]:
        residuals: List[Tuple[float, float]] = []
        for key, row in self.annotations.items():
            if exclude_key and key == exclude_key:
                continue
            if row.get("scene") != scene:
                continue
            if image_name is not None and row.get("image_name") != image_name:
                continue
            if point_name is not None and row.get("point_name") != point_name:
                continue
            try:
                mx = float(row.get("manual_x") or "nan")
                my = float(row.get("manual_y") or "nan")
                px = float(row.get("projected_x") or "nan")
                py = float(row.get("projected_y") or "nan")
            except ValueError:
                continue
            if row.get("visible") != "1" or not all(v == v for v in [mx, my, px, py]):
                continue
            residuals.append((mx - px, my - py))
        return residuals

    def image_sequence(self, image_name: str) -> Optional[int]:
        match = re.search(r"_(\d{4})_D\.JPG$", image_name, re.IGNORECASE)
        if not match:
            return None
        return int(match.group(1))

    def saved_residual_records(
        self,
        scene: str,
        exclude_key: Optional[tuple[str, str, str]] = None,
    ) -> List[Dict[str, float | str | int | None]]:
        records: List[Dict[str, float | str | int | None]] = []
        for key, row in self.annotations.items():
            if exclude_key and key == exclude_key:
                continue
            if row.get("scene") != scene:
                continue
            try:
                mx = float(row.get("manual_x") or "nan")
                my = float(row.get("manual_y") or "nan")
                px = float(row.get("projected_x") or "nan")
                py = float(row.get("projected_y") or "nan")
            except ValueError:
                continue
            if row.get("visible") != "1" or not all(v == v for v in [mx, my, px, py]):
                continue
            dx = mx - px
            dy = my - py
            records.append(
                {
                    "dx": dx,
                    "dy": dy,
                    "projected_x": px,
                    "projected_y": py,
                    "image_name": row.get("image_name", ""),
                    "point_name": row.get("point_name", ""),
                    "seq": self.image_sequence(row.get("image_name", "")),
                    "norm": math.hypot(dx, dy),
                }
            )
        return records

    def median_residual(self, residuals: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        if not residuals:
            return None
        return (statistics.median([r[0] for r in residuals]), statistics.median([r[1] for r in residuals]))

    def robust_filter_records(self, records: List[Dict[str, float | str | int | None]]) -> List[Dict[str, float | str | int | None]]:
        if len(records) < 6:
            return records
        norms = [float(r["norm"]) for r in records]
        med = statistics.median(norms)
        mad = statistics.median([abs(v - med) for v in norms])
        if mad < 1e-6:
            return records
        threshold = med + 3.5 * 1.4826 * mad
        kept = [r for r in records if float(r["norm"]) <= threshold]
        return kept if len(kept) >= 4 else records

    def weighted_scene_correction(self, cand: Dict[str, str]) -> tuple[Optional[Tuple[float, float]], str]:
        records = self.robust_filter_records(self.saved_residual_records(cand["scene"], exclude_key=self.key(cand)))
        if len(records) < 4:
            return None, f"weighted scene model needs >=4 residuals, has n={len(records)}"
        px = float(cand["pixel_x"])
        py = float(cand["pixel_y"])
        seq = self.image_sequence(cand["image_name"])
        scored = []
        for r in records:
            spatial = math.hypot(float(r["projected_x"]) - px, float(r["projected_y"]) - py)
            rseq = r.get("seq")
            seq_dist = abs(int(rseq) - seq) if seq is not None and rseq is not None else 0
            score = spatial / 900.0 + seq_dist / 30.0
            scored.append((score, r))
        scored.sort(key=lambda x: x[0])
        nearest = scored[: min(30, len(scored))]
        weights = [1.0 / ((1.0 + score) ** 2) for score, _ in nearest]
        total_w = sum(weights)
        if total_w <= 0:
            return None, "weighted scene model has zero weight"
        dx = sum(w * float(r["dx"]) for w, (_, r) in zip(weights, nearest)) / total_w
        dy = sum(w * float(r["dy"]) for w, (_, r) in zip(weights, nearest)) / total_w
        return (dx, dy), f"weighted all-history residual model k={len(nearest)}/n={len(records)}"

    def correction_for_candidate(self, cand: Dict[str, str]) -> tuple[Optional[Tuple[float, float]], str]:
        scene = cand["scene"]
        image_name = cand["image_name"]
        point_name = cand["point_name"]
        exclude = self.key(cand)
        same_image = self.saved_residuals(scene, image_name=image_name, exclude_key=exclude)
        med = self.median_residual(same_image)
        if med is not None:
            return med, f"same-image median residual from n={len(same_image)}"
        weighted, info = self.weighted_scene_correction(cand)
        if weighted is not None:
            return weighted, info
        same_point = self.saved_residuals(scene, point_name=point_name, exclude_key=exclude)
        med = self.median_residual(same_point)
        if med is not None:
            return med, f"same-point median residual from n={len(same_point)}"
        same_scene = self.saved_residuals(scene, exclude_key=exclude)
        med = self.median_residual(same_scene)
        if med is not None:
            return med, f"same-scene median residual from n={len(same_scene)}"
        return None, "no correction history"

    def show_item(self) -> None:
        if not self.candidates:
            self.info.configure(text="No candidates.")
            return
        cand = self.candidates[self.idx]
        image_path = self.resolve_image_path(cand)
        if not image_path.exists():
            messagebox.showerror("Missing image", f"Image does not exist:\n{image_path}\n\nUse Image root to point to the scene folder.")
            img = Image.new("RGB", (self.crop_size, self.crop_size), "black")
        else:
            img = Image.open(image_path).convert("RGB")
        px = float(cand["pixel_x"])
        py = float(cand["pixel_y"])
        correction, correction_info = self.correction_for_candidate(cand)
        ann = self.annotations.get(self.key(cand))
        center_x, center_y = px, py
        if ann and ann.get("manual_x") and ann.get("manual_y"):
            center_x = float(ann["manual_x"])
            center_y = float(ann["manual_y"])
        elif correction is not None:
            center_x = px + correction[0]
            center_y = py + correction[1]
        half = self.crop_size // 2
        left = int(round(center_x - half))
        top = int(round(center_y - half))
        crop = Image.new("RGB", (self.crop_size, self.crop_size), "black")
        src_box = (
            max(0, left),
            max(0, top),
            min(img.width, left + self.crop_size),
            min(img.height, top + self.crop_size),
        )
        paste_xy = (max(0, -left), max(0, -top))
        if src_box[2] > src_box[0] and src_box[3] > src_box[1]:
            crop.paste(img.crop(src_box), paste_xy)
        self.crop_origin = (left, top)
        self.current_crop = crop
        self.current_candidate_xy = (px - left, py - top)
        self.current_corrected_xy = None
        self.current_correction_info = correction_info
        if correction is not None:
            dx, dy = correction
            self.current_corrected_xy = (px + dx - left, py + dy - top)
        self.current_manual = None
        self.current_manual_crop_xy = None
        self.note_var.set("")
        if ann and ann.get("manual_x") and ann.get("manual_y"):
            mx = float(ann["manual_x"]) - left
            my = float(ann["manual_y"]) - top
            self.current_manual = (float(ann["manual_x"]), float(ann["manual_y"]))
            self.current_manual_crop_xy = (mx, my)
            self.note_var.set(ann.get("note", ""))
        self.render_current_view(center_active=True)
        self.update_listbox()
        self.info.configure(
            text=(
                f"{self.idx+1}/{len(self.candidates)}  {cand['scene']}  {cand['point_name']}  "
                f"{cand['image_name']}  rank={cand.get('rank_for_gcp','')} score={cand.get('center_score','')}"
            )
        )
        old = self.annotations.get(self.key(cand))
        if old:
            residual_text = self.residual_status_text(cand)
            self.status.configure(
                text=(
                    f"Saved: visible={old.get('visible')} quality={old.get('quality')}  "
                    f"{residual_text}  zoom={self.view_zoom:.2f}x"
                )
            )
        else:
            self.status.configure(
                text=(
                    "Yellow = coarse projection. Magenta = corrected hint. Cyan = manual mark. "
                    "Click true GCP center, then press 1/2/3/4/5/6. Arrow keys nudge manual mark by 0.1 px."
                )
            )

    def draw_cross(
        self,
        draw: ImageDraw.ImageDraw,
        x: float,
        y: float,
        color: tuple[int, int, int],
        arm: int = 11,
        width: int = 3,
    ) -> None:
        draw.line([(x - arm, y), (x + arm, y)], fill=color, width=width)
        draw.line([(x, y - arm), (x, y + arm)], fill=color, width=width)

    def active_crop_xy(self) -> Optional[Tuple[float, float]]:
        if self.current_manual_crop_xy is not None:
            return self.current_manual_crop_xy
        if self.current_corrected_xy is not None:
            return self.current_corrected_xy
        if self.current_candidate_xy is not None:
            return self.current_candidate_xy
        return None

    def canvas_view_size(self) -> tuple[int, int]:
        width = max(1, int(self.canvas.winfo_width() or self.default_canvas_size))
        height = max(1, int(self.canvas.winfo_height() or self.default_canvas_size))
        return width, height

    def canvas_fit_side(self) -> int:
        width, height = self.canvas_view_size()
        return max(1, min(width, height))

    def center_pan_on_active(self, render_size: int) -> None:
        active = self.active_crop_xy()
        if active is None:
            self.clamp_pan(render_size)
            return
        canvas_w, canvas_h = self.canvas_view_size()
        ax, ay = active
        self.pan_x = canvas_w / 2 - ax * self.render_scale
        self.pan_y = canvas_h / 2 - ay * self.render_scale
        self.clamp_pan(render_size)

    def render_current_view(self, center_active: bool = False) -> None:
        if self.current_crop is None:
            return
        canvas_w, canvas_h = self.canvas_view_size()
        render_size = max(1, int(round(self.canvas_fit_side() * self.view_zoom)))
        self.render_scale = render_size / self.crop_size
        rendered = self.current_crop.resize((render_size, render_size), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(rendered)
        if self.current_candidate_xy is not None:
            cx, cy = self.current_candidate_xy
            self.draw_cross(draw, cx * self.render_scale, cy * self.render_scale, (255, 230, 0), arm=10, width=2)
        if self.current_corrected_xy is not None:
            px, py = self.current_corrected_xy
            self.draw_cross(draw, px * self.render_scale, py * self.render_scale, (255, 0, 255), arm=9, width=2)
        if self.current_manual_crop_xy is not None:
            mx, my = self.current_manual_crop_xy
            self.draw_cross(draw, mx * self.render_scale, my * self.render_scale, (0, 255, 255))
        if center_active:
            self.center_pan_on_active(render_size)
        else:
            self.clamp_pan(render_size)
        viewport = Image.new("RGB", (canvas_w, canvas_h), "black")
        pan_x = int(round(self.pan_x))
        pan_y = int(round(self.pan_y))
        src_x = max(0, -pan_x)
        src_y = max(0, -pan_y)
        dst_x = max(0, pan_x)
        dst_y = max(0, pan_y)
        visible_w = min(canvas_w - dst_x, render_size - src_x)
        visible_h = min(canvas_h - dst_y, render_size - src_y)
        if visible_w > 0 and visible_h > 0:
            viewport.paste(rendered.crop((src_x, src_y, src_x + visible_w, src_y + visible_h)), (dst_x, dst_y))
        self.photo = ImageTk.PhotoImage(viewport)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def clamp_pan(self, render_size: int | None = None) -> None:
        if render_size is None:
            render_size = max(1, int(round(self.canvas_fit_side() * self.view_zoom)))
        canvas_w, canvas_h = self.canvas_view_size()
        if render_size <= canvas_w:
            self.pan_x = (canvas_w - render_size) / 2
        else:
            min_pan_x = canvas_w - render_size
            self.pan_x = min(0.0, max(float(min_pan_x), self.pan_x))
        if render_size <= canvas_h:
            self.pan_y = (canvas_h - render_size) / 2
        else:
            min_pan_y = canvas_h - render_size
            self.pan_y = min(0.0, max(float(min_pan_y), self.pan_y))
        return

    def on_click(self, event) -> None:
        left, top = self.crop_origin
        crop_x = (event.x - self.pan_x) / self.render_scale
        crop_y = (event.y - self.pan_y) / self.render_scale
        if crop_x < 0 or crop_y < 0 or crop_x >= self.crop_size or crop_y >= self.crop_size:
            self.status.configure(text="Click is outside the image crop; right-drag or zoom to reposition the crop.")
            return
        x = left + crop_x
        y = top + crop_y
        self.set_manual_point(x, y)

    def set_manual_point(self, x: float, y: float) -> None:
        cand = self.candidates[self.idx]
        left, top = self.crop_origin
        self.current_manual = (x, y)
        existing = self.annotations.get(self.key(cand))
        row = existing or self.base_annotation(cand)
        row["manual_x"] = f"{x:.3f}"
        row["manual_y"] = f"{y:.3f}"
        if existing is None or row.get("visible") == "0":
            row["visible"] = ""
            row["quality"] = ""
            row["confidence"] = ""
        row["annotator"] = self.annotator
        row["note"] = self.note_var.get()
        row["updated_at"] = dt.datetime.now().isoformat(timespec="seconds")
        self.annotations[self.key(cand)] = row
        self.current_manual_crop_xy = (x - left, y - top)
        self.render_current_view(center_active=False)
        self.update_listbox()
        self.status.configure(text=self.residual_status_text(cand))

    def nudge_manual(self, dx: float, dy: float) -> None:
        cand = self.candidates[self.idx]
        if self.current_manual is not None:
            x, y = self.current_manual
        else:
            active = self.active_crop_xy() or self.current_candidate_xy
            if active is None:
                return
            x = self.crop_origin[0] + active[0]
            y = self.crop_origin[1] + active[1]
        self.set_manual_point(x + dx, y + dy)

    def residual_status_text(self, cand: Dict[str, str]) -> str:
        row = self.annotations.get(self.key(cand))
        if not row or not row.get("manual_x") or not row.get("manual_y"):
            return self.current_correction_info
        try:
            mx = float(row["manual_x"])
            my = float(row["manual_y"])
            px = float(row["projected_x"])
            py = float(row["projected_y"])
        except ValueError:
            return self.current_correction_info
        dx = mx - px
        dy = my - py
        norm = (dx * dx + dy * dy) ** 0.5
        corrected = ""
        correction, info = self.correction_for_candidate(cand)
        if correction is not None:
            cdx = dx - correction[0]
            cdy = dy - correction[1]
            cnorm = (cdx * cdx + cdy * cdy) ** 0.5
            corrected = f"; vs corrected hint residual={cnorm:.1f}px"
        return f"manual-coarse residual=({dx:+.1f},{dy:+.1f})px norm={norm:.1f}px{corrected}; correction={info}"

    def base_annotation(self, cand: Dict[str, str]) -> Dict[str, str]:
        image_path = self.resolve_image_path(cand)
        return {
            "schema": "m3m_gcp_manual_image_observation_v1",
            "scene": cand["scene"],
            "point_name": cand["point_name"],
            "image_name": cand["image_name"],
            "image_path": str(image_path),
            "rank_for_gcp": cand.get("rank_for_gcp", ""),
            "candidate_score": cand.get("center_score", ""),
            "projected_x": f"{float(cand['pixel_x']):.3f}",
            "projected_y": f"{float(cand['pixel_y']):.3f}",
            "manual_x": "",
            "manual_y": "",
            "visible": "",
            "quality": "",
            "confidence": "",
            "annotator": self.annotator,
            "note": "",
            "updated_at": "",
        }

    def mark(self, visible: str, quality: str, confidence: str) -> None:
        cand = self.candidates[self.idx]
        row = self.annotations.get(self.key(cand), self.base_annotation(cand))
        row["visible"] = visible
        row["quality"] = quality
        row["confidence"] = confidence
        row["annotator"] = self.annotator
        row["note"] = self.note_var.get()
        row["updated_at"] = dt.datetime.now().isoformat(timespec="seconds")
        self.annotations[self.key(cand)] = row
        self.save()
        self.next_item()

    def clear_current_status(self) -> None:
        if not self.candidates:
            return
        cand = self.candidates[self.idx]
        row = self.annotations.get(self.key(cand), self.base_annotation(cand))
        row["visible"] = ""
        row["quality"] = ""
        row["confidence"] = ""
        row["annotator"] = self.annotator
        row["note"] = self.note_var.get()
        row["updated_at"] = dt.datetime.now().isoformat(timespec="seconds")
        self.annotations[self.key(cand)] = row
        self.update_listbox()
        self.status.configure(text="Cleared current status to U/unselected; manual point is preserved if present.")

    def save(self) -> None:
        rows = [self.annotations[k] for k in sorted(self.annotations)]
        write_csv(self.out_csv, rows, ANNOTATION_FIELDS)
        self.status.configure(text=f"Saved {len(rows)} annotations to {self.out_csv}")

    def candidate_status(self, cand: Dict[str, str]) -> tuple[str, str, str]:
        ann = self.annotations.get(self.key(cand))
        if not ann:
            return "B", "blank", "#f4f4f4"
        if ann.get("visible") == "1":
            quality = ann.get("quality")
            if quality == "good":
                return "G", "good", "#dff2df"
            if quality == "ambiguous":
                return "A", "ambiguous", "#fff2bf"
            return "U", "unselected", "#e5ecff"
        if ann.get("visible") == "0":
            return "NV", "not_visible", "#eeeeee"
        return "U", "unselected", "#e5ecff"

    def update_listbox(self) -> None:
        if not hasattr(self, "listbox"):
            return
        self.updating_listbox = True
        self.listbox.delete(0, tk.END)
        for i, cand in enumerate(self.candidates):
            marker, status, bg = self.candidate_status(cand)
            label = (
                f"{i+1:03d} {marker:<2} {cand.get('point_name','')} "
                f"r{cand.get('rank_for_gcp','')} {cand.get('image_name','')}"
            )
            self.listbox.insert(tk.END, label)
            self.listbox.itemconfig(i, background=bg)
        self.listbox.selection_clear(0, tk.END)
        if self.candidates:
            self.listbox.selection_set(self.idx)
            self.listbox.activate(self.idx)
            self.listbox.see(self.idx)
        self.updating_listbox = False

    def on_listbox_select(self, event) -> None:
        if self.updating_listbox:
            return
        selection = self.listbox.curselection()
        if not selection:
            return
        idx = int(selection[0])
        if idx == self.idx:
            return
        self.idx = max(0, min(len(self.candidates) - 1, idx))
        self.drag_last = None
        self.show_item()

    def next_item(self) -> None:
        self.idx = min(len(self.candidates) - 1, self.idx + 1)
        self.drag_last = None
        self.show_item()

    def prev_item(self) -> None:
        self.idx = max(0, self.idx - 1)
        self.drag_last = None
        self.show_item()

    def reset_view(self) -> None:
        self.view_zoom = 1.0
        self.zoom_var.set(f"{self.view_zoom:.2f}")
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.drag_last = None

    def zoom_at(self, factor: float, anchor_x: float | None = None, anchor_y: float | None = None) -> None:
        if self.current_crop is None:
            return
        if anchor_x is None or anchor_y is None:
            canvas_w, canvas_h = self.canvas_view_size()
            anchor_x = canvas_w / 2
            anchor_y = canvas_h / 2
        crop_x = (anchor_x - self.pan_x) / self.render_scale
        crop_y = (anchor_y - self.pan_y) / self.render_scale
        old_zoom = self.view_zoom
        self.view_zoom = min(self.max_view_zoom, max(self.min_view_zoom, self.view_zoom * factor))
        if abs(self.view_zoom - old_zoom) < 1e-9:
            return
        self.zoom_var.set(f"{self.view_zoom:.2f}")
        render_size = max(1, int(round(self.canvas_fit_side() * self.view_zoom)))
        new_scale = render_size / self.crop_size
        self.pan_x = anchor_x - crop_x * new_scale
        self.pan_y = anchor_y - crop_y * new_scale
        self.render_current_view(center_active=False)
        self.status.configure(
            text=(
                f"Image zoom={self.view_zoom:.2f}x. Mouse wheel/+/- zoom image, right-drag pan, 0 reset."
            )
        )

    def zoom_in(self) -> None:
        self.zoom_at(1.25)

    def zoom_out(self) -> None:
        self.zoom_at(1 / 1.25)

    def zoom_reset(self) -> None:
        self.reset_view()
        self.render_current_view(center_active=True)
        self.status.configure(text="Image zoom reset. Mouse wheel/+/- zoom image, right-drag pan, 0 reset.")

    def apply_zoom_entry(self) -> None:
        try:
            zoom = float(self.zoom_var.get())
        except ValueError:
            self.zoom_var.set(f"{self.view_zoom:.2f}")
            return
        self.view_zoom = min(self.max_view_zoom, max(self.min_view_zoom, zoom))
        self.zoom_var.set(f"{self.view_zoom:.2f}")
        self.render_current_view(center_active=True)

    def on_canvas_configure(self, event) -> None:
        if self.current_crop is not None:
            self.render_current_view(center_active=False)

    def on_mousewheel(self, event) -> None:
        anchor_x = event.x if event.widget is self.canvas else None
        anchor_y = event.y if event.widget is self.canvas else None
        if event.delta > 0:
            self.zoom_at(1.25, anchor_x, anchor_y)
        elif event.delta < 0:
            self.zoom_at(1 / 1.25, anchor_x, anchor_y)
        return "break"

    def on_right_press(self, event) -> None:
        self.drag_last = (event.x, event.y)

    def on_right_drag(self, event) -> None:
        if self.drag_last is None:
            return
        last_x, last_y = self.drag_last
        self.pan_x += event.x - last_x
        self.pan_y += event.y - last_y
        self.drag_last = (event.x, event.y)
        self.clamp_pan()
        self.render_current_view()

    def on_right_release(self, event) -> None:
        self.drag_last = None

    def quit(self) -> None:
        self.save()
        self.root.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual GCP annotator for projected candidate crops.")
    parser.add_argument("--candidates_csv", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--point_name", default=None, help="Optional point-name filter.")
    parser.add_argument("--max_rows", type=int, default=0, help="Optional limit for quick sessions.")
    parser.add_argument("--crop_size", type=int, default=720)
    parser.add_argument("--display_size", type=int, default=860)
    parser.add_argument("--annotator", default="user", help="Annotator id written to the output CSV.")
    parser.add_argument("--image_root", default="", help="Optional root used to resolve image_name when candidate image_path is stale.")
    args = parser.parse_args()
    candidates_path = Path(args.candidates_csv)
    candidates = read_csv(candidates_path)
    if args.point_name:
        candidates = [r for r in candidates if r.get("point_name") == args.point_name]
    candidates = sort_candidates(candidates)
    if args.max_rows and args.max_rows > 0:
        candidates = candidates[: args.max_rows]
    root = tk.Tk()
    Annotator(
        root,
        candidates,
        Path(args.out_csv),
        crop_size=args.crop_size,
        display_size=args.display_size,
        annotator=args.annotator,
        candidates_csv=candidates_path,
        image_root=Path(args.image_root) if args.image_root else None,
        point_name_filter=args.point_name,
        max_rows=args.max_rows,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
