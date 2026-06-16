from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path
import tkinter as tk
from tkinter import ttk
from typing import Dict, List

from PIL import Image, ImageDraw, ImageTk


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


class Annotator:
    def __init__(
        self,
        root: tk.Tk,
        candidates: List[Dict[str, str]],
        out_csv: Path,
        crop_size: int,
        display_size: int,
        annotator: str,
    ):
        self.root = root
        self.candidates = candidates
        self.out_csv = out_csv
        self.crop_size = int(crop_size)
        self.canvas_size = int(display_size)
        self.display_size = self.canvas_size
        self.view_zoom = 1.0
        self.min_view_zoom = 1.0
        self.max_view_zoom = 12.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.drag_last = None
        self.current_crop = None
        self.current_candidate_xy = None
        self.current_manual_crop_xy = None
        self.render_scale = self.canvas_size / self.crop_size
        self.annotator = annotator
        self.idx = 0
        self.annotations: Dict[tuple[str, str, str], Dict[str, str]] = {}
        if out_csv.exists():
            for row in read_csv(out_csv):
                self.annotations[(row["scene"], row["point_name"], row["image_name"])] = row
        self.root.title("GCP manual annotator")
        self.info = ttk.Label(root, text="", font=("Arial", 11))
        self.info.pack(fill=tk.X, padx=8, pady=4)
        self.canvas = tk.Canvas(root, width=self.canvas_size, height=self.canvas_size, bg="black")
        self.canvas.pack(padx=8, pady=4)
        self.status = ttk.Label(root, text="", font=("Arial", 10))
        self.status.pack(fill=tk.X, padx=8, pady=4)
        btns = ttk.Frame(root)
        btns.pack(fill=tk.X, padx=8, pady=4)
        for text, cmd in [
            ("Visible good (v)", lambda: self.mark("1", "good", "1.0")),
            ("Ambiguous (a)", lambda: self.mark("1", "ambiguous", "0.5")),
            ("Not visible (x)", lambda: self.mark("0", "not_visible", "0.0")),
            ("Prev (p)", self.prev_item),
            ("Next (n)", self.next_item),
            ("Save (s)", self.save),
        ]:
            ttk.Button(btns, text=text, command=cmd).pack(side=tk.LEFT, padx=3)
        self.note_var = tk.StringVar()
        ttk.Entry(root, textvariable=self.note_var).pack(fill=tk.X, padx=8, pady=4)
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<ButtonPress-3>", self.on_right_press)
        self.canvas.bind("<B3-Motion>", self.on_right_drag)
        self.canvas.bind("<ButtonRelease-3>", self.on_right_release)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        root.bind("v", lambda e: self.mark("1", "good", "1.0"))
        root.bind("a", lambda e: self.mark("1", "ambiguous", "0.5"))
        root.bind("x", lambda e: self.mark("0", "not_visible", "0.0"))
        root.bind("n", lambda e: self.next_item())
        root.bind("p", lambda e: self.prev_item())
        root.bind("s", lambda e: self.save())
        root.bind("q", lambda e: self.quit())
        root.bind("+", lambda e: self.zoom_in())
        root.bind("=", lambda e: self.zoom_in())
        root.bind("-", lambda e: self.zoom_out())
        root.bind("0", lambda e: self.zoom_reset())
        self.photo = None
        self.crop_origin = (0, 0)
        self.scale = 1.0
        self.current_manual = None
        self.show_item()

    def key(self, cand: Dict[str, str]) -> tuple[str, str, str]:
        return (cand["scene"], cand["point_name"], cand["image_name"])

    def show_item(self) -> None:
        if not self.candidates:
            self.info.configure(text="No candidates.")
            return
        cand = self.candidates[self.idx]
        image_path = Path(cand["image_path"])
        img = Image.open(image_path).convert("RGB")
        px = float(cand["pixel_x"])
        py = float(cand["pixel_y"])
        half = self.crop_size // 2
        left = int(round(px - half))
        top = int(round(py - half))
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
        ann = self.annotations.get(self.key(cand))
        self.current_manual = None
        self.current_manual_crop_xy = None
        self.note_var.set("")
        if ann and ann.get("manual_x") and ann.get("manual_y"):
            mx = float(ann["manual_x"]) - left
            my = float(ann["manual_y"]) - top
            self.current_manual = (float(ann["manual_x"]), float(ann["manual_y"]))
            self.current_manual_crop_xy = (mx, my)
            self.note_var.set(ann.get("note", ""))
        self.render_current_view()
        self.info.configure(
            text=(
                f"{self.idx+1}/{len(self.candidates)}  {cand['scene']}  {cand['point_name']}  "
                f"{cand['image_name']}  rank={cand.get('rank_for_gcp','')} score={cand.get('center_score','')}"
            )
        )
        old = self.annotations.get(self.key(cand))
        if old:
            self.status.configure(
                text=(
                    f"Saved: visible={old.get('visible')} quality={old.get('quality')}  "
                    f"zoom={self.view_zoom:.2f}x  (mouse wheel/+/- zoom image, right-drag pan, 0 reset)"
                )
            )
        else:
            self.status.configure(
                text=(
                    "Yellow cross = coarse projection. Cyan cross = manual mark. "
                    "Click true GCP center, then press v/a/x/n. Mouse wheel/+/- zoom image, right-drag pan, 0 reset."
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

    def render_current_view(self) -> None:
        if self.current_crop is None:
            return
        render_size = max(1, int(round(self.canvas_size * self.view_zoom)))
        self.render_scale = render_size / self.crop_size
        rendered = self.current_crop.resize((render_size, render_size), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(rendered)
        if self.current_candidate_xy is not None:
            cx, cy = self.current_candidate_xy
            self.draw_cross(draw, cx * self.render_scale, cy * self.render_scale, (255, 230, 0))
        if self.current_manual_crop_xy is not None:
            mx, my = self.current_manual_crop_xy
            self.draw_cross(draw, mx * self.render_scale, my * self.render_scale, (0, 255, 255))
        self.clamp_pan(render_size)
        viewport = Image.new("RGB", (self.canvas_size, self.canvas_size), "black")
        pan_x = int(round(self.pan_x))
        pan_y = int(round(self.pan_y))
        src_x = max(0, -pan_x)
        src_y = max(0, -pan_y)
        dst_x = max(0, pan_x)
        dst_y = max(0, pan_y)
        visible_w = min(self.canvas_size - dst_x, render_size - src_x)
        visible_h = min(self.canvas_size - dst_y, render_size - src_y)
        if visible_w > 0 and visible_h > 0:
            viewport.paste(rendered.crop((src_x, src_y, src_x + visible_w, src_y + visible_h)), (dst_x, dst_y))
        self.photo = ImageTk.PhotoImage(viewport)
        self.canvas.configure(width=self.canvas_size, height=self.canvas_size)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def clamp_pan(self, render_size: int | None = None) -> None:
        if render_size is None:
            render_size = max(1, int(round(self.canvas_size * self.view_zoom)))
        if render_size <= self.canvas_size:
            self.pan_x = (self.canvas_size - render_size) / 2
            self.pan_y = (self.canvas_size - render_size) / 2
            return
        min_pan = self.canvas_size - render_size
        self.pan_x = min(0.0, max(float(min_pan), self.pan_x))
        self.pan_y = min(0.0, max(float(min_pan), self.pan_y))

    def on_click(self, event) -> None:
        cand = self.candidates[self.idx]
        left, top = self.crop_origin
        crop_x = (event.x - self.pan_x) / self.render_scale
        crop_y = (event.y - self.pan_y) / self.render_scale
        if crop_x < 0 or crop_y < 0 or crop_x >= self.crop_size or crop_y >= self.crop_size:
            self.status.configure(text="Click is outside the image crop; right-drag or zoom to reposition the crop.")
            return
        x = left + crop_x
        y = top + crop_y
        self.current_manual = (x, y)
        row = self.annotations.get(self.key(cand), self.base_annotation(cand))
        row["manual_x"] = f"{x:.3f}"
        row["manual_y"] = f"{y:.3f}"
        row["visible"] = "1"
        row["quality"] = row.get("quality") or "good"
        row["confidence"] = row.get("confidence") or "1.0"
        row["annotator"] = self.annotator
        row["note"] = self.note_var.get()
        row["updated_at"] = dt.datetime.now().isoformat(timespec="seconds")
        self.annotations[self.key(cand)] = row
        self.show_item()

    def base_annotation(self, cand: Dict[str, str]) -> Dict[str, str]:
        return {
            "schema": "m3m_gcp_manual_image_observation_v1",
            "scene": cand["scene"],
            "point_name": cand["point_name"],
            "image_name": cand["image_name"],
            "image_path": cand["image_path"],
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

    def save(self) -> None:
        rows = [self.annotations[k] for k in sorted(self.annotations)]
        write_csv(self.out_csv, rows, ANNOTATION_FIELDS)
        self.status.configure(text=f"Saved {len(rows)} annotations to {self.out_csv}")

    def next_item(self) -> None:
        self.idx = min(len(self.candidates) - 1, self.idx + 1)
        self.reset_view()
        self.show_item()

    def prev_item(self) -> None:
        self.idx = max(0, self.idx - 1)
        self.reset_view()
        self.show_item()

    def reset_view(self) -> None:
        self.view_zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.drag_last = None

    def zoom_at(self, factor: float, anchor_x: float | None = None, anchor_y: float | None = None) -> None:
        if self.current_crop is None:
            return
        if anchor_x is None or anchor_y is None:
            anchor_x = self.canvas_size / 2
            anchor_y = self.canvas_size / 2
        crop_x = (anchor_x - self.pan_x) / self.render_scale
        crop_y = (anchor_y - self.pan_y) / self.render_scale
        old_zoom = self.view_zoom
        self.view_zoom = min(self.max_view_zoom, max(self.min_view_zoom, self.view_zoom * factor))
        if abs(self.view_zoom - old_zoom) < 1e-9:
            return
        render_size = max(1, int(round(self.canvas_size * self.view_zoom)))
        new_scale = render_size / self.crop_size
        self.pan_x = anchor_x - crop_x * new_scale
        self.pan_y = anchor_y - crop_y * new_scale
        self.render_current_view()
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
        self.render_current_view()
        self.status.configure(text="Image zoom reset. Mouse wheel/+/- zoom image, right-drag pan, 0 reset.")

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
    args = parser.parse_args()
    candidates = read_csv(Path(args.candidates_csv))
    if args.point_name:
        candidates = [r for r in candidates if r.get("point_name") == args.point_name]
    candidates.sort(key=lambda r: (r["point_name"], int(float(r.get("rank_for_gcp") or 9999)), r["image_name"]))
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
    )
    root.mainloop()


if __name__ == "__main__":
    main()
