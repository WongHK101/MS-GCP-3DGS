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
    "scene",
    "point_name",
    "image_name",
    "image_path",
    "projected_x",
    "projected_y",
    "manual_x",
    "manual_y",
    "visible",
    "quality",
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
    def __init__(self, root: tk.Tk, candidates: List[Dict[str, str]], out_csv: Path, crop_size: int, display_size: int):
        self.root = root
        self.candidates = candidates
        self.out_csv = out_csv
        self.crop_size = int(crop_size)
        self.display_size = int(display_size)
        self.idx = 0
        self.annotations: Dict[tuple[str, str, str], Dict[str, str]] = {}
        if out_csv.exists():
            for row in read_csv(out_csv):
                self.annotations[(row["scene"], row["point_name"], row["image_name"])] = row
        self.root.title("GCP manual annotator")
        self.info = ttk.Label(root, text="", font=("Arial", 11))
        self.info.pack(fill=tk.X, padx=8, pady=4)
        self.canvas = tk.Canvas(root, width=self.display_size, height=self.display_size, bg="black")
        self.canvas.pack(padx=8, pady=4)
        self.status = ttk.Label(root, text="", font=("Arial", 10))
        self.status.pack(fill=tk.X, padx=8, pady=4)
        btns = ttk.Frame(root)
        btns.pack(fill=tk.X, padx=8, pady=4)
        for text, cmd in [
            ("Visible good (v)", lambda: self.mark("1", "good")),
            ("Ambiguous (a)", lambda: self.mark("1", "ambiguous")),
            ("Not visible (x)", lambda: self.mark("0", "not_visible")),
            ("Prev (p)", self.prev_item),
            ("Next (n)", self.next_item),
            ("Save (s)", self.save),
        ]:
            ttk.Button(btns, text=text, command=cmd).pack(side=tk.LEFT, padx=3)
        self.note_var = tk.StringVar()
        ttk.Entry(root, textvariable=self.note_var).pack(fill=tk.X, padx=8, pady=4)
        self.canvas.bind("<Button-1>", self.on_click)
        root.bind("v", lambda e: self.mark("1", "good"))
        root.bind("a", lambda e: self.mark("1", "ambiguous"))
        root.bind("x", lambda e: self.mark("0", "not_visible"))
        root.bind("n", lambda e: self.next_item())
        root.bind("p", lambda e: self.prev_item())
        root.bind("s", lambda e: self.save())
        root.bind("q", lambda e: self.quit())
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
        overlay = crop.copy()
        draw = ImageDraw.Draw(overlay)
        cx, cy = px - left, py - top
        draw.line([(cx - 24, cy), (cx + 24, cy)], fill=(255, 230, 0), width=3)
        draw.line([(cx, cy - 24), (cx, cy + 24)], fill=(255, 230, 0), width=3)
        ann = self.annotations.get(self.key(cand))
        self.current_manual = None
        self.note_var.set("")
        if ann and ann.get("manual_x") and ann.get("manual_y"):
            mx = float(ann["manual_x"]) - left
            my = float(ann["manual_y"]) - top
            self.current_manual = (float(ann["manual_x"]), float(ann["manual_y"]))
            draw.ellipse((mx - 8, my - 8, mx + 8, my + 8), outline=(0, 255, 255), width=4)
            self.note_var.set(ann.get("note", ""))
        overlay.thumbnail((self.display_size, self.display_size), Image.Resampling.LANCZOS)
        self.scale = overlay.width / self.crop_size
        self.photo = ImageTk.PhotoImage(overlay)
        self.canvas.configure(width=overlay.width, height=overlay.height)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.info.configure(
            text=(
                f"{self.idx+1}/{len(self.candidates)}  {cand['scene']}  {cand['point_name']}  "
                f"{cand['image_name']}  rank={cand.get('rank_for_gcp','')} score={cand.get('center_score','')}"
            )
        )
        old = self.annotations.get(self.key(cand))
        if old:
            self.status.configure(text=f"Saved: visible={old.get('visible')} quality={old.get('quality')}")
        else:
            self.status.configure(text="Yellow cross = coarse projection. Click true GCP center, then press v/a/x/n.")

    def on_click(self, event) -> None:
        cand = self.candidates[self.idx]
        left, top = self.crop_origin
        x = left + event.x / self.scale
        y = top + event.y / self.scale
        self.current_manual = (x, y)
        row = self.annotations.get(self.key(cand), self.base_annotation(cand))
        row["manual_x"] = f"{x:.3f}"
        row["manual_y"] = f"{y:.3f}"
        row["visible"] = "1"
        row["quality"] = row.get("quality") or "good"
        row["note"] = self.note_var.get()
        row["updated_at"] = dt.datetime.now().isoformat(timespec="seconds")
        self.annotations[self.key(cand)] = row
        self.show_item()

    def base_annotation(self, cand: Dict[str, str]) -> Dict[str, str]:
        return {
            "scene": cand["scene"],
            "point_name": cand["point_name"],
            "image_name": cand["image_name"],
            "image_path": cand["image_path"],
            "projected_x": f"{float(cand['pixel_x']):.3f}",
            "projected_y": f"{float(cand['pixel_y']):.3f}",
            "manual_x": "",
            "manual_y": "",
            "visible": "",
            "quality": "",
            "note": "",
            "updated_at": "",
        }

    def mark(self, visible: str, quality: str) -> None:
        cand = self.candidates[self.idx]
        row = self.annotations.get(self.key(cand), self.base_annotation(cand))
        row["visible"] = visible
        row["quality"] = quality
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
        self.show_item()

    def prev_item(self) -> None:
        self.idx = max(0, self.idx - 1)
        self.show_item()

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
    args = parser.parse_args()
    candidates = read_csv(Path(args.candidates_csv))
    if args.point_name:
        candidates = [r for r in candidates if r.get("point_name") == args.point_name]
    candidates.sort(key=lambda r: (r["point_name"], int(float(r.get("rank_for_gcp") or 9999)), r["image_name"]))
    if args.max_rows and args.max_rows > 0:
        candidates = candidates[: args.max_rows]
    root = tk.Tk()
    Annotator(root, candidates, Path(args.out_csv), crop_size=args.crop_size, display_size=args.display_size)
    root.mainloop()


if __name__ == "__main__":
    main()

