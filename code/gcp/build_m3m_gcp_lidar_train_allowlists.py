#!/usr/bin/env python3
"""Materialize the six frozen train-view allowlists used by LiDAR v1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


SCENES = {
    "gcp_3000_20260602": 82,
    "gcp_5000_20260602": 88,
    "gcp_20000_20260602": 260,
    "gcp_10000_20260610": 854,
    "gcp_50000_20260610": 1932,
    "gcp_100000_20260610": 2196,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: dict[str, Any]) -> str:
    clean = {key: value for key, value in payload.items() if key != "canonical_sha256"}
    raw = json.dumps(clean, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    split = json.loads(args.split.read_text(encoding="utf-8"))
    rows_by_scene = {str(row["scene"]): row for row in split["scenes"]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for scene, expected_count in SCENES.items():
        assignments = rows_by_scene[scene]["assignments"]
        names = [str(row["image_name"]) for row in assignments if row["split_role"] == "train"]
        if len(names) != expected_count or len(set(names)) != expected_count:
            raise RuntimeError(f"{scene}: train allowlist count/uniqueness mismatch")
        output = args.output_dir / f"{scene}.csv"
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["image_name"])
            writer.writerows((name,) for name in names)
        manifest_rows.append(
            {
                "scene": scene,
                "path": output.as_posix(),
                "train_view_count": expected_count,
                "sha256": sha256_file(output),
            }
        )

    manifest = {
        "schema": "m3m_gcp_lidar_train_view_allowlists_v1",
        "protocol_id": "m3m_gcp_lidar_rendered_surface_v1",
        "source_split_path": args.split.as_posix(),
        "source_split_file_sha256": sha256_file(args.split),
        "source_split_canonical_sha256": split["manifest_sha256"],
        "ordering": "frozen split-manifest assignment order",
        "rows": manifest_rows,
    }
    manifest["canonical_sha256"] = canonical_sha256(manifest)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
