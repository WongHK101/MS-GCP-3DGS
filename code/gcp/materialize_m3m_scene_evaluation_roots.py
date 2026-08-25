#!/usr/bin/env python3
"""Materialize scene-generic GCP and held-out RGB camera roots.

This is a thin scene-general wrapper around the already frozen native-quarter
camera-subset materializer.  It derives counts from the formal input manifest,
keeps only active control/checkpoint observations for the GCP root, and makes
an evaluation-only held-out root with an explicit empty points3D.bin member.
It never changes training input or exposes a held-out image to training.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

from materialize_gs_gcp_native_quarter_inputs import (
    canonical_sha256 as formal_manifest_canonical_sha256,
    sha256_file,
)
from materialize_m3m_native_quarter_evaluation_subset import materialize_subset

HERE = Path(__file__).resolve().parent
COLMAP_UTILS = HERE.parent / "colmap" / "utils"
import sys

if str(COLMAP_UTILS) not in sys.path:
    sys.path.insert(0, str(COLMAP_UTILS))

from read_write_model import write_points3D_binary  # noqa: E402


PROTOCOL_ID = "m3m_gcp_native_quarter_geometry_v2"
RELEASE_MANIFEST_SHA256 = "21fbac75d66433169535ea7440c31393f7a5ecdb4ed94fcefd31d1780c28bea4"
EXPECTED_SCENE_COUNTS = {
    "gcp_20000_20260602": {
        "full": 298,
        "train": 260,
        "test": 38,
        "gcp_observations": 116,
        "gcp_views": 103,
        "controls": 5,
        "checkpoints": 4,
    }
}


def canonical_sha256(payload: dict[str, Any]) -> str:
    import hashlib

    body = {key: value for key, value in payload.items() if key != "canonical_sha256"}
    encoded = json.dumps(
        body, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def require_absent(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)


def filter_gcp_observations(
    *, source: Path, scene: str, destination: Path
) -> tuple[list[dict[str, str]], list[str]]:
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [
            dict(row)
            for row in reader
            if row.get("scene") == scene
            and str(row.get("active_formal_eligible", "")).lower() == "true"
            and row.get("active_role") in {"control", "checkpoint"}
            and row.get("audit_disposition") == "formal_primary"
        ]
    if not fieldnames or not rows:
        raise RuntimeError("active scene GCP observation set is empty")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows, fieldnames


def materialize_rgb_root(
    *, formal_scene_root: Path, formal: dict[str, Any], output_root: Path
) -> dict[str, Any]:
    require_absent(output_root)
    source = formal_scene_root / "test"
    source_images = source / "images"
    source_sparse = source / "sparse" / "0"
    if not source_images.is_dir() or not source_sparse.is_dir():
        raise FileNotFoundError(source)
    test_rows = [row for row in formal["images"] if row.get("role") == "test"]
    expected_names = {str(row["image_name"]) for row in test_rows}
    actual_names = {path.name for path in source_images.iterdir() if path.is_file()}
    if actual_names != expected_names:
        raise RuntimeError("held-out RGB inventory differs from formal manifest")

    output_sparse = output_root / "sparse" / "0"
    output_sparse.mkdir(parents=True)
    os.symlink(source_images, output_root / "images", target_is_directory=True)
    identities: dict[str, dict[str, Any]] = {}
    for name in ("cameras.bin", "images.bin", "points3D.ply"):
        src = source_sparse / name
        if not src.is_file():
            raise FileNotFoundError(src)
        dst = output_sparse / name
        os.link(src, dst)
        identities[name] = {
            "bytes": dst.stat().st_size,
            "sha256": sha256_file(dst),
        }
    points = output_sparse / "points3D.bin"
    write_points3D_binary({}, points)
    identities["points3D.bin"] = {
        "bytes": points.stat().st_size,
        "sha256": sha256_file(points),
    }
    payload: dict[str, Any] = {
        "schema": "m3m_gcp_scene_rgb_evaluation_camera_root_v1",
        "status": "PASS_RGB_EVALUATION_CAMERA_ROOT",
        "scene": formal["scene"],
        "purpose": "held-out RGB evaluation only; never training, prior, checkpoint selection, or tuning",
        "formal_input_manifest_canonical_sha256": formal["manifest_sha256"],
        "view_count": len(test_rows),
        "image_names": sorted(expected_names),
        "images_symlink_target": str(source_images.resolve()),
        "sparse_files": identities,
        "points3d_bin_point_count": 0,
        "training_or_prior_use_forbidden": True,
        "heldout_rgb_parameter_fitting_forbidden": True,
    }
    payload["canonical_sha256"] = canonical_sha256(payload)
    write_json(output_root / "RGB_EVALUATION_CAMERA_ROOT_MANIFEST.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--formal-input-manifest", type=Path, required=True)
    parser.add_argument("--protocol-release", type=Path, required=True)
    parser.add_argument("--gcp-output-root", type=Path, required=True)
    parser.add_argument("--rgb-output-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    args = parser.parse_args()

    scene = args.scene
    expected = EXPECTED_SCENE_COUNTS.get(scene)
    if expected is None:
        raise ValueError(f"scene has no reviewed count binding: {scene}")
    formal_manifest = args.formal_input_manifest.expanduser().resolve()
    formal_scene_root = formal_manifest.parent
    protocol_release = args.protocol_release.expanduser().resolve()
    gcp_output_root = args.gcp_output_root.expanduser().resolve()
    rgb_output_root = args.rgb_output_root.expanduser().resolve()
    evidence_root = args.evidence_root.expanduser().resolve()
    for path in (gcp_output_root, rgb_output_root, evidence_root):
        require_absent(path)
    release_manifest = protocol_release / "protocol_release_manifest.json"
    if sha256_file(release_manifest) != RELEASE_MANIFEST_SHA256:
        raise RuntimeError("protocol release manifest identity mismatch")
    formal = json.loads(formal_manifest.read_text(encoding="utf-8"))
    if formal_manifest_canonical_sha256(formal) != formal.get("manifest_sha256"):
        raise RuntimeError("formal input manifest canonical identity mismatch")
    actual_counts = {
        "full": int(formal.get("full_view_count", -1)),
        "train": int(formal.get("train_view_count", -1)),
        "test": int(formal.get("test_view_count", -1)),
    }
    if formal.get("scene") != scene or actual_counts != {
        key: expected[key] for key in ("full", "train", "test")
    }:
        raise RuntimeError("formal input scene/count binding mismatch")

    evidence_root.mkdir(parents=True)
    observations = protocol_release / "observation_semantics.csv"
    filtered = evidence_root / "active_gcp_observations.csv"
    rows, _ = filter_gcp_observations(
        source=observations, scene=scene, destination=filtered
    )
    points_by_role = {
        role: {row["point_name"] for row in rows if row["active_role"] == role}
        for role in ("control", "checkpoint")
    }
    image_names = sorted({row["image_name"] for row in rows})
    if (
        len(rows) != expected["gcp_observations"]
        or len(image_names) != expected["gcp_views"]
        or len(points_by_role["control"]) != expected["controls"]
        or len(points_by_role["checkpoint"]) != expected["checkpoints"]
    ):
        raise RuntimeError("active GCP count binding mismatch")

    subset = materialize_subset(
        scene=scene,
        formal_input_manifest_path=formal_manifest,
        protocol_observations_path=filtered,
        output_root=gcp_output_root,
        file_mode="hardlink",
    )
    if (
        subset.get("observation_count") != expected["gcp_observations"]
        or subset.get("camera_view_count") != expected["gcp_views"]
    ):
        raise RuntimeError("materialized GCP camera-root count mismatch")
    allowlist = evidence_root / "gcp_camera_allowlist.csv"
    with allowlist.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_name"], lineterminator="\n")
        writer.writeheader()
        writer.writerows({"image_name": name} for name in image_names)
    rgb = materialize_rgb_root(
        formal_scene_root=formal_scene_root,
        formal=formal,
        output_root=rgb_output_root,
    )
    if rgb["view_count"] != expected["test"]:
        raise RuntimeError("materialized RGB camera-root count mismatch")

    receipt: dict[str, Any] = {
        "schema": "m3m_gcp_scene_evaluation_roots_receipt_v1",
        "status": "PASS",
        "scene": scene,
        "protocol_id": PROTOCOL_ID,
        "formal_input_manifest": {
            "path": str(formal_manifest),
            "file_sha256": sha256_file(formal_manifest),
            "canonical_sha256": formal["manifest_sha256"],
        },
        "protocol_release_manifest_sha256": RELEASE_MANIFEST_SHA256,
        "active_gcp_observations": {
            "path": str(filtered),
            "sha256": sha256_file(filtered),
            "observation_count": len(rows),
            "camera_view_count": len(image_names),
            "control_points": sorted(points_by_role["control"]),
            "checkpoint_points": sorted(points_by_role["checkpoint"]),
        },
        "gcp_camera_root": {
            "path": str(gcp_output_root),
            "manifest_sha256": subset["manifest_sha256"],
            "allowlist": str(allowlist),
            "allowlist_sha256": sha256_file(allowlist),
        },
        "rgb_camera_root": {
            "path": str(rgb_output_root),
            "manifest_sha256": rgb["canonical_sha256"],
        },
    }
    receipt["canonical_sha256"] = canonical_sha256(receipt)
    write_json(evidence_root / "evaluation_roots_receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
