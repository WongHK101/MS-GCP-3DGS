#!/usr/bin/env python3
"""Freeze camera parity samples and audit path-backed image materialization."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from gs_gcp_stage0_5 import (
    read_cameras_binary,
    read_images_binary,
    write_cameras_binary,
    write_images_binary,
)


THREE_K = "gcp_3000_20260602"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def _canonical_rank(scene: str, image_name: str) -> str:
    return hashlib.sha256(f"{scene}\0{image_name}".encode("utf-8")).hexdigest()


def freeze_samples(split: dict[str, Any], generator_commit: str | None = None) -> dict[str, Any]:
    scenes = []
    for scene_payload in sorted(split["scenes"], key=lambda row: row["scene"].encode("utf-8")):
        scene = scene_payload["scene"]
        assignments = list(scene_payload["assignments"])
        if scene == THREE_K:
            selected = assignments
            rule = "all_images"
        else:
            selected_by_name: dict[str, dict[str, Any]] = {}
            by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in assignments:
                by_stratum[row["stratum_id"]].append(row)
            for rows in by_stratum.values():
                ordered = sorted(rows, key=lambda row: (
                    row["capture_timestamp"], int(row["capture_sequence"]),
                    row["image_name"].encode("utf-8"), int(row["image_id"]),
                ))
                for row in (ordered[0], ordered[(len(ordered) - 1) // 2], ordered[-1]):
                    selected_by_name[row["image_name"]] = row
                for role in ("train", "test"):
                    candidates = [row for row in rows if row["split_role"] == role]
                    if candidates:
                        row = min(candidates, key=lambda item: _canonical_rank(scene, item["image_name"]))
                        selected_by_name[row["image_name"]] = row
            modal_dimensions = Counter(
                (int(row["decoded_width"]), int(row["decoded_height"])) for row in assignments
            ).most_common(1)[0][0]
            for row in assignments:
                if (int(row["decoded_width"]), int(row["decoded_height"])) != modal_dimensions:
                    selected_by_name[row["image_name"]] = row
            if len(selected_by_name) < 16:
                for row in sorted(assignments, key=lambda item: _canonical_rank(scene, item["image_name"])):
                    selected_by_name[row["image_name"]] = row
                    if len(selected_by_name) >= 16:
                        break
            selected = sorted(selected_by_name.values(), key=lambda row: row["image_name"].encode("utf-8"))
            rule = "per_stratum_first_middle_last_plus_min_hash_train_test_nonmodal_dimensions_minimum_16"
        scenes.append({
            "scene": scene,
            "selection_rule": rule,
            "full_image_count": len(assignments),
            "selected_image_count": len(selected),
            "images": [{
                "image_id": int(row["image_id"]),
                "image_name": row["image_name"],
                "split_role": row["split_role"],
                "stratum_id": row["stratum_id"],
                "decoded_width": int(row["decoded_width"]),
                "decoded_height": int(row["decoded_height"]),
                "image_sha256": row["image_sha256"],
            } for row in selected],
        })
    payload = {
        "schema": "gs_gcp_original_3dgs_camera_parity_sample_manifest_v1",
        "selection_frozen_before_pixel_comparison": True,
        "split_manifest_sha256": split["manifest_sha256"],
        "generator_provenance": {
            "git_commit": generator_commit,
            "script_relative_path": "code/gcp/original_3dgs_camera_compatibility.py",
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "working_tree_requirement": "clean committed generator before frozen output",
        },
        "scenes": scenes,
    }
    payload["manifest_sha256"] = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    return payload


def _jpeg_fds() -> list[str]:
    if os.name != "posix":
        return []
    rows = []
    for entry in Path("/proc/self/fd").iterdir():
        try:
            target = os.readlink(entry)
        except OSError:
            continue
        if target.lower().endswith((".jpg", ".jpeg")):
            rows.append(target)
    return sorted(rows)


def audit_all_images(split: dict[str, Any], data_root: Path, output: Path) -> dict[str, Any]:
    records = []
    for scene_payload in sorted(split["scenes"], key=lambda row: row["scene"].encode("utf-8")):
        scene = scene_payload["scene"]
        image_root = data_root / scene / "images"
        for row in sorted(scene_payload["assignments"], key=lambda item: item["image_name"].encode("utf-8")):
            image_path = image_root / row["image_name"]
            before = len(_jpeg_fds())
            with Image.open(image_path) as image:
                decoded_mode = image.mode
                decoded_size = image.size
                resized_size = (round(image.size[0] / 4), round(image.size[1] / 4))
                resized = image.resize(resized_size)
                array = np.array(resized)
                tensor = torch.from_numpy(array) / 255.0
                if len(tensor.shape) == 3:
                    tensor = tensor.permute(2, 0, 1)
                else:
                    tensor = tensor.unsqueeze(dim=-1).permute(2, 0, 1)
                tensor_sha = hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()
            after = len(_jpeg_fds())
            if sha256_file(image_path) != row["image_sha256"]:
                raise ValueError(f"image SHA mismatch: {scene}/{row['image_name']}")
            if decoded_size != (int(row["decoded_width"]), int(row["decoded_height"])):
                raise ValueError(f"decoded dimensions mismatch: {scene}/{row['image_name']}")
            if after != before:
                raise ValueError(f"JPEG FD leak: {scene}/{row['image_name']}")
            records.append({
                "scene": scene,
                "image_name": row["image_name"],
                "image_sha256": row["image_sha256"],
                "decoded_mode": decoded_mode,
                "decoded_width": decoded_size[0],
                "decoded_height": decoded_size[1],
                "resized_width": resized_size[0],
                "resized_height": resized_size[1],
                "chw_float_tensor_sha256": tensor_sha,
                "jpeg_fd_before": before,
                "jpeg_fd_after": after,
            })
    payload = {
        "schema": "gs_gcp_original_3dgs_path_backed_full_image_audit_v1",
        "status": "PASS",
        "image_count": len(records),
        "expected_image_count": 6187,
        "records": records,
    }
    if len(records) != 6187:
        raise ValueError(f"full image count mismatch: {len(records)} != 6187")
    payload["records_root_sha256"] = hashlib.sha256(canonical_bytes(records)).hexdigest()
    write_json(output, payload)
    return payload


def compare_reports(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    fields = [
        "image_id", "image_name", "camera_id", "loaded_width", "loaded_height",
        "channels", "dtype", "device", "tensor_bytes", "tensor_sha256",
        "loaded_fx", "loaded_fy", "loaded_cx", "loaded_cy",
        "R", "T", "FoVx", "FoVy",
        "world_view_transform_sha256", "projection_matrix_sha256",
        "full_proj_transform_sha256", "camera_center_sha256",
    ]
    ref_rows = reference["camera_records"]
    cand_rows = candidate["camera_records"]
    checks = [{
        "name": "camera_count_and_order",
        "passed": [row["image_name"] for row in ref_rows] == [row["image_name"] for row in cand_rows],
    }]
    for field in fields:
        checks.append({
            "name": field,
            "passed": [row.get(field) for row in ref_rows] == [row.get(field) for row in cand_rows],
        })
    ray_error = max(
        float(reference.get("max_normalized_ray_coordinate_error", 0.0)),
        float(candidate.get("max_normalized_ray_coordinate_error", 0.0)),
    )
    checks.append({"name": "normalized_ray_tolerance", "passed": ray_error <= 1e-12, "evidence": ray_error})
    passed = all(row["passed"] for row in checks)
    return {
        "schema": "gs_gcp_original_3dgs_camera_report_parity_v1",
        "status": "PASS" if passed else "BLOCKER",
        "checks": checks,
    }


def materialize_parity_subset(samples: dict[str, Any], scene: str, source_root: Path, output_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(output_root)
    scene_row = next((row for row in samples["scenes"] if row["scene"] == scene), None)
    if scene_row is None:
        raise ValueError(f"scene missing from parity sample manifest: {scene}")
    sparse = source_root / "sparse" / "0"
    cameras = read_cameras_binary(sparse / "cameras.bin")
    images = read_images_binary(sparse / "images.bin")
    by_name = {row.name: row for row in images.values()}
    selected_images = {}
    camera_ids = set()
    image_root = output_root / "images"
    model_root = output_root / "sparse" / "0"
    image_root.mkdir(parents=True)
    model_root.mkdir(parents=True)
    for row in scene_row["images"]:
        image = by_name.get(row["image_name"])
        if image is None or int(image.id) != int(row["image_id"]):
            raise ValueError(f"COLMAP image identity mismatch: {row['image_name']}")
        source_image = source_root / "images" / row["image_name"]
        if sha256_file(source_image) != row["image_sha256"]:
            raise ValueError(f"source image SHA mismatch: {row['image_name']}")
        os.symlink(source_image, image_root / row["image_name"])
        selected_images[int(image.id)] = image
        camera_ids.add(int(image.camera_id))
    selected_cameras = {camera_id: cameras[camera_id] for camera_id in sorted(camera_ids)}
    write_cameras_binary(selected_cameras, model_root / "cameras.bin")
    write_images_binary(selected_images, model_root / "images.bin")
    shutil.copy2(sparse / "points3D.ply", model_root / "points3D.ply")
    payload = {
        "schema": "gs_gcp_original_3dgs_camera_parity_subset_v1",
        "scene": scene,
        "sample_manifest_sha256": samples["manifest_sha256"],
        "image_count": len(selected_images),
        "image_names": [row["image_name"] for row in scene_row["images"]],
        "cameras_bin_sha256": sha256_file(model_root / "cameras.bin"),
        "images_bin_sha256": sha256_file(model_root / "images.bin"),
        "points3d_tracks_present": False,
    }
    payload["manifest_sha256"] = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    write_json(output_root.parent / "CAMERA_SUBSET_MANIFEST.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze-samples")
    freeze.add_argument("--split_manifest", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--generator_commit", required=True)
    audit = subparsers.add_parser("audit-all-images")
    audit.add_argument("--split_manifest", type=Path, required=True)
    audit.add_argument("--data_root", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    compare = subparsers.add_parser("compare-reports")
    compare.add_argument("--reference", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    materialize = subparsers.add_parser("materialize-parity-subset")
    materialize.add_argument("--sample_manifest", type=Path, required=True)
    materialize.add_argument("--scene", required=True)
    materialize.add_argument("--source_root", type=Path, required=True)
    materialize.add_argument("--output_root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze-samples":
        split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
        write_json(args.output, freeze_samples(split, args.generator_commit))
        return 0
    if args.command == "audit-all-images":
        split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
        audit_all_images(split, args.data_root.resolve(), args.output)
        return 0
    if args.command == "materialize-parity-subset":
        samples = json.loads(args.sample_manifest.read_text(encoding="utf-8"))
        result = materialize_parity_subset(samples, args.scene, args.source_root.resolve(), args.output_root.resolve())
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    result = compare_reports(
        json.loads(args.reference.read_text(encoding="utf-8")),
        json.loads(args.candidate.read_text(encoding="utf-8")),
    )
    write_json(args.output, result)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
