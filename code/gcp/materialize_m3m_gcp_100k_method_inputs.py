#!/usr/bin/env python3
"""Materialize 100K method inputs with the exact reviewed 3K split semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


SCENE = "gcp_100000_20260610"
TRAIN_COUNT = 2196
TEST_COUNT = 314
FORMAL_FILE_SHA = "c2cf9e951d95fee12a28d942e95c5c420df55bc364738b3f8737fed1c78bef3d"
FORMAL_CANONICAL_SHA = "5b4fe34743310bd2225feb2dd236200606be933002fec19d2c9ecb9f3ba6769d"
FORMAL_CAMERA_SHA = "6669584ba1ba326cf5b372b878a5abf182f8cfe0bfe0845da3a0c4f7aed8fe5e"
FORMAL_IMAGES_SHA = "dfc1a5d17532aebb3da670598635baea5c8fbf999592b6b567504251a01c9f72"
INITIAL_PLY_SHA = "9f653655a34c05007e58f339afec593136bd857a56b13a612c79d8e53913364e"
FULL_HASHES = {
    "cameras.bin": FORMAL_CAMERA_SHA,
    "images.bin": "57163927bceee6ca330c113c9caf06cafe1a84a7ca21ac0f055680dcbe8eff6e",
    "points3D.bin": "09fc811f32558a11a47bada7393bf7bce2585cbe68eb4872ffce72025b0fc9aa",
    "points3D.ply": INITIAL_PLY_SHA,
    "frames.bin": "f443daf1aa92ed665195a88bcd5d7a0bba025d49a01cc1b95343294f8282dfc5",
    "rigs.bin": "73e4d5f0da0a84a5711b46ae716a149f1532832f95bb09b4b01e73cf2c5afbe9",
}
FULL_AUDIT_SHA = "3c883378e93593328dcf4d864d3aa0d7795e67e24b3f3a2c9c47986626cffe9d"
CITY_HASHES = {
    "cameras.bin": FORMAL_CAMERA_SHA,
    "images.bin": "825fb831886d96bb50d7d25f110909d6938a4a80afb29d3f047873d03d18dbe5",
    "points3D.bin": FULL_HASHES["points3D.bin"],
}
METRO_HASHES = {
    "cameras.bin": FORMAL_CAMERA_SHA,
    "images.bin": CITY_HASHES["images.bin"],
    "points3D.bin": "fcbb06d2b52770281b2b2c88f6d1a9deb5b2435e4578e63ca77bb8f197c37e7f",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: dict) -> str:
    clean = {key: value for key, value in payload.items() if key != "canonical_sha256"}
    raw = json.dumps(clean, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def require_hashes(root: Path, expected: dict[str, str], label: str) -> dict[str, dict]:
    result = {}
    for name, wanted in expected.items():
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != wanted:
            raise RuntimeError(f"{label} {name} SHA mismatch: {actual}")
        result[name] = {"bytes": path.stat().st_size, "sha256": actual}
    return result


def load_bound_evidence(path: Path, expected_sha: str, schema: str) -> dict:
    if sha256(path) != expected_sha:
        raise RuntimeError(f"evidence SHA mismatch: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != schema or value.get("status") != "PASS" or value.get("passed") is not True:
        raise RuntimeError(f"evidence did not pass: {path}")
    return value


def link_dir(target: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(target, link, target_is_directory=True)


def link_sparse(source: Path, output: Path, names: tuple[str, ...], ply: Path) -> None:
    output.mkdir(parents=True)
    for name in names:
        os.link(source / name, output / name)
    os.link(ply, output / "points3D.ply")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-scene-root", type=Path, required=True)
    parser.add_argument("--full-model-root", type=Path, required=True)
    parser.add_argument("--full-package-audit", type=Path, required=True)
    parser.add_argument("--city-track-model", type=Path, required=True)
    parser.add_argument("--city-track-evidence", type=Path, required=True)
    parser.add_argument("--expected-city-track-evidence-sha256", required=True)
    parser.add_argument("--metro-track-model", type=Path, required=True)
    parser.add_argument("--metro-track-evidence", type=Path, required=True)
    parser.add_argument("--expected-metro-track-evidence-sha256", required=True)
    parser.add_argument("--common-root", type=Path, required=True)
    parser.add_argument("--qgs-root", type=Path, required=True)
    parser.add_argument("--citygaussian-root", type=Path, required=True)
    parser.add_argument("--citygs-root", type=Path, required=True)
    parser.add_argument("--metrogs-root", type=Path, required=True)
    parser.add_argument("--evidence-output", type=Path, required=True)
    args = parser.parse_args()

    formal = args.formal_scene_root.resolve()
    full = args.full_model_root.resolve()
    city = args.city_track_model.resolve()
    metro = args.metro_track_model.resolve()
    common = args.common_root.resolve()
    roots = {
        "qgs": args.qgs_root.resolve(),
        "citygaussian_v2": args.citygaussian_root.resolve(),
        "citygs_x": args.citygs_root.resolve(),
        "metrogs": args.metrogs_root.resolve(),
    }
    evidence_output = args.evidence_output.resolve()
    for path in (common, *roots.values(), evidence_output):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing existing output: {path}")

    formal_manifest = formal / "NATIVE_QUARTER_INPUT_MANIFEST.json"
    if sha256(formal_manifest) != FORMAL_FILE_SHA:
        raise RuntimeError("formal manifest file SHA mismatch")
    manifest = json.loads(formal_manifest.read_text(encoding="utf-8"))
    if manifest.get("manifest_sha256") != FORMAL_CANONICAL_SHA:
        raise RuntimeError("formal manifest canonical SHA mismatch")
    train = {row["image_name"] for row in manifest["images"] if row["role"] == "train"}
    test = {row["image_name"] for row in manifest["images"] if row["role"] == "test"}
    if len(train) != TRAIN_COUNT or len(test) != TEST_COUNT or train & test:
        raise RuntimeError(f"formal split differs from {TRAIN_COUNT}/{TEST_COUNT}")
    train_images = (formal / "train" / "images").resolve()
    if {path.name for path in train_images.iterdir() if path.is_file()} != train:
        raise RuntimeError("formal train image inventory mismatch")
    formal_sparse = (formal / "train" / "sparse" / "0").resolve()
    formal_id = require_hashes(
        formal_sparse,
        {"cameras.bin": FORMAL_CAMERA_SHA, "images.bin": FORMAL_IMAGES_SHA, "points3D.ply": INITIAL_PLY_SHA},
        "formal train",
    )
    full_id = require_hashes(full, FULL_HASHES, "full all-image SfM")
    audit_path = args.full_package_audit.resolve()
    if sha256(audit_path) != FULL_AUDIT_SHA:
        raise RuntimeError("full all-image package audit SHA mismatch")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "pass" or audit.get("counts", {}).get("images") != 2510:
        raise RuntimeError("full all-image package audit did not pass")
    city_id = require_hashes(city, CITY_HASHES, "City compatibility")
    metro_id = require_hashes(metro, METRO_HASHES, "Metro closure")
    city_ev_path = args.city_track_evidence.resolve()
    metro_ev_path = args.metro_track_evidence.resolve()
    city_ev = load_bound_evidence(
        city_ev_path,
        args.expected_city_track_evidence_sha256,
        "m3m_gcp_native_quarter_city_track_compatibility_streaming_v1",
    )
    metro_ev = load_bound_evidence(
        metro_ev_path,
        args.expected_metro_track_evidence_sha256,
        "m3m_gcp_colmap_streaming_frozen_train_track_closure_v1",
    )
    if city_ev["derived_model"]["sha256"] != {key: value for key, value in CITY_HASHES.items()}:
        raise RuntimeError("City evidence/model identity mismatch")
    if metro_ev["derived_model"]["sha256"] != {key: value for key, value in METRO_HASHES.items()}:
        raise RuntimeError("Metro evidence/model identity mismatch")

    made: list[Path] = []
    try:
        common.mkdir(parents=True); made.append(common)
        link_dir(train_images, common / "images")
        link_sparse(formal_sparse, common / "sparse" / "0", ("cameras.bin", "images.bin"), formal_sparse / "points3D.ply")

        qgs = roots["qgs"]; qgs.mkdir(parents=True); made.append(qgs)
        link_dir(train_images, qgs / "images")
        link_dir(train_images, qgs / "images_undistorted_1.0")
        link_dir(formal / "train" / "sparse", qgs / "sparse")

        for method in ("citygaussian_v2", "citygs_x"):
            root = roots[method]; root.mkdir(parents=True); made.append(root)
            link_dir(train_images, root / "images")
            link_sparse(city, root / "sparse" / "0", ("cameras.bin", "images.bin", "points3D.bin"), formal_sparse / "points3D.ply")

        metro_root = roots["metrogs"]; metro_root.mkdir(parents=True); made.append(metro_root)
        link_dir(train_images, metro_root / "images")
        link_sparse(metro, metro_root / "sparse" / "0", ("cameras.bin", "images.bin", "points3D.bin"), formal_sparse / "points3D.ply")
    except Exception:
        for path in reversed(made):
            shutil.rmtree(path, ignore_errors=True)
        raise

    payload = {
        "schema": "m3m_gcp_100k_per_method_input_preparation_v2",
        "scene": SCENE,
        "status": "PASS_PER_METHOD_INPUT_PREPARATION_NO_TRAINING_NO_PRIOR",
        "formal_manifest": {
            "path": str(formal_manifest), "file_sha256": FORMAL_FILE_SHA,
            "canonical_sha256": FORMAL_CANONICAL_SHA, "train_views": TRAIN_COUNT, "test_views": TEST_COUNT,
        },
        "shared_all_image_sfm": {
            "path": str(full), "image_count": 2510, "files": full_id,
            "package_audit": {"path": str(audit_path), "sha256": FULL_AUDIT_SHA, "status": "pass"},
        },
        "formal_train_view": {"path": str(formal_sparse), "files": formal_id},
        "city_track_compatibility": {
            "path": str(city), "files": city_id,
            "evidence": {"path": str(city_ev_path), "sha256": args.expected_city_track_evidence_sha256},
        },
        "metrogs_track_closure": {
            "path": str(metro), "files": metro_id,
            "evidence": {"path": str(metro_ev_path), "sha256": args.expected_metro_track_evidence_sha256},
        },
        "method_inputs": {
            "3dgs_original/2dgs/pgsr/rade_gs/gsprior/sof": {"root": str(common), "semantic": "formal 2196-view train root plus shared all-image-SfM PLY"},
            "qgs": {"root": str(roots["qgs"]), "semantic": "exact formal train sparse and dual image aliases, matching reviewed 3K"},
            "citygaussian_v2": {"root": str(roots["citygaussian_v2"]), "semantic": "train records selected after all-image SfM; byte-identical full points3D.bin"},
            "citygs_x": {"root": str(roots["citygs_x"]), "semantic": "same reviewed compatibility semantics as CityGaussianV2"},
            "metrogs": {"root": str(roots["metrogs"]), "semantic": "reciprocal train-only track closure derived after all-image SfM"},
        },
        "access_boundary": {
            "all_images_participated_in_sfm": True, "heldout_rgb_opened_by_preparation": 0,
            "gcp_opened": 0, "lidar_opened": 0, "training_started": False, "external_prior_started": False,
            "rgb_bytes_copied": 0,
        },
        "materializer": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
    }
    payload["canonical_sha256"] = canonical_sha256(payload)
    evidence_output.parent.mkdir(parents=True, exist_ok=True)
    evidence_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
