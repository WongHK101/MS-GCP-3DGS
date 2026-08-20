#!/usr/bin/env python3
"""Verify all formal scene inputs and emit a path-independent preparation manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from materialize_gs_gcp_native_quarter_inputs import (  # noqa: E402
    sha256_file,
    verify_materialization,
)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def inventory(root: Path) -> tuple[int, int]:
    files = [path for path in root.rglob("*") if path.is_file()]
    return len(files), sum(path.stat().st_size for path in files)


def materialization_provenance(manifest: dict[str, Any]) -> dict[str, str]:
    """Normalize optional physical-layout provenance without changing identity.

    The earliest formal 3K manifest predates this informational field.  Its
    complete file inventory and hashes remain normative, so absence must not
    invalidate an otherwise byte-identical formal input.
    """
    value = manifest.get("file_materialization")
    if value is None:
        return {
            "mode": "legacy_manifest_field_absent",
            "semantic_identity": "file bytes and manifest hashes are normative",
        }
    if not isinstance(value, dict):
        raise TypeError("file_materialization must be an object when present")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--repository-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decode-images", action="store_true")
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    release_root = args.release_root.resolve()
    formal_root = release_root / "formal_inputs"
    scene_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for frozen in contract["scenes"]:
        scene = frozen["scene"]
        root = formal_root / scene
        verification = verify_materialization(root, decode_images=args.decode_images)
        if not verification["passed"]:
            failures.extend(f"{scene}: {item}" for item in verification["errors"])
            continue
        manifest_path = root / "NATIVE_QUARTER_INPUT_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_counts = (
            frozen["full_views"],
            frozen["train_views"],
            frozen["test_views"],
        )
        actual_counts = (
            manifest["full_view_count"],
            manifest["train_view_count"],
            manifest["test_view_count"],
        )
        if actual_counts != expected_counts:
            failures.append(
                f"{scene}: formal counts {actual_counts} != contract {expected_counts}"
            )
            continue
        file_count, logical_bytes = inventory(root)
        scene_rows.append(
            {
                "scene": scene,
                "relative_path": f"formal_inputs/{scene}",
                "manifest_file_sha256": sha256_file(manifest_path),
                "manifest_canonical_sha256": manifest["manifest_sha256"],
                "source_package_audit_file_sha256": manifest[
                    "source_package_audit_file_sha256"
                ],
                "source_image_manifest_file_sha256": manifest[
                    "source_image_manifest_file_sha256"
                ],
                "source_model_sha256": manifest["source_model_sha256"],
                "full_views": actual_counts[0],
                "train_views": actual_counts[1],
                "test_views": actual_counts[2],
                "file_materialization": materialization_provenance(manifest),
                "file_count": file_count,
                "logical_bytes": logical_bytes,
                "decoded_images_checked": verification["decoded_images_checked"],
                "verification": "PASS",
            }
        )

    payload: dict[str, Any] = {
        "schema": "m3m_gcp_six_scene_common_preparation_manifest_v1",
        "status": "PASS_COMMON_SCENE_PREPARATION_NO_TRAINING"
        if not failures and len(scene_rows) == 6
        else "FAIL",
        "protocol_id": contract["source_geometry_protocol_id"],
        "lidar_protocol_candidate_id": contract["protocol_id"],
        "repository_head": args.repository_head,
        "release_root_digest_sha256": contract["source_data_release"][
            "release_root_digest_sha256"
        ],
        "contract_file_sha256": sha256_file(args.contract),
        "materializer_script_sha256": sha256_file(
            HERE / "materialize_gs_gcp_native_quarter_inputs.py"
        ),
        "builder_script_sha256": sha256_file(Path(__file__).resolve()),
        "scene_count": len(scene_rows),
        "scenes": scene_rows,
        "training_started": False,
        "gpu_required_for_this_preparation": False,
        "method_specific_prior_generation": "NOT_STARTED_REQUIRES_SEPARATE_SCENE_PLAN",
        "formal_evaluation": "NOT_STARTED",
        "failures": failures,
    }
    clean = dict(payload)
    clean.pop("canonical_sha256", None)
    payload["canonical_sha256"] = hashlib.sha256(
        json.dumps(clean, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if payload["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
