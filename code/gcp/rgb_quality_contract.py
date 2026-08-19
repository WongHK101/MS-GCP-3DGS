#!/usr/bin/env python3
"""Shared contracts for native-quarter heldout RGB render adapters."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


SUITE_ID = "m3m_gcp_native_quarter_rgb_quality_v1"
CONTRACT_SCHEMA = "m3m_gcp_native_quarter_rgb_quality_contract_v1"
RENDER_MANIFEST_SCHEMA = "m3m_gcp_native_quarter_rgb_render_manifest_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def git_identity(repo: Path) -> dict[str, str]:
    repo = repo.expanduser().resolve()

    def run(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception as exc:  # noqa: BLE001
            return f"ERROR:{type(exc).__name__}:{exc}"

    try:
        tracked_diff = subprocess.check_output(
            ["git", "-C", str(repo), "diff", "HEAD", "--binary", "--no-ext-diff"],
            stderr=subprocess.DEVNULL,
        )
        diff_sha = hashlib.sha256(tracked_diff).hexdigest()
        modified_names = subprocess.check_output(
            ["git", "-C", str(repo), "diff", "HEAD", "--name-only", "-z"],
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").split("\0")
        modified_files = {
            name: sha256_file(repo / name)
            for name in modified_names
            if name and (repo / name).is_file()
        }
    except Exception as exc:  # noqa: BLE001
        diff_sha = f"ERROR:{type(exc).__name__}:{exc}"
        modified_files = {}
    return {
        "path": str(repo),
        "commit": run("rev-parse", "HEAD"),
        "tree": run("rev-parse", "HEAD^{tree}"),
        "status_porcelain_v1": run("status", "--porcelain=v1"),
        "tracked_diff_sha256": diff_sha,
        "tracked_modified_files_sha256": modified_files,
    }


def load_contract(path: Path, *, allow_review_candidate: bool = True) -> dict[str, Any]:
    path = path.expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != CONTRACT_SCHEMA or payload.get("suite_id") != SUITE_ID:
        raise ValueError(f"not the {SUITE_ID} contract: {path}")
    allowed = {"ACTIVE_FROZEN"}
    if allow_review_candidate:
        allowed.add("REVIEW_CANDIDATE_NOT_FORMAL")
    if payload.get("status") not in allowed:
        raise ValueError(f"contract status is not executable: {payload.get('status')}")
    return payload


def load_bound_input_manifest(
    contract: dict[str, Any], input_manifest_path: Path, scene: str
) -> dict[str, Any]:
    input_manifest_path = input_manifest_path.expanduser().resolve()
    bindings = contract["input_binding"]["scene_bindings"]
    if scene not in bindings:
        raise ValueError(f"scene is not bound by RGB quality contract: {scene}")
    binding = bindings[scene]
    actual_file_hash = sha256_file(input_manifest_path)
    if actual_file_hash != binding["formal_input_manifest_file_sha256"]:
        raise ValueError(
            f"input manifest SHA mismatch: {actual_file_hash} != "
            f"{binding['formal_input_manifest_file_sha256']}"
        )
    payload = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    expected = {
        "scene": scene,
        "manifest_sha256": binding["formal_input_manifest_canonical_sha256"],
        "release_root_digest_sha256": contract["input_binding"]["release_root_digest_sha256"],
        "pixel_domain": contract["input_binding"]["pixel_domain"],
        "holdout_semantics": contract["input_binding"]["holdout_semantics"],
        "full_view_count": binding["full_view_count"],
        "train_view_count": binding["train_view_count"],
        "test_view_count": binding["test_view_count"],
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"input manifest field mismatch for {key}: {payload.get(key)!r} != {value!r}")
    images = payload.get("images", [])
    if len(images) != binding["full_view_count"]:
        raise ValueError("input manifest image count mismatch")
    names = [str(row["image_name"]) for row in images]
    if len(set(names)) != len(names):
        raise ValueError("duplicate image name in input manifest")
    test_rows = [row for row in images if row.get("role") == "test"]
    train_rows = [row for row in images if row.get("role") == "train"]
    if len(test_rows) != binding["test_view_count"] or len(train_rows) != binding["train_view_count"]:
        raise ValueError("input manifest role counts mismatch")
    for row in images:
        if int(row["width"]) != binding["width"] or int(row["height"]) != binding["height"]:
            raise ValueError(f"unexpected image shape in input manifest: {row['image_name']}")
    return payload


def role_rows(input_manifest: dict[str, Any], role: str) -> list[dict[str, Any]]:
    rows = [dict(row) for row in input_manifest["images"] if row.get("role") == role]
    return sorted(rows, key=lambda row: (int(row["image_id"]), str(row["image_name"])))


def _to_chw_float32(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3 or array.shape[0] != 3:
        raise ValueError(f"RGB render must have shape [3,H,W], got {array.shape}")
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("RGB render contains NaN or infinity")
    return array


class RgbRenderWriter:
    """Write a fail-closed set of method renders and its identity manifest."""

    def __init__(
        self,
        *,
        contract_path: Path,
        input_manifest_path: Path,
        scene: str,
        method_id: str,
        output_dir: Path,
        manifest_path: Path | None = None,
    ) -> None:
        self.contract_path = contract_path.expanduser().resolve()
        self.contract = load_contract(self.contract_path)
        self.input_manifest_path = input_manifest_path.expanduser().resolve()
        self.input_manifest = load_bound_input_manifest(
            self.contract, self.input_manifest_path, scene
        )
        self.scene = scene
        self.method_id = method_id
        self.output_dir = output_dir.expanduser().resolve()
        self.manifest_path = (
            manifest_path.expanduser().resolve()
            if manifest_path
            else self.output_dir.parent / "rgb_render_manifest.json"
        )
        if self.output_dir.exists():
            raise FileExistsError(self.output_dir)
        if self.manifest_path.exists():
            raise FileExistsError(self.manifest_path)
        self.output_dir.mkdir(parents=True)
        self.expected_rows = role_rows(self.input_manifest, "test")
        self.expected_by_name = {str(row["image_name"]): row for row in self.expected_rows}
        stems = [Path(name).stem for name in self.expected_by_name]
        if len(set(stems)) != len(stems):
            raise ValueError("test image stems are not unique")
        self.rows: list[dict[str, Any]] = []
        self._written: set[str] = set()

    @property
    def expected_names(self) -> list[str]:
        return [str(row["image_name"]) for row in self.expected_rows]

    def allowlist_map(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for name in self.expected_names:
            for alias in (name, Path(name).name, Path(name).stem):
                if alias in result and result[alias] != name:
                    raise ValueError(f"ambiguous image alias: {alias}")
                result[alias] = name
        return result

    def save(self, image_name: str, rgb: Any, *, camera_record: dict[str, Any] | None = None) -> dict[str, Any]:
        supplied_name = str(image_name)
        image_name = Path(supplied_name).name
        if supplied_name != image_name:
            raise ValueError(f"render image name must be a basename: {supplied_name!r}")
        if image_name not in self.expected_by_name:
            raise ValueError(f"render is not a frozen test image: {image_name}")
        if image_name in self._written:
            raise ValueError(f"duplicate RGB render: {image_name}")
        expected = self.expected_by_name[image_name]
        array = _to_chw_float32(rgb)
        height, width = int(array.shape[1]), int(array.shape[2])
        if width != int(expected["width"]) or height != int(expected["height"]):
            raise ValueError(
                f"render shape mismatch for {image_name}: {(width, height)} != "
                f"{(expected['width'], expected['height'])}"
            )
        below = int(np.count_nonzero(array < 0.0))
        above = int(np.count_nonzero(array > 1.0))
        quantized = np.floor(np.clip(array, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        quantized = np.moveaxis(quantized, 0, 2)
        output_path = self.output_dir / f"{Path(image_name).stem}.png"
        Image.fromarray(quantized, mode="RGB").save(
            output_path, format="PNG", optimize=False, compress_level=6
        )
        with Image.open(output_path) as check:
            if check.mode != "RGB" or check.size != (width, height):
                raise RuntimeError(f"written PNG identity check failed: {output_path}")
        row = {
            "index": len(self.rows),
            "image_name": image_name,
            "source_image_id": int(expected["image_id"]),
            "prediction_relative_path": output_path.relative_to(self.manifest_path.parent).as_posix(),
            "prediction_png_sha256": sha256_file(output_path),
            "prediction_png_bytes": output_path.stat().st_size,
            "width": width,
            "height": height,
            "mode": "RGB",
            "source_tensor_dtype": "float32",
            "source_tensor_min": float(array.min()),
            "source_tensor_max": float(array.max()),
            "clipped_below_zero_value_count": below,
            "clipped_above_one_value_count": above,
            "ground_truth_relative_path": str(expected["relative_path"]),
            "ground_truth_jpeg_sha256": str(expected["jpeg_sha256"]),
            "camera_record": camera_record or {},
        }
        self.rows.append(row)
        self._written.add(image_name)
        return row

    def finalize(self, *, provenance: dict[str, Any]) -> dict[str, Any]:
        missing = sorted(set(self.expected_by_name) - self._written)
        extra = sorted(self._written - set(self.expected_by_name))
        if missing or extra or len(self.rows) != len(self.expected_rows):
            raise RuntimeError(f"incomplete RGB render set: missing={missing} extra={extra}")
        payload = {
            "schema": RENDER_MANIFEST_SCHEMA,
            "suite_id": SUITE_ID,
            "contract_status": self.contract["status"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "scene": self.scene,
            "method_id": self.method_id,
            "contract_path": str(self.contract_path),
            "contract_file_sha256": sha256_file(self.contract_path),
            "input_manifest_path": str(self.input_manifest_path),
            "input_manifest_file_sha256": sha256_file(self.input_manifest_path),
            "input_manifest_canonical_sha256": self.input_manifest["manifest_sha256"],
            "source_data_release_root_digest_sha256": self.input_manifest[
                "release_root_digest_sha256"
            ],
            "pixel_domain": self.input_manifest["pixel_domain"],
            "holdout_semantics": self.input_manifest["holdout_semantics"],
            "required_test_view_count": len(self.expected_rows),
            "rendered_test_view_count": len(self.rows),
            "complete_test_coverage": True,
            "prediction_encoding": self.contract["prediction_contract"],
            "renders": self.rows,
            "provenance": provenance,
        }
        write_json(self.manifest_path, payload)
        return payload


def arithmetic_mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    if not items:
        raise ValueError("mean requires at least one value")
    if any(math.isinf(value) and value > 0 for value in items):
        return math.inf
    if not all(math.isfinite(value) for value in items):
        raise ValueError("mean contains a non-finite value")
    return math.fsum(items) / len(items)
