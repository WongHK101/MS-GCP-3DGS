#!/usr/bin/env python3
"""Dependency-light validators for immutable 100K phase products."""

from __future__ import annotations

import hashlib
import math
import struct
import zipfile
from pathlib import Path
from typing import Any


PLY_TYPES: dict[str, tuple[str, int]] = {
    "char": ("b", 1),
    "int8": ("b", 1),
    "uchar": ("B", 1),
    "uint8": ("B", 1),
    "short": ("h", 2),
    "int16": ("h", 2),
    "ushort": ("H", 2),
    "uint16": ("H", 2),
    "int": ("i", 4),
    "int32": ("i", 4),
    "uint": ("I", 4),
    "uint32": ("I", 4),
    "float": ("f", 4),
    "float32": ("f", 4),
    "double": ("d", 8),
    "float64": ("d", 8),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ply_header(path: Path, *, method_id: str) -> tuple[int, list[tuple[str, str]], int]:
    if not path.is_file():
        raise RuntimeError(f"Gaussian PLY is missing: {path}")
    with path.open("rb") as handle:
        data = handle.read(1024 * 1024)
    marker = b"end_header\n"
    index = data.find(marker)
    if index < 0:
        marker = b"end_header\r\n"
        index = data.find(marker)
    if index < 0:
        raise RuntimeError(f"Gaussian PLY has no bounded end_header: {path}")
    header_bytes = data[: index + len(marker)]
    try:
        lines = header_bytes.decode("ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"Gaussian PLY header is not ASCII: {path}") from exc
    if not lines or lines[0] != "ply" or "format binary_little_endian 1.0" not in lines:
        raise RuntimeError(f"Gaussian PLY is not binary little-endian PLY 1.0: {path}")
    vertex_count: int | None = None
    vertex_properties: list[tuple[str, str]] = []
    current_element: str | None = None
    for line in lines[1:]:
        parts = line.split()
        if not parts or parts[0] in {"comment", "obj_info", "format", "end_header"}:
            continue
        if parts[0] == "element":
            if len(parts) != 3:
                raise RuntimeError(f"malformed PLY element declaration: {path}")
            current_element = parts[1]
            count = int(parts[2])
            if current_element == "vertex":
                if vertex_count is not None:
                    raise RuntimeError(f"duplicate PLY vertex element: {path}")
                vertex_count = count
            elif count != 0:
                raise RuntimeError(f"Gaussian PLY carries an unexpected non-empty element: {path}")
        elif parts[0] == "property" and current_element == "vertex":
            if len(parts) != 3 or parts[1] == "list" or parts[1] not in PLY_TYPES:
                raise RuntimeError(f"unsupported Gaussian PLY vertex property: {line}")
            vertex_properties.append((parts[2], parts[1]))
    if vertex_count is None or vertex_count <= 0:
        raise RuntimeError(f"Gaussian PLY vertex count is not positive: {path}")
    names = [name for name, _ in vertex_properties]
    if len(names) != len(set(names)):
        raise RuntimeError(f"Gaussian PLY has duplicate vertex properties: {path}")
    common = {"x", "y", "z", "opacity", "rot_0", "rot_1", "rot_2", "rot_3"}
    if method_id == "citygs_x":
        required = {
            *common, "level", "extra_level", "info",
            *{f"f_offset_{index}" for index in range(30)},
            *{f"f_anchor_feat_{index}" for index in range(32)},
            *{f"scale_{index}" for index in range(6)},
        }
    else:
        rest_count = 24 if method_id == "metrogs" else 45
        scale_count = 2 if method_id in {"2dgs", "metrogs"} else 3
        required = {
            *common, "f_dc_0", "f_dc_1", "f_dc_2",
            *{f"f_rest_{index}" for index in range(rest_count)},
            *{f"scale_{index}" for index in range(scale_count)},
        }
    missing = sorted(required - set(names))
    if missing:
        raise RuntimeError(
            f"Gaussian PLY lacks required trained-Gaussian fields: {path}: {missing}"
        )
    return vertex_count, vertex_properties, len(header_bytes)


def validate_gaussian_ply(path: Path, *, method_id: str = "3dgs_original") -> dict[str, Any]:
    """Validate header, Gaussian schema, binary extent and sampled finite values."""
    path = path.resolve()
    vertex_count, properties, header_bytes = _ply_header(path, method_id=method_id)
    record_bytes = sum(PLY_TYPES[type_name][1] for _, type_name in properties)
    expected_bytes = header_bytes + vertex_count * record_bytes
    if path.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"Gaussian PLY binary extent mismatch: {path}: "
            f"expected {expected_bytes}, got {path.stat().st_size}"
        )
    offsets: dict[str, tuple[int, str]] = {}
    offset = 0
    for name, type_name in properties:
        format_code, width = PLY_TYPES[type_name]
        offsets[name] = (offset, format_code)
        offset += width
    sampled_candidates = {
        "x", "y", "z", "opacity", "rot_0", "rot_1", "rot_2", "rot_3"
    } | (
        {"level", "extra_level", "info", "scale_0", "scale_1", "scale_2"}
        if method_id == "citygs_x"
        else {"f_dc_0", "f_dc_1", "f_dc_2", "scale_0", "scale_1"}
    )
    sampled_fields = sorted(name for name in sampled_candidates if name in offsets)
    sample_indices = sorted({0, vertex_count // 2, vertex_count - 1})
    with path.open("rb") as handle:
        for vertex_index in sample_indices:
            base = header_bytes + vertex_index * record_bytes
            for name in sampled_fields:
                property_offset, format_code = offsets[name]
                handle.seek(base + property_offset)
                width = struct.calcsize("<" + format_code)
                raw = handle.read(width)
                if len(raw) != width:
                    raise RuntimeError(f"Gaussian PLY sample is truncated: {path}")
                value = struct.unpack("<" + format_code, raw)[0]
                if isinstance(value, float) and not math.isfinite(value):
                    raise RuntimeError(
                        f"Gaussian PLY contains non-finite sampled {name}: {path}"
                    )
    property_hash = hashlib.sha256(
        "\n".join(f"{type_name} {name}" for name, type_name in properties).encode("ascii")
    ).hexdigest()
    return {
        "kind": "gaussian_ply_v1",
        "method_id": method_id,
        "vertex_count": vertex_count,
        "vertex_record_bytes": record_bytes,
        "property_schema_sha256": property_hash,
    }


def validate_torch_checkpoint(path: Path) -> dict[str, Any]:
    """Validate a PyTorch container without importing or unpickling model code."""
    path = path.resolve()
    if not path.is_file() or path.stat().st_size < 1024:
        raise RuntimeError(f"PyTorch checkpoint is missing or implausibly small: {path}")
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if archive.testzip() is not None:
                raise RuntimeError(f"PyTorch checkpoint ZIP CRC failed: {path}")
        basenames = {Path(name).name for name in names}
        if "data.pkl" not in basenames or "version" not in basenames:
            raise RuntimeError(f"ZIP is not a PyTorch checkpoint container: {path}")
        container = "pytorch_zip"
    else:
        with path.open("rb") as handle:
            prefix = handle.read(2)
        if len(prefix) != 2 or prefix[0] != 0x80:
            raise RuntimeError(f"file is not a recognized PyTorch checkpoint: {path}")
        container = "legacy_pickle_stream"
    return {"kind": "torch_checkpoint_v1", "container": container}


def validate_npz(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file() or not zipfile.is_zipfile(path):
        raise RuntimeError(f"NPZ container is missing or invalid: {path}")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if archive.testzip() is not None or not names or any(not name.endswith(".npy") for name in names):
            raise RuntimeError(f"NPZ member inventory/CRC is invalid: {path}")
    return {
        "kind": "npz_container_v1",
        "member_count": len(names),
        "member_names_sha256": hashlib.sha256("\n".join(sorted(names)).encode("utf-8")).hexdigest(),
    }


def phase_product_row(
    path: Path, *, validate_model_container: bool, method_id: str | None = None
) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"phase product is missing or empty: {path}")
    validation: dict[str, Any] = {"kind": "hash_bound_file_v1"}
    if validate_model_container:
        if path.suffix.lower() == ".ply":
            if not method_id:
                raise RuntimeError("Gaussian model product validation requires a method ID")
            validation = validate_gaussian_ply(path, method_id=method_id)
        elif path.suffix.lower() in {".ckpt", ".pth"}:
            validation = validate_torch_checkpoint(path)
        elif path.suffix.lower() == ".npz":
            validation = validate_npz(path)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "validation": validation,
    }


def revalidate_phase_product_row(row: dict[str, Any]) -> Path:
    if set(row) != {"path", "bytes", "sha256", "validation"}:
        raise RuntimeError("phase product row field inventory mismatch")
    path = Path(str(row.get("path", "")))
    if not path.is_absolute() or not path.is_file():
        raise RuntimeError(f"phase product file is missing: {path}")
    if path.stat().st_size != row.get("bytes") or sha256_file(path) != row.get("sha256"):
        raise RuntimeError(f"phase product file changed: {path}")
    validation = row.get("validation")
    if not isinstance(validation, dict):
        raise RuntimeError(f"phase product validation metadata is invalid: {path}")
    kind = validation.get("kind")
    if kind == "gaussian_ply_v1":
        method_id = str(validation.get("method_id", ""))
        actual = validate_gaussian_ply(path, method_id=method_id)
    elif kind == "torch_checkpoint_v1":
        actual = validate_torch_checkpoint(path)
    elif kind == "npz_container_v1":
        actual = validate_npz(path)
    elif kind == "hash_bound_file_v1" and set(validation) == {"kind"}:
        actual = {"kind": "hash_bound_file_v1"}
    else:
        raise RuntimeError(f"unsupported phase product validator: {path}: {kind}")
    if actual != validation:
        raise RuntimeError(f"phase product validation metadata changed: {path}")
    return path.resolve()
