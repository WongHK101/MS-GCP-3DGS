"""Shared promoted 100K geometry artifact-path bindings."""

from __future__ import annotations

from pathlib import Path
from typing import Any


LIDAR_FULL_TRAIN_PACKET_ROOT_NAME = "lidar_packets_100k_success_v3"
LIDAR_HELDOUT_CANDIDATE_PACKET_ROOT_NAME = (
    "lidar_packets_100k_heldout_candidate_v1"
)


def lidar_full_train_packet_root(run_root: Path) -> Path:
    return (
        Path(run_root).expanduser().resolve()
        / "formal_evaluation"
        / LIDAR_FULL_TRAIN_PACKET_ROOT_NAME
    )


def lidar_full_train_packet_manifest(run_root: Path) -> Path:
    return lidar_full_train_packet_root(run_root) / "depth_export_manifest.json"


def lidar_heldout_candidate_packet_root(run_root: Path) -> Path:
    return (
        Path(run_root).expanduser().resolve()
        / "formal_evaluation"
        / LIDAR_HELDOUT_CANDIDATE_PACKET_ROOT_NAME
    )


def lidar_heldout_candidate_packet_manifest(run_root: Path) -> Path:
    return (
        lidar_heldout_candidate_packet_root(run_root)
        / "depth_export_manifest.json"
    )


def formal_input_manifest_canonical_sha256(payload: dict[str, Any]) -> str:
    """Read the native-quarter manifest's declared canonical identity field."""

    value = str(payload.get("manifest_sha256", ""))
    if len(value) != 64:
        raise ValueError("formal input manifest lacks a valid manifest_sha256")
    return value
