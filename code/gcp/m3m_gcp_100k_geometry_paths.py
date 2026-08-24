"""Shared promoted 100K geometry artifact-path bindings."""

from __future__ import annotations

from pathlib import Path


LIDAR_FULL_TRAIN_PACKET_ROOT_NAME = "lidar_packets_100k_success_v3"


def lidar_full_train_packet_root(run_root: Path) -> Path:
    return (
        Path(run_root).expanduser().resolve()
        / "formal_evaluation"
        / LIDAR_FULL_TRAIN_PACKET_ROOT_NAME
    )


def lidar_full_train_packet_manifest(run_root: Path) -> Path:
    return lidar_full_train_packet_root(run_root) / "depth_export_manifest.json"
