#!/usr/bin/env python3
"""Query one remote COLMAP native-quarter job without modifying it."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shlex
from pathlib import Path


SCENE_PATTERN = re.compile(r"^gcp_[0-9]+_[0-9]{8}$")


def load_connector(path: Path):
    spec = importlib.util.spec_from_file_location("autodl901_connector", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import AutoDL connector: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--connector", required=True, type=Path)
    parser.add_argument("--remote-batch-root", required=True)
    args = parser.parse_args()
    if not SCENE_PATTERN.fullmatch(args.scene):
        raise ValueError(f"Invalid scene name: {args.scene}")
    remote_root = f"{args.remote_batch_root.rstrip('/')}/{args.scene}"
    if not remote_root.startswith("/root/autodl-tmp/runs/gs-gcp-v13/"):
        raise ValueError(f"Refusing unexpected remote root: {remote_root}")

    connector = load_connector(args.connector)
    client = connector.connect(connector.credential())
    try:
        root = shlex.quote(remote_root)
        raw = connector.run_remote(
            client,
            "set -u; "
            f"r={root}; "
            "state=$(test -f \"$r/state.txt\" && cat \"$r/state.txt\" || echo ABSENT); "
            "output_images=$(find \"$r/output/images\" -maxdepth 1 -type f 2>/dev/null | wc -l); "
            "last_progress=$(test -f \"$r/undistort.stderr.log\" && "
            "grep -a 'Undistorting image' \"$r/undistort.stderr.log\" | tail -1 || true); "
            "launcher_pid=$(test -f \"$r/launcher.pid\" && cat \"$r/launcher.pid\" || echo NA); "
            "colmap_pid=$(test -f \"$r/colmap.pid\" && cat \"$r/colmap.pid\" || echo NA); "
            "elapsed_seconds=$(test -f \"$r/elapsed_seconds.txt\" && cat \"$r/elapsed_seconds.txt\" || echo NA); "
            "memory_current=$(cat /sys/fs/cgroup/memory.current); "
            "memory_peak=$(cat /sys/fs/cgroup/memory.peak 2>/dev/null || echo NA); "
            "oom_kill=$(awk '$1==\"oom_kill\" {print $2}' /sys/fs/cgroup/memory.events); "
            "disk_available=$(df -B1 --output=avail /root/autodl-tmp | tail -1); "
            "printf 'state=%s\\noutput_images=%s\\nlast_progress=%s\\n' "
            "\"$state\" \"$output_images\" \"$last_progress\"; "
            "printf 'launcher_pid=%s\\ncolmap_pid=%s\\nelapsed_seconds=%s\\n' "
            "\"$launcher_pid\" \"$colmap_pid\" \"$elapsed_seconds\"; "
            "printf 'memory_current=%s\\nmemory_peak=%s\\noom_kill=%s\\ndisk_available=%s\\n' "
            "\"$memory_current\" \"$memory_peak\" \"$oom_kill\" \"$disk_available\"",
            timeout=60,
        )
    finally:
        client.close()
    values: dict[str, object] = {"scene": args.scene, "remote_root": remote_root}
    for line in raw.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    for key in (
        "output_images",
        "launcher_pid",
        "colmap_pid",
        "elapsed_seconds",
        "memory_current",
        "memory_peak",
        "oom_kill",
        "disk_available",
    ):
        value = values.get(key)
        if isinstance(value, str) and value.isdigit():
            values[key] = int(value)
    print(json.dumps(values, ensure_ascii=False))


if __name__ == "__main__":
    main()
