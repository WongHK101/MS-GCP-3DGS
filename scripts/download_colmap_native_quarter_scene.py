#!/usr/bin/env python3
"""Download and verify one completed COLMAP native-quarter scene."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
import re
import shlex
import threading
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


SCENE_PATTERN = re.compile(r"^gcp_[0-9]+_[0-9]{8}$")
CHUNK_SIZE = 4 * 1024 * 1024


def load_connector(path: Path):
    spec = importlib.util.spec_from_file_location("autodl901_connector", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import AutoDL connector: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Progress:
    def __init__(self, total_files: int, total_bytes: int, done_files: int, done_bytes: int):
        self.total_files = total_files
        self.total_bytes = total_bytes
        self.done_files = done_files
        self.done_bytes = done_bytes
        self.initial_bytes = done_bytes
        self.started = time.monotonic()
        self.lock = threading.Lock()
        self.stop = threading.Event()

    def add(self, count: int) -> None:
        with self.lock:
            self.done_bytes += count

    def finish(self) -> None:
        with self.lock:
            self.done_files += 1

    def snapshot(self, scene: str) -> dict[str, object]:
        with self.lock:
            elapsed = max(time.monotonic() - self.started, 0.001)
            copied = max(self.done_bytes - self.initial_bytes, 0)
            rate = copied / elapsed
            remaining = max(self.total_bytes - self.done_bytes, 0)
            return {
                "status": "downloading",
                "scene": scene,
                "files": f"{self.done_files}/{self.total_files}",
                "percent": round(100 * self.done_bytes / self.total_bytes, 3),
                "MiB_per_second": round(rate / 1024**2, 2),
                "eta_minutes": round(remaining / rate / 60, 1) if rate else None,
            }


def download_bucket(
    connector,
    password: str,
    remote_output: str,
    records: list[dict[str, object]],
    progress: Progress,
) -> None:
    client = connector.connect(password)
    sftp = client.open_sftp()
    try:
        for record in records:
            relative = str(record["relative"])
            destination = Path(str(record["destination"]))
            size = int(record["bytes"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            partial = destination.with_name(destination.name + ".part")
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > size:
                raise IOError(f"Oversized local partial: {partial}")
            remote_path = remote_output + "/" + relative
            with sftp.open(remote_path, "rb") as remote, partial.open("ab") as local:
                if offset:
                    remote.seek(offset)
                try:
                    remote.prefetch(size, max_concurrent_requests=64)
                except Exception:
                    pass
                while offset < size:
                    block = remote.read(min(CHUNK_SIZE, size - offset))
                    if not block:
                        raise IOError(
                            f"Unexpected remote EOF: {relative} ({offset}/{size})"
                        )
                    local.write(block)
                    offset += len(block)
                    progress.add(len(block))
            if partial.stat().st_size != size:
                raise IOError(f"Local partial size mismatch: {relative}")
            os.replace(partial, destination)
            progress.finish()
    finally:
        sftp.close()
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--connector", required=True, type=Path)
    parser.add_argument("--remote-batch-root", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if not SCENE_PATTERN.fullmatch(args.scene):
        raise ValueError(f"Invalid scene name: {args.scene}")
    if args.workers < 1 or args.workers > 16:
        raise ValueError("--workers must be between 1 and 16")
    if not args.connector.is_file():
        raise FileNotFoundError(args.connector)
    remote_root = f"{args.remote_batch_root.rstrip('/')}/{args.scene}"
    if not remote_root.startswith("/root/autodl-tmp/runs/gs-gcp-v13/"):
        raise ValueError(f"Refusing unexpected remote root: {remote_root}")
    remote_output = remote_root + "/output"

    preparation_path = args.candidate_root / "evidence" / "PREPARATION.json"
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    expected_images = int(preparation["raw_images"]["count"])
    images_root = args.candidate_root / "images"
    native_sparse = (
        args.candidate_root / "evidence" / "native_output" / "sparse"
    )
    images_root.mkdir(parents=True, exist_ok=True)
    native_sparse.mkdir(parents=True, exist_ok=True)

    connector = load_connector(args.connector)
    password = connector.credential()
    client = connector.connect(password)
    try:
        state = connector.run_remote(
            client,
            f"set -eu; test -f {shlex.quote(remote_root + '/state.txt')}; "
            f"cat {shlex.quote(remote_root + '/state.txt')}",
            timeout=30,
        ).strip()
        if state != "SUCCESS":
            raise RuntimeError(f"Remote scene is not complete: state={state!r}")
        raw_hashes = connector.run_remote(
            client,
            f"cat {shlex.quote(remote_root + '/output.sha256')}",
            timeout=120,
        )
        expected_hashes: dict[str, str] = {}
        for line in raw_hashes.splitlines():
            digest, relative = line.split(maxsplit=1)
            relative = relative.removeprefix("./")
            expected_hashes[relative] = digest.lower()
        inventory = connector.run_remote(
            client,
            f"set -eu; cd {shlex.quote(remote_output)}; "
            "find images sparse -type f -printf '%p\\t%s\\n' | LC_ALL=C sort",
            timeout=120,
        )
        records: list[dict[str, object]] = []
        for line in inventory.splitlines():
            relative, size_text = line.rsplit("\t", 1)
            if relative not in expected_hashes:
                raise ValueError(f"Remote output lacks hash for {relative}")
            pure = PurePosixPath(relative)
            if pure.parts[0] == "images" and len(pure.parts) == 2:
                destination = images_root / pure.name
            elif pure.parts[0] == "sparse" and len(pure.parts) == 2:
                destination = native_sparse / pure.name
            else:
                raise ValueError(f"Unexpected remote output path: {relative}")
            records.append(
                {
                    "relative": relative,
                    "bytes": int(size_text),
                    "sha256": expected_hashes[relative],
                    "destination": str(destination),
                }
            )
        image_records = [x for x in records if str(x["relative"]).startswith("images/")]
        sparse_names = {
            PurePosixPath(str(x["relative"])).name
            for x in records
            if str(x["relative"]).startswith("sparse/")
        }
        if len(image_records) != expected_images:
            raise ValueError(
                f"Remote image count mismatch: {len(image_records)} != {expected_images}"
            )
        required_sparse = {
            "cameras.bin",
            "images.bin",
            "points3D.bin",
            "rigs.bin",
            "frames.bin",
        }
        if not required_sparse.issubset(sparse_names):
            raise ValueError(f"Incomplete remote sparse model: {sparse_names}")

        pending = []
        done_files = 0
        done_bytes = 0
        for record in records:
            destination = Path(str(record["destination"]))
            size = int(record["bytes"])
            if destination.is_file() and destination.stat().st_size == size:
                done_files += 1
                done_bytes += size
                continue
            partial = destination.with_name(destination.name + ".part")
            if partial.exists():
                if partial.stat().st_size > size:
                    raise IOError(f"Oversized local partial: {partial}")
                done_bytes += partial.stat().st_size
            pending.append(record)
        total_bytes = sum(int(record["bytes"]) for record in records)
        progress = Progress(len(records), total_bytes, done_files, done_bytes)
        print(json.dumps(progress.snapshot(args.scene)), flush=True)
        buckets: list[list[dict[str, object]]] = [
            [] for _ in range(args.workers)
        ]
        bucket_bytes = [0] * args.workers
        for record in sorted(pending, key=lambda item: int(item["bytes"]), reverse=True):
            index = min(range(args.workers), key=bucket_bytes.__getitem__)
            buckets[index].append(record)
            bucket_bytes[index] += int(record["bytes"])

        def reporter() -> None:
            while not progress.stop.wait(30):
                print(json.dumps(progress.snapshot(args.scene)), flush=True)

        report_thread = threading.Thread(target=reporter, daemon=True)
        report_thread.start()
        try:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.workers
            ) as pool:
                futures = [
                    pool.submit(
                        download_bucket,
                        connector,
                        password,
                        remote_output,
                        bucket,
                        progress,
                    )
                    for bucket in buckets
                    if bucket
                ]
                for future in futures:
                    future.result()
        finally:
            progress.stop.set()
            report_thread.join(timeout=5)
        if progress.done_files != len(records) or progress.done_bytes != total_bytes:
            raise RuntimeError("Download counters did not reach expected totals")

        mismatches = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            actual_hashes = pool.map(
                sha256_file, [Path(str(record["destination"])) for record in records]
            )
            for record, actual in zip(records, actual_hashes):
                if actual != record["sha256"]:
                    mismatches.append(str(record["relative"]))
        if mismatches:
            raise ValueError(f"Downloaded SHA256 mismatches: {mismatches[:10]}")

        remote_evidence = args.candidate_root / "evidence" / "remote_901"
        remote_evidence.mkdir(parents=True, exist_ok=True)
        evidence_names = (
            "state.txt",
            "started_utc.txt",
            "finished_utc.txt",
            "exit_code.txt",
            "elapsed_seconds.txt",
            "launcher.pid",
            "colmap.pid",
            "resource.tsv",
            "input_sha256_check.log",
            "undistort.stdout.log",
            "undistort.stderr.log",
            "launcher.log",
            "output.sha256",
            "run_undistorter.sh",
        )
        sftp = client.open_sftp()
        try:
            for name in evidence_names:
                remote = remote_root + "/" + name
                local = remote_evidence / name
                try:
                    sftp.get(remote, str(local))
                except FileNotFoundError:
                    if name not in {"launcher.log"}:
                        raise
        finally:
            sftp.close()
    finally:
        client.close()

    report = {
        "schema": "gs-gcp-colmap-native-quarter-download-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scene": args.scene,
        "status": "pass",
        "remote_root": remote_root,
        "image_count": len(image_records),
        "file_count": len(records),
        "bytes": total_bytes,
        "sha256_mismatch_count": 0,
        "images_root": str(images_root.resolve()),
        "native_sparse_root": str(native_sparse.resolve()),
    }
    report_path = args.candidate_root / "evidence" / "DOWNLOAD.json"
    with report_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
