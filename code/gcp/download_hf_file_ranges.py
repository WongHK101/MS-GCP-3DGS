#!/usr/bin/env python3
"""Download an immutable Hugging Face file as independently verified byte ranges.

This utility is intended for unreliable mirrors where a resumed ``curl -C -``
request may occasionally be answered with a full response and truncate the
destination.  Each range is written to its own file, accepted only after an
exact HTTP Content-Range and byte-count check, and the final target is replaced
only after its complete SHA-256 digest matches the frozen expectation.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from typing import Iterable


CONTENT_RANGE_RE = re.compile(
    r"^content-range:\s*bytes\s+(\d+)-(\d+)/(\d+)\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)
CHUNK_NAME_RE = re.compile(r"^chunk_(\d+)_(\d+)\.bin$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ranges(total_size: int, chunk_size: int) -> Iterable[tuple[int, int]]:
    for start in range(0, total_size, chunk_size):
        yield start, min(total_size - 1, start + chunk_size - 1)


def chunk_path(chunk_dir: Path, start: int, end: int) -> Path:
    return chunk_dir / f"chunk_{start:012d}_{end:012d}.bin"


def valid_existing_chunk(path: Path, expected_size: int) -> bool:
    return path.is_file() and path.stat().st_size == expected_size


def seed_from_larger_chunks(
    source_dir: Path,
    destination_dir: Path,
    requested_ranges: list[tuple[int, int]],
) -> int:
    """Split previously validated range files into smaller resumable ranges."""
    if not source_dir.is_dir() or source_dir.resolve() == destination_dir.resolve():
        return 0
    source_ranges: list[tuple[int, int, Path]] = []
    for path in sorted(source_dir.iterdir()):
        match = CHUNK_NAME_RE.fullmatch(path.name)
        if match is None or not path.is_file():
            continue
        start, end = int(match.group(1)), int(match.group(2))
        if end < start or path.stat().st_size != end - start + 1:
            continue
        source_ranges.append((start, end, path))

    seeded = 0
    for start, end in requested_ranges:
        destination = chunk_path(destination_dir, start, end)
        expected_length = end - start + 1
        if valid_existing_chunk(destination, expected_length):
            continue
        containing = next(
            (
                (source_start, source_path)
                for source_start, source_end, source_path in source_ranges
                if source_start <= start and end <= source_end
            ),
            None,
        )
        if containing is None:
            continue
        source_start, source_path = containing
        temporary = destination.with_name(
            destination.name + f".seed.{os.getpid()}.tmp"
        )
        try:
            with source_path.open("rb") as source:
                source.seek(start - source_start)
                payload = source.read(expected_length)
            if len(payload) != expected_length:
                raise RuntimeError(f"short seed read from {source_path}")
            with temporary.open("wb") as output:
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, destination)
            seeded += 1
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    return seeded


def download_one(
    *,
    url: str,
    chunk_dir: Path,
    start: int,
    end: int,
    total_size: int,
    max_attempts: int,
    connect_timeout: int,
    max_time: int,
) -> tuple[int, int, bool]:
    destination = chunk_path(chunk_dir, start, end)
    expected_length = end - start + 1
    if valid_existing_chunk(destination, expected_length):
        return start, end, True

    for attempt in range(1, max_attempts + 1):
        unique = f"{os.getpid()}.{threading.get_ident()}.{attempt}"
        body_tmp = destination.with_name(destination.name + f".{unique}.tmp")
        headers_tmp = destination.with_name(destination.name + f".{unique}.headers")
        try:
            command = [
                "curl",
                "--location",
                "--http1.1",
                "--fail",
                "--silent",
                "--show-error",
                "--connect-timeout",
                str(connect_timeout),
                "--max-time",
                str(max_time),
                "--header",
                "Accept-Encoding: identity",
                "--range",
                f"{start}-{end}",
                "--dump-header",
                str(headers_tmp),
                "--output",
                str(body_tmp),
                url,
            ]
            completed = subprocess.run(
                command,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"curl exited {completed.returncode}: {completed.stderr.strip()}"
                )
            headers = headers_tmp.read_text(encoding="latin-1")
            received_ranges = {
                (int(found_start), int(found_end), int(found_total))
                for found_start, found_end, found_total in CONTENT_RANGE_RE.findall(headers)
            }
            wanted_range = (start, end, total_size)
            if wanted_range not in received_ranges:
                raise RuntimeError(
                    f"missing exact Content-Range {start}-{end}/{total_size}; "
                    f"received={sorted(received_ranges)}"
                )
            actual_length = body_tmp.stat().st_size
            if actual_length != expected_length:
                raise RuntimeError(
                    f"byte count {actual_length} != expected {expected_length}"
                )
            os.replace(body_tmp, destination)
            headers_tmp.unlink(missing_ok=True)
            return start, end, False
        except Exception as error:  # retry all transport/validation failures
            body_tmp.unlink(missing_ok=True)
            headers_tmp.unlink(missing_ok=True)
            if attempt == max_attempts:
                raise RuntimeError(
                    f"range {start}-{end} failed after {max_attempts} attempts: {error}"
                ) from error
            delay = min(30.0, 2.0 ** min(attempt - 1, 5))
            print(
                f"RETRY range={start}-{end} attempt={attempt}/{max_attempts} "
                f"delay_s={delay:g} error={error}",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)

    raise AssertionError("unreachable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-size", required=True, type=int)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--chunk-dir", required=True, type=Path)
    parser.add_argument("--seed-from-chunk-dir", type=Path)
    parser.add_argument("--chunk-size", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=12)
    parser.add_argument("--connect-timeout", type=int, default=20)
    parser.add_argument("--max-time", type=int, default=1800)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    expected_sha256 = args.expected_sha256.lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("--expected-sha256 must contain 64 hexadecimal characters")
    if args.expected_size <= 0 or args.chunk_size <= 0 or args.workers <= 0:
        raise ValueError("sizes and worker count must be positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.chunk_dir.mkdir(parents=True, exist_ok=True)

    if args.output.is_file() and args.output.stat().st_size == args.expected_size:
        existing_sha256 = sha256_file(args.output)
        if existing_sha256 == expected_sha256:
            print(f"ALREADY_COMPLETE output={args.output} sha256={existing_sha256}")
            return 0

    requested_ranges = list(ranges(args.expected_size, args.chunk_size))
    seeded_count = 0
    if args.seed_from_chunk_dir is not None:
        seeded_count = seed_from_larger_chunks(
            args.seed_from_chunk_dir.resolve(), args.chunk_dir, requested_ranges
        )
        print(
            f"SEEDED chunks={seeded_count} source={args.seed_from_chunk_dir}",
            flush=True,
        )
    print(
        f"START url={args.url} output={args.output} size={args.expected_size} "
        f"chunks={len(requested_ranges)} workers={args.workers}",
        flush=True,
    )
    completed_count = 0
    reused_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                download_one,
                url=args.url,
                chunk_dir=args.chunk_dir,
                start=start,
                end=end,
                total_size=args.expected_size,
                max_attempts=args.max_attempts,
                connect_timeout=args.connect_timeout,
                max_time=args.max_time,
            ): (start, end)
            for start, end in requested_ranges
        }
        for future in concurrent.futures.as_completed(futures):
            start, end, reused = future.result()
            completed_count += 1
            reused_count += int(reused)
            print(
                f"CHUNK_OK {completed_count}/{len(requested_ranges)} "
                f"range={start}-{end} reused={int(reused)}",
                flush=True,
            )

    assembled_tmp = args.output.with_name(
        args.output.name + f".assembled.{os.getpid()}.tmp"
    )
    digest = hashlib.sha256()
    bytes_written = 0
    try:
        with assembled_tmp.open("wb") as assembled:
            for start, end in requested_ranges:
                source = chunk_path(args.chunk_dir, start, end)
                expected_length = end - start + 1
                if not valid_existing_chunk(source, expected_length):
                    raise RuntimeError(f"missing or invalid chunk before assembly: {source}")
                with source.open("rb") as handle:
                    for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                        assembled.write(block)
                        digest.update(block)
                        bytes_written += len(block)
            assembled.flush()
            os.fsync(assembled.fileno())
        actual_sha256 = digest.hexdigest()
        if bytes_written != args.expected_size:
            raise RuntimeError(
                f"assembled size {bytes_written} != expected {args.expected_size}"
            )
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"assembled SHA-256 {actual_sha256} != expected {expected_sha256}"
            )
        os.replace(assembled_tmp, args.output)
        args.output.chmod(0o444)
    except Exception:
        assembled_tmp.unlink(missing_ok=True)
        raise

    manifest = {
        "url": args.url,
        "output": str(args.output),
        "expected_size": args.expected_size,
        "sha256": expected_sha256,
        "chunk_size": args.chunk_size,
        "chunk_count": len(requested_ranges),
        "reused_chunk_count": reused_count,
        "seeded_chunk_count": seeded_count,
        "seed_source": (
            str(args.seed_from_chunk_dir.resolve())
            if args.seed_from_chunk_dir is not None
            else None
        ),
    }
    manifest_path = args.output.with_name(args.output.name + ".download.json")
    manifest_tmp = manifest_path.with_name(manifest_path.name + f".{os.getpid()}.tmp")
    manifest_tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(manifest_tmp, manifest_path)
    print(f"COMPLETE output={args.output} sha256={expected_sha256}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
