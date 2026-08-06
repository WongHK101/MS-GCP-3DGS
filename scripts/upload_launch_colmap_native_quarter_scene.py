#!/usr/bin/env python3
"""Resumably upload one raw scene and launch CPU-only COLMAP undistortion."""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import re
import shlex
import threading
import time
from pathlib import Path


SCENE_PATTERN = re.compile(r"^gcp_[0-9]+_[0-9]{8}$")
CHUNK_SIZE = 4 * 1024 * 1024


def load_connector(path: Path):
    spec = importlib.util.spec_from_file_location("autodl901_connector", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import AutoDL connector: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_manifest(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        name = name.strip()
        if Path(name).name != name or not name.endswith("_D.JPG"):
            raise ValueError(f"Unexpected manifest filename: {name}")
        records.append({"name": name, "sha256": digest.lower()})
    if not records or len({x["name"] for x in records}) != len(records):
        raise ValueError("Raw manifest is empty or contains duplicate names")
    return records


def remote_file_sizes(connector, client, root: str) -> dict[str, int]:
    command = (
        f"if [ -d {shlex.quote(root)} ]; then "
        f"find {shlex.quote(root)} -maxdepth 1 -type f -printf '%f\\t%s\\n'; "
        "fi"
    )
    sizes: dict[str, int] = {}
    for line in connector.run_remote(client, command, timeout=120).splitlines():
        name, size = line.rsplit("\t", 1)
        sizes[name] = int(size)
    return sizes


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
            transferred = max(self.done_bytes - self.initial_bytes, 0)
            rate = transferred / elapsed
            remaining = max(self.total_bytes - self.done_bytes, 0)
            return {
                "status": "uploading",
                "scene": scene,
                "files": f"{self.done_files}/{self.total_files}",
                "percent": round(100 * self.done_bytes / self.total_bytes, 3),
                "MiB_per_second": round(rate / 1024**2, 2),
                "eta_minutes": round(remaining / rate / 60, 1) if rate else None,
            }


def upload_bucket(
    connector,
    password: str,
    raw_root: Path,
    remote_images: str,
    records: list[dict[str, object]],
    progress: Progress,
) -> None:
    client = connector.connect(password)
    sftp = client.open_sftp()
    try:
        for record in records:
            name = str(record["name"])
            source = raw_root / name
            size = source.stat().st_size
            destination = f"{remote_images}/{name}"
            partial = f"{remote_images}/.{name}.part"
            offset = 0
            try:
                offset = sftp.stat(partial).st_size
            except FileNotFoundError:
                pass
            if offset > size:
                raise IOError(f"Oversized remote partial: {name} ({offset}>{size})")
            mode = "ab" if offset else "wb"
            with source.open("rb") as local, sftp.open(partial, mode) as remote:
                if offset:
                    local.seek(offset)
                try:
                    remote.set_pipelined(True)
                except AttributeError:
                    pass
                while offset < size:
                    block = local.read(min(CHUNK_SIZE, size - offset))
                    if not block:
                        raise IOError(f"Unexpected local EOF: {source}")
                    remote.write(block)
                    offset += len(block)
                    progress.add(len(block))
            if sftp.stat(partial).st_size != size:
                raise IOError(f"Remote partial size mismatch: {name}")
            try:
                sftp.posix_rename(partial, destination)
            except IOError:
                sftp.rename(partial, destination)
            progress.finish()
    finally:
        sftp.close()
        client.close()


def upload_small(sftp, source: Path, destination: str) -> None:
    partial = destination + ".part"
    sftp.put(str(source), partial)
    try:
        sftp.posix_rename(partial, destination)
    except IOError:
        sftp.rename(partial, destination)


def make_wrapper(
    remote_root: str, colmap: str, expected_count: int
) -> str:
    root = shlex.quote(remote_root)
    executable = shlex.quote(colmap)
    return f"""#!/usr/bin/env bash
set -uo pipefail
root={root}
state="$root/state.txt"
echo RUNNING > "$state"
date -u +%FT%TZ > "$root/started_utc.txt"
failure() {{
  rc=$?
  trap - ERR INT TERM
  if [[ "${{child:-}}" =~ ^[0-9]+$ ]] && kill -0 "$child" 2>/dev/null; then
    kill "$child" 2>/dev/null || true
    wait "$child" 2>/dev/null || true
  fi
  echo "$rc" > "$root/exit_code.txt"
  date -u +%FT%TZ > "$root/finished_utc.txt"
  echo FAILED > "$state"
  exit "$rc"
}}
trap failure ERR INT TERM
cd "$root/input/images"
sha256sum -c ../raw_images.sha256 > "$root/input_sha256_check.log" 2>&1
test "$(grep -c ': OK$' "$root/input_sha256_check.log")" -eq {expected_count}
test ! -e "$root/output"
start_epoch=$(date +%s)
env CUDA_VISIBLE_DEVICES= {executable} image_undistorter \\
  --image_path "$root/input/images" \\
  --input_path "$root/input/sparse/0" \\
  --output_path "$root/output" \\
  --output_type COLMAP \\
  --max_image_size 1414 \\
  --num_threads 1 \\
  > "$root/undistort.stdout.log" \\
  2> "$root/undistort.stderr.log" &
child=$!
echo "$child" > "$root/colmap.pid"
printf 'utc\\tmemory_current_bytes\\toutput_images\\n' > "$root/resource.tsv"
while kill -0 "$child" 2>/dev/null; do
  memory=$(cat /sys/fs/cgroup/memory.current 2>/dev/null || echo NA)
  if test -d "$root/output/images"; then
    output_count=$(find "$root/output/images" -maxdepth 1 -type f | wc -l)
  else
    output_count=0
  fi
  printf '%s\\t%s\\t%s\\n' "$(date -u +%FT%TZ)" "$memory" "$output_count" >> "$root/resource.tsv"
  sleep 60
done
wait "$child"
end_epoch=$(date +%s)
echo "$((end_epoch-start_epoch))" > "$root/elapsed_seconds.txt"
count=$(find "$root/output/images" -maxdepth 1 -type f | wc -l)
test "$count" -eq {expected_count}
for required in cameras.bin images.bin points3D.bin rigs.bin frames.bin; do
  test -f "$root/output/sparse/$required"
done
cd "$root/output"
find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > "$root/output.sha256"
echo 0 > "$root/exit_code.txt"
date -u +%FT%TZ > "$root/finished_utc.txt"
echo SUCCESS > "$state"
trap - ERR INT TERM
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--prepared-root", required=True, type=Path)
    parser.add_argument("--connector", required=True, type=Path)
    parser.add_argument("--remote-batch-root", required=True)
    parser.add_argument("--colmap", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if not SCENE_PATTERN.fullmatch(args.scene):
        raise ValueError(f"Invalid scene name: {args.scene}")
    if args.workers < 1 or args.workers > 16:
        raise ValueError("--workers must be between 1 and 16")
    for path in (args.raw_root, args.prepared_root, args.connector):
        if not path.exists():
            raise FileNotFoundError(path)
    remote_root = f"{args.remote_batch_root.rstrip('/')}/{args.scene}"
    if not remote_root.startswith("/root/autodl-tmp/runs/gs-gcp-v13/"):
        raise ValueError(f"Refusing unexpected remote root: {remote_root}")

    manifest_path = args.prepared_root / "raw_images.sha256"
    sparse_root = args.prepared_root / "sparse" / "0"
    records = load_manifest(manifest_path)
    for record in records:
        source = args.raw_root / str(record["name"])
        if not source.is_file():
            raise FileNotFoundError(source)
        record["bytes"] = source.stat().st_size
    total_bytes = sum(int(record["bytes"]) for record in records)

    connector = load_connector(args.connector)
    password = connector.credential()
    client = connector.connect(password)
    try:
        environment = connector.run_remote(
            client,
            "set -eu; "
            "test -x " + shlex.quote(args.colmap) + "; "
            "printf 'memory_max='; cat /sys/fs/cgroup/memory.max; "
            "printf 'disk_available='; df -B1 --output=avail /root/autodl-tmp | tail -1; "
            "printf 'colmap_version='; " + shlex.quote(args.colmap) + " -h | head -1",
            timeout=120,
        )
        if "memory_max=2147483648" not in environment:
            raise RuntimeError(f"Unexpected remote memory limit: {environment!r}")
        connector.run_remote(
            client,
            "set -eu; "
            f"mkdir -p {shlex.quote(remote_root + '/input/images')} "
            f"{shlex.quote(remote_root + '/input/sparse/0')}; "
            f"test ! -f {shlex.quote(remote_root + '/state.txt')} || "
            f"test \"$(cat {shlex.quote(remote_root + '/state.txt')})\" != RUNNING",
            timeout=120,
        )
        state = connector.run_remote(
            client,
            f"test -f {shlex.quote(remote_root + '/state.txt')} && "
            f"cat {shlex.quote(remote_root + '/state.txt')} || true",
            timeout=30,
        ).strip()
        if state == "SUCCESS":
            print(json.dumps({"status": "already_success", "scene": args.scene}))
            return
        if state:
            raise RuntimeError(f"Remote scene has non-resumable state {state!r}")

        sftp = client.open_sftp()
        try:
            upload_small(
                sftp, manifest_path, remote_root + "/input/raw_images.sha256"
            )
            for name in ("cameras.bin", "images.bin", "points3D.bin"):
                upload_small(
                    sftp,
                    sparse_root / name,
                    remote_root + "/input/sparse/0/" + name,
                )
        finally:
            sftp.close()

        existing = remote_file_sizes(
            connector, client, remote_root + "/input/images"
        )
        pending = []
        done_files = 0
        done_bytes = 0
        for record in records:
            name = str(record["name"])
            size = int(record["bytes"])
            remote_size = existing.get(name)
            if remote_size is None:
                partial_size = existing.get(f".{name}.part", 0)
                if partial_size > size:
                    raise IOError(
                        f"Existing remote partial is oversized: {name} "
                        f"({partial_size} > {size})"
                    )
                done_bytes += partial_size
                pending.append(record)
            elif remote_size == size:
                done_files += 1
                done_bytes += size
            else:
                raise IOError(
                    f"Existing remote file has wrong size: {name} "
                    f"({remote_size} != {size})"
                )
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
                        upload_bucket,
                        connector,
                        password,
                        args.raw_root,
                        remote_root + "/input/images",
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
        print(json.dumps(progress.snapshot(args.scene)), flush=True)
        if progress.done_files != len(records) or progress.done_bytes != total_bytes:
            raise RuntimeError("Upload counters did not reach the expected totals")

        print(
            json.dumps({"status": "verifying_remote_sha256", "scene": args.scene}),
            flush=True,
        )
        check_output = connector.run_remote(
            client,
            f"set -eu; cd {shlex.quote(remote_root + '/input/images')}; "
            "sha256sum -c ../raw_images.sha256;",
            timeout=None,
        )
        if check_output.count(": OK\n") != len(records):
            raise RuntimeError("Remote SHA256 verification count mismatch")
        wrapper = make_wrapper(remote_root, args.colmap, len(records))
        local_wrapper = args.prepared_root / "run_remote_undistorter.sh"
        local_wrapper.write_text(wrapper, encoding="utf-8", newline="\n")
        sftp = client.open_sftp()
        try:
            upload_small(sftp, local_wrapper, remote_root + "/run_undistorter.sh")
            sftp.chmod(remote_root + "/run_undistorter.sh", 0o755)
        finally:
            sftp.close()
        launcher_pid = connector.run_remote(
            client,
            f"nohup bash {shlex.quote(remote_root + '/run_undistorter.sh')} "
            f"> {shlex.quote(remote_root + '/launcher.log')} 2>&1 </dev/null "
            "& echo $!",
            timeout=30,
        ).strip()
        if not launcher_pid.isdigit():
            raise RuntimeError(f"Unexpected launcher PID: {launcher_pid!r}")
        connector.run_remote(
            client,
            f"printf '%s\\n' {shlex.quote(launcher_pid)} > "
            f"{shlex.quote(remote_root + '/launcher.pid')}",
            timeout=30,
        )
        time.sleep(3)
        status = connector.run_remote(
            client,
            f"set -eu; cat {shlex.quote(remote_root + '/state.txt')}; "
            f"ps -o pid=,stat=,etime=,cmd= -p {shlex.quote(launcher_pid)} || true",
            timeout=30,
        )
        report = {
            "schema": "gs-gcp-colmap-native-quarter-remote-launch-v1",
            "scene": args.scene,
            "status": "launched",
            "remote_root": remote_root,
            "launcher_pid": int(launcher_pid),
            "image_count": len(records),
            "raw_bytes": total_bytes,
            "workers": args.workers,
            "environment": environment,
            "initial_status": status,
            "command_contract": {
                "colmap": args.colmap,
                "CUDA_VISIBLE_DEVICES": "",
                "max_image_size": 1414,
                "num_threads": 1,
                "output_type": "COLMAP",
            },
        }
        report_path = args.prepared_root.parent / "REMOTE_LAUNCH.json"
        with report_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(json.dumps(report, ensure_ascii=False), flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    main()
