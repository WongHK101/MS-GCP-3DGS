#!/usr/bin/env python3
"""Supervise the approved remaining native-quarter scenes in fixed order."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


SCENES = (
    "gcp_100000_20260610",
    "gcp_50000_20260610",
    "gcp_20000_20260602",
    "gcp_10000_20260610",
    "gcp_5000_20260602",
)
ALL_SCENES = ("gcp_3000_20260602",) + SCENES
POLL_SECONDS = 60


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


class Supervisor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.script_root = Path(__file__).resolve().parent
        self.status_path = args.candidate_root / "BATCH_STATUS.json"
        self.audit_path = args.candidate_root / "BATCH_AUDIT.json"
        self.log_root = args.candidate_root / "_batch_logs"
        self.log_root.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_root / "supervisor.log"
        self.started_utc = utc_now()
        self.completed: list[str] = []
        self.current_scene: str | None = None
        self.current_stage = "starting"
        self.last_remote_status: dict[str, object] | None = None

    def log(self, message: str) -> None:
        line = f"{utc_now()} {message}"
        with self.log_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")
        print(line, flush=True)

    def update_status(
        self, status: str = "running", error: str | None = None
    ) -> None:
        value = {
            "schema": "gs-gcp-colmap-native-quarter-batch-status-v1",
            "updated_utc": utc_now(),
            "started_utc": self.started_utc,
            "status": status,
            "scene_order": list(SCENES),
            "completed_scenes": list(self.completed),
            "current_scene": self.current_scene,
            "current_stage": self.current_stage,
            "last_remote_status": self.last_remote_status,
            "error": error,
            "gpu_used": False,
            "training_started": False,
            "configured_num_threads": self.args.num_threads,
        }
        write_json(self.status_path, value)

    def command(self, script: str, *arguments: str) -> list[str]:
        return [
            sys.executable,
            "-u",
            str(self.script_root / script),
            *arguments,
        ]

    def run_stage(
        self,
        stage: str,
        command: list[str],
        retries: int = 0,
        retry_seconds: int = 60,
    ) -> None:
        self.current_stage = stage
        self.update_status()
        stage_log = self.log_root / f"{self.current_scene}.{stage}.log"
        for attempt in range(1, retries + 2):
            self.log(
                f"stage={stage} attempt={attempt}/{retries + 1} command="
                + json.dumps(command, ensure_ascii=False)
            )
            with stage_log.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(
                    f"\n[{utc_now()}] attempt {attempt}/{retries + 1}\n"
                )
                process = subprocess.Popen(
                    command,
                    cwd=str(self.args.repo_root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    handle.write(line)
                    handle.flush()
                    self.log(f"stage={stage} {line.rstrip()}")
                returncode = process.wait()
            if returncode == 0:
                self.log(f"stage={stage} result=pass")
                return
            self.log(f"stage={stage} result=fail returncode={returncode}")
            if attempt <= retries:
                time.sleep(retry_seconds)
        raise RuntimeError(
            f"Stage {stage} failed after {retries + 1} attempt(s); "
            f"see {stage_log}"
        )

    def query_remote_once(self, scene: str) -> dict[str, object]:
        completed = subprocess.run(
            self.command(
                "query_colmap_native_quarter_scene.py",
                "--scene",
                scene,
                "--connector",
                str(self.args.connector),
                "--remote-batch-root",
                self.args.remote_batch_root,
            ),
            cwd=str(self.args.repo_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Remote query failed ({completed.returncode}): "
                f"{completed.stderr[-1000:]}"
            )
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        if not lines:
            raise ValueError("Remote query returned no JSON")
        return json.loads(lines[-1])

    def query_remote(self, scene: str) -> dict[str, object]:
        failures = 0
        while True:
            try:
                return self.query_remote_once(scene)
            except Exception as exc:
                failures += 1
                self.log(
                    f"stage=remote_query transient_failure={failures} "
                    f"error={type(exc).__name__}: {exc}"
                )
                self.current_stage = "waiting_for_remote_connection"
                self.update_status(error=f"{type(exc).__name__}: {exc}")
                time.sleep(POLL_SECONDS)

    def wait_remote(self, scene: str) -> None:
        self.current_stage = "remote_undistortion"
        while True:
            status = self.query_remote(scene)
            self.last_remote_status = status
            state = status.get("state")
            self.update_status()
            self.log(
                "remote_status="
                + json.dumps(
                    {
                        "scene": scene,
                        "state": state,
                        "output_images": status.get("output_images"),
                        "last_progress": status.get("last_progress"),
                        "memory_current": status.get("memory_current"),
                        "oom_kill": status.get("oom_kill"),
                    },
                    ensure_ascii=False,
                )
            )
            if state == "SUCCESS":
                return
            if state == "FAILED":
                raise RuntimeError(f"Remote COLMAP failed for {scene}: {status}")
            if state != "RUNNING":
                raise RuntimeError(
                    f"Unexpected remote state while waiting for {scene}: {state!r}"
                )
            if int(status.get("oom_kill", 0)) != 0:
                raise RuntimeError(f"Remote cgroup reports OOM kill: {status}")
            time.sleep(POLL_SECONDS)

    def ensure_remote_result(self, scene: str, candidate: Path) -> None:
        status = self.query_remote(scene)
        self.last_remote_status = status
        state = status.get("state")
        if state == "ABSENT":
            preparation = json.loads(
                (candidate / "evidence" / "PREPARATION.json").read_text(
                    encoding="utf-8"
                )
            )
            estimated_hours = float(
                preparation["runtime_gate"]["estimated_hours"]
            )
            if estimated_hours >= 10.0:
                raise RuntimeError(
                    f"Refusing {scene}: estimate {estimated_hours:.3f} h >= 10 h"
                )
            upload_command = self.command(
                "upload_launch_colmap_native_quarter_scene.py",
                "--scene",
                scene,
                "--raw-root",
                str(self.args.raw_root / scene),
                "--prepared-root",
                str(candidate / "evidence" / "pose_only_input"),
                "--connector",
                str(self.args.connector),
                "--remote-batch-root",
                self.args.remote_batch_root,
                "--colmap",
                self.args.remote_colmap,
                "--workers",
                str(self.args.transfer_workers),
                "--num-threads",
                str(self.args.num_threads),
            )
            for upload_attempt in range(1, 6):
                try:
                    self.run_stage("upload_launch", upload_command)
                    break
                except Exception:
                    status = self.query_remote(scene)
                    self.last_remote_status = status
                    state = status.get("state")
                    if state in {"RUNNING", "SUCCESS"}:
                        self.log(
                            "upload_launch returned nonzero after remote "
                            f"state became {state}; continuing remote supervision"
                        )
                        break
                    if state != "ABSENT" or upload_attempt == 5:
                        raise
                    self.log(
                        f"upload_launch resumable failure {upload_attempt}/5; "
                        "remote state remains ABSENT"
                    )
                    time.sleep(POLL_SECONDS)
            status = self.query_remote(scene)
            self.last_remote_status = status
            state = status.get("state")
        if state == "RUNNING":
            self.wait_remote(scene)
            return
        if state == "SUCCESS":
            return
        if state == "FAILED":
            raise RuntimeError(f"Remote COLMAP already failed for {scene}: {status}")
        raise RuntimeError(f"Unexpected remote state for {scene}: {state!r}")

    def finalize_scene(self, scene: str, candidate: Path) -> None:
        download_report = candidate / "evidence" / "DOWNLOAD.json"
        if not download_report.is_file():
            self.run_stage(
                "download",
                self.command(
                    "download_colmap_native_quarter_scene.py",
                    "--scene",
                    scene,
                    "--candidate-root",
                    str(candidate),
                    "--connector",
                    str(self.args.connector),
                    "--remote-batch-root",
                    self.args.remote_batch_root,
                    "--workers",
                    str(self.args.transfer_workers),
                ),
                retries=4,
            )
        package_audit = candidate / "evidence" / "PACKAGE_AUDIT.json"
        if not package_audit.is_file():
            self.run_stage(
                "materialize_audit",
                self.command(
                    "materialize_audit_colmap_native_quarter_scene.py",
                    "--scene",
                    scene,
                    "--candidate-root",
                    str(candidate),
                    "--frozen-model",
                    str(self.args.frozen_root / scene / "sparse" / "0"),
                    "--read-write-model",
                    str(self.args.read_write_model),
                    "--local-colmap",
                    str(self.args.local_colmap),
                    "--image-workers",
                    "4",
                ),
            )
        audit = json.loads(package_audit.read_text(encoding="utf-8"))
        if audit.get("status") != "pass":
            raise ValueError(f"Package audit is not pass for {scene}: {audit}")

    def build_batch_audit(self) -> None:
        scenes = []
        num_threads_by_scene = {}
        for scene in ALL_SCENES:
            package_path = (
                self.args.candidate_root
                / scene
                / "evidence"
                / "PACKAGE_AUDIT.json"
            )
            package = json.loads(package_path.read_text(encoding="utf-8"))
            if package.get("status") != "pass":
                raise ValueError(f"Non-pass package audit for {scene}")
            image_generation = package.get("image_generation")
            if image_generation is None and scene == "gcp_3000_20260602":
                # The approved pilot predates PACKAGE_AUDIT v2. Its frozen
                # launch evidence records the original single-thread command.
                num_threads_by_scene[scene] = 1
            else:
                num_threads_by_scene[scene] = int(
                    image_generation["num_threads"]
                )
            scenes.append(
                {
                    "scene": scene,
                    "candidate_root": str(
                        (self.args.candidate_root / scene).resolve()
                    ),
                    "package_audit": {
                        "path": str(package_path.resolve()),
                        "bytes": package_path.stat().st_size,
                        "sha256": sha256_file(package_path),
                    },
                    "camera": package.get("camera"),
                    "counts": package.get("counts"),
                    "reprojection": package.get("reprojection"),
                    "gpu_used": package.get("gpu_used"),
                    "training_started": package.get("training_started"),
                }
            )
        audit = {
            "schema": "gs-gcp-colmap-native-quarter-six-scene-audit-v1",
            "created_utc": utc_now(),
            "status": "pass",
            "scene_order": list(ALL_SCENES),
            "generation_order_after_pilot": list(SCENES),
            "generator": "COLMAP 4.0.4 image_undistorter",
            "command_contract": {
                "CUDA_VISIBLE_DEVICES": "",
                "max_image_size": 1414,
                "num_threads_by_scene": num_threads_by_scene,
                "output_type": "COLMAP",
            },
            "gpu_used": False,
            "training_started": False,
            "scene_count": len(scenes),
            "scenes": scenes,
        }
        write_json(self.audit_path, audit)

    def run(self) -> None:
        self.update_status()
        for scene in SCENES:
            self.current_scene = scene
            candidate = self.args.candidate_root / scene
            preparation_path = candidate / "evidence" / "PREPARATION.json"
            if not preparation_path.is_file():
                raise FileNotFoundError(
                    f"Prepared input is missing for {scene}: {preparation_path}"
                )
            package_path = candidate / "evidence" / "PACKAGE_AUDIT.json"
            if package_path.is_file():
                package = json.loads(package_path.read_text(encoding="utf-8"))
                if package.get("status") != "pass":
                    raise ValueError(f"Existing package audit is not pass: {scene}")
                self.log(f"scene={scene} result=already_complete")
            else:
                self.ensure_remote_result(scene, candidate)
                self.finalize_scene(scene, candidate)
            self.completed.append(scene)
            self.current_stage = "scene_complete"
            self.update_status()
            self.log(f"scene={scene} result=pass")
        self.current_scene = None
        self.current_stage = "building_batch_audit"
        self.update_status()
        self.build_batch_audit()
        self.current_stage = "complete"
        self.update_status(status="complete")
        self.log(f"batch=complete audit={self.audit_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--frozen-root", required=True, type=Path)
    parser.add_argument("--read-write-model", required=True, type=Path)
    parser.add_argument("--local-colmap", required=True, type=Path)
    parser.add_argument("--connector", required=True, type=Path)
    parser.add_argument("--remote-batch-root", required=True)
    parser.add_argument("--remote-colmap", required=True)
    parser.add_argument("--transfer-workers", type=int, default=8)
    parser.add_argument("--num-threads", type=int, default=1)
    args = parser.parse_args()

    for path in (
        args.repo_root,
        args.candidate_root,
        args.raw_root,
        args.frozen_root,
        args.read_write_model,
        args.local_colmap,
        args.connector,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.num_threads < 1 or args.num_threads > 64:
        raise ValueError("--num-threads must be between 1 and 64")
    supervisor = Supervisor(args)
    try:
        supervisor.run()
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        supervisor.log(f"batch=failed error={message}")
        supervisor.log(traceback.format_exc())
        supervisor.update_status(status="failed", error=message)
        raise


if __name__ == "__main__":
    main()
