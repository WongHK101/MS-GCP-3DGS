#!/usr/bin/env python3
"""Run the frozen single-block CityGaussianV2 coarse/fine/merge pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


COARSE_CONFIG = "configs/citygsv2_mc_aerial_coarse_sh2.yaml"
FINE_CONFIG = "configs/citygsv2_mc_aerial_sh2_trim.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_record(args: list[str]) -> list[str]:
    return [str(value) for value in args]


def run_checked(
    args: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    capture_stdout: bool = False,
) -> subprocess.CompletedProcess[str]:
    print("RUN", json.dumps(command_record(args), ensure_ascii=False), flush=True)
    completed = subprocess.run(
        args,
        cwd=cwd,
        env=env,
        check=False,
        text=True,
        stdout=subprocess.PIPE if capture_stdout else None,
        stderr=subprocess.PIPE if capture_stdout else None,
    )
    if capture_stdout:
        log_path.write_text(
            (completed.stdout or "") + "\n--- STDERR ---\n" + (completed.stderr or ""),
            encoding="utf-8",
        )
    else:
        log_path.write_text(
            json.dumps(
                {
                    "command": command_record(args),
                    "returncode": completed.returncode,
                    "note": "stdout and stderr are inherited by the outer immutable resource probe",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            args,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    return completed


def find_step_checkpoint(directory: Path, expected_step: int) -> Path:
    candidates = sorted(directory.glob("*.ckpt"))
    matched = []
    for path in candidates:
        match = re.search(r"step=(\d+)", path.name)
        if match and int(match.group(1)) == expected_step:
            matched.append(path)
    if len(matched) != 1:
        raise RuntimeError(
            f"expected exactly one step={expected_step} checkpoint in {directory}, got {matched}"
        )
    return matched[0].resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--mode", choices=("qualification", "formal"), required=True)
    parser.add_argument("--coarse_steps", type=int, required=True)
    parser.add_argument("--fine_steps", type=int, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    # Preserve the virtual-environment launcher instead of resolving its
    # symlink to the system interpreter and losing the frozen packages.
    python = Path(os.path.abspath(os.fspath(args.python)))
    dataset = args.dataset.resolve()
    output_root = args.output_root.resolve()
    if args.coarse_steps <= 0 or args.fine_steps <= 0:
        raise ValueError("training step counts must be positive")
    if args.mode == "formal" and (args.coarse_steps, args.fine_steps) != (30_000, 60_000):
        raise ValueError("formal CityGaussianV2 route is frozen to coarse 30K and fine 60K")
    if output_root.exists():
        raise FileExistsError(f"output root already exists: {output_root}")
    for required in (
        repo / COARSE_CONFIG,
        repo / FINE_CONFIG,
        repo / "main.py",
        repo / "utils" / "partition_citygs.py",
        repo / "utils" / "train_citygs_partitions.py",
        repo / "utils" / "merge_citygs_ckpts.py",
        python,
        dataset / "images",
        dataset / "sparse" / "0" / "cameras.bin",
        dataset / "sparse" / "0" / "images.bin",
        dataset / "sparse" / "0" / "points3D.bin",
        dataset / "estimated_depths",
        dataset / "estimated_depth_scales.json",
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    output_root.mkdir(parents=True)
    runtime = output_root / "runtime"
    runtime.mkdir()
    state_path = output_root / "pipeline_state.txt"
    state_path.write_text("PREPARED\n", encoding="utf-8")

    env = dict(os.environ)
    env["PATH"] = str(python.parent) + os.pathsep + env.get("PATH", "")
    env["WANDB_MODE"] = "offline"
    env["WANDB_SILENT"] = "true"
    env["PYTHONUNBUFFERED"] = "1"
    # The frozen upstream utilities load full Lightning checkpoints produced
    # by this same trusted pipeline. PyTorch 2.6 changed torch.load's default
    # to weights_only=True, which rejects those checkpoints before the fine
    # stage. Restore the pre-2.6 behavior for this isolated subprocess tree.
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

    coarse_name = "coarse"
    fine_name = "fine"
    coarse_save = f"[{args.coarse_steps}]"
    fine_saves = (
        "[30000,60000]" if args.mode == "formal" else f"[{args.fine_steps}]"
    )
    coarse_command = [
        str(python),
        "main.py",
        "fit",
        "--config",
        COARSE_CONFIG,
        "--seed_everything=0",
        f"--data.path={dataset}",
        "--data.parser.init_args.down_sample_factor=1",
        "--logger=tensorboard",
        f"--output={output_root}",
        f"--name={coarse_name}",
        f"--trainer.max_steps={args.coarse_steps}",
        f"--model.gaussian.init_args.optimization.means_lr_scheduler.init_args.max_steps={args.coarse_steps}",
        f"--model.metric.init_args.depth_loss_weight.max_steps={args.coarse_steps}",
        f"--save_iterations={coarse_save}",
    ]
    state_path.write_text("COARSE_RUNNING\n", encoding="utf-8")
    run_checked(
        coarse_command,
        cwd=repo,
        env=env,
        log_path=runtime / "coarse_command.json",
    )
    coarse_checkpoint = find_step_checkpoint(
        output_root / coarse_name / "checkpoints", args.coarse_steps
    )
    state_path.write_text("COARSE_PASS\n", encoding="utf-8")

    fine_print_command = [
        str(python),
        "main.py",
        "fit",
        "--config",
        FINE_CONFIG,
        "--seed_everything=0",
        f"--model.initialize_from={coarse_checkpoint}",
        f"--data.path={dataset}",
        "--data.parser.init_args.down_sample_factor=1",
        "--data.parser.init_args.block_dim=[1,1]",
        "--data.parser.init_args.content_threshold=0.05",
        "--logger=tensorboard",
        f"--output={output_root}",
        f"--name={fine_name}",
        f"--trainer.max_steps={args.fine_steps}",
        f"--model.gaussian.init_args.optimization.means_lr_scheduler.init_args.max_steps={args.fine_steps}",
        f"--model.metric.init_args.depth_loss_weight.max_steps={args.fine_steps}",
        f"--save_iterations={fine_saves}",
        "--print_config",
    ]
    fine_print = run_checked(
        fine_print_command,
        cwd=repo,
        env=env,
        log_path=runtime / "fine_config_generation.log",
        capture_stdout=True,
    )
    fine_config = runtime / f"{fine_name}.yaml"
    fine_config.write_text(fine_print.stdout or "", encoding="utf-8")
    if not fine_config.read_text(encoding="utf-8").startswith("# lightning.pytorch==2.3.3"):
        raise RuntimeError("resolved fine config is missing the frozen Lightning header")

    state_path.write_text("PARTITION_RUNNING\n", encoding="utf-8")
    partition_command = [
        str(python),
        "utils/partition_citygs.py",
        "--config_path",
        str(fine_config),
        "--force",
    ]
    run_checked(
        partition_command,
        cwd=repo,
        env=env,
        log_path=runtime / "partition_command.json",
    )
    state_path.write_text("PARTITION_PASS\n", encoding="utf-8")

    state_path.write_text("FINE_RUNNING\n", encoding="utf-8")
    # block_dim is frozen to [1, 1], so this is the exact single command that
    # upstream train_citygs_partitions.py constructs for block 0. Running it
    # directly makes the worker return code observable; the upstream scheduler
    # can return zero even when its ProcessPool worker fails before training.
    fine_command = [
        str(python),
        "main.py",
        "fit",
        "--config",
        str(fine_config),
        "--data.parser.block_id",
        "0",
        f"-n={fine_name}",
        "--project",
        "m3m-gcp-native-quarter",
        "--logger",
        "wandb",
    ]
    run_checked(
        fine_command,
        cwd=repo,
        env=env,
        log_path=runtime / "fine_command.json",
    )
    block_checkpoint = find_step_checkpoint(
        output_root / fine_name / "blocks" / "block_0" / "checkpoints",
        args.fine_steps,
    )
    state_path.write_text("FINE_PASS\n", encoding="utf-8")

    state_path.write_text("MERGE_RUNNING\n", encoding="utf-8")
    merge_command = [
        str(python),
        "utils/merge_citygs_ckpts.py",
        str(output_root / fine_name),
    ]
    run_checked(
        merge_command,
        cwd=repo,
        env=env,
        log_path=runtime / "merge_command.json",
    )
    merged_candidates = sorted((output_root / fine_name / "checkpoints").glob("*.ckpt"))
    if len(merged_candidates) != 1:
        raise RuntimeError(f"expected one merged checkpoint, got {merged_candidates}")
    merged_checkpoint = merged_candidates[0].resolve()

    summary: dict[str, Any] = {
        "schema": "m3m_gcp_native_quarter_citygaussian_v2_pipeline_run_v1",
        "protocol_id": "m3m_gcp_native_quarter_geometry_v2",
        "method_id": "citygaussian_v2",
        "mode": args.mode,
        "status": "PIPELINE_PASS",
        "seed": 0,
        "down_sample_factor": 1.0,
        "block_dim": [1, 1],
        "content_threshold": 0.05,
        "coarse_steps": args.coarse_steps,
        "fine_steps": args.fine_steps,
        "coarse_checkpoint": {
            "path": str(coarse_checkpoint),
            "bytes": coarse_checkpoint.stat().st_size,
            "sha256": sha256(coarse_checkpoint),
        },
        "block_checkpoint": {
            "path": str(block_checkpoint),
            "bytes": block_checkpoint.stat().st_size,
            "sha256": sha256(block_checkpoint),
        },
        "merged_checkpoint": {
            "path": str(merged_checkpoint),
            "bytes": merged_checkpoint.stat().st_size,
            "sha256": sha256(merged_checkpoint),
        },
        "resolved_fine_config": {
            "path": str(fine_config),
            "sha256": sha256(fine_config),
        },
        "commands": {
            "coarse": command_record(coarse_command),
            "fine_config": command_record(fine_print_command),
            "partition": command_record(partition_command),
            "fine": command_record(fine_command),
            "merge": command_record(merge_command),
        },
    }
    (output_root / "pipeline_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    state_path.write_text("PIPELINE_PASS\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
