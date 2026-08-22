#!/usr/bin/env python3
"""Run the frozen CityGaussianV2 100K coarse/fine 4x4/merge pipeline."""

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
PARTITION_RELATIVE = Path(
    "partition/partitions-dim_4_4_visibility_0.05/partitions.pt"
)


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


def find_optional_step_checkpoint(
    directory: Path, expected_step: int
) -> Path | None:
    """Return one exact final checkpoint, allowing a genuinely missing block."""
    candidates = sorted(directory.glob("*.ckpt"))
    matched = []
    for path in candidates:
        match = re.search(r"step=(\d+)", path.name)
        if match and int(match.group(1)) == expected_step:
            matched.append(path)
    if len(matched) > 1:
        raise RuntimeError(
            f"multiple step={expected_step} checkpoints in {directory}: {matched}"
        )
    return matched[0].resolve() if matched else None


def hardlink_checkpoint(source: Path, destination: Path) -> dict[str, Any]:
    """Reuse a verified checkpoint without duplicating multi-gigabyte payloads."""
    if source.is_symlink():
        raise RuntimeError(f"symlinked resume checkpoint is forbidden: {source}")
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.stat().st_dev != destination.parent.stat().st_dev:
        raise RuntimeError(
            "resume checkpoints must be on the same filesystem; copying is forbidden"
        )
    digest = sha256(source)
    os.link(source, destination)
    if not destination.is_file() or not os.path.samefile(source, destination):
        raise RuntimeError("checkpoint hardlink verification failed")
    return {
        "source": str(source),
        "destination": str(destination),
        "bytes": source.stat().st_size,
        "sha256": digest,
        "reuse_mode": "same_filesystem_hardlink",
    }


def materialize_resume_checkpoints(
    *,
    resume_root: Path,
    output_root: Path,
    coarse_steps: int,
    fine_steps: int,
    block_count: int = 16,
) -> tuple[Path, dict[int, Path], list[dict[str, Any]]]:
    """Hardlink completed stages from an older diagnostic attempt into a fresh one."""
    resume_root = resume_root.resolve()
    output_root = output_root.resolve()
    if not resume_root.is_dir():
        raise FileNotFoundError(resume_root)
    if resume_root == output_root:
        raise ValueError("resume source and fresh output root must differ")

    coarse_source = find_step_checkpoint(
        resume_root / "coarse" / "checkpoints", coarse_steps
    )
    coarse_destination = (
        output_root / "coarse" / "checkpoints" / coarse_source.name
    )
    records = [hardlink_checkpoint(coarse_source, coarse_destination)]
    reused_blocks: dict[int, Path] = {}
    for block_id in range(block_count):
        source = find_optional_step_checkpoint(
            resume_root
            / "fine"
            / "blocks"
            / f"block_{block_id}"
            / "checkpoints",
            fine_steps,
        )
        if source is None:
            continue
        destination = (
            output_root
            / "fine"
            / "blocks"
            / f"block_{block_id}"
            / "checkpoints"
            / source.name
        )
        record = hardlink_checkpoint(source, destination)
        record["block_id"] = block_id
        records.append(record)
        reused_blocks[block_id] = destination
    return coarse_destination, reused_blocks, records


def cleanup_transient_checkpoints(
    output_root: Path, merged_checkpoint: Path, inventory_path: Path
) -> dict[str, Any]:
    """Hash-inventory and remove only coarse/block checkpoints after merge."""
    output_root = output_root.resolve()
    merged_checkpoint = merged_checkpoint.resolve()
    inventory_path = inventory_path.resolve()
    if inventory_path.exists():
        raise FileExistsError(f"transient checkpoint inventory exists: {inventory_path}")
    candidates = sorted(
        {
            *output_root.glob("coarse/checkpoints/*.ckpt"),
            *output_root.glob("fine/blocks/block_*/checkpoints/*.ckpt"),
        },
        key=lambda path: path.as_posix(),
    )
    if len(candidates) < 17:
        raise RuntimeError(
            f"expected at least one coarse plus 16 block checkpoints, got {len(candidates)}"
        )
    rows = []
    for raw_path in candidates:
        if raw_path.is_symlink():
            raise RuntimeError(f"symlinked transient checkpoint is forbidden: {raw_path}")
        path = raw_path.resolve()
        try:
            relative = path.relative_to(output_root)
        except ValueError as exc:
            raise RuntimeError(f"transient checkpoint escaped output root: {path}") from exc
        if path == merged_checkpoint or not path.is_file():
            raise RuntimeError(f"unsafe transient checkpoint candidate: {path}")
        rows.append(
            {
                "relative_path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    payload = {
        "schema": "m3m_gcp_citygaussian_v2_transient_checkpoint_inventory_v1",
        "policy": "remove only hashed coarse/block checkpoints after the merged checkpoint is verified",
        "merged_checkpoint": {
            "path": str(merged_checkpoint),
            "bytes": merged_checkpoint.stat().st_size,
            "sha256": sha256(merged_checkpoint),
        },
        "files": rows,
        "file_count": len(rows),
        "logical_bytes_removed": sum(row["bytes"] for row in rows),
    }
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        inventory_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        inventory_path.unlink(missing_ok=True)
        raise
    for row in rows:
        (output_root / row["relative_path"]).unlink()
    if any((output_root / row["relative_path"]).exists() for row in rows):
        raise RuntimeError("one or more transient checkpoints survived cleanup")
    if not merged_checkpoint.is_file() or sha256(merged_checkpoint) != payload["merged_checkpoint"]["sha256"]:
        raise RuntimeError("merged checkpoint changed during transient cleanup")
    return {
        "inventory_path": str(inventory_path),
        "inventory_sha256": sha256(inventory_path),
        "file_count": payload["file_count"],
        "logical_bytes_removed": payload["logical_bytes_removed"],
        "all_inventoried_files_removed": True,
        "merged_checkpoint_retained": True,
        "files": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--mode", choices=("qualification", "formal"), required=True)
    parser.add_argument("--coarse_steps", type=int, required=True)
    parser.add_argument("--fine_steps", type=int, required=True)
    parser.add_argument(
        "--sequential_blocks",
        action="store_true",
        help="Train missing 4x4 blocks sequentially on the one assigned GPU.",
    )
    parser.add_argument(
        "--resume_from",
        type=Path,
        help=(
            "Reuse completed coarse/block checkpoints from an older attempt via "
            "same-filesystem hardlinks; the output root must still be fresh."
        ),
    )
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
    if args.resume_from is not None and not args.sequential_blocks:
        raise ValueError("--resume_from requires --sequential_blocks")
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
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONHASHSEED"] = "0"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
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
    resume_root = args.resume_from.resolve() if args.resume_from is not None else None
    reused_blocks: dict[int, Path] = {}
    reuse_records: list[dict[str, Any]] = []
    if resume_root is None:
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
        coarse_execution = "trained_in_this_attempt"
    else:
        state_path.write_text("RESUME_MATERIALIZING\n", encoding="utf-8")
        coarse_checkpoint, reused_blocks, reuse_records = (
            materialize_resume_checkpoints(
                resume_root=resume_root,
                output_root=output_root,
                coarse_steps=args.coarse_steps,
                fine_steps=args.fine_steps,
            )
        )
        coarse_execution = "reused_by_hardlink"
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
        "--data.parser.init_args.block_dim=[4,4]",
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
    partition_path = dataset / PARTITION_RELATIVE
    if resume_root is None:
        run_checked(
            partition_command,
            cwd=repo,
            env=env,
            log_path=runtime / "partition_command.json",
        )
        partition_execution = "generated_in_this_attempt"
    else:
        if not partition_path.is_file():
            raise FileNotFoundError(
                f"resume source requires the existing deterministic partition: {partition_path}"
            )
        partition_execution = "reused_from_shared_dataset"
    state_path.write_text("PARTITION_PASS\n", encoding="utf-8")

    state_path.write_text("FINE_RUNNING\n", encoding="utf-8")
    fine_command = [
        str(python),
        "utils/train_citygs_partitions.py",
        "--config_name",
        fine_name,
        "--config_dir",
        str(runtime),
        "--project_name",
        "m3m-gcp-native-quarter",
    ]
    sequential_commands: dict[int, list[str]] = {}
    if args.sequential_blocks:
        block_checkpoint_by_id = dict(reused_blocks)
        for block_id in range(16):
            if block_id in block_checkpoint_by_id:
                continue
            state_path.write_text(
                f"FINE_BLOCK_{block_id:02d}_RUNNING\n", encoding="utf-8"
            )
            block_command = [
                str(python),
                "main.py",
                "fit",
                "--config",
                str(fine_config),
                "--data.parser.block_id",
                str(block_id),
                f"-n={fine_name}",
                "--project",
                "m3m-gcp-native-quarter",
                "--logger",
                "wandb",
            ]
            sequential_commands[block_id] = block_command
            run_checked(
                block_command,
                cwd=repo,
                env=env,
                log_path=runtime / f"fine_block_{block_id:02d}_command.json",
            )
            block_checkpoint_by_id[block_id] = find_step_checkpoint(
                output_root
                / fine_name
                / "blocks"
                / f"block_{block_id}"
                / "checkpoints",
                args.fine_steps,
            )
        block_checkpoints = [block_checkpoint_by_id[block_id] for block_id in range(16)]
        fine_execution: Any = {
            "mode": "sequential_missing_blocks",
            "reused_block_ids": sorted(reused_blocks),
            "trained_block_ids": sorted(sequential_commands),
            "commands": {
                str(block_id): command_record(command)
                for block_id, command in sequential_commands.items()
            },
        }
    else:
        run_checked(
            fine_command,
            cwd=repo,
            env=env,
            log_path=runtime / "fine_command.json",
        )
        block_checkpoints = [
            find_step_checkpoint(
                output_root
                / fine_name
                / "blocks"
                / f"block_{block_id}"
                / "checkpoints",
                args.fine_steps,
            )
            for block_id in range(16)
        ]
        fine_execution = {
            "mode": "legacy_upstream_partition_scheduler",
            "command": command_record(fine_command),
        }
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

    cleanup = cleanup_transient_checkpoints(
        output_root,
        merged_checkpoint,
        runtime / "transient_checkpoint_inventory_pre_cleanup.json",
    )
    cleanup_rows = cleanup.pop("files")
    cleanup_by_relative = {
        row["relative_path"]: row for row in cleanup_rows
    }

    def removed_checkpoint_record(path: Path) -> dict[str, Any]:
        relative = path.resolve().relative_to(output_root).as_posix()
        row = cleanup_by_relative.get(relative)
        if row is None:
            raise RuntimeError(f"final-stage checkpoint missing from cleanup inventory: {path}")
        return {
            "path": str(path),
            "bytes": row["bytes"],
            "sha256": row["sha256"],
            "retained_after_merge": False,
        }

    summary: dict[str, Any] = {
        "schema": "m3m_gcp_native_quarter_citygaussian_v2_pipeline_run_v2",
        "protocol_id": "m3m_gcp_native_quarter_geometry_v2",
        "method_id": "citygaussian_v2",
        "scene": "gcp_100000_20260610",
        "mode": args.mode,
        "formal_result": args.mode == "formal",
        "status": "PIPELINE_PASS",
        "seed": 0,
        "down_sample_factor": 1.0,
        "block_dim": [4, 4],
        "content_threshold": 0.05,
        "coarse_steps": args.coarse_steps,
        "fine_steps": args.fine_steps,
        "execution_lifecycle": {
            "sequential_blocks": args.sequential_blocks,
            "resume_source": str(resume_root) if resume_root is not None else None,
            "coarse": coarse_execution,
            "partition": partition_execution,
            "partition_path": str(partition_path),
            "partition_sha256": sha256(partition_path),
            "reused_block_ids": sorted(reused_blocks),
            "trained_block_ids": sorted(sequential_commands),
            "resume_hardlinks": reuse_records,
        },
        "coarse_checkpoint": removed_checkpoint_record(coarse_checkpoint),
        "block_checkpoints": [
            {
                "block_id": block_id,
                **removed_checkpoint_record(path),
            }
            for block_id, path in enumerate(block_checkpoints)
        ],
        "merged_checkpoint": {
            "path": str(merged_checkpoint),
            "bytes": merged_checkpoint.stat().st_size,
            "sha256": sha256(merged_checkpoint),
        },
        "resolved_fine_config": {
            "path": str(fine_config),
            "sha256": sha256(fine_config),
        },
        "transient_checkpoint_cleanup": cleanup,
        "commands": {
            "coarse": (
                command_record(coarse_command) if resume_root is None else None
            ),
            "fine_config": command_record(fine_print_command),
            "partition": (
                command_record(partition_command) if resume_root is None else None
            ),
            "fine": fine_execution,
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
