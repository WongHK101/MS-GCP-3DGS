#!/usr/bin/env python3
"""Build the GS-GCP Stage 0 protocol/instrumentation review package."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")


def write_json(path: Path, value: Any) -> None:
    write_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, errors="replace")


def unique(path: Path) -> Path:
    if not path.exists():
        return path
    stamp = datetime.now().strftime("%H%M%S")
    return path.with_name(f"{path.stem}_{stamp}_01{path.suffix}")


def run_test(command: list[str], destination: Path) -> dict[str, Any]:
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    write_text(destination, result.stdout)
    write_text(destination.with_suffix(destination.suffix + ".stderr.txt"), result.stderr)
    record = {
        "command": subprocess.list2cmdline(command),
        "returncode": result.returncode,
        "stdout": destination.name,
        "stderr": destination.with_suffix(destination.suffix + ".stderr.txt").name,
    }
    if result.returncode != 0:
        raise RuntimeError(f"Stage 0 package test failed: {record}")
    return record


def package_entries(root: Path) -> list[dict[str, Any]]:
    entries = []
    for path in sorted(
        (candidate for candidate in root.rglob("*") if candidate.is_file()),
        key=lambda item: item.relative_to(root).as_posix().encode("utf-8"),
    ):
        relative = path.relative_to(root).as_posix()
        if relative == "PACKAGE_MANIFEST.json":
            continue
        entries.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return entries


def build(args: argparse.Namespace) -> tuple[Path, Path]:
    status = git("status", "--porcelain")
    if status:
        raise RuntimeError(f"Stage 0 package requires clean worktree: {status!r}")
    head = git("rev-parse", "HEAD").strip()
    branch = git("branch", "--show-current").strip()
    release_root = args.release_root.resolve()
    evidence_dir = args.evidence_dir.resolve()
    if not release_root.is_dir() or not evidence_dir.is_dir():
        raise FileNotFoundError("release root and evidence directory are required")

    review_root = unique(args.review_output_parent.resolve() / "gs_gcp_stage0_review_20260717")
    review_root.mkdir(parents=True)
    tests = review_root / "tests"
    tests.mkdir()
    commands = [
        ([sys.executable, "-m", "py_compile", "code/gcp/validate_gs_gcp_method_registry.py", "code/gcp/run_with_resource_probe.py", "code/gcp/validate_gs_gcp_stage0.py", "code/gcp/build_gs_gcp_stage0_review_package.py"], "py_compile.txt"),
        ([sys.executable, "code/gcp/test_validate_gs_gcp_method_registry.py"], "method_registry_tests.txt"),
        ([sys.executable, "code/gcp/test_run_with_resource_probe.py"], "resource_probe_tests.txt"),
        ([sys.executable, "code/gcp/test_validate_gs_gcp_stage0.py"], "stage0_readiness_tests.txt"),
        ([sys.executable, "code/gcp/test_gs_gcp_resolution.py"], "resolution_tests.txt"),
        ([sys.executable, "code/gcp/test_validate_gcp_v13_workspace_isolation.py"], "workspace_isolation_tests.txt"),
        ([sys.executable, "code/gcp/test_validate_gs_gcp_v13_original_3dgs_recipe.py"], "original_3dgs_recipe_tests.txt"),
        ([sys.executable, "code/gcp/test_gcp_release_v1_3.py", "--real_release_dir", str(release_root)], "v1_3_real_release_tests.txt"),
        ([sys.executable, "code/gcp/validate_gs_gcp_stage0.py", "--repo_root", str(ROOT), "--release_root", str(release_root), "--method_id", "3dgs_original"], "stage0_readiness_report.json"),
    ]
    command_records = [run_test(command, tests / name) for command, name in commands]

    snapshots = [
        "README.md",
        "configs/gs_gcp_method_registry_v1.json",
        "configs/gs_gcp_resource_probe_contract_v1.json",
        "configs/gs_gcp_training_resolution_v1.json",
        "configs/gs_gcp_v13_original_3dgs_recipe_v2.json",
        "configs/gcp_v13_workspace_isolation_v1.json",
        "configs/gs_gcp_v13_release_review_status_v1.json",
        "configs/gs_gcp_v13_data_mirror_v1.json",
        "configs/gs_gcp_autodl740_runtime_status_v1.json",
        "configs/gs_gcp_repository_promotion_status_v1.json",
        "code/gcp/validate_gs_gcp_method_registry.py",
        "code/gcp/test_validate_gs_gcp_method_registry.py",
        "code/gcp/run_with_resource_probe.py",
        "code/gcp/test_run_with_resource_probe.py",
        "code/gcp/validate_gs_gcp_stage0.py",
        "code/gcp/test_validate_gs_gcp_stage0.py",
        "scripts/gcp_v13/run_original_3dgs_3k_30k.sh",
        "docs/GS_GCP_STAGE0_FREEZE.md",
        "docs/GCP_V1_3_GEOMETRY_METHOD_CANDIDATES.md",
        "docs/GCP_V1_3_GEOMETRY_METHOD_PUBLICATION_AUDIT.md",
    ]
    for relative in snapshots:
        source = ROOT / relative
        destination = review_root / "source_snapshots" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    shutil.copytree(evidence_dir, review_root / "server_evidence")

    write_text(review_root / "git_commit.txt", head + "\n")
    write_text(review_root / "git_branch.txt", branch + "\n")
    write_text(review_root / "git_status_porcelain.txt", status)
    write_text(review_root / "git_show.patch", git("show", "--stat", "--patch", "--format=fuller", "HEAD"))
    write_json(review_root / "exact_commands.json", command_records)
    write_text(
        review_root / "release_root_record.sha256",
        f"{sha256_file(release_root / 'v1_3_0_release_root_digest.json')}  v1_3_0_release_root_digest.json\n",
    )
    write_json(
        review_root / "stage0_summary.json",
        {
            "schema": "gs_gcp_stage0_review_summary_v1",
            "status": "contracts_and_instrumentation_pass_training_not_authorized",
            "commit": head,
            "branch": branch,
            "release_root_digest": "513f8999fe4b110f15bcbecad7932895781cee755ee9ccd7a14ff10298546d75",
            "method_count": 10,
            "formal_core_method_count": 6,
            "scalability_extension_count": 2,
            "conditional_method_count": 2,
            "data_mirror_file_count": 6267,
            "data_mirror_bytes": 64661981667,
            "gpu_used_for_training": False,
            "training_run": False,
            "formal_metrics_computed": False,
            "remaining_blockers": [
                "v1.3.0 external GPT PASS not recorded",
                "GitHub repository rename to GS-GCP-Benchmark pending",
                "AutoDL-740 selected GPU must pass the frozen idle gate; all devices were externally busy at the Stage 0 capture",
            ],
        },
    )
    write_text(
        review_root / "REVIEW_BRIEF.md",
        f"""# GS-GCP Stage 0 Protocol And Instrumentation Review

Status: contracts and instrumentation PASS candidate; training remains blocked.

- Commit: `{head}`
- Branch: `{branch}`
- Release root: `513f8999fe4b110f15bcbecad7932895781cee755ee9ccd7a14ff10298546d75`
- AutoDL-740 mirror: 6,267 files / 64,661,981,667 bytes, read-only, full-hash verified
- Method registry: 6 formal core, 2 scalability extensions, 2 conditional
- Training resolution: original 3DGS `-r -1`, 1600-pixel width cap
- Resource probe: external GNU time plus 1 Hz `nvidia-smi`; no loss/autograd changes
- Training executed: no
- Formal metrics computed: no

The launcher now calls the Stage 0 readiness gate before training and runs the
unchanged child training command through the external resource probe. The gate
intentionally remains closed because an external GPT PASS for the v1.3.0
release is not recorded. The local repository is independent and clean, but
the GitHub repository name still uses the retired MS prefix.
""",
    )

    entries = package_entries(review_root)
    write_json(
        review_root / "PACKAGE_MANIFEST.json",
        {
            "schema": "gs_gcp_stage0_review_package_manifest_v1",
            "self_included": False,
            "file_count": len(entries),
            "files": entries,
        },
    )
    args.package_dir.mkdir(parents=True, exist_ok=True)
    package = unique(args.package_dir.resolve() / "GPT_GS_GCP_STAGE0_PROTOCOL_INSTRUMENTATION_REVIEW_20260717.zip")
    with zipfile.ZipFile(package, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(
            (candidate for candidate in review_root.rglob("*") if candidate.is_file()),
            key=lambda item: item.relative_to(review_root).as_posix().encode("utf-8"),
        ):
            archive.write(path, path.relative_to(review_root).as_posix())
    detached = package.with_suffix(package.suffix + ".sha256")
    write_text(detached, f"{sha256_file(package)}  {package.name}\n")
    return package, detached


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release_root", type=Path, required=True)
    parser.add_argument("--evidence_dir", type=Path, required=True)
    parser.add_argument("--review_output_parent", type=Path, required=True)
    parser.add_argument("--package_dir", type=Path, required=True)
    args = parser.parse_args()
    package, detached = build(args)
    print(package)
    print(detached)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
