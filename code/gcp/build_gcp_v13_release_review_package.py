"""Build the unified GS-GCP v1.3.0 release-freeze review package."""

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


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RELEASE = Path(r"E:\datasets\M3M-GCP\scenes\gcp_manual_annotations_v1_3_0")
DEFAULT_PROJECT = Path(__file__).resolve().parents[2]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def write_json(path: Path, payload: Any) -> None:
    write_text(path, json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True, errors="replace")


def run_command(command: list[str], output_path: Path) -> dict[str, Any]:
    result = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True)
    write_text(output_path, result.stdout)
    write_text(output_path.with_suffix(output_path.suffix + ".stderr.txt"), result.stderr)
    record = {
        "command": subprocess.list2cmdline(command),
        "returncode": result.returncode,
        "stdout": output_path.name,
        "stderr": output_path.with_suffix(output_path.suffix + ".stderr.txt").name,
    }
    if result.returncode != 0:
        raise RuntimeError(f"review-package command failed: {record}")
    return record


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stamp = datetime.now().strftime("%H%M%S")
    return path.with_name(f"{path.stem}_{stamp}_01{path.suffix}")


def package_manifest(root: Path) -> list[dict[str, Any]]:
    excluded = {"PACKAGE_MANIFEST.json"}
    records = []
    for path in sorted((candidate for candidate in root.rglob("*") if candidate.is_file()), key=lambda p: p.relative_to(root).as_posix().encode("utf-8")):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        records.append({"path": relative, "bytes": path.stat().st_size, "sha256": file_sha256(path)})
    return records


def build(args: argparse.Namespace) -> tuple[Path, Path]:
    release = Path(args.release_dir)
    if not release.is_dir():
        raise FileNotFoundError(release)
    status = git_output("status", "--porcelain")
    if status:
        raise ValueError(f"review package requires a clean worktree: {status!r}")
    head = git_output("rev-parse", "HEAD").strip()
    root_record = json.loads((release / "v1_3_0_release_root_digest.json").read_text(encoding="utf-8"))
    release_config = json.loads((release / "gcp_benchmark_release_v1_3_0.json").read_text(encoding="utf-8"))

    review_parent = Path(args.review_output_parent)
    run_root = unique_path(review_parent / "gcp_release_v1_3_0_freeze_review_20260717")
    run_root.mkdir(parents=True)
    tests_dir = run_root / "tests"
    tests_dir.mkdir()
    exact_commands = []
    commands = [
        ([sys.executable, "-m", "py_compile", "code/gcp/gcp_pixel_domain_v1_2.py", "code/gcp/gcp_pixel_domain_v1_3.py", "code/gcp/generate_gcp_release_v1_3.py", "code/gcp/evaluate_gaussian_gcp_geometry.py"], "py_compile.txt"),
        ([sys.executable, "code/gcp/test_gcp_pixel_domain_v1_3.py"], "test_gcp_pixel_domain_v1_3.json"),
        ([sys.executable, "code/gcp/test_generate_gcp_release_v1_3.py"], "test_generate_gcp_release_v1_3.json"),
        ([sys.executable, "code/gcp/test_gcp_release_v1_3.py", "--real_release_dir", str(release)], "test_gcp_release_v1_3.json"),
        ([sys.executable, "code/gcp/test_gcp_evaluator_protocol.py"], "test_gcp_evaluator_protocol.json"),
        ([sys.executable, "code/gcp/test_gcp_release_v1_2.py", "--real_release_dir", str(Path(args.release_v122))], "test_gcp_release_v1_2_2.json"),
        ([sys.executable, "code/gcp/test_validate_gcp_v13_workspace_isolation.py"], "test_workspace_isolation.json"),
    ]
    for command, name in commands:
        exact_commands.append(run_command(command, tests_dir / name))

    shutil.copytree(release, run_root / "release_snapshot")
    source_root = run_root / "source_snapshots"
    for relative in [
        "code/gcp/gcp_pixel_domain_v1_2.py",
        "code/gcp/gcp_pixel_domain_v1_3.py",
        "code/gcp/generate_gcp_release_v1_3.py",
        "code/gcp/evaluate_gaussian_gcp_geometry.py",
        "code/gcp/test_gcp_pixel_domain_v1_3.py",
        "code/gcp/test_generate_gcp_release_v1_3.py",
        "code/gcp/test_gcp_release_v1_3.py",
        "code/gcp/validate_gcp_v13_workspace_isolation.py",
        "code/gcp/test_validate_gcp_v13_workspace_isolation.py",
        "configs/gcp_v13_release_inputs_v1.json",
        "configs/gcp_v13_workspace_isolation_v1.json",
        "docs/GCP_V1_3_COMPLETE_EXPERIMENT_PLAN.md",
        "docs/GCP_V1_3_METHOD_WORKSPACE_ISOLATION_POLICY.md",
    ]:
        source = REPO_ROOT / relative
        destination = source_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    write_text(run_root / "git_commit.txt", head + "\n")
    write_text(run_root / "git_status_porcelain.txt", status)
    write_text(run_root / "git_log.txt", git_output("log", "--oneline", "-12"))
    write_text(run_root / "git_diff_release_implementation.patch", git_output("diff", "f1ded09..HEAD"))
    write_json(run_root / "exact_commands.json", exact_commands)
    write_text(
        run_root / "release_root_record.sha256",
        f"{file_sha256(release / 'v1_3_0_release_root_digest.json')}  v1_3_0_release_root_digest.json\n",
    )
    write_json(
        run_root / "freeze_summary.json",
        {
            "status": "PASS_AFTER_INDEPENDENT_POST_PUBLISH_VERIFICATION",
            "release_id": release_config["release_id"],
            "release_path": str(release),
            "payload_manifest_sha256": root_record["payload_manifest_sha256"],
            "payload_root_digest_sha256": root_record["payload_root_digest_sha256"],
            "payload_file_count": root_record["payload_file_count"],
            "release_generator_commit": release_config["generator_provenance"]["generator_git_commit"],
            "current_review_commit": head,
            "rows": release_config["frozen_counts"]["row_count"],
            "annotation_good": release_config["frozen_counts"]["annotation_good_count"],
            "formal_eligible": release_config["frozen_counts"]["formal_eligible_count"],
            "mapping_records": release_config["frozen_counts"]["annotated_image_count"],
            "training_views": release_config["frozen_counts"]["training_view_count"],
            "v1_2_2_observation_ids_preserved": release_config["frozen_counts"]["v1_2_2_preserved_observation_count"],
            "gpu_used": False,
            "training_run": False,
            "depth_tensor_values_read": False,
            "formal_metrics_computed": False,
        },
    )
    write_text(
        run_root / "POST_PUBLISH_CLEANUP_INCIDENT.md",
        """# Post-publish cleanup incident

Both staging payloads completed and compared byte-identically before the atomic
rename. The rename then published the complete formal directory. Cleanup of the
already-validated compare staging failed because byte-copied RTK evidence kept
the Windows read-only attribute. No release payload file was changed.

The formal directory was independently revalidated: payload/root integrity,
1,383-row real-release loader smoke, evaluator protocol tests, and v1.2.2
regression tests all passed. The two temporary staging directories were then
removed. Commit 4107b96 changes cleanup order so read-only compare staging is
removed before atomic publication and adds a dedicated regression test.
""",
    )
    write_text(
        run_root / "REVIEW_BRIEF.md",
        f"""# GS-GCP v1.3.0 Release Freeze Review

Status: PASS candidate for external review.

- Release: `{release_config['release_id']}`
- Formal directory: `{release}`
- Release generator commit: `{release_config['generator_provenance']['generator_git_commit']}`
- Review/code HEAD: `{head}`
- Canonical rows: 1,383
- Annotation Good: 1,155
- Formal eligible: 1,069
- Formal scene/point split rows: 87 (48 control, 39 checkpoint)
- Annotated image mappings: 951 unique image-level records
- Frozen training views: 6,187
- Preserved v1.2.2 observation IDs: 611/611
- Payload files: {root_record['payload_file_count']}
- Payload root digest: `{root_record['payload_root_digest_sha256']}`

The release keeps v1.2.2 unchanged as a sparse-control diagnostic track. No GPU,
training, packet export, residual generation, Sim(3), or formal metric run was
performed during this freeze. Raw scene directories were read-only inputs.

Two clicked non-formal rows are explicitly preserved as diagnostic out-of-bounds
records: 3K G11 image 0021 (Ambiguous) and 20K wy3_1 image 0062 (Not visible).
No formal observation is out of bounds.
""",
    )

    records = package_manifest(run_root)
    write_json(
        run_root / "PACKAGE_MANIFEST.json",
        {
            "schema": "ms_gcp_v1_3_0_release_review_package_manifest_v1",
            "self_included": False,
            "file_count": len(records),
            "files": records,
        },
    )
    package_dir = Path(args.package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)
    package_path = unique_path(package_dir / "GPT_GCP_POINTSET_RELEASE_V1_3_0_CONTROL_HEAVY_REVIEW_20260717.zip")
    with zipfile.ZipFile(package_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted((candidate for candidate in run_root.rglob("*") if candidate.is_file()), key=lambda p: p.relative_to(run_root).as_posix().encode("utf-8")):
            archive.write(path, path.relative_to(run_root).as_posix())
    detached = package_path.with_suffix(package_path.suffix + ".sha256")
    write_text(detached, f"{file_sha256(package_path)}  {package_path.name}\n")
    return package_path, detached


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release_dir", default=str(DEFAULT_RELEASE))
    parser.add_argument("--release_v122", default=str(DEFAULT_RELEASE.parent / "gcp_manual_annotations_v1_2_2"))
    parser.add_argument("--review_output_parent", default=str(DEFAULT_PROJECT / "outputs"))
    parser.add_argument("--package_dir", default=str(DEFAULT_PROJECT / "outputs" / "gpt_review_packages"))
    args = parser.parse_args()
    package, detached = build(args)
    print(
        json.dumps(
            {
                "status": "PASS",
                "package": str(package),
                "package_sha256": file_sha256(package),
                "detached_sha256": str(detached),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

