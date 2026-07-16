# MS-GCP v1.3 Method Workspace Isolation Policy

Status: mandatory preflight contract for every training, export, or evaluation run.

## Non-negotiable boundaries

- Raw images, COLMAP sources, v1.3 release files, RTK files, and annotations are read-only method inputs.
- Algorithm code uses one clean fixed-commit worktree per method. Runtime build products must not enter that worktree.
- Each method uses its own locked environment. No global `pip`, user-site package, or another project's environment is allowed.
- Each method/run uses a separate CUDA extension build cache through `TORCH_EXTENSIONS_DIR`.
- Models, packets, evaluator outputs, logs, diagnostics, and temporary files stay under one unique non-overwriting run root.
- Dataset hardlinks are forbidden. Read-only symlinks are allowed only when their target identity is recorded and the method cannot write through them.
- A source hash/inventory is recorded before and after each formal run. Any source mutation invalidates the run.

## Canonical server layout

```text
/root/autodl-tmp/datasets/ms-gcp-v13/<release-digest>/       # read-only mirror
/root/autodl-tmp/worktrees/ms-gcp-v13/<method>/<commit>/     # clean code
/root/autodl-tmp/envs/ms-gcp-v13/<method>/<env-hash>/        # isolated env
/root/autodl-tmp/build/ms-gcp-v13/<method>/<commit>/<run>/   # build/CUDA cache
/root/autodl-tmp/runs/ms-gcp-v13/<method>/<scene>/<run>/     # all run outputs
```

Run-root layout:

```text
00_preflight/
01_training/
02_checkpoints/
03_packets/
04_evaluation/
05_diagnostics/
06_audit/
tmp/
```

The run root must not exist before launch. Retries get a new run ID and may not overwrite or merge a previous run.

## Environment isolation

Every launch manifest must set and record:

```text
PYTHONNOUSERSITE=1
TORCH_EXTENSIONS_DIR=<method/run-specific build root>/torch_extensions
TMPDIR=<method/scene/run root>/tmp
```

Conda/pip lock data, Python/CUDA/PyTorch/compiler versions, repository commits, submodule trees, and source hashes are immutable run provenance. Dependency installation ends before the formal run manifest is signed.

## Enforcement

Before training, packet export, or evaluation, run:

```bash
python code/gcp/validate_gcp_v13_workspace_isolation.py \
  --manifest <method_run_layout.json> \
  --require_nonexistent_run_root \
  --report <preflight_report.json>
```

The preflight fails on dataset/release/code pollution, shared caches, unsafe or cross-method paths, output paths outside the run root, overwrite policy, global Python installation, missing hashes, or malformed IDs.

Release generation is a separate transactional publication operation. It may create a new staging/final release directory, but algorithm runners never write into release or raw-data roots.
