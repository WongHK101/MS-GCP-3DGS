# MS-GCP-3DGS

Research code and protocols for large-area multispectral Gaussian
reconstruction with RTK/GCP-based georeferenced evaluation.

## Directory Layout

- `code/`: project-specific source code and launchers.
- `outputs/`: experiment results, diagnostics, reports, and lightweight logs.
- `evidence/`: small canonical coordinate tables and provenance snapshots.
- `transfer_cache/`: large upload archives or temporary RGB copies; not paper results.
- `configs/`: experiment and evaluation configurations.
- `docs/`: design notes, protocols, and manuscript planning.
- `manuscript/`: future second-paper manuscript source.
- `scripts/`: reproducible environment and experiment entry points.

## Data Boundary

Raw UAV imagery, exact GCP coordinates, acquisition reports, checkpoints,
and experiment caches are intentionally excluded from Git. Dataset release
artifacts will be added only after privacy, licensing, and anonymization
review.

## Current Feasibility Work

The isolated COLMAP 4.0.4 CUDA/cuDSS feasibility test completed on a
976-image scene. It preserved registration and reconstruction quality while
reducing mapper time by about 15%. The result does not automatically replace
the established protocol because the comparison changes both COLMAP version
and solver backend. See
[`docs/GPU_BA_FEASIBILITY_RESULT_20260615.md`](docs/GPU_BA_FEASIBILITY_RESULT_20260615.md).

## Project Boundary

This repository is independent from the first-paper UMGS/TGRS codebase.
Reusable components are promoted through explicit, reviewed copies rather
than silent edits across both projects.
