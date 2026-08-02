# GS-GCP Repository Code-Boundary Audit

Date: 2026-08-02

## Repository identity

- Public repository: `https://github.com/WongHK101/GS-GCP-Benchmark`
- Local repository role: independent GS-GCP benchmark orchestration and evaluator
- Audited commit before this report: `46beb3ba474d54d6a3f40097e49d371b86ea5633`
- Tracked files audited: 177
- Local and remote refs audited: 54
- Git submodules: none

## Result

No vendored Gaussian-method training implementation was found in the tracked tree,
the reachable Git history, or any audited local/remote ref. In particular, the
audit found no repository-owned copies of:

- Original 3DGS `train.py`, `gaussian_renderer/`, or `scene/gaussian_model.py`
- 2DGS, PGSR, RaDe-GS, Gaussian Opacity Fields, CityGaussianV2, CityGS-X,
  MetroGS, GFSGS/GFGS, or QGS training implementations
- `diff-gaussian-rasterization` or Simple-KNN source trees
- UMGS, SpectralIndexGS, or multispectral training code

The repository therefore remains a benchmark/evaluator repository rather than a
method implementation repository.

## Intentional external-method interfaces

The following files refer to Gaussian method code without vendoring it:

- `code/gcp/export_gaussian_depth_maps.py` imports a separately supplied,
  fixed-commit method runtime to export evaluation-only metric-depth packets.
- `code/gcp/gs_gcp_common_measurement.py` measures a separately supplied method
  runtime and frozen checkpoint.
- `scripts/gcp_v13/*.sh` invoke `train.py` from isolated external method
  worktrees/environments.
- `code/gcp/test_metric_depth_packet_cuda.py` is an integration test against an
  externally installed rasterizer.
- `configs/gs_gcp_method_registry_v1.json` records upstream method identities,
  commits, licenses, and admission status.

These interfaces do not implement or alter Gaussian training algorithms.

## Compatibility patches retained

Two small compatibility patches are retained as benchmark provenance/build
assets, not as algorithm source:

- `patches/3dgs_original/simple_knn_86710c2_cuda12_cfloat.patch`
- `patches/ceres_respect_explicit_cuda_architectures.patch`

They are applied only to isolated build copies under frozen recipes. They do not
vendor a method repository or change formal benchmark geometry semantics.

## Reproducible audit scope

The audit checked tracked paths, every visible local/remote ref, reachable object
paths, `.gitmodules`, and references to common Gaussian training entry points.
The excluded-path patterns covered `train.py`, `gaussian_renderer/`,
`scene/gaussian_model.py`, rasterizer/Simple-KNN source trees, UMGS,
SpectralIndexGS, and multispectral method code.

Result counts:

- Current tracked algorithm-source path hits: 0
- Ref-level algorithm-source path hits: 0
- Reachable-history algorithm-source path hits: 0
- Submodule declarations: 0

Any future addition of method training source, algorithm submodules, model
weights, datasets, or run outputs must be rejected by repository review. Method
training remains in isolated official upstream worktrees at frozen commits.
