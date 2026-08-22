# GS-GCP Benchmark

GS-GCP is an RGB UAV benchmark and evaluation toolkit for measuring the
georeferenced geometry accuracy of Gaussian-splatting methods with surveyed
ground control points (GCPs).

The repository contains the benchmark protocol, release validation, camera and
pixel-domain contracts, residual-blind control/checkpoint split tooling,
metric-depth packet interfaces, formal evaluator, independent result verifier,
and isolated experiment launch contracts. Training implementations remain in
their official upstream repositories and are connected through reviewed
adapters; this repository does not vendor a Gaussian training method.

## Reproducible 3K Ten-Method Snapshot

The annotated tag `m3m-gcp-3k-10method-v1` records the benchmark-side code
used for the completed seed-0 3K runs of the ten active methods. The snapshot
contains the frozen source revisions, recipes, compatibility patches, renderer
adapters, evaluation code, and formal evidence. It intentionally excludes
third-party training repositories, virtual environments, datasets, checkpoints,
and rendered outputs.

| Method | Official training repository and frozen revision | GS-GCP integration |
| --- | --- | --- |
| 3DGS | [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) @ `2eee0e26d2d5fd00ec462df47752223952f6bf4e` | [recipe](configs/m3m_gcp_native_quarter_3dgs_3k_recipe_v1.json), [renderer adapter](configs/m3m_gcp_native_quarter_3dgs_renderer_adapter_v1.json), `patches/3dgs_original/` |
| 2DGS | [hbb1/2d-gaussian-splatting](https://github.com/hbb1/2d-gaussian-splatting) @ `335ad612f2e783a4e57b9cbc4d1e167bd599fc98` | [recipe](configs/m3m_gcp_native_quarter_2dgs_3k_recipe_v1.json), [renderer adapter](configs/m3m_gcp_native_quarter_2dgs_renderer_adapter_v1.json), `patches/2dgs/` |
| PGSR | [zju3dv/PGSR](https://github.com/zju3dv/PGSR) @ `de24f1a38b350387e8d8fe381b2cd70c1ae946e7` | [recipe](configs/m3m_gcp_native_quarter_pgsr_3k_recipe_v1.json), [renderer adapter](configs/m3m_gcp_native_quarter_pgsr_renderer_adapter_v1.json), `patches/pgsr/`, `compat/pgsr/` |
| RaDe-GS | [HKUST-SAIL/RaDe-GS](https://github.com/HKUST-SAIL/RaDe-GS) @ `d72f20792005ae1d6555a82aa2d15345f247604e` | [recipe](configs/m3m_gcp_native_quarter_rade_gs_3k_recipe_v1.json), [renderer adapter](configs/m3m_gcp_native_quarter_rade_gs_renderer_adapter_v1.json), `patches/rade_gs/` |
| QGS | [will-zzy/QGS](https://github.com/will-zzy/QGS) @ `74d05c945e99fcaef7afe5a8831903be71ad9b55` | [recipe](configs/m3m_gcp_native_quarter_qgs_3k_recipe_v1.json), [renderer adapter](configs/m3m_gcp_native_quarter_qgs_renderer_adapter_v1.json), `patches/qgs/` |
| GSPrior | [takeshie/GSPrior](https://github.com/takeshie/GSPrior) @ `dcb7c89fb6b60f068b440de45d064ecc7fbcba55` | [recipe](configs/m3m_gcp_native_quarter_gsprior_3k_recipe_v1.json), [renderer adapter](configs/m3m_gcp_native_quarter_gsprior_renderer_adapter_v1.json), `patches/gsprior/`, `compat/gsprior/` |
| SOF | [r4dl/SOF](https://github.com/r4dl/SOF) @ `b9eb4170c843014f5f96d54924976161bd675469` | [recipe](configs/m3m_gcp_native_quarter_sof_3k_recipe_v1.json), [renderer adapter](configs/m3m_gcp_native_quarter_sof_renderer_adapter_v1.json), `patches/sof/` |
| CityGaussianV2 | [Linketic/CityGaussian](https://github.com/Linketic/CityGaussian) @ `e84c7c8774dd11d3f4189be3488e1220afa20a86` | [recipe](configs/m3m_gcp_native_quarter_citygaussian_v2_3k_recipe_v1.json), [renderer adapter](configs/m3m_gcp_native_quarter_citygaussian_v2_renderer_adapter_v1.json), `patches/citygaussian_v2/` |
| CityGS-X | [gyy456/CityGS-X](https://github.com/gyy456/CityGS-X) @ `27617f2486505e3b6fe75345edf7c2b11161bc2a` | [recipe](configs/m3m_gcp_native_quarter_citygs_x_3k_recipe_v1.json), [renderer adapter](configs/m3m_gcp_native_quarter_citygs_x_renderer_adapter_v1.json), `patches/citygs_x/`, `compat/citygs_x/` |
| MetroGS | [M3phist0/MetroGS](https://github.com/M3phist0/MetroGS) @ `8cf9ac13c0c34b65c1a935d181c4634909e60f3f` | [recipe](configs/m3m_gcp_native_quarter_metrogs_3k_recipe_v1.json), [renderer adapter](configs/m3m_gcp_native_quarter_metrogs_renderer_adapter_v1.json), `patches/metrogs/` |

To reproduce an integration, clone the method's official repository at the
listed revision, follow its recipe to create the isolated environment and apply
only the listed compatibility changes, bind the common native-quarter dataset,
and export the required RGB/depth buffers through the corresponding renderer
adapter. The exported packet is then evaluated by the common implementation in
`code/gcp/`; no GCP observation or test residual is exposed to training.

The exact upstream source inventory and redistribution boundary are recorded in
the [method registry](configs/m3m_gcp_native_quarter_method_registry_v3.json)
and the [eight-method source freeze](docs/protocol_evidence/m3m_eight_method_source_freeze_v1.json).

## Scope

- Six UAV scenes: 3K, 5K, 10K, 20K, 50K, and 100K.
- Frozen raw-pixel annotation provenance and deterministic projection into the
  benchmark undistorted camera track.
- Surveyed control/checkpoint registration and independent checkpoint metrics.
- Formal metric depth: alpha-normalized expected camera-z, `M1 / A`.
- Reproducible per-method worktrees, environments, build caches, run roots,
  manifests, hashes, and failure records.

GS-GCP is method-independent. GCP annotations, split roles, survey coordinates,
and residuals are never exposed to training, checkpoint selection, early
stopping, or hyperparameter selection.

## Formal Experiment Policy

The sole active entry point is
[`configs/m3m_gcp_native_quarter_current.json`](configs/m3m_gcp_native_quarter_current.json).
It currently selects protocol `m3m_gcp_native_quarter_geometry_v2`, the frozen
COLMAP-native-quarter data release, the v2 protocol release pin, and the v2
method registry.

Every method must use the exact same COLMAP 4.0.4 undistorter images, PINHOLE
cameras, initial sparse model, and train/holdout split from
`M3M-GCP-colmap-native-quarter-v1`. No method-specific R4 resize or the former
1600-pixel loading rule is part of the active benchmark. The complete human
contract is
[`docs/GS_GCP_NATIVE_QUARTER_ACTIVE_PROTOCOL_V2.md`](docs/GS_GCP_NATIVE_QUARTER_ACTIVE_PROTOCOL_V2.md).

Training authorization and completed-result state are owned only by
[`configs/m3m_gcp_native_quarter_method_registry_v3.json`](configs/m3m_gcp_native_quarter_method_registry_v3.json).
At the `m3m-gcp-3k-10method-v1` snapshot, original 3DGS, 2DGS, PGSR,
RaDe-GS, QGS, GSPrior, SOF, CityGaussianV2, CityGS-X, and MetroGS have completed
their single-seed 3K formal runs. GOF is historical-complete-retired and is not
part of the active ten-method pool. The six-scene matrix remains locked at this
snapshot. Older clean-R4 and native-quarter-v1 execution assets are provenance
only and cannot authorize or supply a new result.

## Repository Layout

- `code/gcp/`: release, annotation, packet, evaluator, and verification code.
- `code/colmap/`: benchmark SfM preparation and validation.
- `configs/`: frozen protocol and isolation contracts.
- `docs/`: benchmark, metric, and experiment specifications.
- `patches/`: narrowly scoped build-compatibility patches for upstream methods.
- `scripts/`: reproducible benchmark and experiment entry points.

## Data Boundary

Raw imagery, surveyed coordinates, checkpoints, packets, and run outputs are
excluded from Git. They are referenced by immutable manifests and SHA-256
records. Algorithms receive read-only source mirrors and write only to unique,
method-specific run directories.

## Name And Compatibility

The public benchmark name is **GS-GCP Benchmark**. Some already-frozen release
and packet files contain older schema strings. Those strings remain accepted as
immutable wire identifiers so their hashes and reproducibility are preserved;
they are not the project name and must not be used for new run namespaces.
See [`docs/LEGACY_WIRE_IDENTIFIERS.md`](docs/LEGACY_WIRE_IDENTIFIERS.md).
