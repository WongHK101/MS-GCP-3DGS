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
At this revision, original 3DGS and 2DGS retain completed and re-locked 3K
formal runs; eight additional methods are in one gated seed-0 3K batch, GOF is
historical-complete-retired, and the six-scene matrix remains locked. PGSR has passed
its technical qualification and is the sole method on the current one-use 3K/30K
training allowlist; the other seven batch methods remain locked. Older clean-R4 and
native-quarter-v1 execution assets are provenance only and cannot authorize or supply
a new result.

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
