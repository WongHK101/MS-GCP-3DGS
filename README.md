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

The current primary protocol uses release v1.3.0 and a shared benchmark camera
track. Every method must use the same images, COLMAP cameras, initial sparse
model, train/evaluation split, and loaded image dimensions. The common
resolution follows the original 3DGS `--resolution -1` rule: images wider than
1600 pixels are downscaled to width 1600 with aspect ratio preserved; smaller
images are not enlarged.

Each method must first pass the complete 3K pipeline before any six-scene
training is allowed. See
[`docs/GS_GCP_FAIR_EXPERIMENT_PROTOCOL.md`](docs/GS_GCP_FAIR_EXPERIMENT_PROTOCOL.md).

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
