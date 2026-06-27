# 3K Archived 0.252 m Provenance Diagnostic

This diagnostic audits the archived `gcp_3000_20260602` P1 checkpoint RMSE-3D
of approximately `0.252161 m`. It is not a formal evaluator change.

The runner distinguishes:

- **Stage A1 exact archived replay**: requires exact old source, command, config,
  depth files, annotations, GCP coordinates, split, COLMAP model, patch settings,
  depth semantics, and Sim(3) implementation.
- **Stage A2 forensic reconstruction**: uses recovered but incompletely verified
  artifacts and present-day hashes, and therefore cannot be called exact replay.
- **Stage B one-factor reconciliation**: runs only from a numerically reproduced
  fixed baseline and changes one factor at a time. Incompatible substitutions are
  marked `not_identifiable_as_single_factor`.

No GPU, depth export, training, checkpoint mutation, pointset/split/annotation
mutation, or formal evaluator modification is permitted.
