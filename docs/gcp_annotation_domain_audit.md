# 3K Annotation Pixel-Domain Audit

This diagnostic is a read-only audit for `gcp_3000_20260602`. It checks whether
the frozen release v1.1 annotation pixels are in the same image domain as the
current metric-depth packets and COLMAP cameras used by the Gaussian GCP
evaluator.

The audit does not modify release annotations, GCP coordinates, control or
checkpoint splits, metric-depth packets, checkpoints, Gaussian support, or the
formal evaluator. It uses the frozen annotation rows as the authoritative row
spine and left-joins source/manual rows, archived undistorted rows, packet
manifests, source-view manifests, COLMAP cameras, and evaluator outputs.

The three no-GPU coordinate variants are:

- `A_current_release`: release v1.1 `manual_x/manual_y` as currently used.
- `B_recomputed_raw_to_undistorted`: official manual-source pixels transformed
  from raw distorted COLMAP camera coordinates to the training/rendering
  undistorted COLMAP camera domain.
- `C_archived_undistorted`: previously archived undistorted annotation pixels,
  used only when the pixel-domain transform provenance is present.

The final classification is one of `confirmed`, `likely`, `not_supported`, or
`unresolved`. It is based on camera-model provenance, coordinate-transform
evidence, source/target domain hashes, and pre-declared A/B/C evaluator
diagnostics. It is not selected by whichever coordinate version gives the
lowest residual.
