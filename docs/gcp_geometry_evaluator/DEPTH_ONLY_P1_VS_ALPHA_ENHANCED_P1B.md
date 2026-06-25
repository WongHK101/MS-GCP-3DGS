# Depth-Only P1 vs Alpha-Enhanced P1b

## P1: Depth-Only Evaluator

P1 uses rendered depth maps and manually annotated GCP pixels. It extracts a
robust depth value in a local patch, backprojects the pixel to a method-derived
3D point, aggregates multi-view observations per GCP, and evaluates a global
Sim(3) control/checkpoint split.

P1 records the following diagnostics:

- finite-depth ratio in the patch;
- robust patch depth median or trimmed mean;
- depth MAD in the patch;
- optional local discontinuity indicator;
- valid observation count per GCP;
- multi-view scatter per GCP.

In the current P1 implementation, patch MAD, discontinuity indicators, and
multi-view scatter are recorded diagnostics unless a nonzero threshold is
explicitly configured and preregistered. They must not be described as active
formal validity gates by default.

P1 must record that no alpha or second-moment filtering was used unless those
maps are explicitly available.

The current Graphdeco-derived renderer artifact exported by
`export_gaussian_depth_maps.py` is labeled
`alpha_weighted_unnormalized_inverse_camera_z`. That artifact is not an
acceptable formal P1 metric depth input until alpha/weight normalization or a
direct camera-z/ray-distance export is available.

## P1b: Alpha-Enhanced Evaluator

P1b may add:

- accumulated alpha / opacity maps;
- alpha threshold sweeps;
- alpha-weighted expected camera-z depth;
- depth second moment or variance;
- opacity-aware failure reasons.

P1b is not a blocking requirement for P1. It requires renderer support and must
not be silently substituted into P1 results.
