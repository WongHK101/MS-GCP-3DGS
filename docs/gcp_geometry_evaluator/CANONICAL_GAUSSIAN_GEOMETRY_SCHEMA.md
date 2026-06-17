# Canonical Gaussian Geometry Schema

This schema defines the geometry fields required for the Level-1 canonical
Gaussian geometry evaluator. It is an evaluation-side schema, not a training
format.

## Required Fields

Per Gaussian:

- `x`, `y`, `z`: center in the method/COLMAP model frame.
- `opacity`: scalar opacity before or after activation, with activation stated
  in metadata.
- `scale`: anisotropic scale, usually `scale_x`, `scale_y`, `scale_z` or log
  scales with activation stated.
- `rotation`: orientation, usually quaternion or rotation matrix with
  convention stated.

Scene metadata:

- `scene_id`;
- source method name and version;
- coordinate frame;
- camera model source;
- whether the geometry is a single shared support or one branch among multiple
  supports;
- renderer/depth semantics used to generate depth maps.

## Optional Fields

- SH coefficients;
- RGB/color/features;
- semantic branch name;
- per-Gaussian uncertainty or confidence;
- per-Gaussian timestamps or provenance.

## P1 Scope

P1 does not require direct PLY parsing in the evaluator core. It evaluates depth
maps produced from the canonical geometry by a renderer wrapper. The schema
still matters because the wrapper must use a declared geometry representation
instead of method-specific hidden state.

