# Control and Checkpoint Split Policy

## Principles

- A scene has one fixed split used by all methods.
- Control GCPs fit the global Sim(3).
- Checkpoint GCPs are held out and provide the accuracy claim.
- Checkpoints must never participate in the Sim(3) fit.
- Control and checkpoint residuals are always reported separately.

## Scene-Adaptive Split

The split depends on available GCP count, spatial coverage, and height variation.
Large scenes should use approximately 20-30% control points and 70-80%
checkpoints, with at least 6-8 controls when enough GCPs exist.

Small scenes such as 3k and 5k may not satisfy this requirement. They can be
used for smoke tests or small-area diagnostics, but they should not carry the
formal georeferenced accuracy claim alone.

## Control Selection

Prefer controls that:

- cover the scene boundary and center;
- include known/base points when available;
- cover height variation when meaningful;
- avoid near-collinear geometry;
- remain visible in enough images for all compared methods.

## Supplementary Stability Checks

- Leave-one-out residuals may be reported for small scenes.
- Control-count sensitivity may be reported for larger scenes, e.g. 4, 6, 8,
  and 12 controls.

