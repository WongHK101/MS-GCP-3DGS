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

Small scenes such as 3k and 5k may not satisfy this requirement. They still
enter per-scene formal reporting with their frozen split, but they should not
alone carry the survey-scale georeferenced accuracy claim. The 5k scene is a
formal low-light challenge scene and should be interpreted as an illumination
robustness case, not only as an area-scale case.

Formal release evaluation requires all frozen control GCPs to be valid for the
method. If any frozen control is unavailable, the primary status is
`incomplete_fixed_control_coverage`. A separately labeled diagnostic fit may use
the available subset, but it must not enter the formal primary table.

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
