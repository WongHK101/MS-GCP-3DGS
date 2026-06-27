# MS-GCP-3DGS Protocol Spec

## Purpose

MS-GCP-3DGS is an RGB UAV georeferenced geometry benchmark for Gaussian
Splatting reconstruction methods. The benchmark estimates method-derived
surface/support positions at
manually annotated GCP image locations, registers those positions to surveyed
GCP coordinates with a global Sim(3), and reports held-out checkpoint residuals.

The protocol evaluates the trained Gaussian geometry. COLMAP provides the
shared camera frame and image rays; COLMAP triangulation is not a method result.

## Evaluation Tracks

### Annotation and camera smoke

This track validates that manual GCP image observations, the surveyed GCP table,
camera intrinsics/extrinsics, and Sim(3) residual scripts are internally
consistent. It may use COLMAP-camera triangulation, but it must be reported as a
sanity check only.

### Gaussian geometry evaluation

This is the method-comparison track. For each method, a canonical Gaussian
geometry output is rendered or otherwise converted to metric depth at annotated
views. The evaluator extracts 3D positions at GCP pixels, aggregates
multi-view observations per GCP, fits a global Sim(3) on control points, and
reports held-out checkpoint residuals.

Formal release-mode evaluation is locked to the frozen release config. The
release config defines the annotation tables, GCP table, scene metadata, and
control/checkpoint split. Formal invocations may supply the release config,
depth manifest, scene id, method id, COLMAP model, and output directory, but
must not override annotations, GCPs, splits, metadata, depth semantics, tensor
keys, or depth directory contents outside the manifest. Release mode fixes
`min_valid_observations=1` for all scenes and methods; each GCP reports whether
aggregation used one view, two-view median, or robust multi-view median.

## Required Inputs

- A COLMAP sparse model defining the camera frame used by the method.
- A surveyed GCP table in CGCS2000 / 3-degree Gauss-Kruger CM 108E
  (EPSG:4545) with normal height in the 1985 National Height Datum.
- Manual GCP image observations with image name, pixel coordinates, visibility,
  quality, and confidence.
- A method output compatible with the canonical Gaussian geometry contract, or
  a depth map set rendered from that output by the benchmark renderer.
- A fixed control/checkpoint split for the scene.
- A CRS, pixel-domain, and depth-semantics manifest.

## Main Metrics

Per scene:

- valid observation count and ratio;
- valid GCP count and ratio;
- per-GCP multi-view scatter median, p90, and max;
- control residual RMSE-H, RMSE-Z, and RMSE-3D;
- checkpoint residual RMSE-H, RMSE-Z, and RMSE-3D;
- checkpoint median, p90/p95, and max 3D error;
- failure counts by reason.

The formal release reports all six real-world urban UAV scenes: 3k, 5k, 10k,
20k, 50k, and 100k. Aggregate reporting should include scene-weighted and
GCP-weighted summaries, with 5k identified as a deliberately underexposed
low-light challenge scene rather than a pure area-effect sample.

## Prohibited Operations

- GCP-supervised training.
- Using checkpoint points in the Sim(3) fit.
- Affine shear, local stretching, TPS, or non-rigid warps.
- Silent invalid filtering.
- Selecting the best MS-Splatting branch after seeing the residuals.
- Reporting COLMAP-camera triangulation as Gaussian method accuracy.

## Scene Scale Policy

All six scenes enter per-scene formal reporting. The 3k scene is not only a
smoke test; it is a small formal scene with limited spatial scale. The 5k scene
is a formal low-light challenge scene and should be reported separately when
interpreting illumination sensitivity. Large-area claims should emphasize the
50k and 100k scenes because they provide stronger spatial coverage and are more
representative of survey-scale mapping. Each scene uses its own fixed split,
and all methods use the same split.
