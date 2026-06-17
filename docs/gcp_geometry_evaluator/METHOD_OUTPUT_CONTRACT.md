# Method Output Contract

## Goal

Methods are evaluated through geometry outputs after training. Training code
does not need to consume GCPs. The benchmark evaluator only requires a geometry
representation that can be converted to method-derived depth at annotated
camera views.

## Primary Contract: Canonical Gaussian Geometry

A method should provide a canonical Gaussian geometry output with:

- Gaussian centers in the same COLMAP/model frame used by the camera model;
- opacity;
- scale;
- rotation;
- coordinate-frame metadata;
- depth-rendering compatible schema;
- optional color, SH, or feature fields.

The exact file format may be PLY or another method-specific export, but it must
be convertible into the canonical schema before evaluation.

## Depth-Map Contract

P1 uses a depth-only evaluator core. A renderer wrapper may produce depth maps
from the canonical Gaussian geometry. The depth maps must declare:

- depth semantics: camera-z, ray distance, inverse camera-z, or inverse ray
  distance;
- pixel domain: distorted or undistorted;
- dimensions and filename mapping to COLMAP image names;
- any scale/offset used for integer depth formats;
- invalid-depth encoding.

## MS-Splatting Policy

The main table evaluates each method's predefined default geometry output. If
MS-Splatting exports multiple independent band-specific supports, the benchmark
must not select the best branch after observing the errors. In that case:

- the default branch is specified before evaluation;
- branch-specific residuals may be reported in supplement;
- cross-branch GCP geometry divergence may be reported as a conditional
  diagnostic.

## Disallowed Method-Specific Adaptations

- GCP loss or GCP labels in training.
- Manual branch selection based on GCP residuals.
- Method-specific local georeferencing beyond the common global Sim(3).
- Deleting invalid observations without reporting failure reasons.

