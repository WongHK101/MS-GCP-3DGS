# COLMAP Smoke vs Gaussian Evaluation Boundary

## COLMAP Smoke

COLMAP-camera GCP triangulation may be used to validate:

- manual annotation consistency;
- GCP coordinate table integrity;
- camera intrinsics and distortion handling;
- Sim(3) residual code;
- approximate expected error scale.

It is not a Gaussian method result and must not be used to compare UMGS,
MS-Splatting, or any other Gaussian reconstruction method.

## Gaussian Geometry Evaluation

Gaussian evaluation uses method-derived depth or support positions from each
trained Gaussian model. The method output is queried at the same annotated GCP
image locations. The resulting method-derived GCP points are then registered to
surveyed GCP coordinates with the common Sim(3) protocol.

## Reporting Boundary

Any table or figure comparing methods must use Gaussian-derived points, not
COLMAP-triangulated points. COLMAP smoke can appear in an implementation,
data-quality, or evaluator-validation section only.

