# Six-scene annotation-domain audit

This document defines the Stage-1 read-only audit implemented by
`code/gcp/audit_6scene_annotation_domain.py`.

The audit checks whether release v1.1 manual annotation pixels, raw COLMAP
cameras, and training/render metric-depth packet cameras occupy compatible
pixel domains for the six MS-GCP scenes. It preserves every release v1.1
observation row and records missing or unresolved artifacts explicitly.

Stage 1 may draft a v1.2 pixel-domain schema, but it does not freeze release
v1.2, does not generate formal v1.2 CSVs, does not change the formal evaluator,
and does not run GPU packet export or training.

The pose-equivalence gate is frozen at:

- camera-center difference <= `1e-8` model units;
- rotation angular difference <= `1e-8` radians.

If a source camera and target camera are not pose equivalent and no verified
explicit COLMAP undistortion remap exists, the observation is marked
`non_equivalent_camera_pose_unmappable_without_depth`.

The canonical normalized source ray draft used for v1.2 planning is:

- `normalized_x`, `normalized_y`: source-camera distortion-inverted normalized
  image-plane coordinates;
- source camera-frame ray: `[normalized_x, normalized_y, 1]`;
- `normalized_unit_ray`: normalized source camera-frame ray;
- pixel convention: 0-based pixel centers;
- source image orientation: actual decoded pixel matrix orientation;
- camera convention: COLMAP world-to-camera.
