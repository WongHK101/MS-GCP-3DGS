# Validity and Failure Reason Codes

Observation-level failure reasons:

- `missing_image_in_colmap`: annotated image not found in the COLMAP model.
- `missing_depth_map`: no rendered depth map for the image.
- `pixel_out_of_bounds`: annotated pixel outside the depth image domain.
- `domain_mismatch`: annotation and depth/camera pixel domains are inconsistent.
- `insufficient_finite_depth`: patch finite-depth ratio below threshold.
- `nonpositive_depth`: extracted depth is non-positive after semantics conversion.
- `high_patch_mad`: patch depth MAD exceeds the configured scene-normalized rule.
- `projection_behind_camera`: backprojected or projected point is invalid.
- `low_annotation_confidence`: manual annotation confidence below threshold.
- `annotation_not_visible`: annotation marked not visible or rejected.
- `unsupported_depth_semantics`: evaluator cannot convert the provided depth semantics.

GCP-level failure reasons:

- `insufficient_valid_observations`: fewer than the required valid observations.
- `high_multiview_scatter`: multi-view scatter exceeds the configured threshold.
- `missing_survey_coordinate`: surveyed GCP coordinate is unavailable.

Scene-level status values:

- `ok`: enough valid controls and checkpoints.
- `smoke_only`: evaluator ran, but split size or scene coverage is insufficient
  for formal benchmark claims.
- `failed`: evaluator could not produce valid control/checkpoint residuals.

