# GCP v1.3 Image Exclusion and v1.2.2 Retention Policy

Status: protocol-freeze candidate, 2026-07-17.

## Image `DJI_20260610161948_0002_D.JPG`

This is an **image-level exclusion**, not an annotation-quality label.

For the v1.3 candidate:

- retain the byte-identical raw acquisition file;
- exclude the image from every method's canonical training image list and
  source camera model;
- exclude every annotation observation attached to the image before applying
  per-observation `Good`, `Ambiguous`, `Not visible`, or blur QC;
- do not reinterpret those observations as `Not visible` or blur;
- record the reason as
  `predeclared_image_level_feature_pose_qc_not_gcp_or_3dgs_residual`.

The distinction matters:

- `Not visible`, blur, and `Ambiguous` are point-image observation states. They
  reject one observation but do not normally remove the image from training.
- the `0002` rule rejects the entire image for both training and formal
  observations, independent of which GCP is marked in it.

The exclusion is residual-blind. It was predeclared from image feature/pose QC,
not selected from GCP or 3DGS errors.

## Frozen v1.2.2 release

`gcp_benchmark_release_v1_2_2_pixel_domain_20260628` is the accepted historical
pixel-domain release under
`E:\datasets\M3M-GCP\scenes\gcp_manual_annotations_v1_2_2`.

It freezes:

- 611 canonical observations across six scenes;
- the v1.2.2 point table and sparse-control split;
- raw-pixel coordinates, raw image hashes, decoded orientation records, and
  deterministic observation identities;
- verified raw-to-undistorted benchmark target projections;
- source/target camera and pose provenance;
- 420 unique image-level mapping records;
- inclusion provenance, scene metadata, payload manifest, and release root
  digest.

It does not contain the later 1,383-row multi-view working annotation set or
the proposed v1.3 control-heavy split.

## Retention decision

Keep v1.2.2 byte-for-byte and read-only. It remains necessary for:

1. reproducing and auditing the prior sparse-control diagnostic results;
2. proving how the v1.1 raw/undistorted pixel-domain defect was corrected;
3. validating historical packet/evaluator manifests and hashes;
4. preserving the six direct v1.2.2 references to `0002`.

Do not edit v1.2.2 to apply the new image exclusion. Once v1.3 is approved,
v1.2.2 should be labeled `superseded_for_primary_experiments_by_v1_3` while
remaining available as a versioned sparse-control diagnostic/audit release.
The clean v1.3 training source must instead be built from a manifest-driven
view that omits `0002`; physical deletion from raw acquisition is unnecessary
and would break historical provenance.
