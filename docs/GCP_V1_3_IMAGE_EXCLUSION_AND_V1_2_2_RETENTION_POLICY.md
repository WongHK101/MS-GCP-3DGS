# GCP v1.3 Observation QC and v1.2.2 Retention Policy

Status: protocol-freeze candidate, 2026-07-17.

## Image `DJI_20260610161948_0002_D.JPG`

The final manual review supersedes the earlier image-level exclusion candidate.

For the v1.3 candidate:

- retain the byte-identical raw acquisition file as a normal training image;
- retain its camera record and do not rebuild a method's source model merely to
  remove this image;
- keep the G33 and G39 rows as reviewed provenance observations with
  `quality=ambiguous` because the marks are too blurred for formal use;
- exclude those two rows from formal evaluation through the common Good-only
  observation rule;
- do not apply image-level exclusion, GPS-specific filtering, or a special GPS
  failure narrative to this image.

`Ambiguous` and `Not visible` remain distinct observation states. Both are
ineligible for the formal Good-only row set, but neither normally removes the
underlying image from method training. The earlier image-exclusion audit and
manifest are retained only as superseded diagnostic evidence and must not be
used as a v1.3 split or training-image input.

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

Do not edit v1.2.2 to apply later observation-QC decisions. Once v1.3 is approved,
v1.2.2 should be labeled `superseded_for_primary_experiments_by_v1_3` while
remaining available as a versioned sparse-control diagnostic/audit release.
The v1.3 training source keeps `0002` as a normal image. Formal observations are
derived independently by the Good-only eligibility rule, so the two reviewed
blurred rows are excluded without changing the training image list.
