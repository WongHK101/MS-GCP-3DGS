# M3M-GCP LiDAR rendered-surface formal protocol v1

## 1. Status and scope

This document defines the candidate publication protocol for the six-scene
LiDAR track.  Until the exact repository commit receives an independent
review verdict, it is **not execution-authorized**.  The formal scene set is
3K, 5K, 20K, 10K, 50K, and 100K.  The non-monotonic naming order is retained
from data collection; aggregation never weights a scene by its name, area,
image count, or LiDAR point count.

The LiDAR track is evaluation-only.  LiDAR data, derived reference points,
reference masks, and LiDAR metrics are forbidden during training, prior
generation, hyperparameter choice, checkpoint choice, seed choice, or failure
recovery.  One seed and each method's frozen native recipe remain unchanged.

## 2. Frozen registration and height frames

Each scene uses the single common Sim(3) already frozen by
`m3m_gcp_native_quarter_geometry_v2`.  Per-method Sim(3), ICP, result-dependent
alignment, and metric-driven refitting are forbidden.  Reconstruction points
are mapped through that Sim(3), converted from EPSG:4545 to EPSG:32649 with
`always_xy=True`, and evaluated in metres.

The LiDAR Z values are ellipsoid heights.  They receive the one release-wide,
method-independent normal-minus-ellipsoid offset of
`23.980600991639484 m`.  This is a declared datum approximation, not a fitted
method correction.  All KD-tree coordinates and distances are float64 in a
scene-local metric frame.  Absolute UTM float32 coordinates are forbidden.

## 3. Reference definition and ROI

For each scene, the ROI is the convex hull of all active frozen control and
checkpoint points in EPSG:32649, buffered by exactly 8 m.  The same ROI is used
for every method.  No result-dependent crop, visibility mask, or manual
failure mask is allowed.

All finite LiDAR returns inside the ROI define the reference sample set.  No
LAS-class or return-number filter is applied because this release does not
provide a semantically reliable classification contract (the validated 3K
tiles are class 0).  This choice, including lower vegetation returns and real
temporal change, is part of the benchmark definition and must be discussed as
a limitation rather than adjusted after seeing method results.

Reference points are quantized on a 5 cm grid anchored by flooring the ROI's
minimum UTM X/Y to that grid, with Z origin 0.  Packed voxel IDs are decoded to
deterministic voxel centres.  First-point representatives and order-dependent
deduplication are forbidden.

## 4. Reconstructed surface definition

The common surface is the union of backprojected
`alpha_normalized_expected_camera_z` samples from the frozen metric-depth
packet schema.  It is a rendered expected-depth surface, not raw Gaussian
centres and not a universal physical mesh.

For every scene and method:

- the view allowlist is exactly every `train` image in the frozen RGB split;
- test RGB is never read and GCP annotations do not select views;
- all methods must have the identical image-name set;
- camera domain is the COLMAP 4.0.4 native-quarter PINHOLE output;
- pixel centres are zero-based;
- accumulated alpha must be at least 0.5;
- pixels are sampled at stride 4 from origin `(0, 0)`;
- samples outside the common ROI are discarded;
- the reconstruction uses the same 5 cm deterministic voxel grid as LiDAR.

The 3K pilot's 66-view packets therefore validate the numeric implementation
but do not become formal v1 results.  Existing models do not need retraining;
3K only needs a formal v1 depth export and LiDAR rerun using all 82 training
views.

## 5. Metrics and display policy

Bidirectional nearest-neighbour distances are evaluated with a fixed
`1e-9 m` threshold-comparison epsilon.  The evaluator retains all 17 numeric
fields for audit and diagnostics.

The paper's main LiDAR table contains only:

1. F1 at 10 cm;
2. Precision at 10 cm;
3. Recall at 10 cm;
4. symmetric Chamfer-L1, defined as the mean of reconstruction-to-reference
   mean distance and reference-to-reconstruction mean distance.

The appendix reports F1 at 5 cm, F1 at 20 cm, accuracy p95, and completeness
p95.  Rank changes across 5/10/20 cm are reported as sensitivity, not used to
choose a new threshold.  Normal consistency, Hausdorff, F-AUC, and additional
headline metrics are not authorized by v1.

## 6. Ranking and failures

Within a scene, methods are ordered by F1@10 cm descending, then Chamfer-L1
ascending, then Precision@10 cm descending.  The numeric tie tolerance is
`1e-9`.

Dataset-level values are unweighted arithmetic means of the six scene-level
metrics.  Pooling all points across scenes is forbidden because it would let
50K and 100K dominate.  A method receives a formal six-scene rank only after
all six scenes are `COMPLETE_RANKED`.  OOM, failure, and incomplete scenes are
never assigned fabricated zero-valued geometry metrics.  Such methods remain
visible with completed-scene count and a clearly labelled partial macro
diagnostic, but are unranked overall.

Official rank is computed within each frozen method-registry input class:
RGB+COLMAP only, and RGB+COLMAP with frozen external pretrained geometry
priors.  A combined table may be sorted descriptively, but it must retain the
input-class column and must not claim a cross-class official winner.

## 7. Evidence and storage lifecycle

Every result must bind the exact source release, split, scene Sim(3), ROI,
camera allowlist, model checkpoint, recipe, renderer adapter, evaluator,
verifier, packet manifest, and LiDAR payload hashes.  Float64 bidirectional
distance arrays are retained and independently recomputed into all published
metrics.  The validator must pass centimetre preservation, deterministic
voxel uniqueness, point-order invariance, and chunk-order invariance tests.

Before creating any formal output, the launch gate hashes the nine actual LAZ
files and every NPZ depth packet of the selected method.  It checks each
packet's byte count, SHA-256, exact ten-key inventory, array dtype and camera
shape, image name, and `split=train` declaration.  A hash of only an inventory
or packet manifest is insufficient.

Raw rendered depth packets are scratch artifacts.  They may be deleted only
after independent metric recomputation passes and a lightweight, hash-bound
archive containing metrics, distance arrays, manifests, logs, and failure
evidence passes verification.  Final models and formal evidence remain on
901.  A failed or OOM run retains its command, environment, resource trace,
last valid progress, stderr, and cgroup OOM counters.

## 8. Declared reference limitations

LiDAR was acquired on 2026-08-12.  RGB was acquired on 2026-06-02 for 3K,
5K, and 20K (71 days), and on 2026-06-10 for 10K, 50K, and 100K (63 days).
Vegetation, vehicles, movable objects, and construction changes are therefore
real reference uncertainty.  The constant vertical bridge is also an
approximation.  Neither limitation may be removed through method-specific
alignment, result-dependent masks, or post-hoc threshold selection.

## 9. Byte-level source and method identities

The protocol ID alone is not an identity.  Formal v1 binds the geometry-v2
release pin (`7bf9db0c...ffe8e6`), release manifest
(`21fbac75...28bea4`), the six individual `common_sim3.json` hashes, the
surveyed GCP coordinate and role files, and the source native-quarter release
and split hashes.  It additionally binds the file SHA-256 and canonical
SHA-256 of all six `NATIVE_QUARTER_INPUT_MANIFEST.json` files, requires the
exact four COLMAP model files in each manifest, and re-hashes those model bytes
at launch.  The exact values are normative in the machine contract.

The LiDAR contract contains the relative path, byte count and SHA-256 of
`cloud0.laz` through `cloud8.laz`.  The evaluator has no LAZ-directory
override: it always reads the exact `lidars/terra_laz_1_4` directory under the
reviewed LiDAR release root.

The active method registry is bound by SHA-256
`b409b164...a5043`.  The formal pool is exactly ten methods.  3DGS, 2DGS,
PGSR, RaDe-GS, QGS, GSPrior and SoF are in `rgb_colmap_only`;
CityGaussianV2, CityGS-X and MetroGS are in
`rgb_colmap_external_geometry_prior`.  GOF is not in the active formal pool.
Every scene attempt supplies one immutable, ordered ten-method manifest whose
checkpoint, recipe, renderer adapter and packet manifest paths and hashes are
rechecked before any output directory is created.  Subset or ad-hoc method
manifests are rejected.

## 10. Frozen implementation, artifacts and launch gate

The exact evaluator, independent verifier, six-scene ranker, artifact schema
and launch gate paths and SHA-256 values are frozen in the machine contract.
The artifact schema fixes the NPZ key names, float64 local-metre coordinate and
distance dtypes, shapes, units, origins and point counts.  It also fixes JSON
canonicalization, lightweight archive inventory rules, the 17 diagnostic
metric fields, scene-execution authorization, method-result, independent
verification-report and six-scene results-manifest field sets.

Ranking compares each key in order and treats an absolute difference at most
`1e-9` as tied for that key.  If every key is tied, methods receive the same
competition rank; equal-rank display is by `method_id`, and the next rank skips
the number of tied rows.  The executable ranker accepts exactly the frozen ten
methods and six scenes.  For every `COMPLETE_RANKED` scene it requires a
hash-bound independent `PASS_VERIFIED_FORMAL_V1` report, checks all result
identity/count/metric fields, implementation hashes, retained-array hashes and
recomputed metrics, then enforces unweighted six-scene macro averaging.
Failed/OOM scenes may not carry result or verifier artifacts, and only 6/6
complete methods receive an official within-class rank.

Formal execution is impossible while this contract is a review candidate.
After review, a separate canonical activation manifest must bind the approved
contract and artifact schema hashes plus the exact clean benchmark commit and
tree.  Protocol activation alone does not authorize a scene.  Each scene also
requires a separately reviewed execution-authorization manifest binding the
exact formal input, full ten-method manifest, benchmark commit/tree and one
fresh absolute output root per method.  The launch gate rechecks that
authorization, all implementation hashes,
the geometry release and scene Sim(3), LiDAR inventory, GCP coordinates,
formal input manifest/COLMAP bytes, method registry/class mapping and all
per-method assets.  It then requires a previously nonexistent absolute output
root.  Formal resume or overwrite is forbidden.

The normative machine-readable contract is
`configs/m3m_gcp_lidar_formal_v1.json`.
