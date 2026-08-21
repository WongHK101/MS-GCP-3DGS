# M3M-GCP 100K ten-method time/space execution plan v1

> **SUPERSEDED / DO NOT EXECUTE.** Activation v1 exposed a deterministic
> `RLIMIT_NOFILE` preflight omission. Its artifacts are retained only for audit;
> the current executable candidate is
> `M3M_GCP_100K_TEN_METHOD_TIME_SPACE_EXECUTION_PLAN_V2.md`.

Status: `REVIEW_CANDIDATE_NOT_EXECUTION_AUTHORIZED`

Plan ID: `m3m-gcp-native-quarter-100k-ten-method-seed0-v1`

This plan covers only `gcp_100000_20260610`.  The prepared 5K, 20K, 10K and
50K scenes remain locked and must not be trained, rendered or formally
evaluated under this plan.  No GPU work is authorized until the exact plan and
the revised rolling-packet LiDAR candidate receive the second independent
review verdict.

## 1. Frozen scientific scope

- Input is the byte-frozen native-quarter release.  The 100K scene contains
  2,196 train images and 314 held-out images at 1414 x 1024.  Training reads
  every train image and no held-out RGB, GCP annotation, surveyed coordinate,
  LiDAR file or LiDAR-derived artifact.
- The method pool and order are exactly: 3DGS, 2DGS, PGSR, RaDe-GS, QGS,
  GSPrior, SoF, CityGaussianV2, CityGS-X and MetroGS.  GOF is retired.
- Every method uses seed 0 only.  There is no multi-seed run and no
  significance claim.
- RGB+COLMAP-only and frozen external-pretrained-geometry-prior methods remain
  separate official ranking classes.  GSPrior's scene-internal TSDF remains in
  the RGB+COLMAP-only class; CityGaussianV2, CityGS-X and MetroGS remain in the
  external-prior class.
- Iteration budgets, loss weights, thresholds and prior routes are inherited
  from the already qualified frozen recipes.  Only scene-derived inventories
  are recomputed from the 2,196 frozen training cameras: image lists, neighbor
  lists, bounding boxes and official partition assignments.  No benchmark
  result may change them.

## 2. Method actions and fixed budgets

| Order | Method | 100K action | Frozen budget/route | 100K time envelope including packet + LiDAR |
|---:|---|---|---|---:|
| 0 | 3DGS | reuse the existing validated 100K model; do not retrain | 30K checkpoint already complete, SHA-256 `8d923601...72e1f5` | 2.5--4 h |
| 1 | 2DGS | new seed-0 run | 30K iterations | 3.5--7 h |
| 2 | PGSR | new seed-0 run | 30K iterations, frozen nearest-eight rule | 4--8 h |
| 3 | RaDe-GS | new seed-0 run or record OOM | 30K iterations, frozen nearest-eight rule | 3--7 h or earlier OOM |
| 4 | QGS | new seed-0 run or record OOM | 30K iterations | 12.5--28 h or earlier OOM |
| 5 | GSPrior | new seed-0 run | 40K iterations and frozen internal TSDF schedule | 7.5--18 h |
| 6 | SoF | new seed-0 run | 30K iterations, frozen unbounded/TnT route | 3.5--8 h |
| 7 | CityGaussianV2 | new seed-0 run | official MatrixCity-aerial 4 x 4 partitions; 30K coarse + 60K fine per official block route, then official merge | 39--76 h |
| 8 | CityGS-X | new seed-0 run | frozen DAv2 prior and official 100K all-block 100K-iteration route | 15--40 h |
| 9 | MetroGS | new seed-0 run | frozen Pi3-Align + MoGe-2 route; 150K effective image iterations / 37.5K optimizer steps | 21--52 h |

The ranges are capacity estimates, not budgets or early-stop thresholds.  They
use the measured 3K training times and the existing 3DGS 100K export.  The
existing 3DGS exporter used 740.56 s for 211 views (3.51 s/view), projecting
about 2.14 h for 2,196 views.  Allowing renderer variation gives 2--3.5 h of
packet export per successful method.  The entire successful queue is therefore
expected to take roughly 5.5--12 wall-clock days on one GPU; 7--10 days is the
working expectation.  OOM/failure can shorten, not lengthen, an individual
attempt because recipe-changing rescue runs are forbidden.

## 3. Two-phase execution

### Phase A: models and immutable attempts

1. Revalidate the formal 100K input bytes, clean code/source commits, exact
   environments, free disk and absence of foreign GPU processes.
2. Validate the existing 3DGS checkpoint byte count (2,340,432,588), SHA-256
   and evidence; no 3DGS retraining is permitted.
3. For each remaining method in the fixed order, create one fresh run root,
   generate only its authorized prior, run the frozen budget once, and retain
   the final formal checkpoint plus resource and command evidence.
   At guard admission for either the prior or training phase, the authorized
   run root must be completely absent (including symlinks).  The prior phase
   never creates that root; the training guard exclusively creates a new empty
   root immediately before launch, and the training child must create the declared final model
   products inside it.  Therefore a pre-positioned valid checkpoint/PLY plus a
   zero-work child cannot be accepted as a formal attempt.
   A prior child is successful only after its exact prior manifest/PASS marker
   and every declared prior product pass method-specific validation.  For
   CityGaussianV2 this rehashes all 2,196 depth arrays and the 2,196-row scale
   file; for CityGS-X it additionally rehashes all 2,196 multi-view masks; for
   MetroGS it rehashes all 2,196 joint depth/mask arrays, the scale and
   multi-view files, four Pi3 block pointmaps and the merged pointmap.
   Deleting or changing one item invalidates the phase.  Training cannot start without that
   immutable prior success marker and a second revalidation of the prior
   product.  Likewise, a zero exit from training is not success until the
   method's frozen final iteration/checkpoint and required companion files are
   present, non-empty and internally hash-consistent.  Gaussian PLYs are parsed
   as binary PLY, must have a positive vertex count, exact binary extent,
   finite sampled values and the method-specific field schema (including the
   distinct 2DGS, MetroGS and CityGS-X layouts).  Torch checkpoint and NPZ
   containers are structurally checked without unsafe model loading.  A zero exit without the
   required product is closed as structured `FAILED_UNRANKED` evidence.
4. A method-specific CUDA OOM, host OOM or technical failure is closed as
   `OOM_UNRANKED` or `FAILED_UNRANKED`; its immutable failure evidence is
   retained and the queue continues.  Batch size, resolution, partition rule,
   loss, checkpoint or iteration budget is not changed to rescue it.
5. After all ten attempts, freeze one exact ordered ten-row attempt manifest.
   Each `READY_FOR_EVALUATION` row binds an immutable model-identity manifest
   containing the final model/config inventory and any generated prior
   manifest/PASS marker, plus recipe and renderer bytes.  The packet guard
   rehashes every inventoried file immediately before export; freezing only the
   identity-manifest file is insufficient.
   Failed/OOM rows have null model fields and bind failure evidence.  One
   failed method therefore cannot invalidate successful methods.

   Except for the reused packet-only 3DGS model, every READY identity must
   contain the exact fixed-path training `phase_success.json`; methods with an
   authorized prior must also contain the exact prior success marker.  Both the
   attempt builder and packet guard re-expand the frozen phase command and
   require its SHA-256 to equal the marker's `command_sha256`.  Missing,
   relocated, extra-phase, or wrong-command markers fail closed.

The execution-plan file, recipe manifest, method registry, attempt-manifest
output, model-identity directory and scene-freeze output are all exact paths in
the frozen plan.  The builders reject alternate CLI destinations.  Each model
identity must inventory the actual method-specific final model validated by
the runner; a valid but unrelated checkpoint or decoy PLY cannot satisfy the
freeze.
Every exclusive `phase_success.json` records the frozen budget, completion
evidence, and a sorted absolute path/byte-count/SHA/validator inventory for all
phase products.  The attempt builder rehashes those rows and requires the
training rows to equal the independently reconstructed method-specific final
model set.  Packet launch repeats this nested success-product validation, so a
prior or model changed after the attempt freeze is rejected.

Before Phase A formal launches, each 100K scene recipe must be materialized as
a hash-bound file and mechanically checked against its qualified 3K parent:
only scene identity, train-camera-derived lists/partitions and paths may differ.
The CityGaussianV2 4 x 4 choice is the frozen upstream MatrixCity-aerial
configuration, selected before any 100K result; it is not a result-driven
memory workaround.

The manifest contains ten recipes: nine fresh-training recipes and one
packet-only recipe for the already frozen 3DGS model.  Every recipe contains
an exact all-2,196-view packet-export command, evaluation-adapter source/file
bindings and conformance-evidence hashes.  The reused 3DGS recipe exposes no
training phase and the guard rejects any attempt to retrain it.

The authoritative initialization is the already published, standard COLMAP
4.0.4 native-quarter model produced from **all 2,510 images before the split**.
Its 2,510-image `images.bin`, 1,262,896-point `points3D.bin`, PINHOLE camera and
package audit are hash-bound.  No new SfM, train-first undistortion, feature
extraction, matching or bundle adjustment is part of this plan.  The obsolete
failed train-first undistorter candidate was removed under a hash-bound cleanup
receipt.

Per-method views inherit the exact reviewed 3K semantics.  3DGS, 2DGS, PGSR,
RaDe-GS, QGS and SoF consume the exact formal 2,196-view training root directly;
QGS receives only its two required image aliases.  GSPrior derives its required
camera-coordinate normalization from that same hash-bound formal root just in
time, and the guard verifies the normalization manifest, source hashes, output
hashes, image symlink and sparse-file symlinks before training or packet export.
CityGaussianV2 and CityGS-X
receive the 2,196 training image records selected byte-for-byte from the
all-image model plus the byte-identical shared all-image `points3D.bin`, because
their qualified consumers explicitly select training observations.  MetroGS
alone receives a reciprocal post-SfM training track closure: held-out image
records and their track elements are removed, 511 points having no remaining
training observation are omitted, and every retained image record, point
XYZ/RGB/error and track index remains byte-identical.  No pixel is decoded,
resampled or re-encoded by either compatibility materializer.

All method RGB directories are symlinks to the frozen formal train JPEGs and
all reusable sparse files are same-filesystem hardlinks.  The preparation
evidence has status
`PASS_PER_METHOD_INPUT_PREPARATION_NO_TRAINING_NO_PRIOR`, proves that all-image
SfM precedes the split, and is bound by exact SHA into the plan and every recipe.

Packet export uses a separate evaluation-only all-train camera root.  Its
2,196 camera records, initial PLY and RGB symlink are byte-identical to the
formal train input.  Because generic COLMAP loaders require a complete
`cameras.bin` / `images.bin` / `points3D.bin` triplet, it adds only the
deterministic eight-byte zero-point `points3D.bin` already qualified by the
3DGS 100K evaluation path.  The file contains no geometry or tracks, is never
visible to training or prior construction, and has SHA-256
`af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc`.
The root manifest and external preparation evidence are byte-identical with
SHA-256 `6b31e460ba80b17e85ac284c55165bfbc6c6b3a85411ad88e785ed8fe6645aac`;
both the outer guard and packet dispatcher rehash the full sparse identity and
verify that no held-out RGB, GCP or LiDAR input is present.

MetroGS's official Pi3 partitioner initially emits byte copies under its block
directories.  After the official partition is fixed and before Pi3 runs, the
preparation wrapper replaces each copy with a same-filesystem hardlink to its
byte-identical frozen train JPEG, then verifies an exact device/inode multiset.
This changes no partition, name or pixel and adds zero physical RGB bytes.

### Phase B: rolling packet evaluation

For each `READY_FOR_EVALUATION` method, and never for a failed method:

1. export exactly all 2,196 train-view metric-depth packets into that method's
   plan-frozen packet root;
2. create a per-method pre-result authorization binding the full ten-attempt
   manifest, selected method ID, exact packet-manifest path/SHA and one fresh
   formal output root;
3. run the fail-closed launch gate, formal LiDAR evaluator and independent
   metric verifier;
4. build and verify the lightweight archive containing result JSON, float64
   bidirectional distance arrays, manifests, hashes, logs and failure/resource
   evidence;
5. only after verifier and archive PASS, delete that method's raw packet NPZs
   and transient surface/reference copies.  Final models, distance arrays,
   metrics and evidence stay on 901.

Only one method's raw packet set may exist at a time.  The final ten-method
scene results manifest includes all failures with null result/report paths and
never substitutes zero metrics.

## 4. Storage model and hard gates

Observed on 901 in no-card mode before this review:

- free persistent bytes after corrected input preparation and obsolete-attempt
  cleanup: 602,138,341,376 (about 561 GiB);
- frozen RGB release: 6,798,585,468 bytes;
- LiDAR release: 4,201,008,981 bytes;
- existing project run tree: 118,762,414,978 bytes;
- existing 3DGS 100K final PLY: 2,340,432,588 bytes.

Measured packet density projects 83--90 GiB for one 2,196-view method.  Keeping
all ten would require about 0.83 TiB and is prohibited.  The rolling lifecycle
caps packet scratch at 100 GiB.  Retained new models, exact external priors and
evidence are provisionally budgeted at 150 GiB; active training/merge scratch
at 100 GiB.  The expected additional peak is therefore about 350 GiB, within
the currently observed free space but subject to the following fail-closed
gates:

- at least 300 GiB free before starting a new training/prior attempt;
- at least 180 GiB free before starting any full 2,196-view packet export;
- stop before the next launch if either gate fails; never delete a final model,
  formal result, distance array or failure evidence to force continuation;
- intermediate iteration checkpoints, compiler caches and temporary merged
  files may be removed only after their designated final artifact and a full
  hash inventory exist; raw packets additionally require independent formal
  verification and full lightweight-archive byte verification;
- CityGaussianV2 writes an exclusive, fsynced hash inventory for every coarse
  and block checkpoint after the merged checkpoint is verified, then removes
  only those inventoried transient files; the merged checkpoint, resolved
  configuration, command logs and cleanup inventory remain retained;
- MetroGS likewise inventories and fsyncs the single-GPU rank checkpoint after
  the merged checkpoint and PLY conversion pass, deletes only that duplicated
  rank file, and retains the merged checkpoint, PLY, configuration, summary
  and inventory;
- external prior payloads used for training remain retained through the 100K
  audit.  Any later deletion requires a separate reviewed retention decision.

These are executable checks in `run_m3m_gcp_100k_guarded.py`, not operator
reminders.  The guard rejects a different plan/recipe/commit/tree, a dirty or
mis-hashed method source, missing external-weight bytes, a non-idle GPU, an
unmatched prepared per-method input evidence file, insufficient free space, a
second packet mutex, a changed file inside the frozen model-identity inventory,
an unbound packet-state or packet-set path, or packet growth beyond 100 GiB.
Packet release can delete only the method-specific packet root frozen in its
recipe and requires both independent formal verification and full
archive-inventory byte verification; the mutex persists on failure.
Every prior, training and packet-export child is followed by a method-specific
product postcondition before an exclusive phase-success marker can be written.
Packet success additionally requires an exact 2,196-name inventory, every NPZ
byte count/SHA and recomputation PASS, and an identical mapping CSV.  A child
that exits zero but fails any of these checks is recorded as immutable
`FAILED_UNRANKED` (prior/training) or `INCOMPLETE_UNRANKED` (packet export).

## 5. Result and retry policy

- `READY_FOR_EVALUATION` plus verifier PASS becomes `COMPLETE_RANKED` for this
  scene.  OOM, failed and incomplete rows remain unranked.
- No PSNR, GCP or LiDAR result is used to select routes, retries, partitions,
  checkpoints or hyperparameters.
- A fail-closed guard rejection before child creation is not an experiment
  attempt and may be relaunched only after the exact guard cause is corrected.
  Once a child process starts, any exit—including zero reported optimizer
  progress, technical failure or OOM—is final for that method.  Phase logs,
  failure evidence and success markers are exclusive-create and never reused.
- Formal output roots are absent at prior/training guard admission, are
  exclusively created empty for training, and are never resumed or overwritten.
- The other four prepared scenes remain untouched after 100K completion; their
  later authorization requires a new user instruction and review.

## 6. Review decision requested

The reviewer is asked to return one exact verdict:

- `PASS_100K_TIME_SPACE_EXECUTION_PLAN_V1`; or
- `REVISE_100K_TIME_SPACE_EXECUTION_PLAN_V1` with concrete blocking items.

This plan remains non-executable until the PASS verdict is recorded against
the exact clean commit/tree, execution-plan file/canonical SHA and ten-recipe
manifest file/canonical SHA, and the LiDAR contract is activated with the
reviewed rolling authorization schema.  The reviewed activation must contain
the exact verdict string; the previous protocol-review PASS is insufficient.
The prerequisite protocol/data review is independently frozen as
`PASS_LIDAR_V1_AND_SIX_SCENE_PREPARATION_V2` at commit
`e9c3414b808b374bd8632a45ee965e3f6acc1ac0`, tree
`d1d6c73852e42bc02c519d0853e26c114dcb1f8f`; it cannot substitute for the
second execution-plan review.
