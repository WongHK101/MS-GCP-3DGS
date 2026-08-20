# M3M-GCP 100K ten-method time/space execution plan v1

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
| 4 | SoF | new seed-0 run | 30K iterations, frozen unbounded/TnT route | 3.5--8 h |
| 5 | GSPrior | new seed-0 run | 40K iterations and frozen internal TSDF schedule | 7.5--18 h |
| 6 | QGS | new seed-0 run or record OOM | 30K iterations | 12.5--28 h or earlier OOM |
| 7 | CityGaussianV2 | new seed-0 run | official MatrixCity-aerial 4 x 4 partitions; 30K coarse + 60K fine per official block route, then official merge | 39--76 h |
| 8 | CityGS-X | new seed-0 run | frozen DAv2 prior and official 100K all-block 100K-iteration route | 15--40 h |
| 9 | MetroGS | new seed-0 run | frozen Pi3-Align + MoGe-2 route; 150K effective image iterations / 37.5K optimizer steps | 21--52 h |

The ranges are capacity estimates, not budgets or early-stop thresholds.  They
use the measured 3K training times and the existing 3DGS 100K export.  The
existing 3DGS exporter used 740.56 s for 211 views (3.51 s/view), projecting
about 2.14 h for 2,196 views.  Allowing renderer variation gives 2--3.5 h of
packet export per successful method.  The entire successful queue is therefore
expected to take roughly 5--10 wall-clock days on one GPU; 6--9 days is the
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
4. A method-specific CUDA OOM, host OOM or technical failure is closed as
   `OOM_UNRANKED` or `FAILED_UNRANKED`; its immutable failure evidence is
   retained and the queue continues.  Batch size, resolution, partition rule,
   loss, checkpoint or iteration budget is not changed to rescue it.
5. After all ten attempts, freeze one exact ordered ten-row attempt manifest.
   `READY_FOR_EVALUATION` rows bind model/recipe/renderer bytes.  Failed/OOM
   rows have null model fields and bind failure evidence.  One failed method
   therefore cannot invalidate successful methods.

Before Phase A formal launches, each 100K scene recipe must be materialized as
a hash-bound file and mechanically checked against its qualified 3K parent:
only scene identity, train-camera-derived lists/partitions and paths may differ.
The CityGaussianV2 4 x 4 choice is the frozen upstream MatrixCity-aerial
configuration, selected before any 100K result; it is not a result-driven
memory workaround.

### Phase B: rolling packet evaluation

For each `READY_FOR_EVALUATION` method, and never for a failed method:

1. export exactly all 2,196 train-view metric-depth packets into that method's
   own run root;
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

- free persistent bytes: 604,395,982,848 (about 563 GiB);
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
- intermediate iteration checkpoints, compiler caches, temporary merged files
  and raw packets may be removed only after their designated final artifact,
  full hash inventory and independent verification exist;
- external prior payloads used for training remain retained through the 100K
  audit.  Any later deletion requires a separate reviewed retention decision.

## 5. Result and retry policy

- `READY_FOR_EVALUATION` plus verifier PASS becomes `COMPLETE_RANKED` for this
  scene.  OOM, failed and incomplete rows remain unranked.
- No PSNR, GCP or LiDAR result is used to select routes, retries, partitions,
  checkpoints or hyperparameters.
- A failure before the training child starts or before any optimizer progress
  may receive only a byte-identical infrastructure retry under an already
  frozen compatibility rule.  A progressed/OOM run is final for that method.
- Formal output roots are fresh and never resumed or overwritten.
- The other four prepared scenes remain untouched after 100K completion; their
  later authorization requires a new user instruction and review.

## 6. Review decision requested

The reviewer is asked to return one exact verdict:

- `PASS_100K_TIME_SPACE_EXECUTION_PLAN_V1`; or
- `REVISE_100K_TIME_SPACE_EXECUTION_PLAN_V1` with concrete blocking items.

This plan remains non-executable until the PASS verdict is recorded against
the exact clean commit/tree and the LiDAR contract is activated with the
reviewed rolling authorization schema.

