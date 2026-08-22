# M3M-GCP 100K qualification plan (flex v1)

Status: review candidate; do not launch GPU training before the batch review.

## Purpose

This batch repairs the 100K execution routes without rewriting the earlier
diagnostic attempts. It is a qualification batch, not a ranked formal release.
The valid frozen 3DGS 100K model is reused; the other methods receive one clean,
traceable attempt after the known infrastructure and lifecycle defects are fixed.
If the post-batch review confirms that the scientific contract was unchanged, a
complete saved model and its metrics may be promoted in place; promotion must not
require retraining or choose between attempts by accuracy.

## What is fixed, and what may move

The scientific comparison contract is fixed:

- native-quarter image bytes, all-image COLMAP camera domain, frozen train/test
  split, scene identity, seed 0 and each method's declared optimization budget;
- source commit/tree plus declared compatibility patches;
- no GCP or LiDAR truth in training or priors;
- common RGB, GCP and LiDAR evaluators and no result-driven hyperparameter tuning;
- no resolution, densification, loss-weight or method-semantic changes to rescue a
  failure.

Engineering details may be corrected during qualification with a new attempt ID:

- absolute paths, interpreter and `PYTHONPATH` bindings;
- manifest location, environment variables, logging and resource telemetry;
- save-before-evaluate ordering and separation of training from offline evaluation;
- single-GPU scheduling and reuse of byte-identical completed checkpoints;
- a diagnostic rerun for an ambiguous host kill when the first attempt lacks useful
  telemetry.

An engineering correction is logged in the attempt receipt but does not trigger a
per-method audit. A method-supported large-scene option that could affect the model
may be tested only as a clearly labelled qualification variant; it cannot enter the
ranking until the post-batch review. Any change to data, split, camera domain,
truth access, evaluator, metric, optimization budget, losses or algorithm semantics
is a scientific red line and stops before execution.

Every launch still performs the necessary live scientific input check: it binds the
reviewed `per-method-inputs-v2.json`, verifies the exact 2,196 training image names,
sizes and hashes, checks the method-specific sparse-model hashes and input profile,
and validates the GSPrior normalized-root lineage. These are data-fairness checks,
not separate human approval stages.

## Corrected routes

| Method | Qualification action |
| --- | --- |
| 3DGS | Reuse the validated 30K 100K-scene model; no retraining. |
| 2DGS | Train and save at 30K; defer its in-training test to iteration 30001, then evaluate offline. |
| PGSR | Restore the same minimal PyTorch3D transform compatibility path used by the successful 3K route. Keep its declared test/save schedule. |
| RaDe-GS | Train and save at 30K; defer its in-training test to iteration 30001, then evaluate offline. Keep the 15K checkpoint. |
| QGS | Use the unchanged formal configuration with the common expandable allocator. If it still OOMs on the assigned 96 GB GPU, record OOM; do not tune densification or resolution. |
| GSPrior | Restore the 3K compatibility and benchmark helper paths explicitly. Keep its declared 20K/30K/40K schedule. |
| SoF | Run one clean attempt with telemetry. If host-memory initialization still fails and no author-supported lazy/large-scene route exists, record host OOM. |
| CityGaussianV2 | Reuse only the predeclared coarse 30K and block 0 fine 60K checkpoints by same-filesystem hardlinks into a fresh root. Reject any unlisted completed block, train blocks 1–15 sequentially with the official per-block command, merge, verify, then remove only hash-inventoried transient links/checkpoints. |
| CityGS-X | Defer in-training evaluation to 100001 so the 100K model is saved before the common offline evaluator runs. |
| MetroGS | Read the authoritative formal manifest from the formal-input root rather than requiring a duplicate inside the derived prior root. Keep the 150K effective-image budget. |

All new attempts inherit `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` as an
algorithm-neutral allocator setting. This does not authorize method-specific model
changes.

The CityGaussianV2 reuse binding fixes coarse 30K to 3,353,977,999 bytes / SHA-256
`fbe8776f...36f13`, block 0 at 60K to 2,526,153,999 bytes / SHA-256
`4646b0a9...1d19`, and the 75,381-byte partition to SHA-256
`86d405ec...3a80`. No block other than block 0 may be imported from the diagnostic
root. Resource telemetry uses the frozen bundled GNU time binary rather than an
assumed system `/usr/bin/time`.

## Attempt and failure policy

New outputs use:

`/root/autodl-tmp/runs/m3m-gcp-native-quarter/qualification-100k-v1/gcp_100000_20260610/<method>/attempt-<UTC_ID>`

Old `formal-100k-v2`/later diagnostic outputs remain immutable evidence and are
labelled superseded for execution, not deleted and not ranked.

- Admission/path/import/manifest/scheduler/logging failure before useful training:
  correct it and use a new attempt ID.
- Deterministic GPU or host OOM with complete telemetry: terminal for the current
  hardware/configuration; do not repeat blindly.
- Ambiguous SIGKILL without telemetry: allow one diagnostic repeat with telemetry.
- Model saved but evaluation failed: keep the model and repair/retry only the
  offline adapter/evaluator; do not retrain.
- Result selection across retries is forbidden. A newer attempt supersedes an older
  one only because a documented execution defect was corrected, never because its
  metric is better.

## Execution order and storage control

After one batch approval, perform a GPU preflight and execute methods serially on
the single assigned GPU. Reuse 3DGS first; then run the short/common routes (2DGS,
PGSR, RaDe-GS, GSPrior), the prior-backed routes (CityGS-X, MetroGS), the clean
failure checks (QGS, SoF), and CityGaussianV2 last because its remaining 15 blocks
are expected to dominate wall time (about 75 GPU-hours if the observed ~5 hours per
block persists).

After every saved-model validation, run the common packet/RGB/GCP evaluators as a
separate phase. LiDAR evaluation may be added later to the same saved model. Keep
large models on 901 and pull only receipts, logs, configs and metrics to the local
archive. Delete transient checkpoints only after the final/merged model is hashed
and reload-validated. Do not delete old diagnostic evidence during qualification.

## Minimal audit schedule

1. One batch review now: this plan, generated recipes, wrapper changes and CPU tests.
2. One batch outcome review after all practical 100K attempts finish, including
   honest OOM/unsupported rows.
3. One final protocol/ranking freeze before expanding the same routes to all six
   scenes.

There is no per-method approval gate unless a scientific red line is proposed.
Operational fixes are accumulated in the change log and reviewed in the next batch.
