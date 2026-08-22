# M3M-GCP 100K activation-v4 successor

This successor changes only the executable guard context used to revalidate an
already successful prior before training. It does not change the scene, image
split, method pool, method order, seed, budgets, source commits, prior commands,
training commands, renderer adapters, metrics, or no-retry policy.

## Trigger and classification

MetroGS prior generation completed successfully under activation-v3. The first
training guard invocation stopped before creating the training child or run
root because `validate_prior_phase_success()` rebuilt the prior command with the
active training `source_root`. MetroGS intentionally has different frozen
source roots for prior generation and training, so that reconstructed command
could not equal the immutable prior evidence.

This is a pre-child guard implementation error. It is not an algorithm failure,
does not consume the one formal MetroGS training attempt, and does not authorize
a new prior run.

## Exact fix

When a training phase revalidates a prior phase, the guard now reconstructs the
prior command only from the frozen prior bindings:

- `source_root` comes from `source_bindings.prior.root`;
- `dataset_root` and `prior_root` come from `phase_roots.prior`;
- the benchmark repository, authorized run root, and packet root remain the
  common frozen replacements;
- the prior source commit, tree, runtime status, and required files are
  revalidated before the prior command hash is accepted.

There is no alternative hash, compatibility fallback, manual bypass, or
result-dependent retry.

## Continuity

The receipt
`docs/protocol_evidence/m3m_gcp_100k_activation_v3_to_v4_continuity.json`
binds activation-v3, the reused 3DGS model, all eight terminal failed/OOM
attempts, MetroGS prior PASS and its principal products, and the MetroGS
pre-child guard rejection. Activation-v4 therefore:

- forbids relaunch of all eight terminal failed/OOM methods;
- forbids 3DGS retraining;
- forbids MetroGS prior regeneration;
- authorizes only the unfinished MetroGS training phase before the attempt
  manifest is frozen;
- continues in the existing `formal-100k-v2` run namespace without changing
  any inherited artifact bytes.

The active successor artifacts are:

- execution plan:
  `configs/m3m_gcp_native_quarter_100k_ten_method_execution_plan_v4.json`;
- unchanged recipe manifest:
  `configs/m3m_gcp_native_quarter_100k_recipe_manifest_v3.json`;
- activation:
  `/root/autodl-tmp/runs/m3m-gcp-native-quarter/formal-100k-v2/activation_v4.json`;
- attempt manifest and freeze:
  `scene_attempts_v4.json` and `scene_attempt_freeze_v4.json`.

Activation-v3 remains immutable continuity evidence and is not an executable
fallback after activation-v4 is issued.

