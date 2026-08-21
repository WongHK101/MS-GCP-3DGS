# M3M-GCP 100K ten-method execution plan v2

Status: review candidate; not execution-authorized until the exact clean commit and tree receive the existing formal verdict token `PASS_100K_TIME_SPACE_EXECUTION_PLAN_V1` from task `019ff12c-cb29-7cb2-8fb6-1d82c5f8c54b`.

This document supersedes the **execution instance** described by `M3M_GCP_100K_TEN_METHOD_TIME_SPACE_EXECUTION_PLAN_V1.md`. It does not change the LiDAR metric protocol, RGB split, method pool, method budgets, renderer adapters, packet rules, or the no-result-driven-retry rule. The verdict token remains v1 because it is part of the already reviewed LiDAR artifact-schema interface; commit, tree, plan SHA, recipe-manifest SHA, activation path, and run namespaces distinguish this new instance.

## Why v1 cannot continue

The first 2DGS child under activation v1 inherited `RLIMIT_NOFILE soft=1024` while loading 2,196 training images. It exited before optimizer progress and before GPU allocation with `OSError: [Errno 24] Too many open files`. Reviewer ruling classifies this as a deterministic infrastructure-preflight omission, not an algorithm failure.

The old activation, run root, logs, environment evidence, and failure record are immutable audit artifacts. Their exact local and remote hashes are sealed in `docs/protocol_evidence/m3m_gcp_100k_activation_v1_infrastructure_supersession.json`. They are non-rankable and do not consume the single formal algorithm attempt. No later table, attempt freeze, result verifier, ranker, or archive may treat the old `FAILED_UNRANKED` record as a method outcome.

## Authoritative v2 artifacts and namespaces

- Plan: `configs/m3m_gcp_native_quarter_100k_ten_method_execution_plan_v2.json`
- Recipe manifest: `configs/m3m_gcp_native_quarter_100k_recipe_manifest_v2.json`
- Recipes: `configs/m3m_gcp_native_quarter_100k_recipes_v2/*.json`
- Activation: `/root/autodl-tmp/runs/m3m-gcp-native-quarter/formal-100k-v2/activation_v2.json`
- New method runs and evidence: `/root/autodl-tmp/runs/m3m-gcp-native-quarter/formal-100k-v2/gcp_100000_20260610/...`
- Attempt manifest, model identities, scene freeze, and packet mutex/scratch are all under `formal-100k-v2` and use v2 filenames.

The previously frozen 3DGS model remains the sole permitted cross-namespace reuse. Its exact 2.340 GB PLY identity is unchanged and retraining remains forbidden.

## Mandatory file-descriptor contract

For every `prior`, `training`, and `packet` child:

1. The guard reads the parent `RLIMIT_NOFILE` before creating a run root, evidence directory, log, packet state, or child.
2. The hard limit must be at least 65,536. A lower hard limit is a guard rejection and creates no attempt artifacts.
3. The guard sets the parent soft limit to exactly 65,536 and re-reads it.
4. Immediately after child creation, the guard reads `/proc/<pid>/limits` and requires the child soft limit to be exactly 65,536 and the child hard limit to be at least 65,536.
5. `environment.json` records parent before, parent after, and actual child values. A successful phase marker binds the environment path and SHA; a failure record already binds them.

The resource requirement is frozen identically in every method recipe and in the execution-plan closure. It is not a method-specific accommodation.

## Unchanged attempt semantics

- A guard rejection before child creation is not an attempt and may be relaunched only after the exact guard cause is corrected.
- Once a v2 child starts, every exit—including zero progress and OOM—is final for that method.
- No metric, preview, GCP, LiDAR, held-out RGB, or intermediate result may drive a retry, budget change, partition change, resolution change, or checkpoint selection.
- The other four prepared scenes remain locked.

## Required prelaunch proof

Before generating activation v2, the exact candidate must pass local and Linux tests covering:

- hard-limit-below-65,536 rejection before any child or artifact;
- exact parent soft-limit change and unchanged sufficient hard limit;
- actual child inheritance through `/proc`;
- environment evidence identity and successful-marker binding;
- byte revalidation of every old v1 audit artifact;
- absence of v1 run, evidence, packet, attempt-freeze, and model-identity paths from all v2 recipes;
- all existing input, camera-root, phase-product, packet, failure, and ranking invariants.

Only after the reviewer binds the exact clean commit/tree may activation v2 be generated outside the checkout. The queue then restarts at 2DGS seed0; PGSR and all later methods remain unstarted until that point.
