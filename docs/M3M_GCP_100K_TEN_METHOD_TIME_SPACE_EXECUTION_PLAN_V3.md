# M3M-GCP 100K ten-method execution continuation v3

This document supersedes only the executable authorization issued by
`M3M_GCP_100K_TEN_METHOD_TIME_SPACE_EXECUTION_PLAN_V2.md`. It does not erase or
reclassify any child-started v2 outcome. The LiDAR protocol, RGB split, ten-method
pool, method budgets, frozen v2 recipes, renderer adapters, packet lifecycle,
capacity gates, and no-result-driven-retry rule are unchanged.

## Why a new activation is required

The v2 guard used a generic `.strip()` helper to read
`git status --porcelain=v1`. For a worktree-only modification, the first porcelain
XY column is a significant leading space. Removing it made the runtime status
different from the byte-exact frozen recipe even when the source was correct.

PGSR was rejected at this source-identity gate. No PGSR child, run root, evidence
directory, GPU allocation, or formal attempt was created. This is therefore a
pre-child infrastructure rejection, not a method result. Manual guard bypass and
recipe-status rewriting are forbidden.

The fix keeps the generic scalar Git reader for commits and trees, while a
dedicated porcelain reader removes trailing CR/LF only and preserves every status
column and embedded newline. Tests cover clean, unstaged, staged, and multiline
statuses, a real temporary Git repository, every phase binding in all ten frozen
recipes, and the actual 901 source roots.

## Immutable continuity

`docs/protocol_evidence/m3m_gcp_100k_activation_v2_to_v3_continuity.json`
byte-binds:

- activation v2 and its reviewed commit/tree;
- the unchanged v2 execution plan and recipe manifest;
- the final 2DGS `FAILED_UNRANKED` evidence and explanatory supplement;
- the PGSR pre-child guard console and the fact that its run/evidence roots were
  absent when sealed.

The receipt is rehashed by the activation builder, guarded runner, attempt-manifest
builder, and attempt freezer. Activation v3 inherits the 2DGS final outcome and
must reject any attempt to launch 2DGS again. PGSR retains its one formal attempt.

## v3 executable identity

- Plan: `configs/m3m_gcp_native_quarter_100k_ten_method_execution_plan_v3.json`
- Frozen recipes: `configs/m3m_gcp_native_quarter_100k_recipe_manifest_v2.json`
  (byte-identical to v2)
- Activation: `/root/autodl-tmp/runs/m3m-gcp-native-quarter/formal-100k-v2/activation_v3.json`
- Continued method namespace:
  `/root/autodl-tmp/runs/m3m-gcp-native-quarter/formal-100k-v2/gcp_100000_20260610`
- Final attempt manifest/freeze: `scene_attempts_v3.json` and
  `scene_attempt_freeze_v3.json` under the same continued namespace root.

Only a new clean commit/tree, complete Windows and Linux 100K/LiDAR regressions,
and a fresh reviewer verdict bound to that exact commit/tree may authorize
activation v3. The existing activation v2 is immutable and must never be edited or
used with the new checkout.
