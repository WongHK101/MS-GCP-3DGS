# M3M-GCP 100K ten-method execution continuation v3

This document supersedes only the executable authorization issued by
`M3M_GCP_100K_TEN_METHOD_TIME_SPACE_EXECUTION_PLAN_V2.md`. It does not erase or
reclassify any child-started v2 outcome. The LiDAR protocol, RGB split, ten-method
pool, method budgets, renderer behavior, packet lifecycle, capacity gates, and
no-result-driven-retry rule are unchanged. Nine v2 recipe rows remain byte
identical. The 3DGS packet recipe has one Linux source-identity metadata
correction described below; it does not change source bytes or rendering.

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

The first complete 23-phase preflight then exposed a separate, isolated identity
issue in the reused 3DGS packet source: the v2 recipe contained the SHA-256 of a
Windows checkout of `rasterize_points.h`, while the already-qualified formal Linux
source contains the LF byte identity recorded by the frozen-patch Linux proof.
No source file, command, budget, input, renderer behavior, child process, or
attempt changed. Manifest v3 replaces only the 3DGS recipe row, binds all eight
Linux patched-file hashes, and permits exactly one formal hash per file. The old
v2 manifest and recipe remain immutable audit evidence.

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
must reject any attempt to launch 2DGS again. PGSR remains eligible for its one
formal attempt because its earlier rejection occurred before child start.

`docs/protocol_evidence/m3m_gcp_100k_3dgs_linux_source_binding_correction_v1.json`
independently binds the failing 23-phase audit, the formal Linux identity proof,
both frozen patches, both old v2 identities, and the new v3 identities. Activation,
guard, attempt-manifest, and attempt-freeze entry points revalidate this receipt;
activation and every guarded launch additionally require the actual 901 source
roots to pass all 23 phase bindings.

## v3 executable identity

- Plan: `configs/m3m_gcp_native_quarter_100k_ten_method_execution_plan_v3.json`
- Frozen recipes: `configs/m3m_gcp_native_quarter_100k_recipe_manifest_v3.json`
  (nine byte-identical v2 rows plus the single 3DGS Linux identity correction)
- Activation: `/root/autodl-tmp/runs/m3m-gcp-native-quarter/formal-100k-v2/activation_v3.json`
- Continued method namespace:
  `/root/autodl-tmp/runs/m3m-gcp-native-quarter/formal-100k-v2/gcp_100000_20260610`
- Final attempt manifest/freeze: `scene_attempts_v3.json` and
  `scene_attempt_freeze_v3.json` under the same continued namespace root.

Only a new clean commit/tree, complete Windows and Linux 100K/LiDAR regressions,
and a fresh reviewer verdict bound to that exact commit/tree may authorize
activation v3. The existing activation v2 is immutable and must never be edited or
used with the new checkout.
