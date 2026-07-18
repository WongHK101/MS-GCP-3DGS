# GS-GCP Original 3DGS 3K Qualification External Review Record

Review date: 2026-07-18

Record type: CODEX transcription and local reconciliation of the
user-forwarded external GPT review.

## Reviewed Evidence

- `GPT_GS_GCP_ORIGINAL_3DGS_3K_QUALIFICATION_REVIEW_20260718.zip`
  - SHA-256: `1349e1ffde057b4b450e82b6c5b1bd152692cc8288875ef18e1ea6bb0f178bd3`
  - Registered payloads independently verified by the external reviewer:
    193/193 size and SHA-256 checks passed.
- User-forwarded external review text
  - Local attachment SHA-256:
    `2a20663aba1e5f025efcfb617a41093e15f9724290ad2514af44df13e20bedfc`
- Reviewed Git bundle heads:
  - benchmark: `6e4b88786ab0dc53ae574f0af6b5603f45868ed4`
  - metric adapter: `69842bcbcf1d3a159d08256a8cac557261234d36`
  - metric rasterizer: `c7c8ec385986ea5230dcdd517b8f6cc06db0049d`
- Release root digest:
  `513f8999fe4b110f15bcbecad7932895781cee755ee9ccd7a14ff10298546d75`.

## External Disposition

- `original 3DGS 3K qualification: PASS`
- `original 3DGS remaining-five-scene pipeline: APPROVED`
- `other methods: NOT REVIEWED / NOT APPROVED`

The approval permits the frozen original-3DGS method identity to run 5K, 10K,
20K, 50K, and 100K. It does not claim that those scenes have results or have
passed review.

## Frozen Identity And Scope

The approval remains valid only for:

- official training source commit
  `2eee0e26d2d5fd00ec462df47752223952f6bf4e`;
- official 30K recipe, seed 0;
- `graphdeco_rminus1_1600_width_cap_v1`;
- formal iteration-30000 point-cloud model;
- metric adapter commit `69842bcbcf1d3a159d08256a8cac557261234d36`;
- metric rasterizer commit `c7c8ec385986ea5230dcdd517b8f6cc06db0049d`;
- formal tensor `alpha_normalized_expected_camera_z = M1/A` with `camera_z`
  semantics;
- v1.3.0 release root
  `513f8999fe4b110f15bcbecad7932895781cee755ee9ccd7a14ff10298546d75`;
- frozen split, patch, aggregation, failure gates, and Sim(3) protocol.

Only scene-bound inputs may vary. No GCP residual may select a seed, training
iteration, learning rate, densification setting, opacity-reset setting,
checkpoint, depth formula, patch, aggregation rule, failure gate, or Sim(3)
implementation.

## Non-Blocking Clarifications Preserved

- The formal model is the iteration-30000 point-cloud PLY, not an optimizer
  state checkpoint.
- Training GPU time and packet-export GPU time must be reported separately.
- The Linux legacy fixture invocation was not portable because it retained
  Windows absolute paths. Current Linux runtime wrapper validation passed; the
  matching Windows compatibility fixture matrix passed 50/50.
- The Windows 50/50 fixture renderer commit is generic test-fixture evidence,
  not the current metric rasterizer runtime identity.
- Training preflight used benchmark commit `6d1e1dd927bf4744825ca6ef29c8df388fc75e6a`;
  packet-camera compatibility and formal evaluation used
  `6e4b88786ab0dc53ae574f0af6b5603f45868ed4`.
- The review package did not embed the 849 MB formal PLY or complete packet
  arrays. It bound them through size, SHA, schema, recomputation, and pre/post
  packet identities.
- Multiview scatter must remain reported alongside aggregate checkpoint error.

## CODEX Local Reconciliation

CODEX accepts the disposition as consistent with the submitted package and
local evidence. The method registry may therefore record:

```text
three_k_qualification_status = PASS
full_scene_matrix_eligible = true
external_review_status = PASS
```

This record does not authorize any other method and does not alter the release,
training recipe, packet values, evaluator math, split, or formal protocol.
