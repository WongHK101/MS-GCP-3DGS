# GS-GCP v1.3.0 And Stage 0 External Review Record

Review date: 2026-07-18

Record type: CODEX transcription and local reconciliation of the user-forwarded
external GPT review.

## Reviewed Evidence

- `GPT_GCP_POINTSET_RELEASE_V1_3_0_CONTROL_HEAVY_REVIEW_20260717.zip`
  - SHA-256: `f87cbd043cc568155c66e641a623ab1711070a82ec42b2933b2b5381746f2a11`
  - External audit: 80/80 registered files verified.
- `GPT_GS_GCP_STAGE0_901_EXECUTION_READINESS_REVIEW_20260718.zip`
  - SHA-256: `bd8eb6f66b86a2563185de36a65fecc3c110a0a5711ae21e1d0b17bdbfe90cfb`
  - External audit: 64/64 registered files verified.
- Repository source at commit
  `9b2d26367320c9f9776b0f4b8557b8898a94bb91`, tree
  `325d69526b33e97c938c0d86a57b594c9e7a5ced`.
- Release root digest
  `513f8999fe4b110f15bcbecad7932895781cee755ee9ccd7a14ff10298546d75`.

## External Disposition

- `v1.3.0 release: PASS`
- `Stage 0 901 execution readiness: PASS`
- `original 3DGS 3K qualification smoke: APPROVE`

The approval authorizes entry into the frozen original-3DGS 3K qualification
pipeline. It does not assert that training, packet export, Sim(3), or formal
metrics have already run, and it does not authorize the other five scenes or
any other method.

## CODEX Local Reconciliation

CODEX independently recomputed the following from the formal release after the
external response was returned:

- 1,383 canonical rows and 1,383 unique observation IDs;
- 1,155 annotation-good rows and 1,069 formal rows;
- 593 control and 476 checkpoint formal observations;
- 87 scene-point assignments, representing 50 unique point IDs;
- 951 image-level mappings and 6,187 frozen training views;
- zero formal out-of-bounds rows and two preserved non-formal clicked
  out-of-bounds diagnostics;
- 17 point IDs have different roles in different independently reconstructed
  scenes; no point has both roles within one scene;
- control/checkpoint minimum good-view counts are 9/8;
- authoritative RTK file hashes match the frozen release.

The external reviewer correctly identified one transcription error in the
handoff. The authoritative isolated GNU time binary SHA-256 is:

`7310b9b4c51a8f4d26c1af0da250f03a49ec8a8141033123e79196ad18f6c81b`

The reviewer also identified documentation/provenance clarifications that do
not alter release data or experimental semantics:

- report 87 as scene-point assignments, not 87 independent physical GCPs;
- make the v1.3.0 control-heavy split override the older generic 20-30%
  control guideline;
- describe method-specific pre-registered recipes and transparently reported
  resource budgets rather than claiming identical iteration budgets;
- distinguish legacy hash-verified `ms-gcp-v13` read-only input asset paths on
  AutoDL-901 from new `gs-gcp-v13` build/run output namespaces;
- retain the distinction between captured readiness evidence and live runtime
  GPU availability.

## Scope Boundary

This record changes the external review gate only. It does not modify the
v1.3.0 payload, split, annotations, survey coordinates, camera track, formal
depth semantics, method recipe hyperparameters, or evaluator. A fresh live
preflight remains mandatory immediately before any GPU child is launched.
