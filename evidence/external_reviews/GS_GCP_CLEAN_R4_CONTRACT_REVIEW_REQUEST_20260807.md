# GS-GCP clean-R4 contract review request (2026-08-07)

## Requested decision

Please return one of:

- `CLEAN_R4_CONTRACT_PASS`: the exact contract and 3K materialized input may enter formal from-scratch 3K qualification;
- `BLOCKER`: list the exact blocking invariant and required correction.

This request does **not** ask for full-matrix authorization. The full matrix remains locked until the new 3K end-to-end qualification passes and its evidence is accepted.

## Scope reset

The old 1600-width, path-backed and serializer-modified route is historical negative/diagnostic evidence only. Its recipes, checkpoints, results and qualification decisions cannot be inherited. The clean route uses the unmodified official 3DGS training source.

## Exact review targets

| Artifact | SHA-256 |
|---|---|
| `configs/gs_gcp_r4_input_materialization_v1.json` | `1816299d4a2aabcba64103cd095e6704be76d11590ed395512b80518136389c4` |
| `code/gcp/materialize_gs_gcp_r4_inputs.py` | `53c7c80adcafbf6046364cbb9b08aa728cf1d04cba257e201104ab39b1299825` |
| `configs/gs_gcp_v13_original_3dgs_recipe_v3.json` | `b6a4defb5a5b4eb9c2795208a237536d1fa3595caedd2c5b1f89c461e4aed23e` |
| `configs/gs_gcp_method_registry_v1.json` | `c169c7e5e19368dbfa028e26a5880092b94097723c4c2c6ee9a68dec2720e016` |
| `configs/gs_gcp_v13_original_3dgs_full_matrix_v2.json` | `6044241a6d0b20da44b288bc777df0a779e5cd00148d0df86769368bee800b74` |
| `scripts/gcp_v13/run_original_3dgs_3k_30k.sh` | `88b3ab3cf2964f619b579030ca294c0ea6a6365da67b8995f2b656ae3c35bdee` |

Official source identity:

- commit `2eee0e26d2d5fd00ec462df47752223952f6bf4e`
- tree `5eee127dfc0942bf83d9fdd72328e03ec0cbf6c4`
- local official worktree clean at validation time
- runtime serializer patch forbidden

## Input semantics

- Primary rule: per-axis Python `round(decoded_size / 4)`, ties-to-even.
- Pillow 11.1.0, `Image.resize(size)` with default BICUBIC.
- Output: lossless RGB PNG with decoded uint8 identity recorded per image.
- Camera: PINHOLE dimensions and `fx/fy/cx/cy` scaled by actual per-axis R4 ratios; qvec/tvec preserved.
- Split: frozen 82 train / 12 test for 3K; roots are physically separate.
- Both COLMAP subsets contain zero POINTS2D tracks and no `points3D.bin`.
- Both roots receive the exact frozen `points3D.ply`; training receives no test RGB or test camera record.
- Official 3DGS binds only `<scene>/train`, uses `--resolution 1`, seed 0, 30K, no resume and no runtime source patch.
- Equivalence reference is frozen full-resolution JPEG loaded by official 3DGS with `--resolution 4`.

## Materialized 3K evidence

Local independent root:

`E:\datasets\M3M-GCP\gs_gcp_v13_clean_r4_inputs_20260807\gcp_3000_20260602`

- `R4_INPUT_MANIFEST.json` canonical SHA: `88e354a7cc387975f6686020cf15a3584bfe28769c46360400dcfc027d82921c`
- 101 files, 250,576,506 file bytes
- 82 train images, 12 test images
- full verification: PASS, zero errors
- per-source JPEG SHA/dimensions checked
- every output PNG decoded and compared pixel-for-pixel with the official R4 resize
- camera FoV and normalized principal-ray errors bounded by `1e-12`
- source frozen mirror remained read-only

## Gate state

- Method registry: 3DGS admitted for 3K only; `three_k_qualification_status = NOT_RUN_CLEAN_R4`.
- `full_scene_matrix_eligible = false`.
- Full-matrix plan validator passes only as a locked plan and reports `launch_authorized = false`.
- Legacy full-matrix launcher is hard-disabled.
- GPU UUID is deliberately not inherited; a fresh per-run hardware manifest is required.

## Tests

All 47 focused R4/protocol tests passed:

- materializer: 4/4
- recipe validator: 10/10
- method registry: 7/7
- locked full-matrix plan: 10/10
- Stage 0.5 resolution/split: 7/7
- Stage 0 readiness: 9/9

The broad `pytest code/gcp -q` invocation aborted in an unrelated pre-existing pandas/NumPy correlation test (`test_audit_annotation_gps_pose_association.py`) inside native MKL/NumPy code. It did not report a failure in any clean-R4 target. This environment-level abort is disclosed and is not counted as a passing full suite.
