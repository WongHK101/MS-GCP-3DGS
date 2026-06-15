# COLMAP GPU Bundle-Adjustment Feasibility Result

## Decision

The isolated COLMAP 4.0.4 GPU-BA toolchain completed the
`gcp_10000_20260610` mapper test successfully and preserved reconstruction
quality. It reduced mapper time by about 15%, but it does not automatically
replace the established COLMAP 3.9.1 protocol.

The comparison changes both the COLMAP version and the bundle-adjustment
backend. The measured runtime difference must not be attributed exclusively
to GPU bundle adjustment.

## Fixed Input

- Scene: `gcp_10000_20260610`
- RGB frames: 976
- Matching database: copy of the completed spatial-matching database
- Candidate mapper settings: established mapper settings plus
  `Mapper.ba_use_gpu=1` and `Mapper.ba_gpu_index=0`
- First-paper UMGS code, environments, data, and runs: unchanged

## Result

| Measure | COLMAP 3.9.1 baseline | COLMAP 4.0.4 GPU-BA | Difference |
|---|---:|---:|---:|
| Mapper time | 421.676 min | 357.700 min | -63.976 min |
| Runtime speedup | 1.000x | 1.179x | 15.2% reduction |
| Registered images | 976 | 976 | 0 |
| Sparse points | 505,382 | 505,422 | +40 |
| Observations | 6,640,249 | 6,640,973 | +724 |
| Mean track length | 13.139069 | 13.139462 | +0.000393 |
| Mean reprojection error | 1.228308 px | 1.230787 px | +0.002478 px |
| WGS84-derived ENU mean alignment error | 0.212385 m | 0.212384 m | -0.000001 m |
| WGS84-derived ENU median alignment error | 0.202549 m | 0.202651 m | +0.000102 m |

The candidate output contains a complete 976-image reconstruction in
`sparse/1` and a redundant 17-image candidate in `sparse/0`. Downstream code
must select the largest successful model rather than assume that model `0` is
the final reconstruction.

## GPU Evidence

- GPU trace samples: 4,084 at approximately five-second intervals
- Peak GPU memory: 1,287 MiB
- Peak sampled utilization: 100%
- Nonzero utilization samples: 16
- Peak power: 123.4 W

The GPU solver path was initialized and executed, but its duty cycle was low
relative to the CPU-heavy mapper workflow. The speedup varied substantially
between global-BA stages. This is consistent with a mixed CPU/GPU pipeline,
not full-GPU reconstruction.

The mapper emitted six nonfatal dense-Cholesky step failures and continued to
a complete model. The final registration, sparse statistics, and alignment
checks passed.

## Protocol Recommendation

1. Retain this build as an isolated second-paper toolchain.
2. Do not mix COLMAP 3.9.1 and 4.0.4 results inside one claimed controlled
   benchmark without recording the protocol version per scene.
3. Do not retroactively change first-paper results.
4. Before making GPU BA the formal second-paper default, either:
   - run a same-version CPU/GPU solver control, or
   - adopt COLMAP 4.0.4 for all formal second-paper reconstructions and rerun
     the scenes used in cross-scene quantitative comparisons.
5. The interrupted `gcp_50000_20260610` run should remain paused until this
   protocol choice is approved.

## Evidence

Local lightweight evidence:

`outputs/gpu_ba_feasibility_20260615/evidence_bundle`

Archive:

`outputs/gpu_ba_feasibility_20260615/gcp10000_gpu_ba_feasibility_evidence_20260615.tar.gz`

Archive SHA256:

`6EF7F539A63029FF5992B739D43935146E947B6781EDE99E2F6243A1E1ED62F9`
