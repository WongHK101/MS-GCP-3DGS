# GPU Bundle-Adjustment Feasibility Protocol

## Objective

Measure whether a modern COLMAP/Ceres build with CUDA and cuDSS materially
reduces incremental-mapper time on `gcp_10000_20260610` without reducing
registration completeness or reconstruction quality.

## Isolation

- First-paper UMGS code, environments, datasets, and runs are not modified.
- Source, toolchain, code, data, and runs use distinct second-paper roots.
- All operational paths are configurable through environment variables.

## Fixed Inputs

- Scene: `gcp_10000_20260610`
- RGB images: 976
- Existing spatial-matching database is copied from the completed CPU-BA run.
- Mapper settings remain unchanged except for the explicitly audited GPU-BA
  solver flags.

## Comparison

Baseline:

- COLMAP 3.9.1
- CPU Ceres bundle adjustment
- Mapper time: recorded from the completed run

Candidate:

- COLMAP 4.0.4
- pinned Ceres development commit with CUDA support
- cuDSS 0.8.0.10 for CUDA 12
- `Mapper.ba_use_gpu=1`

## Acceptance Gates

1. Logs must confirm that Ceres uses GPU solvers rather than silently falling
   back to CPU.
2. Registration must remain 976/976.
3. The number of reconstructed points and reprojection statistics must remain
   comparable; any material degradation blocks protocol adoption.
4. WGS84/ENU alignment is evaluated after mapping using the same custom Sim(3)
   procedure.
5. Runtime and GPU traces must be retained.

The test is a feasibility comparison. It does not automatically replace the
current benchmark protocol.

## Completed Result

The test completed on 2026-06-15. The candidate registered 976/976 images and
reduced mapper time from 421.676 to 357.700 minutes while preserving sparse
model and WGS84-derived ENU alignment quality. Because the comparison changes
both COLMAP version and solver backend, the toolchain remains a feasibility
candidate rather than an automatic protocol replacement. See
`docs/GPU_BA_FEASIBILITY_RESULT_20260615.md`.
