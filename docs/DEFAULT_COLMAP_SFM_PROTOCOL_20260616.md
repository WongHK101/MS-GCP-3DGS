# Default COLMAP/SfM Protocol for GS-GCP

This document fixes the default large-scene SfM protocol after the June 2026
GCP scene diagnostics. It is intended for the six M3M-GCP campus scenes and
future second-paper runs unless a later protocol change is explicitly approved.

## Scope

- Input: RGB-only DJI `*_D.JPG` images.
- Output: COLMAP sparse reconstruction plus a diagnostic Sim3 alignment to
  EXIF/RTK-derived WGS84 local ENU camera priors.
- Data boundary: raw datasets are read-only; all databases, sparse models, logs,
  and summaries are written under a dedicated run root.

## Fixed Protocol

- COLMAP executable: `COLMAP 4.0.4` with CUDA/cuDSS and GPU bundle adjustment.
- Camera model: `SIMPLE_RADIAL`.
- Feature extraction: SIFT GPU enabled.
- Matching: COLMAP `spatial_matcher`.
- Spatial matcher parameters:
  - `SpatialMatching.max_num_neighbors=80`
  - `SpatialMatching.max_distance=500`
- Feature matcher parameter:
  - `FeatureMatching.max_num_matches=32768`
- Mapper: COLMAP 4 `global_mapper`.
- Global mapper GPU options:
  - `GlobalMapper.gp_use_gpu=1`
  - `GlobalMapper.gp_gpu_index=0`
  - `GlobalMapper.ba_ceres_use_gpu=1`
  - `GlobalMapper.ba_ceres_gpu_index=0`

The default run root is:

```text
/root/autodl-tmp/runs/gs-gcp/colmap-4.0.4-global-formal-20260616
```

The default RGB data root is:

```text
/root/autodl-tmp/datasets/M3M-GCP/scenes_rgb_20260615
```

## Explicit Non-Defaults

The following variants are diagnostics only and are not part of the default
formal protocol:

- FOV-aware pair-list matching.
- Incremental `mapper` for the large campus scenes.
- COLMAP 3.9.1 runs.
- Mixed-protocol claims inside one controlled result table.

## Entry Points

Run one scene:

```bash
bash /root/autodl-tmp/GS-GCP/scripts/run_large_scene_global_colmap_scene.sh gcp_10000_20260610
```

Run an explicit queue:

```bash
bash /root/autodl-tmp/GS-GCP/scripts/run_formal_colmap_gpu_ba_queue.sh \
  gcp_3000_20260602 gcp_5000_20260602 gcp_20000_20260602
```

The queue runs scenes sequentially. Do not run multiple large COLMAP jobs on the
same GPU unless this is intentionally scheduled and documented.

## Current Formal Evidence

- `gcp_50000_20260610`: 2209/2209 registered with the global protocol.
- `gcp_100000_20260610`: 2510/2510 registered with the global protocol.
- `gcp_10000_20260610`: formal rerun started after this protocol was fixed.

The GCP-50000 run was generated before this document was written but uses the
same scientific protocol: spatial matcher database followed by COLMAP 4.0.4
`global_mapper`. It should be normalized into this run-root convention when
space and time allow, but the completed sparse result remains valid evidence.
