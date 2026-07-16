# MS-GCP v1.3 Original 3DGS F5 Recipe

Status: frozen recipe for the first 3K end-to-end reference smoke.

## Method source

- Official repository: `graphdeco-inria/gaussian-splatting`.
- Training commit: `2eee0e26d2d5fd00ec462df47752223952f6bf4e`.
- Rasterizer submodule: `59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d`.
- `simple-knn` submodule: `44f764299fa305faf6ec5ebd99939e0508331503`.
- The official training checkout is not patched. A one-line upstream `<cfloat>` compatibility patch is applied only to the isolated `simple-knn` build copy so CUDA 12.8 can compile `FLT_MAX`.

This commit is the final official main commit on the original 2023 implementation line before the repository's 2024 feature release. It avoids later depth regularization, anti-aliasing, exposure compensation, and sparse optimizer changes.

## Training contract

The formal 3K run uses all 94 source images at `--resolution 8`, seed 0, 30,000 iterations, black background, SH degree 3, and every optimizer/densification default recorded in `configs/gcp_v13_original_3dgs_recipe_v1.json`. There is no held-out image set and no GCP, split, survey coordinate, residual, or release observation visible to training.

The formal checkpoint is:

```text
02_checkpoints/model/point_cloud/iteration_30000/point_cloud.ply
```

The point cloud, `cameras.json`, and `cfg_args` remain method outputs under the unique run root. Nothing is written to the dataset mirror, v1.3 release, official source checkout, or another method's directory.

## Isolation

Server roots are disjoint:

```text
/root/autodl-tmp/datasets/ms-gcp-v13/<release-digest>/...
/root/autodl-tmp/worktrees/ms-gcp-v13/3dgs-original/<commit>/...
/root/autodl-tmp/envs/ms-gcp-v13/3dgs-original/<environment>/...
/root/autodl-tmp/build/ms-gcp-v13/3dgs-original/<commit>/<run-id>/...
/root/autodl-tmp/runs/ms-gcp-v13/3dgs-original/gcp_3000_20260602/<run-id>/...
```

The launcher refuses pre-existing build/run roots, verifies the recipe, environment lock, release integrity, source hashes, clean method source and submodules, and read-only dataset/release trees before training. It recomputes dataset and source status after training. Any mutation invalidates the run.

## Execution order

1. Validate recipe and workspace isolation.
2. Verify release payload integrity and source hashes.
3. Run official 3DGS training at 30K.
4. Verify the final checkpoint and immutable inputs.
5. Build the separate evaluation-only metric-depth adapter.
6. Export v2 packets only for v1.3 formal annotated views.
7. Run release-mode formal evaluator and independent metric recomputation.

Packet export does not modify this training checkout or checkpoint. The packet adapter has its own fixed source, environment, build cache, and provenance record.
