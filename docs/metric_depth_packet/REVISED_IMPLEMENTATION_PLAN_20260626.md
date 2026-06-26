# Metric Depth Packet Revised Implementation Plan

Date: 2026-06-26

## Scope

This implementation is evaluation-only. It extends the Gaussian rasterizer output API and the GCP evaluator depth-export path so that a metric depth packet can be exported and audited without changing training, checkpoints, Gaussian attributes, densification, pruning, optimizer state, losses, support, GCP point sets, or control/checkpoint splits.

The formal P1 evaluator primary camera-depth tensor is fixed as:

`alpha_normalized_expected_camera_z = M1 / A`

where `A = sum_i alpha_i T_i` and `M1 = sum_i alpha_i T_i z_i`.

## Implemented Repositories

- Renderer/training repository: `E:\Multispectral`
- GCP benchmark/evaluator repository: `E:\M3M-GCP-3DGS`
- Branch in both repositories: `codex/gcp-metric-depth-packet-20260626`

## Renderer Changes

The renderer keeps the existing return path by default. The new packet is returned only when `return_metric_depth_packet=True` is explicitly set.

The CUDA rasterizer now optionally returns a 9-channel tensor:

1. `accumulated_alpha`
2. `weighted_camera_z_sum`
3. `weighted_camera_z_second_moment`
4. `weighted_inverse_camera_z_sum`
5. `alpha_normalized_expected_camera_z`
6. `alpha_normalized_expected_inverse_camera_z`
7. `harmonic_camera_z`
8. `camera_z_variance`
9. `metric_depth_valid_mask`

The legacy `depth` payload remains the historical unnormalized inverse-depth artifact and is not changed.

## Evaluator and Exporter Changes

The exporter writes one `.npz` packet per image and a v2 manifest. Each packet stores all raw accumulators, all derived tensors, the valid mask, and the legacy payload under the explicit name:

`historical_invalid_unnormalized_inverse_depth`

The release evaluator accepts v2 packets only if the manifest and packet satisfy the locked schema, tensor names, dtype, shape, image-domain, pixel convention, per-file SHA-256, and model/checkpoint content-hash requirements.

## Test Status

Completed locally:

- Python syntax checks.
- CPU synthetic metric-depth packet tests.
- Evaluator protocol and v2 packet validation tests.

Completed in the local compatible CUDA environment:

- Tiny CUDA compile/export parity test.
- The base Python environment was incompatible (`torch cu124` versus CUDA toolkit 11.8), so the local `gs` conda environment (`torch 2.4.1+cu118`) was used.
- CUDA extension editable install succeeded in `gs`.
- The synthetic packet test passed and generated a representative `.npz` packet plus manifest under `outputs/metric_depth_packet_20260626/cuda_tiny/`.

## Execution Boundary

Not executed:

- 3K/5K/100K evaluator regression.
- Scene rendering/export.
- Training.
- Checkpoint mutation.
- Support or split mutation.
- External baseline adaptation.
