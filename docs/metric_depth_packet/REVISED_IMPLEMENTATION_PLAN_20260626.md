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
- The final safety pass additionally verified independent raw `A/M1/M2/H` assertions, multi-opacity behavior, off-axis camera-z behavior, metric-packet non-differentiability, and eval-enabled/disabled gradient parity.

## Final protocol-lock fix

Following file-level review, `alpha_cutoff` and `early_termination_threshold` are no longer exporter CLI parameters. They are recorded as fixed rasterizer behavior:

- `alpha_cutoff = 1/255`
- `early_termination_threshold = 1e-4`

If future work studies these thresholds, it must use a separate diagnostic protocol that actually connects threshold values to the CUDA kernel.

The evaluator now recomputes derived packet tensors using `numerical_support_floor` and `variance_clamp_tolerance` from the manifest, and requires both fields to be finite and non-negative.

## Variance recomputation validation fix

Real-image regression exposed only a `camera_z_variance` recomputation tolerance issue. Formal P1 depth (`M1/A`), raw accumulators (`A/M1/M2/H`), CUDA packet formulas, `numerical_support_floor`, and `variance_clamp_tolerance` remain unchanged.

The evaluator/exporter now lock a separate variance recomputation audit policy in the metric-depth manifest:

- `variance_validation_policy = float_forward_error_bound_v1`
- `variance_validation_abs_floor = 1e-5`
- `variance_validation_ulp_factor = 8`
- `variance_validation_dtype = float32`
- `variance_validation_rtol = 0`

For valid pixels the variance audit uses `abs(packet_variance - (M2/A - (M1/A)^2)) <= abs_floor + ulp_factor * eps(dtype) * max(abs(M2/A), abs((M1/A)^2), 1.0)`. Other derived tensors keep the existing fixed recomputation tolerance. The policy records the worst pixel, valid-pixel count, failing-pixel count, maximum absolute error, maximum allowed error, and max error-to-bound ratio. This policy is not a CLI knob and does not change packet values.

## Execution Boundary

Not executed:

- 3K/5K/100K evaluator regression.
- Scene rendering/export.
- Training.
- Checkpoint mutation.
- Support or split mutation.
- External baseline adaptation.
