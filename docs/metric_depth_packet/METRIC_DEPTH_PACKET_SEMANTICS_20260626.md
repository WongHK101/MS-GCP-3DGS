# Metric Depth Packet Tensor Semantics

Date: 2026-06-26

For each pixel, each contributing Gaussian sample has camera-axis depth `z_i`, opacity `alpha_i`, and front-to-back transmittance `T_i`. Define:

`w_i = alpha_i T_i`

`A = sum_i w_i`

`M1 = sum_i w_i z_i`

`M2 = sum_i w_i z_i^2`

`H = sum_i w_i / z_i`

The packet stores:

| Tensor | Formula | Formal P1 role |
|---|---|---|
| `accumulated_alpha` | `A` | raw audit accumulator |
| `weighted_camera_z_sum` | `M1` | raw audit accumulator |
| `weighted_camera_z_second_moment` | `M2` | raw audit accumulator |
| `weighted_inverse_camera_z_sum` | `H` | raw audit accumulator |
| `alpha_normalized_expected_camera_z` | `M1 / A` when `A > floor` else NaN | formal P1 primary camera-z |
| `alpha_normalized_expected_inverse_camera_z` | `H / A` when `A > floor` else NaN | sensitivity diagnostic only |
| `harmonic_camera_z` | `A / H` when `A > floor` and `H > 0` else NaN | sensitivity diagnostic only |
| `camera_z_variance` | `M2/A - (M1/A)^2` when `A > floor` else NaN | validity/scatter diagnostic |
| `metric_depth_valid_mask` | `A > numerical_support_floor` | depth-packet validity mask |

`normalization_epsilon` is reserved metadata for future compatibility. It is not an active denominator in the current formal P1 definitions. The active definitions use strict `M1/A`, `H/A`, and `A/H` after validity gating. Empty or near-empty support pixels must remain invalid and NaN rather than being made finite by epsilon-only division.

The protocol records `alpha_cutoff=1/255` and `early_termination_threshold=1e-4` as fixed behavior of the current rasterizer. The exporter does not expose these as CLI parameters. Any future threshold-sensitivity experiment must be a separate diagnostic protocol that actually wires the threshold values into the CUDA kernel.

## Invalid Pixel Policy

- If `A <= numerical_support_floor`, all derived depth/variance tensors are NaN and `metric_depth_valid_mask=false`.
- If `H <= 0`, `harmonic_camera_z=NaN`.
- Tiny negative variance from floating-point roundoff may be clamped to zero only within `variance_clamp_tolerance`.
- Clearly negative variance is a test failure.

## Legacy Depth Payload

The old renderer payload is:

`sum_i alpha_i T_i / z_i`

It is stored, if present, only as:

`historical_invalid_unnormalized_inverse_depth`

It must not enter formal P1 evaluation, formal ranking, or `camera_z=1/depth` backprojection.
