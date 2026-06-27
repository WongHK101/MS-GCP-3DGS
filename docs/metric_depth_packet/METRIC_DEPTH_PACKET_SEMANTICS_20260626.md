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
| `camera_z_variance` | raw float32 `M2/A - (M1/A)^2` when `A > floor` else NaN | raw validity/scatter diagnostic |
| `camera_z_variance_diagnostic` | diagnostic-only non-negative view after forward-bound validation; unresolved pixels are NaN | variance diagnostic only; not stored as raw packet |
| `camera_z_variance_diagnostic_valid_mask` | boolean mask for pixels whose variance packet/ref consistency and non-negativity checks pass | variance diagnostic only |
| `metric_depth_valid_mask` | `A > numerical_support_floor` | depth-packet validity mask |

`normalization_epsilon` is reserved metadata for future compatibility. It is not an active denominator in the current formal P1 definitions. The active definitions use strict `M1/A`, `H/A`, and `A/H` after validity gating. Empty or near-empty support pixels must remain invalid and NaN rather than being made finite by epsilon-only division.

The protocol records `alpha_cutoff=1/255` and `early_termination_threshold=1e-4` as fixed behavior of the current rasterizer. The exporter does not expose these as CLI parameters. Any future threshold-sensitivity experiment must be a separate diagnostic protocol that actually wires the threshold values into the CUDA kernel.

## Invalid Pixel Policy

- If `A <= numerical_support_floor`, all derived depth/variance tensors are NaN and `metric_depth_valid_mask=false`.
- If `H <= 0`, `harmonic_camera_z=NaN`.
- Raw `camera_z_variance` is never overwritten in CUDA, exporter, or NPZ output, including values within `variance_clamp_tolerance`.
- Variance packet/ref consistency failure remains a hard packet-validation failure.
- Negative raw variance whose packet/ref consistency passes but whose packet or reference value is below the frozen non-negativity bound is not a formal P1 blocker; it is recorded as `float32_variance_nonnegativity_unresolved` and excluded from the diagnostic variance view.
- Accepted cancellation-consistent negative values are classified as `float_cancellation_consistent_with_zero`; only the downstream diagnostic view is zero-clamped.

## Variance Recomputaton Validation

`camera_z_variance` is still the packet value emitted by the renderer/export path. Validation uses a scale-aware forward-error bound only to audit recomputation from the raw accumulators:

`mu = M1/A`

`second = M2/A`

`variance_ref = second - mu^2`

`scale = max(abs(second), abs(mu^2), 1.0)`

`allowed_error = variance_validation_abs_floor + variance_validation_ulp_factor * eps(dtype) * scale`

The locked manifest policy is:

- `variance_validation_policy = float_forward_error_bound_v1`
- `variance_validation_abs_floor = 1e-5`
- `variance_validation_ulp_factor = 8`
- `variance_validation_dtype = float32`
- `variance_validation_rtol = 0`

This validation policy does not modify packet values and is separate from `variance_clamp_tolerance`.

For non-negativity, the same frozen bound is applied to both the packet value and the raw-accumulator recomputation:

- `abs(packet - variance_ref) <= allowed_error`
- `packet >= -allowed_error`
- `variance_ref >= -allowed_error`

The first condition is the packet-level hard gate. The latter two conditions determine whether the variance diagnostic view is valid. If packet/ref consistency passes but either non-negativity condition fails, the formal P1 depth tensor `alpha_normalized_expected_camera_z` remains valid, while the variance diagnostic mask is false for that pixel.

The manifest locks:

- `variance_nonnegativity_policy = float_forward_error_bound_v1`
- `variance_negative_handling = preserve_raw_and_zero_clamp_diagnostic_only`
- `variance_raw_packet_modified = false`

`variance_clamp_tolerance` remains recorded for protocol traceability but does not authorize mutation of the raw NPZ tensor.

## Legacy Depth Payload

The old renderer payload is:

`sum_i alpha_i T_i / z_i`

It is stored, if present, only as:

`historical_invalid_unnormalized_inverse_depth`

It must not enter formal P1 evaluation, formal ranking, or `camera_z=1/depth` backprojection.
