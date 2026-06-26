# Metric Depth Packet Manifest v2 Notes

Date: 2026-06-26

The manifest schema is:

`ms_gcp_metric_depth_packet_manifest_v2`

Required protocol locks:

- `packet_schema`
- `primary_depth_tensor`
- `primary_depth_semantics`
- `tensor_names`
- `tensor_formulas`
- `dtype`
- `image_domain`
- `distorted_or_undistorted`
- `pixel_coordinate_convention`
- `camera_model_source`
- `alpha_cutoff`
- `early_termination_threshold`
- `numerical_support_floor`
- `normalization_epsilon`
- `variance_clamp_tolerance`
- `model_content_hash`
- `renderer_commit`
- `rasterizer_commit`
- `exporter_commit`
- `depth_index`

Each `depth_index` row must include:

- `image_name`
- `packet_path`
- `packet_sha256`
- `packet_bytes`
- `height`
- `width`
- `dtype`
- `primary_depth_tensor`
- `primary_depth_semantics`

The formal evaluator rejects:

- missing required tensor;
- tensor shape mismatch;
- dtype mismatch;
- packet file hash mismatch;
- unknown manifest schema;
- primary tensor other than `alpha_normalized_expected_camera_z`;
- primary semantics other than `camera_z`;
- missing model/checkpoint content hash;
- missing finite non-negative `numerical_support_floor`;
- missing finite non-negative `variance_clamp_tolerance`;
- manual release-mode CLI overrides for semantics, key, scale, offset, domain, or convention.

`normalization_epsilon` is reserved metadata in the current schema. It is recorded for protocol traceability but is not an active denominator in formal P1 metric-depth calculation.

`alpha_cutoff=1/255` and `early_termination_threshold=1e-4` are fixed rasterizer behavior fields. The exporter must not expose CLI options that write different values to the manifest unless a future diagnostic protocol also wires those values into the CUDA kernel.
