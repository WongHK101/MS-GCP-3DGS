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
- manual release-mode CLI overrides for semantics, key, scale, offset, domain, or convention.

