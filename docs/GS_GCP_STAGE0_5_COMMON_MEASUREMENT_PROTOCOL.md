# GS-GCP Stage 0.5 Common Measurement Protocol

## Scope

Stage 0.5 freezes `graphdeco_quarter_resolution_v1`,
`gs_gcp_rgb_holdout_split_v1`, and `gs_gcp_common_measurement_suite_v1`.
Only Original 3DGS may run the 3K qualification in this stage. The other five
scenes and all other methods remain closed.

The RGB protocol is **image-loss-held-out, pose-known under a shared all-image
SfM initialization** (`image_loss_holdout_under_shared_all_image_sfm_v1`). It
is not strict unseen-image reconstruction. Test RGB is excluded from loss,
method-specific priors, appearance/exposure fitting, checkpoint selection,
seed selection, and parameter selection. Test poses and shared sparse
initialization come from all-image SfM; shared point RGB and the initial PLY may
therefore contain test-view contributions. Every admitted method receives the
same frozen shared-SfM information and may not independently access test RGB,
test features, test 2D tracks, or test visibility statistics.

## Shared-SfM Assets

Each scene has three distinct assets:

1. `common_full_sfm`: byte-frozen full `cameras.bin`, `images.bin`,
   `points3D.bin`, and initial PLY. It is provenance/shared initialization, not
   a direct training source.
2. `train_camera_subset`: train RGB links, train camera/image records, and the
   shared PLY. It contains no `points3D.bin` and is not represented as a
   self-contained SfM reconstruction.
3. `test_camera_subset`: test RGB/camera records, benchmark GT, and shared PLY.
   It is opened only after training.

The training allowlist is shared point XYZ/RGB (and point error only after an
explicit method admission), train intrinsics/extrinsics, and train RGB. A
method requiring consistent tracks needs a separately reviewed deterministic
sanitized-track artifact.

## Resolution And RGB GT

Original Graphdeco `--resolution 4` uses independently computed Python
`round(decoded_dimension / 4)` dimensions and `PIL.Image.resize(size)`. The
decoded matrix ignores EXIF orientation; there is no crop, pad, renumbering,
or second resize. Pillow 11.1.0 resolves the RGB default to BICUBIC. The 3K
golden case is `5654 x 4098 -> 1414 x 1024`.

Benchmark GT follows the same decode and resize path, then the official
`np.array / 255`, CHW conversion, and torchvision 0.22.1 `save_image` PNG
serialization. Every source, tensor, and PNG is hashed. Render and GT names,
counts, and dimensions must match exactly. PSNR, SSIM, and LPIPS-VGG use the
frozen official definitions and full 64-character weight hashes. Primary RGB
aggregation is per-image, scene mean, then six-scene macro; incomplete scenes
have diagnostic successful-only means but no primary scene mean.

## Split

The six scene splits are generated once before training. Test count is
`ceil(N/8)`. Capture order is DJI timestamp, sequence, image name, then COLMAP
image ID. Camera centers use `C=-R^T t` in frozen COLMAP model coordinates.
Trajectory heading is derived from center displacement in model X/Y, including
the frozen near-zero, gap, turn, expansion, strip, transition, azimuth-octant,
Hamilton quota, and path-length bin rules in the split contract and generator.
GCP data, pixels, image quality labels, residuals, and method results are not
selection inputs.

## Measurement And Gates

Single-GPU reference rendering uses the frozen 901 GPU UUID, batch one, one
full warm-up, and five CUDA-event timing rounds. Method-native rendering is
secondary and records GPU count and per/aggregate VRAM.

The external resource probe samples process-tree RSS/FD, cgroup memory, and
GPU state at 1 Hz. Camera-load feasibility for 50K and 100K must pass all
host/GPU headroom, cgroup event, FD stability, JPEG closure, tensor-byte, ray,
and process-tree/cgroup consistency gates. Passing does not guarantee full
training success.

Probe on/off acceptance first requires a byte-exact synthetic child, then two
seed-0 100-iteration 3K runs with exact command, image order, trace, checkpoint
schema/count/order, and densification identities. Bitwise checkpoint equality
is preferred. Numeric tolerance is available only after separately proving
pre-existing CUDA nondeterminism and cannot waive count/order/schema or
densification differences.

The formal geometry remains metric packet v2
`alpha_normalized_expected_camera_z = M1/A`, camera-z semantics, patch 7
(radius 3), frozen multiview aggregation, and frozen control-only Sim(3).
Legacy 1600-width results are `high_resolution_1600_diagnostic_track` only.
