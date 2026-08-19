# M3M-GCP native-quarter heldout RGB quality suite v1

Status: **REVIEW_CANDIDATE_NOT_FORMAL**
Suite ID: `m3m_gcp_native_quarter_rgb_quality_v1`

This is an additive measurement suite bound to
`m3m_gcp_native_quarter_geometry_v2`. It does not change the geometry
evaluator, Sim(3), GCP coverage gate, completed geometry evidence, method
training recipes, or method pool.

## Frozen comparison

- Ground truth is decoded directly from the exact test-role JPEG bytes in the
  native-quarter input manifest. No secondary PNG ground-truth dataset is
  created.
- Every method renders the same 12 heldout 3K cameras at their exact
  1414x1025 resolution through its frozen official inference renderer and
  formal checkpoint.
- The adapter writes lossless RGB PNG using one benchmark-owned quantization
  rule. It performs no resize, crop, padding, color calibration, exposure fit,
  or heldout-image optimization.
- One benchmark-owned evaluator computes PSNR, SSIM, and LPIPS-VGG for every
  method. Method-native metric scripts are not used for the comparable table.
- Before computing a formal metric, that evaluator binds the render manifest
  to the frozen registry and rechecks the exact adapter, renderer source tree,
  camera root, model/checkpoint, auxiliary weights, and config hashes.
- A scene mean is publishable only when all frozen heldout views pass identity,
  shape, mode, finite-value, and hash checks. Successful-subset means are
  diagnostic only.

## Why render adapters are method-specific

Checkpoint and renderer APIs differ, but the comparison domain and metric code
must not. The adapters only load a frozen model/camera set and emit the common
PNG+manifest contract. Metric computation starts after that boundary.

Some Graphdeco-family camera loaders decode image bytes while constructing a
camera object. The adapter records that fact and clears `original_image`
before calling the renderer; renderer source identity is frozen, and no
heldout tensor is available at the renderer/appearance-policy boundary.

MetroGS is the only active 3K method requiring an explicit novel-view
appearance rule. Its official renderer trains one appearance embedding per
training view. For each heldout view, the adapter executes the frozen official
`find_most_similar_cameras(alpha=0.7)` rule on poses and copies the selected
training `camera.idx` appearance ID, matching the renderer's official setup
(`use_app_time=false`, `use_app_robust=false`). Heldout RGB is never read by the adapter and no
appearance parameter is optimized.

RaDe-GS and SOF use appearance modules only in their training losses. Their
formal RGB output is the deployable canonical base render with no heldout
embedding. SOF additionally reloads its frozen model-side `config.json` so its
official rasterizer uses the same sorting and culling settings as training.

## Metrics

The evaluator pins the Graphdeco reference implementations and both VGG weight
files by SHA-256. Inputs are full-frame RGB float32 sRGB in [0,1]:

- PSNR: official full-frame RGB MSE definition;
- SSIM: official 11x11, sigma-1.5 Gaussian window with zero padding;
- LPIPS-VGG: the Graphdeco-bundled `lpipsPyTorch` v0.1 VGG16 path.

Per-view values are averaged arithmetically for a scene. Once the six-scene
matrix exists, the paper-level summary uses an unweighted scene macro mean.

The machine-readable authority is
`configs/m3m_gcp_native_quarter_rgb_quality_v1.json`. Formal execution remains
locked while its status is `REVIEW_CANDIDATE_NOT_FORMAL`.
