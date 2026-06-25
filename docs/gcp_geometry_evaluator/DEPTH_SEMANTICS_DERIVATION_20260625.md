# Renderer Depth Semantics Derivation

This note documents the depth tensor exported by
`code/gcp/export_gaussian_depth_maps.py` for the current local UMGS/Graphdeco
renderer stack. It is an implementation audit note, not a formal metric-depth
approval.

## Source Trace

The exporter calls:

```python
payload = render(...)
depth = payload["depth"]
```

The relevant local source files are:

- `E:\Multispectral\gaussian_renderer\__init__.py`
- `E:\Multispectral\submodules\diff-gaussian-rasterization\diff_gaussian_rasterization\__init__.py`
- `E:\Multispectral\submodules\diff-gaussian-rasterization\cuda_rasterizer\forward.cu`

The Python renderer returns:

```python
rendered_image, radii, depth_image = rasterizer(...)
out = {"render": rendered_image, ..., "depth": depth_image}
```

The Python rasterizer wrapper returns `invdepths` from the C++/CUDA extension:

```python
num_rendered, color, radii, ..., invdepths = _C.rasterize_gaussians(...)
return color, radii, invdepths
```

In `forward.cu`, each Gaussian stores camera-space z:

```cpp
depths[idx] = p_view.z;
```

The per-pixel depth output accumulates:

```cpp
expected_invdepth += (1 / depths[collected_id[j]]) * alpha * T;
invdepth[pix_id] = expected_invdepth;
```

## Mathematical Definition

For pixel \(p\), sorted contributing Gaussians \(j\), camera-space depth
\(z_j\), per-Gaussian alpha \(\alpha_j(p)\), and transmittance \(T_j(p)\), the
exported tensor is:

\[
D(p) = \sum_j \frac{\alpha_j(p) T_j(p)}{z_j}.
\]

No accumulated alpha/weight normalization is applied in this source path. Thus
`payload["depth"]` is an alpha/transmittance-weighted unnormalized inverse
camera-z accumulation. It is not:

- metric camera-z;
- ray distance;
- normalized expected inverse camera-z;
- first-surface depth;
- a LiDAR/GCP ground-truth depth map.

## Consequence for the P1 Evaluator

If a single plane at \(z=20\) contributes with opacity/weight \(w\), the stored
value is \(D=w/20\). A naive conversion \(z=1/D\) recovers \(20/w\), not \(20\).
Therefore changing opacity changes the recovered depth even though geometry is
unchanged.

The formal GCP geometry evaluator must reject this artifact unless a future
exporter also provides the accumulated weight/alpha needed to normalize it, or
exports metric `camera_z` / `ray_distance` directly.

The exporter now records this tensor as:

```text
alpha_weighted_unnormalized_inverse_camera_z
```

and the evaluator rejects that semantics in formal depth-only evaluation.
