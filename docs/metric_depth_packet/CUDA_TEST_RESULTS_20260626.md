# CUDA Tiny Test Results

Date: 2026-06-26

The required tiny synthetic CUDA compile/export test was completed locally in a compatible CUDA environment.

## Runtime

- Environment: `D:\anaconda\envs\gs`
- PyTorch: `2.4.1+cu118`
- PyTorch CUDA runtime: `11.8`
- CUDA toolkit: `11.8`
- GPU: NVIDIA GeForce RTX 3050

The default base Python environment could not compile the extension because it uses PyTorch `2.6.0+cu124` while the local CUDA toolkit is `11.8`. The compatible `gs` environment was therefore used for the required tiny synthetic test.

## Compile/Import

Command:

```powershell
& 'D:\anaconda\envs\gs\python.exe' -m pip install -e E:\Multispectral\submodules\diff-gaussian-rasterization
```

Result:

- `diff_gaussian_rasterization` editable install succeeded.
- CUDA extension compile/import path is available in the `gs` environment.

## Test Command

```powershell
& 'D:\anaconda\envs\gs\python.exe' E:\M3M-GCP-3DGS\code\gcp\test_metric_depth_packet_cuda.py `
  --train_repo E:\Multispectral `
  --out_dir E:\M3M-GCP-3DGS\outputs\metric_depth_packet_20260626\cuda_tiny
```

## Results

Output file:

`E:\M3M-GCP-3DGS\outputs\metric_depth_packet_20260626\cuda_tiny\metric_depth_packet_cuda_test_matrix.json`

Status: `PASS`

Validated checks:

- eval-disabled backward compatibility proxy:
  - RGB bitwise equality: true
  - old depth payload bitwise equality: true
  - max RGB abs error: 0
  - max old-depth abs error: 0
- single-plane expected camera-z opacity invariance:
  - expected z = 20, actual z = 20
  - harmonic z = 20
  - variance = 0
- two-layer raw-vs-derived packet consistency:
  - `alpha_normalized_expected_camera_z` max abs error: `9.15e-7`
  - `alpha_normalized_expected_inverse_camera_z` max abs error: `4.56e-10`
  - `harmonic_camera_z` max abs error: `2.09e-7`
  - `camera_z_variance` max abs error: `2.80e-5`
  - historical invalid inverse-depth payload matches `H` within `3.73e-9`
- derived tensor recomputation from `A/M1/M2/H`: pass
- zero-alpha invalid/NaN policy: pass

Representative artifacts:

- `tiny_metric_depth_packet_two_layer.npz`
- `tiny_metric_depth_packet_cuda_manifest.json`

No scene regression, training, checkpoint mutation, support mutation, split mutation, or real data rendering was run.
