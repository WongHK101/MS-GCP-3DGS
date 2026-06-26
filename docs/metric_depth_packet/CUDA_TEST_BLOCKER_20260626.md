# CUDA Tiny Test Blocker

Date: 2026-06-26

The required tiny synthetic CUDA compile/export test has not yet been completed in this pass.

Local environment:

- GPU: NVIDIA GeForce RTX 3050 detected by `nvidia-smi`.
- Python PyTorch: `torch 2.6.0+cu124`, CUDA runtime `12.4`.
- Installed CUDA toolkit: `11.8`.

Attempted local compile:

`pip install -e E:\Multispectral\submodules\diff-gaussian-rasterization`

Result:

Failed before compiling the extension because PyTorch was built with CUDA 12.4 but the detected toolkit was CUDA 11.8.

AutoDL status during this pass:

- Main AutoDL port `30970`: connection refused.
- Backup AutoDL port `28881`: connection refused.

Shared-server status during this pass:

- `172.18.23.177` SSH probe did not complete in the non-interactive check window.

Required next step before submitting the final metric-depth packet review package:

Run `code/gcp/test_metric_depth_packet_cuda.py` on a CUDA environment whose toolkit and PyTorch CUDA versions are compatible. The test must produce:

- CUDA extension compile/import success;
- all-packet tensor availability;
- eval-disabled legacy RGB/depth bitwise compatibility proxy;
- single-plane expected camera-z opacity invariance;
- two-layer raw-vs-derived expected/harmonic/variance checks;
- zero-alpha invalid/NaN policy;
- representative `.npz` packet and manifest;
- numeric absolute/relative errors and tolerances.

