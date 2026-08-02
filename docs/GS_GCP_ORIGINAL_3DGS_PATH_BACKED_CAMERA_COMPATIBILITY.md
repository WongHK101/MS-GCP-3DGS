# Original 3DGS Path-Backed Camera Compatibility

## Accepted blocker diagnosis

The 50K Stage 0.5 preflight exposed overlapping eager camera materialization:
all-image PIL/source-image backing remained live while quarter-resolution
`Camera` tensors were materialized sequentially. Host/cgroup memory therefore
continued to grow before the temporary source-image backing could be released.
The evidence does not establish that CUDA RGB tensors alone exhausted host RAM.

## Candidate order

Candidate A is `path_backed_cuda_resident`: COLMAP camera records retain paths
and metadata, each JPEG is decoded once during camera-list construction and
closed immediately, and the resulting quarter-resolution tensor remains on
CUDA. Candidate B is the same path-backed implementation with the official
`--data_device cpu`; it is eligible only if Candidate A passes every host,
lifecycle, camera, ray, and protocol gate and fails only the frozen GPU-memory
gate. A host or non-GPU failure stops the audit.

Neither candidate changes the image set, ordering, resolution, PIL resize,
camera matrices, training loop, viewpoint RNG, loss, densification, or formal
geometry protocol. Per-iteration JPEG decoding, caches, workers, prefetch,
pinned-memory pipelines, and resource-gate changes are outside this contract.

## Evidence sequence

Before large-scene preflight, the implementation freezes the cross-scene parity
sample manifest, compares all 3K train/test images and the preregistered samples,
and performs a path-backed identity/dimension/FD-closure audit over all 6,187
images. Large-scene selection is Candidate A 100K, Candidate A 50K, then
Candidate B only under the GPU-only rule. The selected contract must pass the
frozen 3K synthetic, eager-reference, candidate, and external-probe equivalence
checks before the 30K Stage 0.5 qualification may start.

Resource blocks are infrastructure feasibility outcomes, not geometry failures.
The canonical host classification is `HOST_RAM_BLOCKED`, with
`failure_stage=camera_load` and
`failure_reason=host_cgroup_peak_exceeded_frozen_gate`.
