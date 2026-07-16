# GCP v1.3 Geometry Method Candidate Set

Status: protocol candidate, not yet a frozen experiment matrix.

UMGS is excluded from the paper's method comparison. Its retained artifacts
are historical implementation and audit evidence only.

## Admission criteria

A method may enter the formal matrix only if it:

1. has an official public implementation and a fixed source commit;
2. accepts a static multi-view custom COLMAP scene without using test GCPs;
3. makes an explicit geometry, depth, surface, or large-scene geometry claim;
4. can train and render on the frozen v1.3 image/camera contract;
5. can expose the approved metric-packet v2 accumulators or an independently
   validated equivalent adapter with formal camera-z `M1/A`;
6. passes the 3K compile, train, checkpoint, packet, and evaluator smoke;
7. uses a checkpoint rule and hyperparameters fixed before checkpoint errors
   are inspected.

## Formal core candidates

| Method | Role | Geometry relevance | Official implementation | Key integration risk |
|---|---|---|---|---|
| 3D Gaussian Splatting (3DGS) | canonical base method | establishes whether geometry-oriented variants improve survey-grade absolute accuracy | https://github.com/graphdeco-inria/gaussian-splatting | metric-packet v2 rasterizer adapter |
| 2D Gaussian Splatting (2DGS) | geometry primitive baseline | oriented planar disks, perspective-correct ray-splat intersection, depth distortion and normal consistency | https://github.com/hbb1/2d-gaussian-splatting | surfel depth semantics and packet parity |
| PGSR | planar/multi-view geometry method | planar Gaussians, unbiased depth rendering, normal and multi-view constraints | https://github.com/zju3dv/PGSR | freeze custom-scene neighbor and depth-filter settings without GCP tuning |
| RaDe-GS | 3D Gaussian depth method | rasterized depth/normal for standard 3D Gaussians and geometry regularization | https://github.com/HKUST-SAIL/RaDe-GS | reconcile its native depth with formal `M1/A` while preserving native diagnostics |
| Gaussian Opacity Fields (GOF) | implicit-surface geometry method | opacity-field regularization and adaptive surface extraction in unbounded scenes | https://github.com/autonomousvision/gaussian-opacity-fields | older CUDA stack and packet adapter |
| CityGaussianV2 | large-scene geometry baseline | 2DGS-derived large-scene geometry, depth regression, elongation control, and geometry benchmark | https://github.com/Linketic/CityGaussian | partition/config adaptation to relatively small 3K smoke |
| CityGS-X | scalable large-scene geometry method | hierarchical distributed representation and progressive RGB-depth-normal geometry optimization | https://github.com/gyy456/CityGS-X | multi-GPU path, external depth prior provenance, packet export from hierarchy |
| MetroGS | recent large-scene geometry method | distributed 2DGS, structured dense initialization, hybrid mono/multi-view geometry optimization | https://github.com/M3phist0/MetroGS | pointmap dependency, distributed checkpoint merge, hardware feasibility |

The formal matrix is not frozen merely by listing these methods. Each row must
pass the identical 3K smoke gate. A failed method is reported with a concrete
protocol-compatible failure reason; it is not silently replaced or tuned using
GCP residuals.

## Conditional 3K feasibility candidates

| Method | Reason to test | Promotion gate |
|---|---|---|
| Quadratic Gaussian Splatting (QGS) | second-order geometric primitives and strong surface-reconstruction claim | official code must accept the benchmark camera track and expose a validated camera-z packet without changing its representation |
| Geometry Field Splatting with Gaussian Surfels (GFSGS) | explicit geometry-field rendering with Gaussian surfels | must work on unbounded UAV scenes and pass memory/runtime plus metric-packet parity gates |

These methods are not part of the promised formal matrix until the 3K gate
passes. Their failure cannot delay the eight-method core matrix.

## Excluded from the primary camera-z ranking

- SuGaR and GS2Mesh are useful mesh extraction/post-processing approaches, but
  their primary surface intersection is not the same output contract as the
  alpha-composited camera-z packet. They may be evaluated later in a separate
  mesh/surface diagnostic track.
- Sparse-view-specialized, dynamic-scene, relighting, transparent-object, and
  generative methods are outside the benchmark question.
- UMGS is not a paper method or baseline. Existing UMGS artifacts remain only
  where required to reproduce historical packet/protocol audits.

## Frozen smoke order

1. 3DGS
2. 2DGS
3. PGSR
4. RaDe-GS
5. GOF
6. CityGaussianV2
7. CityGS-X
8. MetroGS
9. QGS (conditional)
10. GFSGS (conditional)

Every method starts with `gcp_3000_20260602`. Full six-scene training is
allowed only after the method passes source-view identity, training, fixed
checkpoint, metric-packet, release-loader, and independent metric-recompute
checks on 3K.
