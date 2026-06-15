# GCP-50000 COLMAP Fragmentation Diagnosis

Date: 2026-06-15 (CST)

## Current run

- Scene: `gcp_50000_20260610`
- Images: 2,209 RGB frames from one continuous sortie
- Capture interval: 2026-06-10 16:19:44 to 18:04:39
- COLMAP: 4.0.4 with CUDA and GPU bundle adjustment
- Matching: spatial, 80 neighbors, 500 m maximum distance
- Mapper: incremental, multiple models enabled

After approximately three hours of mapping, ten saved submodels existed. The
largest saved model registered 160 images, and the union of saved models
contained only 562 unique images. The current branch was still growing, but
its image IDs were concentrated in another local block. This is severe
fragmentation rather than a nearly complete reconstruction with a few
outliers.

## Match-graph evidence

The verified two-view graph is not disconnected:

- verified edges: 80,302
- connected components: 1
- largest component: 2,209 images
- isolated images: 0
- degree: minimum 39, median 71, mean 72.70, maximum 114
- strong-edge degree (`>=100` inliers): minimum 14, median 47

Sequential and local matches are also strong:

- sequence offset 1: 2,207 edges, median 3,178 inliers
- offsets 2--5: 5,079 edges, median 1,231 inliers
- offsets 41--80: 12,885 edges, median 532 inliers

Therefore, the failure is not explained by missing adjacent-frame matches or
a disconnected database. This is a campus scene containing roads, buildings,
and vegetation, not an agricultural scene, so generic low-texture farmland is
not the primary explanation. A connected graph can still contain inconsistent
or locally dominant geometry when nadir and oblique images are selected only
by camera-position proximity. The incremental mapper is repeatedly forming
locally coherent reconstructions but is not maintaining one globally
extensible model.

## Spatial-matching orientation audit

The scene contains 318 nadir images and 1,891 approximately 45-degree oblique
images. COLMAP 4.0.4 spatial matching:

- uses nearest camera positions;
- defaults to `SpatialMatching.ignore_z=1`;
- does not use gimbal yaw, pitch, or the projected camera footprint.

The current camera has an estimated horizontal/vertical field of view of
approximately 71.6/56.8 degrees. At about 50 m relative altitude, a 45-degree
oblique camera's optical-axis ground intercept is shifted roughly 50 m from
the camera position. Camera-position proximity is therefore not equivalent to
view-frustum overlap.

The live database confirms candidate-budget waste:

- oblique pairs with yaw difference below 30 degrees: 30,215 candidates,
  99.3% geometrically verified, median 773 inliers;
- oblique pairs with yaw difference from 60 to 120 degrees: 32,055 candidates,
  81.4% verified, median 87 inliers;
- oblique pairs with yaw difference from 150 to 180 degrees: 11,022
  candidates, only 2.8% verified, median 30 inliers.

For an oblique image, the median candidate degree is 75, but only 22 candidates
have a yaw difference below 30 degrees; 16 have an approximately opposite
heading. Thus, the 80-neighbor spatial budget is substantially occupied by
camera-near but often footprint-distant views.

The configured 500 m distance cap is not the active limiting factor: verified
pairs are within about 79 m because the nearest-80 limit is reached first.
Changing only the distance cap or enabling altitude distance cannot solve the
orientation/FOV mismatch. Height handling matters if a scene mixes flight
levels, but this scene is predominantly at about 50 m relative altitude.

The saved submodels share registered images and may be mergeable in part, but
their current union covers only a minority of the input. Post-hoc merging alone
cannot recover the remaining unregistered frames.

## Parameter judgment

Do not use the following as the primary fix:

- lowering absolute-pose inlier thresholds;
- lowering `min_model_size`;
- disabling multiple models;
- accepting a union of unrelated fragments;
- exhaustive matching over all 2,209 images.

These changes either do not create missing global consistency or risk accepting
unstable poses. The current registered images already have thousands of visible
3D-point correspondences, so permissive registration thresholds are not the
main bottleneck.

## Recommended staged response

1. Preserve the current feature/match database and logs as a failed
   incremental-mapper diagnostic.
2. Stop the queue before it automatically launches the 100,000-square-metre
   scene with the same large-scene mapper protocol.
3. Run COLMAP 4.0.4 `global_mapper` on a copy of the existing database and a
   separate output root. This reuses all extracted features and matches.
4. Evaluate registration count, reprojection error, model connectivity, GPS
   alignment residuals, and GCP residuals before accepting the model.
5. If global mapping is incomplete, test `hierarchical_mapper` with overlapping
   clusters (`leaf_max_num_images` around 300--500 and `image_overlap` around
   50--100).
6. Only if both mapping strategies expose weak geographic regions, augment the
   existing database without re-extracting features:
   - always include sequential neighbors;
   - generate custom pairs from EXIF position, gimbal yaw/pitch, relative
     altitude, and estimated ground-footprint overlap;
   - optionally add guided or vocabulary-tree matching for long-range links.

Increasing spatial neighbors alone is a fallback, not the preferred repair,
because it also increases the number of orientation-incompatible pairs.

The protocol change must be recorded in machine-readable provenance. It may be
described as a scale-aware SfM backend for large scenes, but should not be
hidden when reporting reproducibility.

## External technical basis

- COLMAP's official tutorial recommends spatial matching when accurate GPS is
  available, vocabulary-tree matching for collections of several thousand
  images, and combining matching modes because previously matched pairs are
  skipped with low overhead.
- The tutorial recommends additional/guided matching when images remain
  unregistered.
- COLMAP documents `hierarchical_mapper` as a pipeline for large scenes.
- COLMAP 4 includes the GLOMAP global mapper. The ECCV 2024 GLOMAP paper
  motivates global SfM as substantially more scalable than repeated
  incremental registration and bundle adjustment while retaining comparable
  reconstruction quality.

References:

- https://colmap.github.io/tutorial.html
- https://colmap.github.io/faq.html
- https://colmap.github.io/changelog.html
- https://lpanaf.github.io/eccv24_glomap/
- https://demuc.de/papers/pan2024glomap.pdf
