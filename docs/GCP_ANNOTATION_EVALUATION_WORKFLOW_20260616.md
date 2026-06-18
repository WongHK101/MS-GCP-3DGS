# GCP Annotation and Evaluation Workflow

This workflow defines the lightweight GCP image-observation scaffold for the
second-paper M3M-GCP dataset. It does not run COLMAP, train 3DGS, mutate
checkpoints, or change surveyed GCP coordinates.

## Inputs

- Canonical primary-usable GCP table:
  `evidence/gcp_coordinates/gcp_points_primary_usable_cgcs2000_cm108_20260615.csv`
- Raw scene images:
  `E:\datasets\M3M-GCP\scenes\<scene_id>\*_D.JPG`
- Completed COLMAP sparse models, when available:
  `/root/autodl-tmp/runs/ms-gcp-3dgs/colmap-4.0.4-global-formal-20260616/<scene_id>/RGB/sparse_aligned/0`

## Stage 1: Coarse Candidate Discovery

Build projected candidates from DJI EXIF/Gimbal metadata:

```powershell
python code\gcp\build_gcp_projection_candidates.py `
  --scene gcp_3000_20260602 `
  --max_candidates_per_gcp 12
```

The output under `outputs/gcp_annotation_candidates_20260616/<scene_id>/`
contains:

- `image_metadata.csv`
- `gcp_projection_candidates.csv`
- `gcp_visibility_summary.csv`
- `contact_sheets/<point_name>.jpg`
- `projection_manifest.json`

These projections are only candidate-discovery aids. They are not visibility
proof and must not be used directly as GCP observations.

## Stage 2: Manual Image Observation

Open the annotator on a candidate CSV:

```powershell
python code\gcp\manual_gcp_annotator.py `
  --candidates_csv outputs\gcp_annotation_candidates_20260616\gcp_3000_20260602\gcp_projection_candidates.csv `
  --out_csv outputs\gcp_annotations\gcp_3000_20260602_manual_annotations.csv `
  --annotator user
```

The yellow cross is the coarse projection. Click the actual GCP center, then
mark:

- `v`: visible and good;
- `a`: visible but ambiguous;
- `x`: not visible;
- `n` / `p`: next / previous;
- `s`: save;
- `q`: save and quit.

The annotator also provides scene/file switching controls:

- `Candidates`: load another candidate CSV without restarting the program;
- `Output CSV`: choose where the current scene's manual observations are saved;
- `Image root`: override stale image paths by resolving `image_name` under a
  selected scene folder;
- `Reload current`: reload the paths currently typed in the controls.

The display overlays are:

- yellow cross: coarse projected candidate;
- magenta cross: correction-assisted hint, estimated only from already saved
  2D residuals in the same image, same GCP, or same scene;
- cyan cross: the manual click that will be saved.

The correction hint is never saved as an observation by itself. It is only a
navigation aid; the evaluation CSV continues to use the manually clicked pixel.
The status bar reports the manual-to-coarse residual in pixels and, when
available, the residual relative to the correction-assisted hint.

The manual CSV stores scene, point name, image name, projected pixel, manual
pixel, visibility, quality, confidence, annotator, note, and timestamp.

## Stage 3: Observation Packaging

Summarize one or more manual annotation files:

```powershell
python code\gcp\summarize_gcp_annotations.py `
  --annotation_csv outputs\gcp_annotations\gcp_3000_20260602_manual_annotations.csv `
  --out_dir outputs\gcp_annotations\summary_20260616
```

This exports:

- `gcp_image_observations_for_evaluation.csv`
- `gcp_annotation_summary_by_scene.csv`
- `gcp_annotation_summary_manifest.json`
- `gcp_annotation_summary.md`

Only rows with `visible=1`, manual coordinates, and quality not equal to
`not_visible` are exported as evaluation-ready observations.

## Stage 4: Future Geometry Evaluation

After formal COLMAP models are available for all six scenes, triangulate manual
2D observations into model-space GCP points:

```powershell
python code\gcp\triangulate_gcp_points.py `
  --colmap_model <COLMAP_SPARSE_MODEL_DIR> `
  --observations_csv outputs\gcp_annotations\summary_20260616\gcp_image_observations_for_evaluation.csv `
  --out_dir outputs\gcp_evaluation\<scene_id>\triangulated `
  --scene <scene_id>
```

This exports `triangulated_gcp_model_points.csv`, which can be paired with the
surveyed GCP table.

Fit a global Sim(3) transform with explicit control points and evaluate held-out
checkpoints:

```powershell
python code\gcp\fit_gcp_sim3.py `
  --model_points_csv outputs\gcp_evaluation\<scene_id>\triangulated\triangulated_gcp_model_points.csv `
  --gcp_csv evidence\gcp_coordinates\gcp_points_primary_usable_cgcs2000_cm108_20260615.csv `
  --control_points G01,G02,G03,NC94 `
  --out_dir outputs\gcp_evaluation\<scene_id>\sim3_eval
```

The evaluator uses a single global similarity transform:

```text
target_xyz = scale * rotation @ model_xyz + translation
```

It intentionally does not perform local stretching, TPS warping, or non-rigid
deformation. Control residuals describe transform fit quality; checkpoint
residuals are the independent accuracy evidence when checkpoints are not used in
fitting.

Future rendered-depth or surface diagnostics should consume:

- `gcp_image_observations_for_evaluation.csv`;
- canonical 3D GCP coordinates;
- COLMAP cameras/images from `RGB/sparse_aligned/0`;
- optional rendered-depth or surface diagnostics.

They should report image reprojection residuals, scene-level residual statistics,
and, where appropriate, rendered-depth or surface-distance diagnostics.
