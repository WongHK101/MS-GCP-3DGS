#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/root/autodl-tmp/envs/spectralindexgs_bw/bin/python}
TRAIN_REPO=${TRAIN_REPO:-/root/autodl-tmp/Multispectral}
SOURCE=${SOURCE:-/root/autodl-tmp/runs/ms-gcp-3dgs/gaussian-rgb-full6-r8-30k-20260618_1028/sources/gcp_3000_20260602}
RUN_ROOT=${RUN_ROOT:-/root/autodl-tmp/runs/ms-gcp-3dgs/gaussian-rgb-gcp3000-r1-30k-20260618_1519}
MODEL=${MODEL:-$RUN_ROOT/models/gcp_3000_20260602/Model_RGB}
ITERATIONS=${ITERATIONS:-30000}
RESOLUTION=${RESOLUTION:-1}

mkdir -p "$RUN_ROOT"/{models,logs,status,summary}
cat > "$RUN_ROOT/run_manifest.json" <<JSON
{
  "schema": "ms_gcp_rgb_gaussian_single_scene_r1_v1",
  "created_at_cst": "$(TZ=Asia/Shanghai date -Is)",
  "purpose": "R1 RGB-only Gaussian training for 3K depth-evaluator comparison against the R8 queue result.",
  "scene": "gcp_3000_20260602",
  "python": "$PYTHON",
  "training_repo": "$TRAIN_REPO",
  "source": "$SOURCE",
  "model": "$MODEL",
  "iterations": $ITERATIONS,
  "resolution": $RESOLUTION,
  "data_device": "cpu",
  "modality_kind": "rgb",
  "notes": [
    "No band training.",
    "No checkpoint mutation of the R8 queue.",
    "Reuses the COLMAP image_undistorter source generated from sparse_aligned/0."
  ]
}
JSON

STATUS="$RUN_ROOT/status/gcp_3000_20260602.status"
LOG="$RUN_ROOT/logs/gcp_3000_20260602.train.log"
TIMING="$RUN_ROOT/logs/gcp_3000_20260602.timing.txt"
echo "START gcp_3000_20260602 $(TZ=Asia/Shanghai date -Is)" | tee "$STATUS"
start=$(date +%s)
set +e
(
  cd "$TRAIN_REPO"
  "$PYTHON" train.py \
    -s "$SOURCE" \
    --images images \
    -m "$MODEL" \
    -r "$RESOLUTION" \
    --iterations "$ITERATIONS" \
    --checkpoint_iterations "$ITERATIONS" \
    --save_iterations "$ITERATIONS" \
    --test_iterations "$ITERATIONS" \
    --disable_viewer \
    --data_device cpu \
    --modality_kind rgb \
    --ss_enable false \
    --ss_prune_before_thermal false \
    --ss_prune_after_rgb false \
    --clamp_scale_after_densify false \
    --clamp_scale_after_rgb_final false \
    --thermal_reset_features false \
    --t_struct_grad_w 0.0 \
    --sgf_disable false \
    --baseline_modules_off false \
    --baseline_restore_ssp false \
    --baseline_restore_stt false
) > "$LOG" 2>&1
rc=$?
set -e
end=$(date +%s)
{
  echo "start_epoch=$start"
  echo "end_epoch=$end"
  echo "elapsed_seconds=$((end-start))"
  echo "rc=$rc"
} > "$TIMING"
if [[ "$rc" -eq 0 ]]; then
  echo "DONE gcp_3000_20260602 $(TZ=Asia/Shanghai date -Is)" | tee "$STATUS"
else
  echo "FAILED gcp_3000_20260602 rc=$rc $(TZ=Asia/Shanghai date -Is)" | tee "$STATUS"
fi
exit "$rc"
