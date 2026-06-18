#!/usr/bin/env bash
set -euo pipefail

# RGB-only Gaussian training queue for the six MS-GCP-3DGS scenes.
# This script is intended to run on the AutoDL backup server.
# It uses the first-paper Gaussian training implementation as an external
# renderer/trainer, while keeping all second-paper inputs and outputs under
# /root/autodl-tmp/runs/ms-gcp-3dgs.

PYTHON=${PYTHON:-/root/autodl-tmp/envs/spectralindexgs_bw/bin/python}
TRAIN_REPO=${TRAIN_REPO:-/root/autodl-tmp/Multispectral}
COLMAP_BIN=${COLMAP_BIN:-/root/autodl-tmp/opt/ms-gcp-3dgs/colmap-4.0.4-gpu-ba/bin/colmap}
COLMAP_ROOT=${COLMAP_ROOT:-/root/autodl-tmp/runs/ms-gcp-3dgs/colmap-4.0.4-global-formal-20260616}
RUN_ROOT=${RUN_ROOT:-/root/autodl-tmp/runs/ms-gcp-3dgs/gaussian-rgb-full6-r8-30k-20260618_1028}
ITERATIONS=${ITERATIONS:-30000}
RESOLUTION=${RESOLUTION:-8}

SCENES=(
  gcp_3000_20260602
  gcp_5000_20260602
  gcp_10000_20260610
  gcp_20000_20260602
  gcp_50000_20260610
  gcp_100000_20260610
)

mkdir -p "$RUN_ROOT"/{sources,models,logs,status,summary}

cat > "$RUN_ROOT/run_manifest.json" <<JSON
{
  "schema": "ms_gcp_rgb_gaussian_full6_queue_v1",
  "created_at_cst": "$(TZ=Asia/Shanghai date -Is)",
  "purpose": "RGB-only Gaussian training for depth-map based GCP geometry evaluator; no band training.",
  "python": "$PYTHON",
  "training_repo": "$TRAIN_REPO",
  "colmap_bin": "$COLMAP_BIN",
  "colmap_root": "$COLMAP_ROOT",
  "run_root": "$RUN_ROOT",
  "iterations": $ITERATIONS,
  "resolution": $RESOLUTION,
  "data_device": "cpu",
  "source_sparse": "RGB/sparse_aligned/0",
  "image_dir": "RGB/input",
  "training_source": "per-scene COLMAP image_undistorter output from sparse_aligned/0",
  "scenes": [
    "gcp_3000_20260602",
    "gcp_5000_20260602",
    "gcp_10000_20260610",
    "gcp_20000_20260602",
    "gcp_50000_20260610",
    "gcp_100000_20260610"
  ]
}
JSON

prepare_source() {
  local scene="$1"
  local src="$RUN_ROOT/sources/$scene"
  local col="$COLMAP_ROOT/$scene/RGB"
  local sparse="$col/sparse_aligned/0"
  local images="$col/input"
  local undistort_log="$RUN_ROOT/logs/${scene}.undistort.log"

  [[ -d "$images" ]] || { echo "missing images: $images" >&2; return 2; }
  [[ -f "$sparse/cameras.bin" && -f "$sparse/images.bin" && -f "$sparse/points3D.bin" ]] || {
    echo "missing sparse_aligned bin files: $sparse" >&2
    return 2
  }

  if [[ -f "$src/.undistort_done" ]]; then
    return 0
  fi

  rm -rf "$src"
  mkdir -p "$src"
  "$COLMAP_BIN" image_undistorter \
    --image_path "$images" \
    --input_path "$sparse" \
    --output_path "$src" \
    --output_type COLMAP > "$undistort_log" 2>&1

  if [[ -f "$src/sparse/cameras.bin" && ! -d "$src/sparse/0" ]]; then
    mkdir -p "$src/sparse/0"
    find "$src/sparse" -maxdepth 1 -type f \( -name '*.bin' -o -name '*.txt' \) -exec mv {} "$src/sparse/0/" \;
  fi

  [[ -d "$src/images" ]] || { echo "image_undistorter did not create images: $src/images" >&2; return 3; }
  [[ -f "$src/sparse/0/cameras.bin" && -f "$src/sparse/0/images.bin" && -f "$src/sparse/0/points3D.bin" ]] || {
    echo "image_undistorter did not create sparse/0 bin files: $src/sparse/0" >&2
    return 3
  }

  find "$images" -maxdepth 1 -type f -iname '*.jpg' | wc -l > "$src/input_rgb_count.txt"
  cp -f "$sparse/georegistration_alignment_summary.json" "$src/sparse/0/" 2>/dev/null || true
  touch "$src/.undistort_done"
}

run_scene() {
  local scene="$1"
  local src="$RUN_ROOT/sources/$scene"
  local model="$RUN_ROOT/models/$scene/Model_RGB"
  local log="$RUN_ROOT/logs/${scene}.train.log"
  local timing="$RUN_ROOT/logs/${scene}.timing.txt"
  local status="$RUN_ROOT/status/${scene}.status"

  echo "START $scene $(TZ=Asia/Shanghai date -Is)" | tee "$status"
  prepare_source "$scene"

  local start
  start=$(date +%s)
  set +e
  (
    cd "$TRAIN_REPO"
    "$PYTHON" train.py \
      -s "$src" \
      --images images \
      -m "$model" \
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
  ) > "$log" 2>&1
  local rc=$?
  set -e
  local end
  end=$(date +%s)
  {
    echo "start_epoch=$start"
    echo "end_epoch=$end"
    echo "elapsed_seconds=$((end-start))"
    echo "rc=$rc"
  } > "$timing"

  if [[ "$rc" -eq 0 ]]; then
    echo "DONE $scene $(TZ=Asia/Shanghai date -Is)" | tee "$status"
  else
    echo "FAILED $scene rc=$rc $(TZ=Asia/Shanghai date -Is)" | tee "$status"
    return "$rc"
  fi
}

echo "QUEUE_START $(TZ=Asia/Shanghai date -Is)" | tee "$RUN_ROOT/status/queue.status"
for scene in "${SCENES[@]}"; do
  run_scene "$scene"
done
echo "QUEUE_DONE $(TZ=Asia/Shanghai date -Is)" | tee "$RUN_ROOT/status/queue.status"
