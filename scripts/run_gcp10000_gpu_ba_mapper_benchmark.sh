#!/usr/bin/env bash
set -euo pipefail

COLMAP=${COLMAP:-/root/autodl-tmp/opt/ms-gcp-3dgs/colmap-4.0.4-gpu-ba/bin/colmap}
BASELINE=${BASELINE:-/root/autodl-tmp/runs/ms-gcp-3dgs/legacy-colmap-3.9.1/gcp_10000_20260610}
RUN_ROOT=${RUN_ROOT:-/root/autodl-tmp/runs/ms-gcp-3dgs/gpu-ba-feasibility-20260615}
GPU_ID=${GPU_ID:-0}
RUN="$RUN_ROOT/gcp_10000_20260610"
RGB="$RUN/RGB"
LOGS="$RUN/logs"
OUT="$RGB/distorted/sparse"

mkdir -p "$RGB/distorted" "$LOGS" "$OUT"
rm -rf "$OUT"/*

if [[ ! -f "$BASELINE/RGB/distorted/database.db" ]]; then
  echo "Missing baseline database: $BASELINE/RGB/distorted/database.db" >&2
  exit 2
fi
if [[ ! -d "$BASELINE/RGB/input" ]]; then
  echo "Missing baseline RGB input: $BASELINE/RGB/input" >&2
  exit 2
fi

cp --reflink=auto "$BASELINE/RGB/distorted/database.db" \
  "$RGB/distorted/database.db"
ln -sfn "$BASELINE/RGB/input" "$RGB/input"

cat > "$RUN/run_manifest.txt" <<EOF
purpose=COLMAP 4.0.4 GPU-BA mapper feasibility
scene=gcp_10000_20260610
baseline=$BASELINE
colmap=$COLMAP
started_at=$(date --iso-8601=seconds)
mapper_args=Mapper.ba_use_gpu=1,Mapper.ba_gpu_index=$GPU_ID
EOF

(
  echo "timestamp,index,memory.used,memory.total,utilization.gpu,power.draw"
  while true; do
    nvidia-smi \
      --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,power.draw \
      --format=csv,noheader,nounits || true
    sleep 5
  done
) > "$LOGS/gpu_trace.csv" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT

set +e
START_EPOCH=$(date +%s)
env CUDA_VISIBLE_DEVICES="$GPU_ID" "$COLMAP" mapper \
  --database_path "$RGB/distorted/database.db" \
  --image_path "$RGB/input" \
  --output_path "$OUT" \
  --Mapper.multiple_models 1 \
  --Mapper.min_model_size 10 \
  --Mapper.init_min_num_inliers 100 \
  --Mapper.abs_pose_min_num_inliers 30 \
  --Mapper.ba_use_gpu 1 \
  --Mapper.ba_gpu_index "$GPU_ID" \
  > "$LOGS/mapper.log" 2>&1
RC=$?
END_EPOCH=$(date +%s)
set -e

{
  echo "start_epoch=$START_EPOCH"
  echo "end_epoch=$END_EPOCH"
  echo "elapsed_seconds=$((END_EPOCH - START_EPOCH))"
  echo "mapper_rc=$RC"
} > "$LOGS/time.txt"

kill "$MONITOR_PID" 2>/dev/null || true
wait "$MONITOR_PID" 2>/dev/null || true
trap - EXIT

echo "mapper_rc=$RC" >> "$RUN/run_manifest.txt"
echo "finished_at=$(date --iso-8601=seconds)" >> "$RUN/run_manifest.txt"
exit "$RC"
