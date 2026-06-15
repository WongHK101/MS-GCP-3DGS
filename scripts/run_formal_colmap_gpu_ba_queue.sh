#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/root/autodl-tmp/MS-GCP-3DGS}
RUN_ROOT=${RUN_ROOT:-/root/autodl-tmp/runs/ms-gcp-3dgs/colmap-4.0.4-gpu-ba-formal-20260615}
DATA=${DATA:-/root/autodl-tmp/datasets/M3M-GCP/scenes_rgb_20260615}
GPU_ID=${GPU_ID:-0}
MIN_FREE_GB=${MIN_FREE_GB:-30}

SCENES=(
  gcp_3000_20260602
  gcp_5000_20260602
  gcp_20000_20260602
  gcp_50000_20260610
  gcp_100000_20260610
)

mkdir -p "$RUN_ROOT/status" "$RUN_ROOT/logs"
exec > >(tee -a "$RUN_ROOT/logs/queue.log") 2>&1

echo "queue_started_at=$(date --iso-8601=seconds)"
echo "repo_head=$(git -C "$REPO" rev-parse HEAD)"
echo "scenes=${SCENES[*]}"

overall_rc=0
for scene in "${SCENES[@]}"; do
  free_kb=$(df -Pk /root/autodl-tmp | awk 'NR==2 {print $4}')
  free_gb=$((free_kb / 1024 / 1024))
  if (( free_gb < MIN_FREE_GB )); then
    echo "[$(date --iso-8601=seconds)] BLOCKED_DISK scene=$scene free_gb=$free_gb min_free_gb=$MIN_FREE_GB"
    overall_rc=3
    break
  fi
  if [[ ! -d "$DATA/$scene" ]]; then
    echo "[$(date --iso-8601=seconds)] MISSING_DATA scene=$scene"
    overall_rc=2
    continue
  fi
  echo "[$(date --iso-8601=seconds)] QUEUE_START scene=$scene free_gb=$free_gb"
  set +e
  REPO="$REPO" RUN_ROOT="$RUN_ROOT" DATA="$DATA" GPU_ID="$GPU_ID" \
    bash "$REPO/code/launchers/run_m3m_gcp_colmap_scene.sh" "$scene"
  rc=$?
  set -e
  echo "[$(date --iso-8601=seconds)] QUEUE_END scene=$scene rc=$rc"
  if (( rc != 0 )); then
    overall_rc=$rc
  fi
done

echo "queue_finished_at=$(date --iso-8601=seconds)"
echo "queue_rc=$overall_rc"
exit "$overall_rc"
