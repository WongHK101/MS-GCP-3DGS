#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "Usage: $0 COLMAP DATABASE IMAGE_PATH OUTPUT_ROOT SCENE_ID [GPU_INDEX]" >&2
  exit 2
fi

COLMAP="$1"
SOURCE_DATABASE="$2"
IMAGE_PATH="$3"
OUTPUT_ROOT="$4"
SCENE_ID="$5"
GPU_INDEX="${6:-0}"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${OUTPUT_ROOT}/${SCENE_ID}_global_${STAMP}"
DATABASE="${RUN_ROOT}/database.db"
SPARSE="${RUN_ROOT}/sparse"
LOG="${RUN_ROOT}/global_mapper.log"
MANIFEST="${RUN_ROOT}/run_manifest.json"

mkdir -p "$RUN_ROOT" "$SPARSE"
cp --reflink=auto "$SOURCE_DATABASE" "$DATABASE"

python3 - "$MANIFEST" "$COLMAP" "$SOURCE_DATABASE" "$DATABASE" "$IMAGE_PATH" "$SPARSE" "$GPU_INDEX" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

manifest, colmap, source_db, copied_db, image_path, sparse, gpu = sys.argv[1:]
Path(manifest).write_text(
    json.dumps(
        {
            "schema": "m3m_gcp_global_mapper_run_v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "colmap": colmap,
            "source_database": source_db,
            "copied_database": copied_db,
            "image_path": image_path,
            "output_sparse": sparse,
            "gpu_index": int(gpu),
            "source_database_is_preserved": True,
            "feature_extraction_reused": True,
            "matching_reused": True,
        },
        indent=2,
    ),
    encoding="utf-8",
)
PY

echo "RUN_ROOT=$RUN_ROOT"
"$COLMAP" global_mapper \
  --database_path "$DATABASE" \
  --image_path "$IMAGE_PATH" \
  --output_path "$SPARSE" \
  --GlobalMapper.gp_use_gpu 1 \
  --GlobalMapper.gp_gpu_index "$GPU_INDEX" \
  --GlobalMapper.ba_ceres_use_gpu 1 \
  --GlobalMapper.ba_ceres_gpu_index "$GPU_INDEX" \
  2>&1 | tee "$LOG"
