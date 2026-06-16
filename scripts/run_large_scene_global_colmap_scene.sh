#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/root/autodl-tmp/MS-GCP-3DGS}
PY=${PY:-/root/autodl-tmp/envs/ms-gcp-3dgs/bin/python}
COLMAP=${COLMAP:-/root/autodl-tmp/opt/ms-gcp-3dgs/colmap-4.0.4-gpu-ba/bin/colmap}
DATA=${DATA:-/root/autodl-tmp/datasets/M3M-GCP/scenes_rgb_20260615}
RUN_ROOT=${RUN_ROOT:-/root/autodl-tmp/runs/ms-gcp-3dgs/colmap-4.0.4-global-formal-20260616}
GPU_ID=${GPU_ID:-0}
SCENE=${1:?scene id required}

SCENE_ROOT="$RUN_ROOT/$SCENE"
RGB="$SCENE_ROOT/RGB"
INPUT="$RGB/input"
LOGS="$SCENE_ROOT/logs"
SUMMARY="$SCENE_ROOT/summary"
GLOBAL="$SCENE_ROOT/global_mapper"
GLOBAL_DB="$GLOBAL/database.db"
GLOBAL_SPARSE="$GLOBAL/sparse"
FINAL_SPARSE="$RGB/sparse/0"
FINAL_ALIGNED="$RGB/sparse_aligned/0"

mkdir -p "$RUN_ROOT/status" "$RUN_ROOT/summary" "$LOGS" "$SUMMARY"

event() {
  echo "[$(date --iso-8601=seconds)] $*" | tee -a "$RUN_ROOT/status/events.log"
}

sample_gpu() {
  local out_csv="$1"
  echo "timestamp,index,memory.used,memory.total,utilization.gpu,power.draw" > "$out_csv"
  while true; do
    nvidia-smi \
      --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,power.draw \
      --format=csv,noheader,nounits >> "$out_csv" 2>/dev/null || true
    sleep 10
  done
}

run_stage() {
  local stage="$1"
  shift
  local log="$LOGS/${stage}.log"
  local trace="$LOGS/${stage}_gpu_trace.csv"
  local timing="$LOGS/${stage}_timing.txt"
  local start_epoch end_epoch rc
  event "START $SCENE $stage"
  start_epoch=$(date +%s)
  sample_gpu "$trace" &
  local mon=$!
  set +e
  CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONUNBUFFERED=1 SIGS_COLMAP_EXECUTABLE="$COLMAP" "$@" > "$log" 2>&1
  rc=$?
  set -e
  end_epoch=$(date +%s)
  kill "$mon" 2>/dev/null || true
  wait "$mon" 2>/dev/null || true
  printf 'start_epoch=%s\nend_epoch=%s\nelapsed_seconds=%s\nrc=%s\n' \
    "$start_epoch" "$end_epoch" "$((end_epoch-start_epoch))" "$rc" > "$timing"
  event "END $SCENE $stage rc=$rc elapsed_seconds=$((end_epoch-start_epoch))"
  return "$rc"
}

materialize_rgb_input() {
  local raw="$DATA/$SCENE"
  if [[ ! -d "$raw" ]]; then
    echo "Missing scene data: $raw" >&2
    exit 2
  fi
  rm -rf "$RGB"
  mkdir -p "$INPUT"
  find "$raw" -maxdepth 1 -type f -name '*_D.JPG' -print0 | sort -z |
    while IFS= read -r -d '' f; do
      ln "$f" "$INPUT/$(basename "$f")" 2>/dev/null || cp "$f" "$INPUT/$(basename "$f")"
    done
  find "$INPUT" -maxdepth 1 -type f -name '*_D.JPG' | wc -l > "$SCENE_ROOT/input_rgb_count.txt"
}

write_manifest() {
  "$PY" - "$SCENE_ROOT/run_manifest.json" "$REPO" "$COLMAP" "$DATA/$SCENE" "$SCENE_ROOT" "$GPU_ID" <<'PY'
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

manifest, repo, colmap, data, scene_root, gpu = sys.argv[1:]
try:
    head = subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip()
except Exception:
    head = "unknown"
try:
    colmap_head = subprocess.check_output([colmap, "-h"], text=True, stderr=subprocess.STDOUT).splitlines()[0]
except Exception as exc:
    colmap_head = f"unavailable: {exc}"
payload = {
    "schema": "ms_gcp_large_scene_global_colmap_v1",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "scene": Path(data).name,
    "repo": repo,
    "repo_head": head,
    "colmap": colmap,
    "colmap_help_header": colmap_head,
    "data_dir": data,
    "scene_root": scene_root,
    "gpu_id": int(gpu),
    "protocol": {
        "input": "RGB-only DJI _D.JPG images",
        "camera_model": "SIMPLE_RADIAL",
        "feature_extraction": "COLMAP feature_extractor, SIFT GPU enabled",
        "matching": "COLMAP spatial_matcher with max_num_neighbors=80 and max_distance=500",
        "mapper": "COLMAP 4.0.4 global_mapper with GPU graph partitioning and GPU Ceres BA",
        "fov_aware_matching": False,
        "incremental_mapper": False,
    },
}
Path(manifest).write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
}

run_global_mapper() {
  rm -rf "$GLOBAL"
  mkdir -p "$GLOBAL_SPARSE"
  cp --reflink=auto "$RGB/distorted/database.db" "$GLOBAL_DB"
  run_stage global_mapper "$COLMAP" global_mapper \
    --database_path "$GLOBAL_DB" \
    --image_path "$INPUT" \
    --output_path "$GLOBAL_SPARSE" \
    --GlobalMapper.gp_use_gpu 1 \
    --GlobalMapper.gp_gpu_index "$GPU_ID" \
    --GlobalMapper.ba_ceres_use_gpu 1 \
    --GlobalMapper.ba_ceres_gpu_index "$GPU_ID"
}

summarize_and_align() {
  "$PY" - "$REPO" "$COLMAP" "$SCENE" "$RGB" "$GLOBAL_SPARSE" "$FINAL_SPARSE" "$FINAL_ALIGNED" "$SUMMARY/colmap_global_quality_summary.json" <<'PY'
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

repo, colmap, scene, rgb, global_sparse, final_sparse, final_aligned, out_json = sys.argv[1:]
repo = Path(repo)
rgb = Path(rgb)
global_sparse = Path(global_sparse)
final_sparse = Path(final_sparse)
final_aligned = Path(final_aligned)
out_json = Path(out_json)
sys.path.insert(0, str(repo / "code" / "colmap"))
sys.path.insert(0, str(repo / "code" / "colmap" / "utils"))

from prepare_scene_colmap import (  # noqa: E402
    custom_sim3_georegister_model,
    export_model_as_txt,
    populate_pose_priors_from_exif,
)
from read_write_model import read_model  # noqa: E402

def stats_for_model(model_dir: Path) -> dict:
    cameras, images, points = read_model(str(model_dir))
    observations = sum(int((image.point3D_ids >= 0).sum()) for image in images.values())
    point_errors = [float(point.error) for point in points.values()]
    track_lengths = [len(point.image_ids) for point in points.values()]
    return {
        "model_path": str(model_dir),
        "camera_count": len(cameras),
        "registered_image_count": len(images),
        "point3D_count": len(points),
        "observation_count": observations,
        "mean_reprojection_error_px": sum(point_errors) / len(point_errors) if point_errors else None,
        "mean_track_length": sum(track_lengths) / len(track_lengths) if track_lengths else None,
    }

candidates = []
for d in sorted(global_sparse.iterdir()):
    if not d.is_dir():
        continue
    try:
        stats = stats_for_model(d)
    except Exception:
        continue
    candidates.append(stats)
if not candidates:
    raise RuntimeError(f"No readable global_mapper sparse models under {global_sparse}")

selected = max(candidates, key=lambda row: (row["registered_image_count"], row["point3D_count"]))
selected_path = Path(selected["model_path"])
if final_sparse.parent.exists():
    shutil.rmtree(final_sparse.parent)
final_sparse.parent.mkdir(parents=True, exist_ok=True)
shutil.copytree(selected_path, final_sparse)
export_model_as_txt(colmap, final_sparse)
final_stats = stats_for_model(final_sparse)

gps_map = populate_pose_priors_from_exif(
    rgb / "distorted" / "database.db",
    rgb / "input",
    exiftool_exe="exiftool",
    wgs84_code=0,
    prior_position_std_m=1.0,
    swap_latlon=False,
)
alignment = None
if len(gps_map) >= 3:
    if final_aligned.parent.exists():
        shutil.rmtree(final_aligned.parent)
    final_aligned.parent.mkdir(parents=True, exist_ok=True)
    alignment = custom_sim3_georegister_model(final_sparse, final_aligned, gps_map)
    export_model_as_txt(colmap, final_aligned)

input_count = len(list((rgb / "input").glob("*_D.JPG")))
payload = {
    "schema": "ms_gcp_large_scene_global_colmap_quality_v1",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "scene": scene,
    "input_rgb_count": input_count,
    "registered_count": final_stats["registered_image_count"],
    "registration_rate": final_stats["registered_image_count"] / input_count if input_count else None,
    "selected_global_model": selected,
    "global_model_count": len(candidates),
    "global_models": candidates,
    "final_sparse": final_stats,
    "pose_priors_from_exif_count": len(gps_map),
    "alignment": alignment,
    "protocol": "COLMAP spatial matcher database followed by COLMAP 4.0.4 global_mapper; no FOV-aware pair list.",
}
out_json.parent.mkdir(parents=True, exist_ok=True)
out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False))
PY
  cp "$SUMMARY/colmap_global_quality_summary.json" "$RUN_ROOT/summary/${SCENE}_colmap_global_quality_summary.json"
}

event "REQUEST $SCENE"
mkdir -p "$SCENE_ROOT"
materialize_rgb_input
write_manifest

run_stage prepare_spatial_database "$PY" "$REPO/code/colmap/prepare_scene_colmap.py" \
  -s "$RGB" \
  --colmap_executable "$COLMAP" \
  --exiftool_executable exiftool \
  --camera SIMPLE_RADIAL \
  --matching spatial \
  --matcher_args "--SpatialMatching.max_num_neighbors=80 --SpatialMatching.max_distance=500" \
  --sift_num_threads -1 \
  --sift_max_image_size 3200 \
  --sift_max_num_features 8192 \
  --sift_matching_max_num_matches 32768 \
  --sift_use_gpu 1 \
  --sift_matching_use_gpu 1 \
  --prior_position_std_m 1.0 \
  --georegistration_mode off \
  --stop_after_matching

run_global_mapper
summarize_and_align

event "DONE $SCENE"
