#!/usr/bin/env bash
set -euo pipefail
REPO=${REPO:-/root/autodl-tmp/MS-GCP-3DGS}
PY=${PY:-/root/autodl-tmp/envs/ms-gcp-3dgs/bin/python}
COLMAP=${COLMAP:-/root/autodl-tmp/opt/ms-gcp-3dgs/colmap-4.0.4-gpu-ba/bin/colmap}
DATA=${DATA:-/root/autodl-tmp/datasets/M3M-GCP/scenes_rgb_20260615}
RUN_ROOT=${RUN_ROOT:-/root/autodl-tmp/runs/ms-gcp-3dgs/colmap-4.0.4-gpu-ba-formal-20260615}
GPU_ID=${GPU_ID:-0}
MAPPER_ARGS=${MAPPER_ARGS:---Mapper.ba_use_gpu 1 --Mapper.ba_gpu_index 0}
SCENE=${1:?scene id required}
mkdir -p "$RUN_ROOT/status" "$RUN_ROOT/logs" "$RUN_ROOT/summary"
event(){ echo "[$(date --iso-8601=seconds)] $*" | tee -a "$RUN_ROOT/status/events.log"; }
sample_gpu(){ local out_csv="$1"; echo "timestamp,index,memory.used,memory.total,utilization.gpu,power.draw" > "$out_csv"; while true; do nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits >> "$out_csv" 2>/dev/null || true; sleep 10; done; }
run_stage(){ local scene_id="$1"; local stage="$2"; shift 2; local scene_dir="$RUN_ROOT/$scene_id"; mkdir -p "$scene_dir/logs"; local log="$scene_dir/logs/${stage}.log"; local trace="$scene_dir/logs/${stage}_gpu_trace.csv"; local timing="$scene_dir/logs/${stage}_timing.txt"; local start_epoch end_epoch; event "START $scene_id $stage"; start_epoch=$(date +%s); sample_gpu "$trace" & local mon=$!; set +e; CUDA_VISIBLE_DEVICES=$GPU_ID PYTHONUNBUFFERED=1 SIGS_COLMAP_EXECUTABLE="$COLMAP" "$@" > "$log" 2>&1; local rc=$?; set -e; end_epoch=$(date +%s); kill "$mon" 2>/dev/null || true; wait "$mon" 2>/dev/null || true; printf 'start_epoch=%s\nend_epoch=%s\nelapsed_seconds=%s\nrc=%s\n' "$start_epoch" "$end_epoch" "$((end_epoch-start_epoch))" "$rc" > "$timing"; event "END $scene_id $stage rc=$rc elapsed_seconds=$((end_epoch-start_epoch))"; return "$rc"; }
materialize_rgb_input(){ local scene_id="$1"; local raw_dir="$DATA/$scene_id"; local rgb_root="$RUN_ROOT/$scene_id/RGB"; local input_dir="$rgb_root/input"; if [ ! -d "$raw_dir" ]; then event "MISSING_RAW $scene_id $raw_dir"; return 1; fi; rm -rf "$rgb_root"; mkdir -p "$input_dir"; find "$raw_dir" -maxdepth 1 -type f -name '*_D.JPG' -print0 | sort -z | while IFS= read -r -d '' f; do ln "$f" "$input_dir/$(basename "$f")" 2>/dev/null || cp "$f" "$input_dir/$(basename "$f")"; done; find "$input_dir" -maxdepth 1 -type f -name '*_D.JPG' | wc -l > "$RUN_ROOT/$scene_id/input_rgb_count.txt"; }
clean_colmap_outputs(){ local rgb_root="$1"; rm -rf "$rgb_root/distorted" "$rgb_root/images" "$rgb_root/sparse"; }
write_quality_summary(){ local scene_id="$1"; local rgb_root="$2"; local matching_used="$3"; local out_json="$4"; "$PY" - "$scene_id" "$rgb_root" "$matching_used" "$out_json" <<'PY'
import json, struct, sys
from pathlib import Path
scene_id, rgb_root, matching_used, out_json = sys.argv[1:5]
rgb_root=Path(rgb_root); out_json=Path(out_json)
input_dir=rgb_root/'input'; images_txt=rgb_root/'sparse'/'0'/'images.txt'; points_txt=rgb_root/'sparse'/'0'/'points3D.txt'
expected=sorted(p.name for p in input_dir.glob('*_D.JPG'))
def names_from_images_txt(path):
    if not path.exists(): return []
    lines=[l.strip() for l in path.read_text(encoding='utf-8',errors='replace').splitlines() if l.strip() and not l.startswith('#')]
    out=[]
    for i in range(0,len(lines),2):
        parts=lines[i].split()
        if len(parts)>=10: out.append(Path(parts[9]).name)
    return sorted(set(out))
def count_images_bin(path):
    if not path.exists(): return None
    with path.open('rb') as f: return int(struct.unpack('<Q', f.read(8))[0])
def count_points_txt(path):
    if not path.exists(): return None
    return sum(1 for line in path.read_text(encoding='utf-8',errors='replace').splitlines() if line.strip() and not line.startswith('#'))
registered=names_from_images_txt(images_txt); missing=sorted(set(expected)-set(registered))
models=[]; distorted_sparse=rgb_root/'distorted'/'sparse'
if distorted_sparse.exists():
    for d in sorted([p for p in distorted_sparse.iterdir() if p.is_dir() and p.name.isdigit()], key=lambda p:int(p.name)):
        models.append({'model_id':int(d.name),'image_count':count_images_bin(d/'images.bin'),'images_bin':str(d/'images.bin')})
georeg_json=images_txt.parent/'georegistration_alignment_summary.json'; georeg=None
if georeg_json.exists():
    try: georeg=json.loads(georeg_json.read_text(encoding='utf-8'))
    except Exception as exc: georeg={'parse_error':str(exc),'path':str(georeg_json)}
payload={'scene_id':scene_id,'protocol':'benchmark COLMAP protocol: spatial first, exhaustive only if spatial command fails; RGB-only input for COLMAP quality check','matching_used':matching_used,'input_rgb_count':len(expected),'registered_count':len(registered),'registration_rate':(len(registered)/len(expected)) if expected else None,'missing_count':len(missing),'missing_images':missing,'selected_sparse0_images_txt':str(images_txt),'selected_sparse0_points3d_count':count_points_txt(points_txt),'distorted_sparse_model_count':len(models),'distorted_sparse_largest_model_image_count':max([m['image_count'] or 0 for m in models], default=0),'distorted_sparse_models':models,'georegistration_summary':georeg}
out_json.parent.mkdir(parents=True, exist_ok=True); out_json.write_text(json.dumps(payload,indent=2),encoding='utf-8'); print(json.dumps(payload,ensure_ascii=False))
PY
}
run_colmap_scene(){ local scene_id="$1"; local scene_dir="$RUN_ROOT/$scene_id"; local rgb_root="$scene_dir/RGB"; mkdir -p "$scene_dir/logs" "$scene_dir/summary"; materialize_rgb_input "$scene_id"; printf '%s\n' "$MAPPER_ARGS" > "$scene_dir/mapper_args.txt"; { echo "scene=$scene_id"; echo "started_at=$(date --iso-8601=seconds)"; echo "repo_head=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"; echo "colmap=$COLMAP"; "$COLMAP" -h 2>&1 | head -1 || true; echo "mapper_args=$MAPPER_ARGS"; } > "$scene_dir/run_manifest.txt"; local matching_used="spatial"; if ! run_stage "$scene_id" colmap_spatial_gpu "$PY" "$REPO/code/colmap/prepare_scene_colmap.py" -s "$rgb_root" --colmap_executable "$COLMAP" --exiftool_executable exiftool --camera SIMPLE_RADIAL --matching spatial --matcher_args "--SpatialMatching.max_num_neighbors=80 --SpatialMatching.max_distance=500" --mapper_args "$MAPPER_ARGS" --sift_num_threads -1 --sift_max_image_size 3200 --sift_max_num_features 8192 --sift_matching_max_num_matches 32768 --sift_use_gpu 1 --sift_matching_use_gpu 1 --prior_position_std_m 1.0 --georegistration_mode auto --georegistration_backend custom_sim3; then echo colmap_spatial_failed_retry_exhaustive > "$scene_dir/WARN_SPATIAL_FAILED"; clean_colmap_outputs "$rgb_root"; matching_used="exhaustive"; if ! run_stage "$scene_id" colmap_exhaustive_gpu "$PY" "$REPO/code/colmap/prepare_scene_colmap.py" -s "$rgb_root" --colmap_executable "$COLMAP" --exiftool_executable exiftool --camera SIMPLE_RADIAL --matching exhaustive --mapper_args "$MAPPER_ARGS" --sift_num_threads -1 --sift_max_image_size 3200 --sift_max_num_features 8192 --sift_matching_max_num_matches 32768 --sift_use_gpu 1 --sift_matching_use_gpu 1 --prior_position_std_m 1.0 --georegistration_mode auto --georegistration_backend custom_sim3; then echo colmap_failed > "$scene_dir/FAILED_COLMAP"; echo "finished_at=$(date --iso-8601=seconds)" >> "$scene_dir/run_manifest.txt"; echo "status=failed" >> "$scene_dir/run_manifest.txt"; return 1; fi; fi; write_quality_summary "$scene_id" "$rgb_root" "$matching_used" "$scene_dir/summary/colmap_quality_summary.json" > "$scene_dir/logs/colmap_quality_summary.log" 2>&1 || true; cp "$scene_dir/summary/colmap_quality_summary.json" "$RUN_ROOT/summary/${scene_id}_colmap_quality_summary.json" 2>/dev/null || true; echo "finished_at=$(date --iso-8601=seconds)" >> "$scene_dir/run_manifest.txt"; echo "status=done" >> "$scene_dir/run_manifest.txt"; }
event "REQUEST $SCENE"
run_colmap_scene "$SCENE"
event "DONE $SCENE"
