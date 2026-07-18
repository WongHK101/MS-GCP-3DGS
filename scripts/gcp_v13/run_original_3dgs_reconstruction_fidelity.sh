#!/usr/bin/env bash
set -euo pipefail

: "${SCENE_ID:?Set SCENE_ID}"
: "${SOURCE_RUN_ROOT:?Set the completed original-3DGS run root}"
: "${FIDELITY_RUN_ID:?Set a unique FIDELITY_RUN_ID}"
: "${EXPECTED_VIEW_COUNT:?Set the frozen train-view count}"

UPSTREAM_COMMIT=2eee0e26d2d5fd00ec462df47752223952f6bf4e
UPSTREAM_TREE=5eee127dfc0942bf83d9fdd72328e03ec0cbf6c4
RELEASE_DIGEST=513f8999fe4b110f15bcbecad7932895781cee755ee9ccd7a14ff10298546d75
RENDER_SHA256=3d3949021947d31138ced30aac73771d7d050ac2229d630e3b0056cb95192b14
METRICS_SHA256=bda39191dde1fad93abf56a994d6f799bc02209b8ac5000b038e5ecf3345d6d3
CODE_ROOT=/root/autodl-tmp/worktrees/ms-gcp-v13/3dgs-original/$UPSTREAM_COMMIT/official-train
ENV_ROOT=/root/autodl-tmp/envs/ms-gcp-v13/3dgs-original/py310-torch2.7.1-cu128-v1
ORCH_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SOURCE_MODEL=$SOURCE_RUN_ROOT/02_checkpoints/model
RUN_ROOT=/root/autodl-tmp/runs/gs-gcp-v13/3dgs-original-fidelity/$SCENE_ID/$FIDELITY_RUN_ID
BUILD_ROOT=/root/autodl-tmp/build/gs-gcp-v13/3dgs-original-fidelity/$SCENE_ID/$FIDELITY_RUN_ID
MODEL_MIRROR=$RUN_ROOT/model_mirror
CHUNKS_ROOT=$RUN_ROOT/metrics_chunks
MERGED_ROOT=$RUN_ROOT/merged_metrics
RESOURCE_CONTRACT=$ORCH_ROOT/configs/gs_gcp_resource_probe_contract_v1.json
FIDELITY_CONTRACT=$ORCH_ROOT/configs/gs_gcp_reconstruction_fidelity_v1.json
PROBE=$ORCH_ROOT/code/gcp/run_with_resource_probe.py
FIDELITY_TOOL=$ORCH_ROOT/code/gcp/original_3dgs_reconstruction_fidelity.py
METRICS_RUNNER=$ORCH_ROOT/code/gcp/run_official_metrics_no_grad.py
GNU_TIME=/root/autodl-tmp/tools/gs-gcp-v13/gnu-time/ubuntu-jammy-time-1.9-v1/root/usr/bin/time

for path in "$RUN_ROOT" "$BUILD_ROOT"; do
  if test -e "$path"; then
    echo "refusing existing output root: $path" >&2
    exit 2
  fi
done

test -f "$SOURCE_MODEL/point_cloud/iteration_30000/point_cloud.ply"
test -f "$SOURCE_MODEL/cfg_args"
test -f "$SOURCE_MODEL/cameras.json"
test -z "$(git -C "$CODE_ROOT" status --porcelain)"
test -z "$(git -C "$ORCH_ROOT" status --porcelain)"
test "$(git -C "$CODE_ROOT" rev-parse HEAD)" = "$UPSTREAM_COMMIT"
test "$(git -C "$CODE_ROOT" rev-parse 'HEAD^{tree}')" = "$UPSTREAM_TREE"
test "$(sha256sum "$CODE_ROOT/render.py" | cut -d' ' -f1)" = "$RENDER_SHA256"
test "$(sha256sum "$CODE_ROOT/metrics.py" | cut -d' ' -f1)" = "$METRICS_SHA256"

mkdir -p "$BUILD_ROOT/tmp" "$RUN_ROOT"/{00_preflight,01_render,02_metrics,03_audit} "$MODEL_MIRROR"
ln -s "$SOURCE_MODEL/cfg_args" "$MODEL_MIRROR/cfg_args"
ln -s "$SOURCE_MODEL/cameras.json" "$MODEL_MIRROR/cameras.json"
ln -s "$SOURCE_MODEL/input.ply" "$MODEL_MIRROR/input.ply"
ln -s "$SOURCE_MODEL/point_cloud" "$MODEL_MIRROR/point_cloud"

export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$ENV_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export GIT_OPTIONAL_LOCKS=0
export TORCH_EXTENSIONS_DIR="$BUILD_ROOT/torch_extensions"
export TMPDIR="$BUILD_ROOT/tmp"
mkdir -p "$TORCH_EXTENSIONS_DIR"

cp "$FIDELITY_CONTRACT" "$RESOURCE_CONTRACT" "$RUN_ROOT/00_preflight/"
cp "${BASH_SOURCE[0]}" "$RUN_ROOT/03_audit/exact_launcher.sh"
git -C "$ORCH_ROOT" rev-parse HEAD > "$RUN_ROOT/00_preflight/orchestrator_commit.txt"
git -C "$CODE_ROOT" rev-parse HEAD > "$RUN_ROOT/00_preflight/upstream_commit.txt"
sha256sum "$CODE_ROOT/render.py" "$CODE_ROOT/metrics.py" "$FIDELITY_CONTRACT" \
  "$SOURCE_MODEL/point_cloud/iteration_30000/point_cloud.ply" \
  > "$RUN_ROOT/00_preflight/frozen_inputs.sha256"
printf '%s\n' "$SCENE_ID" > "$RUN_ROOT/00_preflight/scene.txt"
printf '%s\n' "$SOURCE_RUN_ROOT" > "$RUN_ROOT/00_preflight/source_run_root.txt"
printf '%s\n' "$EXPECTED_VIEW_COUNT" > "$RUN_ROOT/00_preflight/expected_view_count.txt"

python3 "$PROBE" \
  --contract "$RESOURCE_CONTRACT" \
  --phase render \
  --output_dir "$RUN_ROOT/01_render/resource_probe" \
  --working_directory "$CODE_ROOT" \
  --gpu_indices 0 \
  --time_binary "$GNU_TIME" \
  -- "$ENV_ROOT/bin/python" render.py \
    --model_path "$MODEL_MIRROR" \
    --iteration 30000 \
    --skip_test \
    --quiet

RENDERS=$MODEL_MIRROR/train/ours_30000/renders
GT=$MODEL_MIRROR/train/ours_30000/gt
test "$(find "$RENDERS" -maxdepth 1 -type f -name '*.png' | wc -l)" -eq "$EXPECTED_VIEW_COUNT"
test "$(find "$GT" -maxdepth 1 -type f -name '*.png' | wc -l)" -eq "$EXPECTED_VIEW_COUNT"

"$ENV_ROOT/bin/python" "$FIDELITY_TOOL" prepare \
  --renders_dir "$RENDERS" \
  --gt_dir "$GT" \
  --output_root "$CHUNKS_ROOT" \
  --expected_view_count "$EXPECTED_VIEW_COUNT" \
  --chunk_size 64 \
  --method_name ours_30000 \
  > "$RUN_ROOT/02_metrics/prepare_chunks_stdout.json"

mapfile -t CHUNK_PATHS < <("$ENV_ROOT/bin/python" - "$CHUNKS_ROOT/chunk_manifest.json" <<'PY'
import json, sys
for row in json.load(open(sys.argv[1], encoding="utf-8"))["chunks"]:
    print(row["model_path"])
PY
)
for chunk_path in "${CHUNK_PATHS[@]}"; do
  chunk_id=$(basename "$chunk_path")
  python3 "$PROBE" \
    --contract "$RESOURCE_CONTRACT" \
    --phase evaluation \
    --output_dir "$RUN_ROOT/02_metrics/${chunk_id}_resource_probe" \
    --working_directory "$CODE_ROOT" \
    --gpu_indices 0 \
    --time_binary "$GNU_TIME" \
    -- "$ENV_ROOT/bin/python" "$METRICS_RUNNER" \
      --metrics_script "$CODE_ROOT/metrics.py" \
      --expected_sha256 "$METRICS_SHA256" \
      --model_path "$chunk_path"
  test -s "$chunk_path/results.json"
  test -s "$chunk_path/per_view.json"
done

"$ENV_ROOT/bin/python" "$FIDELITY_TOOL" merge \
  --chunks_root "$CHUNKS_ROOT" \
  --output_dir "$MERGED_ROOT" \
  --expected_view_count "$EXPECTED_VIEW_COUNT" \
  --method_name ours_30000 \
  > "$RUN_ROOT/02_metrics/merge_stdout.json"

"$ENV_ROOT/bin/python" - "$SOURCE_MODEL/cameras.json" "$MERGED_ROOT/per_view.json" "$RUN_ROOT/02_metrics/render_index_to_image_name.json" "$EXPECTED_VIEW_COUNT" <<'PY'
import json, sys
cameras_path, metrics_path, output_path, expected = sys.argv[1:]
cameras = json.load(open(cameras_path, encoding="utf-8"))
metrics = json.load(open(metrics_path, encoding="utf-8"))["ours_30000"]
expected = int(expected)
if len(cameras) != expected:
    raise SystemExit(f"camera count mismatch: {len(cameras)} != {expected}")
names = sorted(metrics["PSNR"])
if names != [f"{index:05d}.png" for index in range(expected)]:
    raise SystemExit("render index set is not contiguous")
rows = []
for index, camera in enumerate(cameras):
    rendered_name = f"{index:05d}.png"
    rows.append({
        "rendered_name": rendered_name,
        "source_image_name": camera["img_name"],
        "PSNR": metrics["PSNR"][rendered_name],
        "SSIM": metrics["SSIM"][rendered_name],
        "LPIPS": metrics["LPIPS"][rendered_name],
    })
open(output_path, "w", encoding="utf-8").write(json.dumps({"rows": rows}, indent=2, sort_keys=True) + "\n")
PY

sha256sum "$MERGED_ROOT"/*.json "$RUN_ROOT/02_metrics/render_index_to_image_name.json" \
  > "$RUN_ROOT/03_audit/fidelity_outputs.sha256"
find "$RUN_ROOT" -type f -printf '%P\n' | LC_ALL=C sort > "$RUN_ROOT/03_audit/file_inventory.txt"
echo "completed reconstruction fidelity: $RUN_ROOT"
