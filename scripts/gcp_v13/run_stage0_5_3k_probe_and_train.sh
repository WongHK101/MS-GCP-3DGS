#!/usr/bin/env bash
set -euo pipefail

: "${RUN_ID:?Set a unique RUN_ID}"

RELEASE_DIGEST=513f8999fe4b110f15bcbecad7932895781cee755ee9ccd7a14ff10298546d75
SCENE=gcp_3000_20260602
METHOD_COMMIT=db8deebca67e8d5e1507e67c98de603eca0dfd85
ORCH_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
METHOD_ROOT=/root/autodl-tmp/worktrees/gs-gcp-v13/3dgs-original/$METHOD_COMMIT/serialization-safe
ENV_ROOT=/root/autodl-tmp/envs/ms-gcp-v13/3dgs-original/py310-torch2.7.1-cu128-v1
SOURCE_ROOT=/root/autodl-tmp/datasets/ms-gcp-v13/$RELEASE_DIGEST/$SCENE
RUN_ROOT=/root/autodl-tmp/runs/gs-gcp-v13/stage0-5/$RUN_ID
SPLIT_MANIFEST=$ORCH_ROOT/configs/gs_gcp_rgb_holdout_split_manifest_v1.json
RESOURCE_CONTRACT=$ORCH_ROOT/configs/gs_gcp_resource_probe_contract_v2.json
GNU_TIME=/root/autodl-tmp/tools/gs-gcp-v13/gnu-time/ubuntu-jammy-time-1.9-v1/root/usr/bin/time
ASSET_ROOT=$RUN_ROOT/assets/$SCENE
TRAIN_ROOT=$ASSET_ROOT/train_camera_subset
TEST_ROOT=$ASSET_ROOT/test_camera_subset
MODEL_ROOT=$RUN_ROOT/02_checkpoints/model

test ! -e "$RUN_ROOT"
test -z "$(git -C "$ORCH_ROOT" status --porcelain)"
test -z "$(git -C "$METHOD_ROOT" status --porcelain)"
test "$(git -C "$METHOD_ROOT" rev-parse HEAD)" = "$METHOD_COMMIT"
mkdir -p "$RUN_ROOT"/{00_preflight,01_micro,02_checkpoints,03_render,04_rgb_metrics,05_packets,06_gcp_evaluation,07_measurement,08_audit,tmp}
cp "${BASH_SOURCE[0]}" "$RUN_ROOT/08_audit/exact_launcher.sh"
cp "$SPLIT_MANIFEST" "$RESOURCE_CONTRACT" "$RUN_ROOT/08_audit/"

export CUDA_VISIBLE_DEVICES=0 CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$ENV_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export TORCH_EXTENSIONS_DIR="$RUN_ROOT/tmp/torch_extensions" TMPDIR="$RUN_ROOT/tmp"
mkdir -p "$TORCH_EXTENSIONS_DIR"

"$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/gs_gcp_stage0_5.py" materialize-subsets \
  --split_manifest "$SPLIT_MANIFEST" --scene "$SCENE" --source_root "$SOURCE_ROOT" \
  --output_root "$ASSET_ROOT" --image_mode symlink > "$RUN_ROOT/00_preflight/materialize.log"
"$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/gs_gcp_stage0_5.py" generate-gt \
  --split_manifest "$SPLIT_MANIFEST" --scene "$SCENE" --source_root "$SOURCE_ROOT" \
  --output_root "$RUN_ROOT/03_render/benchmark_gt" > "$RUN_ROOT/00_preflight/generate_gt.log"

strace -f -e trace=open,openat,openat2 -o "$RUN_ROOT/00_preflight/camera_access.strace" \
  "$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/original_3dgs_camera_load_preflight.py" \
    --method_root "$METHOD_ROOT" --source_root "$TRAIN_ROOT" --report "$RUN_ROOT/00_preflight/3k_camera_report.json" \
    --resolution 4 --data_device cuda --stabilization_seconds 1
"$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/validate_stage0_5_file_access.py" \
  --trace "$RUN_ROOT/00_preflight/camera_access.strace" --train_root "$TRAIN_ROOT" \
  --forbidden_root "$TEST_ROOT" --forbidden_root "$ASSET_ROOT/common_full_sfm" \
  --output "$RUN_ROOT/00_preflight/training_file_access_validation.json"

# Synthetic child: exactly the same child argv and output path are used direct and probed.
SYNTHETIC=$RUN_ROOT/01_micro/synthetic.bin
"$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/stage0_5_probe_synthetic_child.py" --output "$SYNTHETIC"
mv "$SYNTHETIC" "$RUN_ROOT/01_micro/synthetic_direct.bin"
"$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/run_with_resource_probe_v2.py" \
  --contract "$RESOURCE_CONTRACT" --output_dir "$RUN_ROOT/01_micro/synthetic_probe" \
  --working_directory "$ORCH_ROOT" --gpu_indices 0 --time_binary "$GNU_TIME" -- \
  "$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/stage0_5_probe_synthetic_child.py" --output "$SYNTHETIC"
cmp "$RUN_ROOT/01_micro/synthetic_direct.bin" "$SYNTHETIC"
sha256sum "$RUN_ROOT/01_micro/synthetic_direct.bin" "$SYNTHETIC" > "$RUN_ROOT/01_micro/synthetic.sha256"

MICRO_WORK=$RUN_ROOT/01_micro/work
MICRO_MODEL=$MICRO_WORK/model
MICRO_TRACE=$MICRO_WORK/trace.json
mkdir -p "$MICRO_WORK"
micro_command=(
  "$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/original_3dgs_micro_child.py"
  --method_root "$METHOD_ROOT" --trace_path "$MICRO_TRACE" --
  --source_path "$TRAIN_ROOT" --model_path "$MICRO_MODEL" --images images --resolution 4
  --data_device cuda --iterations 100 --save_iterations 100 --test_iterations 7000
  --quiet --ip 127.0.0.1 --port 6025
)
"${micro_command[@]}" > "$RUN_ROOT/01_micro/direct_stdout.log" 2> "$RUN_ROOT/01_micro/direct_stderr.log"
mv "$MICRO_MODEL" "$RUN_ROOT/01_micro/direct_model"
mv "$MICRO_TRACE" "$RUN_ROOT/01_micro/direct_trace.json"
"$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/run_with_resource_probe_v2.py" \
  --contract "$RESOURCE_CONTRACT" --output_dir "$RUN_ROOT/01_micro/probed_resource" \
  --working_directory "$ORCH_ROOT" --gpu_indices 0 --time_binary "$GNU_TIME" -- \
  "${micro_command[@]}"
mv "$MICRO_MODEL" "$RUN_ROOT/01_micro/probed_model"
mv "$MICRO_TRACE" "$RUN_ROOT/01_micro/probed_trace.json"
"$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/compare_original_3dgs_micro_runs.py" \
  --direct_model "$RUN_ROOT/01_micro/direct_model" --probed_model "$RUN_ROOT/01_micro/probed_model" \
  --direct_trace "$RUN_ROOT/01_micro/direct_trace.json" --probed_trace "$RUN_ROOT/01_micro/probed_trace.json" \
  --output "$RUN_ROOT/01_micro/equivalence.json"

train_command=(
  "$ENV_ROOT/bin/python" train.py --source_path "$TRAIN_ROOT" --model_path "$MODEL_ROOT"
  --images images --resolution 4 --sh_degree 3 --data_device cuda --iterations 30000
  --position_lr_init 0.00016 --position_lr_final 0.0000016 --position_lr_delay_mult 0.01
  --position_lr_max_steps 30000 --feature_lr 0.0025 --opacity_lr 0.05 --scaling_lr 0.005
  --rotation_lr 0.001 --percent_dense 0.01 --lambda_dssim 0.2 --densification_interval 100
  --opacity_reset_interval 3000 --densify_from_iter 500 --densify_until_iter 15000
  --densify_grad_threshold 0.0002 --test_iterations 7000 30000 --save_iterations 7000 30000
  --quiet --ip 127.0.0.1 --port 6026
)
printf '%q ' "${train_command[@]}" > "$RUN_ROOT/08_audit/exact_training_command.sh"
printf '\n' >> "$RUN_ROOT/08_audit/exact_training_command.sh"
"$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/run_with_resource_probe_v2.py" \
  --contract "$RESOURCE_CONTRACT" --output_dir "$RUN_ROOT/02_checkpoints/training_resource" \
  --working_directory "$METHOD_ROOT" --gpu_indices 0 --time_binary "$GNU_TIME" -- \
  "${train_command[@]}"
test -f "$MODEL_ROOT/point_cloud/iteration_30000/point_cloud.ply"
sha256sum "$MODEL_ROOT/point_cloud/iteration_30000/point_cloud.ply" > "$RUN_ROOT/08_audit/formal_checkpoint.sha256"

# Stage 0.5 stops this launcher after the formal checkpoint. Post-training
# render, packet, and evaluator commands are launched by the audited continuation
# only after this root passes checkpoint and source/data integrity validation.
echo "$RUN_ROOT"
