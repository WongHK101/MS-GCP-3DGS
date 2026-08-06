#!/usr/bin/env bash
set -euo pipefail

: "${RUN_ID:?Set a unique RUN_ID before launch}"

SCENE=gcp_3000_20260602
METHOD_COMMIT=2eee0e26d2d5fd00ec462df47752223952f6bf4e
METHOD_TREE=5eee127dfc0942bf83d9fdd72328e03ec0cbf6c4
ENV_LOCK=29f8997ba141357bbeddca9014757ab5a97acb9dd5ac312beda9e5f94acce0ed
RELEASE_DIGEST=513f8999fe4b110f15bcbecad7932895781cee755ee9ccd7a14ff10298546d75
R4_MANIFEST_SHA=88e354a7cc387975f6686020cf15a3584bfe28769c46360400dcfc027d82921c
CODE_ROOT=/root/autodl-tmp/worktrees/ms-gcp-v13/3dgs-original/$METHOD_COMMIT/official-train
ENV_ROOT=/root/autodl-tmp/envs/ms-gcp-v13/3dgs-original/py310-torch2.7.1-cu128-v1
INPUT_ROOT=/root/autodl-tmp/datasets/gs-gcp-v13/r4_clean_v1/$SCENE
RELEASE_ROOT=/root/autodl-tmp/datasets/ms-gcp-v13/$RELEASE_DIGEST/release_v1_3_0
ORCH_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BUILD_ROOT=/root/autodl-tmp/build/gs-gcp-v13/3dgs-original/$METHOD_COMMIT/$RUN_ID
RUN_ROOT=/root/autodl-tmp/runs/gs-gcp-v13/3dgs-original-clean-r4/$SCENE/$RUN_ID
MODEL_ROOT=$RUN_ROOT/02_checkpoints/model
RECIPE=$ORCH_ROOT/configs/gs_gcp_v13_original_3dgs_recipe_v3.json
REGISTRY=$ORCH_ROOT/configs/gs_gcp_method_registry_v1.json
RESOURCE_CONTRACT=$ORCH_ROOT/configs/gs_gcp_resource_probe_contract_v1.json
GNU_TIME=/root/autodl-tmp/tools/gs-gcp-v13/gnu-time/ubuntu-jammy-time-1.9-v1/root/usr/bin/time
DEPLOYMENT_EVIDENCE=/root/autodl-tmp/transfer_audits/GS_GCP_STAGE0_901_DEPLOYMENT_CURRENT.json
TRAIN_PORT=${TRAIN_PORT:-6013}

for path in "$BUILD_ROOT" "$RUN_ROOT"; do
  if test -e "$path"; then
    echo "refusing existing run/build root: $path" >&2
    exit 2
  fi
done
for path in "$CODE_ROOT" "$ENV_ROOT" "$INPUT_ROOT" "$RELEASE_ROOT"; do
  test -e "$path" || { echo "required path missing: $path" >&2; exit 3; }
done

mkdir -p "$BUILD_ROOT"/{preflight,torch_extensions,tmp}
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export GIT_OPTIONAL_LOCKS=0
export TMPDIR="$BUILD_ROOT/tmp"
export TORCH_EXTENSIONS_DIR="$BUILD_ROOT/torch_extensions"

"$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/validate_gs_gcp_v13_original_3dgs_recipe.py" \
  --recipe "$RECIPE" \
  --official_source "$CODE_ROOT" \
  --report "$BUILD_ROOT/preflight/recipe_validation.json"
"$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/validate_gs_gcp_method_registry.py" \
  --registry "$REGISTRY" \
  --repo_root "$ORCH_ROOT" \
  --report "$BUILD_ROOT/preflight/method_registry_validation.json"
"$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/validate_gs_gcp_original_3dgs_full_matrix.py" \
  --repo_root "$ORCH_ROOT" \
  --report "$BUILD_ROOT/preflight/full_matrix_lock_validation.json"
"$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/materialize_gs_gcp_r4_inputs.py" verify \
  --input_root "$INPUT_ROOT" \
  > "$BUILD_ROOT/preflight/r4_input_verification.json"
"$ENV_ROOT/bin/python" - "$INPUT_ROOT/R4_INPUT_MANIFEST.json" "$R4_MANIFEST_SHA" <<'PY'
import json
import sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
if manifest["manifest_sha256"] != sys.argv[2]:
    raise SystemExit("R4 input manifest SHA does not match the frozen recipe")
if (manifest["train_view_count"], manifest["test_view_count"]) != (82, 12):
    raise SystemExit("R4 input split counts do not match the frozen recipe")
PY
"$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/validate_gs_gcp_stage0.py" \
  --repo_root "$ORCH_ROOT" \
  --release_root "$RELEASE_ROOT" \
  --method_id 3dgs_original \
  --deployment_evidence "$DEPLOYMENT_EVIDENCE" \
  --report "$BUILD_ROOT/preflight/stage0_readiness.json" \
  --require_training_ready

test "$(git -C "$CODE_ROOT" rev-parse HEAD)" = "$METHOD_COMMIT"
test "$(git -C "$CODE_ROOT" rev-parse 'HEAD^{tree}')" = "$METHOD_TREE"
test -z "$(git -C "$CODE_ROOT" status --porcelain)"
test -z "$(git -C "$CODE_ROOT/submodules/diff-gaussian-rasterization" status --porcelain)"
test -z "$(git -C "$CODE_ROOT/submodules/simple-knn" status --porcelain)"
test "$(find "$INPUT_ROOT" -perm /222 -print -quit)" = ""

mkdir -p "$RUN_ROOT"/{00_preflight,01_training,02_checkpoints,03_packets,04_evaluation,05_diagnostics,06_audit,tmp}
cp "$BUILD_ROOT/preflight/"*.json "$RUN_ROOT/00_preflight/"
cp "$RECIPE" "$REGISTRY" "$ORCH_ROOT/configs/gs_gcp_r4_input_materialization_v1.json" "$RUN_ROOT/00_preflight/"
cp "$INPUT_ROOT/R4_INPUT_MANIFEST.json" "$RUN_ROOT/00_preflight/"
cp "${BASH_SOURCE[0]}" "$RUN_ROOT/06_audit/exact_launcher.sh"

export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$ENV_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST=12.0
export TMPDIR="$RUN_ROOT/tmp"
export PYTHONHASHSEED=0

git -C "$CODE_ROOT" rev-parse HEAD > "$RUN_ROOT/00_preflight/method_commit.txt"
git -C "$CODE_ROOT" rev-parse 'HEAD^{tree}' > "$RUN_ROOT/00_preflight/method_tree.txt"
git -C "$ORCH_ROOT" rev-parse HEAD > "$RUN_ROOT/00_preflight/orchestration_commit.txt"
"$ENV_ROOT/bin/python" -m pip freeze --all | LC_ALL=C sort > "$RUN_ROOT/00_preflight/environment.freeze.txt"
echo "$ENV_LOCK  $RUN_ROOT/00_preflight/environment.freeze.txt" | sha256sum -c -
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv,noheader > "$RUN_ROOT/00_preflight/gpu.txt"
test "$(wc -l < "$RUN_ROOT/00_preflight/gpu.txt")" -eq 1
nvcc --version > "$RUN_ROOT/00_preflight/nvcc.txt"
gcc --version > "$RUN_ROOT/00_preflight/gcc.txt"

cat > "$RUN_ROOT/06_audit/exact_command.sh" <<COMMAND
'$ENV_ROOT/bin/python' '$ORCH_ROOT/code/gcp/run_with_resource_probe.py' \\
  --contract '$RESOURCE_CONTRACT' --phase train \\
  --output_dir '$RUN_ROOT/01_training/resource_probe' \\
  --working_directory '$CODE_ROOT' --gpu_indices 0 --time_binary '$GNU_TIME' -- \\
  '$ENV_ROOT/bin/python' train.py \\
  --source_path '$INPUT_ROOT/train' --model_path '$MODEL_ROOT' --images images \\
  --resolution 1 --iterations 30000 --test_iterations 7000 30000 \\
  --save_iterations 7000 30000 --ip 127.0.0.1 --port '$TRAIN_PORT' --quiet
COMMAND

date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_ROOT/01_training/start_utc.txt"
set +e
"$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/run_with_resource_probe.py" \
  --contract "$RESOURCE_CONTRACT" \
  --phase train \
  --output_dir "$RUN_ROOT/01_training/resource_probe" \
  --working_directory "$CODE_ROOT" \
  --gpu_indices 0 \
  --time_binary "$GNU_TIME" \
  -- \
  "$ENV_ROOT/bin/python" train.py \
  --source_path "$INPUT_ROOT/train" \
  --model_path "$MODEL_ROOT" \
  --images images \
  --resolution 1 \
  --iterations 30000 \
  --test_iterations 7000 30000 \
  --save_iterations 7000 30000 \
  --ip 127.0.0.1 \
  --port "$TRAIN_PORT" \
  --quiet
STATUS=$?
set -e
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_ROOT/01_training/end_utc.txt"
printf '%s\n' "$STATUS" > "$RUN_ROOT/01_training/exit_code.txt"
test "$STATUS" -eq 0

test -f "$MODEL_ROOT/point_cloud/iteration_30000/point_cloud.ply"
sha256sum "$MODEL_ROOT/point_cloud/iteration_7000/point_cloud.ply" \
  "$MODEL_ROOT/point_cloud/iteration_30000/point_cloud.ply" \
  > "$RUN_ROOT/06_audit/checkpoints.sha256"
"$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/materialize_gs_gcp_r4_inputs.py" verify \
  --input_root "$INPUT_ROOT" \
  > "$RUN_ROOT/06_audit/r4_input_verification_after.json"
git -C "$CODE_ROOT" status --porcelain > "$RUN_ROOT/06_audit/method_status_after.txt"
test ! -s "$RUN_ROOT/06_audit/method_status_after.txt"
touch "$RUN_ROOT/TRAINING_COMPLETE_NOT_YET_QUALIFIED"

printf 'RUN_ROOT=%s\n' "$RUN_ROOT"
printf 'Training finished; held-out RGB, packet export, formal GCP evaluation, and external acceptance remain.\n'
