#!/usr/bin/env bash
set -euo pipefail

: "${SCENE_ID:?Set one approved remaining-five SCENE_ID}"
: "${RUN_ID:?Set a unique RUN_ID before launch}"

METHOD_COMMIT=2eee0e26d2d5fd00ec462df47752223952f6bf4e
ENV_LOCK=29f8997ba141357bbeddca9014757ab5a97acb9dd5ac312beda9e5f94acce0ed
RELEASE_DIGEST=513f8999fe4b110f15bcbecad7932895781cee755ee9ccd7a14ff10298546d75
CODE_ROOT=/root/autodl-tmp/worktrees/ms-gcp-v13/3dgs-original/$METHOD_COMMIT/official-train
ENV_ROOT=/root/autodl-tmp/envs/ms-gcp-v13/3dgs-original/py310-torch2.7.1-cu128-v1
DATASET_ROOT=/root/autodl-tmp/datasets/ms-gcp-v13/$RELEASE_DIGEST/$SCENE_ID
RELEASE_ROOT=/root/autodl-tmp/datasets/ms-gcp-v13/$RELEASE_DIGEST/release_v1_3_0
ORCH_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BUILD_ROOT=/root/autodl-tmp/build/gs-gcp-v13/3dgs-original/$METHOD_COMMIT/$RUN_ID
RUN_ROOT=/root/autodl-tmp/runs/gs-gcp-v13/3dgs-original/$SCENE_ID/$RUN_ID
MODEL_ROOT=$RUN_ROOT/02_checkpoints/model
RECIPE=$ORCH_ROOT/configs/gs_gcp_v13_original_3dgs_recipe_v2.json
FULL_MATRIX_PLAN=$ORCH_ROOT/configs/gs_gcp_v13_original_3dgs_full_matrix_v1.json
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

mkdir -p "$BUILD_ROOT"/torch_extensions "$BUILD_ROOT"/preflight "$BUILD_ROOT"/tmp
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export GIT_OPTIONAL_LOCKS=0
export TMPDIR="$BUILD_ROOT/tmp"

"$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/validate_gcp_v13_workspace_isolation.py" \
  --manifest <("$ENV_ROOT/bin/python" - "$SCENE_ID" "$RUN_ID" "$CODE_ROOT" "$ENV_ROOT" "$DATASET_ROOT" "$RELEASE_ROOT" "$BUILD_ROOT" "$RUN_ROOT" <<'PY'
import json
import sys
scene, run_id, code, env, data, release, build, run = sys.argv[1:]
print(json.dumps({
    "schema": "gs_gcp_method_run_layout_v1",
    "method_id": "3dgs-original",
    "scene": scene,
    "run_id": run_id,
    "code_root": code,
    "code_commit": "2eee0e26d2d5fd00ec462df47752223952f6bf4e",
    "environment_root": env,
    "environment_lock_sha256": "29f8997ba141357bbeddca9014757ab5a97acb9dd5ac312beda9e5f94acce0ed",
    "dataset_root": data,
    "release_root": release,
    "release_root_digest": "513f8999fe4b110f15bcbecad7932895781cee755ee9ccd7a14ff10298546d75",
    "build_root": build,
    "run_root": run,
    "torch_extensions_dir": build + "/torch_extensions",
    "temp_root": run + "/tmp",
    "output_subdirs": {
        "preflight": run + "/00_preflight",
        "training": run + "/01_training",
        "checkpoints": run + "/02_checkpoints",
        "packets": run + "/03_packets",
        "evaluation": run + "/04_evaluation",
        "diagnostics": run + "/05_diagnostics",
        "audit": run + "/06_audit",
    },
    "policies": {
        "dataset_access": "read_only",
        "release_access": "read_only",
        "code_runtime_writes": "forbidden",
        "overwrite_policy": "fail_if_exists",
        "global_python_install_allowed": False,
        "shared_cuda_cache_allowed": False,
        "hardlink_to_dataset_allowed": False,
    },
    "env_vars": {
        "PYTHONNOUSERSITE": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "TORCH_EXTENSIONS_DIR": build + "/torch_extensions",
        "TMPDIR": run + "/tmp",
    },
}))
PY
  ) \
  --require_nonexistent_run_root \
  --report "$BUILD_ROOT/preflight/isolation_validation.json"

# Persist the exact layout only after the isolation validator accepts it.
"$ENV_ROOT/bin/python" - "$BUILD_ROOT/preflight/isolation_validation.json" "$BUILD_ROOT/preflight/run_layout.json" "$SCENE_ID" "$RUN_ID" "$CODE_ROOT" "$ENV_ROOT" "$DATASET_ROOT" "$RELEASE_ROOT" "$BUILD_ROOT" "$RUN_ROOT" <<'PY'
import json
import sys
validation, output, scene, run_id, code, env, data, release, build, run = sys.argv[1:]
if json.load(open(validation, encoding="utf-8"))["status"] != "pass":
    raise SystemExit("isolation validation did not pass")
record = {
    "schema": "gs_gcp_method_run_layout_v1",
    "method_id": "3dgs-original",
    "scene": scene,
    "run_id": run_id,
    "code_root": code,
    "code_commit": "2eee0e26d2d5fd00ec462df47752223952f6bf4e",
    "environment_root": env,
    "environment_lock_sha256": "29f8997ba141357bbeddca9014757ab5a97acb9dd5ac312beda9e5f94acce0ed",
    "dataset_root": data,
    "release_root": release,
    "release_root_digest": "513f8999fe4b110f15bcbecad7932895781cee755ee9ccd7a14ff10298546d75",
    "build_root": build,
    "run_root": run,
    "torch_extensions_dir": build + "/torch_extensions",
    "temp_root": run + "/tmp",
    "output_subdirs": {name: run + "/" + value for name, value in {
        "preflight": "00_preflight", "training": "01_training", "checkpoints": "02_checkpoints",
        "packets": "03_packets", "evaluation": "04_evaluation", "diagnostics": "05_diagnostics",
        "audit": "06_audit"}.items()},
    "policies": {"dataset_access": "read_only", "release_access": "read_only", "code_runtime_writes": "forbidden", "overwrite_policy": "fail_if_exists", "global_python_install_allowed": False, "shared_cuda_cache_allowed": False, "hardlink_to_dataset_allowed": False},
    "env_vars": {"PYTHONNOUSERSITE": "1", "GIT_OPTIONAL_LOCKS": "0", "TORCH_EXTENSIONS_DIR": build + "/torch_extensions", "TMPDIR": run + "/tmp"},
}
open(output, "w", encoding="utf-8").write(json.dumps(record, indent=2, sort_keys=True) + "\n")
PY

"$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/validate_gs_gcp_v13_original_3dgs_recipe.py" \
  --recipe "$RECIPE" \
  --official_source "$CODE_ROOT" \
  --report "$BUILD_ROOT/preflight/recipe_validation.json"
"$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/validate_gs_gcp_stage0.py" \
  --repo_root "$ORCH_ROOT" \
  --release_root "$RELEASE_ROOT" \
  --method_id 3dgs_original \
  --deployment_evidence "$DEPLOYMENT_EVIDENCE" \
  --require_full_scene_matrix_eligible \
  --report "$BUILD_ROOT/preflight/stage0_readiness.json" \
  --require_training_ready
"$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/validate_gs_gcp_original_3dgs_full_matrix.py" \
  --repo_root "$ORCH_ROOT" \
  --plan "$FULL_MATRIX_PLAN" \
  --scene "$SCENE_ID" \
  --scene_root "$DATASET_ROOT" \
  --release_root "$RELEASE_ROOT" \
  --report "$BUILD_ROOT/preflight/full_matrix_scene_validation.json"

ORIGINAL_WIDTH=$("$ENV_ROOT/bin/python" - "$FULL_MATRIX_PLAN" "$SCENE_ID" <<'PY'
import json, sys
row = next(r for r in json.load(open(sys.argv[1], encoding='utf-8'))['scenes'] if r['scene'] == sys.argv[2])
print(row['original_width'])
PY
)
ORIGINAL_HEIGHT=$("$ENV_ROOT/bin/python" - "$FULL_MATRIX_PLAN" "$SCENE_ID" <<'PY'
import json, sys
row = next(r for r in json.load(open(sys.argv[1], encoding='utf-8'))['scenes'] if r['scene'] == sys.argv[2])
print(row['original_height'])
PY
)
"$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/gs_gcp_resolution.py" \
  --contract "$ORCH_ROOT/configs/gs_gcp_training_resolution_v1.json" \
  --width "$ORIGINAL_WIDTH" \
  --height "$ORIGINAL_HEIGHT" \
  > "$BUILD_ROOT/preflight/resolution_contract_validation.json"

mkdir -p "$RUN_ROOT"/{00_preflight,01_training,02_checkpoints,03_packets,04_evaluation,05_diagnostics,06_audit,tmp}
cp "$BUILD_ROOT/preflight/"*.json "$RUN_ROOT/00_preflight/"
cp "$RECIPE" "$FULL_MATRIX_PLAN" "$RESOURCE_CONTRACT" "$RUN_ROOT/00_preflight/"
cp "${BASH_SOURCE[0]}" "$RUN_ROOT/06_audit/exact_launcher.sh"

export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$ENV_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST=12.0
export TORCH_EXTENSIONS_DIR="$BUILD_ROOT/torch_extensions"
export TMPDIR="$RUN_ROOT/tmp"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0

source_digest() {
  local root="$1"
  find "$root" -type f -print0 \
    | LC_ALL=C sort -z \
    | while IFS= read -r -d '' path; do
        rel=${path#"$root"/}
        printf '%s  %s\n' "$(sha256sum "$path" | cut -d' ' -f1)" "$rel"
      done \
    | sha256sum \
    | cut -d' ' -f1
}

test -z "$(git -C "$CODE_ROOT" status --porcelain)"
test -z "$(git -C "$CODE_ROOT/submodules/diff-gaussian-rasterization" status --porcelain)"
test -z "$(git -C "$CODE_ROOT/submodules/simple-knn" status --porcelain)"
test -z "$(git -C "$ORCH_ROOT" status --porcelain)"
test "$(git -C "$CODE_ROOT" rev-parse HEAD)" = "$METHOD_COMMIT"
test "$(find "$DATASET_ROOT" "$RELEASE_ROOT" -perm /222 -print -quit)" = ""

SOURCE_PRE=$(source_digest "$DATASET_ROOT")
printf '%s\n' "$SOURCE_PRE" > "$RUN_ROOT/00_preflight/dataset_tree_digest_before.txt"
sha256sum "$RECIPE" "$FULL_MATRIX_PLAN" > "$RUN_ROOT/00_preflight/frozen_contracts.sha256"
sha256sum "$RELEASE_ROOT/v1_3_0_release_root_digest.json" > "$RUN_ROOT/00_preflight/release_root_record.sha256"
git -C "$CODE_ROOT" rev-parse HEAD > "$RUN_ROOT/00_preflight/method_commit.txt"
git -C "$CODE_ROOT" rev-parse 'HEAD^{tree}' > "$RUN_ROOT/00_preflight/method_tree.txt"
git -C "$ORCH_ROOT" rev-parse HEAD > "$RUN_ROOT/00_preflight/orchestration_commit.txt"
"$ENV_ROOT/bin/python" -m pip freeze --all | LC_ALL=C sort > "$RUN_ROOT/00_preflight/environment.freeze.txt"
echo "$ENV_LOCK  $RUN_ROOT/00_preflight/environment.freeze.txt" | sha256sum -c -
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv,noheader > "$RUN_ROOT/00_preflight/gpu.txt"
nvcc --version > "$RUN_ROOT/00_preflight/nvcc.txt"
gcc --version > "$RUN_ROOT/00_preflight/gcc.txt"

cat > "$RUN_ROOT/06_audit/exact_command.sh" <<COMMAND
'$ENV_ROOT/bin/python' '$ORCH_ROOT/code/gcp/run_with_resource_probe.py' \
  --contract '$RESOURCE_CONTRACT' --phase train \
  --output_dir '$RUN_ROOT/01_training/resource_probe' \
  --working_directory '$CODE_ROOT' --gpu_indices 0 --time_binary '$GNU_TIME' -- \
  '$ENV_ROOT/bin/python' train.py --source_path '$DATASET_ROOT' --model_path '$MODEL_ROOT' \
  --images images --resolution -1 --sh_degree 3 --data_device cuda --iterations 30000 \
  --position_lr_init 0.00016 --position_lr_final 0.0000016 --position_lr_delay_mult 0.01 \
  --position_lr_max_steps 30000 --feature_lr 0.0025 --opacity_lr 0.05 --scaling_lr 0.005 \
  --rotation_lr 0.001 --percent_dense 0.01 --lambda_dssim 0.2 --densification_interval 100 \
  --opacity_reset_interval 3000 --densify_from_iter 500 --densify_until_iter 15000 \
  --densify_grad_threshold 0.0002 --test_iterations 7000 30000 --save_iterations 7000 30000 \
  --ip 127.0.0.1 --port '$TRAIN_PORT'
COMMAND

date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_ROOT/01_training/start_utc.txt"
set +e
"$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/run_with_resource_probe.py" \
  --contract "$RESOURCE_CONTRACT" --phase train \
  --output_dir "$RUN_ROOT/01_training/resource_probe" \
  --working_directory "$CODE_ROOT" --gpu_indices 0 --time_binary "$GNU_TIME" -- \
  "$ENV_ROOT/bin/python" train.py --source_path "$DATASET_ROOT" --model_path "$MODEL_ROOT" \
  --images images --resolution -1 --sh_degree 3 --data_device cuda --iterations 30000 \
  --position_lr_init 0.00016 --position_lr_final 0.0000016 --position_lr_delay_mult 0.01 \
  --position_lr_max_steps 30000 --feature_lr 0.0025 --opacity_lr 0.05 --scaling_lr 0.005 \
  --rotation_lr 0.001 --percent_dense 0.01 --lambda_dssim 0.2 --densification_interval 100 \
  --opacity_reset_interval 3000 --densify_from_iter 500 --densify_until_iter 15000 \
  --densify_grad_threshold 0.0002 --test_iterations 7000 30000 --save_iterations 7000 30000 \
  --ip 127.0.0.1 --port "$TRAIN_PORT"
STATUS=$?
set -e
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_ROOT/01_training/end_utc.txt"
printf '%s\n' "$STATUS" > "$RUN_ROOT/01_training/exit_code.txt"
test "$STATUS" -eq 0

test -f "$MODEL_ROOT/point_cloud/iteration_30000/point_cloud.ply"
sha256sum "$MODEL_ROOT/point_cloud/iteration_7000/point_cloud.ply" "$MODEL_ROOT/point_cloud/iteration_30000/point_cloud.ply" > "$RUN_ROOT/06_audit/checkpoints.sha256"
find "$MODEL_ROOT" -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum > "$RUN_ROOT/06_audit/model_metadata.sha256"

SOURCE_POST=$(source_digest "$DATASET_ROOT")
printf '%s\n' "$SOURCE_POST" > "$RUN_ROOT/06_audit/dataset_tree_digest_after.txt"
test "$SOURCE_PRE" = "$SOURCE_POST"
git -C "$CODE_ROOT" status --porcelain > "$RUN_ROOT/06_audit/method_status_after.txt"
git -C "$ORCH_ROOT" status --porcelain > "$RUN_ROOT/06_audit/orchestration_status_after.txt"
test ! -s "$RUN_ROOT/06_audit/method_status_after.txt"
test ! -s "$RUN_ROOT/06_audit/orchestration_status_after.txt"
test -z "$(git -C "$CODE_ROOT/submodules/diff-gaussian-rasterization" status --porcelain)"
test -z "$(git -C "$CODE_ROOT/submodules/simple-knn" status --porcelain)"

printf 'RUN_ROOT=%s\n' "$RUN_ROOT"
printf 'SOURCE_DIGEST=%s\n' "$SOURCE_POST"
cat "$RUN_ROOT/06_audit/checkpoints.sha256"
