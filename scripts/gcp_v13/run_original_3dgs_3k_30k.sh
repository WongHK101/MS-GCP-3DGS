#!/usr/bin/env bash
set -euo pipefail

: "${RUN_ID:?Set a unique RUN_ID before launch}"

METHOD_COMMIT=2eee0e26d2d5fd00ec462df47752223952f6bf4e
ENV_LOCK=29f8997ba141357bbeddca9014757ab5a97acb9dd5ac312beda9e5f94acce0ed
RELEASE_DIGEST=513f8999fe4b110f15bcbecad7932895781cee755ee9ccd7a14ff10298546d75
SOURCE_MANIFEST_SHA=442c7d74ba0d79f7611b75f9f9155c7d1bf0d09ea71f2985cfb08f68aed24b7d
CODE_ROOT=/root/autodl-tmp/worktrees/gs-gcp-v13/3dgs-original/$METHOD_COMMIT/official-train
ENV_ROOT=/root/autodl-tmp/envs/gs-gcp-v13/3dgs-original/py310-torch2.7.1-cu128-v1
DATASET_ROOT=/root/autodl-tmp/datasets/gs-gcp-v13/$RELEASE_DIGEST/gcp_3000_20260602
RELEASE_ROOT=/root/autodl-tmp/datasets/gs-gcp-v13/$RELEASE_DIGEST/release_v1_3_0
ORCH_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BUILD_ROOT=/root/autodl-tmp/build/gs-gcp-v13/3dgs-original/$METHOD_COMMIT/$RUN_ID
RUN_ROOT=/root/autodl-tmp/runs/gs-gcp-v13/3dgs-original/gcp_3000_20260602/$RUN_ID
MODEL_ROOT=$RUN_ROOT/02_checkpoints/model
RECIPE=$ORCH_ROOT/configs/gs_gcp_v13_original_3dgs_recipe_v2.json
RESOURCE_CONTRACT=$ORCH_ROOT/configs/gs_gcp_resource_probe_contract_v1.json
GNU_TIME=/root/autodl-tmp/tools/gs-gcp-v13/gnu-time/ubuntu-jammy-time-1.9-v1/root/usr/bin/time
DEPLOYMENT_EVIDENCE=/root/autodl-tmp/transfer_audits/GS_GCP_STAGE0_DEPLOYMENT_CURRENT.json

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

cat > "$BUILD_ROOT/preflight/run_layout.json" <<JSON
{
  "schema": "gs_gcp_method_run_layout_v1",
  "method_id": "3dgs-original",
  "scene": "gcp_3000_20260602",
  "run_id": "$RUN_ID",
  "code_root": "$CODE_ROOT",
  "code_commit": "$METHOD_COMMIT",
  "environment_root": "$ENV_ROOT",
  "environment_lock_sha256": "$ENV_LOCK",
  "dataset_root": "$DATASET_ROOT",
  "release_root": "$RELEASE_ROOT",
  "release_root_digest": "$RELEASE_DIGEST",
  "build_root": "$BUILD_ROOT",
  "run_root": "$RUN_ROOT",
  "torch_extensions_dir": "$BUILD_ROOT/torch_extensions",
  "temp_root": "$RUN_ROOT/tmp",
  "output_subdirs": {
    "preflight": "$RUN_ROOT/00_preflight",
    "training": "$RUN_ROOT/01_training",
    "checkpoints": "$RUN_ROOT/02_checkpoints",
    "packets": "$RUN_ROOT/03_packets",
    "evaluation": "$RUN_ROOT/04_evaluation",
    "diagnostics": "$RUN_ROOT/05_diagnostics",
    "audit": "$RUN_ROOT/06_audit"
  },
  "policies": {
    "dataset_access": "read_only",
    "release_access": "read_only",
    "code_runtime_writes": "forbidden",
    "overwrite_policy": "fail_if_exists",
    "global_python_install_allowed": false,
    "shared_cuda_cache_allowed": false,
    "hardlink_to_dataset_allowed": false
  },
  "env_vars": {
    "PYTHONNOUSERSITE": "1",
    "GIT_OPTIONAL_LOCKS": "0",
    "TORCH_EXTENSIONS_DIR": "$BUILD_ROOT/torch_extensions",
    "TMPDIR": "$RUN_ROOT/tmp"
  }
}
JSON

"$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/validate_gcp_v13_workspace_isolation.py" \
  --manifest "$BUILD_ROOT/preflight/run_layout.json" \
  --require_nonexistent_run_root \
  --report "$BUILD_ROOT/preflight/isolation_validation.json"
"$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/validate_gs_gcp_v13_original_3dgs_recipe.py" \
  --recipe "$RECIPE" \
  --official_source "$CODE_ROOT" \
  --report "$BUILD_ROOT/preflight/recipe_validation.json"
"$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/validate_gs_gcp_stage0.py" \
  --repo_root "$ORCH_ROOT" \
  --release_root "$RELEASE_ROOT" \
  --method_id 3dgs_original \
  --deployment_evidence "$DEPLOYMENT_EVIDENCE" \
  --report "$BUILD_ROOT/preflight/stage0_readiness.json" \
  --require_training_ready
"$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/gs_gcp_resolution.py" \
  --contract "$ORCH_ROOT/configs/gs_gcp_training_resolution_v1.json" \
  --width 5654 \
  --height 4098 \
  > "$BUILD_ROOT/preflight/resolution_contract_validation.json"

mkdir -p "$RUN_ROOT"/{00_preflight,01_training,02_checkpoints,03_packets,04_evaluation,05_diagnostics,06_audit,tmp}
cp "$BUILD_ROOT/preflight/run_layout.json" "$RUN_ROOT/00_preflight/"
cp "$BUILD_ROOT/preflight/isolation_validation.json" "$RUN_ROOT/00_preflight/"
cp "$BUILD_ROOT/preflight/recipe_validation.json" "$RUN_ROOT/00_preflight/"
cp "$BUILD_ROOT/preflight/stage0_readiness.json" "$RUN_ROOT/00_preflight/"
cp "$BUILD_ROOT/preflight/resolution_contract_validation.json" "$RUN_ROOT/00_preflight/"
cp "$RECIPE" "$RUN_ROOT/00_preflight/"
cp "$RESOURCE_CONTRACT" "$RUN_ROOT/00_preflight/"
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
test "$(git -C "$ORCH_ROOT" status --porcelain)" = ""
test "$(git -C "$CODE_ROOT" rev-parse HEAD)" = "$METHOD_COMMIT"
test "$(find "$DATASET_ROOT" "$RELEASE_ROOT" -perm /222 -print -quit)" = ""
echo "$SOURCE_MANIFEST_SHA  $DATASET_ROOT/SOURCE_MANIFEST.json" | sha256sum -c -

SOURCE_PRE=$(source_digest "$DATASET_ROOT")
printf '%s\n' "$SOURCE_PRE" > "$RUN_ROOT/00_preflight/dataset_tree_digest_before.txt"
sha256sum "$RECIPE" > "$RUN_ROOT/00_preflight/recipe.sha256"
sha256sum "$RELEASE_ROOT/v1_3_0_release_root_digest.json" > "$RUN_ROOT/00_preflight/release_root_record.sha256"
git -C "$CODE_ROOT" rev-parse HEAD > "$RUN_ROOT/00_preflight/method_commit.txt"
git -C "$CODE_ROOT" rev-parse 'HEAD^{tree}' > "$RUN_ROOT/00_preflight/method_tree.txt"
git -C "$ORCH_ROOT" rev-parse HEAD > "$RUN_ROOT/00_preflight/orchestration_commit.txt"
"$ENV_ROOT/bin/python" -m pip freeze --all | LC_ALL=C sort > "$RUN_ROOT/00_preflight/environment.freeze.txt"
echo "$ENV_LOCK  $RUN_ROOT/00_preflight/environment.freeze.txt" | sha256sum -c -
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv,noheader > "$RUN_ROOT/00_preflight/gpu.txt"
nvcc --version > "$RUN_ROOT/00_preflight/nvcc.txt"
gcc --version > "$RUN_ROOT/00_preflight/gcc.txt"

PYTHONPATH="$ORCH_ROOT/code/gcp" "$ENV_ROOT/bin/python" - "$RELEASE_ROOT" > "$RUN_ROOT/00_preflight/release_integrity.json" <<'PY'
import json
import sys
from pathlib import Path
from gcp_pixel_domain_v1_2 import verify_payload_integrity

root = Path(sys.argv[1])
result = verify_payload_integrity(
    root,
    root / "v1_3_0_release_file_manifest.json",
    root / "v1_3_0_release_root_digest.json",
)
print(json.dumps(result, indent=2))
if not result["passed"]:
    raise SystemExit(1)
PY

cat > "$RUN_ROOT/06_audit/exact_command.sh" <<COMMAND
'$ENV_ROOT/bin/python' '$ORCH_ROOT/code/gcp/run_with_resource_probe.py' \\
  --contract '$RESOURCE_CONTRACT' \\
  --phase train \\
  --output_dir '$RUN_ROOT/01_training/resource_probe' \\
  --working_directory '$CODE_ROOT' \\
  --gpu_indices 0 \\
  --time_binary '$GNU_TIME' \\
  -- \\
  '$ENV_ROOT/bin/python' train.py \\
  --source_path '$DATASET_ROOT' \\
  --model_path '$MODEL_ROOT' \\
  --images images \\
  --resolution -1 \\
  --sh_degree 3 \\
  --data_device cuda \\
  --iterations 30000 \\
  --position_lr_init 0.00016 \\
  --position_lr_final 0.0000016 \\
  --position_lr_delay_mult 0.01 \\
  --position_lr_max_steps 30000 \\
  --feature_lr 0.0025 \\
  --opacity_lr 0.05 \\
  --scaling_lr 0.005 \\
  --rotation_lr 0.001 \\
  --percent_dense 0.01 \\
  --lambda_dssim 0.2 \\
  --densification_interval 100 \\
  --opacity_reset_interval 3000 \\
  --densify_from_iter 500 \\
  --densify_until_iter 15000 \\
  --densify_grad_threshold 0.0002 \\
  --test_iterations 7000 30000 \\
  --save_iterations 7000 30000 \\
  --ip 127.0.0.1 \\
  --port 6013
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
  --source_path "$DATASET_ROOT" \
  --model_path "$MODEL_ROOT" \
  --images images \
  --resolution -1 \
  --sh_degree 3 \
  --data_device cuda \
  --iterations 30000 \
  --position_lr_init 0.00016 \
  --position_lr_final 0.0000016 \
  --position_lr_delay_mult 0.01 \
  --position_lr_max_steps 30000 \
  --feature_lr 0.0025 \
  --opacity_lr 0.05 \
  --scaling_lr 0.005 \
  --rotation_lr 0.001 \
  --percent_dense 0.01 \
  --lambda_dssim 0.2 \
  --densification_interval 100 \
  --opacity_reset_interval 3000 \
  --densify_from_iter 500 \
  --densify_until_iter 15000 \
  --densify_grad_threshold 0.0002 \
  --test_iterations 7000 30000 \
  --save_iterations 7000 30000 \
  --ip 127.0.0.1 \
  --port 6013
STATUS=$?
set -e
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_ROOT/01_training/end_utc.txt"
printf '%s\n' "$STATUS" > "$RUN_ROOT/01_training/exit_code.txt"
test "$STATUS" -eq 0

test -f "$MODEL_ROOT/point_cloud/iteration_30000/point_cloud.ply"
sha256sum "$MODEL_ROOT/point_cloud/iteration_7000/point_cloud.ply" "$MODEL_ROOT/point_cloud/iteration_30000/point_cloud.ply" > "$RUN_ROOT/06_audit/checkpoints.sha256"
sha256sum "$MODEL_ROOT/cameras.json" "$MODEL_ROOT/cfg_args" > "$RUN_ROOT/06_audit/model_metadata.sha256"

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
