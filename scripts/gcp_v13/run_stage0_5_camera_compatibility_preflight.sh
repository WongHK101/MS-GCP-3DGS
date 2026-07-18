#!/usr/bin/env bash
set -euo pipefail

: "${RUN_ID:?Set a unique RUN_ID}"

RELEASE_DIGEST=513f8999fe4b110f15bcbecad7932895781cee755ee9ccd7a14ff10298546d75
CANDIDATE_COMMIT=ac8e622f7347cc01ed2af4e0381a05191025834b
ORCH_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CANDIDATE_ROOT=/root/autodl-tmp/worktrees/gs-gcp-v13/3dgs-original/$CANDIDATE_COMMIT/path-backed
ENV_ROOT=/root/autodl-tmp/envs/ms-gcp-v13/3dgs-original/py310-torch2.7.1-cu128-v1
DATA_ROOT=/root/autodl-tmp/datasets/ms-gcp-v13/$RELEASE_DIGEST
RUN_ROOT=/root/autodl-tmp/runs/gs-gcp-v13/stage0-5/$RUN_ID
SPLIT_MANIFEST=$ORCH_ROOT/configs/gs_gcp_rgb_holdout_split_manifest_v1.json
RESOURCE_CONTRACT=$ORCH_ROOT/configs/gs_gcp_resource_probe_contract_v2.json
COMPATIBILITY_CONTRACT=$ORCH_ROOT/configs/gs_gcp_original_3dgs_camera_materialization_compatibility_v1.json
GNU_TIME=/root/autodl-tmp/tools/gs-gcp-v13/gnu-time/ubuntu-jammy-time-1.9-v1/root/usr/bin/time

test ! -e "$RUN_ROOT"
test -z "$(git -C "$ORCH_ROOT" status --porcelain)"
test -z "$(git -C "$CANDIDATE_ROOT" status --porcelain)"
test "$(git -C "$CANDIDATE_ROOT" rev-parse HEAD)" = "$CANDIDATE_COMMIT"
mkdir -p "$RUN_ROOT"/{assets,preflight,audit,tmp}
cp "${BASH_SOURCE[0]}" "$RUN_ROOT/audit/exact_launcher.sh"
cp "$SPLIT_MANIFEST" "$RESOURCE_CONTRACT" "$COMPATIBILITY_CONTRACT" "$RUN_ROOT/audit/"
git -C "$ORCH_ROOT" rev-parse HEAD > "$RUN_ROOT/audit/orchestration_commit.txt"

export CUDA_VISIBLE_DEVICES=0 CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$ENV_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export TORCH_EXTENSIONS_DIR="$RUN_ROOT/tmp/torch_extensions" TMPDIR="$RUN_ROOT/tmp"
mkdir -p "$TORCH_EXTENSIONS_DIR"

for scene in gcp_50000_20260610 gcp_100000_20260610; do
  "$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/gs_gcp_stage0_5.py" materialize-subsets \
    --split_manifest "$SPLIT_MANIFEST" --scene "$scene" --source_root "$DATA_ROOT/$scene" \
    --output_root "$RUN_ROOT/assets/$scene" --image_mode symlink \
    > "$RUN_ROOT/preflight/${scene}_materialize.log"
done

LAST_STATUS=
run_one() {
  local label="$1" scene="$2" device="$3"
  local evidence="$RUN_ROOT/preflight/$label"
  local train_root="$RUN_ROOT/assets/$scene/train_camera_subset"
  mkdir -p "$evidence"
  set +e
  "$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/run_with_resource_probe_v2.py" \
    --contract "$RESOURCE_CONTRACT" --output_dir "$evidence/resource_probe" \
    --working_directory "$CANDIDATE_ROOT" --gpu_indices 0 --time_binary "$GNU_TIME" \
    --timeout_seconds 3600 --enforce_contract_gates --failure_stage camera_load -- \
    "$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/original_3dgs_camera_load_preflight.py" \
      --method_root "$CANDIDATE_ROOT" --source_root "$train_root" \
      --resolution 4 --data_device "$device" --stabilization_seconds 30 \
      --expected_materialization path_backed \
      --lifecycle_report "$evidence/lifecycle.jsonl" \
      --report "$evidence/camera_load_report.json"
  local outer=$?
  set -e
  printf '%s\n' "$outer" > "$evidence/outer_probe_exit_code.txt"
  LAST_STATUS=$("$ENV_ROOT/bin/python" - "$evidence/resource_probe/resource_summary.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["status"])
PY
)
  if [[ "$LAST_STATUS" == PASS ]]; then
    "$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/validate_stage0_5_resource_preflight.py" \
      --contract "$RESOURCE_CONTRACT" \
      --resource_summary "$evidence/resource_probe/resource_summary.json" \
      --camera_report "$evidence/camera_load_report.json" \
      --output "$evidence/validation.json"
  fi
}

run_one candidate_a_50k gcp_50000_20260610 cuda
A50=$LAST_STATUS
if [[ "$A50" == PASS ]]; then
  run_one candidate_a_100k gcp_100000_20260610 cuda
  A100=$LAST_STATUS
else
  A100=
fi

if [[ "$A50" == PASS && "$A100" == PASS ]]; then
  printf '{"status":"SELECTED_A"}\n' > "$RUN_ROOT/preflight/selection_seed.json"
elif [[ "$A50" == GPU_MEMORY_BLOCKED || "$A100" == GPU_MEMORY_BLOCKED ]]; then
  test "$A50" = PASS -o "$A50" = GPU_MEMORY_BLOCKED
  test -z "$A100" -o "$A100" = PASS -o "$A100" = GPU_MEMORY_BLOCKED
  run_one candidate_b_50k gcp_50000_20260610 cpu
  B50=$LAST_STATUS
  test "$B50" = PASS
  run_one candidate_b_100k gcp_100000_20260610 cpu
  B100=$LAST_STATUS
  test "$B100" = PASS
else
  echo "Candidate A failed a non-GPU gate: 50K=$A50 100K=${A100:-NOT_RUN}" >&2
  exit 1
fi

"$ENV_ROOT/bin/python" - "$RUN_ROOT" "${A50}" "${A100:-}" "${B50:-}" "${B100:-}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
statuses = {"a50": sys.argv[2] or None, "a100": sys.argv[3] or None, "b50": sys.argv[4] or None, "b100": sys.argv[5] or None}
for key, status in statuses.items():
    if status is not None:
        path = root / "preflight" / f"selection_{key}.json"
        path.write_text(json.dumps({"status": status}) + "\n", encoding="utf-8")
PY

selection_args=(--a50 "$RUN_ROOT/preflight/selection_a50.json" --output "$RUN_ROOT/preflight/selected_contract.json")
test -z "${A100:-}" || selection_args+=(--a100 "$RUN_ROOT/preflight/selection_a100.json")
test -z "${B50:-}" || selection_args+=(--b50 "$RUN_ROOT/preflight/selection_b50.json")
test -z "${B100:-}" || selection_args+=(--b100 "$RUN_ROOT/preflight/selection_b100.json")
"$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/select_original_3dgs_camera_contract.py" "${selection_args[@]}"

echo "$RUN_ROOT"
