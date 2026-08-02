#!/usr/bin/env bash
set -euo pipefail

: "${RUN_ID:?Set a unique RUN_ID}"

RELEASE_DIGEST=513f8999fe4b110f15bcbecad7932895781cee755ee9ccd7a14ff10298546d75
METHOD_COMMIT=db8deebca67e8d5e1507e67c98de603eca0dfd85
ORCH_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
METHOD_ROOT=/root/autodl-tmp/worktrees/gs-gcp-v13/3dgs-original/$METHOD_COMMIT/serialization-safe
ENV_ROOT=/root/autodl-tmp/envs/ms-gcp-v13/3dgs-original/py310-torch2.7.1-cu128-v1
DATA_ROOT=/root/autodl-tmp/datasets/ms-gcp-v13/$RELEASE_DIGEST
RUN_ROOT=/root/autodl-tmp/runs/gs-gcp-v13/stage0-5/$RUN_ID
SPLIT_MANIFEST=$ORCH_ROOT/configs/gs_gcp_rgb_holdout_split_manifest_v1.json
RESOURCE_CONTRACT=$ORCH_ROOT/configs/gs_gcp_resource_probe_contract_v2.json
GNU_TIME=/root/autodl-tmp/tools/gs-gcp-v13/gnu-time/ubuntu-jammy-time-1.9-v1/root/usr/bin/time

test ! -e "$RUN_ROOT"
test -z "$(git -C "$ORCH_ROOT" status --porcelain)"
test -z "$(git -C "$METHOD_ROOT" status --porcelain)"
test "$(git -C "$METHOD_ROOT" rev-parse HEAD)" = "$METHOD_COMMIT"
test -f "$SPLIT_MANIFEST"
mkdir -p "$RUN_ROOT"/{assets,preflight,audit,tmp}
cp "${BASH_SOURCE[0]}" "$RUN_ROOT/audit/exact_launcher.sh"
cp "$SPLIT_MANIFEST" "$RESOURCE_CONTRACT" "$RUN_ROOT/audit/"
git -C "$ORCH_ROOT" rev-parse HEAD > "$RUN_ROOT/audit/orchestration_commit.txt"

export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$ENV_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export MALLOC_TRIM_THRESHOLD_=0
export TORCH_EXTENSIONS_DIR="$RUN_ROOT/tmp/torch_extensions"
export TMPDIR="$RUN_ROOT/tmp"
mkdir -p "$TORCH_EXTENSIONS_DIR"

for scene in gcp_100000_20260610 gcp_50000_20260610; do
  source_root="$DATA_ROOT/$scene"
  asset_root="$RUN_ROOT/assets/$scene"
  evidence_root="$RUN_ROOT/preflight/$scene"
  mkdir -p "$evidence_root"
  test -z "$(find "$source_root" -perm /222 -print -quit)"
  "$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/gs_gcp_stage0_5.py" materialize-subsets \
    --split_manifest "$SPLIT_MANIFEST" --scene "$scene" --source_root "$source_root" \
    --output_root "$asset_root" --image_mode symlink \
    > "$evidence_root/materialize.log"
  train_root="$asset_root/train_camera_subset"
  test ! -e "$train_root/sparse/0/points3D.bin"
  set +e
  "$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/run_with_resource_probe_v2.py" \
    --contract "$RESOURCE_CONTRACT" --output_dir "$evidence_root/resource_probe" \
    --working_directory "$METHOD_ROOT" --gpu_indices 0 --time_binary "$GNU_TIME" \
    --timeout_seconds 3600 --enforce_contract_gates --failure_stage camera_load -- \
    "$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/original_3dgs_camera_load_preflight.py" \
      --method_root "$METHOD_ROOT" --source_root "$train_root" \
      --resolution 4 --data_device cuda --stabilization_seconds 30 \
      --host_allocator_policy glibc_malloc_trim_threshold_zero_v1 \
      --expected_materialization eager \
      --lifecycle_report "$evidence_root/lifecycle.jsonl" \
      --report "$evidence_root/camera_load_report.json"
  status=$?
  set -e
  printf '%s\n' "$status" > "$evidence_root/probe_exit_code.txt"
  test "$status" -eq 0
  "$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/validate_stage0_5_resource_preflight.py" \
    --contract "$RESOURCE_CONTRACT" \
    --resource_summary "$evidence_root/resource_probe/resource_summary.json" \
    --camera_report "$evidence_root/camera_load_report.json" \
    --output "$evidence_root/validation.json"
done

"$ENV_ROOT/bin/python" - "$RUN_ROOT" > "$RUN_ROOT/preflight/combined_status.json" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
rows = []
for scene in ("gcp_100000_20260610", "gcp_50000_20260610"):
    payload = json.loads((root / "preflight" / scene / "validation.json").read_text())
    rows.append({"scene": scene, "status": payload["status"], "failed_count": payload["failed_count"]})
out = {"schema": "gs_gcp_stage0_5_large_scene_preflight_summary_v1", "status": "PASS" if all(r["status"] == "PASS" for r in rows) else "BLOCKER", "scenes": rows, "scope": "camera_load_feasibility_only_not_training_guarantee"}
print(json.dumps(out, indent=2, sort_keys=True))
if out["status"] != "PASS": raise SystemExit(1)
PY

echo "$RUN_ROOT"
