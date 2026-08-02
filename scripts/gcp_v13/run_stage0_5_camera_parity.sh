#!/usr/bin/env bash
set -euo pipefail

: "${RUN_ID:?Set a unique RUN_ID}"

RELEASE_DIGEST=513f8999fe4b110f15bcbecad7932895781cee755ee9ccd7a14ff10298546d75
REFERENCE_COMMIT=db8deebca67e8d5e1507e67c98de603eca0dfd85
CANDIDATE_COMMIT=ac8e622f7347cc01ed2af4e0381a05191025834b
ORCH_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
REFERENCE_ROOT=/root/autodl-tmp/worktrees/gs-gcp-v13/3dgs-original/$REFERENCE_COMMIT/serialization-safe
CANDIDATE_ROOT=/root/autodl-tmp/worktrees/gs-gcp-v13/3dgs-original/$CANDIDATE_COMMIT/path-backed
ENV_ROOT=/root/autodl-tmp/envs/ms-gcp-v13/3dgs-original/py310-torch2.7.1-cu128-v1
DATA_ROOT=/root/autodl-tmp/datasets/ms-gcp-v13/$RELEASE_DIGEST
RUN_ROOT=/root/autodl-tmp/runs/gs-gcp-v13/stage0-5/$RUN_ID
SPLIT=$ORCH_ROOT/configs/gs_gcp_rgb_holdout_split_manifest_v1.json
SAMPLES=$ORCH_ROOT/configs/gs_gcp_original_3dgs_camera_parity_samples_v1.json

test ! -e "$RUN_ROOT"
test -z "$(git -C "$ORCH_ROOT" status --porcelain)"
test -z "$(git -C "$REFERENCE_ROOT" status --porcelain)"
test -z "$(git -C "$CANDIDATE_ROOT" status --porcelain)"
test "$(git -C "$REFERENCE_ROOT" rev-parse HEAD)" = "$REFERENCE_COMMIT"
test "$(git -C "$CANDIDATE_ROOT" rev-parse HEAD)" = "$CANDIDATE_COMMIT"
mkdir -p "$RUN_ROOT"/{assets,reports,audit,tmp}
cp "${BASH_SOURCE[0]}" "$RUN_ROOT/audit/exact_launcher.sh"
cp "$SPLIT" "$SAMPLES" "$RUN_ROOT/audit/"

export CUDA_VISIBLE_DEVICES=0 CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$ENV_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export MALLOC_TRIM_THRESHOLD_=0
export TORCH_EXTENSIONS_DIR="$RUN_ROOT/tmp/torch_extensions" TMPDIR="$RUN_ROOT/tmp"
mkdir -p "$TORCH_EXTENSIONS_DIR"

"$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/original_3dgs_camera_compatibility.py" audit-all-images \
  --split_manifest "$SPLIT" --data_root "$DATA_ROOT" \
  --output "$RUN_ROOT/reports/full_6187_path_backed_image_audit.json"

for scene in \
  gcp_3000_20260602 gcp_5000_20260602 gcp_10000_20260610 \
  gcp_20000_20260602 gcp_50000_20260610 gcp_100000_20260610; do
  scene_root="$RUN_ROOT/assets/$scene"
  subset_root="$scene_root/parity_camera_subset"
  mkdir -p "$scene_root"
  "$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/original_3dgs_camera_compatibility.py" materialize-parity-subset \
    --sample_manifest "$SAMPLES" --scene "$scene" --source_root "$DATA_ROOT/$scene" \
    --output_root "$subset_root" > "$RUN_ROOT/reports/${scene}_materialize.log"
  "$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/original_3dgs_camera_load_preflight.py" \
    --method_root "$REFERENCE_ROOT" --source_root "$subset_root" --resolution 4 --data_device cuda \
    --host_allocator_policy glibc_malloc_trim_threshold_zero_v1 \
    --stabilization_seconds 1 --expected_materialization eager --include_tensor_hashes \
    --lifecycle_report "$RUN_ROOT/reports/${scene}_eager_lifecycle.jsonl" \
    --report "$RUN_ROOT/reports/${scene}_eager_camera_report.json"
  "$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/original_3dgs_camera_load_preflight.py" \
    --method_root "$CANDIDATE_ROOT" --source_root "$subset_root" --resolution 4 --data_device cuda \
    --host_allocator_policy glibc_malloc_trim_threshold_zero_v1 \
    --stabilization_seconds 1 --expected_materialization path_backed --include_tensor_hashes \
    --lifecycle_report "$RUN_ROOT/reports/${scene}_candidate_lifecycle.jsonl" \
    --report "$RUN_ROOT/reports/${scene}_candidate_camera_report.json"
  "$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/original_3dgs_camera_compatibility.py" compare-reports \
    --reference "$RUN_ROOT/reports/${scene}_eager_camera_report.json" \
    --candidate "$RUN_ROOT/reports/${scene}_candidate_camera_report.json" \
    --output "$RUN_ROOT/reports/${scene}_camera_parity.json"
done

"$ENV_ROOT/bin/python" - "$RUN_ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
rows = []
for path in sorted((root / "reports").glob("gcp_*_camera_parity.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows.append({"scene": path.name.removesuffix("_camera_parity.json"), "status": payload["status"]})
out = {"schema": "gs_gcp_original_3dgs_camera_parity_summary_v1", "status": "PASS" if len(rows) == 6 and all(r["status"] == "PASS" for r in rows) else "BLOCKER", "scenes": rows}
(root / "reports" / "camera_parity_summary.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if out["status"] != "PASS": raise SystemExit(1)
PY

echo "$RUN_ROOT"
