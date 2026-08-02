#!/usr/bin/env bash
set -euo pipefail

: "${RUN_ID:?Set a unique RUN_ID}"
: "${PARITY_RUN_ROOT:?Set the completed camera-parity run root}"
: "${COMPATIBILITY_RUN_ROOT:?Set the completed A/B resource-preflight run root}"

RELEASE_DIGEST=513f8999fe4b110f15bcbecad7932895781cee755ee9ccd7a14ff10298546d75
SCENE=gcp_3000_20260602
REFERENCE_COMMIT=db8deebca67e8d5e1507e67c98de603eca0dfd85
CANDIDATE_COMMIT=ac8e622f7347cc01ed2af4e0381a05191025834b
ADAPTER_COMMIT=69842bcbcf1d3a159d08256a8cac557261234d36
RASTERIZER_COMMIT=c7c8ec385986ea5230dcdd517b8f6cc06db0049d
ORCH_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
REFERENCE_ROOT=/root/autodl-tmp/worktrees/gs-gcp-v13/3dgs-original/$REFERENCE_COMMIT/serialization-safe
CANDIDATE_ROOT=/root/autodl-tmp/worktrees/gs-gcp-v13/3dgs-original/$CANDIDATE_COMMIT/path-backed
ADAPTER_ROOT=/root/autodl-tmp/worktrees/gs-gcp-v13/3dgs-original-metric-adapter/${ADAPTER_COMMIT}-clean2
ADAPTER_SITE=/root/autodl-tmp/build/gs-gcp-v13/3dgs-original-metric-adapter/${ADAPTER_COMMIT}-retry2/site
ENV_ROOT=/root/autodl-tmp/envs/ms-gcp-v13/3dgs-original/py310-torch2.7.1-cu128-v1
DATA_ROOT=/root/autodl-tmp/datasets/ms-gcp-v13/$RELEASE_DIGEST
SOURCE_ROOT=$DATA_ROOT/$SCENE
RELEASE_ROOT=$DATA_ROOT/release_v1_3_0
RELEASE_CONFIG=$RELEASE_ROOT/gcp_benchmark_release_v1_3_0.json
ANNOTATIONS=$RELEASE_ROOT/${SCENE}_gcp_annotations_pixel_domain_v1_3_0.csv
RUN_ROOT=/root/autodl-tmp/runs/gs-gcp-v13/stage0-5/$RUN_ID
SPLIT_MANIFEST=$ORCH_ROOT/configs/gs_gcp_rgb_holdout_split_manifest_v1.json
RESOURCE_CONTRACT=$ORCH_ROOT/configs/gs_gcp_resource_probe_contract_v2.json
GNU_TIME=/root/autodl-tmp/tools/gs-gcp-v13/gnu-time/ubuntu-jammy-time-1.9-v1/root/usr/bin/time
STRACE=/root/autodl-tmp/tools/gs-gcp-v13/strace/root/usr/bin/strace
VGG16_WEIGHTS=/root/.cache/torch/hub/checkpoints/vgg16-397923af.pth
LPIPS_WEIGHTS=/root/.cache/torch/hub/checkpoints/vgg.pth
ASSET_ROOT=$RUN_ROOT/assets/$SCENE
TRAIN_ROOT=$ASSET_ROOT/train_camera_subset
TEST_ROOT=$ASSET_ROOT/test_camera_subset
MODEL_ROOT=$RUN_ROOT/02_checkpoints/model
EVAL_MODEL=$RUN_ROOT/05_packets/evaluation_model
PACKET_DIR=$RUN_ROOT/05_packets/metric_depth_packets
DEPTH_MANIFEST=$RUN_ROOT/05_packets/metric_depth_manifest.json
DEPTH_MAPPING=$RUN_ROOT/05_packets/depth_mapping.csv
COMPAT_ROOT=$RUN_ROOT/05_packets/compatibility_v1_1
FORMAL_ROOT=$RUN_ROOT/06_gcp_evaluation/formal_expected_camera_z
PARITY_SUMMARY=${PARITY_SUMMARY:-$PARITY_RUN_ROOT/reports/camera_parity_summary.json}

test ! -e "$RUN_ROOT"
for repo in "$ORCH_ROOT" "$REFERENCE_ROOT" "$CANDIDATE_ROOT" "$ADAPTER_ROOT"; do
  test -z "$(git -C "$repo" status --porcelain)"
done
test "$(git -C "$REFERENCE_ROOT" rev-parse HEAD)" = "$REFERENCE_COMMIT"
test "$(git -C "$CANDIDATE_ROOT" rev-parse HEAD)" = "$CANDIDATE_COMMIT"
test "$(git -C "$ADAPTER_ROOT" rev-parse HEAD)" = "$ADAPTER_COMMIT"
test "$(git -C "$ADAPTER_ROOT/submodules/diff-gaussian-rasterization" rev-parse HEAD)" = "$RASTERIZER_COMMIT"
test "$(sha256sum "$GNU_TIME" | awk '{print $1}')" = 7310b9b4c51a8f4d26c1af0da250f03a49ec8a8141033123e79196ad18f6c81b
test "$(sha256sum "$STRACE" | awk '{print $1}')" = 38a5c75cb29dd85ddd7780d54f5bf595554d7a1b5c42524b23065f5dc4c4b01d
test "$(sha256sum "$VGG16_WEIGHTS" | awk '{print $1}')" = 397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0
test "$(sha256sum "$LPIPS_WEIGHTS" | awk '{print $1}')" = a78928a0af1e5f0fcb1f3b9e8f8c3a2a5a3de244d830ad5c1feddc79b8432868

"$ENV_ROOT/bin/python" - "$PARITY_SUMMARY" \
  "$COMPATIBILITY_RUN_ROOT/preflight/selected_contract.json" > /dev/null <<'PY'
import json, sys
parity = json.load(open(sys.argv[1], encoding="utf-8"))
selection = json.load(open(sys.argv[2], encoding="utf-8"))
if parity.get("status") != "PASS" or selection.get("status") not in {"SELECTED_A", "SELECTED_B"}:
    raise SystemExit(f"unmet parity/selection gate: {parity.get('status')} {selection.get('status')}")
PY

SELECTED_STATUS=$("$ENV_ROOT/bin/python" -c "import json; print(json.load(open('$COMPATIBILITY_RUN_ROOT/preflight/selected_contract.json'))['status'])")
if [[ "$SELECTED_STATUS" == SELECTED_A ]]; then
  SELECTED_DEVICE=cuda
  SELECTED_RESIDENCY=cuda_resident_official
else
  SELECTED_DEVICE=cpu
  SELECTED_RESIDENCY=cpu_backed_official
fi

mkdir -p "$RUN_ROOT"/{00_preflight,01_micro,02_checkpoints,03_render,04_rgb_metrics,05_packets,06_gcp_evaluation,07_measurement,08_audit,tmp}
cp "${BASH_SOURCE[0]}" "$RUN_ROOT/08_audit/exact_launcher.sh"
cp "$SPLIT_MANIFEST" "$RESOURCE_CONTRACT" "$RUN_ROOT/08_audit/"
cp /root/autodl-tmp/tools/gs-gcp-v13/strace/version.txt \
  /root/autodl-tmp/tools/gs-gcp-v13/strace/SHA256SUMS "$RUN_ROOT/08_audit/"
cp "$PARITY_SUMMARY" "$RUN_ROOT/00_preflight/camera_parity_summary.json"
cp "$COMPATIBILITY_RUN_ROOT/preflight/selected_contract.json" "$RUN_ROOT/00_preflight/"
printf '%s\n' "$SELECTED_RESIDENCY" > "$RUN_ROOT/00_preflight/original_3dgs_data_residency.txt"
git -C "$ORCH_ROOT" rev-parse HEAD > "$RUN_ROOT/00_preflight/orchestration_commit.txt"
git -C "$CANDIDATE_ROOT" rev-parse HEAD > "$RUN_ROOT/00_preflight/method_commit.txt"
git -C "$CANDIDATE_ROOT" rev-parse 'HEAD^{tree}' > "$RUN_ROOT/00_preflight/method_tree.txt"
git -C "$ORCH_ROOT" bundle create "$RUN_ROOT/08_audit/orchestration.bundle" --all
git -C "$CANDIDATE_ROOT" bundle create "$RUN_ROOT/08_audit/original_3dgs_compatibility.bundle" --all

export CUDA_VISIBLE_DEVICES=0 CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$ENV_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$ADAPTER_SITE:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ADAPTER_SITE:$ADAPTER_ROOT"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export MALLOC_TRIM_THRESHOLD_=0
export TORCH_EXTENSIONS_DIR="$RUN_ROOT/tmp/torch_extensions" TMPDIR="$RUN_ROOT/tmp"
mkdir -p "$TORCH_EXTENSIONS_DIR"

"$ENV_ROOT/bin/python" -m py_compile \
  "$ORCH_ROOT/code/gcp/gs_gcp_common_measurement.py" \
  "$ORCH_ROOT/code/gcp/export_gaussian_depth_maps.py" \
  "$ORCH_ROOT/code/gcp/gcp_packet_camera_compatibility.py" \
  "$ORCH_ROOT/code/gcp/evaluate_gaussian_gcp_geometry.py" \
  > "$RUN_ROOT/00_preflight/py_compile.txt" 2>&1
"$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/test_gcp_release_v1_3.py" --real_release_dir "$RELEASE_ROOT" \
  > "$RUN_ROOT/00_preflight/real_release_v1_3_tests.txt" 2>&1
"$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/test_gcp_evaluator_protocol.py" \
  > "$RUN_ROOT/00_preflight/evaluator_protocol_tests.txt" 2>&1
cat > "$RUN_ROOT/00_preflight/legacy_v1_2_2_packet_compatibility_test_scope.json" <<'JSON'
{
  "decision": "not_run_not_part_of_v1_3_stage0_5",
  "reason": "The legacy test fixture is bound to withdrawn v1.2.2/R8 artifacts and Windows-only source paths. The v1.3.0 packet-camera contract is validated against the newly exported formal packet set by the runtime wrapper and release-mode evaluator gates later in this launcher.",
  "test": "test_gcp_packet_camera_compatibility.py"
}
JSON

"$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/gs_gcp_stage0_5.py" materialize-subsets \
  --split_manifest "$SPLIT_MANIFEST" --scene "$SCENE" --source_root "$SOURCE_ROOT" \
  --output_root "$ASSET_ROOT" --image_mode symlink > "$RUN_ROOT/00_preflight/materialize.log"
"$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/gs_gcp_stage0_5.py" generate-gt \
  --split_manifest "$SPLIT_MANIFEST" --scene "$SCENE" --source_root "$SOURCE_ROOT" \
  --output_root "$RUN_ROOT/03_render/benchmark_gt" > "$RUN_ROOT/00_preflight/generate_gt.log"

"$STRACE" -f -e trace=open,openat,openat2 -o "$RUN_ROOT/00_preflight/camera_access.strace" \
  "$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/original_3dgs_camera_load_preflight.py" \
    --method_root "$CANDIDATE_ROOT" --source_root "$TRAIN_ROOT" \
    --report "$RUN_ROOT/00_preflight/3k_camera_report.json" --resolution 4 \
    --data_device "$SELECTED_DEVICE" --stabilization_seconds 1 \
    --host_allocator_policy glibc_malloc_trim_threshold_zero_v1 \
    --expected_materialization path_backed --lifecycle_report "$RUN_ROOT/00_preflight/3k_lifecycle.jsonl"
"$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/validate_stage0_5_file_access.py" \
  --trace "$RUN_ROOT/00_preflight/camera_access.strace" --train_root "$TRAIN_ROOT" \
  --forbidden_root "$TEST_ROOT" --forbidden_root "$ASSET_ROOT/common_full_sfm" \
  --output "$RUN_ROOT/00_preflight/training_file_access_validation.json"

SYNTHETIC=$RUN_ROOT/01_micro/synthetic.bin
"$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/stage0_5_probe_synthetic_child.py" --output "$SYNTHETIC"
mv "$SYNTHETIC" "$RUN_ROOT/01_micro/synthetic_direct.bin"
"$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/run_with_resource_probe_v2.py" \
  --contract "$RESOURCE_CONTRACT" --output_dir "$RUN_ROOT/01_micro/synthetic_probe" \
  --working_directory "$ORCH_ROOT" --gpu_indices 0 --time_binary "$GNU_TIME" --failure_stage synthetic_probe -- \
  "$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/stage0_5_probe_synthetic_child.py" --output "$SYNTHETIC"
cmp "$RUN_ROOT/01_micro/synthetic_direct.bin" "$SYNTHETIC"
sha256sum "$RUN_ROOT/01_micro/synthetic_direct.bin" "$SYNTHETIC" > "$RUN_ROOT/01_micro/synthetic.sha256"

MICRO_WORK=$RUN_ROOT/01_micro/work
mkdir -p "$MICRO_WORK"
run_micro_direct() {
  local label="$1" method="$2" device="$3" port="$4"
  local model=$MICRO_WORK/model trace=$MICRO_WORK/trace.json
  test ! -e "$model"; test ! -e "$trace"
  "$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/original_3dgs_micro_child.py" \
    --method_root "$method" --trace_path "$trace" -- \
    --source_path "$TRAIN_ROOT" --model_path "$model" --images images --resolution 4 \
    --data_device "$device" --iterations 100 --save_iterations 100 --test_iterations 7000 \
    --quiet --ip 127.0.0.1 --port "$port" \
    > "$RUN_ROOT/01_micro/${label}_stdout.log" 2> "$RUN_ROOT/01_micro/${label}_stderr.log"
  mv "$model" "$RUN_ROOT/01_micro/${label}_model"
  mv "$trace" "$RUN_ROOT/01_micro/${label}_trace.json"
}
run_micro_probed() {
  local label="$1" method="$2" device="$3" port="$4"
  local model=$MICRO_WORK/model trace=$MICRO_WORK/trace.json
  test ! -e "$model"; test ! -e "$trace"
  "$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/run_with_resource_probe_v2.py" \
    --contract "$RESOURCE_CONTRACT" --output_dir "$RUN_ROOT/01_micro/${label}_resource" \
    --working_directory "$ORCH_ROOT" --gpu_indices 0 --time_binary "$GNU_TIME" --failure_stage micro_training -- \
    "$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/original_3dgs_micro_child.py" \
      --method_root "$method" --trace_path "$trace" -- \
      --source_path "$TRAIN_ROOT" --model_path "$model" --images images --resolution 4 \
      --data_device "$device" --iterations 100 --save_iterations 100 --test_iterations 7000 \
      --quiet --ip 127.0.0.1 --port "$port"
  mv "$model" "$RUN_ROOT/01_micro/${label}_model"
  mv "$trace" "$RUN_ROOT/01_micro/${label}_trace.json"
}

run_micro_direct eager_reference "$REFERENCE_ROOT" cuda 6031
run_micro_direct candidate_a "$CANDIDATE_ROOT" cuda 6032
"$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/compare_original_3dgs_micro_runs.py" \
  --direct_model "$RUN_ROOT/01_micro/eager_reference_model" \
  --probed_model "$RUN_ROOT/01_micro/candidate_a_model" \
  --direct_trace "$RUN_ROOT/01_micro/eager_reference_trace.json" \
  --probed_trace "$RUN_ROOT/01_micro/candidate_a_trace.json" \
  --output "$RUN_ROOT/01_micro/eager_vs_candidate_a.json"

if [[ "$SELECTED_DEVICE" == cpu ]]; then
  run_micro_direct selected_direct "$CANDIDATE_ROOT" cpu 6033
  "$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/compare_original_3dgs_micro_runs.py" \
    --direct_model "$RUN_ROOT/01_micro/candidate_a_model" \
    --probed_model "$RUN_ROOT/01_micro/selected_direct_model" \
    --direct_trace "$RUN_ROOT/01_micro/candidate_a_trace.json" \
    --probed_trace "$RUN_ROOT/01_micro/selected_direct_trace.json" \
    --allowed_argv_value_option=--data_device \
    --output "$RUN_ROOT/01_micro/candidate_a_vs_candidate_b.json"
else
  ln -s "$RUN_ROOT/01_micro/candidate_a_model" "$RUN_ROOT/01_micro/selected_direct_model"
  ln -s "$RUN_ROOT/01_micro/candidate_a_trace.json" "$RUN_ROOT/01_micro/selected_direct_trace.json"
fi
run_micro_probed selected_probed "$CANDIDATE_ROOT" "$SELECTED_DEVICE" 6034
"$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/compare_original_3dgs_micro_runs.py" \
  --direct_model "$RUN_ROOT/01_micro/selected_direct_model" \
  --probed_model "$RUN_ROOT/01_micro/selected_probed_model" \
  --direct_trace "$RUN_ROOT/01_micro/selected_direct_trace.json" \
  --probed_trace "$RUN_ROOT/01_micro/selected_probed_trace.json" \
  --output "$RUN_ROOT/01_micro/selected_direct_vs_probed.json"

train_command=(
  "$ENV_ROOT/bin/python" "$CANDIDATE_ROOT/train.py" --source_path "$TRAIN_ROOT" --model_path "$MODEL_ROOT"
  --images images --resolution 4 --sh_degree 3 --data_device "$SELECTED_DEVICE" --iterations 30000
  --position_lr_init 0.00016 --position_lr_final 0.0000016 --position_lr_delay_mult 0.01
  --position_lr_max_steps 30000 --feature_lr 0.0025 --opacity_lr 0.05 --scaling_lr 0.005
  --rotation_lr 0.001 --percent_dense 0.01 --lambda_dssim 0.2 --densification_interval 100
  --opacity_reset_interval 3000 --densify_from_iter 500 --densify_until_iter 15000
  --densify_grad_threshold 0.0002 --test_iterations 7000 30000 --save_iterations 7000 30000
  --quiet --ip 127.0.0.1 --port 6035
)
printf '%q ' "${train_command[@]}" > "$RUN_ROOT/08_audit/exact_training_command.sh"; printf '\n' >> "$RUN_ROOT/08_audit/exact_training_command.sh"
"$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/run_with_resource_probe_v2.py" \
  --contract "$RESOURCE_CONTRACT" --output_dir "$RUN_ROOT/02_checkpoints/training_resource" \
  --working_directory "$CANDIDATE_ROOT" --gpu_indices 0 --time_binary "$GNU_TIME" \
  --enforce_contract_gates --failure_stage training -- "${train_command[@]}"
test -f "$MODEL_ROOT/point_cloud/iteration_30000/point_cloud.ply"
sha256sum "$MODEL_ROOT/point_cloud/iteration_30000/point_cloud.ply" > "$RUN_ROOT/08_audit/formal_checkpoint.sha256"

"$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/run_with_resource_probe_v2.py" \
  --contract "$RESOURCE_CONTRACT" --output_dir "$RUN_ROOT/03_render/heldout_render_resource" \
  --working_directory "$ORCH_ROOT" --gpu_indices 0 --time_binary "$GNU_TIME" --failure_stage heldout_render -- \
  "$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/gs_gcp_common_measurement.py" render-heldout \
    --method_root "$CANDIDATE_ROOT" --model_path "$MODEL_ROOT" --test_source "$TEST_ROOT" \
    --iteration 30000 --output_dir "$RUN_ROOT/03_render/heldout"
"$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/run_with_resource_probe_v2.py" \
  --contract "$RESOURCE_CONTRACT" --output_dir "$RUN_ROOT/03_render/reference_benchmark_resource" \
  --working_directory "$ORCH_ROOT" --gpu_indices 0 --time_binary "$GNU_TIME" --failure_stage render_benchmark -- \
  "$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/gs_gcp_common_measurement.py" render-benchmark \
    --method_root "$CANDIDATE_ROOT" --model_path "$MODEL_ROOT" --test_source "$TEST_ROOT" \
    --iteration 30000 --gpu_index 0 --output "$RUN_ROOT/03_render/single_gpu_reference_render.json"
"$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/run_with_resource_probe_v2.py" \
  --contract "$RESOURCE_CONTRACT" --output_dir "$RUN_ROOT/04_rgb_metrics/resource" \
  --working_directory "$ORCH_ROOT" --gpu_indices 0 --time_binary "$GNU_TIME" --failure_stage rgb_metrics -- \
  "$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/gs_gcp_common_measurement.py" rgb-metrics \
    --render_dir "$RUN_ROOT/03_render/heldout/renders" --gt_root "$RUN_ROOT/03_render/benchmark_gt" \
    --gt_manifest "$RUN_ROOT/03_render/benchmark_gt/GT_MANIFEST.json" --method_root "$CANDIDATE_ROOT" \
    --output_dir "$RUN_ROOT/04_rgb_metrics/results" --device cuda \
    --vgg16_weights "$VGG16_WEIGHTS" --lpips_weights "$LPIPS_WEIGHTS"
"$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/gs_gcp_common_measurement.py" inspect-3dgs \
  --model_path "$MODEL_ROOT" --iteration 30000 --output "$RUN_ROOT/07_measurement/representation.json"

"$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/gs_gcp_common_measurement.py" prepare-eval-model \
  --trained_model "$MODEL_ROOT" --evaluation_model "$EVAL_MODEL" --full_source "$SOURCE_ROOT" \
  > "$RUN_ROOT/05_packets/prepare_evaluation_model.log"
packet_command=(
  "$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/export_gaussian_depth_maps.py"
  --train_repo "$ADAPTER_ROOT" --source_path "$SOURCE_ROOT" --model_path "$EVAL_MODEL"
  --images images --resolution 4 --iteration 30000 --camera_sets all
  --image_list_csv "$ANNOTATIONS" --image_name_column target_image_name
  --image_list_status_column formal_eligible --image_list_status_values true
  --depth_output_dir "$PACKET_DIR" --manifest_path "$DEPTH_MANIFEST" --mapping_csv "$DEPTH_MAPPING"
)
printf '%q ' "${packet_command[@]}" > "$RUN_ROOT/05_packets/exact_packet_export_command.sh"; printf '\n' >> "$RUN_ROOT/05_packets/exact_packet_export_command.sh"
"$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/run_with_resource_probe_v2.py" \
  --contract "$RESOURCE_CONTRACT" --output_dir "$RUN_ROOT/05_packets/resource" \
  --working_directory "$ORCH_ROOT" --gpu_indices 0 --time_binary "$GNU_TIME" \
  --enforce_contract_gates --failure_stage packet_export -- "${packet_command[@]}"
"$ENV_ROOT/bin/python" - "$ANNOTATIONS" "$DEPTH_MAPPING" "$DEPTH_MANIFEST" <<'PY' > "$RUN_ROOT/05_packets/post_export_validation.json"
import csv, json, sys
from pathlib import Path
annotations, mapping, manifest = map(Path, sys.argv[1:])
rows = list(csv.DictReader(annotations.open("r", encoding="utf-8-sig", newline="")))
expected = {r["target_image_name"] for r in rows if r["formal_eligible"].lower() == "true"}
mapped = list(csv.DictReader(mapping.open("r", encoding="utf-8", newline="")))
actual = {r["image_name"] for r in mapped}
payload = json.load(manifest.open(encoding="utf-8"))
result = {
    "expected_count": len(expected), "mapping_count": len(mapped), "actual_count": len(actual),
    "missing": sorted(expected-actual), "extra": sorted(actual-expected),
    "packet_schema": payload.get("packet_schema"), "primary_depth_tensor": payload.get("primary_depth_tensor"),
    "primary_depth_semantics": payload.get("primary_depth_semantics"), "formal_depth_formula": payload.get("formal_depth_formula"),
    "all_packet_recompute_passed": all(r["packet_recompute_passed"].lower() == "true" for r in mapped),
    "variance_consistency_fail_count_sum": sum(int(r["variance_consistency_fail_count"]) for r in mapped),
}
result["passed"] = len(expected)==len(mapped)==len(actual)==66 and expected==actual and payload.get("rendered_view_count")==66 and payload.get("packet_schema")=="ms_gcp_metric_depth_packet_v2" and payload.get("primary_depth_tensor")=="alpha_normalized_expected_camera_z" and payload.get("primary_depth_semantics")=="camera_z" and payload.get("formal_depth_formula")=="M1/A" and result["all_packet_recompute_passed"] and result["variance_consistency_fail_count_sum"]==0
print(json.dumps(result, indent=2, sort_keys=True))
if not result["passed"]: raise SystemExit(1)
PY
find "$PACKET_DIR" -maxdepth 1 -type f -name '*.npz' -printf '%f\0' | sort -z | xargs -0 -I{} sha256sum "$PACKET_DIR/{}" > "$RUN_ROOT/05_packets/packet_files.sha256"
test "$(wc -l < "$RUN_ROOT/05_packets/packet_files.sha256")" -eq 66

mkdir -p "$COMPAT_ROOT"
"$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/gcp_packet_camera_compatibility.py" \
  --release_config "$RELEASE_CONFIG" --scene "$SCENE" --depth_manifest "$DEPTH_MANIFEST" \
  --model_dir "$EVAL_MODEL" --renderer_repo "$ADAPTER_ROOT" --out_dir "$COMPAT_ROOT" \
  --packet_search_root "$PACKET_DIR" --require_local_packets > "$COMPAT_ROOT/wrapper_generation_stdout.json"
WRAPPER=$COMPAT_ROOT/metric_depth_packet_resolution_compatibility_v1_1.json
WRAPPER_SHA=$(sha256sum "$WRAPPER" | awk '{print $1}')
"$ENV_ROOT/bin/python" - "$ORCH_ROOT" "$WRAPPER" "$WRAPPER_SHA" "$DEPTH_MANIFEST" "$RELEASE_CONFIG" "$RELEASE_ROOT" "$SCENE" "$PACKET_DIR" <<'PY' > "$COMPAT_ROOT/runtime_wrapper_validation.json"
import json, sys
from pathlib import Path
bench, wrapper, expected_sha, manifest_path, config_path, release_root, scene, packet_dir = sys.argv[1:]
sys.path.insert(0, str(Path(bench)/"code"/"gcp"))
from gcp_packet_camera_compatibility import validate_compatibility_wrapper
result = validate_compatibility_wrapper(Path(wrapper), expected_wrapper_sha256=expected_sha, depth_manifest=json.load(open(manifest_path)), depth_manifest_path=Path(manifest_path), release_config=json.load(open(config_path)), release_dir=Path(release_root), scene=scene, patch_size=7, packet_search_roots=[Path(packet_dir)], require_local_packets=True)
print(json.dumps(result, indent=2, sort_keys=True))
PY

eval_command=(
  "$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/evaluate_gaussian_gcp_geometry.py"
  --release_config "$RELEASE_CONFIG" --scene "$SCENE" --method_id original_3dgs
  --colmap_model "$SOURCE_ROOT/sparse/0" --depth_manifest "$DEPTH_MANIFEST"
  --packet_compatibility_manifest "$WRAPPER" --packet_compatibility_manifest_sha256 "$WRAPPER_SHA"
  --out_dir "$FORMAL_ROOT"
)
printf '%q ' "${eval_command[@]}" > "$RUN_ROOT/06_gcp_evaluation/exact_formal_evaluator_command.sh"; printf '\n' >> "$RUN_ROOT/06_gcp_evaluation/exact_formal_evaluator_command.sh"
"$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/run_with_resource_probe_v2.py" \
  --contract "$RESOURCE_CONTRACT" --output_dir "$RUN_ROOT/06_gcp_evaluation/resource" \
  --working_directory "$ORCH_ROOT" --gpu_indices 0 --time_binary "$GNU_TIME" --failure_stage formal_evaluation -- \
  "${eval_command[@]}"
"$ENV_ROOT/bin/python" "$ORCH_ROOT/code/gcp/verify_gaussian_gcp_eval_outputs.py" \
  --eval_dir "$FORMAL_ROOT" --out "$RUN_ROOT/06_gcp_evaluation/independent_recomputation_v1.json" --tolerance 1e-9 \
  > "$RUN_ROOT/06_gcp_evaluation/independent_recomputation_stdout.txt"

sha256sum -c "$RUN_ROOT/05_packets/packet_files.sha256" > "$RUN_ROOT/08_audit/packet_sha_post_evaluation.txt"
for repo in "$ORCH_ROOT" "$REFERENCE_ROOT" "$CANDIDATE_ROOT" "$ADAPTER_ROOT"; do
  test -z "$(git -C "$repo" status --porcelain)"
done
find "$RUN_ROOT" -type f ! -path '*/tmp/*' -printf '%P\0' | sort -z | xargs -0 -I{} sha256sum "$RUN_ROOT/{}" > "$RUN_ROOT/08_audit/run_outputs.sha256"
printf 'RUN_ROOT=%s\nSELECTED_CONTRACT=%s\n' "$RUN_ROOT" "$SELECTED_RESIDENCY"
