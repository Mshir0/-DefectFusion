#!/usr/bin/env bash
set -euo pipefail

# Final MVTec protocol: 1/2/4/8 normal-only shots and 8-normal defect-typing runs.
DATA_ROOT="${DATA_ROOT:-/mnt/sda1/mvtec_anomaly}"
MODEL="${MODEL:-/mnt/sda1/DINOv3/dinov3-vitl16-pretrain-lvd1689m}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
NORMAL_FIT_MAX_PATCHES="${NORMAL_FIT_MAX_PATCHES:-50000}"
OOM_RETRY_FIT_MAX_PATCHES="${OOM_RETRY_FIT_MAX_PATCHES:-30000}"

common_args=(
  --data-root "$DATA_ROOT"
  --model "$MODEL"
  --device "$DEVICE"
  --seed "$SEED"
  --image-size 672
  --pixel-image-size-override cable=896
  --pixel-image-size-override transistor=896
  --pixel-multiscale-size-override cable=672
  --pixel-multiscale-size-override transistor=672
  --pixel-multiscale-weight 0.25
  --resize-mode direct
  --normal-augment-count 30
  --normal-augmentations rotate
  --no-augment-categories transistor
  --feature-layers=1,17,21,23
  --layer-aggregation mean
  --layer-normalization none
  --dual-branch
  --anomaly-method pca_knn_anoco
  --knn-weight 0.5
  --pixel-anoco-weight 0.35
  --pixel-anoco-categories cable transistor
  --pixel-anoco-weight-override cable=0.10
  --pixel-anoco-norm-compatibility
  --pixel-anoco-norm-compatibility-categories transistor
  --anoco-neighbors 16
  --anoco-query-weight 2.0
  --anoco-temperature 0.07
  --anoco-affinity softmax
  --anoco-anchor-ranking mean
  --anoco-weight 0.25
  --anoco-layer-consensus
  --fusion-mode fixed
  --image-score mtop1p
  --image-top-ratio 0.01
  --image-min-component-size 1
  --image-fusion-stage patch
  --memory-max-patches 50000
  --normal-fit-max-patches "$NORMAL_FIT_MAX_PATCHES"
  --knn-chunk-size 256
  --knn-backend torch
  --knn-dtype float16
  --knn-spatial-radius 0.10
  --knn-spatial-categories cable transistor
  --map-postprocess none
  --type-matching bidirectional_patch
  --top-k-ratio 0.05
)
if [[ "$SKIP_COMPLETED" == "1" ]]; then
  common_args+=(--skip-completed-categories)
elif [[ "$SKIP_COMPLETED" != "0" ]]; then
  printf 'SKIP_COMPLETED must be 0 or 1, got: %s\n' "$SKIP_COMPLETED" >&2
  exit 2
fi

run_experiment() {
  local normal_shots="$1"
  local defect_shots="$2"
  local name="normal-${normal_shots}shot-defect-${defect_shots}shot"
  local output="$OUTPUT_ROOT/mvtec-$name"

  if [[ "$SKIP_COMPLETED" == "1" && -f "$output/results.json" ]]; then
    printf '[mvtec-shots] skipping %s (complete: %s)\n' "$name" "$output/results.json"
    return
  fi

  printf '[mvtec-shots] starting %s\n' "$name"
  set +e
  "$PYTHON" -m defectfusion.cli evaluate-mvtec \
      "${common_args[@]}" \
      --normal-shots "$normal_shots" \
      --defect-shots "$defect_shots" \
      --output "$output"
  local status="$?"
  set -e
  if [[ "$status" -ne 0 ]]; then
    if [[ "$status" -eq 137 && "$OOM_RETRY_FIT_MAX_PATCHES" =~ ^[1-9][0-9]*$ ]]; then
      printf '[mvtec-shots] %s was killed; retrying from the beginning with normal_fit_max_patches=%s\n' \
        "$name" "$OOM_RETRY_FIT_MAX_PATCHES" >&2
      "$PYTHON" -m defectfusion.cli evaluate-mvtec \
        "${common_args[@]}" \
        --normal-fit-max-patches "$OOM_RETRY_FIT_MAX_PATCHES" \
        --normal-shots "$normal_shots" \
        --defect-shots "$defect_shots" \
        --output "$output"
    else
      return "$status"
    fi
  fi
  printf '[mvtec-shots] completed %s\n' "$name"
}

for shots in 1 2 4 8; do
  run_experiment "$shots" 0
done
for defect_shots in 1 2 4 8; do
  run_experiment 8 "$defect_shots"
done

printf '[mvtec-shots] all experiments completed\n'
