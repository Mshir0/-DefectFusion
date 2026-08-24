#!/usr/bin/env bash
set -euo pipefail

# Final VisA protocol: 1/2/4/8 normal-only shots and one 8-normal/8-defect run.
DATA_ROOT="${DATA_ROOT:-/mnt/sda1/VisA_20220922}"
MODEL="${MODEL:-/mnt/sda1/DINOv3/dinov3-vitl16-pretrain-lvd1689m}"
SPLIT_CSV="${SPLIT_CSV:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
SKIP_COMPLETED="${SKIP_COMPLETED:-0}"

common_args=(
  --data-root "$DATA_ROOT"
  --model "$MODEL"
  --device "$DEVICE"
  --seed "$SEED"
  --image-size 672
  --image-size-override macaroni2=896
  --image-size-override pcb2=896
  --image-size-override pcb3=896
  --pixel-image-size-override fryum=896
  --pixel-image-size-override macaroni1=896
  --pixel-image-size-override pcb4=896
  --image-head-size-override pcb4=896
  --pixel-multiscale-size-override macaroni1=672
  --pixel-multiscale-size-override macaroni2=672
  --pixel-multiscale-size-override fryum=672
  --pixel-multiscale-size-override pcb2=672
  --pixel-multiscale-size-override pcb3=672
  --pixel-multiscale-size-override pcb4=672
  --pixel-multiscale-weight 0.25
  --normal-augment-count 30
  --normal-augmentations rotate
  --affine-categories macaroni1 macaroni2
  --feature-layers=1,17,21,23
  --layer-aggregation mean
  --layer-normalization none
  --dual-branch
  --anomaly-method pca_knn_anoco
  --knn-weight 0.5
  --pixel-anoco-weight 0.10
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
  --image-min-component-size 2
  --component-reject-categories macaroni1 macaroni2
  --image-fusion-stage patch
  --memory-max-patches 50000
  --knn-chunk-size 256
  --knn-backend torch
  --knn-dtype float16
  --knn-spatial-radius 0.10
  --knn-spatial-categories pcb2 pcb3 pcb4
  --map-postprocess none
  --type-matching bidirectional_patch
  --top-k-ratio 0.05
)
if [[ -n "$SPLIT_CSV" ]]; then
  if [[ ! -f "$SPLIT_CSV" ]]; then
    printf 'SPLIT_CSV does not exist: %s\n' "$SPLIT_CSV" >&2
    exit 2
  fi
  common_args+=(--split-csv "$SPLIT_CSV")
fi

run_experiment() {
  local normal_shots="$1"
  local defect_shots="$2"
  local name="normal-${normal_shots}shot-defect-${defect_shots}shot"
  local output="$OUTPUT_ROOT/visa-$name"

  if [[ "$SKIP_COMPLETED" == "1" && -f "$output/results.json" ]]; then
    printf '[visa-shots] skipping %s (complete: %s)\n' "$name" "$output/results.json"
    return
  fi

  printf '[visa-shots] starting %s\n' "$name"
  "$PYTHON" -m defectfusion.cli evaluate-visa \
    "${common_args[@]}" \
    --normal-shots "$normal_shots" \
    --defect-shots "$defect_shots" \
    --output "$output"
  printf '[visa-shots] completed %s\n' "$name"
}

for shots in 1 2 4 8; do
  run_experiment "$shots" 0
done
run_experiment 8 8

printf '[visa-shots] all experiments completed\n'
