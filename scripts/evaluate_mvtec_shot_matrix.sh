#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/mnt/sda1/mvtec_anomaly}"
MODEL="${MODEL:-/mnt/sda1/DINOv3/dinov3-vitl16-pretrain-lvd1689m}"

common_args=(
  --data-root "$DATA_ROOT"
  --model "$MODEL"
  --device cuda
  --seed 42
  --image-size 672
  --pixel-image-size-override cable=896
  --pixel-image-size-override transistor=896
  --pixel-multiscale-size-override cable=672
  --pixel-multiscale-size-override transistor=672
  --pixel-multiscale-weight 0.25
  --resize-mode direct
  --normal-augmentations rotate
  --no-augment-categories transistor
  --feature-layers=1,17,21,23
  --layer-aggregation mean
  --layer-normalization none
  --dual-branch
  --anomaly-method pca_knn_anoco
  --knn-weight 0.5
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
  --knn-chunk-size 256
  --knn-backend torch
  --knn-dtype float16
  --knn-spatial-radius -1
  --map-postprocess none
  --type-matching bidirectional_patch
  --top-k-ratio 0.05
)

run_experiment() {
  local normal_name="$1"
  local normal_shots="$2"
  local defect_shots="$3"
  local augment_count="$4"
  local name="normal-${normal_name}-defect-${defect_shots}shot"

  echo "[shot-matrix] starting $name"
  python -m defectfusion.cli evaluate-mvtec \
    "${common_args[@]}" \
    --normal-shots "$normal_shots" \
    --defect-shots "$defect_shots" \
    --normal-augment-count "$augment_count" \
    --output "outputs/mvtec-$name"
  echo "[shot-matrix] completed $name"
}

run_experiment 1shot 1 0 30
run_experiment 1shot 1 1 30

run_experiment 2shot 2 0 30
run_experiment 2shot 2 1 30
run_experiment 2shot 2 2 30

run_experiment 4shot 4 0 30
run_experiment 4shot 4 1 30
run_experiment 4shot 4 2 30
run_experiment 4shot 4 4 30

run_experiment fullshot -1 0 0
run_experiment fullshot -1 1 0
run_experiment fullshot -1 2 0
run_experiment fullshot -1 4 0

echo "[shot-matrix] all experiments completed"
