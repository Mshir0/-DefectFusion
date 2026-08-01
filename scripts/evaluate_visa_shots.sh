#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/mnt/sda1/VisA_20220922}"
MODEL="${MODEL:-/mnt/sda1/DINOv3/dinov3-vitl16-pretrain-lvd1689m}"

common_args=(
  --data-root "$DATA_ROOT"
  --model "$MODEL"
  --device cuda
  --defect-shots 0
  --seed 42
  --image-size 672
  --image-size-override macaroni2=896
  --image-size-override pcb2=896
  --image-size-override pcb3=896
  --pixel-image-size-override fryum=896
  --image-head-size-override pcb4=896
  --pixel-multiscale-size-override macaroni2=672
  --pixel-multiscale-size-override pcb2=672
  --pixel-multiscale-size-override pcb3=672
  --pixel-multiscale-weight 0.25
  --normal-augmentations rotate
  --affine-categories macaroni1 macaroni2
  --feature-layers=1,17,21,23
  --layer-aggregation mean
  --layer-normalization none
  --dual-branch
  --anomaly-method pca_knn_anoco
  --knn-weight 0.5
  --anoco-neighbors 16
  --anoco-query-weight 1.0
  --anoco-temperature 0.07
  --anoco-weight 0.25
  --anoco-layer-consensus
  --image-score mtop1p
  --image-top-ratio 0.01
  --image-min-component-size 2
  --component-reject-categories macaroni1 macaroni2
  --image-fusion-stage patch
  --memory-max-patches 50000
  --knn-chunk-size 256
  --knn-backend torch
  --knn-dtype float16
  --knn-spatial-radius -1
  --map-postprocess none
)

run_experiment() {
  local name="$1"
  local normal_shots="$2"
  local augment_count="$3"
  local fit_max_patches="$4"
  local output="outputs/visa-normal-$name"

  if [[ -f "$output/summary.csv" ]]; then
    echo "[visa-shots] skipping $name (complete: $output/summary.csv)"
    return
  fi

  echo "[visa-shots] starting $name"
  python -m defectfusion.cli evaluate-visa \
    "${common_args[@]}" \
    --normal-shots "$normal_shots" \
    --normal-augment-count "$augment_count" \
    --normal-fit-max-patches "$fit_max_patches" \
    --output "$output"
  echo "[visa-shots] completed $name"
}

run_experiment 1shot 1 30 0
run_experiment 2shot 2 30 0
run_experiment 4shot 4 30 0
run_experiment fullshot -1 0 50000

echo "[visa-shots] all experiments completed"
