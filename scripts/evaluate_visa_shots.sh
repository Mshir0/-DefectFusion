#!/usr/bin/env bash
set -euo pipefail

# Final VisA protocol: 1/2/4/8 normal-only shots and 8-normal defect-typing runs.
DATA_ROOT="${DATA_ROOT:-/mnt/sda1/VisA_20220922}"
MODEL="${MODEL:-/mnt/sda1/DINOv3/dinov3-vitl16-pretrain-lvd1689m}"
SPLIT_CSV="${SPLIT_CSV:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
NORMAL_FIT_MAX_PATCHES="${NORMAL_FIT_MAX_PATCHES:-50000}"
OOM_RETRY_FIT_MAX_PATCHES="${OOM_RETRY_FIT_MAX_PATCHES:-30000}"
MODE="${MODE:-matrix}"
TUNING_FAMILY="${TUNING_FAMILY:-all}"
DETECTION_BALANCED_THRESHOLD="${DETECTION_BALANCED_THRESHOLD:-0.80}"
TYPE_MACRO_F1_THRESHOLD="${TYPE_MACRO_F1_THRESHOLD:-0.60}"

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
  --normal-fit-max-patches "$NORMAL_FIT_MAX_PATCHES"
  --knn-chunk-size 256
  --knn-backend torch
  --knn-dtype float16
  --knn-spatial-radius 0.10
  --knn-spatial-categories pcb2 pcb3 pcb4
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
  set +e
  "$PYTHON" -m defectfusion.cli evaluate-visa \
      "${common_args[@]}" \
      --normal-shots "$normal_shots" \
      --defect-shots "$defect_shots" \
      --output "$output"
  local status="$?"
  set -e
  if [[ "$status" -ne 0 ]]; then
    if [[ "$status" -eq 137 && "$OOM_RETRY_FIT_MAX_PATCHES" =~ ^[1-9][0-9]*$ ]]; then
      printf '[visa-shots] %s was killed; retrying from the beginning with normal_fit_max_patches=%s\n' \
        "$name" "$OOM_RETRY_FIT_MAX_PATCHES" >&2
      "$PYTHON" -m defectfusion.cli evaluate-visa \
        "${common_args[@]}" \
        --normal-fit-max-patches "$OOM_RETRY_FIT_MAX_PATCHES" \
        --normal-shots "$normal_shots" \
        --defect-shots "$defect_shots" \
        --output "$output"
    else
      return "$status"
    fi
  fi
  printf '[visa-shots] completed %s\n' "$name"
}

run_tuning_experiment() {
  local family="$1"
  local source="$2"
  local normal_shots="$3"
  local defect_shots="$4"
  local category="$5"
  local variant="$6"
  shift 6
  local output="$OUTPUT_ROOT/visa-tuning/$family/$source/$category/$variant"

  if [[ "$SKIP_COMPLETED" == "1" && -f "$output/results.json" ]]; then
    printf '[visa-tuning] skipping %s/%s/%s/%s (complete)\n' "$family" "$source" "$category" "$variant"
    return
  fi

  printf '[visa-tuning] starting %s/%s/%s/%s\n' "$family" "$source" "$category" "$variant"
  set +e
  "$PYTHON" -m defectfusion.cli evaluate-visa \
      "${common_args[@]}" \
      --categories "$category" \
      --normal-shots "$normal_shots" \
      --defect-shots "$defect_shots" \
      --output "$output" \
      "$@"
  local status="$?"
  set -e
  if [[ "$status" -ne 0 ]]; then
    if [[ "$status" -eq 137 && "$OOM_RETRY_FIT_MAX_PATCHES" =~ ^[1-9][0-9]*$ ]]; then
      printf '[visa-tuning] %s/%s/%s/%s was killed; retrying with normal_fit_max_patches=%s\n' \
        "$family" "$source" "$category" "$variant" "$OOM_RETRY_FIT_MAX_PATCHES" >&2
      "$PYTHON" -m defectfusion.cli evaluate-visa \
        "${common_args[@]}" \
        --normal-fit-max-patches "$OOM_RETRY_FIT_MAX_PATCHES" \
        --categories "$category" \
        --normal-shots "$normal_shots" \
        --defect-shots "$defect_shots" \
        --output "$output" \
        "$@"
    else
      return "$status"
    fi
  fi
  printf '[visa-tuning] completed %s/%s/%s/%s\n' "$family" "$source" "$category" "$variant"
}

run_tuning_matrix() {
  if [[ "$TUNING_FAMILY" == "all" || "$TUNING_FAMILY" == "detection" ]]; then
    for normal_shots in 1 2 4 8; do
      local source="normal-${normal_shots}shot-defect-0shot"
      local results="$OUTPUT_ROOT/visa-$source/results.json"
      if [[ ! -f "$results" ]]; then
        printf '[visa-tuning] skipping detection/%s: missing %s\n' "$source" "$results" >&2
        continue
      fi
      mapfile -t categories < <("$PYTHON" -m defectfusion.tuning \
        --results "$results" --metric balanced_accuracy --below "$DETECTION_BALANCED_THRESHOLD")
      printf '[visa-tuning] detection/%s selected=%s threshold=%s\n' \
        "$source" "${categories[*]:-none}" "$DETECTION_BALANCED_THRESHOLD"
      for category in "${categories[@]}"; do
        run_tuning_experiment detection "$source" "$normal_shots" 0 "$category" augmentation-q099 \
          --normal-decision-calibration augmentation \
          --normal-decision-quantile 0.99 \
          --normal-decision-quantile-method higher \
          --normal-decision-augment-count 30 \
          --normal-decision-seed 142
        run_tuning_experiment detection "$source" "$normal_shots" 0 "$category" augmentation-max \
          --normal-decision-calibration augmentation \
          --normal-decision-quantile 1.0 \
          --normal-decision-quantile-method higher \
          --normal-decision-augment-count 30 \
          --normal-decision-seed 142
      done
    done
  fi

  if [[ "$TUNING_FAMILY" == "all" || "$TUNING_FAMILY" == "typing" ]]; then
    for defect_shots in 1 2 4 8; do
      local source="normal-8shot-defect-${defect_shots}shot"
      local results="$OUTPUT_ROOT/visa-$source/results.json"
      if [[ ! -f "$results" ]]; then
        printf '[visa-tuning] skipping typing/%s: missing %s\n' "$source" "$results" >&2
        continue
      fi
      mapfile -t categories < <("$PYTHON" -m defectfusion.tuning \
        --results "$results" --metric defect_type_macro_f1 --below "$TYPE_MACRO_F1_THRESHOLD")
      printf '[visa-tuning] typing/%s selected=%s threshold=%s\n' \
        "$source" "${categories[*]:-none}" "$TYPE_MACRO_F1_THRESHOLD"
      for category in "${categories[@]}"; do
        run_tuning_experiment typing "$source" 8 "$defect_shots" "$category" bidirectional-top10 \
          --type-matching bidirectional_patch --top-k-ratio 0.10
        run_tuning_experiment typing "$source" 8 "$defect_shots" "$category" prototype-mean \
          --type-matching prototype_mean --top-k-ratio 0.05
        run_tuning_experiment typing "$source" 8 "$defect_shots" "$category" rbf-svm \
          --type-matching rbf_svm --top-k-ratio 0.05
      done
    done
  fi
}

case "$MODE" in
  matrix)
    for shots in 1 2 4 8; do
      run_experiment "$shots" 0
    done
    for defect_shots in 1 2 4 8; do
      run_experiment 8 "$defect_shots"
    done
    ;;
  tune)
    if [[ "$TUNING_FAMILY" != "all" && "$TUNING_FAMILY" != "detection" && "$TUNING_FAMILY" != "typing" ]]; then
      printf 'TUNING_FAMILY must be all, detection, or typing, got: %s\n' "$TUNING_FAMILY" >&2
      exit 2
    fi
    run_tuning_matrix
    ;;
  *)
    printf 'MODE must be matrix or tune, got: %s\n' "$MODE" >&2
    exit 2
    ;;
esac

printf '[visa-shots] mode=%s completed\n' "$MODE"
