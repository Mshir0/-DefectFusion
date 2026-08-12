#!/usr/bin/env bash
set -euo pipefail

# Main DefectFusion PCA evaluation only. No distilled student model, kNN,
# ANoCo, or dual branch is used. Good/anomaly decisions use the 99.5th
# percentile of held-out rotation scores generated with a separate seed.

DATASET="${DATASET:-all}"
PYTHON="${PYTHON:-python}"
MODEL="${MODEL:-/mnt/sda1/DINOv3/dinov3-vitl16-pretrain-lvd1689m}"
DEVICE="${DEVICE:-cuda}"
NORMAL_SHOTS="${NORMAL_SHOTS:-8}"
NORMAL_AUGMENT_COUNT="${NORMAL_AUGMENT_COUNT:-}"
NORMAL_FIT_MAX_PATCHES="${NORMAL_FIT_MAX_PATCHES:-0}"
NORMAL_DECISION_QUANTILE="${NORMAL_DECISION_QUANTILE:-0.995}"
NORMAL_DECISION_AUGMENT_COUNT="${NORMAL_DECISION_AUGMENT_COUNT:-}"
IMAGE_SIZE="${IMAGE_SIZE:-672}"
FEATURE_LAYERS="${FEATURE_LAYERS:-1,17,21,23}"
SEED="${SEED:-42}"
NORMAL_DECISION_SEED="${NORMAL_DECISION_SEED:-$((SEED + 100))}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/pca-good-accuracy-heldout}"
SKIP_COMPLETED="${SKIP_COMPLETED:-0}"

MVTEC_DATA_ROOT="${MVTEC_DATA_ROOT:-/mnt/sda1/mvtec_anomaly}"
VISA_DATA_ROOT="${VISA_DATA_ROOT:-/mnt/sda1/VisA_20220922}"
VISA_SPLIT_CSV="${VISA_SPLIT_CSV:-}"

if [[ "$DATASET" != "mvtec" && "$DATASET" != "visa" && "$DATASET" != "all" ]]; then
  printf '%s\n' 'DATASET must be mvtec, visa, or all.' >&2
  exit 2
fi
if [[ -z "$NORMAL_AUGMENT_COUNT" ]]; then
  if [[ "$NORMAL_SHOTS" == "-1" ]]; then
    NORMAL_AUGMENT_COUNT=0
  else
    NORMAL_AUGMENT_COUNT=30
  fi
fi
if [[ -z "$NORMAL_DECISION_AUGMENT_COUNT" ]]; then
  NORMAL_DECISION_AUGMENT_COUNT="$NORMAL_AUGMENT_COUNT"
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
cd "$repo_root"

common_args=(
  --model "$MODEL"
  --device "$DEVICE"
  --normal-shots "$NORMAL_SHOTS"
  --defect-shots 0
  --seed "$SEED"
  --normal-augment-count "$NORMAL_AUGMENT_COUNT"
  --normal-augmentations rotate
  --normal-fit-max-patches "$NORMAL_FIT_MAX_PATCHES"
  --normal-decision-quantile "$NORMAL_DECISION_QUANTILE"
  --normal-decision-augment-count "$NORMAL_DECISION_AUGMENT_COUNT"
  --normal-decision-seed "$NORMAL_DECISION_SEED"
  --image-size "$IMAGE_SIZE"
  --resize-mode direct
  --feature-layers="$FEATURE_LAYERS"
  --layer-aggregation mean
  --layer-normalization none
  --anomaly-method pca
  --pca-residual-metric squared_l2
  --image-score mtop1p
  --image-top-ratio 0.01
  --image-fusion-stage patch
  --map-postprocess none
  --top-k-ratio 0.05
)

require_directory() {
  local name="$1"
  local path="$2"
  if [[ ! -d "$path" ]]; then
    printf '%s does not exist: %s\n' "$name" "$path" >&2
    exit 2
  fi
}

print_decision_metrics() {
  local summary_path="$1"
  local dataset_name="$2"
  "$PYTHON" - "$summary_path" "$dataset_name" <<'PY'
import csv
import sys

summary_path, dataset_name = sys.argv[1:]
with open(summary_path, encoding="utf-8-sig", newline="") as stream:
    rows = list(csv.DictReader(stream))

print(f"[main-pca] {dataset_name} normal-test decisions")
print("category\tgood_accuracy\tdefect_recall\tbalanced_accuracy\tdecision_accuracy\tTN\tFP\tTP\tFN\tthreshold_source\tquantile\treferences")
for row in rows:
    if row.get("category") != "macro_average":
        print("\t".join([
            row.get("category", ""),
            row.get("good_accuracy", ""),
            row.get("defect_recall", ""),
            row.get("balanced_accuracy", ""),
            row.get("decision_accuracy", ""),
            row.get("good_predicted_normal", ""),
            row.get("good_predicted_anomaly", ""),
            row.get("defect_predicted_anomaly", ""),
            row.get("defect_predicted_normal", ""),
            row.get("good_decision_threshold_source", ""),
            row.get("good_decision_quantile", ""),
            row.get("good_decision_reference_images", ""),
        ]))

macro = next((row for row in rows if row.get("category") == "macro_average"), None)
if macro:
    print(
        f"macro_average good_accuracy={macro.get('good_accuracy', '')} "
        f"defect_recall={macro.get('defect_recall', '')} "
        f"balanced_accuracy={macro.get('balanced_accuracy', '')} "
        f"decision_accuracy={macro.get('decision_accuracy', '')}"
    )
PY
}

run_mvtec() {
  local output="$OUTPUT_ROOT/mvtec"
  require_directory MVTEC_DATA_ROOT "$MVTEC_DATA_ROOT"

  if [[ "$SKIP_COMPLETED" == "1" && -f "$output/summary.csv" ]]; then
    printf '[main-pca] skipping MVTec (complete: %s)\n' "$output/summary.csv"
    print_decision_metrics "$output/summary.csv" "MVTec"
    return
  fi

  printf '[main-pca] starting MVTec PCA evaluation for all categories\n'
  "$PYTHON" -m defectfusion.cli evaluate-mvtec \
    --data-root "$MVTEC_DATA_ROOT" \
    "${common_args[@]}" \
    --no-augment-categories transistor \
    --output "$output"
  print_decision_metrics "$output/summary.csv" "MVTec"
  printf '[main-pca] completed MVTec; results: %s\n' "$output/results.json"
}

run_visa() {
  local output="$OUTPUT_ROOT/visa"
  local -a visa_args=()
  require_directory VISA_DATA_ROOT "$VISA_DATA_ROOT"
  if [[ -n "$VISA_SPLIT_CSV" ]]; then
    if [[ ! -f "$VISA_SPLIT_CSV" ]]; then
      printf 'VISA_SPLIT_CSV does not exist: %s\n' "$VISA_SPLIT_CSV" >&2
      exit 2
    fi
    visa_args+=(--split-csv "$VISA_SPLIT_CSV")
  fi

  if [[ "$SKIP_COMPLETED" == "1" && -f "$output/summary.csv" ]]; then
    printf '[main-pca] skipping VisA (complete: %s)\n' "$output/summary.csv"
    print_decision_metrics "$output/summary.csv" "VisA"
    return
  fi

  printf '[main-pca] starting VisA PCA evaluation for all categories\n'
  "$PYTHON" -m defectfusion.cli evaluate-visa \
    --data-root "$VISA_DATA_ROOT" \
    "${common_args[@]}" \
    --image-size-override macaroni2=896 \
    --image-size-override pcb2=896 \
    --image-size-override pcb3=896 \
    --affine-categories macaroni1 macaroni2 \
    "${visa_args[@]}" \
    --output "$output"
  print_decision_metrics "$output/summary.csv" "VisA"
  printf '[main-pca] completed VisA; results: %s\n' "$output/results.json"
}

if [[ "$DATASET" == "mvtec" || "$DATASET" == "all" ]]; then
  run_mvtec
fi
if [[ "$DATASET" == "visa" || "$DATASET" == "all" ]]; then
  run_visa
fi

printf '[main-pca] requested evaluations completed.\n'
