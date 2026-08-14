#!/usr/bin/env bash
set -euo pipefail

# Evaluate the frozen DINOv3 teacher on VisA before changing student
# distillation. The two profiles differ only in normal-score calibration; they
# do not use anomaly labels to construct a threshold.

usage() {
  cat <<'EOF'
Usage:
  bash scripts/evaluate_visa_teacher.sh \
    --visa-root /path/to/VisA_20220922 \
    --teacher-model /path/to/dinov3-vitb16-pretrain-lvd1689m \
    [options]

Required:
  --visa-root PATH          VisA root containing split_csv/1cls.csv or Data/.
  --teacher-model PATH_OR_ID  Frozen DINOv3 teacher used for direct evaluation.

Options:
  --visa-split-csv PATH     Optional VisA 1cls.csv outside --visa-root.
  --profile NAME            baseline or visa-balanced. Default: visa-balanced.
  --output-root PATH        Default: outputs/visa-teacher.
  --categories CSV          Optional comma-separated category subset.
  --normal-shots N          Normal reference images per category. Default: 8.
  --normal-augment-count N  Normal fitting views per source. Default: 30.
  --image-size N            Default: 672; pcb2, pcb3 and macaroni2 use 896.
  --feature-layers CSV      ViT-B default: 1,6,12.
  --seed N                  Default: 42.
  --device DEVICE           Default: cuda.
  --python COMMAND          Default: python.
  --skip-completed          Do not rerun an existing summary.csv.
  -h, --help                Show this help.

Profiles:
  baseline       Existing conservative LOO calibration: higher q=0.995,
                 30 held-out rotation views per source.
  visa-balanced  Candidate VisA operating point: linear q=0.90,
                 10 held-out rotation views per source. It deliberately
                 lowers the decision threshold; report its normal false
                 positive rate alongside defect recall.
EOF
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"

visa_root=""
visa_split_csv=""
teacher_model=""
profile="visa-balanced"
output_root="outputs/visa-teacher"
categories_csv=""
normal_shots=8
normal_augment_count=30
image_size=672
feature_layers="1,6,12"
seed=42
device="cuda"
python_command="python"
skip_completed=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --visa-root) visa_root="${2:?--visa-root requires a value}"; shift 2 ;;
    --visa-split-csv) visa_split_csv="${2:?--visa-split-csv requires a value}"; shift 2 ;;
    --teacher-model) teacher_model="${2:?--teacher-model requires a value}"; shift 2 ;;
    --profile) profile="${2:?--profile requires a value}"; shift 2 ;;
    --output-root) output_root="${2:?--output-root requires a value}"; shift 2 ;;
    --categories) categories_csv="${2:?--categories requires a value}"; shift 2 ;;
    --normal-shots) normal_shots="${2:?--normal-shots requires a value}"; shift 2 ;;
    --normal-augment-count) normal_augment_count="${2:?--normal-augment-count requires a value}"; shift 2 ;;
    --image-size) image_size="${2:?--image-size requires a value}"; shift 2 ;;
    --feature-layers) feature_layers="${2:?--feature-layers requires a value}"; shift 2 ;;
    --seed) seed="${2:?--seed requires a value}"; shift 2 ;;
    --device) device="${2:?--device requires a value}"; shift 2 ;;
    --python) python_command="${2:?--python requires a value}"; shift 2 ;;
    --skip-completed) skip_completed=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$visa_root" || -z "$teacher_model" ]]; then
  printf '%s\n\n' '--visa-root and --teacher-model are required.' >&2
  usage >&2
  exit 2
fi
if [[ ! -d "$visa_root" ]]; then
  printf 'VisA root does not exist: %s\n' "$visa_root" >&2
  exit 2
fi
if [[ -n "$visa_split_csv" && ! -f "$visa_split_csv" ]]; then
  printf 'VisA split CSV does not exist: %s\n' "$visa_split_csv" >&2
  exit 2
fi
case "$teacher_model" in
  /*|./*|../*)
    if [[ ! -d "$teacher_model" || ! -f "$teacher_model/config.json" ]]; then
      printf 'Teacher model must be a local Hugging Face snapshot containing config.json: %s\n' "$teacher_model" >&2
      exit 2
    fi
    printf '[teacher-visa] local teacher: %s\n' "$(cd -- "$teacher_model" && pwd -P)"
    ;;
  *) printf '[teacher-visa] Hugging Face teacher: %s\n' "$teacher_model" ;;
esac

case "$profile" in
  baseline)
    decision_quantile=0.995
    decision_method="higher"
    decision_augment_count=30
    ;;
  visa-balanced)
    decision_quantile=0.90
    decision_method="linear"
    decision_augment_count=10
    ;;
  *)
    printf 'Unknown --profile: %s (expected baseline or visa-balanced)\n' "$profile" >&2
    exit 2
    ;;
esac

cd "$repo_root"
output="$output_root/$profile"
if [[ "$skip_completed" == "1" && -f "$output/summary.csv" ]]; then
  printf '[teacher-visa] skipping completed profile: %s\n' "$output/summary.csv"
  exit 0
fi

category_args=()
if [[ -n "$categories_csv" ]]; then
  IFS=',' read -r -a categories <<< "$categories_csv"
  category_args+=(--categories "${categories[@]}")
fi
split_args=()
if [[ -n "$visa_split_csv" ]]; then
  split_args+=(--split-csv "$visa_split_csv")
fi

printf '[teacher-visa] profile=%s shots=%s calibration=leave-one-out q=%s method=%s decision_views=%s\n' \
  "$profile" "$normal_shots" "$decision_quantile" "$decision_method" "$decision_augment_count"

"$python_command" -m defectfusion.cli evaluate-visa \
  --data-root "$visa_root" \
  --model "$teacher_model" \
  --device "$device" \
  --normal-shots "$normal_shots" \
  --defect-shots 0 \
  --seed "$seed" \
  --normal-augment-count "$normal_augment_count" \
  --normal-augmentations rotate \
  --affine-categories macaroni1 macaroni2 \
  --normal-decision-calibration leave-one-out \
  --normal-decision-quantile "$decision_quantile" \
  --normal-decision-quantile-method "$decision_method" \
  --normal-decision-augment-count "$decision_augment_count" \
  --normal-decision-fit-augment-count 4 \
  --normal-decision-seed "$((seed + 100))" \
  --image-size "$image_size" \
  --image-size-override macaroni2=896 \
  --image-size-override pcb2=896 \
  --image-size-override pcb3=896 \
  --resize-mode direct \
  --feature-layers="$feature_layers" \
  --layer-aggregation mean \
  --layer-normalization none \
  --anomaly-method pca \
  --pca-residual-metric squared_l2 \
  --image-score mtop1p \
  --image-top-ratio 0.01 \
  --image-fusion-stage patch \
  --map-postprocess none \
  --top-k-ratio 0.05 \
  "${category_args[@]}" \
  "${split_args[@]}" \
  --output "$output"

printf '[teacher-visa] completed. Results: %s\n' "$output/results.json"
printf '[teacher-visa] summary: %s\n' "$output/summary.csv"
