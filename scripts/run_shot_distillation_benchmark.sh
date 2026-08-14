#!/usr/bin/env bash
set -euo pipefail

# Run the main PCA detector at 1/2/4/8 normal shots for MVTec AD and VisA,
# then run one 8-shot ViT-B -> ViT-S+ LoRA distillation for both datasets.
# Every completed result is retained and converted into CSV/Markdown tables.

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_shot_distillation_benchmark.sh \
    --mvtec-root /path/to/mvtec_anomaly \
    --visa-root /path/to/VisA_20220922 \
    --base-model /path/to/dinov3-vitl16 \
    --teacher-model /path/to/dinov3-vitb16 \
    --student-model /path/to/dinov3-vits16 \
    [options]

Required paths:
  --mvtec-root PATH       MVTec AD dataset root.
  --visa-root PATH        VisA dataset root.
  --base-model PATH_OR_ID Backbone used by the main PCA 1/2/4/8-shot runs.
  --teacher-model PATH_OR_ID Frozen ViT-B distillation teacher.
  --student-model PATH_OR_ID ViT-S+ base checkpoint used with saved LoRA adapters.

Options:
  --visa-split-csv PATH   Optional VisA 1cls.csv outside --visa-root.
  --output-root PATH      Default: outputs/shot-distillation-benchmark
  --python COMMAND        Default: python
  --device DEVICE         Default: cuda
  --seed N                Shared shot-sampling seed. Default: 42
  --distill-shots N       One distillation run's normal shots. Default: 8
  --distill-epochs N      Default: 10
  --distill-batch-size N  Default: 2
  --num-workers N         Default: 4
  --skip-completed        Resume without rerunning experiments with final results.
  --no-amp                Disable mixed precision during distillation.
  -h, --help              Show this help.

The main 1-shot run uses augmentation calibration because leave-one-out is not
defined for one source image. The 2/4/8-shot runs use source-disjoint LOO.
DenseCRF is intentionally excluded from this full benchmark because its effect
is category-dependent; use compare_mvtec_crf.sh or compare_visa_crf.sh for it.
EOF
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"

mvtec_root=""
visa_root=""
visa_split_csv=""
base_model=""
teacher_model=""
student_model=""
output_root="outputs/shot-distillation-benchmark"
python_command="python"
device="cuda"
seed=42
distill_shots=8
distill_epochs=10
distill_batch_size=2
num_workers=4
skip_completed=0
amp=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mvtec-root) mvtec_root="${2:?--mvtec-root requires a value}"; shift 2 ;;
    --visa-root) visa_root="${2:?--visa-root requires a value}"; shift 2 ;;
    --visa-split-csv) visa_split_csv="${2:?--visa-split-csv requires a value}"; shift 2 ;;
    --base-model) base_model="${2:?--base-model requires a value}"; shift 2 ;;
    --teacher-model) teacher_model="${2:?--teacher-model requires a value}"; shift 2 ;;
    --student-model) student_model="${2:?--student-model requires a value}"; shift 2 ;;
    --output-root) output_root="${2:?--output-root requires a value}"; shift 2 ;;
    --python) python_command="${2:?--python requires a value}"; shift 2 ;;
    --device) device="${2:?--device requires a value}"; shift 2 ;;
    --seed) seed="${2:?--seed requires a value}"; shift 2 ;;
    --distill-shots) distill_shots="${2:?--distill-shots requires a value}"; shift 2 ;;
    --distill-epochs) distill_epochs="${2:?--distill-epochs requires a value}"; shift 2 ;;
    --distill-batch-size) distill_batch_size="${2:?--distill-batch-size requires a value}"; shift 2 ;;
    --num-workers) num_workers="${2:?--num-workers requires a value}"; shift 2 ;;
    --skip-completed) skip_completed=1; shift ;;
    --no-amp) amp=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$mvtec_root" || -z "$visa_root" || -z "$base_model" || -z "$teacher_model" || -z "$student_model" ]]; then
  printf '%s\n\n' 'All five dataset/model path arguments are required.' >&2
  usage >&2
  exit 2
fi
if [[ ! -d "$mvtec_root" ]]; then
  printf 'MVTec root does not exist: %s\n' "$mvtec_root" >&2
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
if [[ "$distill_shots" == "1" ]]; then
  printf '%s\n' 'Use --distill-shots 2 or greater so distilled evaluation can use LOO.' >&2
  exit 2
fi

report_model_reference() {
  local option_name="$1"
  local reference="$2"
  case "$reference" in
    /*|./*|../*)
      if [[ ! -d "$reference" ]]; then
        printf '%s local model directory does not exist: %s\n' "$option_name" "$reference" >&2
        exit 2
      fi
      if [[ ! -f "$reference/config.json" ]]; then
        printf '%s local model directory is missing config.json: %s\n' "$option_name" "$reference" >&2
        exit 2
      fi
      printf '[models] %s local directory: %s\n' "$option_name" "$(cd -- "$reference" && pwd -P)"
      ;;
    *)
      printf '[models] %s Hugging Face ID: %s\n' "$option_name" "$reference"
      ;;
  esac
}

report_model_reference --base-model "$base_model"
report_model_reference --teacher-model "$teacher_model"
report_model_reference --student-model "$student_model"

cd "$repo_root"
mkdir -p "$output_root/logs" "$output_root/tables"

refresh_tables() {
  "$python_command" "$repo_root/scripts/build_benchmark_tables.py" \
    --input-root "$output_root" \
    --output-dir "$output_root/tables"
}

run_main_shot() {
  local shots="$1"
  local calibration="leave-one-out"
  local quantile_method="higher"
  local destination="$output_root/main/${shots}shot"
  if [[ "$shots" == "1" ]]; then
    calibration="augmentation"
    quantile_method="linear"
  fi

  printf '[benchmark] main PCA: shots=%s calibration=%s\n' "$shots" "$calibration"
  DATASET=all \
  PYTHON="$python_command" \
  MODEL="$base_model" \
  DEVICE="$device" \
  MVTEC_DATA_ROOT="$mvtec_root" \
  VISA_DATA_ROOT="$visa_root" \
  VISA_SPLIT_CSV="$visa_split_csv" \
  NORMAL_SHOTS="$shots" \
  SEED="$seed" \
  NORMAL_DECISION_CALIBRATION="$calibration" \
  NORMAL_DECISION_QUANTILE=0.995 \
  NORMAL_DECISION_QUANTILE_METHOD="$quantile_method" \
  OUTPUT_ROOT="$destination" \
  SKIP_COMPLETED="$skip_completed" \
  bash "$repo_root/scripts/evaluate_pca_good_accuracy.sh" \
    2>&1 | tee "$output_root/logs/main-${shots}shot.log"
  refresh_tables
}

for shots in 1 2 4 8; do
  run_main_shot "$shots"
done

distill_args=(
  --mvtec-root "$mvtec_root"
  --visa-root "$visa_root"
  --teacher-model "$teacher_model"
  --student-model "$student_model"
  --output-root "$output_root/distillation"
  --python "$python_command"
  --normal-shots "$distill_shots"
  --epochs "$distill_epochs"
  --batch-size "$distill_batch_size"
  --num-workers "$num_workers"
  --device "$device"
  --seed "$seed"
)
if [[ -n "$visa_split_csv" ]]; then
  distill_args+=(--visa-split-csv "$visa_split_csv")
fi
if [[ "$skip_completed" == "1" ]]; then
  distill_args+=(--skip-completed)
fi
if [[ "$amp" == "0" ]]; then
  distill_args+=(--no-amp)
fi

printf '[benchmark] distilled ViT-S+ LoRA: shots=%s\n' "$distill_shots"
bash "$repo_root/scripts/distill_all_mvtec_visa.sh" "${distill_args[@]}" \
  2>&1 | tee "$output_root/logs/distillation-${distill_shots}shot.log"

refresh_tables
printf '[benchmark] completed. Markdown table: %s\n' "$output_root/tables/results.md"
printf '[benchmark] per-metric best: %s\n' "$output_root/tables/best_results.csv"
printf '[benchmark] best threshold results: %s\n' "$output_root/tables/best_balanced_results.csv"
