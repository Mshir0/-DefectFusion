#!/usr/bin/env bash
set -euo pipefail

# Train and evaluate one distilled ViT-S model for every detected MVTec AD and
# VisA category. All data/model locations are explicit command-line arguments;
# this script deliberately does not use environment variables or server paths.

usage() {
  cat <<'EOF'
Usage:
  bash scripts/distill_all_mvtec_visa.sh \
    --mvtec-root /path/to/mvtec_anomaly \
    --visa-root /path/to/VisA_20220922 \
    [options]

Required:
  --mvtec-root PATH         MVTec AD root containing category/train/good.
  --visa-root PATH          VisA root containing split_csv/1cls.csv or raw Data/ directories.

Model and output options:
  --teacher-model PATH_OR_ID  Frozen DINOv3 ViT-B checkpoint. Defaults to the Python entry point's Hugging Face ID.
  --student-model PATH_OR_ID  DINOv3 ViT-S checkpoint. Defaults to the Python entry point's Hugging Face ID.
  --visa-split-csv PATH       Optional VisA 1cls.csv outside --visa-root.
  --output-root PATH          Parent directory for the MVTec and VisA experiments. Default: outputs/dinov3-all-categories.
  --python COMMAND            Python executable. Default: python.
  --skip-completed            Skip a dataset when evaluation/results.json already exists.

Training options:
  --normal-shots N          Normal training images per category; -1 uses all. Default: 8.
  --epochs N                Epochs per category. Default: 10.
  --batch-size N            Training batch size. Default: 2.
  --num-workers N           DataLoader workers. Default: 4.
  --device DEVICE           Torch device. Default: cuda.
  --image-size N            Training and evaluation input size. Default: 448.
  --feature-layers CSV      Hidden-state layers. Default: 1,6,12.
  --adaptation MODE         Must be lora; only LoRA weights are exported. Default: lora.
  --last-n-blocks N         Final transformer blocks adapted. Default: 4.
  --lora-rank N             LoRA rank when --adaptation lora. Default: 8.
  --seed N                  Random seed. Default: 42.
  --no-amp                  Disable CUDA mixed precision.
  -h, --help                Show this help.

Examples:
  bash scripts/distill_all_mvtec_visa.sh \
    --mvtec-root /mnt/sda1/mvtec_anomaly \
    --visa-root /mnt/sda1/VisA_20220922 \
    --teacher-model /mnt/sda1/DINOv3/dinov3-vitb16-pretrain-lvd1689m \
    --student-model /mnt/sda1/DINOv3/dinov3-vits16-pretrain-lvd1689m

  # Use every normal training image and write a separate experiment directory.
  bash scripts/distill_all_mvtec_visa.sh \
    --mvtec-root /data/mvtec_anomaly --visa-root /data/VisA \
    --normal-shots -1 --output-root outputs/dinov3-all-fullshot
EOF
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"

mvtec_root=""
visa_root=""
visa_split_csv=""
teacher_model=""
student_model=""
output_root="outputs/dinov3-all-categories"
python_command="python"
normal_shots=8
epochs=10
batch_size=2
num_workers=4
device="cuda"
image_size=448
feature_layers="1,6,12"
adaptation="lora"
last_n_blocks=4
lora_rank=8
seed=42
amp=true
skip_completed=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mvtec-root)
      mvtec_root="${2:?--mvtec-root requires a value}"
      shift 2
      ;;
    --visa-root)
      visa_root="${2:?--visa-root requires a value}"
      shift 2
      ;;
    --visa-split-csv)
      visa_split_csv="${2:?--visa-split-csv requires a value}"
      shift 2
      ;;
    --teacher-model)
      teacher_model="${2:?--teacher-model requires a value}"
      shift 2
      ;;
    --student-model)
      student_model="${2:?--student-model requires a value}"
      shift 2
      ;;
    --output-root)
      output_root="${2:?--output-root requires a value}"
      shift 2
      ;;
    --python)
      python_command="${2:?--python requires a value}"
      shift 2
      ;;
    --normal-shots)
      normal_shots="${2:?--normal-shots requires a value}"
      shift 2
      ;;
    --epochs)
      epochs="${2:?--epochs requires a value}"
      shift 2
      ;;
    --batch-size)
      batch_size="${2:?--batch-size requires a value}"
      shift 2
      ;;
    --num-workers)
      num_workers="${2:?--num-workers requires a value}"
      shift 2
      ;;
    --device)
      device="${2:?--device requires a value}"
      shift 2
      ;;
    --image-size)
      image_size="${2:?--image-size requires a value}"
      shift 2
      ;;
    --feature-layers)
      feature_layers="${2:?--feature-layers requires a value}"
      shift 2
      ;;
    --adaptation)
      adaptation="${2:?--adaptation requires a value}"
      shift 2
      ;;
    --last-n-blocks)
      last_n_blocks="${2:?--last-n-blocks requires a value}"
      shift 2
      ;;
    --lora-rank)
      lora_rank="${2:?--lora-rank requires a value}"
      shift 2
      ;;
    --seed)
      seed="${2:?--seed requires a value}"
      shift 2
      ;;
    --no-amp)
      amp=false
      shift
      ;;
    --skip-completed)
      skip_completed=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown option: %s\n\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$mvtec_root" || -z "$visa_root" ]]; then
  printf '%s\n\n' '--mvtec-root and --visa-root are required.' >&2
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
if [[ "$adaptation" != "lora" ]]; then
  printf 'Only --adaptation lora is supported because outputs contain LoRA weights only.\n' >&2
  exit 2
fi

cd "$repo_root"

eval_normal_decision_calibration="leave-one-out"
if [[ "$normal_shots" == "1" ]]; then
  eval_normal_decision_calibration="training-reference"
  printf '[distill-all] 1-shot cannot use source-disjoint LOO; using training-reference calibration\n'
fi

common_args=(
  --normal-shots "$normal_shots"
  --defect-shots 0
  --epochs "$epochs"
  --batch-size "$batch_size"
  --num-workers "$num_workers"
  --device "$device"
  --image-size "$image_size"
  --feature-layers "$feature_layers"
  --eval-normal-decision-calibration "$eval_normal_decision_calibration"
  --eval-normal-decision-quantile 0.995
  --eval-normal-decision-quantile-method higher
  --eval-normal-decision-augment-count 30
  --eval-normal-decision-fit-augment-count 4
  --adaptation "$adaptation"
  --last-n-blocks "$last_n_blocks"
  --lora-rank "$lora_rank"
  --seed "$seed"
)
if [[ "$amp" == false ]]; then
  common_args+=(--no-amp)
fi
if [[ -n "$teacher_model" ]]; then
  common_args+=(--teacher-model "$teacher_model")
fi
if [[ -n "$student_model" ]]; then
  common_args+=(--student-model "$student_model")
fi

run_dataset() {
  local dataset="$1"
  local data_root="$2"
  local destination="$3"
  shift 3
  local -a dataset_args=("$@")

  if [[ "$skip_completed" == true && -f "$destination/evaluation/results.json" ]]; then
    printf '[distill-all] skipping %s; completed evaluation found at %s\n' "$dataset" "$destination/evaluation/results.json"
    return
  fi

  printf '[distill-all] starting %s for every discovered category\n' "$dataset"
  "$python_command" "$repo_root/distill_dinov3.py" \
    --dataset "$dataset" \
    --data-root "$data_root" \
    --output "$destination" \
    "${common_args[@]}" \
    "${dataset_args[@]}"
  printf '[distill-all] completed %s; metrics: %s\n' "$dataset" "$destination/evaluation/results.json"
}

# Omitting --categories intentionally makes distill_dinov3.py discover all
# classes in each dataset and trains them sequentially under one result root.
run_dataset mvtec "$mvtec_root" "$output_root/mvtec"

visa_args=()
if [[ -n "$visa_split_csv" ]]; then
  visa_args+=(--split-csv "$visa_split_csv")
fi
run_dataset visa "$visa_root" "$output_root/visa" "${visa_args[@]}"

printf '[distill-all] all MVTec AD and VisA categories completed.\n'
