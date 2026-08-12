#!/usr/bin/env bash
set -Eeuo pipefail

# DINOv3 ViT-B -> ViT-S distillation for Linux.
# Required: NORMAL_DIR=/path/to/normal/images
# Optional: DEFECT_DIR=/path/to/synthetic/images MASK_DIR=/path/to/masks
# Run as: NORMAL_DIR=... DEFECT_DIR=... MASK_DIR=... bash scripts/train_dinov3_distill.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
NORMAL_DIR="${NORMAL_DIR:-}"
DEFECT_DIR="${DEFECT_DIR:-}"
MASK_DIR="${MASK_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/dinov3-vits-distilled}"
TEACHER_MODEL="${TEACHER_MODEL:-facebook/dinov3-vitb16-pretrain-lvd1689m}"
STUDENT_MODEL="${STUDENT_MODEL:-facebook/dinov3-vits16-pretrain-lvd1689m}"
DEVICE="${DEVICE:-cuda}"
IMAGE_SIZE="${IMAGE_SIZE:-448}"
BATCH_SIZE="${BATCH_SIZE:-2}"
CENTROID_BATCH_SIZE="${CENTROID_BATCH_SIZE:-2}"
EPOCHS="${EPOCHS:-10}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-42}"

if [[ -z "$NORMAL_DIR" ]]; then
  echo "ERROR: set NORMAL_DIR to normal training images (for example train/good)." >&2
  exit 2
fi
if [[ ! -d "$NORMAL_DIR" ]]; then
  echo "ERROR: NORMAL_DIR does not exist: $NORMAL_DIR" >&2
  exit 2
fi
if [[ -n "$DEFECT_DIR" && ! -d "$DEFECT_DIR" ]]; then
  echo "ERROR: DEFECT_DIR does not exist: $DEFECT_DIR" >&2
  exit 2
fi
if [[ -n "$MASK_DIR" && ! -d "$MASK_DIR" ]]; then
  echo "ERROR: MASK_DIR does not exist: $MASK_DIR" >&2
  exit 2
fi

train_args=(
  --normal-dir "$NORMAL_DIR"
  --teacher-model "$TEACHER_MODEL"
  --student-model "$STUDENT_MODEL"
  --output "$OUTPUT_DIR"
  --device "$DEVICE"
  --image-size "$IMAGE_SIZE"
  --feature-layers 1,6,12
  --adaptation lora
  --last-n-blocks "${LAST_N_BLOCKS:-4}"
  --lora-rank "${LORA_RANK:-8}"
  --lora-alpha "${LORA_ALPHA:-16}"
  --lora-dropout "${LORA_DROPOUT:-0.05}"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --centroid-batch-size "$CENTROID_BATCH_SIZE"
  --lr "${BACKBONE_LR:-1e-4}"
  --head-lr "${HEAD_LR:-1e-3}"
  --weight-decay "${WEIGHT_DECAY:-1e-4}"
  --lambda-feature "${LAMBDA_FEATURE:-1.0}"
  --lambda-map "${LAMBDA_MAP:-1.0}"
  --mask-alpha "${MASK_ALPHA:-2.0}"
  --margin "${ANOMALY_MARGIN:-0.2}"
  --top-ratio "${TOP_RATIO:-0.01}"
  --num-workers "$NUM_WORKERS"
  --seed "$SEED"
)

if [[ -n "$DEFECT_DIR" ]]; then
  train_args+=(--defect-dir "$DEFECT_DIR")
fi
if [[ -n "$MASK_DIR" ]]; then
  train_args+=(--mask-dir "$MASK_DIR")
fi
if [[ "${AMP:-1}" == "0" ]]; then
  train_args+=(--no-amp)
else
  train_args+=(--amp)
fi

echo "[distill] teacher: $TEACHER_MODEL"
echo "[distill] student: $STUDENT_MODEL"
echo "[distill] output:  $OUTPUT_DIR"
"$PYTHON_BIN" -m defectfusion.distill_finetune "${train_args[@]}"

MERGED_MODEL="$OUTPUT_DIR/student_merged"
if [[ ! -f "$MERGED_MODEL/config.json" ]]; then
  echo "ERROR: merged student was not exported to $MERGED_MODEL" >&2
  exit 1
fi
echo "[distill] merged student ready: $MERGED_MODEL"

# Optional end-to-end evaluation. Example:
# RUN_EVAL=1 EVAL_KIND=mvtec EVAL_DATA_ROOT=/data/mvtec \
#   NORMAL_DIR=... bash scripts/train_dinov3_distill.sh
if [[ "${RUN_EVAL:-0}" != "1" ]]; then
  exit 0
fi

EVAL_KIND="${EVAL_KIND:-mvtec}"
EVAL_DATA_ROOT="${EVAL_DATA_ROOT:-}"
if [[ "$EVAL_KIND" != "mvtec" && "$EVAL_KIND" != "visa" ]]; then
  echo "ERROR: EVAL_KIND must be mvtec or visa." >&2
  exit 2
fi
if [[ -z "$EVAL_DATA_ROOT" || ! -d "$EVAL_DATA_ROOT" ]]; then
  echo "ERROR: set EVAL_DATA_ROOT to the MVTec AD or VisA root directory." >&2
  exit 2
fi

eval_command="evaluate-mvtec"
if [[ "$EVAL_KIND" == "visa" ]]; then
  eval_command="evaluate-visa"
fi

"$PYTHON_BIN" -m defectfusion.cli "$eval_command" \
  --data-root "$EVAL_DATA_ROOT" \
  --model "$MERGED_MODEL" \
  --device "$DEVICE" \
  --normal-shots "${NORMAL_SHOTS:--1}" \
  --defect-shots 0 \
  --seed "$SEED" \
  --image-size "${EVAL_IMAGE_SIZE:-672}" \
  --normal-augment-count "${NORMAL_AUGMENT_COUNT:-0}" \
  --feature-layers=1,6,12 \
  --layer-aggregation mean \
  --layer-normalization none \
  --dual-branch \
  --anomaly-method pca_knn_anoco \
  --knn-weight 0.5 \
  --anoco-neighbors 16 \
  --anoco-query-weight 2.0 \
  --anoco-temperature 0.07 \
  --anoco-weight 0.25 \
  --anoco-layer-consensus \
  --fusion-mode fixed \
  --image-score mtop1p \
  --image-top-ratio 0.01 \
  --image-fusion-stage patch \
  --memory-max-patches "${MEMORY_MAX_PATCHES:-50000}" \
  --knn-chunk-size "${KNN_CHUNK_SIZE:-256}" \
  --knn-backend torch \
  --knn-dtype float16 \
  --knn-spatial-radius -1 \
  --map-postprocess none \
  --output "${EVAL_OUTPUT:-outputs/${EVAL_KIND}-distilled-vits}"

echo "[distill] evaluation complete"
