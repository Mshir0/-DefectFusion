#!/usr/bin/env bash
set -euo pipefail

# One-category PCA anomaly-map ablation. Both runs use identical data, model,
# seeds, normal fitting, and threshold calibration; only map post-processing
# changes between the raw baseline and RGB-guided DenseCRF refinement.

PYTHON="${PYTHON:-python}"
MODEL="${MODEL:-/mnt/sda1/DINOv3/dinov3-vitl16-pretrain-lvd1689m}"
MVTEC_DATA_ROOT="${MVTEC_DATA_ROOT:-/mnt/sda1/mvtec_anomaly}"
DEVICE="${DEVICE:-cuda}"
CATEGORY="${CATEGORY:-leather}"
NORMAL_SHOTS="${NORMAL_SHOTS:-8}"
NORMAL_AUGMENT_COUNT="${NORMAL_AUGMENT_COUNT:-30}"
NORMAL_DECISION_AUGMENT_COUNT="${NORMAL_DECISION_AUGMENT_COUNT:-$NORMAL_AUGMENT_COUNT}"
NORMAL_DECISION_QUANTILE="${NORMAL_DECISION_QUANTILE:-0.995}"
NORMAL_FIT_MAX_PATCHES="${NORMAL_FIT_MAX_PATCHES:-0}"
IMAGE_SIZE="${IMAGE_SIZE:-672}"
FEATURE_LAYERS="${FEATURE_LAYERS:-1,17,21,23}"
SEED="${SEED:-42}"
NORMAL_DECISION_SEED="${NORMAL_DECISION_SEED:-$((SEED + 100))}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/mvtec-crf-ablation}"
SKIP_COMPLETED="${SKIP_COMPLETED:-0}"

if [[ ! -d "$MVTEC_DATA_ROOT" ]]; then
  printf 'MVTEC_DATA_ROOT does not exist: %s\n' "$MVTEC_DATA_ROOT" >&2
  exit 2
fi
if [[ ! -d "$MVTEC_DATA_ROOT/$CATEGORY/train/good" ]]; then
  printf 'MVTec category does not exist: %s\n' "$MVTEC_DATA_ROOT/$CATEGORY" >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
cd "$repo_root"

if ! "$PYTHON" -c 'import pydensecrf.densecrf; import pydensecrf.utils' >/dev/null 2>&1; then
  printf "DenseCRF dependency is unavailable. Install it with: %s -m pip install -e '.[crf]'\n" "$PYTHON" >&2
  exit 2
fi

experiment_root="$OUTPUT_ROOT/$CATEGORY"
none_output="$experiment_root/none"
crf_output="$experiment_root/crf"

common_args=(
  --data-root "$MVTEC_DATA_ROOT"
  --categories "$CATEGORY"
  --model "$MODEL"
  --device "$DEVICE"
  --normal-shots "$NORMAL_SHOTS"
  --defect-shots 0
  --seed "$SEED"
  --normal-augment-count "$NORMAL_AUGMENT_COUNT"
  --normal-augmentations rotate
  --no-augment-categories transistor
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
  --top-k-ratio 0.05
)

run_mode() {
  local mode="$1"
  local output="$2"
  if [[ "$SKIP_COMPLETED" == "1" && -f "$output/summary.csv" ]]; then
    printf '[crf-ablation] skipping %s (complete: %s)\n' "$mode" "$output/summary.csv"
    return
  fi
  printf '[crf-ablation] category=%s mode=%s\n' "$CATEGORY" "$mode"
  "$PYTHON" -m defectfusion.cli evaluate-mvtec \
    "${common_args[@]}" \
    --map-postprocess "$mode" \
    --output "$output"
}

run_mode none "$none_output"
run_mode crf "$crf_output"

mkdir -p "$experiment_root"
"$PYTHON" - "$none_output/summary.csv" "$crf_output/summary.csv" \
  "$experiment_root/comparison.csv" "$CATEGORY" <<'PY'
import csv
import sys

none_path, crf_path, output_path, category = sys.argv[1:]
metric_names = (
    "image_auroc", "image_aupr", "pixel_auroc", "pixel_aupr",
    "pixel_aupro", "pixel_f1_max", "total_seconds",
)


def category_row(path):
    with open(path, encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return next(row for row in rows if row.get("category") == category)


none = category_row(none_path)
crf = category_row(crf_path)
fields = ["mode", *metric_names]
fields.extend(f"delta_{name}" for name in metric_names if name != "total_seconds")
rows = []
for mode, values in (("none", none), ("crf", crf)):
    row = {"mode": mode}
    row.update({name: values.get(name, "") for name in metric_names})
    for name in metric_names:
        if name != "total_seconds":
            row[f"delta_{name}"] = float(values[name]) - float(none[name])
    rows.append(row)

with open(output_path, "w", encoding="utf-8-sig", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print(f"[crf-ablation] category={category}")
print("mode\tP-AUROC\tP-AUPR\tP-AUPRO\tP-F1\ttotal_seconds")
for row in rows:
    print("\t".join([
        row["mode"], row["pixel_auroc"], row["pixel_aupr"],
        row["pixel_aupro"], row["pixel_f1_max"], row["total_seconds"],
    ]))
print("crf-minus-none\t" + "\t".join([
    f"{float(crf[name]) - float(none[name]):+.8f}"
    for name in ("pixel_auroc", "pixel_aupr", "pixel_aupro", "pixel_f1_max")
]))
print(f"[crf-ablation] comparison: {output_path}")
PY
