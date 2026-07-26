# DefectFusion

Training-free few-shot anomaly detection and localization with DINOv3 features.
The current project is intentionally detection-only: it learns exclusively
from normal reference images and does not use labeled defect shots.

## Installation

```bash
pip install -e .
```

DenseCRF is optional:

```bash
pip install -e ".[crf]"
```

## MVTec AD Evaluation

The recommended 1-shot configuration uses one normal image, 30 rotated views,
cross-layer DINOv3 features, and fixed PCA-kNN score fusion:

```bash
python -m defectfusion.cli evaluate-mvtec \
  --data-root /mnt/sda1/mvtec_anomaly \
  --model /mnt/sda1/DINOv3/dinov3-vitl16-pretrain-lvd1689m \
  --normal-shots 1 \
  --seed 42 \
  --normal-augment-count 30 \
  --normal-augmentations rotate \
  --image-size 672 \
  --feature-layers=-1,-3,-5,-7 \
  --layer-aggregation mean \
  --image-score mtop1p \
  --anomaly-method pca_knn \
  --fusion-mode fixed \
  --knn-weight 0.5 \
  --memory-max-patches 5000 \
  --knn-backend torch \
  --knn-dtype float16 \
  --output outputs/mvtec-1shot.json
```

`--normal-shots -1` uses all images in `train/good`. The evaluator reports:

- Image AUROC and AUPR
- Pixel AUROC and AUPR
- Pixel AUPRO with 8-connectivity and `FPR <= 0.3`

Test images are never used to fit the detector. Per-image JSONL output includes
the anomaly score, patch-grid anomaly map, PCA score, and kNN score when active.

## Five Seeds

```bash
for seed in 0 1 2 3 4; do
  python -m defectfusion.cli evaluate-mvtec \
    --data-root /mnt/sda1/mvtec_anomaly \
    --model /mnt/sda1/DINOv3/dinov3-vitl16-pretrain-lvd1689m \
    --normal-shots 1 --seed "$seed" \
    --normal-augment-count 30 --normal-augmentations rotate \
    --image-size 672 --feature-layers=-1,-3,-5,-7 \
    --layer-aggregation mean --image-score mtop1p \
    --anomaly-method pca_knn --fusion-mode fixed --knn-weight 0.5 \
    --memory-max-patches 5000 --knn-backend torch --knn-dtype float16 \
    --output "outputs/mvtec-seed-${seed}.json"
done
```

## Fit And Predict

Fit a detector from a directory containing only normal images:

```bash
python -m defectfusion.cli fit \
  --normal-dir examples/data/normal \
  --image-size 672 \
  --feature-layers=-1,-3,-5,-7 \
  --anomaly-method pca_knn \
  --knn-backend torch \
  --knn-dtype float16 \
  --output outputs/model.json
```

Score an image:

```bash
python -m defectfusion.cli predict \
  --model-state outputs/model.json \
  --image examples/data/test.png
```

## Main Parameters

| Parameter | Default | Relevant values |
|---|---:|---|
| `--normal-shots` | `-1` | `1, 2, 4, -1` |
| `--normal-augment-count` | `30` in few-shot mode | `0, 10, 20, 30` |
| `--image-size` | `448` | `448, 672` |
| `--feature-layers` | `-1,-3,-5,-7` | comma-separated hidden-state indices |
| `--feature-layer-preset` | `cross4` | `cross4, last4, middle7` |
| `--layer-aggregation` | `mean` | `mean, concat` |
| `--image-score` | `mtop1p` | `mtop1p, mean, p99, max` |
| `--anomaly-method` | `pca` | `pca, knn, pca_knn` |
| `--fusion-mode` | `fixed` | `fixed, gated` |
| `--knn-weight` | `0.5` | `0.25, 0.5, 0.75` |
| `--memory-max-patches` | `50000` | `5000, 10000, 0` |
| `--knn-backend` | `auto` | `auto, numpy, torch` |
| `--knn-dtype` | `float32` | `float32, float16` |
| `--map-postprocess` | `none` | `none, gaussian, crf` |

`cross4` is `-1,-3,-5,-7`, `last4` is `-1,-2,-3,-4`, and `middle7`
uses the seven intermediate layers evaluated in the SubspaceAD-style ablation.

## Implementation Notes

- `mtop1p` averages the highest-scoring 1% of patches for the image score.
- PCA residual and normal-memory kNN scores are robustly calibrated before
  fixed fusion.
- Torch kNN uses CUDA when the extractor is on CUDA; `float16` reduces memory
  use and improves matrix-matching throughput.
- Normal-memory patches are stored in a compressed NPZ sidecar instead of the
  JSON model state.
- Gaussian smoothing remains optional but is disabled by default after a
  negative MVTec ablation.
