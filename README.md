# DefectFusion

Training-free and few-shot defect detection and defect typing, combining ideas from AnomalyDINO, SubspaceAD, and INSID3.

## Design

- Frozen dense DINO features (DINOv3 ViT-7B by default; a compatible DINO wrapper can be supplied).
- Position debiasing inspired by INSID3.
- Foreground-aware normal modeling with a streaming PCA subspace inspired by SubspaceAD.
- Optional k-nearest-neighbour memory score inspired by AnomalyDINO.
- Few-shot defect prototypes and zero-shot text prototypes are pluggable classifiers.

## Quick start

```bash
python examples/generate_data.py
python -m defectfusion.cli fit --config configs/example.json
python -m defectfusion.cli predict --model-state outputs/example-model.json --image examples/data/prototypes/scratch/scratch_0.png
```

`fit` accepts `--normal-dir` and an optional `--prototype-dir`; each prototype
subdirectory becomes a defect label. Use `--device cuda` when available,
`--unknown-threshold` to control the `unknown` decision, and `--output` to
write prediction JSON. All options can be placed in a JSON file (see
`configs/example.json`), with command-line flags taking precedence.

The default backbone is `facebook/dinov3-vit7b16-pretrain-lvd1689m`. This
checkpoint may require Hugging Face access approval/token and substantial GPU
memory; pass `--model` to use another compatible checkpoint. The default classifier reports `unknown` when no prototype is sufficiently
confident. Model weights are downloaded by HuggingFace/torch at runtime.

## MVTec AD evaluation

Download and unpack MVTec AD so that each category has `train/good`,
`test/<defect_type>`, and `ground_truth/<defect_type>` directories. Then run:

```bash
python -m defectfusion.cli evaluate-mvtec \
  --data-dir data/mvtec/bottle \
  --model facebook/dinov3-vit7b16-pretrain-lvd1689m \
  --output outputs/mvtec-bottle.jsonl
```

The command fits only on `train/good`, writes one JSON object per test image,
and reports Image AUROC/AUPR plus Pixel AUROC/AUPR/AUPRO when ground-truth
masks are available. AUPRO follows the MVTec convention and integrates the
per-region overlap curve over `FPR <= 0.3`.
Use `--data-root data/mvtec` to evaluate every category under the MVTec root;
the CLI prints each image as it is processed and writes one JSONL file per
category. For a multi-category run, `--output` stores the final combined JSON
summary, including macro averages and every category result; per-image files
use `<output-stem>-<category>.jsonl`. Add `--prototype-dir` (subdirectories are defect labels) to enable
defect-type accuracy, macro-F1, and a confusion matrix.

MVTec evaluation separates normal references used for anomaly detection from
labeled defect references used for typing. `--normal-shots 1/2/4` samples that
many images from `train/good`; `-1` uses all normal training images.
`--defect-shots` samples test images per defect type as typing prototypes
(selected defect images are excluded from evaluation):

```bash
python -m defectfusion.cli evaluate-mvtec --data-root data/mvtec \
  --normal-shots 4 --defect-shots 1 --seed 42 \
  --output outputs/mvtec-all.jsonl
```

Enable INSID3 positional debiasing for an ablation with the same split:

```bash
python -m defectfusion.cli evaluate-mvtec --data-root data/mvtec \
  --normal-shots -1 --defect-shots 1 --seed 42 --debias --svd-components 20 \
  --output outputs/mvtec-all-debiased.jsonl
```

Defect typing uses the highest PCA-reconstruction-residual patches instead of
the whole-image mean. Control their fraction with `--top-k-ratio` (default
`0.05`, selected by the MVTec 1-shot ablation); use `1.0` to reproduce
whole-image prototype classification.

Image-level anomaly scores use the mean of the highest-scoring 1% of patch
residuals (`--image-score mtop1p`). Use `--image-score mean` to reproduce the
previous whole-image score; `max` and `p99` are also available for ablation.

Dense features are averaged from the four cross-depth DINOv3 states
`-1,-3,-5,-7` by default, selected after the MVTec ablation. Use
`--feature-layer-preset last4` for the consecutive final-four baseline or
`--feature-layers=-1` for the former last-layer baseline.
Use `--feature-layer-preset middle7` to select the seven intermediate states
`-12,-13,-14,-15,-16,-17,-18` used by the SubspaceAD benchmark. This matches
its hidden-state indices, but not necessarily the same relative network depth
when the backbone has a different number of transformer blocks.

## Ablation experiments

Keep the dataset, backbone, normal/defect shot counts, seed, and all unrelated parameters fixed
when comparing one option. The current recommended setup is:

```bash
python -m defectfusion.cli evaluate-mvtec \
  --data-root data/mvtec \
  --model facebook/dinov3-vit7b16-pretrain-lvd1689m \
  --normal-shots -1 --defect-shots 1 --seed 42 \
  --top-k-ratio 0.05 \
  --type-matching bidirectional_patch \
  --image-score mtop1p \
  --feature-layers=-1,-3,-5,-7 \
  --layer-aggregation mean \
  --output outputs/ablation-recommended.jsonl
```

### Parameters

| Dimension | Argument | Current default | Ablation values | Main affected metrics |
|---|---|---:|---|---|
| Normal references | `--normal-shots` | `-1` (all) | `1, 2, 4, -1` | Detection sample efficiency |
| Defect references | `--defect-shots` | `0` | `0, 1, 3, 5` | Type Accuracy, Macro-F1 |
| Normal augment count | `--normal-augment-count` | `30` in normal few-shot | `0, 10, 30` | Detection sample efficiency |
| Normal augmentations | `--normal-augmentations` | `rotate` | `rotate, hflip, vflip, color_jitter, affine` | Detection metrics |
| Sampling seed | `--seed` | `42` | `0, 1, 2, 42` | Few-shot variance |
| Typing patch ratio | `--top-k-ratio` | `0.05` | `0.05, 0.10, 0.20, 0.30, 1.0` | Type Accuracy, Macro-F1 |
| Image score | `--image-score` | `mtop1p` | `mean, mtop1p, p99, max` | Image AUROC |
| Type matching | `--type-matching` | `bidirectional_patch` | `prototype_mean, bidirectional_patch` | Type Accuracy, Macro-F1 |
| Anomaly detector | `--anomaly-method` | `pca` | `pca, knn, pca_knn` | Image/Pixel AUROC |
| kNN fusion weight | `--knn-weight` | `0.5` | `0.25, 0.5, 0.75` | Active with `pca_knn` |
| PCA-kNN fusion | `--fusion-mode` | `fixed` | `fixed, gated` | Image/Pixel AUROC |
| Gate temperature | `--gate-temperature` | `1.0` | `0.5, 1.0, 2.0` | Active with gated fusion |
| Normal patch memory | `--memory-max-patches` | `50000` | `10000, 25000, 50000, 0` | kNN accuracy, memory, runtime |
| kNN query chunk | `--knn-chunk-size` | `256` | `64, 128, 256` | Runtime/memory only |
| kNN backend | `--knn-backend` | `auto` | `auto, torch, numpy` | Runtime only |
| kNN CUDA dtype | `--knn-dtype` | `float32` | `float32, float16` | Speed, memory, small numeric differences |
| Feature layers | `--feature-layers` | `-1,-3,-5,-7` | `-1`, `-1,-2,-3,-4`, `-1,-3,-5,-7` | All metrics |
| Feature layer preset | `--feature-layer-preset` | `cross4` | `cross4, last4, middle7` | All metrics |
| Layer fusion | `--layer-aggregation` | `mean` | `mean, concat` | All metrics and memory |
| Map post-process | `--map-postprocess` | `none` | `none, gaussian, crf` | Pixel AUROC |
| Gaussian sigma | `--gaussian-sigma` | `1.0` | `0.5, 1.0, 2.0` | Pixel AUROC |
| Positional debiasing | `--debias` | off | off/on | All metrics |
| Debias rank | `--svd-components` | `20` | `2, 5, 10, 20` | Active only with `--debias` |
| Device | `--device` | auto | `cpu, cuda` | Runtime only |
| Input resolution | `--image-size` | `448` | `224, 448, 672` | All metrics and memory |

Negative feature-layer values must use the equals form, such as
`--feature-layers=-1,-3,-5,-7`, so that `argparse` does not interpret them as
new options. `concat` multiplies the PCA feature dimension by the number of
selected layers and therefore uses substantially more memory than `mean`.
Input images are explicitly resized to `--image-size` without center cropping;
the value is passed to the Hugging Face processor rather than relying on its
checkpoint default. Higher resolutions increase patch count quadratically.

Defect typing performs bidirectional matching between the PCA-selected Top-K
query patches and all defect-reference patches for each label. Its score is the
mean of query-to-reference and reference-to-query nearest-neighbour
similarities, which rewards both precise matches and reference coverage.

Anomaly-map post-processing is isolated from image scoring and defect typing.
Use `--map-postprocess none` for the raw-map baseline, `gaussian` for separable
Gaussian smoothing on the patch grid, or `crf` for RGB-guided DenseCRF. CRF
requires installing the optional dependency with `pip install -e '.[crf]'`.
Gaussian smoothing is disabled by default after the MVTec sigma=1.0 ablation
reduced macro pixel AUROC; it remains available only for explicit experiments.

### PCA and normal-patch kNN fusion

`--anomaly-method knn` uses the AnomalyDINO normal-patch memory score: cosine
distance to the closest normalized normal patch. `pca_knn` combines it with
the PCA reconstruction residual. `--fusion-mode fixed` reproduces robust
z-score calibration with the global `--knn-weight`. `gated` maps both raw
scores to empirical normal-tail evidence and assigns a soft kNN weight to
every patch. `--gate-temperature` controls how decisively the gate selects
the stronger expert; lower values approach hard selection.
The default remains `pca`, so existing baselines do not change. kNN queries
are chunked; lower `--knn-chunk-size` if inference runs out of memory. A
positive `--memory-max-patches` deterministically subsamples the normal bank;
use `0` to retain every patch.
With `--knn-backend auto`, a CUDA DINO extractor automatically uses Torch
matrix multiplication and keeps the normalized memory bank on the same GPU.
Use `--knn-backend torch` to require that path explicitly. `--knn-dtype
float16` halves memory-bank and similarity-matrix storage and is normally the
fastest CUDA setting; compare it with `float32` once before using it for final
reported numbers.

Run a single-variable comparison with the current 1-shot protocol:

```bash
for method in pca knn pca_knn; do
  python -m defectfusion.cli evaluate-mvtec --data-root data/mvtec \
    --normal-shots 1 --defect-shots 0 --seed 42 --normal-augment-count 30 \
    --image-size 672 --feature-layers=-1,-3,-5,-7 \
    --anomaly-method "$method" --knn-weight 0.5 \
    --knn-backend torch --knn-dtype float16 \
    --output "outputs/mvtec-${method}-seed42.json"
done
```

Saved fitted kNN models use `<model-state>.normal-memory.npz` alongside the
JSON state. Keep both files together when moving a fitted model.

Compare fixed and training-free gated calibration without changing the
backbone, memory, split, or image aggregation:

```bash
for fusion in fixed gated; do
  python -m defectfusion.cli evaluate-mvtec --data-root data/mvtec \
    --normal-shots 1 --defect-shots 0 --seed 42 --normal-augment-count 30 \
    --image-size 672 --feature-layers=-1,-3,-5,-7 \
    --anomaly-method pca_knn --fusion-mode "$fusion" --gate-temperature 1.0 \
    --knn-backend torch --knn-dtype float16 --memory-max-patches 5000 \
    --output "outputs/mvtec-fusion-${fusion}-seed42.json"
done
```

### Paper-protocol normal shots

Use no labeled defects and vary only normal references to compare with the
1/2/4-shot AnomalyDINO and SubspaceAD table:

```bash
for shots in 1 2 4; do
  python -m defectfusion.cli evaluate-mvtec --data-root data/mvtec \
    --normal-shots "$shots" --defect-shots 0 --seed 42 \
    --normal-augment-count 30 --normal-augmentations rotate \
    --image-size 672 \
    --output "outputs/mvtec-normal-${shots}shot.json"
done
```

`--normal-shots 0` is unsupported because PCA cannot be fitted without normal
features. The legacy `--few-shot` spelling remains an alias for
`--defect-shots`, but new experiments should use the explicit name.
Normal augmentation is active only for normal few-shot runs by default. Each
selected normal image contributes its original view plus 30 random rotations;
`transistor` is excluded by default, matching the SubspaceAD benchmark setup.
Use `--normal-augment-count 0` for the unaugmented baseline.

### Single-variable commands

Reproduce the original whole-image typing and image-score baseline:

```bash
python -m defectfusion.cli evaluate-mvtec --data-root data/mvtec \
  --model facebook/dinov3-vit7b16-pretrain-lvd1689m \
  --normal-shots -1 --defect-shots 1 --seed 42 --top-k-ratio 1.0 --image-score mean \
  --feature-layers=-1 --layer-aggregation mean \
  --output outputs/ablation-original.jsonl
```

Compare PCA-residual Top-K typing ratios:

```bash
for ratio in 0.05 0.10 0.20 0.30 1.0; do
  python -m defectfusion.cli evaluate-mvtec --data-root data/mvtec \
    --normal-shots -1 --defect-shots 1 --seed 42 --top-k-ratio "$ratio" \
    --output "outputs/ablation-topk-${ratio}.jsonl"
done
```

Compare image-level score aggregation while leaving pixel maps and typing
unchanged:

```bash
for score in mean mtop1p p99 max; do
  python -m defectfusion.cli evaluate-mvtec --data-root data/mvtec \
    --normal-shots -1 --defect-shots 1 --seed 42 --image-score "$score" \
    --output "outputs/ablation-image-${score}.jsonl"
done
```

Compare single-layer and multi-layer features:

```bash
python -m defectfusion.cli evaluate-mvtec --data-root data/mvtec \
  --normal-shots -1 --defect-shots 1 --seed 42 --feature-layers=-1 \
  --output outputs/ablation-layer-last.jsonl

python -m defectfusion.cli evaluate-mvtec --data-root data/mvtec \
  --normal-shots -1 --defect-shots 1 --seed 42 --feature-layers=-1,-2,-3,-4 \
  --layer-aggregation mean --output outputs/ablation-layer-last4-mean.jsonl

python -m defectfusion.cli evaluate-mvtec --data-root data/mvtec \
  --normal-shots -1 --defect-shots 1 --seed 42 --feature-layer-preset middle7 \
  --layer-aggregation mean --output outputs/ablation-layer-middle7-mean.jsonl
```

For the paper-protocol anomaly-detection comparison, keep defect references
disabled and change only the layer selection:

```bash
python -m defectfusion.cli evaluate-mvtec --data-root data/mvtec \
  --normal-shots 1 --defect-shots 0 --seed 42 --normal-augment-count 30 \
  --image-size 672 --feature-layer-preset middle7 --layer-aggregation mean \
  --output outputs/mvtec-normal-1shot-middle7.json
```

Measure shot-sampling stability with multiple seeds. Report the mean and standard
deviation across runs rather than selecting the best seed:

```bash
for seed in 0 1 2; do
  python -m defectfusion.cli evaluate-mvtec --data-root data/mvtec \
    --normal-shots 4 --defect-shots 1 --seed "$seed" \
    --output "outputs/ablation-seed-${seed}.jsonl"
done
```

Run the current fixed PCA+kNN 1-shot configuration for the final five-seed
comparison (each summary includes Image AUROC/AUPR and Pixel AUROC/AUPR/AUPRO):

```bash
for seed in 0 1 2 3 4; do
  python -m defectfusion.cli evaluate-mvtec --data-root data/mvtec \
    --model facebook/dinov3-vitl16-pretrain-lvd1689m --device cuda \
    --normal-shots 1 --defect-shots 0 --seed "$seed" \
    --normal-augment-count 30 --image-size 672 \
    --feature-layers=-1,-3,-5,-7 --layer-aggregation mean \
    --anomaly-method pca_knn --fusion-mode fixed --knn-weight 0.5 \
    --memory-max-patches 5000 --knn-backend torch --knn-dtype float16 \
    --knn-chunk-size 1024 \
    --output "outputs/mvtec-fixed-seed-${seed}.json"
done
```
