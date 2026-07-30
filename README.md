# DefectFusion

Training-free few-shot anomaly detection and localization, combining ideas from
AnomalyDINO, SubspaceAD, and INSID3. Labeled-defect typing is retained as an
optional auxiliary task, but it is not part of the default detection setup.

## Design

- Frozen dense DINO features (DINOv3 ViT-7B by default; a compatible DINO wrapper can be supplied).
- Position debiasing inspired by INSID3.
- Foreground-aware normal modeling with a streaming PCA subspace inspired by SubspaceAD.
- Optional k-nearest-neighbour memory score inspired by AnomalyDINO.
- Optional few-shot defect prototypes provide an auxiliary typing head.

## Quick start

```bash
python examples/generate_data.py
python -m defectfusion.cli fit --config configs/example.json
python -m defectfusion.cli predict --model-state outputs/example-model.json --image examples/data/test.png
```

`fit` learns the anomaly detector from `--normal-dir`. The optional
`--prototype-dir` enables the separate defect-typing head; each subdirectory
becomes a defect label. Use `--device cuda` when available,
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
  --data-root /mnt/sda1/mvtec_anomaly \
  --model /mnt/sda1/DINOv3/dinov3-vitl16-pretrain-lvd1689m \
  --normal-shots 1 --defect-shots 0 --seed 42 \
  --normal-augment-count 30 --normal-augmentations rotate \
  --image-size 672 --feature-layers=-1,-3,-5,-7 \
  --layer-aggregation mean --image-score mtop1p \
  --anomaly-method pca_knn --fusion-mode fixed --knn-weight 0.5 \
  --memory-max-patches 5000 --knn-backend torch --knn-dtype float16 \
  --output outputs/mvtec-1shot.json
```

The command fits only on `train/good`, writes one JSON object per test image,
and reports Image AUROC/AUPR plus Pixel AUROC/AUPR/AUPRO when ground-truth
masks are available. AUPRO follows the MVTec convention and integrates the
per-region overlap curve over `FPR <= 0.3`.
Use `--data-root data/mvtec` to evaluate every category under the MVTec root;
the CLI prints each image as it is processed and writes one JSON file per
category. For a multi-category run, `--output` stores the final combined JSON
summary, including macro averages and every category result. Evaluation output
is organized under the `--output` directory as `results.json`, `summary.csv`,
and `categories/<category>.json`. Each category file contains both its metrics
and a `predictions` array with all per-image results. No JSONL files are
generated. If `--output` has a file suffix, its stem becomes
the experiment directory name. Add `--prototype-dir` (subdirectories are defect labels) to enable
defect-type accuracy, macro-F1, and a confusion matrix.

Use `--categories macaroni2` to run a focused category ablation. The optional
`--image-min-component-size 2` rejects isolated Top-K anomaly patches from the
image score while leaving the pixel map unchanged; the default `1` preserves
the validated aggregation. For pose-variable categories, combine it with
`--normal-augmentations rotate affine` before considering a full-dataset run.
For a mixed full-dataset run, keep the baseline `--normal-augmentations rotate`
and scope both changes explicitly:

```bash
--affine-categories macaroni1 macaroni2 \
--image-min-component-size 2 \
--component-reject-categories macaroni1 macaroni2
```

Category-specific input resolution is repeatable with
`--image-size-override CATEGORY=SIZE`. For example,
`--image-size 672 --image-size-override macaroni2=896` keeps all other
categories at 672. The extractor resets its size-dependent positional cache
when switching categories.

Every category JSON records the effective augmentation list, component size,
and whether each override was active.

## VisA evaluation

VisA can be evaluated directly without converting its directory layout. The
loader supports both the official `split_csv/1cls.csv` split and the raw
per-category `Data/Images/{Normal,Anomaly}` layout:

```bash
python -m defectfusion.cli evaluate-visa \
  --data-root /mnt/sda1/VisA \
  --model /mnt/sda1/DINOv3/dinov3-vitl16-pretrain-lvd1689m \
  --normal-shots 1 --defect-shots 0 --seed 42 \
  --normal-augment-count 30 --normal-augmentations rotate \
  --image-size 672 --feature-layers=-1,-3,-5,-7 \
  --dual-branch --anomaly-method pca_knn_anoco \
  --knn-weight 0.5 --anoco-weight 0.25 --anoco-layer-consensus \
  --image-score mtop1p --image-top-ratio 0.01 --image-fusion-stage patch \
  --knn-backend torch --knn-dtype float16 --knn-spatial-radius -1 \
  --output outputs/visa-1shot
```

Use `--split-csv /path/to/1cls.csv` when the split file is stored elsewhere.
The loader reads the CSV `object`, `split`, `label`, `image`, and `mask`
columns, fits each object only on normal training rows, and evaluates the test
rows with their listed masks. Without a root split CSV, it discovers each
category's `Data` directory and pairs anomaly images and masks by filename stem;
selected normal shots are excluded from the normal test pool. Output layout and
metrics match MVTec evaluation.

`--normal-shots 1/2/4` samples that many images from the normal training
partition (`train/good` for MVTec or normal train rows for VisA); `-1` uses
all normal training images. Keep `--defect-shots 0` for standard anomaly
detection comparisons with AnomalyDINO and SubspaceAD. No test anomaly is then
used while fitting the detector.

### Optional defect typing

Typing is an auxiliary experiment, separate from the anomaly-detection setup.
`--defect-shots` samples labeled test images per defect type as prototypes;
those selected images are excluded from classification evaluation:

```bash
python -m defectfusion.cli evaluate-mvtec --data-root data/mvtec \
  --normal-shots 1 --defect-shots 1 --seed 42 \
  --output outputs/mvtec-all
```

Enable INSID3 positional debiasing for an ablation with the same split:

```bash
python -m defectfusion.cli evaluate-mvtec --data-root data/mvtec \
  --normal-shots -1 --defect-shots 1 --seed 42 --debias --svd-components 20 \
  --output outputs/mvtec-all-debiased
```

Defect typing uses the highest PCA-reconstruction-residual patches instead of
the whole-image mean. Control their fraction with `--top-k-ratio` (default
`0.05`, selected by the MVTec 1-shot ablation); use `1.0` to reproduce
whole-image prototype classification.

Image-level anomaly scores use the mean of the highest-scoring 1% of patch
residuals (`--image-score mtop1p --image-top-ratio 0.01`). Use `--image-score mean` to reproduce the
previous whole-image score; `max` and `p99` are also available for ablation.
The historical `mtop1p` name remains for compatibility, while
`--image-top-ratio` controls the actual fraction. Compare P0.2 on the dual
image branch with:

```bash
for ratio in 0.005 0.01 0.02 0.05; do
  python -m defectfusion.cli evaluate-mvtec \
    --data-root /mnt/sda1/mvtec_anomaly \
    --model /mnt/sda1/DINOv3/dinov3-vitl16-pretrain-lvd1689m \
    --normal-shots 1 --defect-shots 0 --seed 42 \
    --normal-augment-count 30 --normal-augmentations rotate \
    --image-size 672 --resize-mode direct \
    --feature-layers=-1,-3,-5,-7 --layer-aggregation mean \
    --layer-normalization none --dual-branch \
    --image-score mtop1p --image-top-ratio "$ratio" \
    --anomaly-method pca_knn --fusion-mode fixed --knn-weight 0.5 \
    --knn-spatial-radius -1 --memory-max-patches 5000 \
    --knn-backend torch --knn-dtype float16 \
    --output "outputs/mvtec-dual-top-${ratio}-seed42.json"
done
```

Dense features are averaged from the four cross-depth DINOv3 states
`-1,-3,-5,-7` by default, selected after the MVTec ablation. Use
`--feature-layer-preset last4` for the consecutive final-four baseline or
`--feature-layers=-1` for the former last-layer baseline.
Use `--feature-layer-preset middle7` to select the seven intermediate states
`-12,-13,-14,-15,-16,-17,-18` used by the SubspaceAD benchmark. This matches
its hidden-state indices, but not necessarily the same relative network depth
when the backbone has a different number of transformer blocks.

## Ablation experiments

Keep the dataset, backbone, normal shot count, seed, and all unrelated
parameters fixed when comparing one detection option. The current recommended
detection setup is:

```bash
python -m defectfusion.cli evaluate-mvtec \
  --data-root data/mvtec \
  --model facebook/dinov3-vit7b16-pretrain-lvd1689m \
  --normal-shots 1 --defect-shots 0 --seed 42 \
  --normal-augment-count 30 --normal-augmentations rotate \
  --image-score mtop1p \
  --feature-layers=-1,-3,-5,-7 \
  --layer-aggregation mean --image-size 672 \
  --anomaly-method pca_knn --fusion-mode fixed --knn-weight 0.5 \
  --memory-max-patches 5000 --knn-backend torch --knn-dtype float16 \
  --output outputs/ablation-recommended
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
| Image Top-K ratio | `--image-top-ratio` | `0.01` | `0.005, 0.01, 0.02, 0.05` | Active with `mtop1p` |
| Image fusion stage | `--image-fusion-stage` | `patch` | `patch, score` | PCA/kNN image evidence |
| Image spatial weight | `--image-spatial-weight` | `0.0` | `0, 0.25, 0.5` | Connected Top-K image evidence |
| Type matching | `--type-matching` | `bidirectional_patch` | `prototype_mean, bidirectional_patch, rbf_svm` | Type Accuracy, Macro-F1 |
| Anomaly detector | `--anomaly-method` | `pca` | `pca, knn, pca_knn` | Image/Pixel AUROC |
| PCA residual metric | `--pca-residual-metric` | `squared_l2` | `squared_l2, mahalanobis` | PCA anomaly evidence |
| kNN fusion weight | `--knn-weight` | `0.5` | `0.25, 0.5, 0.75` | Active with `pca_knn` |
| PCA-kNN fusion | `--fusion-mode` | `fixed` | `fixed, gated` | Image/Pixel AUROC |
| Gate temperature | `--gate-temperature` | `1.0` | `0.5, 1.0, 2.0` | Active with gated fusion |
| Normal patch memory | `--memory-max-patches` | `50000` | `10000, 25000, 50000, 0` | kNN accuracy, memory, runtime |
| kNN query chunk | `--knn-chunk-size` | `256` | `64, 128, 256` | Runtime/memory only |
| kNN spatial radius | `--knn-spatial-radius` | `-1` | `-1, 0.05, 0.10, 0.20` | Local structural matching |
| Align train positions | `--align-training-positions` | off | on/off | Canonical coordinates for rotate/flip normal views |
| kNN backend | `--knn-backend` | `auto` | `auto, torch, numpy` | Runtime only |
| kNN CUDA dtype | `--knn-dtype` | `float32` | `float32, float16` | Speed, memory, small numeric differences |
| Feature layers | `--feature-layers` | `-1,-3,-5,-7` | `-1`, `-1,-2,-3,-4`, `-1,-3,-5,-7` | All metrics |
| Feature layer preset | `--feature-layer-preset` | `cross4` | `cross4, last4, middle7` | All metrics |
| Layer fusion | `--layer-aggregation` | `mean` | `mean, concat` | All metrics and memory |
| Per-layer normalization | `--layer-normalization` | `none` | `none, l2` | Multi-layer feature balance |
| Dual image/pixel branch | `--dual-branch` | off | on/off | Image vs Pixel trade-off |
| Test-time augmentation | `--test-augmentations` | none | `hflip vflip` | Image and Pixel metrics, 3x inference |

Flip TTA always includes the identity view. For the first screening run, use
`--test-augmentations hflip vflip`; image scores are averaged across three
views and patch maps are inverse-aligned before averaging.
| Map post-process | `--map-postprocess` | `none` | `none, gaussian, crf` | Pixel AUROC |
| Gaussian sigma | `--gaussian-sigma` | `1.0` | `0.5, 1.0, 2.0` | Pixel AUROC |
| Positional debiasing | `--debias` | off | off/on | All metrics |
| Debias rank | `--svd-components` | `20` | `2, 5, 10, 20` | Active only with `--debias` |
| Device | `--device` | auto | `cpu, cuda` | Runtime only |
| Input resolution | `--image-size` | `448` | `224, 448, 672` | All metrics and memory |
| Resize geometry | `--resize-mode` | `direct` | `direct, longest_pad` | Geometric fidelity, all metrics |

Negative feature-layer values must use the equals form, such as
`--feature-layers=-1,-3,-5,-7`, so that `argparse` does not interpret them as
new options. `concat` multiplies the PCA feature dimension by the number of
selected layers and therefore uses substantially more memory than `mean`.
Use `--layer-normalization l2` to normalize every patch token independently in
each selected hidden layer before layer fusion; `none` reproduces prior runs.
With `--dual-branch`, one DINOv3 forward pass produces both variants: the L2
branch supplies the image-level score and the raw branch supplies the anomaly
map. This is an optional P0 experiment and does not alter the default branch.

Run the P0.1 dual-branch comparison with the same normal-only protocol:

```bash
python -m defectfusion.cli evaluate-mvtec \
  --data-root /mnt/sda1/mvtec_anomaly \
  --model /mnt/sda1/DINOv3/dinov3-vitl16-pretrain-lvd1689m \
  --normal-shots 1 --defect-shots 0 --seed 42 \
  --normal-augment-count 30 --normal-augmentations rotate \
  --image-size 672 --resize-mode direct \
  --feature-layers=-1,-3,-5,-7 --layer-aggregation mean \
  --layer-normalization none --dual-branch \
  --image-score mtop1p --anomaly-method pca_knn \
  --fusion-mode fixed --knn-weight 0.5 --knn-spatial-radius -1 \
  --memory-max-patches 5000 --knn-backend torch --knn-dtype float16 \
  --output outputs/mvtec-dual-branch-seed42.json
```
Input images are resized to `--image-size` without center cropping. The default
`direct` mode reproduces existing experiments. `longest_pad` preserves aspect
ratio by resizing the longest side and symmetrically padding with the processor
mean color, which becomes approximately zero after normalization. Higher
resolutions increase patch count quadratically.

Compare resize geometry while keeping every other option fixed:

```bash
for mode in direct longest_pad; do
  python -m defectfusion.cli evaluate-mvtec \
    --data-root /mnt/sda1/mvtec_anomaly \
    --model /mnt/sda1/DINOv3/dinov3-vitl16-pretrain-lvd1689m \
    --normal-shots 1 --defect-shots 0 --seed 42 \
    --normal-augment-count 30 --normal-augmentations rotate \
    --image-size 672 --resize-mode "$mode" \
    --feature-layers=-1,-3,-5,-7 --layer-aggregation mean \
    --image-score mtop1p --anomaly-method pca_knn \
    --fusion-mode fixed --knn-weight 0.5 \
    --memory-max-patches 5000 --knn-backend torch --knn-dtype float16 \
    --output "outputs/mvtec-resize-${mode}.json"
done
```

Defect typing performs bidirectional matching between the PCA-selected Top-K
query patches and all defect-reference patches for each label. Its score is the
mean of query-to-reference and reference-to-query nearest-neighbour
similarities, which rewards both precise matches and reference coverage.
Use `--type-matching rbf_svm` to fit a class-balanced nonlinear SVM on the
same normalized Top-K reference patches. The classifier fits lazily once per
category, converts one-vs-rest margins to patch-level softmax confidence, and
averages them into an image-level defect-type score. Detection metrics are
unchanged by this option.

Compare the current matcher and RBF-SVM under the same labeled-defect
split:

```bash
for matcher in bidirectional_patch rbf_svm; do
  python -m defectfusion.cli evaluate-mvtec --data-root data/mvtec \
    --normal-shots -1 --defect-shots 1 --seed 42 \
    --image-size 672 --feature-layers=-1,-3,-5,-7 \
    --top-k-ratio 0.05 --type-matching "$matcher" \
    --anomaly-method pca_knn --fusion-mode fixed --knn-weight 0.5 \
    --memory-max-patches 5000 --knn-backend torch --knn-dtype float16 \
    --output "outputs/mvtec-type-${matcher}.json"
done
```

Anomaly-map post-processing is isolated from image scoring and defect typing.
Use `--map-postprocess none` for the raw-map baseline, `gaussian` for separable
Gaussian smoothing on the patch grid, or `crf` for RGB-guided DenseCRF. CRF
requires installing the optional dependency with `pip install -e '.[crf]'`.
Gaussian smoothing is disabled by default after the MVTec sigma=1.0 ablation
reduced macro pixel AUROC; it remains available only for explicit experiments.

### PCA and normal-patch kNN fusion

`--anomaly-method knn` uses the AnomalyDINO normal-patch memory score: cosine
distance to the closest normalized normal patch. `pca_knn` combines it with
the PCA reconstruction residual. `--pca-residual-metric mahalanobis` weights
each orthogonal residual dimension by its shrinkage-regularized normal variance;
the default `squared_l2` preserves the existing unweighted residual. `--fusion-mode fixed` reproduces robust
z-score calibration with the global `--knn-weight`. `gated` maps both raw
scores to empirical normal-tail evidence and assigns a soft kNN weight to
every patch. `--gate-temperature` controls how decisively the gate selects
the stronger expert; lower values approach hard selection.
The default remains `pca`, so existing baselines do not change. kNN queries
are chunked; lower `--knn-chunk-size` if inference runs out of memory. A
positive `--memory-max-patches` deterministically subsamples the normal bank;
use `0` to retain every patch.

### ANoCo-inspired manifold drift

`--anomaly-method anoco` replaces nearest-normal distance with a training-free
manifold-conformity cost. For every query patch it retrieves a normal anchor,
ranks normal neighbors by their joint similarity to the query and anchor,
pulls the query toward that consistent neighborhood in closed form, and scores
the squared feature displacement times its angular displacement.
`--anomaly-method pca_anoco` robustly calibrates and fuses this score with the
PCA residual using `--anoco-weight` (default `0.5`). The main controls are
`--anoco-neighbors 16`, `--anoco-query-weight 1.0`, and
`--anoco-temperature 0.07`. With multi-layer features,
`--anoco-layer-consensus` replaces aggregate-feature ANoCo evidence with the
per-patch median of independently calibrated layer drift scores. The implementation reuses the normal memory,
spatial candidate mask, CUDA backend, and chunk size used by kNN. It performs
two query-to-bank similarity products per chunk, so it is expected to be
roughly twice as expensive as the kNN head. The validated default detector is
unchanged; screen `anoco` on seed 42 before running `pca_anoco` or five seeds.

For the dual-head configuration, use `--anomaly-method pca_knn_anoco` together
with `--dual-branch`. The raw pixel branch uses calibrated PCA+kNN with
`--knn-weight`, while the L2 image branch uses calibrated PCA+ANoCo with
`--anoco-weight`. Each branch computes its PCA residual once and reuses it for
fusion and aggregation.
Set `--knn-spatial-radius 0.10` to restrict kNN candidates to a local window
in normalized patch coordinates. `-1` retains global matching. If leave-one-out
calibration finds no candidate in the local window, it falls back to the global
memory while still excluding the query patch itself.

For P0.3, `--image-fusion-stage patch` fuses calibrated PCA/kNN values per
patch before Top-K aggregation. `score` aggregates PCA and kNN independently
and then applies `--knn-weight`; the pixel map remains patch-first in both
cases.
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
  --output outputs/ablation-original
```

Compare PCA-residual Top-K typing ratios:

```bash
for ratio in 0.05 0.10 0.20 0.30 1.0; do
  python -m defectfusion.cli evaluate-mvtec --data-root data/mvtec \
    --normal-shots -1 --defect-shots 1 --seed 42 --top-k-ratio "$ratio" \
    --output "outputs/ablation-topk-${ratio}"
done
```

Compare image-level score aggregation while leaving pixel maps and typing
unchanged:

```bash
for score in mean mtop1p p99 max; do
  python -m defectfusion.cli evaluate-mvtec --data-root data/mvtec \
    --normal-shots -1 --defect-shots 1 --seed 42 --image-score "$score" \
    --output "outputs/ablation-image-${score}"
done
```

Compare single-layer and multi-layer features:

```bash
python -m defectfusion.cli evaluate-mvtec --data-root data/mvtec \
  --normal-shots -1 --defect-shots 1 --seed 42 --feature-layers=-1 \
  --output outputs/ablation-layer-last

python -m defectfusion.cli evaluate-mvtec --data-root data/mvtec \
  --normal-shots -1 --defect-shots 1 --seed 42 --feature-layers=-1,-2,-3,-4 \
  --layer-aggregation mean --output outputs/ablation-layer-last4-mean

python -m defectfusion.cli evaluate-mvtec --data-root data/mvtec \
  --normal-shots -1 --defect-shots 1 --seed 42 --feature-layer-preset middle7 \
  --layer-aggregation mean --output outputs/ablation-layer-middle7-mean
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
    --output "outputs/ablation-seed-${seed}"
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
