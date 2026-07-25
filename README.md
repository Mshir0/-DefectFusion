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
and prints image-level and pixel-level AUROC when both classes are present.
Use `--data-root data/mvtec` to evaluate every category under the MVTec root;
the CLI prints each image as it is processed and writes one JSONL file per
category. For a multi-category run, `--output` stores the final combined JSON
summary, including macro averages and every category result; per-image files
use `<output-stem>-<category>.jsonl`. Add `--prototype-dir` (subdirectories are defect labels) to enable
defect-type accuracy, macro-F1, and a confusion matrix.

For a reproducible MVTec few-shot experiment, randomly select N test images
per defect type as prototypes (selected images are excluded from evaluation):

```bash
python -m defectfusion.cli evaluate-mvtec --data-root data/mvtec \
  --few-shot 1 --seed 42 --output outputs/mvtec-all.jsonl
```

Enable INSID3 positional debiasing for an ablation with the same split:

```bash
python -m defectfusion.cli evaluate-mvtec --data-root data/mvtec \
  --few-shot 1 --seed 42 --debias --svd-components 20 \
  --output outputs/mvtec-all-debiased.jsonl
```

Defect typing uses the highest PCA-reconstruction-residual patches instead of
the whole-image mean. Control their fraction with `--top-k-ratio` (default
`0.05`, selected by the MVTec 1-shot ablation); use `1.0` to reproduce
whole-image prototype classification.

Image-level anomaly scores use the mean of the highest-scoring 1% of patch
residuals (`--image-score mtop1p`). Use `--image-score mean` to reproduce the
previous whole-image score; `max` and `p99` are also available for ablation.

Dense features are averaged from the final four DINOv3 transformer blocks by
default. Override this with `--feature-layers=-1` for the former last-layer
baseline, or select layers and concatenation explicitly, for example
`--feature-layers=-1,-3,-5 --layer-aggregation concat`.

## Ablation experiments

Keep the dataset, backbone, few-shot seed, and all unrelated parameters fixed
when comparing one option. The current recommended setup is:

```bash
python -m defectfusion.cli evaluate-mvtec \
  --data-root data/mvtec \
  --model facebook/dinov3-vit7b16-pretrain-lvd1689m \
  --few-shot 1 --seed 42 \
  --top-k-ratio 0.05 \
  --type-matching bidirectional_patch \
  --image-score mtop1p \
  --feature-layers=-1,-2,-3,-4 \
  --layer-aggregation mean \
  --output outputs/ablation-recommended.jsonl
```

### Parameters

| Dimension | Argument | Current default | Ablation values | Main affected metrics |
|---|---|---:|---|---|
| Few-shot count | `--few-shot` | `0` | `0, 1, 3, 5` | Type Accuracy, Macro-F1 |
| Sampling seed | `--seed` | `42` | `0, 1, 2, 42` | Few-shot variance |
| Typing patch ratio | `--top-k-ratio` | `0.05` | `0.05, 0.10, 0.20, 0.30, 1.0` | Type Accuracy, Macro-F1 |
| Image score | `--image-score` | `mtop1p` | `mean, mtop1p, p99, max` | Image AUROC |
| Type matching | `--type-matching` | `bidirectional_patch` | `prototype_mean, bidirectional_patch` | Type Accuracy, Macro-F1 |
| Feature layers | `--feature-layers` | `-1,-2,-3,-4` | `-1`, `-1,-2`, `-1,-2,-3,-4`, `-1,-3,-5` | All metrics |
| Layer fusion | `--layer-aggregation` | `mean` | `mean, concat` | All metrics and memory |
| Multi-scale views | `--multiscale-mode` | `overlap` | `none, overlap` | All metrics and runtime |
| Crop size | `--crop-ratio` | `0.75` | `0.50, 0.75` | All metrics and runtime |
| Crop overlap | `--crop-overlap` | `0.50` | `0.25, 0.50` | All metrics and runtime |
| Positional debiasing | `--debias` | off | off/on | All metrics |
| Debias rank | `--svd-components` | `20` | `2, 5, 10, 20` | Active only with `--debias` |
| Device | `--device` | auto | `cpu, cuda` | Runtime only |

Negative feature-layer values must use the equals form, such as
`--feature-layers=-1,-2,-3,-4`, so that `argparse` does not interpret them as
new options. `concat` multiplies the PCA feature dimension by the number of
selected layers and therefore uses substantially more memory than `mean`.

Defect typing performs bidirectional matching between the PCA-selected Top-K
query patches and all few-shot patches for each defect label. Its score is the
mean of query-to-reference and reference-to-query nearest-neighbour
similarities, which rewards both precise matches and reference coverage.

Multi-scale inference combines the full image with overlapping local crops.
Every view is encoded at the backbone input resolution; crop anomaly maps are
stitched into the original image coordinates and averaged in overlap regions.
Use `--multiscale-mode none` for the former full-image-only baseline. The
default 0.75 crop ratio produces four local crops plus the full-image view.

### Single-variable commands

Reproduce the original whole-image typing and image-score baseline:

```bash
python -m defectfusion.cli evaluate-mvtec --data-root data/mvtec \
  --model facebook/dinov3-vit7b16-pretrain-lvd1689m \
  --few-shot 1 --seed 42 --top-k-ratio 1.0 --image-score mean \
  --feature-layers=-1 --layer-aggregation mean \
  --output outputs/ablation-original.jsonl
```

Compare PCA-residual Top-K typing ratios:

```bash
for ratio in 0.05 0.10 0.20 0.30 1.0; do
  python -m defectfusion.cli evaluate-mvtec --data-root data/mvtec \
    --few-shot 1 --seed 42 --top-k-ratio "$ratio" \
    --output "outputs/ablation-topk-${ratio}.jsonl"
done
```

Compare image-level score aggregation while leaving pixel maps and typing
unchanged:

```bash
for score in mean mtop1p p99 max; do
  python -m defectfusion.cli evaluate-mvtec --data-root data/mvtec \
    --few-shot 1 --seed 42 --image-score "$score" \
    --output "outputs/ablation-image-${score}.jsonl"
done
```

Compare single-layer and multi-layer features:

```bash
python -m defectfusion.cli evaluate-mvtec --data-root data/mvtec \
  --few-shot 1 --seed 42 --feature-layers=-1 \
  --output outputs/ablation-layer-last.jsonl

python -m defectfusion.cli evaluate-mvtec --data-root data/mvtec \
  --few-shot 1 --seed 42 --feature-layers=-1,-2,-3,-4 \
  --layer-aggregation mean --output outputs/ablation-layer-last4-mean.jsonl
```

Measure few-shot stability with multiple seeds. Report the mean and standard
deviation across runs rather than selecting the best seed:

```bash
for seed in 0 1 2; do
  python -m defectfusion.cli evaluate-mvtec --data-root data/mvtec \
    --few-shot 1 --seed "$seed" \
    --output "outputs/ablation-seed-${seed}.jsonl"
done
```
