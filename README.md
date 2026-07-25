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
category. Add `--prototype-dir` (subdirectories are defect labels) to enable
defect-type accuracy, macro-F1, and a confusion matrix.

For a reproducible MVTec few-shot experiment, randomly select N test images
per defect type as prototypes (selected images are excluded from evaluation):

```bash
python -m defectfusion.cli evaluate-mvtec --data-root data/mvtec \
  --few-shot 1 --seed 42 --output outputs/mvtec-all.jsonl
```
