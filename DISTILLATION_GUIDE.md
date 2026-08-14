# DINOv3 Teacher-Student Distillation

All distillation code and dataset parsing live in the standalone
`distill_dinov3.py` file. It does not read environment variables or use a
package-internal distillation module. Dataset, model, and output locations are
provided as normal command-line arguments. Paths may be relative to the
repository or absolute Linux paths. Its post-training evaluation deliberately
calls the existing DefectFusion evaluator so metric definitions stay identical.

The script freezes a DINOv3 ViT-B teacher and adapts one DINOv3 ViT-S+ student
per selected category. It uses hidden states 1, 6, and 12 for patch-feature
distillation, anomaly-map distillation, and QKV LoRA in the final four blocks.
The ViT-S+ base and its LayerNorms remain frozen. Projection heads exist only
while computing the training loss and are discarded after training. After every
MVTec/VisA run, the script reloads the original ViT-S+ checkpoint, attaches the
saved LoRA adapter, and evaluates it with the project's existing DefectFusion
evaluator. This reports the same image-level AUROC/AUPR/F1-max and pixel-level
AUROC/AUPR/AUPRO/F1-max metrics as `evaluate-mvtec` and `evaluate-visa`.
It also writes a thresholded `good`/`anomaly` result for every test image. The
threshold is the maximum score over the selected normal training images; normal
test images are not included in pixel/localization metrics.

## Installation

Use Python 3.10 or newer. Install the PyTorch build matching the Linux
server's CUDA version, followed by the project dependencies:

```bash
git clone https://github.com/Mshir0/-DefectFusion.git
cd ./-DefectFusion

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

The default teacher and student are Hugging Face checkpoints. Pass local model
directories to `--teacher-model` and `--student-model` for offline execution.
A local reference must point at the actual Hugging Face snapshot directory,
including its `config.json`, rather than the parent `DINOv3/` directory. The
trainer resolves an existing local directory to its physical absolute path and
loads it with `local_files_only=True`; no Hub lookup is attempted for it. If a
path that appears to exist still fails, verify it from the same Python
environment that launches training:

```bash
python -c 'from pathlib import Path; p = Path("/mnt/sda1/DINOv3/dinov3-vits16plus-pretrain-lvd1689m"); print(repr(str(p)), p.exists(), p.is_dir(), (p / "config.json").is_file(), p.resolve())'
```

## MVTec AD

The script discovers each category from the standard layout:

```text
mvtec_anomaly/
  bottle/
    train/good/
    test/<defect_type>/
    ground_truth/<defect_type>/*_mask.png
```

Train selected categories with normal data only:

```bash
python distill_dinov3.py \
  --dataset mvtec \
  --data-root ./datasets/mvtec_anomaly \
  --categories bottle cable \
  --normal-shots 8 \
  --defect-shots 0 \
  --teacher-model ./models/dinov3-vitb16-pretrain-lvd1689m \
  --student-model ./models/dinov3-vits16plus-pretrain-lvd1689m \
  --output ./outputs/mvtec-distilled \
  --epochs 10 --batch-size 2 --device cuda
```

Omit `--categories` to process every detected category. Use
`--normal-shots -1` for all `train/good` images.

## VisA

The official `split_csv/1cls.csv` is used automatically. A custom split can be
provided with `--split-csv`. If no split CSV exists, the script accepts the
raw release layout `Data/Images/Normal`, `Data/Images/Anomaly`, and
`Data/Masks/Anomaly`.

```bash
python distill_dinov3.py \
  --dataset visa \
  --data-root ./datasets/VisA_20220922 \
  --categories candle capsules \
  --normal-shots 8 \
  --defect-shots 0 \
  --teacher-model ./models/dinov3-vitb16-pretrain-lvd1689m \
  --student-model ./models/dinov3-vits16plus-pretrain-lvd1689m \
  --output ./outputs/visa-distilled \
  --epochs 10 --batch-size 2 --device cuda
```

`--defect-shots N` explicitly selects up to N test defects per defect type and
uses their masks for defect-weighted distillation. It defaults to `0` because
using test defects is data leakage in the standard unsupervised protocol.
If it is set above zero, the selected test images are excluded from the final
automatic evaluation so reported metrics remain held-out.

## All MVTec AD and VisA Categories

`scripts/distill_all_mvtec_visa.sh` runs the standalone trainer once for each
dataset. It deliberately omits `--categories`, so `distill_dinov3.py` discovers
and trains every valid category sequentially, then writes one project-standard
evaluation summary per dataset. Dataset and model paths are explicit script
arguments; no environment variables or server-specific paths are read.

```bash
bash scripts/distill_all_mvtec_visa.sh \
  --mvtec-root /mnt/sda1/mvtec_anomaly \
  --visa-root /mnt/sda1/VisA_20220922 \
  --teacher-model /mnt/sda1/DINOv3/dinov3-vitb16-pretrain-lvd1689m \
  --student-model /mnt/sda1/DINOv3/dinov3-vits16plus-pretrain-lvd1689m \
  --output-root ./outputs/dinov3-all-8shot
```

The default is the same 8-shot, 448-pixel, rank-8 LoRA configuration used by
the focused example. Use `--normal-shots -1` to train each category with all
available normal images. After a completed run, `--skip-completed` skips a
dataset only when its `evaluation/results.json` exists. The results are stored
under `./outputs/dinov3-all-8shot/mvtec/` and
`./outputs/dinov3-all-8shot/visa/`, each with `evaluation/results.json` and
`evaluation/summary.csv` containing the existing project's metrics.

## Outputs

Each category is trained independently:

```text
outputs/mvtec-distilled/
  bottle/
    lora_adapter.pt
    training_config.json
    summary.json
  cable/
    ...
  evaluation/
    categories/<category>.json
    results.json
    summary.csv
  summary.json
```

`lora_adapter.pt` contains only the trained LoRA matrices. It does not contain
the ViT-S+ base weights, LayerNorm weights, projection heads, processor files,
or teacher weights. Keep the original `--student-model` checkpoint available:
automatic evaluation reloads that checkpoint and attaches each category's
adapter in memory. `summary.json` records both `trainable_parameters` (LoRA
plus training-only projection heads) and `lora_parameters` (the weights that
are actually saved). The automatic `evaluation/` directory uses the same JSON
and CSV layout as the existing evaluator. Its detector defaults are PCA,
`--eval-image-size` equal to the training image size, and the same selected
ViT-S+ hidden states. For 2-shot and above, the image decision threshold defaults
to source-disjoint leave-one-out calibration with a `0.995` higher quantile.
This affects only evaluation and does not change the adapter artifact. A 1-shot
run must explicitly use `--eval-normal-decision-calibration training-reference`,
because source-disjoint LOO has no remaining source from which to fit a fold.
Override detector settings through explicit `--eval-*` flags, for example
`--eval-anomaly-method pca_knn --eval-dual-branch --eval-knn-dtype float16`.
Non-PCA or dual-branch evaluation currently also requires
`--eval-normal-decision-calibration training-reference`.
Use `--no-evaluate` to skip post-training evaluation while retaining only the
adapter artifact. The generic `defectfusion` CLI accepts complete Hugging Face
models, so the adapter-only workflow uses `distill_dinov3.py`'s automatic
post-training evaluation with the same `--student-model` used during training.
Existing full-model artifacts from earlier runs are not deleted automatically;
use a new `--output` directory or remove those old directories after checking
their contents.

Run `python distill_dinov3.py --help` for all LoRA, loss, precision, and
adapter snapshot options.
