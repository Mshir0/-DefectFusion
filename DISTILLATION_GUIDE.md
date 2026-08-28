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
the focused example. MVTec inherits the global robust LOO settings
(`linear@0.95` across sources and `linear@0.90` within each source's views).
VisA defaults to the restored legacy strategy: `higher@0.995` across sources
and the maximum within each source's views. Use `--visa-threshold-profile
inherit` only when intentionally rerunning VisA with the global settings. Use
`--normal-shots -1` to train each category with all available normal images.
After a completed run, `--skip-completed` skips a dataset only when its selected
`evaluation/results.json` exists. The results are stored
under `./outputs/dinov3-all-8shot/mvtec/` and
`./outputs/dinov3-all-8shot/visa/`, each with `evaluation/results.json` and
`evaluation/summary.csv` containing the existing project's metrics.

## Improve The VisA Teacher First

Before retraining the student, evaluate the frozen ViT-B teacher directly on
VisA. This separates teacher quality from student capacity, LoRA adaptation,
and the lower default student input size. Run the current conservative
calibration and the balanced candidate as separate experiments:

```bash
bash scripts/evaluate_visa_teacher.sh \
  --visa-root /mnt/sda1/VisA_20220922 \
  --teacher-model /mnt/sda1/DINOv3/dinov3-vitb16-pretrain-lvd1689m \
  --profile baseline \
  --output-root outputs/visa-teacher

bash scripts/evaluate_visa_teacher.sh \
  --visa-root /mnt/sda1/VisA_20220922 \
  --teacher-model /mnt/sda1/DINOv3/dinov3-vitb16-pretrain-lvd1689m \
  --profile visa-balanced \
  --output-root outputs/visa-teacher
```

`visa-balanced` uses source-disjoint LOO but changes the decision calibration
from `higher@0.995` with 30 held-out views to `linear@0.90` with 10 views.
It is a candidate operating point, not a test-label-derived optimum. Compare
Good Accuracy, Defect Recall, and Balanced Accuracy; use an independent normal
validation split to choose a deployment threshold. The high-resolution VisA
overrides and ViT-B layers `1,6,12` are retained for both profiles.

Once a teacher-side configuration is validated, pass its calibration options
to `scripts/distill_all_mvtec_visa.sh`; that script now supports
`--eval-normal-decision-*` and explicit `--teacher-layers` /
`--student-layers` pairs for a later stronger-teacher experiment.

## Legacy VisA Threshold Evaluation For Existing ViT-S+ Adapters

Threshold calibration is evaluation-only. Existing LoRA adapters do not need
to be distilled again. For example, re-evaluate all existing VisA adapters and
write the restored legacy-threshold results to a separate directory:

```bash
python distill_dinov3.py \
  --dataset visa \
  --data-root /mnt/sda1/VisA_20220922 \
  --teacher-model /mnt/sda1/DINOv3/dinov3-vitb16-pretrain-lvd1689m \
  --student-model /mnt/sda1/DINOv3/dinov3-vits16plus-pretrain-lvd1689m \
  --output outputs/shot-distillation-benchmark/distillation/visa \
  --evaluate-only \
  --eval-output outputs/shot-distillation-benchmark/distillation-thresholds/visa/evaluation \
  --normal-shots 8 \
  --defect-shots 0 \
  --seed 42 \
  --device cuda \
  --eval-normal-decision-calibration leave-one-out \
  --eval-normal-decision-quantile 0.995 \
  --eval-normal-decision-quantile-method higher \
  --eval-normal-decision-view-quantile 1.0 \
  --eval-normal-decision-augment-count 30 \
  --eval-normal-decision-fit-augment-count 4
```

`--output` is the adapter root. When it contains `<category>/lora_adapter.pt`,
`--evaluate-only` reuses those adapters. When the root is missing or contains
no adapters, the entry point creates it and automatically switches to a full
distillation run; in that case, provide a usable `--teacher-model`. `--eval-output`
is the exact directory for the new `results.json`, `summary.csv`, and
per-category JSON files. Use the same `--normal-shots` and `--seed` as the
distillation run so the evaluator reconstructs the same normal-shot selection.

The legacy calibration takes the maximum of the original normal image and its
held-out rotation views within each LOO fold. It then takes the `0.995` higher
quantile across the source-disjoint fold scores. The reported source is
`normal_leave_one_out_quantile`. This changes only binary image decisions;
LoRA files and ranking metrics are not modified.

To re-evaluate both datasets in one command, point the all-category script at
the existing adapter root. In evaluate-only mode, the result root defaults to
`<output-root>-thresholds`. VisA uses the legacy profile by default:

```bash
bash scripts/distill_all_mvtec_visa.sh \
  --mvtec-root /mnt/sda1/mvtec_anomaly \
  --visa-root /mnt/sda1/VisA_20220922 \
  --teacher-model /mnt/sda1/DINOv3/dinov3-vitb16-pretrain-lvd1689m \
  --student-model /mnt/sda1/DINOv3/dinov3-vits16plus-pretrain-lvd1689m \
  --output-root outputs/shot-distillation-benchmark/distillation \
  --eval-output-root outputs/shot-distillation-benchmark/distillation-thresholds \
  --normal-shots 8 \
  --evaluate-only
```

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
ViT-S+ hidden states. The Python entry point retains the historical
`higher@0.995` / per-source maximum defaults. The all-category Bash script
keeps its global robust settings for MVTec but restores those historical
defaults for VisA.
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
