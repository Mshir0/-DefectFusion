# DINOv3 Teacher-Student Distillation

All distillation code and dataset parsing live in the standalone
`distill_dinov3.py` file. It does not read environment variables or use a
package-internal distillation module. Dataset, model, and output locations are
provided as normal command-line arguments. Paths may be relative to the
repository or absolute Linux paths. Its post-training evaluation deliberately
calls the existing DefectFusion evaluator so metric definitions stay identical.

The script freezes a DINOv3 ViT-B teacher and adapts one DINOv3 ViT-S student
per selected category. It uses hidden states 1, 6, and 12 for patch-feature
distillation, anomaly-map distillation, QKV LoRA in the final four blocks, and
trainable LayerNorm/projection parameters. After every MVTec/VisA run it also
evaluates the merged student with the project's existing DefectFusion
evaluator. This reports the same image-level AUROC/AUPR/F1-max and pixel-level
AUROC/AUPR/AUPRO/F1-max metrics as `evaluate-mvtec` and `evaluate-visa`.

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
  --student-model ./models/dinov3-vits16-pretrain-lvd1689m \
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
  --student-model ./models/dinov3-vits16-pretrain-lvd1689m \
  --output ./outputs/visa-distilled \
  --epochs 10 --batch-size 2 --device cuda
```

`--defect-shots N` explicitly selects up to N test defects per defect type and
uses their masks for defect-weighted distillation. It defaults to `0` because
using test defects is data leakage in the standard unsupervised protocol.
If it is set above zero, the selected test images are excluded from the final
automatic evaluation so reported metrics remain held-out.

## Outputs

Each category is trained independently:

```text
outputs/mvtec-distilled/
  bottle/
    student_base/
    processor/
    distill_checkpoint.pt
    student_merged/
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

`student_merged/` is a standard Hugging Face model with LoRA weights merged.
The automatic `evaluation/` directory uses the same JSON and CSV layout as the
existing evaluator. Its detector defaults are PCA, `--eval-image-size` equal
to the training image size, and the same selected ViT-S hidden states. Override
them through explicit `--eval-*` flags, for example
`--eval-anomaly-method pca_knn --eval-dual-branch --eval-knn-dtype float16`.
Use `--no-evaluate` to export a model without the post-training evaluation.

The merged model can also be evaluated manually through the existing CLI:

```bash
python -m defectfusion.cli evaluate-mvtec \
  --data-root ./datasets/mvtec_anomaly --categories bottle \
  --model ./outputs/mvtec-distilled/bottle/student_merged \
  --feature-layers=1,6,12 --image-size 672 --device cuda \
  --normal-shots 8 --defect-shots 0 \
  --dual-branch --anomaly-method pca_knn_anoco \
  --anoco-layer-consensus --output outputs/mvtec-bottle-eval
```

Run `python distill_dinov3.py --help` for all LoRA, loss, precision, and
checkpoint options.
