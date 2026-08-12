# DINOv3 Teacher-Student Distillation

This training path freezes a DINOv3 ViT-B teacher and adapts a DINOv3 ViT-S
student with QKV LoRA in the final four blocks. It distills patch tokens from
hidden states 1, 6, and 12 and the resulting anomaly map. Synthetic masks, when
available, increase the feature and map loss inside defect regions.

## 1. Linux environment

Use Python 3.10 or newer. Install the PyTorch build matching the server's CUDA
version first, then install DefectFusion:

```bash
git clone https://github.com/Mshir0/-DefectFusion.git
cd ./-DefectFusion

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Select the matching command at https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

The default model identifiers are Hugging Face DINOv3 checkpoints. If access
requires authentication, run `hf auth login`. Offline/local model directories
can be supplied through `TEACHER_MODEL` and `STUDENT_MODEL` instead.

## 2. Data layout

The recommended layout is:

```text
data/
  normal/                 # normal training images
  synthetic/images/       # optional generated defect images
  synthetic/masks/        # optional binary masks
```

Mask files may use the image stem (`000.png`) or a suffix such as
`000_mask.png`, `000_label.png`, or `000_seg.png`. Mirrored image/mask
subdirectories are also supported. Defect images can be omitted for
normal-only distillation.

## 3. Train

```bash
chmod +x scripts/train_dinov3_distill.sh

NORMAL_DIR=/data/normal \
DEFECT_DIR=/data/synthetic/images \
MASK_DIR=/data/synthetic/masks \
OUTPUT_DIR=outputs/dinov3-vits-distilled \
BATCH_SIZE=2 EPOCHS=10 \
bash scripts/train_dinov3_distill.sh
```

For local weights:

```bash
TEACHER_MODEL=/models/dinov3-vitb16-pretrain-lvd1689m \
STUDENT_MODEL=/models/dinov3-vits16-pretrain-lvd1689m \
NORMAL_DIR=/data/normal \
DEFECT_DIR=/data/synthetic/images \
MASK_DIR=/data/synthetic/masks \
bash scripts/train_dinov3_distill.sh
```

Useful overrides include `LORA_RANK=8`, `LAST_N_BLOCKS=4`, `MASK_ALPHA=2.0`,
`LAMBDA_FEATURE=1.0`, `LAMBDA_MAP=1.0`, `IMAGE_SIZE=448`, and `AMP=0`. Reduce
`BATCH_SIZE` and `CENTROID_BATCH_SIZE` first if GPU memory is insufficient.

The final output contains:

```text
outputs/dinov3-vits-distilled/
  student_base/           # original ViT-S used to reconstruct adapters
  processor/              # preprocessing config
  distill_checkpoint.pt   # LoRA, LayerNorm, projections, centroids, metadata
  student_merged/         # LoRA-merged standard Hugging Face ViT-S
  training_config.json
  summary.json
```

Use `student_merged/` as the model in the existing detector. Its valid default
hidden states are `1,6,12`, not the ViT-L/7B setting `1,17,21,23`.

## 4. Train and evaluate

MVTec AD:

```bash
RUN_EVAL=1 EVAL_KIND=mvtec \
EVAL_DATA_ROOT=/data/mvtec_anomaly \
NORMAL_DIR=/data/normal \
DEFECT_DIR=/data/synthetic/images \
MASK_DIR=/data/synthetic/masks \
bash scripts/train_dinov3_distill.sh
```

VisA:

```bash
RUN_EVAL=1 EVAL_KIND=visa \
EVAL_DATA_ROOT=/data/VisA_20220922 \
NORMAL_DIR=/data/normal \
DEFECT_DIR=/data/synthetic/images \
MASK_DIR=/data/synthetic/masks \
bash scripts/train_dinov3_distill.sh
```

To evaluate an already-trained model without retraining, use the existing CLI
and set `--model /path/to/student_merged --feature-layers=1,6,12`.
