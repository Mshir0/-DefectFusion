# DefectFusion

Training-free and few-shot defect detection and defect typing, combining ideas from AnomalyDINO, SubspaceAD, and INSID3.

## Design

- Frozen dense DINO features (DINOv2 by default; a compatible DINOv3 wrapper can be supplied).
- Position debiasing inspired by INSID3.
- Foreground-aware normal modeling with a streaming PCA subspace inspired by SubspaceAD.
- Optional k-nearest-neighbour memory score inspired by AnomalyDINO.
- Few-shot defect prototypes and zero-shot text prototypes are pluggable classifiers.

## Quick start

```bash
python -m defectfusion.cli fit --normal-dir data/normal --model facebook/dinov2-small
python -m defectfusion.cli predict --model-state outputs/model.npz --image image.jpg
```

The default classifier reports `unknown` when no prototype is sufficiently confident. Model weights are downloaded by HuggingFace/torch at runtime.
