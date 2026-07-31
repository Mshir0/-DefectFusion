# VisA normalized cosine ANoCo affinity

P2, VisA 1-shot, seed 42. Implementation commit: `3e890de`.

```bash
python -m defectfusion.cli evaluate-visa --data-root /mnt/sda1/VisA_20220922 --model /mnt/sda1/DINOv3/dinov3-vitl16-pretrain-lvd1689m --normal-shots 1 --defect-shots 0 --seed 42 --image-size 672 --image-size-override macaroni2=896 --image-size-override pcb2=896 --image-size-override pcb3=896 --pixel-image-size-override fryum=896 --image-head-size-override pcb4=896 --pixel-multiscale-size-override macaroni2=672 --pixel-multiscale-size-override pcb2=672 --pixel-multiscale-size-override pcb3=672 --pixel-multiscale-weight 0.25 --normal-augment-count 30 --normal-augmentations rotate --affine-categories macaroni1 macaroni2 --feature-layers=1,17,21,23 --dual-branch --anomaly-method pca_knn_anoco --knn-weight 0.5 --anoco-neighbors 16 --anoco-query-weight 1.0 --anoco-temperature 0.07 --anoco-affinity cosine --anoco-weight 0.25 --anoco-layer-consensus --image-score mtop1p --image-top-ratio 0.01 --image-min-component-size 2 --component-reject-categories macaroni1 macaroni2 --image-fusion-stage patch --knn-backend torch --knn-dtype float16 --knn-spatial-radius -1 --output outputs/visa-anoco-cosine-affinity
```

| Metric | Softmax baseline | Cosine | Change (points) |
|---|---:|---:|---:|
| I-AUROC | 93.8059 | 93.6829 | -0.1230 |
| I-AUPR | 93.1937 | 93.0851 | -0.1086 |
| I-F1-MAX | 90.2868 | 90.0937 | -0.1931 |
| P-AUROC | 98.4713 | 98.4713 | 0.0000 |
| P-AUPR | 44.8090 | 44.8090 | 0.0000 |
| PRO | 93.8848 | 93.8848 | 0.0000 |
| P-F1-MAX | 49.8948 | 49.8948 | 0.0000 |

Decision: reject cosine because I-AUROC regressed by 0.1230 points, exceeding
the 0.10-point preservation gate. Retain softmax for P3. Exact equality of all
pixel metrics confirms that this ablation changed only the image ANoCo head.
