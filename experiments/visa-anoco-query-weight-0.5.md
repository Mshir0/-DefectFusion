# VisA ANoCo query weight 0.5

P3 intermediate run, VisA 1-shot, seed 42. Affinity remains `softmax`.

```bash
python -m defectfusion.cli evaluate-visa --data-root /mnt/sda1/VisA_20220922 --model /mnt/sda1/DINOv3/dinov3-vitl16-pretrain-lvd1689m --normal-shots 1 --defect-shots 0 --seed 42 --image-size 672 --image-size-override macaroni2=896 --image-size-override pcb2=896 --image-size-override pcb3=896 --pixel-image-size-override fryum=896 --image-head-size-override pcb4=896 --pixel-multiscale-size-override macaroni2=672 --pixel-multiscale-size-override pcb2=672 --pixel-multiscale-size-override pcb3=672 --pixel-multiscale-weight 0.25 --normal-augment-count 30 --normal-augmentations rotate --affine-categories macaroni1 macaroni2 --feature-layers=1,17,21,23 --dual-branch --anomaly-method pca_knn_anoco --knn-weight 0.5 --anoco-neighbors 16 --anoco-query-weight 0.5 --anoco-temperature 0.07 --anoco-affinity softmax --anoco-weight 0.25 --anoco-layer-consensus --image-score mtop1p --image-top-ratio 0.01 --image-min-component-size 2 --component-reject-categories macaroni1 macaroni2 --image-fusion-stage patch --knn-backend torch --knn-dtype float16 --knn-spatial-radius -1 --output outputs/visa-anoco-query-weight-0.5
```

| Metric | Weight 1.0 baseline | Weight 0.5 | Change (points) |
|---|---:|---:|---:|
| I-AUROC | 93.8059 | 93.8203 | +0.0144 |
| I-AUPR | 93.1937 | 93.1838 | -0.0099 |
| I-F1-MAX | 90.2868 | 90.3160 | +0.0292 |
| P-AUROC | 98.4713 | 98.4713 | 0.0000 |
| P-AUPR | 44.8090 | 44.8090 | 0.0000 |
| PRO | 93.8848 | 93.8848 | 0.0000 |
| P-F1-MAX | 49.8948 | 49.8948 | 0.0000 |

Interim decision: retain as the current P3 leader, but do not promote until
weights `2.0` and `4.0` have been evaluated.
