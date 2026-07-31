# VisA strict-minimum ANoCo anchor ranking

P4, VisA 1-shot, seed 42. Affinity is `softmax` and query weight is `2.0`.

```bash
python -m defectfusion.cli evaluate-visa --data-root /mnt/sda1/VisA_20220922 --model /mnt/sda1/DINOv3/dinov3-vitl16-pretrain-lvd1689m --normal-shots 1 --defect-shots 0 --seed 42 --image-size 672 --image-size-override macaroni2=896 --image-size-override pcb2=896 --image-size-override pcb3=896 --pixel-image-size-override fryum=896 --image-head-size-override pcb4=896 --pixel-multiscale-size-override macaroni2=672 --pixel-multiscale-size-override pcb2=672 --pixel-multiscale-size-override pcb3=672 --pixel-multiscale-weight 0.25 --normal-augment-count 30 --normal-augmentations rotate --affine-categories macaroni1 macaroni2 --feature-layers=1,17,21,23 --dual-branch --anomaly-method pca_knn_anoco --knn-weight 0.5 --anoco-neighbors 16 --anoco-query-weight 2.0 --anoco-temperature 0.07 --anoco-affinity softmax --anoco-anchor-ranking minimum --anoco-weight 0.25 --anoco-layer-consensus --image-score mtop1p --image-top-ratio 0.01 --image-min-component-size 2 --component-reject-categories macaroni1 macaroni2 --image-fusion-stage patch --knn-backend torch --knn-dtype float16 --knn-spatial-radius -1 --output outputs/visa-anoco-anchor-ranking-minimum
```

| Metric | Mean | Minimum | Change (points) |
|---|---:|---:|---:|
| I-AUROC | 93.8020 | 93.5998 | -0.2022 |
| I-AUPR | 93.2365 | 93.0107 | -0.2258 |
| I-F1-MAX | 90.3768 | 90.1789 | -0.1979 |
| P-AUROC | 98.4713 | 98.4713 | 0.0000 |
| P-AUPR | 44.8090 | 44.8090 | 0.0000 |
| PRO | 93.8848 | 93.8848 | 0.0000 |
| P-F1-MAX | 49.8948 | 49.8948 | 0.0000 |

Decision: reject minimum because its I-AUROC regression exceeds the
0.10-point gate and all image metrics decline. Retain mean for P5.
