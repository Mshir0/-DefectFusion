# Improvement roadmap

- [x] Aggregate the image anomaly score from the highest-scoring 1% of patches.
- [x] Fuse features from multiple DINOv3 layers.
- [x] Evaluate foreground/saliency suppression (removed after negative MVTec ablation).
- [x] Add bidirectional patch matching for anomalous regions.
- [x] Evaluate overlapping multi-scale crops (removed after negative MVTec ablation).
- [x] Evaluate clustered defect prototypes (removed after negative MVTec ablation).
- [x] Evaluate Gaussian smoothing (disabled by default after negative MVTec ablation); retain optional CRF refinement.

## Paper-derived next steps

- [x] Fuse PCA residuals with an AnomalyDINO-style normal-patch kNN memory score.
- [x] Fit independent PCA models per feature layer and fuse calibrated anomaly maps.
- [ ] Apply INSID3 positional debiasing only to cross-image defect matching.
- [ ] Add PCA score ablations: cosine residual, dropped PCs, and Mahalanobis residual.
- [ ] Score each defect reference independently and robustly aggregate reference scores.
- [ ] Calibrate uncertainty from agreement across multiple normal references.
