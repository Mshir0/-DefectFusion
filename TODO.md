# Improvement roadmap

- [x] Aggregate the image anomaly score from the highest-scoring 1% of patches.
- [x] Fuse features from multiple DINOv3 layers.
- [x] Evaluate foreground/saliency suppression (removed after negative MVTec ablation).
- [x] Add bidirectional patch matching for anomalous regions.
- [x] Evaluate overlapping multi-scale crops (removed after negative MVTec ablation).
- [x] Evaluate clustered defect prototypes (removed after negative MVTec ablation).
- [x] Evaluate Gaussian smoothing (disabled by default after negative MVTec ablation); retain optional CRF refinement.

## Paper-derived next steps

- [x] Add raw/L2 dual-branch scoring: L2 for image AUROC, raw for pixel maps.
- [x] Add configurable Top-K image aggregation ratios (0.5/1/2/5%).
- [x] Add aspect-ratio-preserving longest-side resize with mean-color padding for ablation.
- [x] Add optional per-layer L2 normalization before DINOv3 feature fusion.
- [x] Add optional spatially constrained normal-memory kNN matching.
- [x] Fuse PCA residuals with an AnomalyDINO-style normal-patch kNN memory score.
- [x] Evaluate independent per-layer PCA score fusion (removed after negative MVTec ablation).
- [x] Add training-free normal-tail-calibrated gating for PCA and kNN fusion.
- [ ] Apply INSID3 positional debiasing only to cross-image defect matching.
- [ ] Add PCA score ablations: cosine residual, dropped PCs, and Mahalanobis residual.
- [ ] Score each defect reference independently and robustly aggregate reference scores.
- [ ] Calibrate uncertainty from agreement across multiple normal references.
- [x] Evaluate Shrinkage LDA defect typing (removed after negative MVTec ablation).
- [x] Add a class-balanced RBF-SVM defect-type classifier for Top-K anomaly patches.
