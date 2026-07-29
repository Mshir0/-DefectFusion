# Improvement roadmap

## Performance remediation

- [x] Skip defect-type Top-K selection and the duplicate raw-PCA score when no defect prototypes are loaded.
- [ ] Reuse raw/L2 PCA and kNN patch scores within each prediction instead of recomputing image-level evidence.
- [ ] Replace full CPU SVD with a benchmarked covariance-eigh or randomized/GPU PCA implementation; retain the current SVD as the numerical reference.
- [ ] Keep patch features and PCA residual scoring on GPU to remove DINO -> CPU PCA -> GPU kNN transfers.
- [ ] Index spatial kNN memory by canonical position so local radius queries avoid full-bank similarity and mask construction.
- [ ] Batch DINO test-image extraction within available VRAM and profile processor/PIL preprocessing separately.
- [ ] Reduce exact metric overhead: retain pixel maps as float32 and share ranking work between Pixel AUROC and AUPR without changing reported values.
- [ ] Add stage-level timing (normal augmentation, feature extraction, PCA fit, kNN calibration, per-image scoring, metric evaluation) to validate each optimization.

- [x] Add configurable Top-K patch aggregation for the image anomaly score.
- [x] Fuse features from multiple DINOv3 layers.
- [x] Evaluate foreground/saliency suppression (removed after negative MVTec ablation).
- [x] Add bidirectional patch matching for anomalous regions.
- [x] Evaluate overlapping multi-scale crops (removed after negative MVTec ablation).
- [x] Evaluate clustered defect prototypes (removed after negative MVTec ablation).
- [x] Evaluate Gaussian smoothing (disabled by default after negative MVTec ablation); retain optional CRF refinement.

## Paper-derived next steps

- [x] Add raw/L2 dual-branch scoring: L2 for image AUROC, raw for pixel maps.
- [x] Add configurable Top-K image aggregation ratios (0.5/1/2/5%).
- [x] Compare patch-first and image-score-first PCA/kNN fusion (score-first was negative).
- [x] Add optional connected Top-K spatial-consistency weighting for image scores.
- [x] Add aspect-ratio-preserving longest-side resize with mean-color padding for ablation.
- [x] Add optional per-layer L2 normalization before DINOv3 feature fusion.
- [x] Add optional spatially constrained normal-memory kNN matching.
- [x] Add canonical coordinate alignment for rotate/flip normal augmentations before spatial kNN.
- [x] Add optional region-level Reason-and-Reject with handcrafted texture evidence.
- [x] Fuse PCA residuals with an AnomalyDINO-style normal-patch kNN memory score.
- [x] Evaluate independent per-layer PCA score fusion (removed after negative MVTec ablation).
- [x] Add training-free normal-tail-calibrated gating for PCA and kNN fusion.
- [ ] Apply INSID3 positional debiasing only to cross-image defect matching.
- [x] Add a shrinkage diagonal-Mahalanobis PCA residual ablation.
- [x] Add flip test-time augmentation with inverse-aligned anomaly-map averaging.
- [ ] Score each normal reference independently and robustly aggregate scores (deferred until the 2/4-shot extension).
- [ ] Calibrate uncertainty from agreement across multiple normal references.
- [x] Evaluate Shrinkage LDA defect typing (removed after negative MVTec ablation).
- [x] Add a class-balanced RBF-SVM defect-type classifier for Top-K anomaly patches.
