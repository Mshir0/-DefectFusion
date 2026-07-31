# Improvement roadmap

## Performance remediation

- [x] Skip defect-type Top-K selection and the duplicate raw-PCA score when no defect prototypes are loaded.
- [x] Reuse raw/L2 PCA and kNN patch scores within each prediction instead of recomputing image-level evidence.
- [ ] Replace full CPU SVD with a benchmarked covariance-eigh or randomized/GPU PCA implementation; retain the current SVD as the numerical reference.
- [ ] Keep patch features and PCA residual scoring on GPU to remove DINO -> CPU PCA -> GPU kNN transfers.
- [ ] Index spatial kNN memory by canonical position so local radius queries avoid full-bank similarity and mask construction.
- [ ] Batch DINO test-image extraction within available VRAM and profile processor/PIL preprocessing separately.
- [x] Reduce exact metric overhead: retain pixel maps as float32 and share ranking work between Pixel AUROC and AUPR without changing reported values.
- [ ] Add stage-level timing (normal augmentation, feature extraction, PCA fit, kNN calibration, per-image scoring, metric evaluation) to validate each optimization.

- [x] Add configurable Top-K patch aggregation for the image anomaly score.
- [x] Fuse features from multiple DINOv3 layers.
- [x] Evaluate foreground/saliency suppression (removed after negative MVTec ablation).
- [x] Add bidirectional patch matching for anomalous regions.
- [x] Evaluate overlapping multi-scale crops (removed after negative MVTec ablation).
- [x] Evaluate clustered defect prototypes (removed after negative MVTec ablation).
- [x] Evaluate Gaussian smoothing (disabled by default after negative MVTec ablation); retain optional CRF refinement.

## Paper-derived next steps

### ANoCo-inspired follow-ups

#### Pixel optimization recovery plan

Run these items in order. Do not start the next numbered item until the
current item has a recorded command, commit, per-category CSV, macro metrics,
and an accept/reject decision. Every output directory must follow
`outputs/<dataset>-<improvement-name>`.

Reference results on VisA 1-shot, seed 42, feature layers `1,17,21,23`:

| Run | I-AUROC | I-AUPR | P-AUROC | P-AUPR | PRO |
|---|---:|---:|---:|---:|---:|
| Restored PCA+kNN pixel baseline | 93.81 | 93.19 | 98.47 | 44.81 | 93.88 |
| Rejected PCA+ANoCo pixel change | 93.50 | 93.02 | 98.05 | 43.13 | 92.35 |
| ANoCo paper target | 92.70 | 93.30 | 98.70 | - | 94.90 |

Merge gates for a pixel-level change:

- [ ] Preserve I-AUROC within 0.10 points of the rerun baseline.
- [ ] Improve both macro P-AUROC and macro PRO; do not trade one for the other.
- [ ] Do not reduce any category PRO by more than 1.00 point without a documented category-specific reason.
- [ ] Validate the winning setting on five seeds, then on 1/2/4-shot VisA and MVTec-AD.

- [x] R0. Revert the failed combined change to raw cosine weights, strict minimum anchor ranking, norm compatibility, and PCA+ANoCo pixel replacement. Keep feature layers `1,17,21,23`.
- [x] R1. Record the failed result and affected categories. The largest PRO regressions were macaroni2, fryum, pcb4, and macaroni1.
- [x] P0. Add image F1-MAX and pixel F1-MAX to evaluation output, CSV reporting, and metric tests so results are directly comparable with ANoCo Table 1.
- [x] P1. Re-run the restored `pca_knn_anoco` baseline twice on the current commit with identical category overrides. Require metric differences below 0.05 points before algorithm ablations.
- [x] P2. Add an experimental ANoCo affinity mode without changing the default. Compare current softmax weights with non-negative cosine weights normalized to sum to one. Keep mean anchor ranking, norm compatibility off, and the pixel branch fixed to PCA+kNN. Rejected normalized cosine: I-AUROC/I-AUPR/I-F1-MAX were `93.6829/93.0851/90.0937`, changes of `-0.1230/-0.1086/-0.1931` points from the reproducible softmax baseline. All pixel metrics were exactly unchanged, confirming branch isolation. Softmax remains the P3 winner. Command, macro decision, and per-category CSV are recorded in `experiments/visa-anoco-cosine-affinity.md` and `.csv`.
- [x] P3. Sweep `anoco_query_weight` over `0.5, 1.0, 2.0, 4.0` only for the winning normalized affinity from P2. Selected `2.0`: versus the `1.0` baseline it changed I-AUROC/I-AUPR/I-F1-MAX by `-0.0039/+0.0428/+0.0900` points while all pixel metrics remained identical. Weight `4.0` was dominated by `2.0` on all three image metrics. Commands and per-category CSVs are recorded under `experiments/visa-anoco-query-weight-*`.
- [x] P4. Compare anchor retrieval scoring while keeping P2-P3 fixed: current mean score versus strict minimum score. Rejected minimum: versus mean it reduced I-AUROC/I-AUPR/I-F1-MAX by `0.2022/0.2258/0.1979` points while pixel metrics remained identical. Retain mean for P5. Command and per-category CSV are recorded in `experiments/visa-anoco-anchor-ranking-minimum.md` and `.csv`.
- [ ] P5. Test norm compatibility only after P4. Multiply the winning affinity by `min(norm_q, norm_r) / max(norm_q, norm_r)`, then renormalize each query's edge weights to sum to one. Never reuse the rejected unnormalized degree formulation.
- [ ] P6. Sweep `anoco_neighbors` over `4, 8, 16, 32` with all earlier choices frozen. Report latency and memory together with metrics.
- [ ] P7. Add ANoCo to the pixel branch as residual evidence instead of replacing PCA+kNN. Start from the restored PCA+kNN score and sweep an independent pixel ANoCo weight over `0.10, 0.25, 0.50`; keep the image-head ANoCo weight unchanged.
- [ ] P8. Inspect per-category maps for macaroni2, fryum, pcb4, and macaroni1. Only after a global P7 winner exists, test a normal-only uncertainty gate that falls back to PCA+kNN when ANoCo and kNN disagree.
- [ ] P9. Promote a setting only after it passes all merge gates and the five-seed, 1/2/4-shot, cross-dataset validation. Update README defaults only at this point.

- [x] Remove soft anchor, adaptive lambda, calibrated drift, exact ANoCo, disagreement gating, view balancing, and active graph after neutral or negative ablations.
- [x] Add optional median consensus across independently calibrated per-layer ANoCo drift.

- [x] Add raw/L2 dual-branch scoring: L2 for image AUROC, raw for pixel maps.
- [x] Add configurable Top-K image aggregation ratios (0.5/1/2/5%).
- [x] Compare patch-first and image-score-first PCA/kNN fusion (score-first was negative).
- [x] Add optional connected Top-K spatial-consistency weighting for image scores.
- [x] Add aspect-ratio-preserving longest-side resize with mean-color padding for ablation.
- [x] Add optional per-layer L2 normalization before DINOv3 feature fusion.
- [x] Add optional spatially constrained normal-memory kNN matching.
- [x] Add canonical coordinate alignment for rotate/flip normal augmentations before spatial kNN.
- [x] Fuse PCA residuals with an AnomalyDINO-style normal-patch kNN memory score.
- [x] Add an optional ANoCo-inspired anchor-consistent manifold-drift head and PCA/ANoCo fusion.
- [x] Add a dual head: raw PCA+kNN for localization and L2 PCA+ANoCo for image detection.
- [x] Evaluate independent per-layer PCA score fusion (removed after negative MVTec ablation).
- [x] Add training-free normal-tail-calibrated gating for PCA and kNN fusion.
- [ ] Apply INSID3 positional debiasing only to cross-image defect matching.
- [x] Add a shrinkage diagonal-Mahalanobis PCA residual ablation.
- [x] Add flip test-time augmentation with inverse-aligned anomaly-map averaging.
- [ ] Score each normal reference independently and robustly aggregate scores (deferred until the 2/4-shot extension).
- [ ] Calibrate uncertainty from agreement across multiple normal references.
- [x] Evaluate Shrinkage LDA defect typing (removed after negative MVTec ablation).
- [x] Add a class-balanced RBF-SVM defect-type classifier for Top-K anomaly patches.
