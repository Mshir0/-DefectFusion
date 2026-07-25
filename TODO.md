# Improvement roadmap

- [x] Aggregate the image anomaly score from the highest-scoring 1% of patches.
- [x] Fuse features from multiple DINOv3 layers.
- [ ] Suppress background patches using foreground/saliency estimates.
- [ ] Add bidirectional patch matching for anomalous regions.
- [ ] Add overlapping multi-scale crop inference.
- [ ] Cluster anomalous patches into multiple defect prototypes.
- [ ] Add Gaussian smoothing and optional CRF map refinement.
