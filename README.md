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
python examples/generate_data.py
python -m defectfusion.cli fit --config configs/example.json
python -m defectfusion.cli predict --model-state outputs/example-model.json --image examples/data/prototypes/scratch/scratch_0.png
```

`fit` accepts `--normal-dir` and an optional `--prototype-dir`; each prototype
subdirectory becomes a defect label. Use `--device cuda` when available,
`--unknown-threshold` to control the `unknown` decision, and `--output` to
write prediction JSON. All options can be placed in a JSON file (see
`configs/example.json`), with command-line flags taking precedence.

The default classifier reports `unknown` when no prototype is sufficiently
confident. Model weights are downloaded by HuggingFace/torch at runtime.
