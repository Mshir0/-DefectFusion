# Shot and Distillation Benchmark

The combined benchmark runs the main DINOv3 PCA detector once at each of
1, 2, 4, and 8 normal shots on both MVTec AD and VisA. It then runs one
8-shot DINOv3 ViT-B to ViT-S LoRA distillation on both datasets and builds
reproducible result tables.

## Script Summary

| Script | Scope | Main purpose | Default output |
| --- | --- | --- | --- |
| `evaluate_pca_good_accuracy.sh` | All MVTec/VisA categories | Main PCA evaluation and normal/defect threshold metrics | `outputs/pca-good-accuracy-loo` |
| `compare_mvtec_crf.sh` | MVTec leather only | Compare raw and DenseCRF pixel maps with the same LOO threshold | `outputs/mvtec-crf-ablation-loo/leather` |
| `compare_visa_crf.sh` | One VisA category (`candle` by default) | Compare raw and DenseCRF pixel maps with the same LOO threshold | `outputs/visa-crf-ablation-loo/<category>` |
| `compare_mvtec_gaussian.sh` | One MVTec category | Compare raw and Gaussian-smoothed pixel maps | `outputs/mvtec-gaussian-ablation-loo/<category>` |
| `distill_all_mvtec_visa.sh` | All MVTec/VisA categories | Train and evaluate one LoRA-only ViT-S adapter per category | `outputs/dinov3-all-categories` |
| `run_shot_distillation_benchmark.sh` | Both complete datasets | Run main 1/2/4/8-shot experiments, one distillation, and result tables | `outputs/shot-distillation-benchmark` |
| `build_benchmark_tables.py` | Completed results under one root | Rebuild macro, category, and best-result tables without rerunning models | `<input-root>/tables` |

## Recorded Focused Results

These are the completed 8-shot, seed-42, q=0.995-higher LOO ablations already
observed before the full benchmark. They are single-category results and must
not be compared directly with full-dataset macro averages.

| Dataset/category | Selected mode | Image AUROC | Pixel AUROC | Pixel AUPR | Pixel AUPRO | Pixel F1-max | Good accuracy | Defect recall | Balanced accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MVTec leather | CRF | 100.00 | 99.27 | **57.45** | **98.61** | **57.08** | 100.00 | 100.00 | 100.00 |
| VisA candle | None | 91.89 | 99.22 | 47.42 | **96.21** | **53.93** | 88.00 | 84.00 | 86.00 |

For VisA candle, CRF raised Pixel AUPR to 50.18 but reduced Pixel AUPRO to
91.70 and Pixel F1-max to 50.36, so `none` is retained as the better overall
pixel-localization setting. The generated full benchmark tables select best
values independently for every metric and use balanced accuracy as the primary
good/anomaly threshold result.

## Experiment Matrix

| Stage | Method | Normal shots | Threshold calibration | Map post-process | Saved result |
| --- | --- | --- | --- | --- | --- |
| 1 | Main DINOv3 PCA | 1 | Held-out augmentation, q=0.995 linear | None | `main/1shot/<dataset>` |
| 2 | Main DINOv3 PCA | 2 | Source-disjoint LOO, q=0.995 higher | None | `main/2shot/<dataset>` |
| 3 | Main DINOv3 PCA | 4 | Source-disjoint LOO, q=0.995 higher | None | `main/4shot/<dataset>` |
| 4 | Main DINOv3 PCA | 8 | Source-disjoint LOO, q=0.995 higher | None | `main/8shot/<dataset>` |
| 5 | Distilled ViT-S LoRA | 8 | Source-disjoint LOO, q=0.995 higher | None | `distillation/<dataset>` |

One shot cannot use LOO because no normal source image remains to fit the
held-out fold. DenseCRF is excluded from the full benchmark because it improved
MVTec leather but reduced Pixel AUPRO and Pixel F1 on VisA candle. The focused
CRF scripts remain available as category-level ablations.

The previously completed full VisA 8-shot LOO baseline is also retained for
reference while the combined benchmark is running:

| Dataset | Method | Shots | Image AUROC | Image AUPR | Pixel AUROC | Pixel AUPR | Pixel AUPRO | Pixel F1-max | Good accuracy | Defect recall | Balanced accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| VisA | Main PCA | 8 | 93.79 | 92.96 | 98.06 | 47.64 | 92.88 | 52.08 | 96.67 | 57.33 | 77.00 |

This row documents the supplied result; the generated tables are authoritative
for the new controlled run because all shots and distillation then share the
same output root, seed, and current code revision.

## Run

Run from the repository root on Linux:

```bash
bash scripts/run_shot_distillation_benchmark.sh \
  --mvtec-root /mnt/sda1/mvtec_anomaly \
  --visa-root /mnt/sda1/VisA_20220922 \
  --base-model /mnt/sda1/DINOv3/dinov3-vitl16-pretrain-lvd1689m \
  --teacher-model /mnt/sda1/DINOv3/dinov3-vitb16-pretrain-lvd1689m \
  --student-model /mnt/sda1/DINOv3/dinov3-vits16-pretrain-lvd1689m \
  --seed 42 \
  --output-root outputs/shot-distillation-benchmark
```

For a local model argument, the script prints a `[models]` line with the
physical directory before work starts and requires `config.json` in that
directory. This distinguishes a mounted model snapshot from a parent directory
or a path that is not visible to the selected Python environment.

Resume a partially completed run without repeating final result directories:

```bash
bash scripts/run_shot_distillation_benchmark.sh \
  --mvtec-root /mnt/sda1/mvtec_anomaly \
  --visa-root /mnt/sda1/VisA_20220922 \
  --base-model /mnt/sda1/DINOv3/dinov3-vitl16-pretrain-lvd1689m \
  --teacher-model /mnt/sda1/DINOv3/dinov3-vitb16-pretrain-lvd1689m \
  --student-model /mnt/sda1/DINOv3/dinov3-vits16-pretrain-lvd1689m \
  --output-root outputs/shot-distillation-benchmark \
  --skip-completed
```

## Outputs

| File | Contents |
| --- | --- |
| `tables/experiment_results.csv` | One macro-average row for every method, dataset, and shot |
| `tables/category_results.csv` | One row per category with threshold audit fields |
| `tables/best_results.csv` | Every per-dataset metric winner, including ties |
| `tables/best_balanced_results.csv` | Best threshold configuration by balanced accuracy |
| `tables/results.md` | Human-readable tables with best values in bold |
| `logs/main-<N>shot.log` | Complete output from each main-model shot run |
| `logs/distillation-8shot.log` | Complete output from the single distillation run |

Each experiment directory also keeps the native `results.json`, `summary.csv`,
and `categories/<category>.json`. Distillation category directories contain
only `lora_adapter.pt` plus training metadata; the ViT-S base model is not
copied into the output.

Balanced accuracy is used as the primary threshold-selection result because
maximizing good accuracy alone can hide a large defect false-negative rate.
The per-metric best table is still retained for complete reporting.

Tables can be rebuilt independently after copying or adding completed result
directories:

```bash
python scripts/build_benchmark_tables.py \
  --input-root outputs/shot-distillation-benchmark \
  --output-dir outputs/shot-distillation-benchmark/tables
```
