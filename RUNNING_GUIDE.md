# DefectFusion 运行参数与脚本指南

本文档集中说明当前推荐配置、单次评估命令、批量 shot 实验、结果汇总和单张热力图生成。以下命令面向 Linux 服务器，并假定项目依赖和 DINOv3 权重已经准备完成。

## 1. 进入项目目录

```bash
cd /path/to/-DefectFusion
```

所有命令均从仓库根目录执行。默认示例路径如下：

```text
MVTec AD: /mnt/sda1/mvtec_anomaly
VisA:     /mnt/sda1/VisA_20220922
DINOv3:   /mnt/sda1/DINOv3/dinov3-vitl16-pretrain-lvd1689m
```

输出目录统一命名为：

```text
outputs/<数据集>-<测试改进项目名称>
```

固定使用表现最好的特征层组合：

```bash
--feature-layers=1,17,21,23
```

注意：等号不能省略。尤其在使用负数层索引时，也必须写成 `--feature-layers=-1,-2,-3,-4`，否则 `argparse` 可能把负数识别为新的参数。

### 正常/异常阈值校准

主 PCA 的 2-shot 及以上实验推荐使用源图隔离的留一法：

```bash
--normal-decision-calibration leave-one-out \
--normal-decision-quantile 0.995 \
--normal-decision-quantile-method higher \
--normal-decision-augment-count 30 \
--normal-decision-fit-augment-count 4 \
--normal-decision-seed 142
```

每一折留出一张正常源图，其余源图拟合临时 PCA；同一留出源图的原图和旋转增强只贡献一个最大分数。因此 8-shot 最终有 8 个独立校准分数，而不是把 240 个相关旋转视图当作独立样本。`0.995 + higher` 在 8 个分数上取保守最大值。`summary.csv` 中查看 `good_accuracy`、`defect_recall`、`balanced_accuracy`、TN/FP/TP/FN，并用 `normal_decision_calibration`、`good_decision_quantile_method`、`normal_decision_folds` 审计阈值来源。

1-shot 无法执行源图隔离留一法，因为没有剩余源图可拟合 PCA。此时应优先提供独立的正常验证集；没有验证集时只能显式使用旧增强校准：

```bash
--normal-decision-calibration augmentation \
--normal-decision-quantile 0.995 \
--normal-decision-quantile-method linear
```

增强校准可用于 1-shot 的工程回退，但增强视图并非独立正常样本，论文中应与独立验证或 LOO 结果分开报告。

## 2. 当前推荐方法

当前推荐配置为双分支：

- Pixel 分支：PCA + kNN，用于生成 patch-wise anomaly map。
- Image 分支：PCA + 跨层 ANoCo 中位数一致性，用于图像级异常分数。
- 对应参数：`--dual-branch --anomaly-method pca_knn_anoco --anoco-layer-consensus`。
- 不推荐用 `--anomaly-method pca_anoco` 全面替代像素 kNN。已有 MVTec 1-shot 实验中，其 Pixel AUROC、Pixel AUPR 和 Pixel F1 均略有下降。

关键参数如下：

| 参数 | 当前值 | 作用 |
|---|---:|---|
| `--feature-layers` | `1,17,21,23` | 提取四个跨深度 DINOv3 hidden states |
| `--layer-aggregation` | `mean` | 聚合主像素分支的多层特征 |
| `--layer-normalization` | `none` | 保留原始层特征幅值 |
| `--dual-branch` | 开启 | 图像和像素使用不同特征分支 |
| `--anomaly-method` | `pca_knn_anoco` | Pixel 使用 PCA+kNN，Image 使用 PCA+ANoCo |
| `--knn-weight` | `0.5` | Pixel PCA/kNN 校准融合中的 kNN 权重 |
| `--anoco-neighbors` | `16` | 每个 query patch 的正常参考邻居数 |
| `--anoco-temperature` | `0.07` | ANoCo softmax 邻接权重温度 |
| `--anoco-weight` | `0.25` | Image PCA/ANoCo 校准融合中的 ANoCo 权重 |
| `--anoco-layer-consensus` | 开启 | 对 1、17、21、23 层独立计算并校准 ANoCo drift，最后逐 patch 取中位数 |
| `--image-score` | `mtop1p` | 使用最高异常 patch 的均值作为图像分数 |
| `--image-top-ratio` | `0.01` | 图像分数使用最高 1% patch |
| `--image-fusion-stage` | `patch` | 先融合 patch 证据，再计算图像分数 |
| `--knn-backend` | `torch` | 在 GPU 上执行 kNN |
| `--knn-dtype` | `float16` | 降低 kNN 显存和计算开销 |
| `--knn-spatial-radius` | `-1` | 全局搜索正常 patch |
| `--map-postprocess` | `none` | 不做额外图像后处理 |

## 3. MVTec AD 单次完整命令

以下为当前 MVTec 1 normal shot、0 defect shot 的完整全类别命令：

```bash
python -m defectfusion.cli evaluate-mvtec \
  --data-root /mnt/sda1/mvtec_anomaly \
  --model /mnt/sda1/DINOv3/dinov3-vitl16-pretrain-lvd1689m \
  --device cuda \
  --normal-shots 1 \
  --defect-shots 0 \
  --seed 42 \
  --image-size 672 \
  --pixel-image-size-override cable=896 \
  --pixel-image-size-override transistor=896 \
  --pixel-multiscale-size-override cable=672 \
  --pixel-multiscale-size-override transistor=672 \
  --pixel-multiscale-weight 0.25 \
  --resize-mode direct \
  --normal-augment-count 30 \
  --normal-augmentations rotate \
  --no-augment-categories transistor \
  --feature-layers=1,17,21,23 \
  --layer-aggregation mean \
  --layer-normalization none \
  --dual-branch \
  --anomaly-method pca_knn_anoco \
  --knn-weight 0.5 \
  --anoco-neighbors 16 \
  --anoco-query-weight 2.0 \
  --anoco-temperature 0.07 \
  --anoco-affinity softmax \
  --anoco-anchor-ranking mean \
  --anoco-weight 0.25 \
  --anoco-layer-consensus \
  --fusion-mode fixed \
  --image-score mtop1p \
  --image-top-ratio 0.01 \
  --image-min-component-size 1 \
  --image-fusion-stage patch \
  --memory-max-patches 50000 \
  --knn-chunk-size 256 \
  --knn-backend torch \
  --knn-dtype float16 \
  --knn-spatial-radius -1 \
  --map-postprocess none \
  --type-matching bidirectional_patch \
  --top-k-ratio 0.05 \
  --output outputs/mvtec-normal-1shot-defect-0shot
```

只测试部分类别时，在命令中增加例如：

```bash
--categories cable pill transistor
```

标准异常检测必须保持 `--defect-shots 0`。测试缺陷类型分类时，才将其改为 `1`、`2` 或 `4`；每种缺陷类型会分别采样对应数量的带标签缺陷图像。

## 4. VisA 单次完整命令

以下为当前 VisA 1 normal shot、0 defect shot 的完整全类别命令：

```bash
python -m defectfusion.cli evaluate-visa \
  --data-root /mnt/sda1/VisA_20220922 \
  --model /mnt/sda1/DINOv3/dinov3-vitl16-pretrain-lvd1689m \
  --device cuda \
  --normal-shots 1 \
  --defect-shots 0 \
  --seed 42 \
  --image-size 672 \
  --image-size-override macaroni2=896 \
  --image-size-override pcb2=896 \
  --image-size-override pcb3=896 \
  --pixel-image-size-override fryum=896 \
  --image-head-size-override pcb4=896 \
  --pixel-multiscale-size-override macaroni2=672 \
  --pixel-multiscale-size-override pcb2=672 \
  --pixel-multiscale-size-override pcb3=672 \
  --pixel-multiscale-weight 0.25 \
  --normal-augment-count 30 \
  --normal-augmentations rotate \
  --affine-categories macaroni1 macaroni2 \
  --feature-layers=1,17,21,23 \
  --layer-aggregation mean \
  --layer-normalization none \
  --dual-branch \
  --anomaly-method pca_knn_anoco \
  --knn-weight 0.5 \
  --anoco-neighbors 16 \
  --anoco-query-weight 1.0 \
  --anoco-temperature 0.07 \
  --anoco-weight 0.25 \
  --anoco-layer-consensus \
  --image-score mtop1p \
  --image-top-ratio 0.01 \
  --image-min-component-size 2 \
  --component-reject-categories macaroni1 macaroni2 \
  --image-fusion-stage patch \
  --memory-max-patches 50000 \
  --knn-chunk-size 256 \
  --knn-backend torch \
  --knn-dtype float16 \
  --knn-spatial-radius -1 \
  --map-postprocess none \
  --output outputs/visa-normal-1shot-defect-0shot
```

只测试单个类别时，例如：

```bash
--categories macaroni2
```

## 5. 批量运行 MVTec shot 排列组合

脚本：`scripts/evaluate_mvtec_shot_matrix.sh`

它顺序执行 13 组实验，并保证 defect shots 不超过 normal shots；full normal shot 允许所有 defect shot 设置。

| Normal shots | Defect shots | 输出目录 |
|---:|---|---|
| 1 | 0、1 | `outputs/mvtec-normal-1shot-defect-*shot` |
| 2 | 0、1、2 | `outputs/mvtec-normal-2shot-defect-*shot` |
| 4 | 0、1、2、4 | `outputs/mvtec-normal-4shot-defect-*shot` |
| full (`-1`) | 0、1、2、4 | `outputs/mvtec-normal-fullshot-defect-*shot` |

使用脚本默认路径运行：

```bash
bash scripts/evaluate_mvtec_shot_matrix.sh
```

覆盖数据集和模型路径后运行：

```bash
DATA_ROOT=/mnt/sda1/mvtec_anomaly \
MODEL=/mnt/sda1/DINOv3/dinov3-vitl16-pretrain-lvd1689m \
bash scripts/evaluate_mvtec_shot_matrix.sh
```

该脚本当前会重新执行所有组合。重新运行前应检查已有输出，避免覆盖或重复长时间实验。

## 6. 批量运行 VisA normal shots

脚本：`scripts/evaluate_visa_shots.sh`

它依次执行 1、2、4 和 full normal-shot，固定 `defect-shots=0`：

| Normal shots | Normal augmentation | 正常拟合 patch 上限 | 输出目录 |
|---:|---:|---:|---|
| 1 | 30 | 不限制 | `outputs/visa-normal-1shot` |
| 2 | 30 | 不限制 | `outputs/visa-normal-2shot` |
| 4 | 30 | 不限制 | `outputs/visa-normal-4shot` |
| full (`-1`) | 0 | 50000 | `outputs/visa-normal-fullshot` |

使用脚本默认路径运行：

```bash
bash scripts/evaluate_visa_shots.sh
```

覆盖数据集和模型路径后运行：

```bash
DATA_ROOT=/mnt/sda1/VisA_20220922 \
MODEL=/mnt/sda1/DINOv3/dinov3-vitl16-pretrain-lvd1689m \
bash scripts/evaluate_visa_shots.sh
```

脚本检测到目标目录中已经存在 `summary.csv` 时会跳过该组实验，因此中断后可以直接重新执行以继续尚未完成的组。

VisA full-shot 不再生成 900 个额外增强视图，而是使用全部正常训练图像、`normal-augment-count=0` 和 `normal-fit-max-patches=50000`。这是为了避免 Linux 显示 `Killed` 的主机内存不足问题。full-shot 与 few-shot 的结果因此不能解释为只改变了 normal shot 数量。

## 7. 汇总所有实验结果

运行：

```bash
python scripts/summarize_results.py \
  --input outputs \
  --output outputs/all-results-summary
```

生成：

```text
outputs/all-results-summary/experiment_metrics.csv
outputs/all-results-summary/category_metrics.csv
```

- `experiment_metrics.csv`：每个完整实验一行，包含宏平均指标和实验配置。
- `category_metrics.csv`：每个实验的每个类别一行，包含分类别指标、耗时、内存 patch 数量和配置。
- 没有最终 `results.json` 的未完成实验不会进入汇总。

每个评估输出目录本身包含：

```text
results.json
summary.csv
categories/<category>.json
```

其中 `summary.csv` 用于快速查看指标，类别 JSON 还保存逐图预测与 anomaly map。

## 8. 生成一张推理热力图

### 8.1 从已有评估结果生成

指定一张图像：

```bash
python scripts/render_heatmaps.py \
  --predictions outputs/mvtec-normal-1shot-defect-0shot/categories/bottle.json \
  --image /mnt/sda1/mvtec_anomaly/bottle/test/broken_large/000.png \
  --lower-percentile 1 \
  --upper-percentile 99 \
  --colormap turbo \
  --output outputs/mvtec-bottle-heatmap.png
```

不提供 `--image` 时，默认从该类别 JSON 中选择图像异常分数最高的样本：

```bash
python scripts/render_heatmaps.py \
  --predictions outputs/mvtec-normal-1shot-defect-0shot/categories/bottle.json \
  --select highest \
  --lower-percentile 1 \
  --upper-percentile 99 \
  --colormap turbo \
  --output outputs/mvtec-bottle-highest-heatmap.png
```

脚本只生成一张纯 heatmap PNG，不生成 overlay、拼图或多张图片。可选色图为 `turbo`、`magma` 和 `jet`。

### 8.2 从已保存模型直接推理

```bash
python scripts/render_heatmaps.py \
  --model-state outputs/model.json \
  --model /mnt/sda1/DINOv3/dinov3-vitl16-pretrain-lvd1689m \
  --image /path/to/test.png \
  --device cuda \
  --image-size 672 \
  --resize-mode direct \
  --feature-layers=1,17,21,23 \
  --layer-aggregation mean \
  --layer-normalization none \
  --colormap turbo \
  --output outputs/custom-single-heatmap.png
```

若保存的检测器使用 kNN，还必须让模型状态 JSON 与其旁边的 `.normal-memory.npz` 文件保持在同一目录。

## 9. Shot 参数含义

| 参数 | 含义 |
|---|---|
| `--normal-shots 1/2/4` | 每个类别从正常训练集采样 1、2 或 4 张参考图 |
| `--normal-shots -1` | 使用该类别全部正常训练图 |
| `--defect-shots 0` | 标准异常检测，不使用任何带标签缺陷样本 |
| `--defect-shots 1/2/4` | 每种缺陷类型采样 1、2 或 4 张带标签样本，用于辅助缺陷分类 |
| `--seed 42` | 固定 normal/defect shot 抽样，保证可复现 |

加入 defect shots 只应影响缺陷类型分类分支，不应改变 Image AUROC、Pixel AUROC、PRO 等异常检测指标。被选为 defect prototype 的样本仍参与 image/pixel 检测指标，但会从缺陷类型分类指标中排除。

## 10. 常见运行问题

### 参数被识别为命令

如果出现：

```text
cli.py: error: unrecognized arguments:
--layer-aggregation: command not found
```

通常是上一行末尾的反斜杠 `\` 后存在空格，导致 Linux 提前结束命令。反斜杠必须是该行最后一个字符：

```bash
--feature-layers=1,17,21,23 \
--layer-aggregation mean
```

不要把带尾随空格的命令直接粘贴到终端。

### 进程只显示 Killed

Linux 只输出 `Killed` 通常表示主机 RAM 被 OOM killer 耗尽，而不是 Python 参数错误。优先检查：

```bash
free -h
dmesg -T | tail -n 50
```

VisA full-shot 应使用仓库中的批量脚本，因为脚本已经关闭 full-shot 增强并限制每个拟合分支最多使用 50000 个正常 patch。不要对 900 张正常图像继续设置 `normal-augment-count=30`。

### 查看完整命令帮助

```bash
python -m defectfusion.cli evaluate-mvtec --help
python -m defectfusion.cli evaluate-visa --help
python scripts/summarize_results.py --help
python scripts/render_heatmaps.py --help
```
