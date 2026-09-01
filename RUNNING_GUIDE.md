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

标准异常检测必须保持 `--defect-shots 0`。最终缺陷类型分类实验固定使用 `--normal-shots 8`，并依次测试 `--defect-shots 1/2/4/8`。

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

## 5. 批量运行 MVTec 最终 shot 组合

脚本：`scripts/evaluate_mvtec_shots.sh`

它顺序执行 8 组实验：1、2、4、8 normal-shot 均不使用缺陷图，然后固定 8 normal-shot，依次使用 1、2、4、8 defect-shot。

| Normal shots | Defect shots | 输出目录 |
|---:|---|---|
| 1 | 0 | `outputs/mvtec-normal-1shot-defect-0shot` |
| 2 | 0 | `outputs/mvtec-normal-2shot-defect-0shot` |
| 4 | 0 | `outputs/mvtec-normal-4shot-defect-0shot` |
| 8 | 0 | `outputs/mvtec-normal-8shot-defect-0shot` |
| 8 | 1 | `outputs/mvtec-normal-8shot-defect-1shot` |
| 8 | 2 | `outputs/mvtec-normal-8shot-defect-2shot` |
| 8 | 4 | `outputs/mvtec-normal-8shot-defect-4shot` |
| 8 | 8 | `outputs/mvtec-normal-8shot-defect-8shot` |

使用脚本默认路径运行：

```bash
bash scripts/evaluate_mvtec_shots.sh
```

覆盖数据集和模型路径后运行：

```bash
DATA_ROOT=/mnt/sda1/mvtec_anomaly \
MODEL=/mnt/sda1/DINOv3/dinov3-vitl16-pretrain-lvd1689m \
bash scripts/evaluate_mvtec_shots.sh
```

脚本默认使用 `SKIP_COMPLETED=1`。已有 `results.json` 的完整组合会直接跳过；组合中断时，已完整写入 `categories/<category>.json` 的类别也会跳过，只继续未完成的类别。使用 `SKIP_COMPLETED=0` 可强制全部重跑。

## 6. 批量运行 VisA 最终 shot 组合

脚本：`scripts/evaluate_visa_shots.sh`

它使用 VisA 已验证的最佳配置运行与 MVTec 相同的 8 组组合：

| Normal shots | Defect shots | 输出目录 |
|---:|---:|---|
| 1 | 0 | `outputs/visa-normal-1shot-defect-0shot` |
| 2 | 0 | `outputs/visa-normal-2shot-defect-0shot` |
| 4 | 0 | `outputs/visa-normal-4shot-defect-0shot` |
| 8 | 0 | `outputs/visa-normal-8shot-defect-0shot` |
| 8 | 1 | `outputs/visa-normal-8shot-defect-1shot` |
| 8 | 2 | `outputs/visa-normal-8shot-defect-2shot` |
| 8 | 4 | `outputs/visa-normal-8shot-defect-4shot` |
| 8 | 8 | `outputs/visa-normal-8shot-defect-8shot` |

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

默认会跳过已完成的组合和类别；使用 `SKIP_COMPLETED=0` 可强制全部重跑。分离存放的官方 split CSV 可通过 `SPLIT_CSV=/path/to/1cls.csv` 指定。

两个批量脚本默认使用 `NORMAL_FIT_MAX_PATCHES=50000`，用于限制 8-shot 多增强、多分支拟合时的主机内存峰值。如果进程仍被 OOM killer 以状态码 137 结束，脚本会自动从头以 `OOM_RETRY_FIT_MAX_PATCHES=30000` 重试当前组合。两个上限都可通过同名环境变量覆盖。

VisA `8 normal-shot + 8 defect-shot` 的低表现类别可通过同一脚本单独调优，不会覆盖正式 shot 结果：

```bash
MODE=tune bash scripts/evaluate_visa_shots.sh
```

`TUNING_FAMILY=detection` 只测试 `candle/capsules/pipe_fryum` 的增强阈值校准，以 `balanced_accuracy` 选型；`TUNING_FAMILY=typing` 只测试 `pcb3/fryum/pcb2/pcb1/capsules` 的 patch 匹配方案，以 `defect_type_macro_f1` 选型。输出位于 `outputs/visa-8shot-8defect-tuning/`，已完成方案默认跳过。

运行完成后汇总调优结果：

```bash
python -m defectfusion.aggregate \
  --input outputs/visa-8shot-8defect-tuning \
  --output outputs/visa-8shot-8defect-tuning-summary
```

## 7. Shot 参数含义

| 参数 | 含义 |
|---|---|
| `--normal-shots 1/2/4/8` | 每个类别从正常训练集采样 1、2、4 或 8 张参考图 |
| `--defect-shots 0` | 标准异常检测，不使用任何带标签缺陷样本 |
| `--defect-shots 1/2/4/8` | 在 8 normal-shot 实验中，每种缺陷类型采样 1、2、4 或 8 张带标签样本用于缺陷分类 |
| `--seed 42` | 固定 normal/defect shot 抽样，保证可复现 |

加入 defect shots 只应影响缺陷类型分类分支，不应改变 Image AUROC、Pixel AUROC、PRO 等异常检测指标。被选为 defect prototype 的样本仍参与 image/pixel 检测指标，但会从缺陷类型分类指标中排除。

`2/4/8 defect-shot` 会在已采样的 defect prototypes 内执行类内留一校准：每次留出一张缺陷图，用其余 prototype 预测该图，再以 LOO macro-F1 选择 unknown threshold。`1 defect-shot` 无法留一，因此保持固定阈值。校准不使用剩余测试图，校准方式、样本数、LOO macro-F1 和阈值会写入每个类别 JSON 及 `summary.csv`。

## 8. 常见运行问题

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

8-shot 批量实验应使用仓库脚本，脚本会限制每个拟合分支使用的正常 patch 数量。如果仍然发生 OOM，请优先将 `NORMAL_FIT_MAX_PATCHES` 从 50000 降为 30000。

### 查看完整命令帮助

```bash
python -m defectfusion.cli evaluate-mvtec --help
python -m defectfusion.cli evaluate-visa --help
```
