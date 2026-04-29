# 实验进度记录（按 `paper/实验内容.png`）

## 当前目标

按任务图完成以下实验：

1. 脱敏方法对比：`DP`、`GAN+DP`、`Dataset Distillation+DP`、`Diffusion+DP`。
2. 隐私预算对比：`epsilon = 0.1, 0.5, 1.0`。
3. 下游模型：`CNN`、`ResNet20`。
4. 记录耗时：合成耗时、训练耗时、预测耗时。
5. 记录精度：分别在**合成数据集**与**原始验证集**上评估。
6. 视觉对比：每种方法每个预算保存“原图最近邻 vs 合成图”。

## 已完成（你此前 + 本轮之前）

- 已完成 `k=40` 的初步网格实验，并得到：
  - `outputs/metrics_cnn_all.csv`
  - `outputs/metrics_resnet20_all.csv`
- 已保存可视化图片到：
  - `outputs/visuals/*.png`
- 已能在 `conda` 环境 `learn_torch` 内完整运行流程。

## 本轮代码增强（刚完成）

- 更新 `trainer.py`：
  - 增加双评估输出：`original_acc`、`synthetic_acc`。
  - 分别记录预测耗时：`original_pred_seconds`、`synthetic_pred_seconds`。
- 更新 `run_pipeline.py`：
  - 训练后同时评估原始验证集与合成集。
  - 结果 CSV 字段扩展为：
    - `synthesis_seconds`
    - `train_seconds`
    - `original_pred_seconds`
    - `synthetic_pred_seconds`
    - `original_acc`
    - `synthetic_acc`

## 本轮已完成（2026-04-21）

- 已完成全量矩阵（与任务图一致）：
  - 方法：`dp, gan_dp, distill_dp, diffusion_dp`
  - 预算：`0.1, 0.5, 1.0`
  - 图片规模：`k=40, 80`（每类图像数）
  - 分类器：`cnn, resnet20`
  - 训练轮次：`epochs=2`
- 已完成输出：
  - 完整指标：`outputs/metrics_task_full.csv`
  - 当前指标快照：`outputs/metrics.csv`
  - 可视化对比图：`outputs/visuals/*.png`

## 结果摘要（本轮）

- 最优原始验证精度（`original_acc`）：
  - `diffusion_dp + epsilon=0.5 + k=80 + cnn`
  - `original_acc=0.2940`
- 最优合成集精度（`synthetic_acc`）：
  - `distill_dp + epsilon=0.5 + k=80 + cnn`
  - `synthetic_acc=1.0000`
- 最快生成耗时：
  - `dp + epsilon=0.5 + k=40`
  - `synthesis_seconds=0.0248s`

## 观察与结论

- `CNN` 在本轮小数据合成设置下显著优于 `ResNet20`（原始集精度更高）。
- `GAN+DP` 生成耗时远高于其他方法，且精度优势不明显。
- `Distillation+DP` 在“合成集自测”上很高，存在过拟合迹象，需要以原始集精度为主判断可用性。
- `Diffusion+DP` 在原始集精度上表现最稳健（本轮最佳）。

## 追加实验（2026-04-21，按你的最新指令）

### 已完成项

- [x] 将 `k` 扩展到 `160`，完成 `k=40/80/160` 三档。
- [x] 将训练轮次扩展到 `epochs=5/10/20`，聚焦 `ResNet20` 排序验证。
- [x] 固定 top2 方法 `dp`、`diffusion_dp`，每组做 `3` 次重复并统计均值与方差。

### 运行设置

- 方法：`dp`, `diffusion_dp`
- 预算：`epsilon=0.1, 0.5, 1.0`
- 图片规模：`k=40, 80, 160`
- 轮次：`epochs=5, 10, 20`
- 重复：`repeats=3`
- 总训练任务：`2 * 3 * 3 * 3 = 54` 组（每组为 ResNet20 训练）

### 新增产物

- 分轮原始结果（9份）：
  - `outputs/repeats/metrics_resnet20_top2_ep5_rep1.csv`
  - `outputs/repeats/metrics_resnet20_top2_ep5_rep2.csv`
  - `outputs/repeats/metrics_resnet20_top2_ep5_rep3.csv`
  - `outputs/repeats/metrics_resnet20_top2_ep10_rep1.csv`
  - `outputs/repeats/metrics_resnet20_top2_ep10_rep2.csv`
  - `outputs/repeats/metrics_resnet20_top2_ep10_rep3.csv`
  - `outputs/repeats/metrics_resnet20_top2_ep20_rep1.csv`
  - `outputs/repeats/metrics_resnet20_top2_ep20_rep2.csv`
  - `outputs/repeats/metrics_resnet20_top2_ep20_rep3.csv`
- 合并明细：
  - `outputs/metrics_resnet20_top2_repeats_detail.csv`
- 均值/方差汇总：
  - `outputs/metrics_resnet20_top2_repeats_summary.csv`

### 排序结论（按 `original_acc_mean`）

- 全局最佳：
  - `diffusion_dp + epsilon=0.5 + k=160 + epochs=20`
  - `original_acc_mean=0.3910`, `original_acc_std=0.0271`
- 分 epoch 最优：
  - `epochs=5`: `diffusion_dp + eps=0.5 + k=160`，`0.3313 ± 0.0380`
  - `epochs=10`: `diffusion_dp + eps=1.0 + k=160`，`0.3503 ± 0.0227`
  - `epochs=20`: `diffusion_dp + eps=0.5 + k=160`，`0.3910 ± 0.0271`

### 本轮观察

- 在 `ResNet20` 上，随着 `k` 与 `epochs` 提升，`k=160` 配置明显优于 `k=40/80`。
- `diffusion_dp` 在中高训练轮次下整体优于 `dp`（以原始集精度均值为准）。
- `dp` 在合成集精度上常接近饱和，说明“合成集自测”不应单独作为模型可用性指标，仍应以原始验证集排序为主。

## 论文产物自动化（2026-04-21，新增）

### 已完成任务

- [x] 自动生成论文图表（`matplotlib + seaborn`，`dpi=400`，`png+pdf` 双格式）。
- [x] 完成 `diffusion_dp vs dp` 的分组 t-test 显著性检验。
- [x] 自动生成论文风格中文结论文档。
- [x] 基于 `outputs/visuals/` 计算 `SSIM/PSNR` 并输出汇总与图表。

### 脚本与输入

- 自动化脚本：
  - `dp_experiments/generate_paper_outputs.py`
- 主要输入：
  - `outputs/metrics_task_full.csv`
  - `outputs/metrics_resnet20_top2_repeats_summary.csv`
  - `outputs/metrics_resnet20_top2_repeats_detail.csv`
  - `outputs/visuals/*.png`

### 图表输出（`outputs/figures/`）

- `fig1_method_epsilon_original_acc_bar`（各方法在不同 epsilon 下 original_acc 柱状图）
- `fig2_k_impact_accuracy_line`（k=40/80/160 对 accuracy 影响折线图）
- `fig3_epochs_impact_accuracy_line`（epochs=5/10/20 对 accuracy 影响图）
- `fig4_synthesis_train_time_comparison`（synthesis_seconds 与 train_seconds 时间对比图）
- `fig5_epsilon_privacy_tradeoff`（epsilon 与 accuracy 隐私权衡图）
- `fig6_image_quality_ssim_bar`（SSIM 图）
- `fig7_image_quality_psnr_bar`（PSNR 图）

### 统计检验输出（`outputs/statistics/`）

- 文件：
  - `ttest_diffusion_vs_dp_by_group.csv`
- 字段：
  - `p_value`
  - `significant_better_than_dp`
  - `mean_diff`
  - `t_stat`
- 结果概览：
  - 分组总数：`27`
  - `diffusion_dp` 显著优于 `dp` 的分组数：`8`
  - 显著占比：`29.63%`

### 结论文本输出（`outputs/report/`）

- 文件：
  - `实验结论.md`
- 内容覆盖：
  - 最佳组合
  - epsilon / k / epochs 趋势
  - diffusion_dp 优势解释
  - dp 优缺点分析

### 图像质量输出

- 汇总：
  - `outputs/metrics_image_quality.csv`
- 明细：
  - `outputs/metrics_image_quality_detail.csv`
- 结论摘要（按 epsilon 汇总）：
  - `diffusion_dp` 在 `epsilon=0.5/1.0` 上 SSIM 较高且稳定；
  - `dp` 在 `epsilon=1.0` 上 SSIM 接近 diffusion；
  - `distill_dp` 的 PSNR 较高，但需结合下游精度评估其实际可用性。

## 1）4类、12个合成数据集在哪里？

### 4类方法

- `dp`
- `gan_dp`
- `distill_dp`
- `diffusion_dp`

### 12组（= 4类 × 3个epsilon）

- `epsilon = 0.1 / 0.5 / 1.0`

### 存放位置（可视化体现）

- `dp_experiments/outputs/visuals/`
- 命名规则：`{method}_eps_{epsilon}_k_{per_class}.png`
- 例如：`diffusion_dp_eps_0.1_k_80.png`

这就是每个方法 + 预算组合的“合成数据视觉对比产物”。  
当存在多个 `k`（`40/80/160`）时，会形成多套 12 组图。

## 2）“对于每一类进行如下记录”分别在哪里？

你要的 4 项性能记录都在 CSV 中：

1. **GAN+DP / Distillation+DP / Diffusion+DP 生成时间**
   - 文件：`dp_experiments/outputs/metrics_task_full.csv`
   - 字段：`synthesis_seconds`
   - 过滤：`method in {gan_dp, distill_dp, diffusion_dp}`

2. **不同隐私预算下、随图片数量变化的生成时间**
   - 文件：`metrics_task_full.csv`（基础）
   - 或：`metrics_resnet20_top2_repeats_summary.csv`（重复实验均值方差）
   - 关键字段：`method, epsilon, per_class, synthesis_seconds`

3. **不同预算下 CNN / ResNet 训练时间（自变量为图片数量）**
   - 文件：`metrics_task_full.csv`
   - 字段：`train_seconds`
   - 模型字段：`classifier`（当前主要是 `cnn`、`resnet20`）

4. **不同预算下 CNN / ResNet 预测精度（合成集 + 原始集）**
   - 文件：`metrics_task_full.csv`
   - 字段：
     - 原始集精度：`original_acc`
     - 合成集精度：`synthetic_acc`

## 3）“随机选5张最近邻视觉差异”现在在哪里？

当前实现位置：

- `dp_experiments/run_pipeline.py` 的 `save_visual_compare(...)`

当前实现现状：

- 每组图里取 `n=min(8, synth_x.size(0))`（默认展示 8 对）
- 不是“随机 5 张”，而是前 `n` 张（并做最近邻匹配）
- 最近邻匹配池是 `train_x[:2000]`

## `metrics_task_full.csv` 字段含义

文件路径：

- `E:/learn_torch/DP-private/dp_experiments/outputs/metrics_task_full.csv`

字段说明：

- `method`：脱敏方法类别（`dp` / `gan_dp` / `distill_dp` / `diffusion_dp`）。
- `epsilon`：差分隐私预算参数（隐私强度控制变量，当前常用 `0.1/0.5/1.0`）。
- `per_class`：每个类别生成的合成图像数量 `k`（如 `40/80/160`）。
- `classifier`：下游分类模型名称（当前主要为 `cnn`、`resnet20`）。
- `synthesis_seconds`：该组合下生成合成数据集耗时（秒）。
- `train_seconds`：使用该合成数据集训练分类模型的耗时（秒）。
- `original_pred_seconds`：在原始验证集上执行预测/推理耗时（秒）。
- `synthetic_pred_seconds`：在合成数据集上执行预测/推理耗时（秒）。
- `original_acc`：模型在原始验证集上的分类精度（Accuracy，范围 `[0,1]`）。
- `synthetic_acc`：模型在合成数据集上的分类精度（Accuracy，范围 `[0,1]`）。
- `synthetic_images`：合成数据总量（通常等于 `per_class * 类别数`，CIFAR10 下类别数为 `10`）。

补充说明：

- `original_acc` 更适合用于评估模型对真实分布的泛化能力。
- `synthetic_acc` 常用于评估模型对合成分布的拟合程度，可能偏高，需与 `original_acc` 结合分析。


## 视觉图与图像质量结果怎么看

对应文件：

- 视觉对比图：`E:/learn_torch/DP-private/dp_experiments/outputs/visuals/{method}_eps_{epsilon}_k_{per_class}.png`
- 汇总指标：`E:/learn_torch/DP-private/dp_experiments/outputs/metrics_image_quality.csv`
- 逐样本明细：`E:/learn_torch/DP-private/dp_experiments/outputs/metrics_image_quality_detail.csv`

### 一、如何看单张视觉对比图（`*.png`）

- 每一行是 1 组样本对比：
  - 左图：原始数据集中欧式距离最近邻（`real-nearest`）
  - 右图：对应方法生成的合成图（标题含 `method` 与 `eps`）
- 观察重点：
  - 语义是否一致（类别是否还能辨识）
  - 结构是否保留（主体轮廓、布局）
  - 噪声强度是否过高（尤其 `epsilon=0.1`）
- 当前实现是“前 `n` 张样本（最多 8 对）+ 最近邻匹配”，不是“随机 5 张”。

### 二、如何看汇总表（`metrics_image_quality.csv`）

字段：

- `method`：脱敏方法
- `epsilon`：隐私预算
- `ssim`：结构相似度（越高越好，结构更接近原图）
- `psnr`：峰值信噪比（越高越好，失真更小）

解读原则：

- 同一方法比较不同 `epsilon`：看隐私预算变化下的视觉退化/恢复趋势。
- 同一 `epsilon` 比较不同方法：看同等隐私约束下谁保留更多可视结构。
- `SSIM` 与 `PSNR` 可能不一致：一个偏结构，一个偏像素误差，需要联合判断。

### 三、如何看明细表（`metrics_image_quality_detail.csv`）

字段：

- `method, epsilon, per_class`：方法与实验条件
- `ssim, psnr`：该样本对的质量指标
- `file`：来源视觉图文件名

用途：

- 发现“均值背后”的离群点（某些样本质量特别好或特别差）
- 分析 `k=40/80/160` 是否影响视觉稳定性
- 回溯到具体 `file` 人工复核图像质量

### 四、当前结果快速结论（按汇总表）

- `epsilon=0.1`：
  - `diffusion_dp` 的 `SSIM` 最高（约 `0.444`），说明低预算下结构保留相对最好。
- `epsilon=0.5`：
  - `diffusion_dp` 与 `dp` 的 `SSIM` 均较高（约 `0.525` vs `0.493`）；
  - `distill_dp` 的 `PSNR` 最高（约 `9.465`），像素误差更小。
- `epsilon=1.0`：
  - `dp` 的 `SSIM` 略高于 `diffusion_dp`（约 `0.529` vs `0.525`）；
  - `distill_dp` 的 `PSNR` 仍最高（约 `9.526`）。
- `gan_dp` 在三档 `epsilon` 下 `SSIM/PSNR` 均整体偏低，视觉保真度相对弱。

结论建议：

- 若更关注结构语义稳定（接近下游分类可用性），优先参考 `SSIM` 与 `original_acc` 联合结论。
- 若更关注像素级还原，可参考 `PSNR`，但不能替代下游任务精度评估。

## 跨数据集一致性验证（CIFAR10 -> MNIST）

### 目标与口径

- 在与 CIFAR10 主线一致的实验口径下，验证方法排序在 MNIST 上是否保持稳定。
- 统一对比字段：`method`、`epsilon`、`per_class(k)`、`classifier`、`original_acc`、`synthetic_acc`、`generalization_gap`、时间指标。

### MNIST 实验设置（已完成）

- 数据集：`MNIST`（输入对齐为 `3x32x32` 以复用现有模型管线）。
- 方法：`dp`、`gan_dp`、`distill_dp`、`diffusion_dp`。
- 隐私预算：`epsilon=0.1/0.5/1.0`。
- 合成规模：`k=40/80/160`。
- 分类器：`cnn`、`resnet20`。
- 训练轮次：`epochs=2`。

### MNIST 核心结果

- 全局最佳组合：
  - `diffusion_dp + eps=0.5 + k=160 + cnn`
  - `original_acc=0.3684`
  - `synthetic_acc=0.4288`
- 各方法平均 `original_acc` 排序：
  - `diffusion_dp: 0.2169`
  - `dp: 0.1916`
  - `distill_dp: 0.1810`
  - `gan_dp: 0.1004`
- epsilon 趋势（平均 `original_acc`）：
  - `0.1 -> 0.1668`
  - `0.5 -> 0.1739`
  - `1.0 -> 0.1767`
- k 趋势（平均 `original_acc`）：
  - `k=40 -> 0.1500`
  - `k=80 -> 0.1678`
  - `k=160 -> 0.1996`
- 模型对比（平均 `original_acc`）：
  - `cnn: 0.2196`
  - `resnet20: 0.1254`
- 过拟合风险（`generalization_gap = synthetic_acc - original_acc`）：
  - `distill_dp: 0.3834`（最高，合成域拟合强但真实域泛化风险大）
  - `diffusion_dp: 0.0434`
  - `dp: 0.0273`
  - `gan_dp: 0.0099`

### 跨数据集一致性结论

- 与 CIFAR10 主结论一致：`diffusion_dp` 在综合可用性上仍领先。
- `k` 增大对 `original_acc` 的提升在 MNIST 上依旧显著，说明该规律具有跨数据集稳定性。
- `distill_dp` 在合成集表现与真实集表现差距较大，跨数据集均呈现过拟合倾向，评估应以 `original_acc` 为主。

### 结果文件

- MNIST 全量指标：`dp_experiments/outputs/metrics_mnist_full.csv`
- MNIST 专项说明：`dp_experiments/outputs/report/MNIST_实验结果说明.md`
