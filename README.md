# DP-PRIVATE

DP-PRIVATE 是一个围绕“隐私保护数据外包训练”场景搭建的实验项目。项目模拟企业在本地对图像数据进行脱敏或合成，再将脱敏后的数据交给第三方训练下游分类模型的流程，用于比较不同隐私保护技术路线在模型可用性、隐私预算、合成样本规模和训练成本之间的权衡。

当前实验主要覆盖 MNIST 和 CIFAR10 两个数据集，比较 `DP`、`GAN+DP`、`Dataset Distillation+DP`、`Diffusion+DP` 四类脱敏/合成方法，并使用 `CNN`、`ResNet18`、`ResNet20` 作为下游分类模型进行评估。

## 项目目标

- 在原始数据不直接外包的前提下，生成可用于下游训练的隐私保护合成数据。
- 比较不同脱敏方法在原始验证集上的模型准确率表现。
- 研究隐私预算 `epsilon`、每类合成样本数 `k`、训练轮数和下游模型结构对结果的影响。
- 输出可复现实验指标、可视化图表、统计结果和论文/报告材料。

## 核心实验设计

项目中的主要评估指标包括：

- `original_acc`：下游分类器在原始验证集上的准确率，是判断方法可用性的主要指标。
- `synthetic_acc`：下游分类器在合成数据验证集上的准确率，用于观察合成域内泛化表现。
- `generalization_gap`：`synthetic_acc - original_acc`，反映合成域与原始域之间的性能差距。
- `synthesis_seconds`：生成合成数据的耗时。
- `train_seconds`：下游分类器训练耗时。
- `original_pred_seconds` / `synthetic_pred_seconds`：在原始验证集和合成验证集上的预测耗时。

后续版本已将合成数据拆分为训练集和验证集，避免使用同一批合成样本同时训练和测试导致 `synthetic_acc` 偏高。

## 目录结构

```text
.
├── README.md
├── .gitattributes
├── dataset/                         # 本地数据集缓存
├── mnist/                           # MNIST 数据缓存
├── paper/                           # 论文、报告和演示材料
├── tmp/                             # 临时生成文件
├── dp_experiments/
│   ├── config.py                    # 默认实验配置
│   ├── data_utils.py                # 数据集加载与张量提取
│   ├── models.py                    # CNN / ResNet20 分类模型
│   ├── privacy.py                   # DP 裁剪与加噪工具
│   ├── synthesizers.py              # DP、GAN+DP、Distill+DP、Diffusion+DP 合成器
│   ├── trainer.py                   # 下游分类器训练与评估
│   ├── run_pipeline.py              # 主实验入口
│   ├── run_resnet20_top2_repeats.py # 第一轮 top2 方法重复实验
│   ├── summarize_next_priority_results.py
│   ├── generate_paper_outputs.py    # 生成论文图表与统计结果
│   ├── generate_*report*.py         # 生成实验报告和 Word 文档
│   ├── outputs/                     # 当前实验输出
│   └── archived_outputs/            # 历史实验输出归档
└── ResNet18_Cifar10_95.46           # CIFAR10 相关参考资源
```

## 环境依赖

建议使用 Python 3.10 或以上版本，并安装以下依赖：

```bash
pip install torch torchvision numpy pandas matplotlib seaborn scipy scikit-image python-docx
```

如果需要使用 GPU，请先安装与本机 CUDA 版本匹配的 PyTorch。

项目中的 CIFAR10 压缩包通过 Git LFS 管理。首次克隆后建议执行：

```bash
git lfs install
git lfs pull
```

## 快速开始

运行默认实验：

```bash
python dp_experiments/run_pipeline.py
```

默认配置位于 `dp_experiments/config.py`。默认数据集为 CIFAR10，输出目录为 `dp_experiments/outputs`。

常用参数示例：

```bash
python dp_experiments/run_pipeline.py ^
  --dataset cifar10 ^
  --data-root dataset ^
  --output-root dp_experiments/outputs/example_run ^
  --methods dp diffusion_dp ^
  --classifiers cnn resnet20 ^
  --epsilons 0.5 1.0 ^
  --per-class 160 320 ^
  --epochs 10 ^
  --synthetic-eval-ratio 0.2
```

MNIST 示例：

```bash
python dp_experiments/run_pipeline.py ^
  --dataset mnist ^
  --output-root dp_experiments/outputs/mnist_example ^
  --methods diffusion_dp ^
  --classifiers cnn resnet20 ^
  --epsilons 0.3 0.5 0.7 ^
  --per-class 640 ^
  --epochs 10
```

在 PowerShell 中可以使用反引号换行，也可以将命令写成单行运行。

## 主要实验产物

常见输出文件包括：

- `metrics.csv`：每组实验的核心指标。
- `visuals/<dataset>/*.png`：合成样本与最近邻原始样本对比图。
- `figures/*.png` / `figures/*.pdf`：论文图表。
- `statistics/*.csv`：统计分析结果。
- `report/*.docx` / `report/*.md`：实验报告和论文草稿材料。

第四轮 Diffusion+DP 扩展实验的主要目录：

```text
dp_experiments/outputs/next_priority_experiments/
├── cifar10_diffusion_sweep/metrics.csv
├── mnist_diffusion_sweep/metrics.csv
├── repeats/
├── repeats_detail.csv
├── repeats_summary.csv
└── report/
```

完整方法矩阵和跨数据集结果位于：

```text
dp_experiments/outputs/complete_original_models/
├── cifar10/metrics.csv
├── mnist/metrics.csv
└── 完整实验结果说明.md
```

## 复现实验命令

完整 MNIST / CIFAR10 方法矩阵示例：

```bash
python dp_experiments/run_pipeline.py ^
  --dataset mnist ^
  --output-root dp_experiments/outputs/complete_original_models/mnist ^
  --methods dp gan_dp distill_dp diffusion_dp ^
  --classifiers cnn resnet20 ^
  --epsilons 0.1 0.5 1.0 ^
  --per-class 160 320 ^
  --epochs 5 ^
  --synthetic-eval-ratio 0.2
```

```bash
python dp_experiments/run_pipeline.py ^
  --dataset cifar10 ^
  --data-root dp_experiments/runtime_data/cifar10 ^
  --output-root dp_experiments/outputs/complete_original_models/cifar10 ^
  --methods dp gan_dp distill_dp diffusion_dp ^
  --classifiers cnn resnet20 ^
  --epsilons 0.1 0.5 1.0 ^
  --per-class 160 320 ^
  --epochs 5 ^
  --synthetic-eval-ratio 0.2
```

Diffusion+DP 扩展实验示例：

```bash
python dp_experiments/run_pipeline.py ^
  --dataset cifar10 ^
  --data-root dp_experiments/runtime_data/cifar10 ^
  --output-root dp_experiments/outputs/next_priority_experiments/cifar10_diffusion_sweep ^
  --methods diffusion_dp ^
  --classifiers cnn resnet20 ^
  --epsilons 0.7 1.0 1.5 2.0 ^
  --per-class 640 1000 ^
  --epochs 10 ^
  --synthetic-eval-ratio 0.2
```

```bash
python dp_experiments/run_pipeline.py ^
  --dataset mnist ^
  --output-root dp_experiments/outputs/next_priority_experiments/mnist_diffusion_sweep ^
  --methods diffusion_dp ^
  --classifiers cnn resnet20 ^
  --epsilons 0.3 0.5 0.7 ^
  --per-class 640 ^
  --epochs 10 ^
  --synthetic-eval-ratio 0.2
```

汇总第四轮结果：

```bash
python dp_experiments/summarize_next_priority_results.py
```

生成论文图表：

```bash
python dp_experiments/generate_paper_outputs.py
```

## 当前结论概览

根据已有实验记录，项目目前得到的主要经验结论包括：

- 在重复实验中，`Diffusion+DP` 的 `original_acc` 整体排名靠前，说明其优势不是单次随机波动造成的。
- 合成样本数 `k` 对 CIFAR10 尤其关键；从较小 `k` 扩展到 `k=1000` 后，模型在原始验证集上的准确率有明显提升。
- MNIST 结构简单，增加样本数和训练轮数后可以较稳定达到高准确率。
- `synthetic_acc` 不能单独作为方法优劣依据，最终仍应以 `original_acc` 为主要判断指标。
- 在当前合成数据质量下，复杂模型并不总是天然优于 CNN；ResNet20 的优势需要足够的数据规模和数据质量支撑。


