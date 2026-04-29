# DP-private 实验骨架

这个目录实现了你图片中提出的核心流程骨架：

- 企业本地数据先做脱敏，再上传第三方训练。
- 脱敏方法对比：`dp`、`gan_dp`、`distill_dp`、`diffusion_dp`。
- 下游模型：`cnn`、`resnet18`。
- 记录指标：合成耗时、训练耗时、预测耗时。
- 可视化：每种方法在不同隐私预算下，抽取与原始图最接近的样本做对比。

## 目录

- `run_pipeline.py`: 主入口，串联合成、训练、评估和可视化。
- `config.py`: 实验配置（数据集、epsilon、样本数等）。
- `data_utils.py`: 数据读取与拆分。
- `models.py`: CNN / ResNet18 分类模型。
- `privacy.py`: DP 裁剪与噪声工具。
- `synthesizers.py`: 4类脱敏数据生成器。
- `trainer.py`: 分类训练与评估。

## 快速开始

1. 安装依赖：

```bash
pip install torch torchvision numpy matplotlib pandas
```

2. 运行默认实验（CIFAR10）：

```bash
python dp_experiments/run_pipeline.py
```

3. 结果输出目录：

- `dp_experiments/outputs/metrics.csv`
- `dp_experiments/outputs/visuals/*.png`

## 说明

- 当前版本是“可运行骨架 + 基线算法”，用于先完成流程验证和自动化记录。
- 若你确认后，我可以继续把 `gan_dp`、`distill_dp`、`diffusion_dp` 替换成更标准的论文级实现（例如 Opacus + DDPM + 梯度匹配蒸馏）。
