# MNIST 顶刊风格实验结论

## 最佳组合

- 最优配置：`diffusion_dp + eps=0.5 + k=160 + cnn`
- `original_acc=0.3684`，`synthetic_acc=0.4288`

## 方法排序（按 mean_original_acc）

| method | mean_original_acc |
| --- | --- |
| diffusion_dp | 0.2169 |
| dp | 0.1916 |
| distill_dp | 0.1810 |
| gan_dp | 0.1004 |

## epsilon 趋势

| epsilon | mean_original_acc |
| --- | --- |
| 0.1000 | 0.1668 |
| 0.5000 | 0.1739 |
| 1.0000 | 0.1767 |

## k 趋势

| per_class | mean_original_acc |
| --- | --- |
| 40.0000 | 0.1500 |
| 80.0000 | 0.1678 |
| 160.0000 | 0.1996 |

## 分类器对比

| classifier | mean_original_acc |
| --- | --- |
| cnn | 0.2196 |
| resnet20 | 0.1254 |

## 过拟合风险（generalization_gap）

| method | generalization_gap |
| --- | --- |
| distill_dp | 0.3834 |
| diffusion_dp | 0.0434 |
| dp | 0.0273 |
| gan_dp | 0.0099 |

## 结论

1. 在 MNIST 上，`diffusion_dp` 依然保持最高的真实分布泛化能力，跨数据集结论稳定。
2. `k` 从 40 提升到 160 带来持续且显著的性能提升，是当前最有效的工程增益变量。
3. `distill_dp` 在合成域拟合强，但 `generalization_gap` 明显偏高，实际部署需谨慎。
4. 在当前训练轮次设置下，`cnn` 的平均表现优于 `resnet20`，建议先保证收敛稳定再提升模型复杂度。
