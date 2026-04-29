# MNIST 实验结果说明（本轮）

## 一、实验配置

- 数据集：MNIST（转换为 3x32x32）
- 方法：dp, gan_dp, distill_dp, diffusion_dp
- 隐私预算：epsilon=0.1/0.5/1.0
- 合成规模：k=40/80/160
- 分类器：cnn, resnet20
- 训练轮次：epochs=2

## 二、总体效果

- 全局最佳组合：`diffusion_dp` + eps=0.5 + k=160 + cnn，original_acc=0.3684，synthetic_acc=0.4288

## 三、按方法平均 original_acc 排序

- diffusion_dp: 0.2169
- dp: 0.1916
- distill_dp: 0.1810
- gan_dp: 0.1004

## 四、隐私预算趋势（epsilon -> original_acc 均值）

- eps=0.1: 0.1668
- eps=0.5: 0.1739
- eps=1.0: 0.1767

## 五、合成规模趋势（k -> original_acc 均值）

- k=40: 0.1500
- k=80: 0.1678
- k=160: 0.1996

## 六、分类器对比（original_acc 均值）

- cnn: 0.2196
- resnet20: 0.1254

## 七、时间开销（按方法平均 synthesis_seconds）

- dp: 0.040s
- distill_dp: 0.103s
- diffusion_dp: 0.669s
- gan_dp: 8.647s

## 八、过拟合风险（synthetic_acc - original_acc 均值）

- distill_dp: 0.3834
- diffusion_dp: 0.0434
- dp: 0.0273
- gan_dp: 0.0099

解释：该值越大，表示模型在合成集上的表现显著高于原始集，潜在过拟合风险越高。
