from pathlib import Path

import pandas as pd


def main() -> None:
    df = pd.read_csv("dp_experiments/outputs/metrics_mnist_full.csv")

    best = df.loc[df["original_acc"].idxmax()]
    by_method = (
        df.groupby("method", as_index=False)["original_acc"]
        .mean()
        .sort_values("original_acc", ascending=False)
    )
    by_eps = (
        df.groupby("epsilon", as_index=False)["original_acc"]
        .mean()
        .sort_values("epsilon")
    )
    by_k = (
        df.groupby("per_class", as_index=False)["original_acc"]
        .mean()
        .sort_values("per_class")
    )
    by_clf = (
        df.groupby("classifier", as_index=False)["original_acc"]
        .mean()
        .sort_values("original_acc", ascending=False)
    )
    by_gen_time = (
        df.groupby("method", as_index=False)["synthesis_seconds"]
        .mean()
        .sort_values("synthesis_seconds")
    )

    gap_df = df.copy()
    gap_df["gap"] = gap_df["synthetic_acc"] - gap_df["original_acc"]
    by_gap = (
        gap_df.groupby("method", as_index=False)["gap"]
        .mean()
        .sort_values("gap", ascending=False)
    )

    lines: list[str] = []
    lines.append("# MNIST 实验结果说明（本轮）\n\n")
    lines.append("## 一、实验配置\n\n")
    lines.append("- 数据集：MNIST（转换为 3x32x32）\n")
    lines.append("- 方法：dp, gan_dp, distill_dp, diffusion_dp\n")
    lines.append("- 隐私预算：epsilon=0.1/0.5/1.0\n")
    lines.append("- 合成规模：k=40/80/160\n")
    lines.append("- 分类器：cnn, resnet20\n")
    lines.append("- 训练轮次：epochs=2\n\n")

    lines.append("## 二、总体效果\n\n")
    lines.append(
        f"- 全局最佳组合：`{best['method']}` + eps={best['epsilon']} + k={int(best['per_class'])} + "
        f"{best['classifier']}，original_acc={best['original_acc']:.4f}，"
        f"synthetic_acc={best['synthetic_acc']:.4f}\n\n"
    )

    lines.append("## 三、按方法平均 original_acc 排序\n\n")
    for _, row in by_method.iterrows():
        lines.append(f"- {row['method']}: {row['original_acc']:.4f}\n")
    lines.append("\n")

    lines.append("## 四、隐私预算趋势（epsilon -> original_acc 均值）\n\n")
    for _, row in by_eps.iterrows():
        lines.append(f"- eps={row['epsilon']}: {row['original_acc']:.4f}\n")
    lines.append("\n")

    lines.append("## 五、合成规模趋势（k -> original_acc 均值）\n\n")
    for _, row in by_k.iterrows():
        lines.append(f"- k={int(row['per_class'])}: {row['original_acc']:.4f}\n")
    lines.append("\n")

    lines.append("## 六、分类器对比（original_acc 均值）\n\n")
    for _, row in by_clf.iterrows():
        lines.append(f"- {row['classifier']}: {row['original_acc']:.4f}\n")
    lines.append("\n")

    lines.append("## 七、时间开销（按方法平均 synthesis_seconds）\n\n")
    for _, row in by_gen_time.iterrows():
        lines.append(f"- {row['method']}: {row['synthesis_seconds']:.3f}s\n")
    lines.append("\n")

    lines.append("## 八、过拟合风险（synthetic_acc - original_acc 均值）\n\n")
    for _, row in by_gap.iterrows():
        lines.append(f"- {row['method']}: {row['gap']:.4f}\n")
    lines.append("\n")
    lines.append("解释：该值越大，表示模型在合成集上的表现显著高于原始集，潜在过拟合风险越高。\n")

    out = Path("dp_experiments/outputs/report/MNIST_实验结果说明.md")
    out.write_text("".join(lines), encoding="utf-8")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()

