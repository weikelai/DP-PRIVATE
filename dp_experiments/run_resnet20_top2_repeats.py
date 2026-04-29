import argparse
import random
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import torch

from data_utils import extract_tensors, load_cifar10
from models import build_classifier
from synthesizers import synthesize_diffusion_dp, synthesize_dp
from trainer import make_loader_from_tensors, stratified_split_tensors, train_classifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="dataset")
    parser.add_argument("--output-root", type=str, default="dp_experiments/outputs")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--methods", nargs="*", default=["dp", "diffusion_dp"])
    parser.add_argument("--epsilons", type=float, nargs="*", default=[0.1, 0.5, 1.0])
    parser.add_argument("--per-class", type=int, nargs="*", default=[40, 80, 160])
    parser.add_argument("--epochs-list", type=int, nargs="*", default=[5, 10, 20])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--synthetic-eval-ratio", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )

    train_loader, val_loader, _ = load_cifar10(
        args.data_root, args.batch_size, args.train_ratio, args.num_workers
    )
    train_x, train_y = extract_tensors(train_loader)
    val_x, val_y = extract_tensors(val_loader)

    synth_map: Dict[str, Callable] = {
        "dp": lambda x, y, eps, k: synthesize_dp(x, y, eps, k),
        "diffusion_dp": lambda x, y, eps, k: synthesize_diffusion_dp(x, y, eps, k),
    }

    rows: List[dict] = []
    for method in args.methods:
        for epsilon in args.epsilons:
            for per_class in args.per_class:
                for epochs in args.epochs_list:
                    for repeat in range(args.repeats):
                        run_seed = args.seed + repeat + int(epsilon * 1000) + per_class + epochs
                        set_seed(run_seed)

                        synth = synth_map[method](train_x, train_y, epsilon, per_class)
                        sx, sy = synth["x"], synth["y"]
                        sx_train, sy_train, sx_eval, sy_eval = stratified_split_tensors(
                            sx,
                            sy,
                            val_ratio=args.synthetic_eval_ratio,
                            seed=run_seed,
                        )

                        synth_train_loader = make_loader_from_tensors(
                            sx_train, sy_train, batch_size=args.batch_size, shuffle=True
                        )
                        original_eval_loader = make_loader_from_tensors(
                            val_x, val_y, batch_size=args.batch_size, shuffle=False
                        )
                        synthetic_eval_loader = make_loader_from_tensors(
                            sx_eval, sy_eval, batch_size=args.batch_size, shuffle=False
                        )

                        model = build_classifier("resnet20", num_classes=10)
                        result = train_classifier(
                            model=model,
                            train_loader=synth_train_loader,
                            original_eval_loader=original_eval_loader,
                            synthetic_eval_loader=synthetic_eval_loader,
                            epochs=epochs,
                            lr=args.lr,
                            device=device,
                            weight_decay=args.weight_decay,
                            label_smoothing=args.label_smoothing,
                        )
                        rows.append(
                            {
                                "method": method,
                                "epsilon": epsilon,
                                "per_class": per_class,
                                "epochs": epochs,
                                "repeat": repeat,
                                "synthesis_seconds": synth["seconds"],
                                "train_seconds": result["train_seconds"],
                                "original_pred_seconds": result["original_pred_seconds"],
                                "synthetic_pred_seconds": result["synthetic_pred_seconds"],
                                "original_acc": result["original_acc"],
                                "synthetic_acc": result["synthetic_acc"],
                                "synthetic_images": int(sx.size(0)),
                                "synthetic_train_images": int(sx_train.size(0)),
                                "synthetic_eval_images": int(sx_eval.size(0)),
                            }
                        )
                        print(
                            f"[{method}|eps={epsilon}|k={per_class}|ep={epochs}|r={repeat}] "
                            f"acc_o={result['original_acc']:.4f} acc_s={result['synthetic_acc']:.4f}"
                        )

    detail_df = pd.DataFrame(rows)
    detail_path = out_root / "metrics_resnet20_top2_repeats_detail.csv"
    detail_df.to_csv(detail_path, index=False)

    summary_df = (
        detail_df.groupby(["method", "epsilon", "per_class", "epochs"], as_index=False)
        .agg(
            original_acc_mean=("original_acc", "mean"),
            original_acc_std=("original_acc", "std"),
            synthetic_acc_mean=("synthetic_acc", "mean"),
            synthetic_acc_std=("synthetic_acc", "std"),
            train_seconds_mean=("train_seconds", "mean"),
            train_seconds_std=("train_seconds", "std"),
            synthesis_seconds_mean=("synthesis_seconds", "mean"),
            synthesis_seconds_std=("synthesis_seconds", "std"),
            runs=("repeat", "count"),
        )
        .sort_values(["per_class", "epochs", "epsilon", "method"])
    )
    summary_path = out_root / "metrics_resnet20_top2_repeats_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"\nDetail metrics saved to: {detail_path}")
    print(f"Summary metrics saved to: {summary_path}")


if __name__ == "__main__":
    main()

