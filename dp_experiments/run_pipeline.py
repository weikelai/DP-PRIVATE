import random
import argparse
from pathlib import Path
from typing import Callable, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from config import ExperimentConfig
from data_utils import extract_tensors, load_dataset
from models import build_classifier
from synthesizers import (
    synthesize_diffusion_dp,
    synthesize_distill_dp,
    synthesize_dp,
    synthesize_gan_dp,
)
from trainer import make_loader_from_tensors, stratified_split_tensors, train_classifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _nearest_image_index(img: torch.Tensor, pool: torch.Tensor) -> int:
    d = ((pool - img.unsqueeze(0)) ** 2).flatten(1).sum(dim=1)
    return int(torch.argmin(d).item())


def save_visual_compare(
    method: str,
    epsilon: float,
    per_class: int,
    synth_x: torch.Tensor,
    real_x: torch.Tensor,
    out_dir: Path,
    sample_count: int = 5,
) -> None:
    n = min(sample_count, synth_x.size(0))
    chosen = torch.randperm(synth_x.size(0))[:n]
    fig, axes = plt.subplots(n, 2, figsize=(5, n * 2))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)
    for i in range(n):
        s = synth_x[chosen[i]]
        ridx = _nearest_image_index(s, real_x)
        r = real_x[ridx]

        axes[i, 0].imshow(np.transpose(r.cpu().numpy(), (1, 2, 0)))
        axes[i, 0].set_title("real-nearest")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(np.transpose(s.cpu().numpy(), (1, 2, 0)))
        axes[i, 1].set_title(f"{method}-eps={epsilon}")
        axes[i, 1].axis("off")

    fig.tight_layout()
    fig.savefig(out_dir / f"{method}_eps_{epsilon}_k_{per_class}.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None, help="Override classifier epochs")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset name")
    parser.add_argument("--data-root", type=str, default=None, help="Override dataset root")
    parser.add_argument("--output-root", type=str, default=None, help="Override output root")
    parser.add_argument("--lr", type=float, default=None, help="Override classifier lr")
    parser.add_argument("--weight-decay", type=float, default=None, help="Override weight decay")
    parser.add_argument(
        "--label-smoothing", type=float, default=None, help="Override CE label smoothing"
    )
    parser.add_argument("--methods", type=str, nargs="*", default=None, help="Override methods")
    parser.add_argument(
        "--classifiers", type=str, nargs="*", default=None, help="Override classifiers"
    )
    parser.add_argument("--epsilons", type=float, nargs="*", default=None, help="Override eps list")
    parser.add_argument(
        "--per-class",
        type=int,
        nargs="*",
        default=None,
        help="Override synthetic images per class list",
    )
    parser.add_argument(
        "--synthetic-eval-ratio",
        type=float,
        default=None,
        help="Hold out this ratio of synthetic data for synthetic-domain evaluation",
    )
    args = parser.parse_args()

    cfg = ExperimentConfig()
    if args.epochs is not None:
        cfg.classifier_epochs = args.epochs
    if args.per_class is not None and len(args.per_class) > 0:
        cfg.synthetic_per_class_list = args.per_class
    if args.methods is not None and len(args.methods) > 0:
        cfg.methods = args.methods
    if args.classifiers is not None and len(args.classifiers) > 0:
        cfg.classifiers = args.classifiers
    if args.epsilons is not None and len(args.epsilons) > 0:
        cfg.epsilons = args.epsilons
    if args.seed is not None:
        cfg.random_seed = args.seed
    if args.dataset is not None:
        cfg.dataset = args.dataset.lower()
    if args.data_root is not None:
        cfg.data_root = args.data_root
    if args.output_root is not None:
        cfg.output_root = args.output_root
    if args.lr is not None:
        cfg.lr = args.lr
    if args.weight_decay is not None:
        cfg.weight_decay = args.weight_decay
    if args.label_smoothing is not None:
        cfg.label_smoothing = args.label_smoothing
    if args.synthetic_eval_ratio is not None:
        cfg.synthetic_eval_ratio = args.synthetic_eval_ratio

    cfg.ensure_dirs()
    set_seed(cfg.random_seed)

    device = torch.device(
        "cuda" if cfg.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    out_root = Path(cfg.output_root)
    dataset_visual_dir = out_root / "visuals" / cfg.dataset
    dataset_visual_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _ = load_dataset(
        cfg.dataset, cfg.data_root, cfg.batch_size, cfg.train_ratio, cfg.num_workers
    )
    train_x, train_y = extract_tensors(train_loader)
    val_x, val_y = extract_tensors(val_loader)

    rows = []
    for per_class in cfg.synthetic_per_class_list:
        synth_map: Dict[str, Callable] = {
            "dp": lambda x, y, e: synthesize_dp(x, y, e, per_class),
            "gan_dp": lambda x, y, e: synthesize_gan_dp(x, y, e, per_class, device),
            "distill_dp": lambda x, y, e: synthesize_distill_dp(x, y, e, per_class),
            "diffusion_dp": lambda x, y, e: synthesize_diffusion_dp(x, y, e, per_class),
        }
        for method in cfg.methods:
            for epsilon in cfg.epsilons:
                synth = synth_map[method](train_x, train_y, epsilon)
                sx, sy = synth["x"], synth["y"]
                split_seed = cfg.random_seed + int(epsilon * 1000) + per_class
                sx_train, sy_train, sx_eval, sy_eval = stratified_split_tensors(
                    sx, sy, val_ratio=cfg.synthetic_eval_ratio, seed=split_seed
                )
                save_visual_compare(
                    method=method,
                    epsilon=epsilon,
                    per_class=per_class,
                    synth_x=sx,
                    real_x=train_x,
                    out_dir=dataset_visual_dir,
                    sample_count=cfg.visual_compare_count,
                )

                synth_train_loader = make_loader_from_tensors(
                    sx_train, sy_train, batch_size=cfg.batch_size, shuffle=True
                )
                original_eval_loader = make_loader_from_tensors(
                    val_x, val_y, batch_size=cfg.batch_size, shuffle=False
                )
                synthetic_eval_loader = make_loader_from_tensors(
                    sx_eval, sy_eval, batch_size=cfg.batch_size, shuffle=False
                )

                for clf_name in cfg.classifiers:
                    clf = build_classifier(clf_name, num_classes=10)
                    result = train_classifier(
                        model=clf,
                        train_loader=synth_train_loader,
                        original_eval_loader=original_eval_loader,
                        synthetic_eval_loader=synthetic_eval_loader,
                        epochs=cfg.classifier_epochs,
                        lr=cfg.lr,
                        device=device,
                        weight_decay=cfg.weight_decay,
                        label_smoothing=cfg.label_smoothing,
                    )
                    rows.append(
                        {
                            "method": method,
                            "epsilon": epsilon,
                            "per_class": per_class,
                            "classifier": clf_name,
                            "synthesis_seconds": synth["seconds"],
                            "train_seconds": result["train_seconds"],
                            "original_pred_seconds": result["original_pred_seconds"],
                            "synthetic_pred_seconds": result["synthetic_pred_seconds"],
                            "original_acc": result["original_acc"],
                            "synthetic_acc": result["synthetic_acc"],
                            "generalization_gap": result["synthetic_acc"] - result["original_acc"],
                            "synthetic_images": int(sx.size(0)),
                            "synthetic_train_images": int(sx_train.size(0)),
                            "synthetic_eval_images": int(sx_eval.size(0)),
                        }
                    )
                    print(
                        f"[{method}|eps={epsilon}|k={per_class}|{clf_name}] "
                        f"gen={synth['seconds']:.2f}s train={result['train_seconds']:.2f}s "
                        f"pred_o={result['original_pred_seconds']:.2f}s pred_s={result['synthetic_pred_seconds']:.2f}s "
                        f"acc_o={result['original_acc']:.4f} acc_s={result['synthetic_acc']:.4f}"
                    )

    df = pd.DataFrame(rows)
    metrics_path = out_root / "metrics.csv"
    df.to_csv(metrics_path, index=False)
    print(f"\nDone. Metrics saved to: {metrics_path}")
    print(f"Visual comparisons saved to: {dataset_visual_dir}")


if __name__ == "__main__":
    main()
