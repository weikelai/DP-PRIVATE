import time
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def make_loader_from_tensors(
    x: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool = True
) -> DataLoader:
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def stratified_split_tensors(
    x: torch.Tensor,
    y: torch.Tensor,
    val_ratio: float = 0.2,
    seed: int | None = None,
):
    """Split synthetic tensors per class so synthetic accuracy is not train-set self-test."""
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    train_parts = []
    val_parts = []
    for cls in torch.unique(y, sorted=True):
        cls_idx = torch.nonzero(y == cls, as_tuple=False).flatten()
        if cls_idx.numel() <= 1:
            train_parts.append(cls_idx)
            continue

        perm = cls_idx[torch.randperm(cls_idx.numel(), generator=generator)]
        val_count = max(1, int(round(cls_idx.numel() * val_ratio)))
        val_count = min(val_count, cls_idx.numel() - 1)
        val_parts.append(perm[:val_count])
        train_parts.append(perm[val_count:])

    train_idx = torch.cat(train_parts)
    val_idx = torch.cat(val_parts) if val_parts else train_idx
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def train_classifier(
    model: torch.nn.Module,
    train_loader: DataLoader,
    original_eval_loader: DataLoader,
    synthetic_eval_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.05,
) -> Dict[str, float]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1)
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    t0 = time.perf_counter()

    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        scheduler.step()

    train_seconds = time.perf_counter() - t0
    original_acc, original_pred_seconds = evaluate_classifier(
        model, original_eval_loader, device
    )
    synthetic_acc, synthetic_pred_seconds = evaluate_classifier(
        model, synthetic_eval_loader, device
    )
    return {
        "train_seconds": train_seconds,
        "original_acc": original_acc,
        "synthetic_acc": synthetic_acc,
        "original_pred_seconds": original_pred_seconds,
        "synthetic_pred_seconds": synthetic_pred_seconds,
    }


def evaluate_classifier(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
):
    model.eval()
    correct = 0
    total = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    pred_seconds = time.perf_counter() - t0
    return correct / max(total, 1), pred_seconds

