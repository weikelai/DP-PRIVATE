import pickle
import tarfile
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class CIFAR10TarDataset(torch.utils.data.Dataset):
    def __init__(self, archive_path: str | Path, train: bool, transform=None):
        self.archive_path = Path(archive_path)
        self.train = train
        self.transform = transform
        self.data, self.targets = self._load_archive()

    def _load_archive(self):
        batch_names = (
            [f"cifar-10-batches-py/data_batch_{i}" for i in range(1, 6)]
            if self.train
            else ["cifar-10-batches-py/test_batch"]
        )
        data_parts = []
        target_parts = []
        with tarfile.open(self.archive_path, "r:gz") as tar:
            for name in batch_names:
                f = tar.extractfile(name)
                if f is None:
                    raise FileNotFoundError(f"{name} not found in {self.archive_path}")
                entry = pickle.load(f, encoding="latin1")
                data_parts.append(entry["data"])
                target_parts.extend(entry["labels"])

        data = np.concatenate(data_parts, axis=0)
        data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        return data, target_parts

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        img = Image.fromarray(self.data[index])
        target = self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def load_cifar10(
    data_root: str,
    batch_size: int,
    train_ratio: float,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    archive_path = Path(data_root) / "cifar-10-python.tar.gz"
    if archive_path.exists():
        train_full = CIFAR10TarDataset(archive_path, train=True, transform=transform)
        test_set = CIFAR10TarDataset(archive_path, train=False, transform=transform)
    else:
        train_full = datasets.CIFAR10(
            root=data_root, train=True, download=True, transform=transform
        )
        test_set = datasets.CIFAR10(
            root=data_root, train=False, download=True, transform=transform
        )

    train_len = int(len(train_full) * train_ratio)
    val_len = len(train_full) - train_len
    train_set, val_set = random_split(train_full, [train_len, val_len])

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, test_loader


def load_mnist(
    data_root: str,
    batch_size: int,
    train_ratio: float,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # 转为 3x32x32，复用现有 CNN/ResNet20 与合成器流程
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_full = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)

    train_len = int(len(train_full) * train_ratio)
    val_len = len(train_full) - train_len
    train_set, val_set = random_split(train_full, [train_len, val_len])

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, test_loader


def load_dataset(
    dataset: str,
    data_root: str,
    batch_size: int,
    train_ratio: float,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    name = dataset.lower()
    if name == "cifar10":
        return load_cifar10(data_root, batch_size, train_ratio, num_workers)
    if name == "mnist":
        return load_mnist(data_root, batch_size, train_ratio, num_workers)
    raise ValueError(f"Unsupported dataset: {dataset}")


def extract_tensors(loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    images, labels = [], []
    for x, y in loader:
        images.append(x)
        labels.append(y)
    return torch.cat(images, dim=0), torch.cat(labels, dim=0)
