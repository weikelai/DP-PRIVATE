from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class ExperimentConfig:
    dataset: str = "cifar10"
    data_root: str = "dataset"
    output_root: str = "dp_experiments/outputs"
    batch_size: int = 128
    num_workers: int = 0
    train_ratio: float = 0.8
    random_seed: int = 42

    # 图中要求的隐私预算中心/标准差组合，按 epsilon 近似管理
    epsilons: List[float] = field(default_factory=lambda: [0.1, 0.5, 1.0])
    synthetic_per_class_list: List[int] = field(default_factory=lambda: [100, 300])

    methods: List[str] = field(
        default_factory=lambda: ["dp", "gan_dp", "distill_dp", "diffusion_dp"]
    )
    classifiers: List[str] = field(default_factory=lambda: ["cnn", "resnet18", "resnet20"])
    classifier_epochs: int = 3
    lr: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    device: str = "cuda"
    visual_compare_count: int = 5
    synthetic_eval_ratio: float = 0.2

    def ensure_dirs(self) -> None:
        Path(self.output_root).mkdir(parents=True, exist_ok=True)
        Path(self.output_root, "visuals").mkdir(parents=True, exist_ok=True)
