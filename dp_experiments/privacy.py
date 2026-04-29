import torch


def dp_sigma_from_epsilon(epsilon: float, base: float = 0.2) -> float:
    # epsilon 越小，噪声越大
    return base / max(epsilon, 1e-6)


def add_gaussian_noise(x: torch.Tensor, epsilon: float) -> torch.Tensor:
    sigma = dp_sigma_from_epsilon(epsilon)
    noisy = x + torch.randn_like(x) * sigma
    return torch.clamp(noisy, -3.0, 3.0)


def clip_and_noise_gradients(model: torch.nn.Module, max_norm: float, noise_scale: float) -> None:
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
    if torch.isfinite(total_norm):
        for p in model.parameters():
            if p.grad is None:
                continue
            p.grad.add_(torch.randn_like(p.grad) * noise_scale)

