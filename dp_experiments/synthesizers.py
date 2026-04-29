import time
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from privacy import add_gaussian_noise, clip_and_noise_gradients, dp_sigma_from_epsilon


def _class_balanced_sample(
    images: torch.Tensor, labels: torch.Tensor, per_class: int, num_classes: int = 10
) -> Tuple[torch.Tensor, torch.Tensor]:
    x_list, y_list = [], []
    for cls in range(num_classes):
        idx = (labels == cls).nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            continue
        chosen = idx[torch.randint(0, len(idx), (per_class,))]
        x_list.append(images[chosen])
        y_list.append(labels[chosen])
    return torch.cat(x_list, dim=0), torch.cat(y_list, dim=0)


def synthesize_dp(
    images: torch.Tensor, labels: torch.Tensor, epsilon: float, per_class: int
) -> Dict[str, torch.Tensor]:
    t0 = time.perf_counter()
    x, y = _class_balanced_sample(images, labels, per_class=per_class)
    x = add_gaussian_noise(x, epsilon)
    return {"x": x, "y": y, "seconds": time.perf_counter() - t0}


class _TinyGenerator(nn.Module):
    def __init__(self, z_dim: int = 64):
        super().__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 3 * 32 * 32),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.net(z)
        return x.view(-1, 3, 32, 32)


class _TinyDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def synthesize_gan_dp(
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    per_class: int,
    device: torch.device,
    steps: int = 100,
) -> Dict[str, torch.Tensor]:
    t0 = time.perf_counter()
    z_dim = 64
    g = _TinyGenerator(z_dim=z_dim).to(device)
    d = _TinyDiscriminator().to(device)
    opt_g = optim.Adam(g.parameters(), lr=2e-4)
    opt_d = optim.Adam(d.parameters(), lr=2e-4)
    bce = nn.BCELoss()

    real_x, real_y = _class_balanced_sample(images, labels, per_class=per_class)
    real_x = real_x.to(device)
    batch_size = min(128, real_x.size(0))
    noise_scale = dp_sigma_from_epsilon(epsilon, base=0.01)

    for _ in range(steps):
        idx = torch.randint(0, real_x.size(0), (batch_size,), device=device)
        xb = real_x[idx]
        true_label = torch.ones(batch_size, 1, device=device)
        fake_label = torch.zeros(batch_size, 1, device=device)

        z = torch.randn(batch_size, z_dim, device=device)
        fake = g(z)

        opt_d.zero_grad()
        loss_d = bce(d(xb), true_label) + bce(d(fake.detach()), fake_label)
        loss_d.backward()
        clip_and_noise_gradients(d, max_norm=1.0, noise_scale=noise_scale)
        opt_d.step()

        opt_g.zero_grad()
        loss_g = bce(d(fake), true_label)
        loss_g.backward()
        clip_and_noise_gradients(g, max_norm=1.0, noise_scale=noise_scale)
        opt_g.step()

    with torch.no_grad():
        total = per_class * 10
        z = torch.randn(total, z_dim, device=device)
        fake_x = g(z).cpu()
        fake_y = torch.arange(0, 10).repeat_interleave(per_class)
        fake_x = add_gaussian_noise(fake_x, epsilon)
    return {"x": fake_x, "y": fake_y, "seconds": time.perf_counter() - t0}


def synthesize_distill_dp(
    images: torch.Tensor, labels: torch.Tensor, epsilon: float, per_class: int
) -> Dict[str, torch.Tensor]:
    t0 = time.perf_counter()
    x_list, y_list = [], []
    sigma = dp_sigma_from_epsilon(epsilon, base=0.05)
    for cls in range(10):
        idx = (labels == cls).nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            continue
        class_x = images[idx]
        proto = class_x.mean(dim=0, keepdim=True)
        proto = proto.repeat(per_class, 1, 1, 1)
        proto = proto + torch.randn_like(proto) * sigma
        x_list.append(torch.clamp(proto, -3.0, 3.0))
        y_list.append(torch.full((per_class,), cls, dtype=torch.long))
    return {
        "x": torch.cat(x_list, dim=0),
        "y": torch.cat(y_list, dim=0),
        "seconds": time.perf_counter() - t0,
    }


def synthesize_diffusion_dp(
    images: torch.Tensor, labels: torch.Tensor, epsilon: float, per_class: int, steps: int = 20
) -> Dict[str, torch.Tensor]:
    t0 = time.perf_counter()
    x, y = _class_balanced_sample(images, labels, per_class=per_class)
    sigma = dp_sigma_from_epsilon(epsilon, base=0.03)
    noise = torch.randn_like(x)
    alpha = torch.linspace(0.1, 0.9, steps)
    out = noise
    for a in alpha:
        out = a * x + (1 - a) * out
        out = out + torch.randn_like(out) * sigma
    out = torch.clamp(out, -3.0, 3.0)
    return {"x": out, "y": y, "seconds": time.perf_counter() - t0}

