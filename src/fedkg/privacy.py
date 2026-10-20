"""Simple client-level differential privacy on model updates (clip + Gaussian noise).
Note: this is a teaching-grade mechanism; use Opacus / an RDP accountant for formal epsilon."""
import torch


def clip_update(delta, clip_norm):
    total = torch.sqrt(sum((d.float() ** 2).sum() for d in delta.values()))
    scale = min(1.0, clip_norm / (total.item() + 1e-12))
    return {k: v * scale for k, v in delta.items()}


def add_noise(avg_delta, clip_norm, noise_multiplier, n_clients):
    std = noise_multiplier * clip_norm / n_clients
    return {k: v + torch.randn_like(v) * std for k, v in avg_delta.items()}
