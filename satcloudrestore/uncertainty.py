from __future__ import annotations

import torch

from .sampling import ddim_sample


@torch.no_grad()
def sample_uncertainty(model, diffusion, cloudy, mask, steps: int, samples: int = 8, seed: int = 42):
    if samples < 1:
        raise ValueError("samples must be positive")
    draws = torch.stack([ddim_sample(model, diffusion, cloudy, mask, steps=steps, seed=seed + i) for i in range(samples)])
    mean = draws.mean(0)
    variance = draws.var(0, unbiased=False)
    return mean, variance.mean(dim=1, keepdim=True), draws
