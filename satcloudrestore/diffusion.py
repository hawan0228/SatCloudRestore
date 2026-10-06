from __future__ import annotations

import math

import torch
from torch import nn


def make_betas(timesteps: int, schedule: str) -> torch.Tensor:
    if schedule == "linear":
        return torch.linspace(1e-4, 0.02, timesteps, dtype=torch.float32)
    if schedule == "cosine":
        steps = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64)
        cumulative = torch.cos(((steps / timesteps + 0.008) / 1.008) * math.pi / 2) ** 2
        cumulative = cumulative / cumulative[0]
        return (1 - cumulative[1:] / cumulative[:-1]).clamp(1e-8, 0.999).float()
    raise ValueError(f"Unknown beta schedule: {schedule}")


def extract(buffer: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size | tuple[int, ...]) -> torch.Tensor:
    if timesteps.ndim != 1:
        raise ValueError("timesteps must be one-dimensional")
    values = buffer.gather(0, timesteps.to(buffer.device))
    return values.reshape(timesteps.shape[0], *((1,) * (len(shape) - 1)))


def stratified_validation_timesteps(batch_size: int, timesteps: int, batch_index: int,
                                    device: torch.device | str) -> torch.Tensor:
    """Deterministically cover the full diffusion horizon in every validation batch."""
    if batch_size < 1 or timesteps < 2:
        raise ValueError("batch_size must be positive and timesteps at least 2")
    if batch_size == 1:
        return torch.tensor([batch_index % timesteps], device=device, dtype=torch.long)
    values = torch.linspace(0, timesteps - 1, batch_size, device=device).round().long()
    return torch.roll(values, shifts=batch_index % batch_size)


class GaussianDiffusion(nn.Module):
    def __init__(self, timesteps: int = 200, beta_schedule: str = "linear"):
        super().__init__()
        betas = make_betas(timesteps, beta_schedule)
        alphas = 1 - betas
        cumulative = torch.cumprod(alphas, dim=0)
        previous = torch.cat((torch.ones(1), cumulative[:-1]))
        posterior_variance = betas * (1 - previous) / (1 - cumulative).clamp_min(1e-20)
        for name, value in {
            "betas": betas, "alphas": alphas, "alpha_bars": cumulative,
            "alpha_bars_prev": previous, "sqrt_alpha_bars": cumulative.sqrt(),
            "sqrt_one_minus_alpha_bars": (1 - cumulative).clamp_min(0).sqrt(),
            "sqrt_recip_alpha_bars": cumulative.rsqrt(),
            "sqrt_recipm1_alpha_bars": (1 / cumulative - 1).clamp_min(0).sqrt(),
            "posterior_variance": posterior_variance.clamp_min(1e-20),
            "posterior_mean_coef1": betas * previous.sqrt() / (1 - cumulative).clamp_min(1e-20),
            "posterior_mean_coef2": (1 - previous) * alphas.sqrt() / (1 - cumulative).clamp_min(1e-20),
        }.items():
            self.register_buffer(name, value.float())
        self.timesteps = timesteps

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        noise = torch.randn_like(x0) if noise is None else noise
        return extract(self.sqrt_alpha_bars, t, x0.shape) * x0 + extract(self.sqrt_one_minus_alpha_bars, t, x0.shape) * noise

    def predict_x0(self, xt: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (extract(self.sqrt_recip_alpha_bars, t, xt.shape) * xt - extract(self.sqrt_recipm1_alpha_bars, t, xt.shape) * noise).clamp(-1, 1)

    def p_sample(self, model: nn.Module, xt: torch.Tensor, t: torch.Tensor, cloudy: torch.Tensor,
                 mask: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        predicted_noise = model(torch.cat((xt, cloudy, mask), dim=1), t)
        x0 = self.predict_x0(xt, t, predicted_noise)
        mean = extract(self.posterior_mean_coef1, t, xt.shape) * x0 + extract(self.posterior_mean_coef2, t, xt.shape) * xt
        noise = torch.randn(xt.shape, device=xt.device, dtype=xt.dtype, generator=generator)
        nonzero = (t != 0).float().reshape(t.shape[0], *((1,) * (xt.ndim - 1)))
        return mean + nonzero * extract(self.posterior_variance, t, xt.shape).sqrt() * noise

    @torch.no_grad()
    def sample(self, model: nn.Module, cloudy: torch.Tensor, mask: torch.Tensor, seed: int = 42) -> torch.Tensor:
        generator = torch.Generator(device=cloudy.device).manual_seed(seed)
        x = torch.randn(cloudy.shape, device=cloudy.device, generator=generator)
        known_noise = torch.randn(cloudy.shape, device=cloudy.device, dtype=cloudy.dtype, generator=generator)
        for step in reversed(range(self.timesteps)):
            t = torch.full((cloudy.shape[0],), step, device=cloudy.device, dtype=torch.long)
            known_xt = self.q_sample(cloudy, t, known_noise)
            x = x * mask + known_xt * (1 - mask)
            x = self.p_sample(model, x, t, cloudy, mask, generator)
        return (x.clamp(-1, 1) * mask + cloudy * (1 - mask)).clamp(-1, 1)


def weighted_noise_loss(prediction: torch.Tensor, noise: torch.Tensor, mask: torch.Tensor, mask_weight: float) -> torch.Tensor:
    if mask.shape[1] != 1:
        raise ValueError("mask must have one channel")
    weights = 1 + float(mask_weight) * mask
    return (weights * (prediction - noise).square()).mean()
