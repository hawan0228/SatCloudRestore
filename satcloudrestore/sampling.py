from __future__ import annotations

import torch
from torch import nn

from .diffusion import GaussianDiffusion, extract


@torch.no_grad()
def ddim_sample(model: nn.Module, diffusion: GaussianDiffusion, cloudy: torch.Tensor, mask: torch.Tensor,
                steps: int = 25, eta: float = 0.0, seed: int = 42, return_intermediates: bool = False):
    if steps < 1 or steps > diffusion.timesteps:
        raise ValueError(f"steps must be in [1, {diffusion.timesteps}]")
    if eta < 0:
        raise ValueError("eta must be non-negative")
    generator = torch.Generator(device=cloudy.device).manual_seed(seed)
    x = torch.randn(cloudy.shape, device=cloudy.device, dtype=cloudy.dtype, generator=generator)
    # Training observes q(x_t | clean) in known regions. Keep those regions on
    # the same forward-diffusion trajectory during inpainting instead of asking
    # the model to generate them freely and replacing them only at the end.
    known_noise = torch.randn(cloudy.shape, device=cloudy.device, dtype=cloudy.dtype, generator=generator)
    sequence = torch.linspace(diffusion.timesteps - 1, 0, steps, device=cloudy.device).long()
    intermediates = []
    for index, current in enumerate(sequence):
        t = torch.full((cloudy.shape[0],), int(current), device=cloudy.device, dtype=torch.long)
        known_xt = diffusion.q_sample(cloudy, t, known_noise)
        x = x * mask + known_xt * (1 - mask)
        predicted_noise = model(torch.cat((x, cloudy, mask), dim=1), t)
        x0 = diffusion.predict_x0(x, t, predicted_noise)
        if index == len(sequence) - 1:
            x = x0
        else:
            previous = sequence[index + 1]
            alpha = extract(diffusion.alpha_bars, t, x.shape)
            alpha_previous = diffusion.alpha_bars[previous].reshape(1, 1, 1, 1)
            sigma = eta * (((1 - alpha_previous) / (1 - alpha).clamp_min(1e-20)) * (1 - alpha / alpha_previous)).clamp_min(0).sqrt()
            direction = (1 - alpha_previous - sigma.square()).clamp_min(0).sqrt() * predicted_noise
            noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator) if eta > 0 else 0
            x = alpha_previous.sqrt() * x0 + direction + sigma * noise
        if return_intermediates:
            intermediates.append(x.detach().clamp(-1, 1).cpu())
    # Preserve known regions exactly; soft edges blend smoothly.
    final = (x.clamp(-1, 1) * mask + cloudy * (1 - mask)).clamp(-1, 1)
    return (final, intermediates) if return_intermediates else final
