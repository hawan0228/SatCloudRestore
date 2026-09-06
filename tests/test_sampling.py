import torch

from satcloudrestore.diffusion import GaussianDiffusion
from satcloudrestore.sampling import ddim_sample
from satcloudrestore.uncertainty import sample_uncertainty


class ZeroModel(torch.nn.Module):
    def forward(self, x, t):
        return torch.zeros_like(x[:, :3])


def test_ddim_short_reproducible_and_composites_known_region():
    diffusion = GaussianDiffusion(timesteps=10)
    cloudy = torch.rand(2, 3, 8, 8) * 2 - 1; mask = torch.zeros(2, 1, 8, 8); mask[:, :, 2:6, 2:6] = 1
    a = ddim_sample(ZeroModel(), diffusion, cloudy, mask, steps=4, seed=7)
    b = ddim_sample(ZeroModel(), diffusion, cloudy, mask, steps=4, seed=7)
    assert a.shape == cloudy.shape and torch.equal(a, b)
    assert torch.equal(a * (1 - mask), cloudy * (1 - mask))


def test_uncertainty_is_single_channel():
    diffusion = GaussianDiffusion(timesteps=6)
    cloudy = torch.zeros(2, 3, 8, 8); mask = torch.ones(2, 1, 8, 8)
    mean, variance, draws = sample_uncertainty(ZeroModel(), diffusion, cloudy, mask, steps=2, samples=3, seed=1)
    assert mean.shape == cloudy.shape
    assert variance.shape == (2, 1, 8, 8)
    assert draws.shape == (3, 2, 3, 8, 8)
