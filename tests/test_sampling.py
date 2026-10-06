import torch

from satcloudrestore.diffusion import GaussianDiffusion
from satcloudrestore.sampling import ddim_sample
from satcloudrestore.uncertainty import sample_uncertainty


class ZeroModel(torch.nn.Module):
    def forward(self, x, t):
        return torch.zeros_like(x[:, :3])


class RecordingModel(ZeroModel):
    def __init__(self):
        super().__init__()
        self.first_xt = None

    def forward(self, x, t):
        if self.first_xt is None:
            self.first_xt = x[:, :3].detach().clone()
        return super().forward(x, t)


def test_ddim_short_reproducible_and_composites_known_region():
    diffusion = GaussianDiffusion(timesteps=10)
    cloudy = torch.rand(2, 3, 8, 8) * 2 - 1; mask = torch.zeros(2, 1, 8, 8); mask[:, :, 2:6, 2:6] = 1
    a = ddim_sample(ZeroModel(), diffusion, cloudy, mask, steps=4, seed=7)
    b = ddim_sample(ZeroModel(), diffusion, cloudy, mask, steps=4, seed=7)
    assert a.shape == cloudy.shape and torch.equal(a, b)
    assert torch.equal(a * (1 - mask), cloudy * (1 - mask))


def test_ddim_conditions_known_region_at_each_reverse_step():
    diffusion = GaussianDiffusion(timesteps=10, beta_schedule="cosine")
    cloudy = torch.rand(1, 3, 8, 8) * 2 - 1
    mask = torch.zeros(1, 1, 8, 8)
    mask[:, :, 2:6, 2:6] = 1
    seed = 17
    generator = torch.Generator().manual_seed(seed)
    torch.randn(cloudy.shape, generator=generator)  # Initial reverse-process noise.
    known_noise = torch.randn(cloudy.shape, generator=generator)
    t = torch.full((1,), diffusion.timesteps - 1, dtype=torch.long)
    expected = diffusion.q_sample(cloudy, t, known_noise)
    model = RecordingModel()
    ddim_sample(model, diffusion, cloudy, mask, steps=4, seed=seed)
    assert torch.allclose(model.first_xt * (1 - mask), expected * (1 - mask))


def test_uncertainty_is_single_channel():
    diffusion = GaussianDiffusion(timesteps=6)
    cloudy = torch.zeros(2, 3, 8, 8); mask = torch.ones(2, 1, 8, 8)
    mean, variance, draws = sample_uncertainty(ZeroModel(), diffusion, cloudy, mask, steps=2, samples=3, seed=1)
    assert mean.shape == cloudy.shape
    assert variance.shape == (2, 1, 8, 8)
    assert draws.shape == (3, 2, 3, 8, 8)
