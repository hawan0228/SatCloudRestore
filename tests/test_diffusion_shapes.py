import torch

from satcloudrestore.diffusion import GaussianDiffusion, extract, weighted_noise_loss


class ZeroModel(torch.nn.Module):
    def forward(self, x, t):
        return torch.zeros_like(x[:, :3])


def test_q_sample_extract_and_reverse_shapes():
    diffusion = GaussianDiffusion(timesteps=10)
    x = torch.randn(3, 3, 8, 8); t = torch.tensor([0, 4, 9]); noise = torch.randn_like(x)
    assert extract(diffusion.alpha_bars, t, x.shape).shape == (3, 1, 1, 1)
    xt = diffusion.q_sample(x, t, noise)
    assert xt.shape == x.shape and torch.isfinite(xt).all()
    result = diffusion.p_sample(ZeroModel(), xt, t, torch.randn_like(x), torch.ones(3, 1, 8, 8))
    assert result.shape == x.shape and torch.isfinite(result).all()
    assert weighted_noise_loss(noise, noise, torch.ones(3, 1, 8, 8), 4).item() == 0
