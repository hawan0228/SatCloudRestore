import torch

from satcloudrestore.model import ConditionalUNet, group_count


def test_model_shape_and_groups():
    model = ConditionalUNet(base_channels=16, channel_multipliers=[1, 2, 4], time_dim=32)
    output = model(torch.randn(2, 7, 64, 64), torch.tensor([0, 10]))
    assert output.shape == (2, 3, 64, 64)
    assert all(channels % group_count(channels) == 0 for channels in (16, 32, 64))
