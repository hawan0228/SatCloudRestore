import pytest
import torch

from satcloudrestore.model import ConditionalUNet, group_count


@pytest.mark.parametrize("upsample_mode", ["transpose", "resize_conv"])
def test_model_shape_and_groups(upsample_mode):
    model = ConditionalUNet(base_channels=16, channel_multipliers=[1, 2, 4], time_dim=32,
                            upsample_mode=upsample_mode)
    output = model(torch.randn(2, 7, 64, 64), torch.tensor([0, 10]))
    assert output.shape == (2, 3, 64, 64)
    assert all(channels % group_count(channels) == 0 for channels in (16, 32, 64))


def test_resize_conv_contains_no_transposed_convolution():
    resize = ConditionalUNet(base_channels=8, channel_multipliers=[1, 2], time_dim=16,
                             upsample_mode="resize_conv")
    legacy = ConditionalUNet(base_channels=8, channel_multipliers=[1, 2], time_dim=16,
                             upsample_mode="transpose")
    assert not any(isinstance(module, torch.nn.ConvTranspose2d) for module in resize.modules())
    assert any(isinstance(module, torch.nn.ConvTranspose2d) for module in legacy.modules())
