from __future__ import annotations

import math

import torch
from torch import nn


def group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        frequencies = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1))
        values = t.float()[:, None] * frequencies[None]
        embedding = torch.cat((values.sin(), values.cos()), dim=1)
        return nn.functional.pad(embedding, (0, self.dim - embedding.shape[1]))


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_channels))
        self.norm2 = nn.GroupNorm(group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        h = self.conv1(nn.functional.silu(self.norm1(x)))
        h = h + self.time(time)[:, :, None, None]
        h = self.conv2(nn.functional.silu(self.norm2(h)))
        return h + self.skip(x)


class ConditionalUNet(nn.Module):
    def __init__(self, in_channels: int = 7, out_channels: int = 3, base_channels: int = 32,
                 channel_multipliers: list[int] | tuple[int, ...] = (1, 2, 4), time_dim: int = 128):
        super().__init__()
        channels = [base_channels * m for m in channel_multipliers]
        self.time_embed = nn.Sequential(SinusoidalTimeEmbedding(time_dim), nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.input = nn.Conv2d(in_channels, channels[0], 3, padding=1)
        self.down_blocks, self.downsamples = nn.ModuleList(), nn.ModuleList()
        for i, ch in enumerate(channels):
            self.down_blocks.append(ResBlock(ch, ch, time_dim))
            if i < len(channels) - 1:
                self.downsamples.append(nn.Conv2d(ch, channels[i + 1], 4, stride=2, padding=1))
        self.middle = nn.ModuleList([ResBlock(channels[-1], channels[-1], time_dim), ResBlock(channels[-1], channels[-1], time_dim)])
        self.up_blocks, self.upsamples = nn.ModuleList(), nn.ModuleList()
        current = channels[-1]
        for i in range(len(channels) - 1, -1, -1):
            ch = channels[i]
            self.up_blocks.append(ResBlock(current + ch, ch, time_dim))
            current = ch
            if i > 0:
                self.upsamples.append(nn.ConvTranspose2d(ch, channels[i - 1], 4, stride=2, padding=1))
                current = channels[i - 1]
        self.output = nn.Sequential(nn.GroupNorm(group_count(channels[0]), channels[0]), nn.SiLU(), nn.Conv2d(channels[0], out_channels, 3, padding=1))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 7:
            raise ValueError(f"Expected Bx7xHxW input, got {tuple(x.shape)}")
        time = self.time_embed(t)
        h, skips = self.input(x), []
        for i, block in enumerate(self.down_blocks):
            h = block(h, time)
            skips.append(h)
            if i < len(self.downsamples):
                h = self.downsamples[i](h)
        for block in self.middle:
            h = block(h, time)
        upsample_index = 0
        for i, block in enumerate(self.up_blocks):
            skip = skips.pop()
            if h.shape[-2:] != skip.shape[-2:]:
                h = nn.functional.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = block(torch.cat((h, skip), dim=1), time)
            if i < len(self.up_blocks) - 1:
                h = self.upsamples[upsample_index](h)
                upsample_index += 1
        return self.output(h)


def build_model(config: dict) -> ConditionalUNet:
    return ConditionalUNet(**config)
