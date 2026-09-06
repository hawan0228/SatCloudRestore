from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, zoom


@dataclass(frozen=True)
class CloudSample:
    clean: np.ndarray
    cloudy: np.ndarray
    soft_mask: np.ndarray
    binary_mask: np.ndarray
    actual_coverage: float
    opacity: float


class SyntheticCloudGenerator:
    def __init__(self, octaves: int = 3):
        if octaves < 1:
            raise ValueError("octaves must be positive")
        self.octaves = octaves

    def _noise(self, height: int, width: int, rng: np.random.Generator, sigma: float) -> np.ndarray:
        total = np.zeros((height, width), dtype=np.float32)
        weight_sum = 0.0
        for octave in range(self.octaves):
            scale = 2 ** (octave + 2)
            small_h, small_w = max(2, height // scale), max(2, width // scale)
            coarse = rng.normal(size=(small_h, small_w)).astype(np.float32)
            layer = zoom(coarse, (height / small_h, width / small_w), order=3)[:height, :width]
            weight = 0.55**octave
            total += weight * layer
            weight_sum += weight
        total = gaussian_filter(total / weight_sum, sigma=max(0.1, sigma), mode="reflect")
        return (total - total.min()) / max(float(total.max() - total.min()), 1e-8)

    def generate(self, clean: np.ndarray, coverage: float, opacity: float, sigma: float = 2.0, seed: int | None = None) -> CloudSample:
        image = np.asarray(clean, dtype=np.float32).copy()
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("clean must have shape HxWx3")
        if image.min() < 0 or image.max() > 1:
            raise ValueError("clean must be in [0, 1]")
        if not 0 < coverage < 1 or not 0 <= opacity <= 1:
            raise ValueError("coverage must be in (0,1) and opacity in [0,1]")
        rng = np.random.default_rng(seed)
        h, w, _ = image.shape
        field = self._noise(h, w, rng, sigma)
        threshold = float(np.quantile(field, 1 - coverage))
        binary = (field >= threshold).astype(np.float32)
        softness = max(0.5, sigma * 0.55)
        soft = gaussian_filter(binary, sigma=softness, mode="reflect")
        soft = np.clip(soft / max(float(soft.max()), 1e-8), 0, 1).astype(np.float32)
        base = rng.uniform(0.78, 1.0, size=(1, 1, 3)).astype(np.float32)
        base[..., 2] = np.clip(base[..., 2] + 0.03, 0, 1)
        texture_noise = gaussian_filter(rng.normal(size=(h, w)).astype(np.float32), sigma=max(1.0, sigma))
        texture_noise /= max(float(np.max(np.abs(texture_noise))), 1e-8)
        texture = np.clip(base + texture_noise[..., None] * 0.08, 0, 1)
        blend = (opacity * soft)[..., None]
        cloudy = np.clip(image * (1 - blend) + texture * blend, 0, 1).astype(np.float32)
        return CloudSample(image, cloudy, soft, binary, float(binary.mean()), float(opacity))
