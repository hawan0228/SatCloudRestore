import numpy as np
import torch

from satcloudrestore.cloud_generator import SyntheticCloudGenerator
from satcloudrestore.utils import to_image_range, to_model_range


def test_cloud_shape_range_coverage_reproducibility():
    clean = np.random.default_rng(1).random((64, 64, 3), dtype=np.float32)
    original = clean.copy(); generator = SyntheticCloudGenerator(3)
    first = generator.generate(clean, 0.3, 0.75, sigma=2, seed=9)
    second = generator.generate(clean, 0.3, 0.75, sigma=2, seed=9)
    assert first.cloudy.shape == clean.shape and first.soft_mask.shape == (64, 64)
    assert first.cloudy.dtype == np.float32 and first.binary_mask.dtype == np.float32
    assert 0 <= first.cloudy.min() <= first.cloudy.max() <= 1
    assert abs(first.actual_coverage - 0.3) < 0.02
    np.testing.assert_array_equal(first.cloudy, second.cloudy)
    np.testing.assert_array_equal(clean, original)


def test_normalization_round_trip():
    x = torch.rand(2, 3, 8, 8)
    assert torch.allclose(to_image_range(to_model_range(x)), x)
