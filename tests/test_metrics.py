import math

import numpy as np
import pytest

from satcloudrestore.metrics import masked_mae, masked_psnr, masked_ssim, uncertainty_error_correlation


def test_masked_metrics_known_values():
    target = np.zeros((8, 8, 3)); prediction = target.copy(); mask = np.zeros((8, 8)); mask[2:6, 2:6] = 1
    assert math.isinf(masked_psnr(prediction, target, mask))
    assert masked_mae(prediction, target, mask) == 0
    assert masked_ssim(prediction, target, mask) == pytest.approx(1)
    prediction[2:6, 2:6] = 0.5
    assert masked_mae(prediction, target, mask) == pytest.approx(0.5)
    assert masked_psnr(prediction, target, mask) == pytest.approx(10 * math.log10(4))


def test_zero_mask_and_constant_correlation_safe():
    image = np.zeros((8, 8, 3)); mask = np.zeros((8, 8))
    with pytest.raises(ValueError): masked_mae(image, image, mask)
    mask[:] = 1
    assert uncertainty_error_correlation(np.ones((8, 8)), np.ones((8, 8)), mask) is None
