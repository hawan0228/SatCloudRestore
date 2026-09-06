from __future__ import annotations

import math
import warnings

import numpy as np
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity


def _validate(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 3:
        raise ValueError("prediction and target must have matching HxWx3 shapes")
    if mask is not None:
        mask = np.asarray(mask, dtype=np.float64).squeeze()
        if mask.shape != prediction.shape[:2] or mask.sum() <= 0:
            raise ValueError("mask must match image dimensions and contain at least one positive pixel")
    return prediction, target, mask


def psnr(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction, target, _ = _validate(prediction, target)
    mse = float(np.mean((prediction - target) ** 2))
    return math.inf if mse == 0 else float(10 * math.log10(1 / mse))


def ssim(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction, target, _ = _validate(prediction, target)
    return float(structural_similarity(target, prediction, data_range=1.0, channel_axis=-1))


def masked_mae(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    prediction, target, mask = _validate(prediction, target, mask)
    return float((np.abs(prediction - target) * mask[..., None]).sum() / (mask.sum() * 3))


def masked_psnr(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    prediction, target, mask = _validate(prediction, target, mask)
    mse = float(((prediction - target) ** 2 * mask[..., None]).sum() / (mask.sum() * 3))
    return math.inf if mse == 0 else float(10 * math.log10(1 / mse))


def masked_ssim(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    prediction, target, mask = _validate(prediction, target, mask)
    _, score_map = structural_similarity(target, prediction, data_range=1.0, channel_axis=-1, full=True)
    if score_map.ndim == 3:
        score_map = score_map.mean(axis=-1)
    return float((score_map * mask).sum() / mask.sum())


def all_metrics(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    return {"full_psnr": psnr(prediction, target), "full_ssim": ssim(prediction, target),
            "cloud_psnr": masked_psnr(prediction, target, mask), "cloud_ssim": masked_ssim(prediction, target, mask),
            "cloud_mae": masked_mae(prediction, target, mask)}


def uncertainty_error_correlation(uncertainty: np.ndarray, error: np.ndarray, mask: np.ndarray) -> float | None:
    active = np.asarray(mask).squeeze() > 0
    x, y = np.asarray(uncertainty).squeeze()[active], np.asarray(error).squeeze()[active]
    if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        value = spearmanr(x, y).statistic
    return None if not np.isfinite(value) else float(value)
