from __future__ import annotations

import cv2
import numpy as np
import torch

from .diffusion import GaussianDiffusion
from .model import build_model
from .utils import load_checkpoint, to_image_range, to_model_range


def telea_inpaint(cloudy: np.ndarray, binary_mask: np.ndarray, radius: float = 3.0) -> np.ndarray:
    rgb = (np.asarray(cloudy).clip(0, 1) * 255).round().astype(np.uint8)
    mask = (np.asarray(binary_mask).squeeze() > 0.5).astype(np.uint8) * 255
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    restored = cv2.inpaint(bgr, mask, float(radius), cv2.INPAINT_TELEA)
    return cv2.cvtColor(restored, cv2.COLOR_BGR2RGB).astype(np.float32) / 255


def load_model_bundle(path: str, config: dict, device: torch.device):
    checkpoint = load_checkpoint(path, device)
    saved = checkpoint.get("config", {})
    if saved:
        for key in ("image_size", "normalization", "model", "diffusion"):
            if saved.get(key) != config.get(key):
                raise ValueError(f"Checkpoint configuration mismatch in '{key}'")
    model = build_model(config["model"]).to(device)
    model.load_state_dict(checkpoint.get("ema", checkpoint["model"]))
    model.eval()
    diffusion = GaussianDiffusion(**{k: config["diffusion"][k] for k in ("timesteps", "beta_schedule")}).to(device)
    return model, diffusion, checkpoint


def tensor_to_rgb(x: torch.Tensor) -> np.ndarray:
    return to_image_range(x.detach().cpu())[0].permute(1, 2, 0).numpy()


def rgb_to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return to_model_range(torch.from_numpy(np.asarray(x, dtype=np.float32)).permute(2, 0, 1)[None]).to(device)
