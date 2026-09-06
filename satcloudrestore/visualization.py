from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def save_comparison(path: str | Path, clean, cloudy, mask, telea, restored, uncertainty=None) -> None:
    error = np.abs(np.asarray(restored) - np.asarray(clean)).mean(axis=-1)
    panels = [(clean, "Clean GT", None), (cloudy, "Cloudy", None), (mask, "Cloud Mask", "gray"),
              (telea, "Telea", None), (restored, "SatCloudRestore", None), (error, "Error Map", "magma")]
    if uncertainty is not None:
        panels.append((uncertainty, "Uncertainty", "viridis"))
    fig, axes = plt.subplots(1, len(panels), figsize=(3 * len(panels), 3), dpi=140)
    for axis, (value, title, cmap) in zip(axes, panels):
        shown = axis.imshow(np.nan_to_num(value).clip(0, 1) if cmap is None else value, cmap=cmap, vmin=0 if cmap is None else None, vmax=1 if cmap is None else None)
        axis.set_title(title)
        axis.axis("off")
        if cmap in {"magma", "viridis"}:
            fig.colorbar(shown, ax=axis, fraction=0.046, pad=0.04)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_gif(frames: list[np.ndarray], path: str | Path, duration: int = 120) -> None:
    if not frames:
        raise ValueError("Cannot save an empty GIF")
    images = [Image.fromarray((np.nan_to_num(frame).clip(0, 1) * 255).astype(np.uint8)) for frame in frames]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    images[0].save(path, save_all=True, append_images=images[1:], duration=duration, loop=0)
