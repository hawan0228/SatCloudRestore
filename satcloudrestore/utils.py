from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str | None = None) -> torch.device:
    """Resolve a device without silently falling back from an explicit CUDA request."""
    name = requested or ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(name)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Invalid device '{name}'") from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested, but torch.cuda.is_available() is False")
    return device


def validate_experiment_name(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", name):
        raise ValueError("experiment name must be 1-80 characters using letters, digits, '.', '_' or '-'")
    return name


def capture_random_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_random_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def to_model_range(x: torch.Tensor) -> torch.Tensor:
    return x.mul(2).sub(1)


def to_image_range(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x.add(1).div(2), nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)


def safe_image_array(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().float()
        if x.ndim == 3 and x.shape[0] in (1, 3):
            x = x.permute(1, 2, 0)
        x = x.numpy()
    return np.nan_to_num(np.asarray(x), nan=0.0, posinf=1.0, neginf=0.0).clip(0, 1)


def save_image(x: torch.Tensor | np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (safe_image_array(x) * 255).round().astype(np.uint8)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    Image.fromarray(arr).save(path)


def save_checkpoint(path: str | Path, state: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"Invalid checkpoint: {path}")
    return checkpoint


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
