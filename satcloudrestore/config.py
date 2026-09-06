from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping")
    required = {"paths", "seed", "image_size", "splits", "cloud", "model", "diffusion", "training", "evaluation"}
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"Missing configuration sections: {', '.join(missing)}")
    if config["image_size"] <= 0 or config["image_size"] % 4:
        raise ValueError("image_size must be a positive multiple of 4")
    if config["model"].get("in_channels") != 7 or config["model"].get("out_channels") != 3:
        raise ValueError("Conditional U-Net requires exactly 7 input and 3 output channels")
    if config["diffusion"].get("timesteps", 0) < 2:
        raise ValueError("diffusion.timesteps must be at least 2")
    if config["diffusion"].get("beta_schedule") not in {"linear", "cosine"}:
        raise ValueError("diffusion.beta_schedule must be linear or cosine")
    for name in ("train", "validation", "test"):
        if int(config["splits"].get(name, 0)) <= 0:
            raise ValueError(f"splits.{name} must be positive")
    for key in ("coverage_range", "opacity_range", "sigma_range"):
        values = config["cloud"].get(key)
        if not isinstance(values, list) or len(values) != 2 or values[0] > values[1]:
            raise ValueError(f"cloud.{key} must be [minimum, maximum]")
    if not 0 <= config["cloud"]["coverage_range"][0] <= config["cloud"]["coverage_range"][1] <= 1:
        raise ValueError("cloud coverage must lie in [0, 1]")
    training = config["training"]
    if int(training.get("batch_size", 0)) < 1 or int(training.get("num_workers", -1)) < 0:
        raise ValueError("training.batch_size must be positive and num_workers non-negative")
    if int(training.get("gradient_accumulation", 1)) < 1:
        raise ValueError("training.gradient_accumulation must be positive")
    if training.get("max_steps") is not None and int(training["max_steps"]) < 1:
        raise ValueError("training.max_steps must be positive when provided")
    return config
