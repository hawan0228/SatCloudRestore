import json
from pathlib import Path

from satcloudrestore.config import load_config


def test_pilot_config_and_colab_notebook_are_valid():
    config = load_config("configs/pilot_gpu.yaml")
    assert config["experiment_name"] == "pilot_gpu"
    assert config["training"]["gradient_accumulation"] == 4
    assert config["diffusion"]["ddim_steps"] == 25
    notebook = json.loads(Path("notebooks/SatCloudRestore_Colab.ipynb").read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert all(not cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code")


def test_resize_configs_preserve_quick_core_settings():
    legacy = load_config("configs/quick.yaml")
    resize = load_config("configs/quick_resize.yaml")
    pilot = load_config("configs/pilot_resize_gpu.yaml")
    assert resize["model"]["upsample_mode"] == "resize_conv"
    assert pilot["model"]["upsample_mode"] == "resize_conv"
    for key in ("base_channels", "channel_multipliers", "time_dim"):
        assert resize["model"][key] == legacy["model"][key]
    assert resize["diffusion"] == legacy["diffusion"]


def test_recommended_cosine_configs_have_near_zero_terminal_requirement():
    pilot = load_config("configs/pilot_cosine_resize_gpu.yaml")
    quick = load_config("configs/quick_cosine_resize.yaml")
    for config in (pilot, quick):
        assert config["diffusion"]["beta_schedule"] == "cosine"
        assert config["diffusion"]["require_near_zero_terminal"] is True
