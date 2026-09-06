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
