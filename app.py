from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from satcloudrestore.cloud_generator import SyntheticCloudGenerator
from satcloudrestore.config import load_config
from satcloudrestore.inference import load_model_bundle, telea_inpaint, tensor_to_rgb
from satcloudrestore.metrics import all_metrics
from satcloudrestore.uncertainty import sample_uncertainty
from satcloudrestore.inference import rgb_to_tensor
from satcloudrestore.utils import resolve_device


def create_app(checkpoint: str, config_path: str = "configs/quick.yaml", device_name: str | None = None):
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Gradio is required: pip install -e .") from exc
    config = load_config(config_path)
    device = resolve_device(device_name)
    model, diffusion, _ = load_model_bundle(checkpoint, config, device)
    generator = SyntheticCloudGenerator(config["cloud"]["octaves"])
    size = int(config["image_size"])
    manifest_path = Path(config["paths"]["manifest"])
    test_choices = []
    if manifest_path.is_file():
        test_choices = [row["path"] for row in json.loads(manifest_path.read_text(encoding="utf-8")) if row.get("split") == "test"][:100]

    def restore(image, test_path, coverage, opacity, steps, samples, seed):
        if image is None and not test_path:
            raise gr.Error("Upload an RGB image or select a test-set image first.")
        array = np.asarray(image) if image is not None else np.asarray(Image.open(test_path).convert("RGB"))
        if array.dtype != np.uint8:
            array = (np.nan_to_num(array).clip(0, 1) * 255).astype(np.uint8) if array.max() <= 1 else np.nan_to_num(array).clip(0, 255).astype(np.uint8)
        clean = np.asarray(Image.fromarray(array).convert("RGB").resize((size, size), Image.Resampling.BICUBIC), dtype=np.float32) / 255
        cloud = generator.generate(clean, float(coverage), float(opacity), seed=int(seed))
        cloudy_t = rgb_to_tensor(cloud.cloudy, device)
        mask_t = torch.from_numpy(cloud.soft_mask)[None, None].to(device)
        mean, variance, _ = sample_uncertainty(model, diffusion, cloudy_t, mask_t, int(steps), int(samples), int(seed))
        restored = tensor_to_rgb(mean); uncertainty = variance[0, 0].cpu().numpy()
        telea = telea_inpaint(cloud.cloudy, cloud.binary_mask, config["evaluation"]["telea_radius"])
        values = all_metrics(restored, clean, cloud.binary_mask)
        values["notice"] = "Synthetic-cloud educational estimate; it cannot recover factual geography hidden by opaque cloud."
        values["device"] = str(device)
        return clean, cloud.cloudy, cloud.soft_mask, telea, restored, uncertainty / max(float(uncertainty.max()), 1e-12), values

    with gr.Blocks(title="SatCloudRestore") as demo:
        gr.Markdown("# SatCloudRestore\nEducational synthetic-cloud restoration. Uploaded images are resized to 64×64; generated content is not factual recovery of obscured geography.")
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="numpy", label="Upload RGB image (takes priority)")
                test_path = gr.Dropdown(test_choices, label="Or select a test-set image", allow_custom_value=False)
            with gr.Column():
                coverage = gr.Slider(0.1, 0.7, value=0.3, step=0.05, label="Cloud coverage")
                opacity = gr.Slider(0.35, 1.0, value=0.8, step=0.05, label="Opacity")
                steps = gr.Dropdown([25, 50], value=25, label="DDIM steps")
                samples = gr.Slider(1, 8, value=2, step=1, label="Sample count")
                seed = gr.Number(value=42, precision=0, label="Seed")
                button = gr.Button("Restore", variant="primary")
        with gr.Row():
            clean_out = gr.Image(label="Clean (resized)"); cloudy_out = gr.Image(label="Cloudy"); mask_out = gr.Image(label="Cloud mask")
        with gr.Row():
            telea_out = gr.Image(label="Telea"); restored_out = gr.Image(label="Diffusion restoration"); uncertainty_out = gr.Image(label="Uncertainty proxy")
        metrics = gr.JSON(label="Metrics")
        button.click(restore, [image, test_path, coverage, opacity, steps, samples, seed], [clean_out, cloudy_out, mask_out, telea_out, restored_out, uncertainty_out, metrics])
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch SatCloudRestore Gradio app")
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--config", default="configs/quick.yaml")
    parser.add_argument("--device", default=None); parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    if not Path(args.checkpoint).is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}. Train a model before launching the app.")
    create_app(args.checkpoint, args.config, args.device).launch(share=args.share)


if __name__ == "__main__":
    main()
