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


DEMO_CSS = """
.gradio-container {
    max-width: 1800px !important;
}

/* Gradio uses object-fit: scale-down for output images, which leaves native
   64 x 64 results tiny. Fill the preview area while preserving aspect ratio. */
.sat-output-image .image-container,
.sat-output-image .image-frame,
.sat-output-image .image-container > button {
    width: 100% !important;
    height: 100% !important;
}

.sat-output-image .image-frame img {
    width: 100% !important;
    height: 100% !important;
    max-width: none !important;
    max-height: none !important;
    object-fit: contain !important;
}
"""


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
            raise gr.Error("請先上傳 RGB 影像，或從測試集中選擇一張影像。")
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
        values["notice"] = "此結果來自合成雲修復，無法還原厚雲遮蔽下的真實地理資訊。"
        values["device"] = str(device)
        return clean, cloud.cloudy, cloud.soft_mask, telea, restored, uncertainty / max(float(uncertainty.max()), 1e-12), values

    with gr.Blocks(title="SatCloudRestore 衛星影像修復") as demo:
        gr.Markdown(
            "# SatCloudRestore 衛星雲遮蔽影像修復\n"
            "本介面會將輸入影像縮放為 64×64，加入合成雲後比較 Telea 與擴散模型的修復結果。"
            "生成內容不代表厚雲遮蔽下的真實地理資訊。"
        )
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="numpy", label="上傳 RGB 影像（優先使用）", height=360)
                test_path = gr.Dropdown(test_choices, label="或選擇測試集影像", allow_custom_value=False)
            with gr.Column():
                coverage = gr.Slider(0.1, 0.7, value=0.3, step=0.05, label="雲層覆蓋率")
                opacity = gr.Slider(0.35, 1.0, value=0.8, step=0.05, label="雲層不透明度")
                steps = gr.Dropdown([25, 50], value=25, label="DDIM 取樣步數")
                samples = gr.Slider(1, 8, value=2, step=1, label="取樣次數")
                seed = gr.Number(value=42, precision=0, label="隨機種子")
                button = gr.Button("開始修復", variant="primary")
        gr.Markdown("## 輸入與合成結果")
        with gr.Row():
            clean_out = gr.Image(label="模型輸入（64×64）", height=420, elem_classes=["sat-output-image"])
            cloudy_out = gr.Image(label="合成雲影像", height=420, elem_classes=["sat-output-image"])
            mask_out = gr.Image(label="雲層遮罩", height=420, elem_classes=["sat-output-image"])
        gr.Markdown("## 修復結果")
        with gr.Row():
            telea_out = gr.Image(label="Telea 傳統修復", height=420, elem_classes=["sat-output-image"])
            restored_out = gr.Image(label="擴散模型修復", height=420, elem_classes=["sat-output-image"])
            uncertainty_out = gr.Image(label="不確定性代理", height=420, elem_classes=["sat-output-image"])
        metrics = gr.JSON(label="評估指標")
        button.click(restore, [image, test_path, coverage, opacity, steps, samples, seed], [clean_out, cloudy_out, mask_out, telea_out, restored_out, uncertainty_out, metrics])
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch SatCloudRestore Gradio app")
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--config", default="configs/quick.yaml")
    parser.add_argument("--device", default=None); parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    if not Path(args.checkpoint).is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}. Train a model before launching the app.")
    create_app(args.checkpoint, args.config, args.device).launch(share=args.share, css=DEMO_CSS)


if __name__ == "__main__":
    main()
