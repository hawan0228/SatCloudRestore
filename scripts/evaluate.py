from __future__ import annotations

import argparse
import copy
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from satcloudrestore.config import load_config
from satcloudrestore.dataset import CloudDataset
from satcloudrestore.inference import load_model_bundle, telea_inpaint, tensor_to_rgb
from satcloudrestore.metrics import all_metrics, uncertainty_error_correlation
from satcloudrestore.sampling import ddim_sample
from satcloudrestore.uncertainty import sample_uncertainty
from satcloudrestore.utils import resolve_device, save_image, to_image_range, write_json
from satcloudrestore.visualization import save_comparison, save_gif


def parse_image_ids(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    return {item.strip() for value in values for item in value.split(",") if item.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Cloudy, Telea, and SatCloudRestore")
    parser.add_argument("--config", default="configs/quick.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--image-ids", nargs="+", default=None, help="Space- or comma-separated manifest image IDs")
    parser.add_argument("--coverage", nargs="+", type=float, default=None, help="Target coverage values, e.g. 0.1 0.3")
    parser.add_argument("--output-dir", default=None, help="Exact result directory for this evaluation")
    parser.add_argument("--device", default=None)
    parser.add_argument("--ddim-steps", type=int, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--save-samples", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = resolve_device(args.device)
    max_images = args.max_images if args.max_images is not None else args.limit
    if max_images is not None and max_images < 1:
        raise ValueError("--max-images must be positive")
    samples = int(args.samples if args.samples is not None else cfg["evaluation"]["uncertainty_samples"])
    ddim_steps = int(args.ddim_steps if args.ddim_steps is not None else cfg["diffusion"]["ddim_steps"])
    if samples < 1:
        raise ValueError("--samples must be positive")
    if not 1 <= ddim_steps <= int(cfg["diffusion"]["timesteps"]):
        raise ValueError("--ddim-steps must lie within the configured diffusion timesteps")
    coverages = set(args.coverage) if args.coverage else None
    if coverages and any(not 0 < value < 1 for value in coverages):
        raise ValueError("--coverage values must lie in (0, 1)")
    requested_ids = parse_image_ids(args.image_ids)

    model, diffusion, checkpoint = load_model_bundle(args.checkpoint, cfg, device)
    experiment_name = checkpoint.get("experiment_name") or Path(args.checkpoint).resolve().parent.name
    out = Path(args.output_dir) if args.output_dir else Path(cfg["paths"]["results_dir"]) / experiment_name
    out = out.resolve()
    if any((out / name).exists() for name in ("metrics.csv", "summary.json")):
        raise FileExistsError(f"Evaluation output already exists: {out}. Select a new --output-dir.")

    dataset = CloudDataset(cfg["paths"]["manifest"], "test", cfg["image_size"], cfg["cloud"], cfg["seed"])
    if requested_ids is not None:
        missing = sorted(requested_ids - {row["image_id"] for row in dataset.rows})
        if missing:
            raise ValueError(f"Requested image IDs not found: {', '.join(missing)}")
    groups = [float(value) for value in cfg["cloud"]["fixed_coverages"]]
    selected = []
    for index, row in enumerate(dataset.rows):
        target_coverage = groups[index % len(groups)]
        if requested_ids is not None and row["image_id"] not in requested_ids:
            continue
        if coverages is not None and not any(abs(target_coverage - value) < 1e-8 for value in coverages):
            continue
        selected.append(index)
        if max_images is not None and len(selected) >= max_images:
            break
    if not selected:
        raise ValueError("Evaluation filters selected no test images")
    loader = DataLoader(Subset(dataset, selected), batch_size=1, shuffle=False, num_workers=0)

    (out / "comparisons").mkdir(parents=True, exist_ok=False)
    (out / "denoising_gifs").mkdir()
    (out / "uncertainty_maps").mkdir()
    snapshot = copy.deepcopy(cfg)
    snapshot["evaluation_run"] = {"checkpoint": str(Path(args.checkpoint).resolve()),
                                  "max_images": max_images, "image_ids": sorted(requested_ids) if requested_ids else None,
                                  "coverage": sorted(coverages) if coverages else None,
                                  "ddim_steps": ddim_steps, "samples": samples}
    (out / "config.yaml").write_text(yaml.safe_dump(snapshot, sort_keys=False), encoding="utf-8")
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "none"
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}; torchvision: {torchvision.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}; device: {device}; GPU: {gpu_name}")
    print(f"Checkpoint: {Path(args.checkpoint).resolve()}")
    print(f"Result directory: {out}")
    print(f"Evaluation images: {len(selected)}; DDIM steps: {ddim_steps}; samples: {samples}")

    rows, correlations = [], []
    for index, batch in enumerate(tqdm(loader, total=len(selected), desc="evaluating")):
        clean_t, cloudy_t, mask_t = (batch[key].to(device) for key in ("clean", "cloudy", "mask"))
        clean = tensor_to_rgb(clean_t)
        cloudy = tensor_to_rgb(cloudy_t)
        binary = batch["binary_mask"][0, 0].numpy()
        telea_start = time.perf_counter()
        telea = telea_inpaint(cloudy, binary, cfg["evaluation"]["telea_radius"])
        telea_time = time.perf_counter() - telea_start
        start = time.perf_counter()
        if samples > 1:
            restored_t, variance_t, draws = sample_uncertainty(
                model, diffusion, cloudy_t, mask_t, ddim_steps, samples, int(batch["seed"][0])
            )
            uncertainty = variance_t[0, 0].cpu().numpy()
            intermediates = None
        else:
            restored_t, intermediates = ddim_sample(
                model, diffusion, cloudy_t, mask_t, ddim_steps, seed=int(batch["seed"][0]),
                return_intermediates=True,
            )
            uncertainty, draws = None, None
        inference_time = time.perf_counter() - start
        restored = tensor_to_rgb(restored_t)
        metadata = {
            "image_id": batch["image_id"][0], "class": int(batch["class_label"][0]),
            "target_coverage": float(batch["target_coverage"][0]),
            "actual_coverage": float(batch["actual_coverage"][0]), "opacity": float(batch["opacity"][0]),
            "seed": int(batch["seed"][0]),
        }
        for name, prediction, elapsed in (("cloudy", cloudy, 0.0), ("telea", telea, telea_time),
                                          ("diffusion", restored, inference_time)):
            rows.append({**metadata, "method": name, **all_metrics(prediction, clean, binary),
                         "inference_time": elapsed,
                         "telea_radius": cfg["evaluation"]["telea_radius"] if name == "telea" else None})
        if uncertainty is not None:
            correlation = uncertainty_error_correlation(uncertainty, np.abs(restored - clean).mean(-1), binary)
            correlations.append(correlation)
            normalized = uncertainty / max(float(uncertainty.max()), 1e-12)
            np.save(out / "uncertainty_maps" / f"{metadata['image_id']}.npy", uncertainty)
            if args.save_samples:
                sample_dir = out / "samples" / metadata["image_id"]
                for sample_index in range(draws.shape[0]):
                    save_image(to_image_range(draws[sample_index, 0]), sample_dir / f"sample_{sample_index:03d}.png")
        else:
            normalized = None
        save_comparison(out / "comparisons" / f"{metadata['image_id']}.png",
                        clean, cloudy, binary, telea, restored, normalized)
        if index == 0:
            if samples > 1:
                gif_frames = [tensor_to_rgb(draws[i]) for i in range(min(samples, 12))]
            else:
                gif_frames = [to_image_range(frame)[0].permute(1, 2, 0).numpy() for frame in intermediates]
            save_gif(gif_frames, out / "denoising_gifs" / "first_sample.gif")

    frame = pd.DataFrame(rows)
    frame.to_csv(out / "metrics.csv", index=False)
    aggregate = frame.groupby(["method", "target_coverage"], dropna=False).agg(
        count=("image_id", "count"), full_psnr=("full_psnr", "mean"), full_ssim=("full_ssim", "mean"),
        cloud_psnr=("cloud_psnr", "mean"), cloud_ssim=("cloud_ssim", "mean"),
        cloud_mae=("cloud_mae", "mean"), average_inference_time=("inference_time", "mean"),
    ).reset_index().to_dict(orient="records")
    valid_correlations = [value for value in correlations if value is not None]
    summary = {
        "experiment_name": experiment_name, "checkpoint": str(Path(args.checkpoint).resolve()),
        "result_dir": str(out), "evaluated_images": len(selected), "ddim_steps": ddim_steps,
        "uncertainty_samples": samples, "image_ids_filter": sorted(requested_ids) if requested_ids else None,
        "coverage_filter": sorted(coverages) if coverages else None,
        "uncertainty_error_spearman_mean": float(np.mean(valid_correlations)) if valid_correlations else None,
        "aggregate": aggregate,
        "warning": "Sample variance is an uncertainty proxy, not calibrated uncertainty.",
    }
    write_json(out / "summary.json", summary)
    print(f"Wrote {out / 'metrics.csv'} and {out / 'summary.json'}")


if __name__ == "__main__":
    main()
