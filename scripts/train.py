from __future__ import annotations

import argparse
import copy
import json
import platform
import sys
from pathlib import Path

import torch
import torchvision
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from satcloudrestore.config import load_config
from satcloudrestore.dataset import CloudDataset
from satcloudrestore.diffusion import GaussianDiffusion, weighted_noise_loss
from satcloudrestore.model import build_model
from satcloudrestore.sampling import ddim_sample
from satcloudrestore.utils import (
    capture_random_state,
    load_checkpoint,
    resolve_device,
    restore_random_state,
    save_checkpoint,
    save_image,
    seed_everything,
    to_image_range,
    validate_experiment_name,
)


def update_ema(ema, model, decay: float) -> None:
    with torch.no_grad():
        for target, source in zip(ema.parameters(), model.parameters()):
            target.mul_(decay).add_(source, alpha=1 - decay)
        for target, source in zip(ema.buffers(), model.buffers()):
            target.copy_(source)


def assert_resume_compatible(saved: dict, current: dict, experiment_name: str) -> None:
    if saved.get("experiment_name") != experiment_name:
        raise ValueError(
            f"Resume experiment mismatch: checkpoint={saved.get('experiment_name')!r}, requested={experiment_name!r}"
        )
    saved_cfg = saved.get("config")
    if not isinstance(saved_cfg, dict):
        raise ValueError("Resume checkpoint has no valid config snapshot")
    for key in ("image_size", "normalization", "model", "diffusion"):
        if saved_cfg.get(key) != current.get(key):
            raise ValueError(f"Resume config mismatch in '{key}'")


def append_log(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SatCloudRestore")
    parser.add_argument("--config", default="configs/quick.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--output-dir", default=None, help="Exact checkpoint directory for this experiment")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation", type=int, default=None)
    parser.add_argument("--overfit-batch", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum optimizer steps, not micro-batches")
    args = parser.parse_args()

    cfg = copy.deepcopy(load_config(args.config))
    tc = cfg["training"]
    for argument, key in ((args.num_workers, "num_workers"), (args.batch_size, "batch_size"),
                          (args.gradient_accumulation, "gradient_accumulation")):
        if argument is not None:
            tc[key] = argument
    if args.max_steps is not None:
        tc["max_steps"] = args.max_steps
    max_steps = tc.get("max_steps")
    if tc["batch_size"] < 1 or tc["num_workers"] < 0 or tc.get("gradient_accumulation", 1) < 1:
        raise ValueError("batch size and gradient accumulation must be positive; num workers must be non-negative")
    if max_steps is not None and int(max_steps) < 1:
        raise ValueError("--max-steps must be positive")

    experiment_name = validate_experiment_name(args.experiment_name or cfg.get("experiment_name") or Path(args.config).stem)
    cfg["experiment_name"] = experiment_name
    checkpoint_root = Path(cfg["paths"]["checkpoint_dir"])
    output = Path(args.output_dir) if args.output_dir else checkpoint_root / experiment_name
    output = output.resolve()
    result_dir = (Path(cfg["paths"]["results_dir"]) / experiment_name).resolve()
    resume_state = None
    if args.resume:
        resume_path = Path(args.resume).resolve()
        if output != resume_path.parent and args.output_dir:
            raise ValueError("--output-dir must be the resume checkpoint's parent directory")
        if not args.output_dir:
            output = resume_path.parent
        resume_state = load_checkpoint(resume_path, "cpu")
        assert_resume_compatible(resume_state, cfg, experiment_name)
    elif any((output / name).exists() for name in ("latest.pt", "best.pt", "config.yaml", "training_log.jsonl")):
        raise FileExistsError(f"Experiment directory already contains a run: {output}. Use --resume or a new experiment name.")

    device = resolve_device(args.device)
    seed_everything(int(cfg["seed"]))
    train_set = CloudDataset(cfg["paths"]["manifest"], "train", cfg["image_size"], cfg["cloud"], cfg["seed"])
    val_set = CloudDataset(cfg["paths"]["manifest"], "validation", cfg["image_size"], cfg["cloud"], cfg["seed"])
    if args.overfit_batch:
        train_set = [train_set[i] for i in range(min(4, len(train_set)))]
        val_set = [val_set[i] for i in range(min(4, len(val_set)))]
    workers = 0 if args.overfit_batch else int(tc["num_workers"])
    batch_size = min(int(tc["batch_size"]), len(train_set))
    accumulation = int(tc.get("gradient_accumulation", 1))
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=not args.overfit_batch,
                        num_workers=workers, pin_memory=device.type == "cuda", persistent_workers=workers > 0)
    val_loader = DataLoader(val_set, batch_size=min(batch_size, len(val_set)), shuffle=False,
                            num_workers=workers, pin_memory=device.type == "cuda", persistent_workers=workers > 0)
    model = build_model(cfg["model"]).to(device)
    ema = copy.deepcopy(model).eval()
    diffusion = GaussianDiffusion(cfg["diffusion"]["timesteps"], cfg["diffusion"]["beta_schedule"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc["learning_rate"], weight_decay=tc["weight_decay"])
    total_epochs = int(tc["epochs"])
    if args.overfit_batch and max_steps:
        total_epochs = max(total_epochs, int(max_steps))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_epochs))
    amp_enabled = bool(tc["amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    start_epoch, global_step, best = 0, 0, float("inf")
    if resume_state:
        model.load_state_dict(resume_state["model"])
        ema.load_state_dict(resume_state["ema"])
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        if resume_state.get("scaler"):
            scaler.load_state_dict(resume_state["scaler"])
        start_epoch = int(resume_state["epoch"]) + 1
        global_step = int(resume_state["global_step"])
        best = float(resume_state["best_validation_loss"])
        restore_random_state(resume_state.get("random_state"))

    output.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "none"
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}; torchvision: {torchvision.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}; device: {device}; GPU: {gpu_name}")
    print(f"Mixed precision: {amp_enabled}")
    print(f"Batch size: {batch_size}; gradient accumulation: {accumulation}; effective batch size: {batch_size * accumulation}")
    print(f"Trainable parameters: {parameter_count:,}")
    print(f"Images: train={len(train_set)}, validation={len(val_set)}")
    print(f"Maximum optimizer steps: {max_steps if max_steps is not None else 'unlimited'}")
    print(f"Checkpoint directory: {output}")
    print(f"Result directory: {result_dir}")

    stop = global_step >= int(max_steps) if max_steps else False
    for epoch in range(start_epoch, total_epochs):
        if stop:
            break
        model.train()
        running = 0.0
        micro_batches = 0
        optimizer.zero_grad(set_to_none=True)
        iterator = tqdm(loader, desc=f"epoch {epoch + 1}/{total_epochs}")
        for micro_index, batch in enumerate(iterator):
            clean, cloudy, mask = (batch[key].to(device, non_blocking=True) for key in ("clean", "cloudy", "mask"))
            t = torch.randint(0, diffusion.timesteps, (clean.shape[0],), device=device)
            noise = torch.randn_like(clean)
            xt = diffusion.q_sample(clean, t, noise)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                prediction = model(torch.cat((xt, cloudy, mask), 1), t)
                raw_loss = weighted_noise_loss(prediction, noise, mask, tc["mask_weight"])
                loss = raw_loss / accumulation
            scaler.scale(loss).backward()
            loss_value = raw_loss.detach().item()
            running += loss_value
            micro_batches += 1
            last_micro_batch = micro_index + 1 == len(loader)
            if (micro_index + 1) % accumulation == 0 or last_micro_batch:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                update_ema(ema, model, tc["ema_decay"])
                global_step += 1
                if max_steps and global_step >= int(max_steps):
                    stop = True
            iterator.set_postfix(loss=f"{loss_value:.4f}", step=global_step)
            if stop:
                break

        model.eval()
        val_total, val_count = 0.0, 0
        with torch.no_grad():
            for index, batch in enumerate(val_loader):
                clean, cloudy, mask = (batch[key].to(device, non_blocking=True) for key in ("clean", "cloudy", "mask"))
                t = torch.full((clean.shape[0],), diffusion.timesteps // 2, device=device, dtype=torch.long)
                generator = torch.Generator(device=device).manual_seed(cfg["seed"] + index)
                noise = torch.randn(clean.shape, device=device, generator=generator)
                prediction = ema(torch.cat((diffusion.q_sample(clean, t, noise), cloudy, mask), 1), t)
                value = weighted_noise_loss(prediction, noise, mask, tc["mask_weight"])
                val_total += value.detach().item() * clean.shape[0]
                val_count += clean.shape[0]
                if args.overfit_batch or index >= 3:
                    break
        train_loss = running / max(1, micro_batches)
        val_loss = val_total / max(1, val_count)
        scheduler.step()
        improved = val_loss < best
        best = min(best, val_loss)
        log_row = {"epoch": epoch, "global_step": global_step, "train_loss": train_loss,
                   "validation_loss": val_loss, "learning_rate": optimizer.param_groups[0]["lr"]}
        append_log(output / "training_log.jsonl", log_row)
        state = {
            "model": model.state_dict(), "ema": ema.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict() if amp_enabled else None,
            "epoch": epoch, "global_step": global_step, "best_validation_loss": best,
            "config": cfg, "experiment_name": experiment_name, "seed": cfg["seed"],
            "random_state": capture_random_state(), "checkpoint_dir": str(output), "result_dir": str(result_dir),
        }
        save_checkpoint(output / "latest.pt", state)
        if (epoch + 1) % int(tc["checkpoint_frequency"]) == 0:
            save_checkpoint(output / f"epoch_{epoch + 1:03d}.pt", state)
        if improved:
            save_checkpoint(output / "best.pt", state)
        if (epoch + 1) % int(tc["preview_frequency"]) == 0 or args.overfit_batch:
            preview_steps = min(10 if args.overfit_batch else cfg["diffusion"]["ddim_steps"], diffusion.timesteps)
            preview = ddim_sample(ema, diffusion, cloudy[:1], mask[:1], steps=preview_steps, seed=cfg["seed"])
            save_image(to_image_range(preview[0]), output / f"preview_epoch_{epoch + 1:03d}.png")
        print(f"train_loss={train_loss:.6f} validation_loss={val_loss:.6f} global_step={global_step}")


if __name__ == "__main__":
    main()
