"""Short local train/resume/subset-evaluation smoke test.

The generated checkpoint receives only three optimizer steps. Its metrics are
pipeline diagnostics, never model-performance results.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from satcloudrestore.dataset import image_id


def next_run_directory(base: Path) -> Path:
    for index in range(10000):
        candidate = base / f"run_{index:03d}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"No available smoke run directory under {base}")


def run(command: list[str], expect_success: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    print(result.stdout, end="")
    if result.stderr and expect_success:
        print(result.stderr, file=sys.stderr, end="")
    if expect_success and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command)
    if not expect_success and result.returncode == 0:
        raise AssertionError("Command unexpectedly succeeded; overwrite protection failed")
    if not expect_success:
        if "already contains a run" not in result.stderr:
            raise AssertionError(f"Unexpected failure while checking overwrite protection: {result.stderr}")
        print("Overwrite protection: passed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU-ready CLI smoke test using local synthetic fixtures")
    parser.add_argument("--output", default="results/gpu_ready_smoke")
    args = parser.parse_args()
    root = next_run_directory(Path(args.output).resolve())
    images = root / "images"
    images.mkdir(parents=True)
    split_names = ("train",) * 4 + ("validation",) * 2 + ("test",) * 4
    rows = []
    for index, split in enumerate(split_names):
        yy, xx = np.mgrid[0:16, 0:16]
        array = np.stack(((xx + index) / 26, (yy + index) / 26,
                          0.3 + 0.2 * np.sin((xx + yy + index) / 3)), -1).clip(0, 1)
        path = (images / f"fixture_{index}.png").resolve()
        Image.fromarray((array * 255).astype(np.uint8)).save(path)
        rows.append({"path": str(path).replace("\\", "/"), "split": split,
                     "class_label": index % 2, "image_id": image_id(str(path))})
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    config = {
        "paths": {"data_dir": str(root), "raw_dir": str(root), "manifest": str(manifest),
                  "checkpoint_dir": str(root / "checkpoints"), "results_dir": str(root / "evaluation")},
        "seed": 42, "experiment_name": "smoke_train", "image_size": 16,
        "splits": {"train": 4, "validation": 2, "test": 4},
        "cloud": {"coverage_range": [0.1, 0.6], "opacity_range": [0.35, 1.0],
                  "sigma_range": [1.0, 2.0], "octaves": 2,
                  "fixed_coverages": [0.1, 0.3, 0.5, 0.7]},
        "normalization": "minus_one_one",
        "model": {"in_channels": 7, "out_channels": 3, "base_channels": 8,
                  "channel_multipliers": [1, 2], "time_dim": 16},
        "diffusion": {"timesteps": 4, "beta_schedule": "linear", "ddim_steps": 2},
        "training": {"batch_size": 1, "gradient_accumulation": 2, "epochs": 5,
                     "learning_rate": 0.0002, "weight_decay": 0.0001, "mask_weight": 4.0,
                     "ema_decay": 0.995, "amp": False, "grad_clip": 1.0,
                     "checkpoint_frequency": 1, "preview_frequency": 1, "num_workers": 0},
        "evaluation": {"batch_size": 1, "uncertainty_samples": 2, "telea_radius": 3},
    }
    config_path = root / "smoke.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    checkpoint_dir = root / "checkpoints" / "smoke_train"
    train = [sys.executable, str(ROOT / "scripts" / "train.py"), "--config", str(config_path),
             "--device", "cpu", "--experiment-name", "smoke_train", "--output-dir", str(checkpoint_dir),
             "--num-workers", "0", "--batch-size", "1", "--gradient-accumulation", "2",
             "--overfit-batch", "--max-steps", "2"]
    run(train)
    run(train, expect_success=False)  # Existing run must never be silently overwritten.
    run(train[:-1] + ["3", "--resume", str(checkpoint_dir / "latest.pt")])

    primary = root / "evaluation" / "primary_subset"
    run([sys.executable, str(ROOT / "scripts" / "evaluate.py"), "--config", str(config_path),
         "--checkpoint", str(checkpoint_dir / "best.pt"), "--device", "cpu", "--output-dir", str(primary),
         "--max-images", "2", "--coverage", "0.1", "0.3", "--ddim-steps", "2", "--samples", "1"])
    first_test_id = next(row["image_id"] for row in rows if row["split"] == "test")
    uncertainty = root / "evaluation" / "uncertainty_subset"
    run([sys.executable, str(ROOT / "scripts" / "evaluate.py"), "--config", str(config_path),
         "--checkpoint", str(checkpoint_dir / "best.pt"), "--device", "cpu", "--output-dir", str(uncertainty),
         "--image-ids", first_test_id, "--coverage", "0.1", "--ddim-steps", "2", "--samples", "2",
         "--save-samples"])
    required = [checkpoint_dir / "best.pt", checkpoint_dir / "latest.pt", checkpoint_dir / "config.yaml",
                checkpoint_dir / "training_log.jsonl", primary / "metrics.csv", primary / "summary.json",
                uncertainty / "metrics.csv", uncertainty / "summary.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Smoke artifacts missing: {missing}")
    print(f"GPU-ready smoke pipeline passed: {root}")
    print("WARNING: the three-step checkpoint and metrics are diagnostics only.")


if __name__ == "__main__":
    main()
