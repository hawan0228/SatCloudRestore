from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from satcloudrestore.cloud_generator import SyntheticCloudGenerator
from satcloudrestore.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic synthetic cloud examples")
    parser.add_argument("--mode", choices=["clouds"], default="clouds")
    parser.add_argument("--config", default="configs/quick.yaml")
    parser.add_argument("--output", default="results/cloud_demo.png")
    args = parser.parse_args()
    config = load_config(args.config)
    size = int(config["image_size"])
    yy, xx = np.mgrid[0:size, 0:size]
    clean = np.stack((0.15 + 0.6 * xx / size, 0.25 + 0.55 * yy / size, 0.2 + 0.25 * np.sin(xx / 6)), axis=-1).clip(0, 1).astype(np.float32)
    coverages = config["cloud"]["fixed_coverages"]
    generator = SyntheticCloudGenerator(config["cloud"]["octaves"])
    fig, axes = plt.subplots(len(coverages), 3, figsize=(8, 2.5 * len(coverages)), dpi=140)
    for i, coverage in enumerate(coverages):
        result = generator.generate(clean, coverage=float(coverage), opacity=0.8, sigma=2.2, seed=int(config["seed"]) + i)
        for axis, image, title in zip(axes[i], (clean, result.soft_mask, result.cloudy),
                                      ("Clean", f"Mask target {coverage:.0%}\nactual {result.actual_coverage:.1%}", "Cloudy")):
            axis.imshow(image, cmap="gray" if image.ndim == 2 else None, vmin=0, vmax=1)
            axis.set_title(title); axis.axis("off")
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(output, bbox_inches="tight"); plt.close(fig)
    print(f"Saved cloud demo to {output}")


if __name__ == "__main__":
    main()
