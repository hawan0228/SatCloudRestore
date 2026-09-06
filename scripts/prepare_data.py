from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from satcloudrestore.config import load_config
from satcloudrestore.dataset import save_manifest, stratified_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Download EuroSAT and create leakage-free image splits")
    parser.add_argument("--config", default="configs/quick.yaml")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    try:
        from torchvision.datasets import EuroSAT
    except Exception as exc:
        raise RuntimeError("torchvision EuroSAT loader is unavailable; install compatible torch/torchvision versions") from exc
    raw = Path(config["paths"]["raw_dir"])
    dataset = EuroSAT(root=raw, download=not args.no_download)
    try:
        samples = [(str(Path(path).resolve().relative_to(ROOT.resolve())), int(label)) for path, label in dataset.samples]
    except ValueError as exc:
        raise ValueError("EuroSAT must be stored under the project root so manifest paths remain relative") from exc
    manifest = stratified_split(samples, config["splits"], int(config["seed"]))
    save_manifest(config["paths"]["manifest"], manifest)
    print(f"Wrote {len(manifest)} rows to {config['paths']['manifest']}")


if __name__ == "__main__":
    main()
