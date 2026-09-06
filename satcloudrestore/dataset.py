from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .cloud_generator import SyntheticCloudGenerator
from .utils import to_model_range, write_json


def image_id(path: str) -> str:
    return hashlib.sha1(path.replace("\\", "/").encode()).hexdigest()[:16]


def stratified_split(samples: Sequence[tuple[str, int]], sizes: dict[str, int], seed: int = 42) -> list[dict]:
    total = sum(int(sizes[k]) for k in ("train", "validation", "test"))
    if len(samples) < total:
        raise ValueError(f"Dataset has {len(samples)} images but requested splits require {total}")
    paths = [str(p) for p, _ in samples]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate source paths detected")
    rng = np.random.default_rng(seed)
    labels = sorted({int(label) for _, label in samples})
    by_label = {label: [(str(p), int(y)) for p, y in samples if int(y) == label] for label in labels}
    for group in by_label.values():
        rng.shuffle(group)
    manifest: list[dict] = []
    for split in ("train", "validation", "test"):
        count = int(sizes[split])
        available = sum(len(group) for group in by_label.values())
        exact = {label: count * len(by_label[label]) / available for label in labels}
        allocation = {label: min(len(by_label[label]), int(exact[label])) for label in labels}
        remainder = count - sum(allocation.values())
        order = sorted(labels, key=lambda label: (exact[label] - allocation[label], len(by_label[label])), reverse=True)
        while remainder:
            progressed = False
            for label in order:
                if allocation[label] < len(by_label[label]) and remainder:
                    allocation[label] += 1; remainder -= 1; progressed = True
            if not progressed:
                raise ValueError("Unable to allocate requested stratified split")
        chosen = []
        for label in labels:
            chosen.extend(by_label[label][-allocation[label]:] if allocation[label] else [])
            if allocation[label]: del by_label[label][-allocation[label]:]
        rng.shuffle(chosen)
        for path, label in chosen:
            manifest.append({"path": path.replace("\\", "/"), "split": split, "class_label": label, "image_id": image_id(path)})
    validate_manifest(manifest, sizes)
    return manifest


def validate_manifest(manifest: Sequence[dict], sizes: dict[str, int] | None = None) -> None:
    paths, ids = {}, {}
    for row in manifest:
        for mapping, key in ((paths, "path"), (ids, "image_id")):
            value = row[key]
            if value in mapping:
                raise ValueError(f"Duplicate/data leakage: {key} {value} occurs more than once")
            mapping[value] = row["split"]
    if sizes:
        for split, expected in sizes.items():
            actual = sum(row["split"] == split for row in manifest)
            if actual != int(expected):
                raise ValueError(f"Split {split} contains {actual}, expected {expected}")


def save_manifest(path: str | Path, manifest: Sequence[dict]) -> None:
    write_json(path, list(manifest))


class CloudDataset(Dataset):
    def __init__(self, manifest_path: str | Path, split: str, image_size: int, cloud_config: dict, seed: int = 42):
        path = Path(manifest_path)
        if not path.is_file():
            raise FileNotFoundError(f"Split manifest not found: {path}. Run scripts/prepare_data.py first.")
        rows = json.loads(path.read_text(encoding="utf-8"))
        validate_manifest(rows)
        self.rows = [row for row in rows if row["split"] == split]
        if not self.rows:
            raise ValueError(f"Manifest contains no rows for split {split}")
        self.split, self.image_size, self.cfg, self.seed = split, image_size, cloud_config, seed
        self.generator = SyntheticCloudGenerator(int(cloud_config.get("octaves", 3)))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | float | int]:
        row = self.rows[index]
        with Image.open(row["path"]) as handle:
            image = np.asarray(handle.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BICUBIC), dtype=np.float32) / 255
        if self.split == "train":
            # DataLoader seeds torch independently per worker; deriving the NumPy
            # generator here makes runs reproducible while keeping each read random.
            rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, ()).item()))
            sample_seed = int(rng.integers(0, 2**31 - 1))
            coverage = float(rng.uniform(*self.cfg["coverage_range"]))
            opacity = float(rng.uniform(*self.cfg["opacity_range"]))
            sigma = float(rng.uniform(*self.cfg["sigma_range"]))
        else:
            sample_seed = self.seed + index * 1009
            rng = np.random.default_rng(sample_seed)
            groups = self.cfg.get("fixed_coverages", [0.1, 0.3, 0.5, 0.7])
            coverage = float(groups[index % len(groups)])
            opacity = float(rng.uniform(*self.cfg["opacity_range"]))
            sigma = float(rng.uniform(*self.cfg["sigma_range"]))
        cloud = self.generator.generate(image, coverage, opacity, sigma, sample_seed)
        clean = torch.from_numpy(cloud.clean).permute(2, 0, 1)
        cloudy = torch.from_numpy(cloud.cloudy).permute(2, 0, 1)
        return {"clean": to_model_range(clean), "cloudy": to_model_range(cloudy),
                "mask": torch.from_numpy(cloud.soft_mask)[None], "binary_mask": torch.from_numpy(cloud.binary_mask)[None],
                "image_id": row["image_id"], "class_label": int(row["class_label"]),
                "target_coverage": coverage, "actual_coverage": cloud.actual_coverage,
                "opacity": opacity, "seed": sample_seed}
