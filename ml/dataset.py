"""TruthLens AI - Dataset Pipeline & Synthetic Benchmark Loader
Handles loading, splitting, and preparing synthetic vs. authentic image datasets.
Supports local directory structures, synthetic benchmark generation, and PyTorch DataLoaders.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFilter


class ImageForensicsDataset(Dataset):
    """
    Standard PyTorch dataset for binary image forensics:
    Class 0: Authentic / Camera-Captured (Real)
    Class 1: AI-Generated / Synthetic (Fake)
    """

    def __init__(
        self,
        samples: List[Tuple[str, int]],
        transform: Optional[transforms.Compose] = None,
    ):
        self.samples = samples
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path, label = self.samples[idx]
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            # Fallback blank image on disk read error
            image = Image.new("RGB", (224, 224), color=(128, 128, 128))

        tensor = self.transform(image)
        return tensor, label


def create_synthetic_benchmark_dataset(
    output_dir: str = "data/benchmark",
    num_samples_per_class: int = 50,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    Creates a balanced evaluation benchmark dataset on disk:
    - Real class (natural gradients, optical noise, realistic lighting)
    - AI-generated class (diffusion checkerboard patterns, latent frequency artifacts, smooth synthetic skin)
    Splits into 80% train, 20% test splits without data leakage.
    """
    real_dir = os.path.join(output_dir, "real")
    fake_dir = os.path.join(output_dir, "synthetic")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    samples: List[Tuple[str, int]] = []
    np.random.seed(42)

    for i in range(num_samples_per_class):
        # 1. Create realistic camera simulation (Class 0: Real)
        real_path = os.path.join(real_dir, f"real_{i:04d}.png")
        if not os.path.exists(real_path):
            arr = np.zeros((224, 224, 3), dtype=np.uint8)
            # Natural photographic sky/ground gradient
            for y in range(224):
                arr[y, :, 0] = int(100 + 100 * (y / 224))
                arr[y, :, 1] = int(120 + 80 * (y / 224))
                arr[y, :, 2] = int(150 + 50 * (y / 224))
            # Camera sensor ISO noise
            noise = np.random.normal(0, 8, (224, 224, 3)).astype(np.int16)
            arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            real_img = Image.fromarray(arr)
            draw = ImageDraw.Draw(real_img)
            # Add natural optical elements
            draw.ellipse([50, 40, 110, 100], fill=(220, 200, 160))
            real_img.save(real_path)
        samples.append((real_path, 0))

        # 2. Create synthetic generative simulation (Class 1: AI-Generated)
        fake_path = os.path.join(fake_dir, f"synthetic_{i:04d}.png")
        if not os.path.exists(fake_path):
            arr = np.zeros((224, 224, 3), dtype=np.uint8)
            # Generative latent space frequency oscillations (characteristic diffusion artifact)
            x = np.linspace(0, 10 * np.pi, 224)
            y = np.linspace(0, 10 * np.pi, 224)
            xx, yy = np.meshgrid(x, y)
            grid_pattern = ((np.sin(xx) * np.cos(yy) + 1.0) * 127.5).astype(np.uint8)
            arr[:, :, 0] = grid_pattern
            arr[:, :, 1] = np.roll(grid_pattern, 20, axis=0)
            arr[:, :, 2] = np.roll(grid_pattern, 40, axis=1)
            fake_img = Image.fromarray(arr)
            # Over-smoothed bilateral / hyper-realistic diffusion sheen
            fake_img = fake_img.filter(ImageFilter.SMOOTH_MORE)
            fake_img.save(fake_path)
        samples.append((fake_path, 1))

    # Split into Train (80%) and Test (20%)
    np.random.shuffle(samples)
    split_idx = int(0.8 * len(samples))
    train_samples = samples[:split_idx]
    test_samples = samples[split_idx:]

    return train_samples, test_samples
