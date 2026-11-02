"""TruthLens AI - Synthetic Sample Generator Utility
Uses trained DCGAN generator or initialized latent weights to synthesize
forensic testing samples and saves them to assets/generated/.
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure repository root is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import List
import torch
from PIL import Image

from ml.gan_model import DCGANGenerator, generate_synthetic_images
from ml.model_loader import CHECKPOINT_DIR

DEFAULT_GAN_WEIGHTS = os.path.join(CHECKPOINT_DIR, "dcgan_generator.pt")
GENERATED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "generated")


def generate_and_save_samples(
    num_samples: int = 6,
    latent_dim: int = 100,
    seed: int = 42,
    output_dir: str = GENERATED_DIR,
    weights_path: str = DEFAULT_GAN_WEIGHTS,
) -> List[str]:
    """Generates synthetic images and saves them to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    generator = DCGANGenerator(latent_dim=latent_dim)

    if os.path.exists(weights_path):
        try:
            ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
            if "generator_state_dict" in ckpt:
                generator.load_state_dict(ckpt["generator_state_dict"])
            print(f"Loaded DCGAN weights from: {weights_path}")
        except Exception as e:
            print(f"Notice: Initializing generator with base weights: {e}")
    else:
        print("Notice: No saved checkpoint found. Using initialized generator.")

    generator.eval()
    images = generate_synthetic_images(
        generator=generator,
        num_samples=num_samples,
        latent_dim=latent_dim,
        seed=seed,
    )

    saved_paths = []
    for idx, img in enumerate(images):
        save_path = os.path.join(output_dir, f"dcgan_synthetic_{seed}_{idx + 1:02d}.png")
        img.save(save_path)
        saved_paths.append(save_path)

    print(f"Successfully generated and saved {len(saved_paths)} synthetic samples to: {output_dir}")
    return saved_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic samples using DCGAN")
    parser.add_argument("--num-samples", type=int, default=4, help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random latent seed")
    args = parser.parse_args()

    generate_and_save_samples(num_samples=args.num_samples, seed=args.seed)
