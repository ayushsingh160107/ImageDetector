"""TruthLens AI - DCGAN Training Script
Trains Deep Convolutional Generative Adversarial Network (DCGAN)
comprising DCGANGenerator and DCGANDiscriminator.
Saves generator checkpoint to checkpoints/dcgan_generator.pt.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Ensure repository root is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.dataset import ImageForensicsDataset, create_synthetic_benchmark_dataset
from ml.gan_model import DCGANDiscriminator, DCGANGenerator, weights_init
from ml.model_loader import CHECKPOINT_DIR

GAN_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "dcgan_generator.pt")


def train_gan(
    epochs: int = 5,
    batch_size: int = 8,
    latent_dim: int = 100,
    lr: float = 0.0002,
    beta1: float = 0.5,
    device: str = "cpu",
    output_path: str = GAN_CHECKPOINT_PATH,
) -> None:
    """Trains DCGAN Generator and Discriminator."""
    print("=== Starting TruthLens DCGAN Training ===")
    print(f"Epochs: {epochs} | Batch size: {batch_size} | Latent dim: {latent_dim} | Device: {device}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    train_samples, _ = create_synthetic_benchmark_dataset(num_samples_per_class=20)
    train_dataset = ImageForensicsDataset(train_samples)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    netG = DCGANGenerator(latent_dim=latent_dim).to(device)
    netD = DCGANDiscriminator().to(device)

    netG.apply(weights_init)
    netD.apply(weights_init)

    criterion = nn.BCELoss()
    optimizerD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        errD_total, errG_total = 0.0, 0.0
        batches = 0

        for batch_x, _ in dataloader:
            b_size = batch_x.size(0)
            # Resize batch to 64x64 for DCGAN standard resolution
            real_data = nn.functional.interpolate(batch_x, size=(64, 64), mode="bilinear", align_corners=False).to(device)
            # Scale to [-1, 1] range expected by DCGAN tanh output
            real_data = torch.clamp(real_data, -1.0, 1.0)

            # --- 1. Train Discriminator: maximize log(D(x)) + log(1 - D(G(z))) ---
            netD.zero_grad()
            real_labels = torch.full((b_size,), 1.0, dtype=torch.float, device=device)
            output_real = netD(real_data)
            errD_real = criterion(output_real, real_labels)
            errD_real.backward()

            # Train with fake batch
            noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
            fake_data = netG(noise)
            fake_labels = torch.full((b_size,), 0.0, dtype=torch.float, device=device)
            output_fake = netD(fake_data.detach())
            errD_fake = criterion(output_fake, fake_labels)
            errD_fake.backward()

            errD = errD_real + errD_fake
            optimizerD.step()

            # --- 2. Train Generator: maximize log(D(G(z))) ---
            netG.zero_grad()
            output_g = netD(fake_data)
            errG = criterion(output_g, real_labels)  # Generator wants discriminator to believe fakes are real
            errG.backward()
            optimizerG.step()

            errD_total += errD.item()
            errG_total += errG.item()
            batches += 1

        duration = time.time() - start_time
        avg_d_loss = errD_total / max(batches, 1)
        avg_g_loss = errG_total / max(batches, 1)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Loss_D: {avg_d_loss:.4f} | Loss_G: {avg_g_loss:.4f} | Time: {duration:.1f}s"
        )

    # Save Generator weights
    torch.save(
        {
            "generator_state_dict": netG.state_dict(),
            "latent_dim": latent_dim,
            "channels": 3,
            "feature_maps": 64,
        },
        output_path,
    )
    print(f"DCGAN Generator checkpoint successfully saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DCGAN Synthetic Image Generator")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--latent-dim", type=int, default=100, help="Latent vector dimension")
    args = parser.parse_args()

    train_gan(epochs=args.epochs, batch_size=args.batch_size, latent_dim=args.latent_dim)
