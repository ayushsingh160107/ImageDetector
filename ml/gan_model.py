"""TruthLens AI - DCGAN (Deep Convolutional Generative Adversarial Network)
Implements Generator and Discriminator neural networks in PyTorch for synthetic image generation
and forensic research experimentation.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


def weights_init(m: nn.Module) -> None:
    """Initializes weights for Conv2d, ConvTranspose2d, and BatchNorm2d layers."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if getattr(m, "bias", None) is not None:
            nn.init.constant_(m.bias.data, 0)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class DCGANGenerator(nn.Module):
    """
    DCGAN Generator:
    Projects random latent vector z (shape: batch_size x latent_dim x 1 x 1)
    up to a synthetic RGB image (shape: batch_size x 3 x 64 x 64).
    """

    def __init__(self, latent_dim: int = 100, feature_maps: int = 64, channels: int = 3):
        super().__init__()
        self.latent_dim = latent_dim
        self.feature_maps = feature_maps
        self.channels = channels

        self.net = nn.Sequential(
            # Input is Z, going into convolution: (latent_dim) -> (feature_maps * 8) x 4 x 4
            nn.ConvTranspose2d(latent_dim, feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),
            # State size: (feature_maps * 8) x 4 x 4 -> (feature_maps * 4) x 8 x 8
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            # State size: (feature_maps * 4) x 8 x 8 -> (feature_maps * 2) x 16 x 16
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            # State size: (feature_maps * 2) x 16 x 16 -> (feature_maps) x 32 x 32
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            # State size: (feature_maps) x 32 x 32 -> (channels) x 64 x 64
            nn.ConvTranspose2d(feature_maps, channels, 4, 2, 1, bias=False),
            nn.Tanh(),  # Output range [-1, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
        return self.net(x)


class DCGANDiscriminator(nn.Module):
    """
    DCGAN Discriminator:
    Binary classifier evaluating whether an input image (3 x 64 x 64) is real (1) or synthetic (0).
    """

    def __init__(self, feature_maps: int = 64, channels: int = 3):
        super().__init__()
        self.feature_maps = feature_maps
        self.channels = channels

        self.net = nn.Sequential(
            # Input: (channels) x 64 x 64 -> (feature_maps) x 32 x 32
            nn.Conv2d(channels, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (feature_maps) x 32 x 32 -> (feature_maps * 2) x 16 x 16
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (feature_maps * 2) x 16 x 16 -> (feature_maps * 4) x 8 x 8
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (feature_maps * 4) x 8 x 8 -> (feature_maps * 8) x 4 x 4
            nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (feature_maps * 8) x 4 x 4 -> 1 x 1 x 1
            nn.Conv2d(feature_maps * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out.view(-1, 1).squeeze(1)


def generate_synthetic_images(
    generator: DCGANGenerator,
    num_samples: int = 4,
    latent_dim: int = 100,
    device: str = "cpu",
    seed: Optional[int] = None,
) -> List[Image.Image]:
    """
    Samples latent vectors and runs generator forward pass to produce PIL Images.
    Images are scaled from [-1, 1] to [0, 255] RGB.
    """
    if seed is not None:
        torch.manual_seed(seed)
    generator.eval()
    generator.to(device)

    with torch.no_grad():
        noise = torch.randn(num_samples, latent_dim, 1, 1, device=device)
        fake_tensors = generator(noise).detach().cpu()

    # Convert from [-1, 1] to [0, 255]
    fake_tensors = ((fake_tensors + 1.0) / 2.0).clamp(0, 1)
    images: List[Image.Image] = []
    for i in range(num_samples):
        img_np = fake_tensors[i].permute(1, 2, 0).numpy()
        img_uint8 = (img_np * 255).astype(np.uint8)
        images.append(Image.fromarray(img_uint8))

    return images
