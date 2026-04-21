"""Lightweight Mask Autoencoder for Joint Image-Mask Co-Generation.

This module provides a small convolutional autoencoder that maps
10-channel binary anatomy masks (in [-1, 1]) to a compact latent
representation of shape (4, 32, 32), matching the image VAE latent
geometry for architectural compatibility with OmniGen.

Architecture:
  MaskEncoder: (10, 256, 256) -> (4, 32, 32)
    - 3 strided conv blocks: 256->128->64->32 spatial
    - Channel progression: 10 -> 32 -> 64 -> 4

  MaskDecoder: (4, 32, 32) -> (10, 256, 256)
    - 3 upsample + conv blocks: 32->64->128->256 spatial
    - Channel progression: 4 -> 64 -> 32 -> 10
    - Final tanh activation for [-1, 1] output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Simple residual block with GroupNorm."""
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class MaskEncoder(nn.Module):
    """Encode 10-channel masks from (10, 256, 256) to latent (4, 32, 32)."""
    def __init__(self, in_channels: int = 10, latent_channels: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            # (10, 256, 256) -> (32, 128, 128)
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            ResBlock(32),
            # (32, 128, 128) -> (64, 64, 64)
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            ResBlock(64),
            # (64, 64, 64) -> (64, 32, 32)
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            ResBlock(64),
            # (64, 32, 32) -> (4, 32, 32)
            nn.Conv2d(64, latent_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 10, 256, 256) in [-1, 1] -> (B, 4, 32, 32)"""
        return self.net(x)


class MaskDecoder(nn.Module):
    """Decode latent (4, 32, 32) back to 10-channel masks (10, 256, 256)."""
    def __init__(self, latent_channels: int = 4, out_channels: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            # (4, 32, 32) -> (64, 32, 32)
            nn.Conv2d(latent_channels, 64, kernel_size=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            ResBlock(64),
            # (64, 32, 32) -> (64, 64, 64)
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            ResBlock(64),
            # (64, 64, 64) -> (32, 128, 128)
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            ResBlock(32),
            # (32, 128, 128) -> (10, 256, 256)
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, 4, 32, 32) -> (B, 10, 256, 256) in [-1, 1]"""
        return self.net(z)


class MaskAutoencoder(nn.Module):
    """Combined mask encoder-decoder for pretraining and joint training."""
    def __init__(self, in_channels: int = 10, latent_channels: int = 4):
        super().__init__()
        self.encoder = MaskEncoder(in_channels, latent_channels)
        self.decoder = MaskDecoder(latent_channels, in_channels)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode mask to latent space."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to mask space."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> dict:
        """Full forward pass for pretraining.

        Args:
            x: (B, 10, 256, 256) binary masks mapped to [-1, 1]

        Returns:
            dict with 'z_mask' (latent), 'x_recon' (reconstruction), 'loss'
        """
        z = self.encoder(x)
        x_recon = self.decoder(z)
        loss = F.mse_loss(x_recon, x)
        return {
            'z_mask': z,
            'x_recon': x_recon,
            'loss': loss,
        }
