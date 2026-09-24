"""
face_stream.py — I3D Variant Face Stream
=========================================
Processes cropped face video clips using a lightweight 3-D convolutional
network (I3D variant) to capture facial expressions and mouth movements
that disambiguate near-minimal-pair glosses.

Input:  (B, T, 3, H, W) — face crop sequence (default 64×64)
Output: (B, T, embed_dim) — per-frame face embeddings
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# 3-D convolutional blocks
# ─────────────────────────────────────────────────────────────────────────────

class Conv3dBNReLU(nn.Module):
    """3-D convolution with batch norm and ReLU."""

    def __init__(self, in_ch: int, out_ch: int,
                 kernel: tuple = (3, 3, 3),
                 stride: tuple = (1, 1, 1),
                 padding: tuple = (1, 1, 1)):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel,
                      stride=stride, padding=padding, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class InceptionBlock3D(nn.Module):
    """
    Lightweight 3-D Inception block with four branches:
        1×1×1 conv
        3×3×3 conv (preceded by 1×1×1 bottleneck)
        1×1×3 + 3×3×1 (factorised spatial)
        1×1×1 projection from max-pool
    """

    def __init__(self, in_ch: int, out_channels: tuple):
        """
        Args:
            in_ch: Input channels.
            out_channels: (branch1, branch2_mid, branch2_out,
                           branch3_mid, branch3_out, branch4_out)
        """
        super().__init__()
        b1, b2m, b2o, b3m, b3o, b4o = out_channels

        # Branch 1: 1×1×1
        self.b1 = Conv3dBNReLU(in_ch, b1, (1, 1, 1), padding=(0, 0, 0))

        # Branch 2: 1×1×1 → 3×3×3
        self.b2 = nn.Sequential(
            Conv3dBNReLU(in_ch, b2m, (1, 1, 1), padding=(0, 0, 0)),
            Conv3dBNReLU(b2m,  b2o, (3, 3, 3)),
        )

        # Branch 3: 1×1×1 → 1×3×3 → 3×1×1 (factorised)
        self.b3 = nn.Sequential(
            Conv3dBNReLU(in_ch, b3m, (1, 1, 1), padding=(0, 0, 0)),
            Conv3dBNReLU(b3m, b3m, (1, 3, 3), padding=(0, 1, 1)),
            Conv3dBNReLU(b3m, b3o, (3, 1, 1), padding=(1, 0, 0)),
        )

        # Branch 4: max-pool → 1×1×1
        self.b4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            Conv3dBNReLU(in_ch, b4o, (1, 1, 1), padding=(0, 0, 0)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Face Stream
# ─────────────────────────────────────────────────────────────────────────────

class FaceStream(nn.Module):
    """
    I3D-variant face stream.

    Architecture:
        Conv3d stem → 2 Inception3D blocks → adaptive temporal pool
        → per-frame projection to embed_dim

    Args:
        embed_dim:  Output embedding dimension (default 256).
        input_size: Spatial size of face crops (default 64).
    """

    def __init__(self, embed_dim: int = 256, input_size: int = 64):
        super().__init__()
        self.embed_dim = embed_dim

        # Stem: spatial and temporal downsampling
        self.stem = nn.Sequential(
            Conv3dBNReLU(3, 64, (3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            Conv3dBNReLU(64, 64,  (1, 1, 1), padding=(0, 0, 0)),
            Conv3dBNReLU(64, 192, (3, 3, 3)),
            nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )

        # Two Inception blocks
        # out_channels: (b1, b2_mid, b2_out, b3_mid, b3_out, b4_out)
        self.inception1 = InceptionBlock3D(192, (64, 96, 128, 16, 32, 32))
        # output channels: 64+128+32+32 = 256
        self.inception2 = InceptionBlock3D(256, (128, 128, 192, 32, 96, 64))
        # output channels: 128+192+96+64 = 480

        # Spatial global average pool
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))

        # Per-frame projection
        self.project = nn.Sequential(
            nn.Linear(480, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, 3, H, W) face crop sequence

        Returns:
            (B, T, embed_dim) per-frame face embeddings
        """
        B, T, C, H, W = x.shape

        # Reshape to (B, C, T, H, W) for 3-D conv
        x3d = x.permute(0, 2, 1, 3, 4)   # (B, 3, T, H, W)

        out = self.stem(x3d)               # (B, 192, T', H', W')
        out = self.inception1(out)         # (B, 256, T', H', W')
        out = self.inception2(out)         # (B, 480, T', H', W')

        out = self.spatial_pool(out)       # (B, 480, T', 1, 1)
        out = out.squeeze(-1).squeeze(-1)  # (B, 480, T')
        out = out.permute(0, 2, 1)         # (B, T', 480)

        # Interpolate back to original T if needed
        if out.size(1) != T:
            out = out.permute(0, 2, 1)                              # (B, 480, T')
            out = nn.functional.interpolate(out, size=T, mode="linear",
                                            align_corners=False)    # (B, 480, T)
            out = out.permute(0, 2, 1)                              # (B, T, 480)

        return self.project(out)           # (B, T, embed_dim)
