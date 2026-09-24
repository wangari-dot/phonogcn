"""
hand_stream.py — EfficientNet-Lite Hand Stream
===============================================
Processes cropped hand image sequences through a lightweight CNN backbone
to produce per-frame hand appearance embeddings.

The stream processes left and right hands independently, then concatenates
their embeddings before projecting to the target dimension.

Default backbone: EfficientNet-Lite0 (chosen for accuracy–speed trade-off
on edge devices; see Appendix A for backbone ablation results).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight depthwise-separable block (EfficientNet-Lite style)
# ─────────────────────────────────────────────────────────────────────────────

class DepthwiseSeparableConv(nn.Module):
    """Depthwise + pointwise conv block with BN and ReLU6."""

    def __init__(self, in_ch: int, out_ch: int,
                 stride: int = 1, expand_ratio: int = 1):
        super().__init__()
        hidden = in_ch * expand_ratio
        self.block = nn.Sequential(
            # Expand (pointwise)
            nn.Conv2d(in_ch, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
            # Depthwise
            nn.Conv2d(hidden, hidden, 3, stride=stride,
                      padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
            # Project (pointwise)
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.use_residual = (stride == 1 and in_ch == out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return out


class EfficientNetLite0(nn.Module):
    """
    Minimal EfficientNet-Lite0 implementation for hand feature extraction.

    Input:  (B, 3, H, W)       — RGB hand crop (default 112×112)
    Output: (B, embed_dim)     — appearance embedding
    """

    def __init__(self, embed_dim: int = 256, input_size: int = 112):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )

        # MBConv blocks (simplified EfficientNet-Lite0 structure)
        self.blocks = nn.Sequential(
            DepthwiseSeparableConv(32,  16,  stride=1, expand_ratio=1),
            DepthwiseSeparableConv(16,  24,  stride=2, expand_ratio=6),
            DepthwiseSeparableConv(24,  24,  stride=1, expand_ratio=6),
            DepthwiseSeparableConv(24,  40,  stride=2, expand_ratio=6),
            DepthwiseSeparableConv(40,  40,  stride=1, expand_ratio=6),
            DepthwiseSeparableConv(40,  80,  stride=2, expand_ratio=6),
            DepthwiseSeparableConv(80,  80,  stride=1, expand_ratio=6),
            DepthwiseSeparableConv(80,  112, stride=1, expand_ratio=6),
            DepthwiseSeparableConv(112, 112, stride=1, expand_ratio=6),
            DepthwiseSeparableConv(112, 192, stride=2, expand_ratio=6),
            DepthwiseSeparableConv(192, 192, stride=1, expand_ratio=6),
            DepthwiseSeparableConv(192, 320, stride=1, expand_ratio=6),
        )

        # Head
        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, 1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True),
        )
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.project = nn.Linear(1280, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.pool(x).flatten(1)
        return self.project(x)


# ─────────────────────────────────────────────────────────────────────────────
# Hand Stream
# ─────────────────────────────────────────────────────────────────────────────

class HandStream(nn.Module):
    """
    Dual-hand CNN stream producing per-frame hand appearance embeddings.

    Each hand is encoded separately; embeddings are concatenated then
    projected to `embed_dim`.

    Args:
        embed_dim:  Output embedding dimension per hand (default 256).
                    Final projected dim = embed_dim (both hands fused).
        backbone:   Backbone name.  Supported: 'efficientnet_lite0'.
                    (Extend _build_backbone() to add MobileNetV3, ResNet-18.)
    """

    def __init__(self,
                 embed_dim: int = 256,
                 backbone: str = "efficientnet_lite0"):
        super().__init__()
        self.embed_dim = embed_dim

        self.right_encoder = self._build_backbone(backbone, embed_dim)
        self.left_encoder  = self._build_backbone(backbone, embed_dim)

        # Fuse both hands
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _build_backbone(name: str, embed_dim: int) -> nn.Module:
        if name == "efficientnet_lite0":
            return EfficientNetLite0(embed_dim=embed_dim)
        raise ValueError(
            f"Unknown hand backbone: {name!r}. "
            "Supported: 'efficientnet_lite0'"
        )

    def forward(self,
                right_hand: torch.Tensor,
                left_hand: torch.Tensor) -> torch.Tensor:
        """
        Args:
            right_hand: (B, T, 3, H, W) right-hand crop sequence
            left_hand:  (B, T, 3, H, W) left-hand crop sequence

        Returns:
            (B, T, embed_dim) per-frame hand embeddings
        """
        B, T, C, H, W = right_hand.shape

        # Flatten temporal dimension for batch processing
        right_flat = right_hand.reshape(B * T, C, H, W)
        left_flat  = left_hand.reshape(B * T, C, H, W)

        r_emb = self.right_encoder(right_flat)    # (B*T, embed_dim)
        l_emb = self.left_encoder(left_flat)      # (B*T, embed_dim)

        combined = torch.cat([r_emb, l_emb], dim=-1)    # (B*T, embed_dim*2)
        fused = self.fusion(combined)                    # (B*T, embed_dim)

        return fused.reshape(B, T, self.embed_dim)
