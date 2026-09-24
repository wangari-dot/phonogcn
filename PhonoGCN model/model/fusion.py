"""
fusion.py — Cross-Modal Fusion Transformer
===========================================
Fuses hand, body, and face stream embeddings using a cross-modal
transformer (§3.2.3).

Architecture:
    1. Per-stream linear projection → d_model
    2. Concatenate streams along the joint/feature axis (not time)
    3. Add positional encoding
    4. N-layer transformer encoder (self-attention across streams)
    5. Output: (B, T, d_model) fused sequence for the translation head

Hyperparameters (from config.py ModelConfig):
    d_model = 512
    n_heads  = 4
    n_layers = 2
    d_ffn    = 2048
"""

import math
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Positional encoding
# ─────────────────────────────────────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model)"""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ─────────────────────────────────────────────────────────────────────────────
# Fusion transformer
# ─────────────────────────────────────────────────────────────────────────────

class FusionTransformer(nn.Module):
    """
    Cross-modal fusion transformer.

    Each modality is projected to d_model and concatenated along the
    feature dimension to form a unified token sequence, then processed
    by a standard transformer encoder.

    Args:
        hand_dim:   Input dimension of the hand stream (default 256).
        body_dim:   Input dimension of the body stream (default 256).
        face_dim:   Input dimension of the face stream (default 256).
        d_model:    Transformer model dimension (default 512).
        n_heads:    Number of attention heads (default 4).
        n_layers:   Number of transformer layers (default 2).
        d_ffn:      Feed-forward network inner dimension (default 2048).
        dropout:    Dropout probability (default 0.1).
    """

    def __init__(self,
                 hand_dim: int = 256,
                 body_dim: int = 256,
                 face_dim: int = 256,
                 d_model: int = 512,
                 n_heads: int = 4,
                 n_layers: int = 2,
                 d_ffn: int = 2048,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # Per-modality projections
        self.hand_proj = nn.Linear(hand_dim, d_model)
        self.body_proj = nn.Linear(body_dim, d_model)
        self.face_proj = nn.Linear(face_dim, d_model)

        # Modality-type embeddings (learned)
        self.modality_embed = nn.Embedding(3, d_model)  # 0=hand, 1=body, 2=face

        self.pos_enc = SinusoidalPositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ffn,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
        )

        # Output projection: 3 × d_model → d_model after stream concatenation
        self.out_proj = nn.Linear(3 * d_model, d_model)

    def forward(self,
                hand_feat: torch.Tensor,
                body_feat: torch.Tensor,
                face_feat: torch.Tensor,
                src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            hand_feat: (B, T, hand_dim)
            body_feat: (B, T, body_dim)
            face_feat: (B, T, face_dim)
            src_key_padding_mask: (B, T) boolean mask; True = padded frame.

        Returns:
            fused: (B, T, d_model)
        """
        B, T, _ = hand_feat.shape

        # Project each stream
        h = self.hand_proj(hand_feat)   # (B, T, d_model)
        b = self.body_proj(body_feat)   # (B, T, d_model)
        f = self.face_proj(face_feat)   # (B, T, d_model)

        # Add modality-type embeddings
        mod_ids = torch.arange(3, device=hand_feat.device)
        h = h + self.modality_embed(mod_ids[0])
        b = b + self.modality_embed(mod_ids[1])
        f = f + self.modality_embed(mod_ids[2])

        # Positional encoding (applied per-stream then fused)
        h = self.pos_enc(h)
        b = self.pos_enc(b)
        f = self.pos_enc(f)

        # Concatenate streams along feature dim, then feed through transformer
        # Strategy: interleave tokens from three modalities at each time step
        # Shape: (B, T, 3*d_model) → (B, T, d_model) after out_proj
        combined = torch.cat([h, b, f], dim=-1)           # (B, T, 3*d_model)
        combined = self.out_proj(combined)                 # (B, T, d_model)

        # Transformer encoder — attends over time
        fused = self.transformer(
            combined,
            src_key_padding_mask=src_key_padding_mask,
        )  # (B, T, d_model)

        return fused
