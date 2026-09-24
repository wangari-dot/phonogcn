"""
body_stream.py — ST-GCN + PAM Body Stream
==========================================
Processes skeletal sequences through a 2-layer Spatial-Temporal GCN whose
adjacency matrix is augmented by the Phonological Adjacency Matrix (PAM).

Input:  (B, T, J, 3)  — batch of skeleton sequences
Output: (B, T, embed_dim)  — per-frame body embeddings
"""

import torch
import torch.nn as nn

from .pam import PAM, build_linguistic_adjacency
from pipeline.clustering import build_body_adjacency  # reuse normalised adj


class STGCNLayer(nn.Module):
    """
    One Spatial-Temporal GCN layer with batch norm and residual connection.

    Spatial step:  graph conv over joints using the PAM-augmented adjacency.
    Temporal step: depthwise 1-D conv over the time axis.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 temporal_kernel: int = 9,
                 dropout: float = 0.2):
        super().__init__()

        self.spatial_fc  = nn.Linear(in_channels, out_channels)
        self.temporal_conv = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=temporal_kernel,
            padding=temporal_kernel // 2,
            groups=out_channels,   # depthwise
        )
        self.bn   = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=dropout)

        self.residual = (nn.Linear(in_channels, out_channels)
                         if in_channels != out_channels else nn.Identity())

    def forward(self, x: torch.Tensor,
                adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   (B, T, J, C_in)
            adj: (J, J) effective adjacency matrix

        Returns:
            (B, T, J, C_out)
        """
        B, T, J, C = x.shape

        # Spatial graph aggregation: x̂ = Â · x
        x_agg = torch.einsum("btjc,jk->btkc", x, adj)          # (B, T, J, C)
        x_s = self.spatial_fc(x_agg)                            # (B, T, J, C_out)

        # Temporal conv over each joint independently
        # Reshape to (B*J, C_out, T)
        x_t = x_s.permute(0, 2, 3, 1).reshape(B * J, -1, T)    # (B·J, C_out, T)
        x_t = self.temporal_conv(x_t)                           # (B·J, C_out, T)
        x_t = x_t.reshape(B, J, -1, T).permute(0, 3, 1, 2)     # (B, T, J, C_out)

        # BN + ReLU on (B·T·J, C_out) flattened
        out_flat = x_t.reshape(B * T * J, -1)
        out_flat = self.relu(self.bn(out_flat))
        out = self.drop(out_flat.reshape(B, T, J, -1))

        # Residual
        return out + self.residual(x)


class BodyStream(nn.Module):
    """
    2-layer ST-GCN body stream with PAM augmentation.

    Args:
        n_joints:      Number of skeleton joints.
        hidden_dim:    Hidden channel dimension.
        embed_dim:     Output embedding dimension.
        temporal_kernel: Temporal convolution kernel size.
        n_stgcn_layers:  Number of ST-GCN layers (default 2).
        lambda_init:   Initial λ for PAM (default 0.5).
    """

    def __init__(self,
                 n_joints: int = 32,
                 hidden_dim: int = 128,
                 embed_dim: int = 256,
                 temporal_kernel: int = 9,
                 n_stgcn_layers: int = 2,
                 lambda_init: float = 0.5):
        super().__init__()

        import numpy as np
        a_phys = build_body_adjacency(n_joints)
        a_ling = build_linguistic_adjacency(n_joints)

        self.pam = PAM(n_joints=n_joints,
                       lambda_init=lambda_init,
                       a_phys=a_phys,
                       a_ling=a_ling)

        dims = [3] + [hidden_dim] * (n_stgcn_layers - 1) + [embed_dim]
        self.layers = nn.ModuleList([
            STGCNLayer(dims[i], dims[i + 1], temporal_kernel)
            for i in range(n_stgcn_layers)
        ])

        # Project from joint dimension to single embedding per frame
        self.joint_pool = nn.Linear(n_joints, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, J, 3) skeleton tensor

        Returns:
            (B, T, embed_dim) per-frame body embeddings
        """
        adj = self.pam()   # (J, J) — recomputed each forward pass

        out = x
        for layer in self.layers:
            out = layer(out, adj)           # (B, T, J, embed_dim)

        # Pool over joints: (B, T, embed_dim, J) → (B, T, embed_dim)
        out = out.permute(0, 1, 3, 2)      # (B, T, embed_dim, J)
        out = self.joint_pool(out).squeeze(-1)  # (B, T, embed_dim)
        return out
