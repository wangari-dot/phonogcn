"""
pam.py — Phonological Adjacency Matrix
=======================================
Implements the learnable phonological graph augmentation described in §3.2.4.

The effective adjacency matrix is:

    A_eff = A_phys + λ · σ(A_ling + M_learn)      (Equation 2)

where:
    A_phys   — fixed physical skeleton adjacency (normalised, binary)
    A_ling   — fixed linguistically motivated adjacency (from KSL phonology,
                Mweri 2018)
    M_learn  — learnable edge-weight matrix (initialised to zero)
    λ        — balancing coefficient (learnable scalar, initialised to 0.5)
    σ        — sigmoid activation

Linguistic edges connect joints that co-articulate in KSL phonology:
    - Dominant hand ↔ non-dominant hand (handshape coordination)
    - Wrist ↔ ipsilateral shoulder (location encoding)
    - Fingertip joints ↔ face landmarks (contact locations)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class PAM(nn.Module):
    """
    Phonological Adjacency Matrix module.

    Args:
        n_joints:       Total number of joints in the skeleton.
        lambda_init:    Initial value for the balancing coefficient λ.
        a_phys:         Fixed physical adjacency matrix (n_joints × n_joints).
                        If None, uses an identity-based fallback.
        a_ling:         Fixed linguistic adjacency matrix (n_joints × n_joints).
                        If None, uses zeros (PAM degenerates to A_phys).
    """

    def __init__(self,
                 n_joints: int = 32,
                 lambda_init: float = 0.5,
                 a_phys: Optional[np.ndarray] = None,
                 a_ling: Optional[np.ndarray] = None):
        super().__init__()
        self.n_joints = n_joints

        # Fixed matrices (not optimised)
        if a_phys is None:
            a_phys = np.eye(n_joints, dtype=np.float32)
        if a_ling is None:
            a_ling = np.zeros((n_joints, n_joints), dtype=np.float32)

        self.register_buffer("A_phys", torch.tensor(a_phys, dtype=torch.float32))
        self.register_buffer("A_ling", torch.tensor(a_ling, dtype=torch.float32))

        # Learnable components
        self.M_learn = nn.Parameter(torch.zeros(n_joints, n_joints))
        self.lambda_pam = nn.Parameter(torch.tensor(lambda_init))

    def forward(self) -> torch.Tensor:
        """
        Compute the effective adjacency matrix.

        Returns:
            A_eff: (n_joints, n_joints) tensor — the effective adjacency matrix
                   used by the ST-GCN body stream.
        """
        A_aug = torch.sigmoid(self.A_ling + self.M_learn)
        A_eff = self.A_phys + self.lambda_pam * A_aug
        return A_eff

    def get_top_edges(self, k: int = 10):
        """
        Return the k strongest learned phonological edges.

        Useful for interpretability analysis (§4.7).

        Args:
            k: Number of top edges to return.

        Returns:
            List of (joint_i, joint_j, weight) tuples, sorted descending.
        """
        with torch.no_grad():
            aug = torch.sigmoid(self.A_ling + self.M_learn)
            weights = (self.lambda_pam * aug).cpu().numpy()

        # Exclude physical edges and self-loops
        phys = self.A_phys.cpu().numpy()
        mask = (phys == 0) & (np.eye(self.n_joints) == 0)
        candidates = [(i, j, weights[i, j])
                      for i in range(self.n_joints)
                      for j in range(i + 1, self.n_joints)
                      if mask[i, j]]
        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates[:k]


# ─────────────────────────────────────────────────────────────────────────────
# Linguistic adjacency construction (KSL phonology, Mweri 2018)
# ─────────────────────────────────────────────────────────────────────────────

def build_linguistic_adjacency(n_joints: int = 32) -> np.ndarray:
    """
    Build the fixed linguistic adjacency matrix A_ling for KSL.

    Phonologically motivated connections:
        1. Dominant (right) wrist ↔ non-dominant (left) wrist
           (bimanual handshape coordination)
        2. Right wrist ↔ right shoulder
           (signing-space location for right hand)
        3. Left wrist ↔ left shoulder
           (signing-space location for left hand)
        4. Right fingertips ↔ face reference points
           (contact-location signs, e.g. THINK, KNOW)
        5. Right hand MCP joints ↔ left hand MCP joints
           (symmetrical two-handed signs)

    Joint index convention (must match pipeline/broadcast_alignment.py):
        0=R-shoulder, 1=L-shoulder, 2=R-elbow, 3=L-elbow,
        4=R-wrist,    5=L-wrist,
        6–26 = right hand (MediaPipe 21 joints)
        27–31 = left hand subset

    NOTE: Adjust joint indices to match your specific skeleton convention.
    """
    A_ling = np.zeros((n_joints, n_joints), dtype=np.float32)

    def add_edge(i: int, j: int):
        if i < n_joints and j < n_joints:
            A_ling[i, j] = 1.0
            A_ling[j, i] = 1.0

    # 1. Bimanual wrist connection (R-wrist=4, L-wrist=5)
    add_edge(4, 5)

    # 2. R-wrist ↔ R-shoulder (location)
    add_edge(4, 0)

    # 3. L-wrist ↔ L-shoulder (location)
    add_edge(5, 1)

    # 4. Right fingertips (indices 10, 14, 18, 22, 26 for ring/pinky/index/middle/thumb tips)
    #    to face reference points — none in this reduced skeleton, so omit unless face joints added

    # 5. Right MCPs ↔ Left MCPs (symmetric signs)
    right_mcps = [7, 11, 16, 21]     # right hand MCP joints
    left_mcps  = [28, 29, 30, 31]    # left hand MCP joints
    for r, l in zip(right_mcps, left_mcps):
        add_edge(r, l)

    return A_ling
