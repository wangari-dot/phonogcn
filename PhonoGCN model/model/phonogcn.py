"""
phonogcn.py — Full PhoneGCN Model
===================================
Top-level module integrating hand, body, and face streams with
cross-modal fusion and multi-task loss heads.

Outputs
-------
Primary:    CTC gloss sequence (recognition)
Auxiliary:  Handshape classifier logits   (for auxiliary loss α)
            Location classifier logits    (for auxiliary loss β)

Translation is handled separately by TranslationHead (translation.py);
PhoneGCN's fused output is passed directly to the translation head during
joint training.

Multi-task loss (Equation 5):
    L = L_ctc + α·L_hand + β·L_loc + γ·L_contrast

See §3.2.5 for full loss specification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .hand_stream  import HandStream
from .body_stream  import BodyStream
from .face_stream  import FaceStream
from .fusion       import FusionTransformer


# ─────────────────────────────────────────────────────────────────────────────
# NT-Xent contrastive loss (for self-supervised pretraining)
# ─────────────────────────────────────────────────────────────────────────────

class NTXentLoss(nn.Module):
    """
    Normalised Temperature-scaled Cross-Entropy loss (Chen et al., 2020).

    Operates on paired embeddings (e.g., two augmented views of the same clip).

    Args:
        tau_init: Initial temperature τ (learnable; default 0.07).
    """

    def __init__(self, tau_init: float = 0.07):
        super().__init__()
        self.log_tau = nn.Parameter(torch.log(torch.tensor(tau_init)))

    @property
    def tau(self) -> torch.Tensor:
        return self.log_tau.exp()

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1: (B, D) normalised embeddings from view 1
            z2: (B, D) normalised embeddings from view 2

        Returns:
            Scalar NT-Xent loss.
        """
        B = z1.size(0)
        z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)  # (2B, D)

        sim = torch.mm(z, z.T) / self.tau    # (2B, 2B)

        # Mask self-similarities
        mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
        sim = sim.masked_fill(mask, float("-inf"))

        # Positive pairs: (i, i+B) and (i+B, i)
        labels = torch.cat([
            torch.arange(B, 2 * B, device=z.device),
            torch.arange(0, B, device=z.device),
        ])  # (2B,)

        loss = F.cross_entropy(sim, labels)
        return loss


# ─────────────────────────────────────────────────────────────────────────────
# PhoneGCN
# ─────────────────────────────────────────────────────────────────────────────

class PhoneGCN(nn.Module):
    """
    Full PhoneGCN model for continuous KSL recognition and translation.

    Args:
        n_glosses:         Vocabulary size (default 607, expert-validated).
        n_handshapes:      Number of handshape classes (default 38; Mweri 2018).
        n_locations:       Number of signing-space location classes (default 12).
        hand_embed_dim:    Hand stream output dim (default 256).
        body_embed_dim:    Body stream output dim (default 256).
        face_embed_dim:    Face stream output dim (default 256).
        fusion_d_model:    Fusion transformer model dim (default 512).
        fusion_n_heads:    Fusion transformer attention heads (default 4).
        fusion_n_layers:   Fusion transformer layers (default 2).
        fusion_d_ffn:      Fusion transformer FFN dim (default 2048).
        tau_init:          Initial NT-Xent temperature (default 0.07).
        lambda_pam_init:   Initial PAM balancing coefficient λ (default 0.5).
    """

    def __init__(self,
                 n_glosses: int = 607,
                 n_handshapes: int = 38,
                 n_locations: int = 12,
                 hand_embed_dim: int = 256,
                 body_embed_dim: int = 256,
                 face_embed_dim: int = 256,
                 fusion_d_model: int = 512,
                 fusion_n_heads: int = 4,
                 fusion_n_layers: int = 2,
                 fusion_d_ffn: int = 2048,
                 tau_init: float = 0.07,
                 lambda_pam_init: float = 0.5):
        super().__init__()

        # ── Streams ──────────────────────────────────────────────────────────
        self.hand_stream = HandStream(embed_dim=hand_embed_dim)
        self.body_stream = BodyStream(embed_dim=body_embed_dim,
                                      lambda_init=lambda_pam_init)
        self.face_stream = FaceStream(embed_dim=face_embed_dim)

        # ── Fusion ───────────────────────────────────────────────────────────
        self.fusion = FusionTransformer(
            hand_dim=hand_embed_dim,
            body_dim=body_embed_dim,
            face_dim=face_embed_dim,
            d_model=fusion_d_model,
            n_heads=fusion_n_heads,
            n_layers=fusion_n_layers,
            d_ffn=fusion_d_ffn,
        )

        # ── Primary head: CTC ────────────────────────────────────────────────
        # +1 for CTC blank token
        self.ctc_head = nn.Linear(fusion_d_model, n_glosses + 1)

        # ── Auxiliary heads ──────────────────────────────────────────────────
        self.handshape_head = nn.Linear(fusion_d_model, n_handshapes)
        self.location_head  = nn.Linear(fusion_d_model, n_locations)

        # ── Loss ─────────────────────────────────────────────────────────────
        self.ctc_loss = nn.CTCLoss(blank=n_glosses, reduction="mean",
                                   zero_infinity=True)
        self.nt_xent  = NTXentLoss(tau_init=tau_init)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def encode(self,
               right_hand: torch.Tensor,
               left_hand: torch.Tensor,
               skeleton: torch.Tensor,
               face: torch.Tensor,
               padding_mask: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode inputs through all three streams and fuse.

        Args:
            right_hand:   (B, T, 3, H, W)
            left_hand:    (B, T, 3, H, W)
            skeleton:     (B, T, J, 3)
            face:         (B, T, 3, H, W)
            padding_mask: (B, T) boolean — True = padded frame

        Returns:
            fused:        (B, T, fusion_d_model)
            hand_feat:    (B, T, hand_embed_dim)
            body_feat:    (B, T, body_embed_dim)
            face_feat:    (B, T, face_embed_dim)
        """
        hand_feat = self.hand_stream(right_hand, left_hand)
        body_feat = self.body_stream(skeleton)
        face_feat = self.face_stream(face)

        fused = self.fusion(hand_feat, body_feat, face_feat,
                            src_key_padding_mask=padding_mask)
        return fused, hand_feat, body_feat, face_feat

    def forward(self,
                right_hand: torch.Tensor,
                left_hand: torch.Tensor,
                skeleton: torch.Tensor,
                face: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.

        Returns:
            Dict with keys:
                ctc_logits      : (T, B, n_glosses+1) — log-softmax for CTC
                handshape_logits: (B, T, n_handshapes)
                location_logits : (B, T, n_locations)
                fused           : (B, T, fusion_d_model) — for translation head
        """
        fused, _, _, _ = self.encode(
            right_hand, left_hand, skeleton, face, padding_mask
        )

        # CTC expects (T, B, C)
        ctc_log_probs = F.log_softmax(self.ctc_head(fused), dim=-1)
        ctc_log_probs = ctc_log_probs.permute(1, 0, 2)   # (T, B, n_glosses+1)

        hand_logits = self.handshape_head(fused)    # (B, T, n_handshapes)
        loc_logits  = self.location_head(fused)     # (B, T, n_locations)

        return {
            "ctc_logits":       ctc_log_probs,
            "handshape_logits": hand_logits,
            "location_logits":  loc_logits,
            "fused":            fused,
        }

    # ------------------------------------------------------------------
    # Precomputed-feature path (KSL-Daily-500 feature files)
    # ------------------------------------------------------------------

    def encode_features(self,
                        right_feat: torch.Tensor,
                        left_feat: torch.Tensor,
                        skeleton: torch.Tensor,
                        face_feat: torch.Tensor,
                        padding_mask: Optional[torch.Tensor] = None
                        ) -> torch.Tensor:
        """
        Encode from precomputed per-frame crop features instead of raw
        image crops. The dataset ships `crops/<clip>/<part>/features.npy`
        (T, 256) per part; these stand in for the CNN stream outputs, so
        the hand/face backbones are bypassed. The skeleton still runs
        through the ST-GCN body stream.

        Args:
            right_feat:   (B, T, hand_embed_dim)
            left_feat:    (B, T, hand_embed_dim)
            skeleton:     (B, T, J, 3)
            face_feat:    (B, T, face_embed_dim)
            padding_mask: (B, T) boolean — True = padded frame

        Returns:
            fused: (B, T, fusion_d_model)
        """
        # Reuse the hand stream's two-hand fusion MLP on the embeddings.
        hand_feat = self.hand_stream.fusion(
            torch.cat([right_feat, left_feat], dim=-1))
        body_feat = self.body_stream(skeleton)

        return self.fusion(hand_feat, body_feat, face_feat,
                           src_key_padding_mask=padding_mask)

    def forward_features(self,
                         right_feat: torch.Tensor,
                         left_feat: torch.Tensor,
                         skeleton: torch.Tensor,
                         face_feat: torch.Tensor,
                         padding_mask: Optional[torch.Tensor] = None
                         ) -> Dict[str, torch.Tensor]:
        """Full forward pass from precomputed features (same outputs as forward())."""
        fused = self.encode_features(right_feat, left_feat, skeleton,
                                     face_feat, padding_mask)

        ctc_log_probs = F.log_softmax(self.ctc_head(fused), dim=-1)
        ctc_log_probs = ctc_log_probs.permute(1, 0, 2)   # (T, B, n_glosses+1)

        return {
            "ctc_logits":       ctc_log_probs,
            "handshape_logits": self.handshape_head(fused),
            "location_logits":  self.location_head(fused),
            "fused":            fused,
        }

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def compute_loss(self,
                     outputs: Dict[str, torch.Tensor],
                     gloss_targets: torch.Tensor,
                     input_lengths: torch.Tensor,
                     target_lengths: torch.Tensor,
                     handshape_labels: Optional[torch.Tensor] = None,
                     location_labels: Optional[torch.Tensor] = None,
                     contrast_z1: Optional[torch.Tensor] = None,
                     contrast_z2: Optional[torch.Tensor] = None,
                     alpha: float = 0.3,
                     beta: float = 0.2,
                     gamma: float = 0.5,
                     pseudo_label_weights: Optional[torch.Tensor] = None,
                     ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss (Equation 5 in §3.2.5).

        L = L_ctc + α·L_hand + β·L_loc + γ·L_contrast

        Args:
            outputs:          Dict from forward()
            gloss_targets:    Concatenated gloss target sequences (CTC format)
            input_lengths:    (B,) temporal lengths before padding
            target_lengths:   (B,) gloss sequence lengths
            handshape_labels: (B, T) integer handshape class per frame (optional)
            location_labels:  (B, T) integer location class per frame (optional)
            contrast_z1/z2:   (B, D) paired contrastive embeddings (optional)
            alpha, beta, gamma: Loss weight coefficients
            pseudo_label_weights: (B,) per-sample weight for pseudo-labelled clips
                                  (delta=0.4 for pseudo, 1.0 for expert-validated)

        Returns:
            Dict with individual and total loss values.
        """
        losses = {}

        # CTC loss — confidence-weighted pseudo-label discount (§3.2.5, δ).
        # Per-sample weighting needs per-sample loss values, so this uses the
        # functional form with reduction="none" and a manual weighted mean
        # instead of the reduction="mean" nn.CTCLoss module.
        if pseudo_label_weights is not None:
            l_ctc_per_sample = F.ctc_loss(
                outputs["ctc_logits"], gloss_targets, input_lengths, target_lengths,
                blank=self.ctc_loss.blank, reduction="none",
                zero_infinity=self.ctc_loss.zero_infinity,
            )
            l_ctc = ((l_ctc_per_sample * pseudo_label_weights).sum()
                    / pseudo_label_weights.sum().clamp_min(1e-8))
        else:
            l_ctc = self.ctc_loss(
                outputs["ctc_logits"], gloss_targets, input_lengths, target_lengths,
            )
        losses["ctc"] = l_ctc

        total_loss = l_ctc

        # Auxiliary handshape loss
        if handshape_labels is not None and alpha > 0:
            B, T, _ = outputs["handshape_logits"].shape
            l_hand = F.cross_entropy(
                outputs["handshape_logits"].reshape(B * T, -1),
                handshape_labels.reshape(B * T),
                ignore_index=-1,
            )
            losses["handshape"] = l_hand
            total_loss = total_loss + alpha * l_hand

        # Auxiliary location loss
        if location_labels is not None and beta > 0:
            B, T, _ = outputs["location_logits"].shape
            l_loc = F.cross_entropy(
                outputs["location_logits"].reshape(B * T, -1),
                location_labels.reshape(B * T),
                ignore_index=-1,
            )
            losses["location"] = l_loc
            total_loss = total_loss + beta * l_loc

        # Contrastive loss
        if contrast_z1 is not None and contrast_z2 is not None and gamma > 0:
            z1 = F.normalize(contrast_z1, dim=-1)
            z2 = F.normalize(contrast_z2, dim=-1)
            l_contrast = self.nt_xent(z1, z2)
            losses["contrastive"] = l_contrast
            total_loss = total_loss + gamma * l_contrast

        losses["total"] = total_loss
        return losses
