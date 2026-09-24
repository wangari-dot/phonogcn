"""
clustering.py
=============
Stage 2 of the PhoneGCN data collection pipeline.

Takes candidate clips from Stage 1 (broadcast_alignment), extracts
ST-GCN embeddings, runs per-gloss K-means clustering, and assigns
pseudo-labels based on proximity to cluster centroids.

Architecture
------------
Encoder: Pretrained ST-GCN (WLASL-2000 / BSL-1K)
    - Input: (T, 32, 3) skeleton sequence
    - Output: 256-dimensional embedding

Pseudo-label assignment:
    - K = 3 clusters per gloss group
    - High-confidence threshold: centroid distance ≤ 1.5 × cluster std
      (sd_threshold = 1.5; sensitivity reported in Appendix B.2)
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight ST-GCN Encoder (feature extraction only)
# ─────────────────────────────────────────────────────────────────────────────

class GraphConvLayer(nn.Module):
    """Single spatial graph convolution layer."""

    def __init__(self, in_channels: int, out_channels: int, n_joints: int):
        super().__init__()
        self.fc = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels * n_joints)
        self.relu = nn.ReLU()
        self.n_joints = n_joints

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: (B, T, J, C_in)
        B, T, J, C = x.shape
        out = torch.einsum("btjc,jk->btkc", x, adj)  # graph aggregation
        out = self.fc(out)                             # (B, T, J, C_out)
        out_flat = out.reshape(B * T, J * out.size(-1))
        out_flat = self.bn(out_flat)
        out = out_flat.reshape(B, T, J, out.size(-1))
        return self.relu(out)


class STGCNEncoder(nn.Module):
    """
    2-layer Spatial-Temporal GCN encoder for skeleton sequences.

    Input:  (B, T, J, 3)  — T frames, J joints, xyz coordinates
    Output: (B, embed_dim) — pooled clip embedding
    """

    def __init__(self, n_joints: int = 32,
                 hidden_dim: int = 128,
                 embed_dim: int = 256,
                 temporal_kernel: int = 9):
        super().__init__()
        self.n_joints = n_joints
        self.embed_dim = embed_dim

        self.gcn1 = GraphConvLayer(3, hidden_dim, n_joints)
        self.gcn2 = GraphConvLayer(hidden_dim, embed_dim, n_joints)

        self.temporal_conv = nn.Conv1d(
            embed_dim, embed_dim,
            kernel_size=temporal_kernel,
            padding=temporal_kernel // 2,
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor,
                adj: torch.Tensor) -> torch.Tensor:
        # x: (B, T, J, 3)
        out = self.gcn1(x, adj)   # (B, T, J, hidden)
        out = self.gcn2(out, adj)  # (B, T, J, embed)

        # Average over joints, then temporal conv
        out = out.mean(dim=2)                 # (B, T, embed)
        out = out.permute(0, 2, 1)            # (B, embed, T)
        out = self.temporal_conv(out)         # (B, embed, T)
        out = self.pool(out).squeeze(-1)      # (B, embed)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Skeleton adjacency matrix (human body + hand topology)
# ─────────────────────────────────────────────────────────────────────────────

def build_body_adjacency(n_joints: int = 32) -> np.ndarray:
    """
    Build a normalised adjacency matrix for the signing-relevant skeleton.

    Joint ordering (0-indexed):
        0–5   : body (shoulders, elbows, wrists — bilateral)
        6–26  : right hand (21 MediaPipe joints)
        27–31 : left hand subset (wrist + 4 MCP joints for compactness)

    NOTE: Replace with your full joint set if using a different convention.
    """
    edges = [
        # Body chain
        (0, 1), (0, 2), (2, 4), (1, 3), (3, 5), (4, 5),
        # Right hand: wrist to MCPs
        (5, 6), (6, 7), (6, 11), (6, 16), (6, 21),
        # Right hand: finger chains (thumb–pinky)
        (7, 8), (8, 9), (9, 10),
        (11, 12), (12, 13), (13, 14),
        (16, 17), (17, 18), (18, 19),
        (21, 22), (22, 23), (23, 24),
        # Left hand subset
        (4, 27), (27, 28), (27, 29), (27, 30), (27, 31),
    ]

    A = np.zeros((n_joints, n_joints), dtype=np.float32)
    for i, j in edges:
        if i < n_joints and j < n_joints:
            A[i, j] = 1.0
            A[j, i] = 1.0

    # Symmetric normalisation: D^{-1/2} A D^{-1/2}
    degree = A.sum(axis=1)
    d_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
    A_norm = d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :]
    return A_norm


# ─────────────────────────────────────────────────────────────────────────────
# Clustering stage
# ─────────────────────────────────────────────────────────────────────────────

class ClusteringPipeline:
    """
    K-means pseudo-labelling for candidate sign clips.

    Usage::

        pipeline = ClusteringPipeline(
            encoder_checkpoint="checkpoints/stgcn_pretrained.pt",
            n_clusters=3,
            sd_threshold=1.5,
        )
        pseudo_labels = pipeline.run(
            candidates_path="output/candidates/all_candidates.json",
            skeleton_dir="output/skeletons/",
            output_path="output/pseudo_labels.json",
        )
    """

    def __init__(self,
                 encoder_checkpoint: str,
                 n_clusters: int = 3,
                 sd_threshold: float = 1.5,
                 n_joints: int = 32,
                 embed_dim: int = 256,
                 device: str = "cpu"):
        self.n_clusters = n_clusters
        self.sd_threshold = sd_threshold
        self.device = torch.device(device)

        adj = build_body_adjacency(n_joints)
        self.adj = torch.tensor(adj).to(self.device)

        self.encoder = STGCNEncoder(n_joints=n_joints, embed_dim=embed_dim).to(self.device)
        self._load_checkpoint(encoder_checkpoint)
        self.encoder.eval()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_checkpoint(self, checkpoint_path: str):
        if not os.path.exists(checkpoint_path):
            logger.warning(
                f"Encoder checkpoint not found: {checkpoint_path}. "
                "Running with random weights (for testing only)."
            )
            return
        state = torch.load(checkpoint_path, map_location=self.device)
        self.encoder.load_state_dict(state["model_state_dict"])
        logger.info(f"Loaded encoder checkpoint: {checkpoint_path}")

    def _load_skeleton(self, skeleton_path: str) -> Optional[torch.Tensor]:
        """
        Load a pre-extracted skeleton file (.npy) and return a (T, J, 3) tensor.

        The skeleton files are produced by running MediaPipe Holistic on the
        candidate clips and saving the joint coordinates as numpy arrays.
        """
        if not os.path.exists(skeleton_path):
            logger.warning(f"Skeleton file not found: {skeleton_path}")
            return None
        arr = np.load(skeleton_path).astype(np.float32)  # (T, J, 3)
        return torch.tensor(arr).unsqueeze(0).to(self.device)  # (1, T, J, 3)

    @torch.no_grad()
    def _encode_clip(self, skeleton_tensor: torch.Tensor) -> np.ndarray:
        """Encode a single clip skeleton to an embedding vector."""
        emb = self.encoder(skeleton_tensor, self.adj)  # (1, embed_dim)
        return emb.cpu().numpy().squeeze(0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_embeddings(self,
                           candidates: List[Dict],
                           skeleton_dir: str) -> Tuple[np.ndarray, List[int]]:
        """
        Extract ST-GCN embeddings for all candidates with matching skeleton files.

        Args:
            candidates: List of candidate dicts from broadcast_alignment.
            skeleton_dir: Directory containing .npy skeleton files named
                          <video_stem>_<start_ms>.npy.

        Returns:
            embeddings: (N_valid, embed_dim) array
            valid_indices: indices into candidates for which skeletons exist
        """
        embeddings, valid_indices = [], []

        for i, cand in enumerate(candidates):
            stem = Path(cand["video_path"]).stem
            start_ms = int(cand["start_sec"] * 1000)
            skel_path = os.path.join(skeleton_dir, f"{stem}_{start_ms}.npy")

            skel = self._load_skeleton(skel_path)
            if skel is None:
                continue

            emb = self._encode_clip(skel)
            embeddings.append(emb)
            valid_indices.append(i)

        if not embeddings:
            raise RuntimeError("No valid skeleton files found. Check skeleton_dir.")

        return np.array(embeddings, dtype=np.float32), valid_indices

    def cluster_by_gloss(self,
                         embeddings: np.ndarray,
                         valid_candidates: List[Dict]
                         ) -> List[Dict]:
        """
        Run per-gloss K-means and assign pseudo-labels with confidence scores.

        For each gloss group:
            1. Fit K-means with K = self.n_clusters.
            2. Compute each sample's distance to its assigned centroid.
            3. Mark samples within sd_threshold × cluster_std as
               high-confidence pseudo-labels (keep=True).

        Args:
            embeddings:       (N, embed_dim) array
            valid_candidates: Candidate dicts corresponding to embeddings

        Returns:
            pseudo_labels: List of dicts with added keys:
                {cluster_id, centroid_distance, normalised_distance,
                 keep, pseudo_label_confidence}
        """
        from collections import defaultdict

        # Group indices by gloss
        gloss_to_indices: Dict[str, List[int]] = defaultdict(list)
        for i, cand in enumerate(valid_candidates):
            gloss_to_indices[cand["gloss"]].append(i)

        pseudo_labels = [dict(c) for c in valid_candidates]

        for gloss, indices in gloss_to_indices.items():
            subset = embeddings[indices]
            n = len(indices)

            k = min(self.n_clusters, n)
            scaler = StandardScaler()
            subset_scaled = scaler.fit_transform(subset)

            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            cluster_ids = km.fit_predict(subset_scaled)
            centroids = km.cluster_centers_

            for local_i, global_i in enumerate(indices):
                cid = cluster_ids[local_i]
                dist = float(np.linalg.norm(subset_scaled[local_i] - centroids[cid]))

                # Intra-cluster distances for std estimation
                cluster_members = subset_scaled[cluster_ids == cid]
                if len(cluster_members) > 1:
                    intra_dists = np.linalg.norm(
                        cluster_members - centroids[cid], axis=1
                    )
                    cluster_std = float(intra_dists.std()) + 1e-8
                else:
                    cluster_std = 1.0

                norm_dist = dist / cluster_std
                keep = norm_dist <= self.sd_threshold
                confidence = float(np.exp(-norm_dist / self.sd_threshold))

                pseudo_labels[global_i].update({
                    "cluster_id": int(cid),
                    "centroid_distance": round(dist, 4),
                    "normalised_distance": round(norm_dist, 4),
                    "keep": bool(keep),
                    "pseudo_label_confidence": round(confidence, 4),
                })

            kept = sum(1 for i in indices if pseudo_labels[i]["keep"])
            logger.info(
                f"  Gloss '{gloss}': {n} clips → {k} clusters → "
                f"{kept} high-confidence ({100*kept/max(n,1):.1f}%)"
            )

        return pseudo_labels

    def run(self,
            candidates_path: str,
            skeleton_dir: str,
            output_path: str) -> List[Dict]:
        """
        Full clustering stage: load candidates → embed → cluster → save.

        Args:
            candidates_path: Path to all_candidates.json from Stage 1.
            skeleton_dir:    Directory containing pre-extracted .npy skeleton files.
            output_path:     Where to write the pseudo-labelled JSON.

        Returns:
            pseudo_labels: Full list of candidates with pseudo-label annotations.
        """
        with open(candidates_path) as f:
            candidates = json.load(f)
        logger.info(f"Loaded {len(candidates)} candidates from {candidates_path}")

        embeddings, valid_indices = self.extract_embeddings(candidates, skeleton_dir)
        valid_candidates = [candidates[i] for i in valid_indices]

        logger.info(f"Extracted embeddings for {len(valid_candidates)} / {len(candidates)} candidates")

        pseudo_labels = self.cluster_by_gloss(embeddings, valid_candidates)

        kept = sum(1 for p in pseudo_labels if p.get("keep", False))
        logger.info(
            f"Pseudo-labelling complete: {kept} / {len(pseudo_labels)} clips "
            f"pass sd_threshold={self.sd_threshold}"
        )

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(pseudo_labels, f, indent=2)
        logger.info(f"Saved pseudo-labels → {output_path}")

        return pseudo_labels


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="PhoneGCN Clustering & Pseudo-labelling")
    parser.add_argument("--candidates",     required=True, help="Path to all_candidates.json")
    parser.add_argument("--skeleton_dir",   required=True, help="Directory of .npy skeleton files")
    parser.add_argument("--output",         required=True, help="Output path for pseudo_labels.json")
    parser.add_argument("--encoder_ckpt",   required=True, help="ST-GCN encoder checkpoint")
    parser.add_argument("--n_clusters",     type=int,   default=3,   help="K-means clusters per gloss")
    parser.add_argument("--sd_threshold",   type=float, default=1.5, help="Centroid distance threshold (in cluster SDs)")
    parser.add_argument("--device",         default="cpu")
    args = parser.parse_args()

    pipeline = ClusteringPipeline(
        encoder_checkpoint=args.encoder_ckpt,
        n_clusters=args.n_clusters,
        sd_threshold=args.sd_threshold,
        device=args.device,
    )
    pipeline.run(args.candidates, args.skeleton_dir, args.output)
