"""
config.py — Centralised hyperparameter and path configuration for PhoneGCN.

All training, pipeline, and model hyperparameters are defined here.
Override individual settings by passing --config_override KEY=VALUE to
train.py or evaluate.py.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Dataset & Paths
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DataConfig:
    # Root directory for KSL-Daily-500 (after pipeline + expert validation)
    data_root: str = "../KSL_Daily_500_dataset/KSL_Daily_500_expert607"
    # Gloss-to-text pairs (JSON: [{"gloss_seq": [...], "text": "..."}])
    gloss_text_pairs: str = "data/gloss_text_pairs.json"
    # Minimal pair evaluation set
    minimal_pairs_dir: str = "data/minimal_pairs"

    # Split configuration (signer-independent), 70/15/15
    train_split: float = 0.70
    val_split:   float = 0.15
    test_split:  float = 0.15

    # Pose skeleton
    n_joints: int = 32          # Body + hand landmark joints used
    n_frames_max: int = 150     # Max temporal length; shorter clips are padded

    # Handshape classes (Mweri 2018 inventory)
    n_handshapes: int = 38
    # Location classes (signing-space regions)
    n_locations: int = 12
    # Total gloss vocabulary (expert-validated list: cleaned_expert_validated.csv)
    n_glosses: int = 607


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    # --- Broadcast alignment ---
    # Keyword-spotter detection threshold (Appendix B.1)
    det_threshold: float = 0.5
    # Pretrained ASL spotter checkpoint (WLASL-2000 pretrained, KSL fine-tuned)
    seed_spotter_ckpt: str = "checkpoints/seed_spotter.pt"
    # Seed set size used for fine-tuning keyword spotter
    seed_set_size: int = 80

    # --- Clustering ---
    n_clusters: int = 3         # K-means clusters per gloss group
    # Proximity threshold for high-confidence pseudo-label assignment (Appendix B.2)
    sd_threshold: float = 1.5
    # Pretrained ST-GCN encoder for feature extraction
    stgcn_pretrain_ckpt: str = "checkpoints/stgcn_pretrained.pt"

    # --- Augmentation ---
    # Geometric augmentation parameters
    rotation_range: float = 10.0        # degrees ±
    scale_range: tuple = (0.8, 1.2)
    flip_horizontal: bool = True

    # Skeletal perturbation for rare classes (Stoll et al., 2020)
    rare_class_threshold: int = 10      # classes with < N examples get augmented
    max_joint_perturbation_deg: float = 15.0   # ±15° per joint axis
    synthetic_variants_per_sample: int = 3

    # Target augmentation factor
    augment_factor: float = 4.2


# ─────────────────────────────────────────────────────────────────────────────
# Model Architecture
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    # --- Hand stream ---
    hand_backbone: str = "efficientnet_lite0"   # Options: efficientnet_lite0, mobilenetv3_large, resnet18
    hand_embed_dim: int = 256
    n_hand_keypoints: int = 21   # MediaPipe Hands per hand, x2 = 42 total

    # --- Body stream (ST-GCN) ---
    stgcn_hidden_dim: int = 256
    stgcn_n_layers: int = 2
    stgcn_temporal_kernel: int = 9   # kernel size for temporal convolution

    # --- Face stream (I3D variant) ---
    face_embed_dim: int = 256

    # --- Phonological Adjacency Matrix (PAM) ---
    # Balancing coefficient λ (Equation 2) — learnable, initialised here
    lambda_pam_init: float = 0.5

    # --- Fusion transformer ---
    fusion_embed_dim: int = 512
    fusion_n_heads: int = 4
    fusion_n_layers: int = 2
    fusion_ffn_dim: int = 2048

    # --- Translation (mBART) ---
    mbart_model: str = "facebook/mbart-large-cc25"
    translation_n_encoder_layers: int = 2
    translation_n_decoder_layers: int = 2
    translation_embed_dim: int = 512
    translation_ffn_dim: int = 2048
    target_language: str = "en_XX"   # "en_XX" for English, "sw_KE" for Swahili


# ─────────────────────────────────────────────────────────────────────────────
# Training Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-4
    lr_schedule: str = "cosine_annealing"   # cosine_annealing | step | none

    # Multi-task loss weights (Equation 5)
    alpha: float = 0.3    # Handshape auxiliary loss weight
    beta: float = 0.2     # Location auxiliary loss weight
    gamma: float = 0.5    # Contrastive pretraining loss weight

    # Confidence-weighted pseudo-label discount
    delta: float = 0.4    # Weight for pseudo-labelled samples (vs 1.0 for expert)

    # NT-Xent contrastive loss temperature (learnable, initialised here)
    tau_init: float = 0.07

    # Transfer learning
    pretrain_dataset: str = "wlasl2000"   # wlasl2000 + bsl1k
    freeze_lower_layers: bool = True
    n_frozen_layers: int = 2   # Freeze first N ST-GCN layers after transfer

    # Reproducibility
    n_seeds: int = 5
    seed: int = 42

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    save_every_n_epochs: int = 10


# ─────────────────────────────────────────────────────────────────────────────
# Master config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def get_config() -> Config:
    """Return the default configuration object."""
    return Config()
