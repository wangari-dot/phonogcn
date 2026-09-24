"""
broadcast_alignment.py
======================
Stage 1 of the PhoneGCN data collection pipeline.

Ingests broadcast video (e.g. KBC/Citizen TV sign language insets),
applies ASR to the audio track to generate a timestamped transcript,
then runs a keyword-spotting model to identify video segments where
a target KSL gloss is being signed.

Architecture
------------
Keyword spotter: 2-layer Temporal Convolutional Network (TCN)
    - Input: MediaPipe Holistic pose features (body + hands)
    - Pretrained: WLASL-2000 (ASL)
    - Fine-tuned: 80 manually annotated KSL seed clips

Detection threshold (tau_det = 0.5) selected via grid search on a
40-clip held-out validation set. See Appendix B.1 for sensitivity analysis.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import cv2
import mediapipe as mp

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Pose Extractor
# ─────────────────────────────────────────────────────────────────────────────

class PoseExtractor:
    """
    Extracts per-frame pose features using MediaPipe Holistic.
    Returns a (T, N_joints, 3) array of 3D coordinates.
    """
    N_BODY_LANDMARKS = 33
    N_HAND_LANDMARKS = 21   # per hand
    USED_BODY_INDICES = [11, 12, 13, 14, 15, 16]  # shoulders, elbows, wrists

    def __init__(self, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        self.holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=1,
        )

    def extract_from_video(self, video_path: str) -> np.ndarray:
        """
        Extract pose features for every frame in a video.

        Args:
            video_path: Path to the video file.

        Returns:
            features: np.ndarray of shape (T, N_features) where
                      N_features = len(USED_BODY_INDICES) * 3
                                 + N_HAND_LANDMARKS * 3 * 2 (both hands)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        frames_features = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.holistic.process(rgb)
            feat = self._parse_landmarks(result)
            frames_features.append(feat)

        cap.release()
        return np.array(frames_features, dtype=np.float32)

    def _parse_landmarks(self, result) -> np.ndarray:
        """Flatten selected landmarks into a feature vector."""
        features = []

        # Body: only signing-relevant joints
        if result.pose_landmarks:
            for idx in self.USED_BODY_INDICES:
                lm = result.pose_landmarks.landmark[idx]
                features.extend([lm.x, lm.y, lm.z])
        else:
            features.extend([0.0] * len(self.USED_BODY_INDICES) * 3)

        # Right hand
        if result.right_hand_landmarks:
            for lm in result.right_hand_landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z])
        else:
            features.extend([0.0] * self.N_HAND_LANDMARKS * 3)

        # Left hand
        if result.left_hand_landmarks:
            for lm in result.left_hand_landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z])
        else:
            features.extend([0.0] * self.N_HAND_LANDMARKS * 3)

        return np.array(features, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Keyword Spotter: 2-layer TCN
# ─────────────────────────────────────────────────────────────────────────────

class TemporalConvBlock(nn.Module):
    """Single dilated causal temporal convolutional block."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.residual = (nn.Conv1d(in_channels, out_channels, 1)
                         if in_channels != out_channels else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Causal: remove future context introduced by symmetric padding
        out = self.conv(x)[:, :, :x.size(2)]
        out = self.relu(self.bn(out))
        return self.dropout(out) + self.residual(x)


class KeywordSpotter(nn.Module):
    """
    2-layer TCN keyword spotter operating on per-frame pose features.

    Input:  (B, T, n_input_features)
    Output: (B, n_glosses) — raw logits for gloss presence per clip
    """

    def __init__(self, n_input_features: int, n_glosses: int,
                 hidden_dim: int = 128):
        super().__init__()
        self.tcn = nn.Sequential(
            TemporalConvBlock(n_input_features, hidden_dim, kernel_size=3, dilation=1),
            TemporalConvBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(hidden_dim, n_glosses)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) → (B, F, T)
        x = x.permute(0, 2, 1)
        x = self.tcn(x)              # (B, H, T)
        x = self.pool(x).squeeze(-1) # (B, H)
        return self.classifier(x)    # (B, n_glosses)


# ─────────────────────────────────────────────────────────────────────────────
# ASR wrapper (expects an external ASR backend)
# ─────────────────────────────────────────────────────────────────────────────

def run_asr(audio_path: str) -> List[Dict]:
    """
    Run ASR on an audio file and return a list of timestamped word tokens.

    Returns:
        List of dicts: [{"word": str, "start": float, "end": float}, ...]

    NOTE: This is a stub. Replace with your preferred ASR backend
    (e.g. SpeechBrain, Whisper, Google STT, etc.).
    """
    raise NotImplementedError(
        "Replace run_asr() with your ASR backend. "
        "Expected output: [{'word': str, 'start': float, 'end': float}, ...]"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline stage
# ─────────────────────────────────────────────────────────────────────────────

class BroadcastAlignmentPipeline:
    """
    Keyword-spotting pipeline for broadcast sign language video.

    Usage::

        pipeline = BroadcastAlignmentPipeline(
            spotter_checkpoint="checkpoints/seed_spotter.pt",
            gloss_vocabulary=["HOSPITAL", "SCHOOL", ...],
            det_threshold=0.5,
        )
        candidates = pipeline.process_video("broadcast_jan_2025.mp4")
        pipeline.save_candidates(candidates, "output/candidates/")
    """

    N_POSE_FEATURES = 6 * 3 + 21 * 3 * 2   # 6 body + 42 hand landmarks × 3D

    def __init__(self, spotter_checkpoint: str,
                 gloss_vocabulary: List[str],
                 det_threshold: float = 0.5,
                 clip_window_sec: float = 2.0,
                 stride_sec: float = 0.5,
                 device: str = "cpu"):
        self.vocabulary = gloss_vocabulary
        self.n_glosses = len(gloss_vocabulary)
        self.det_threshold = det_threshold
        self.clip_window_sec = clip_window_sec
        self.stride_sec = stride_sec
        self.device = torch.device(device)

        self.pose_extractor = PoseExtractor()
        self.spotter = KeywordSpotter(
            n_input_features=self.N_POSE_FEATURES,
            n_glosses=self.n_glosses,
        ).to(self.device)
        self._load_checkpoint(spotter_checkpoint)
        self.spotter.eval()

    def _load_checkpoint(self, checkpoint_path: str):
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint not found: {checkpoint_path}. "
                           "Running with random weights (for testing only).")
            return
        state = torch.load(checkpoint_path, map_location=self.device)
        self.spotter.load_state_dict(state["model_state_dict"])
        logger.info(f"Loaded spotter checkpoint: {checkpoint_path}")

    def process_video(self, video_path: str) -> List[Dict]:
        """
        Process a single broadcast video and return candidate clips.

        Args:
            video_path: Path to the broadcast video file.

        Returns:
            List of candidate dicts:
            [{"gloss": str, "start_sec": float, "end_sec": float,
              "confidence": float, "video_path": str}, ...]
        """
        logger.info(f"Processing: {video_path}")
        pose_features = self.pose_extractor.extract_from_video(video_path)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        window_frames = int(self.clip_window_sec * fps)
        stride_frames = int(self.stride_sec * fps)
        T = len(pose_features)

        candidates = []
        for start_f in range(0, T - window_frames + 1, stride_frames):
            end_f = start_f + window_frames
            clip_feat = pose_features[start_f:end_f]  # (W, F)
            clip_tensor = torch.tensor(clip_feat).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.spotter(clip_tensor)         # (1, n_glosses)
                probs = torch.sigmoid(logits).squeeze(0)   # (n_glosses,)

            for g_idx, prob in enumerate(probs):
                if prob.item() >= self.det_threshold:
                    candidates.append({
                        "gloss": self.vocabulary[g_idx],
                        "start_sec": round(start_f / fps, 3),
                        "end_sec": round(end_f / fps, 3),
                        "confidence": round(prob.item(), 4),
                        "video_path": video_path,
                    })

        logger.info(f"  Found {len(candidates)} candidates in {video_path}")
        return candidates

    def process_directory(self, video_dir: str,
                          output_dir: str) -> List[Dict]:
        """Process all .mp4 files in video_dir and save candidate JSONs."""
        os.makedirs(output_dir, exist_ok=True)
        all_candidates = []

        video_paths = sorted(Path(video_dir).glob("*.mp4"))
        logger.info(f"Found {len(video_paths)} videos in {video_dir}")

        for vp in video_paths:
            candidates = self.process_video(str(vp))
            all_candidates.extend(candidates)

            out_file = os.path.join(output_dir, vp.stem + "_candidates.json")
            with open(out_file, "w") as f:
                json.dump(candidates, f, indent=2)

        summary_path = os.path.join(output_dir, "all_candidates.json")
        with open(summary_path, "w") as f:
            json.dump(all_candidates, f, indent=2)

        logger.info(f"Total candidates: {len(all_candidates)} → {summary_path}")
        return all_candidates

    def save_candidates(self, candidates: List[Dict], output_dir: str):
        """Save candidate clip metadata to JSON."""
        os.makedirs(output_dir, exist_ok=True)
        out = os.path.join(output_dir, "candidates.json")
        with open(out, "w") as f:
            json.dump(candidates, f, indent=2)
        logger.info(f"Saved {len(candidates)} candidates → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="PhoneGCN Broadcast Alignment Pipeline")
    parser.add_argument("--video_dir",      required=True,  help="Directory of broadcast .mp4 files")
    parser.add_argument("--output_dir",     required=True,  help="Output directory for candidate JSONs")
    parser.add_argument("--seed_model",     required=True,  help="Path to keyword spotter checkpoint")
    parser.add_argument("--vocabulary",     required=True,  help="Path to gloss vocabulary JSON list")
    parser.add_argument("--threshold",      type=float, default=0.5, help="Detection threshold (default: 0.5)")
    parser.add_argument("--clip_window",    type=float, default=2.0, help="Clip window in seconds")
    parser.add_argument("--stride",         type=float, default=0.5, help="Sliding window stride in seconds")
    parser.add_argument("--device",         default="cpu")
    args = parser.parse_args()

    with open(args.vocabulary) as f:
        vocabulary = json.load(f)

    pipeline = BroadcastAlignmentPipeline(
        spotter_checkpoint=args.seed_model,
        gloss_vocabulary=vocabulary,
        det_threshold=args.threshold,
        clip_window_sec=args.clip_window,
        stride_sec=args.stride,
        device=args.device,
    )
    pipeline.process_directory(args.video_dir, args.output_dir)
