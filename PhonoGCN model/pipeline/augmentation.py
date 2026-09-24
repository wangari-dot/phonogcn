"""
augmentation.py
===============
Stage 4 of the PhoneGCN data collection pipeline.

Applies geometric, temporal, and skeletal augmentation to the validated
dataset.  Rare-class clips (< 10 validated examples) receive additional
skeletal perturbation following Stoll et al. (2020).

Augmentation strategy (§3.1.4)
-------------------------------
Geometric   : Random rotation (±10°), scale (0.8–1.2×), horizontal flip
Temporal    : Speed perturbation (0.8–1.2×), frame dropout (p=0.05)
Skeletal    : Joint angle perturbation ±15° (rare classes only)

Target augmentation factor: 4.2×
Synthetic variants per rare sample: 3
"""

import os
import json
import logging
import random
import math
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Geometric augmentations
# ─────────────────────────────────────────────────────────────────────────────

def rotate_skeleton(skeleton: np.ndarray,
                    angle_deg: float,
                    axis: str = "z") -> np.ndarray:
    """
    Rotate skeleton coordinates around a given axis.

    Args:
        skeleton:  (T, J, 3) array of joint coordinates.
        angle_deg: Rotation angle in degrees.
        axis:      Rotation axis — 'x', 'y', or 'z'.

    Returns:
        Rotated skeleton of the same shape.
    """
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)

    if axis == "z":
        R = np.array([[c, -s, 0],
                      [s,  c, 0],
                      [0,  0, 1]], dtype=np.float32)
    elif axis == "y":
        R = np.array([[ c, 0, s],
                      [ 0, 1, 0],
                      [-s, 0, c]], dtype=np.float32)
    elif axis == "x":
        R = np.array([[1,  0,  0],
                      [0,  c, -s],
                      [0,  s,  c]], dtype=np.float32)
    else:
        raise ValueError(f"Unknown axis: {axis!r}. Use 'x', 'y', or 'z'.")

    return (skeleton @ R.T).astype(np.float32)


def scale_skeleton(skeleton: np.ndarray, scale: float) -> np.ndarray:
    """Uniformly scale all joint coordinates."""
    return (skeleton * scale).astype(np.float32)


def flip_skeleton_horizontal(skeleton: np.ndarray,
                              left_right_pairs: Optional[List[Tuple[int, int]]] = None
                              ) -> np.ndarray:
    """
    Mirror the skeleton left–right by negating the x coordinate and
    swapping symmetric joint pairs.

    Args:
        skeleton:         (T, J, 3) array.
        left_right_pairs: List of (left_joint_idx, right_joint_idx) pairs
                          to swap after mirroring.  Defaults to standard
                          body pairs (shoulders, elbows, wrists).

    Returns:
        Horizontally flipped skeleton.
    """
    flipped = skeleton.copy()
    flipped[:, :, 0] *= -1.0   # negate x

    if left_right_pairs is None:
        # Body: shoulder(0↔1), elbow(2↔3), wrist(4↔5)
        left_right_pairs = [(0, 1), (2, 3), (4, 5)]

    for l, r in left_right_pairs:
        flipped[:, [l, r], :] = flipped[:, [r, l], :]

    return flipped.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Temporal augmentations
# ─────────────────────────────────────────────────────────────────────────────

def speed_perturbation(skeleton: np.ndarray,
                       speed_factor: float,
                       n_frames_max: int = 150) -> np.ndarray:
    """
    Resample the temporal dimension to simulate signing speed variation.

    A speed_factor > 1 makes the sign faster (fewer output frames);
    speed_factor < 1 makes it slower (more output frames).

    Args:
        skeleton:      (T, J, 3) input sequence.
        speed_factor:  Resampling factor.  Range: 0.8–1.2 recommended.
        n_frames_max:  Maximum output length; longer sequences are truncated.

    Returns:
        Resampled skeleton of shape (T', J, 3) where T' ≤ n_frames_max.
    """
    T, J, C = skeleton.shape
    new_T = max(1, int(round(T / speed_factor)))
    new_T = min(new_T, n_frames_max)

    old_indices = np.linspace(0, T - 1, new_T)
    new_skeleton = np.zeros((new_T, J, C), dtype=np.float32)

    for j in range(J):
        for c in range(C):
            new_skeleton[:, j, c] = np.interp(old_indices,
                                               np.arange(T),
                                               skeleton[:, j, c])
    return new_skeleton


def frame_dropout(skeleton: np.ndarray,
                  p: float = 0.05,
                  rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Zero out randomly selected frames to simulate occlusion or packet loss.

    Args:
        skeleton: (T, J, 3) array.
        p:        Dropout probability per frame.
        rng:      Optional numpy random generator for reproducibility.

    Returns:
        Skeleton with zeroed frames.
    """
    if rng is None:
        rng = np.random.default_rng()
    result = skeleton.copy()
    mask = rng.random(skeleton.shape[0]) < p
    result[mask] = 0.0
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Skeletal perturbation (rare classes)
# ─────────────────────────────────────────────────────────────────────────────

def _local_rotation_matrix(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rodrigues' rotation formula."""
    theta = math.radians(angle_deg)
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    K = np.array([
        [     0, -axis[2],  axis[1]],
        [ axis[2],      0, -axis[0]],
        [-axis[1],  axis[0],      0],
    ], dtype=np.float32)
    return np.eye(3, dtype=np.float32) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)


def perturb_joint_angles(skeleton: np.ndarray,
                         max_angle_deg: float = 15.0,
                         rng: Optional[np.random.Generator] = None
                         ) -> np.ndarray:
    """
    Apply random rotation perturbations to each joint independently.

    Perturbation magnitude is sampled uniformly from [-max_angle_deg, +max_angle_deg]
    for each joint at each frame.  The rotation axis is sampled uniformly
    on the unit sphere.

    Args:
        skeleton:      (T, J, 3) joint coordinate array.
        max_angle_deg: Maximum perturbation magnitude in degrees.
        rng:           Optional numpy random generator.

    Returns:
        Perturbed skeleton of the same shape.
    """
    if rng is None:
        rng = np.random.default_rng()

    T, J, _ = skeleton.shape
    result = skeleton.copy()

    for j in range(J):
        for t in range(T):
            if np.allclose(result[t, j], 0.0):
                continue  # skip zero-padded joints

            angle = rng.uniform(-max_angle_deg, max_angle_deg)
            axis = rng.standard_normal(3).astype(np.float32)
            axis /= np.linalg.norm(axis) + 1e-8

            R = _local_rotation_matrix(axis, angle)
            result[t, j] = R @ result[t, j]

    return result.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Padding
# ─────────────────────────────────────────────────────────────────────────────

def pad_or_truncate(skeleton: np.ndarray,
                    n_frames_max: int = 150) -> np.ndarray:
    """
    Pad (with zeros) or truncate a skeleton sequence to n_frames_max frames.

    Args:
        skeleton:     (T, J, 3) array.
        n_frames_max: Target temporal length.

    Returns:
        (n_frames_max, J, 3) array.
    """
    T, J, C = skeleton.shape
    if T >= n_frames_max:
        return skeleton[:n_frames_max]
    pad = np.zeros((n_frames_max - T, J, C), dtype=np.float32)
    return np.concatenate([skeleton, pad], axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation pipeline
# ─────────────────────────────────────────────────────────────────────────────

class AugmentationPipeline:
    """
    Applies augmentation to the validated KSL-Daily-500 skeleton dataset.

    Usage::

        aug = AugmentationPipeline(
            skeleton_dir="output/skeletons/validated/",
            manifest_path="output/validated/validated_manifest.json",
            output_dir="output/augmented/",
        )
        aug.run()
    """

    def __init__(self,
                 skeleton_dir: str,
                 manifest_path: str,
                 output_dir: str,
                 rotation_range: float = 10.0,
                 scale_range: Tuple[float, float] = (0.8, 1.2),
                 flip_horizontal: bool = True,
                 speed_range: Tuple[float, float] = (0.8, 1.2),
                 frame_dropout_p: float = 0.05,
                 rare_class_threshold: int = 10,
                 max_joint_perturbation_deg: float = 15.0,
                 synthetic_variants_per_sample: int = 3,
                 n_frames_max: int = 150,
                 augment_factor: float = 4.2,
                 seed: int = 42):
        self.skeleton_dir = skeleton_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        with open(manifest_path) as f:
            self.manifest: List[Dict] = json.load(f)

        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.flip_horizontal = flip_horizontal
        self.speed_range = speed_range
        self.frame_dropout_p = frame_dropout_p
        self.rare_threshold = rare_class_threshold
        self.max_joint_deg = max_joint_perturbation_deg
        self.n_variants = synthetic_variants_per_sample
        self.n_frames_max = n_frames_max
        self.augment_factor = augment_factor
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_skeleton(self, clip_id: str) -> Optional[np.ndarray]:
        path = os.path.join(self.skeleton_dir, f"{clip_id}.npy")
        if not os.path.exists(path):
            logger.warning(f"Skeleton not found: {path}")
            return None
        return np.load(path).astype(np.float32)

    def _save_skeleton(self, skeleton: np.ndarray, clip_id: str):
        out = os.path.join(self.output_dir, f"{clip_id}.npy")
        np.save(out, skeleton)

    def _augment_one(self, skeleton: np.ndarray,
                     include_skeletal: bool = False) -> np.ndarray:
        """Apply a random combination of augmentations to one skeleton."""
        aug = skeleton.copy()

        # Geometric
        if self.rotation_range > 0:
            angle = self.rng.uniform(-self.rotation_range, self.rotation_range)
            aug = rotate_skeleton(aug, angle, axis="z")

        scale = self.rng.uniform(*self.scale_range)
        aug = scale_skeleton(aug, scale)

        if self.flip_horizontal and self.rng.random() < 0.5:
            aug = flip_skeleton_horizontal(aug)

        # Temporal
        speed = self.rng.uniform(*self.speed_range)
        aug = speed_perturbation(aug, speed, self.n_frames_max)
        aug = frame_dropout(aug, self.frame_dropout_p, self.rng)

        # Skeletal (rare classes only)
        if include_skeletal:
            aug = perturb_joint_angles(aug, self.max_joint_deg, self.rng)

        return pad_or_truncate(aug, self.n_frames_max)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _gloss_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for entry in self.manifest:
            g = entry["gloss"]
            counts[g] = counts.get(g, 0) + 1
        return counts

    def run(self) -> List[Dict]:
        """
        Augment the validated dataset to the target augmentation factor.

        Returns:
            Augmented manifest (original + synthetic entries).
        """
        gloss_counts = self._gloss_counts()
        rare_glosses = {g for g, c in gloss_counts.items()
                        if c < self.rare_threshold}

        augmented_manifest = list(self.manifest)  # start with originals

        # Copy originals to output_dir
        for entry in self.manifest:
            skel = self._load_skeleton(entry["clip_id"])
            if skel is None:
                continue
            skel = pad_or_truncate(skel, self.n_frames_max)
            self._save_skeleton(skel, entry["clip_id"])

        original_count = len(self.manifest)
        target_count = int(original_count * self.augment_factor)
        n_to_generate = target_count - original_count

        logger.info(
            f"Augmenting {original_count} clips → target {target_count} "
            f"(factor {self.augment_factor}×).  "
            f"Generating {n_to_generate} synthetic clips."
        )

        generated = 0
        entries_pool = list(self.manifest)
        random.seed(42)
        random.shuffle(entries_pool)

        while generated < n_to_generate:
            entry = entries_pool[generated % len(entries_pool)]
            skel = self._load_skeleton(entry["clip_id"])
            if skel is None:
                generated += 1
                continue

            include_skeletal = (entry["gloss"] in rare_glosses)
            aug_skel = self._augment_one(skel, include_skeletal)

            syn_id = f"{entry['clip_id']}_syn{generated:05d}"
            self._save_skeleton(aug_skel, syn_id)

            syn_entry = dict(entry)
            syn_entry["clip_id"] = syn_id
            syn_entry["synthetic"] = True
            syn_entry["source_clip_id"] = entry["clip_id"]
            syn_entry["skeletal_augmentation"] = include_skeletal
            augmented_manifest.append(syn_entry)
            generated += 1

        logger.info(f"Generated {generated} synthetic clips.")

        # Save manifest
        out_manifest = os.path.join(self.output_dir, "augmented_manifest.json")
        with open(out_manifest, "w") as f:
            json.dump(augmented_manifest, f, indent=2)
        logger.info(f"Augmented manifest → {out_manifest}")

        return augmented_manifest


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="PhoneGCN Augmentation Pipeline")
    parser.add_argument("--skeleton_dir",   required=True)
    parser.add_argument("--manifest",       required=True)
    parser.add_argument("--output_dir",     required=True)
    parser.add_argument("--augment_factor", type=float, default=4.2)
    parser.add_argument("--rare_threshold", type=int,   default=10)
    parser.add_argument("--max_joint_deg",  type=float, default=15.0)
    parser.add_argument("--n_variants",     type=int,   default=3)
    parser.add_argument("--seed",           type=int,   default=42)
    args = parser.parse_args()

    pipeline = AugmentationPipeline(
        skeleton_dir=args.skeleton_dir,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        augment_factor=args.augment_factor,
        rare_class_threshold=args.rare_threshold,
        max_joint_perturbation_deg=args.max_joint_deg,
        synthetic_variants_per_sample=args.n_variants,
        seed=args.seed,
    )
    pipeline.run()
