"""
train.py — PhoneGCN Training Script
=====================================
End-to-end training for the PhoneGCN continuous KSL recognition model
on the KSL-Daily-500 dataset (607-gloss expert-validated vocabulary).

Loads REAL per-frame pixel crops (`crops/<clip_id>/{right_hand,left_hand,
face}.npy`, (T,3,112,112)/(T,3,64,64) uint8) plus skeletons
(`skeletons/<clip_id>.npy`, (T,32,3)), produced by
dataset_toolkit/03b_extract_crops.py from real broadcast video — training
runs the standard PhoneGCN.forward() path, so the raw-image CNN backbones
(HandStream/FaceStream) are trained end-to-end rather than bypassed.
(The synthetic-data `forward_features()` precomputed-feature path in
model/phonogcn.py still exists for schema/pipeline smoke tests — see
checkpoints_synthetic_placeholder/ — but real training must not use it,
since random precomputed vectors carry no gloss signal.)

Usage
-----
    python train.py \\
        --data_root E:/KSL_keypoints \\
        --output_dir checkpoints/ \\
        [--config_override train.lr=5e-5 train.batch_size=8]

Reproducibility
---------------
Training is repeated for n_seeds random seeds (default 5), and results
are reported as mean ± std across seeds (see §4.3 and §4.8).
"""

import os
import json
import logging
import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import get_config, Config
from model import PhoneGCN

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class KSLDataset(Dataset):
    """
    KSL-Daily-500 loader (real raw-pixel-crop layout).

    Each sample contains:
        right_hand:  (T, 3, 112, 112) right-hand crop, normalised to [0,1]
        left_hand:   (T, 3, 112, 112) left-hand crop, normalised to [0,1]
        face:        (T, 3, 64, 64) face crop, normalised to [0,1]
        skeleton:    (T, J, 3) pose tensor
        gloss_ids:   (L,) integer gloss sequence
        pseudo_weight: float — 1.0 for expert-validated, <1 for pseudo-labelled

    crops/<clip_id>/{right_hand,left_hand,face}.npy come from
    dataset_toolkit/03b_extract_crops.py's mediapipe Holistic pass over
    real broadcast video — NOT the synthetic dataset's precomputed
    features.npy (random vectors), which model.forward_features() exists
    to consume instead and must not be used for a real training run.
    """

    def __init__(self, manifest_path: str, data_root: str,
                 cfg: Config, split: str = "train"):
        self.cfg       = cfg
        self.split     = split
        self.data_root = Path(data_root)

        with open(manifest_path, encoding="utf-8") as f:
            self.samples = json.load(f)

        logger.info(f"KSLDataset [{split}]: {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        e = self.samples[idx]
        cid = e["clip_id"]

        skeleton = np.load(self.data_root / e["skeleton_path"])
        crops_dir = self.data_root / "crops" / cid
        right_hand = np.load(crops_dir / "right_hand.npy")
        left_hand = np.load(crops_dir / "left_hand.npy")
        face = np.load(crops_dir / "face.npy")

        # Truncate to n_frames_max (padding happens in collate_fn)
        T = min(len(skeleton), self.cfg.data.n_frames_max)

        return {
            "right_hand":    torch.from_numpy(right_hand[:T].astype(np.float32) / 255.0),
            "left_hand":     torch.from_numpy(left_hand[:T].astype(np.float32) / 255.0),
            "face":          torch.from_numpy(face[:T].astype(np.float32) / 255.0),
            "skeleton":      torch.from_numpy(skeleton[:T]).float(),
            "gloss_ids":     torch.tensor(e["gloss_ids"], dtype=torch.long),
            "pseudo_weight": float(e.get("pseudo_weight", 1.0)),
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate samples into a batch dict, padding all variable-length
    sequences to the maximum length in the batch.

    padding_mask is True on padded frames (transformer convention).
    """
    B = len(batch)
    T_max = max(s["skeleton"].shape[0] for s in batch)
    L_max = max(s["gloss_ids"].shape[0] for s in batch)

    def zeros_like_first(key, trailing):
        return torch.zeros(B, T_max, *trailing)

    out = {
        "right_hand":     zeros_like_first("right_hand", batch[0]["right_hand"].shape[1:]),
        "left_hand":      zeros_like_first("left_hand",  batch[0]["left_hand"].shape[1:]),
        "face":           zeros_like_first("face",       batch[0]["face"].shape[1:]),
        "skeleton":       zeros_like_first("skeleton",   batch[0]["skeleton"].shape[1:]),
        "gloss_ids":      torch.zeros(B, L_max, dtype=torch.long),
        "padding_mask":   torch.ones(B, T_max, dtype=torch.bool),
        "input_lengths":  torch.zeros(B, dtype=torch.long),
        "target_lengths": torch.zeros(B, dtype=torch.long),
        "pseudo_weights": torch.tensor([s["pseudo_weight"] for s in batch]),
    }

    for i, s in enumerate(batch):
        T = s["skeleton"].shape[0]
        L = s["gloss_ids"].shape[0]
        for key in ("right_hand", "left_hand", "face", "skeleton"):
            out[key][i, :T] = s[key]
        out["gloss_ids"][i, :L]   = s["gloss_ids"]
        out["padding_mask"][i, :T] = False
        out["input_lengths"][i]    = T
        out["target_lengths"][i]   = L

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Training utilities
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    # Separate LR for fine-tuned PAM parameters (lower LR for stability)
    pam_params     = [p for n, p in model.named_parameters() if "pam" in n.lower()]
    other_params   = [p for n, p in model.named_parameters() if "pam" not in n.lower()]

    return AdamW([
        {"params": other_params, "lr": cfg.train.lr},
        {"params": pam_params,   "lr": cfg.train.lr * 0.1},
    ], weight_decay=cfg.train.weight_decay)


def build_scheduler(optimizer, cfg: Config, n_steps: int):
    if cfg.train.lr_schedule == "cosine_annealing":
        return CosineAnnealingLR(optimizer, T_max=n_steps)
    elif cfg.train.lr_schedule == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=n_steps // 3, gamma=0.1)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def forward_batch(model: PhoneGCN, batch: Dict, device: torch.device) -> Dict:
    # Standard forward() — raw pixel crops through the real CNN backbones,
    # not forward_features()'s precomputed-vector bypass (synthetic-data only).
    return model(
        batch["right_hand"].to(device),
        batch["left_hand"].to(device),
        batch["skeleton"].to(device),
        batch["face"].to(device),
        batch["padding_mask"].to(device),
    )


def train_one_epoch(model: PhoneGCN,
                    loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    scheduler,
                    cfg: Config,
                    device: torch.device,
                    epoch: int) -> Dict[str, float]:
    model.train()
    total_losses: Dict[str, float] = {}
    n_batches = 0

    for batch in loader:
        optimizer.zero_grad()

        outputs = forward_batch(model, batch, device)

        # No frame-level handshape/location labels in this dataset →
        # auxiliary heads are skipped (alpha/beta paths need real labels).
        loss_dict = model.compute_loss(
            outputs=outputs,
            gloss_targets=batch["gloss_ids"].to(device),
            input_lengths=batch["input_lengths"].to(device),
            target_lengths=batch["target_lengths"].to(device),
            alpha=0.0,
            beta=0.0,
            pseudo_label_weights=batch["pseudo_weights"].to(device),
        )

        loss = loss_dict["total"]
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        for k, v in loss_dict.items():
            total_losses[k] = total_losses.get(k, 0.0) + v.item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in total_losses.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation: greedy CTC decoding + WER
# ─────────────────────────────────────────────────────────────────────────────

def ctc_greedy_decode(log_probs: torch.Tensor,
                      input_lengths: torch.Tensor,
                      blank: int) -> List[List[int]]:
    """
    Greedy (best-path) CTC decoding.

    Args:
        log_probs:     (T, B, C) log-softmax outputs
        input_lengths: (B,) valid frame counts
        blank:         blank token index

    Returns:
        List of decoded gloss-id sequences, one per batch element.
    """
    best = log_probs.argmax(dim=-1).T   # (B, T)
    decoded = []
    for b, path in enumerate(best):
        seq, prev = [], blank
        for t in range(int(input_lengths[b])):
            tok = int(path[t])
            if tok != blank and tok != prev:
                seq.append(tok)
            prev = tok
        decoded.append(seq)
    return decoded


def edit_distance(ref: List[int], hyp: List[int]) -> int:
    """Levenshtein distance between two token sequences."""
    m, n = len(ref), len(hyp)
    d = list(range(n + 1))
    for i in range(1, m + 1):
        prev, d[0] = d[0], i
        for j in range(1, n + 1):
            cur = d[j]
            d[j] = min(d[j] + 1,          # deletion
                       d[j - 1] + 1,      # insertion
                       prev + (ref[i - 1] != hyp[j - 1]))  # substitution
            prev = cur
    return d[n]


@torch.no_grad()
def evaluate(model: PhoneGCN,
             loader: DataLoader,
             cfg: Config,
             device: torch.device) -> Dict[str, float]:
    """Evaluate CTC greedy-decoded gloss-level Word Error Rate (WER)."""
    model.eval()
    blank = cfg.data.n_glosses
    total_err, total_ref, total_loss, n_batches = 0, 0, 0.0, 0

    for batch in loader:
        outputs = forward_batch(model, batch, device)
        loss_dict = model.compute_loss(
            outputs=outputs,
            gloss_targets=batch["gloss_ids"].to(device),
            input_lengths=batch["input_lengths"].to(device),
            target_lengths=batch["target_lengths"].to(device),
            alpha=0.0, beta=0.0,
            pseudo_label_weights=batch["pseudo_weights"].to(device),
        )
        total_loss += loss_dict["ctc"].item()
        n_batches  += 1

        hyps = ctc_greedy_decode(outputs["ctc_logits"],
                                 batch["input_lengths"], blank)
        for i, hyp in enumerate(hyps):
            ref = batch["gloss_ids"][i, :batch["target_lengths"][i]].tolist()
            total_err += edit_distance(ref, hyp)
            total_ref += len(ref)

    return {
        "wer":      total_err / max(total_ref, 1),
        "ctc_loss": total_loss / max(n_batches, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Data provenance guard
# ─────────────────────────────────────────────────────────────────────────────

def check_data_provenance(data_root: Path, cfg: Config) -> None:
    """
    Warn loudly when training on synthetic or severely under-sampled data.

    The dataset toolkit's integrity rule is that results reported on
    synthetic clips must be labeled as such (dataset_toolkit/README.md).
    Synthetic clips carry no gloss-conditioned signal — their features are
    random vectors — so CTC will collapse to all-blank and WER will sit at
    1.0 no matter how long training runs. Failing silently there wastes
    compute and invites unlabeled numbers into a paper.
    """
    with open(data_root / "train_manifest.json", encoding="utf-8") as f:
        train = json.load(f)

    n_synth = sum(1 for e in train if e.get("synthetic"))
    n_gloss = len({e["gloss"] for e in train})
    per_gloss = len(train) / max(n_gloss, 1)

    if n_synth:
        logger.warning(
            "\n" + "!" * 70 +
            f"\n  {n_synth}/{len(train)} TRAINING CLIPS ARE SYNTHETIC PLACEHOLDERS."
            "\n  Their features are random vectors with no gloss-conditioned"
            "\n  signal, so CTC collapses to all-blank and WER stays at 1.0."
            "\n  This run validates the pipeline ONLY. Any metric it produces"
            "\n  is meaningless and must never be reported as a result."
            "\n" + "!" * 70)

    if per_gloss < 5:
        logger.warning(
            f"Only {per_gloss:.1f} training clips per gloss "
            f"({len(train)} clips / {n_gloss} glosses). The toolkit's coverage "
            f"floor is 25 clips/gloss - see 05_coverage_report.py. Even with "
            f"real footage this is too few to learn {n_gloss} classes.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    cfg = get_config()

    # Apply overrides
    if args.config_override:
        for override in args.config_override:
            key, val = override.split("=", 1)
            *path, attr = key.split(".")
            obj = cfg
            for p in path:
                obj = getattr(obj, p)
            # Attempt numeric cast
            try:
                val = float(val) if "." in val else int(val)
            except ValueError:
                pass
            setattr(obj, attr, val)

    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Training device: {device}")

    data_root = Path(args.data_root)
    for split in ("train", "val", "test"):
        if not (data_root / f"{split}_manifest.json").exists():
            raise FileNotFoundError(
                f"{data_root / f'{split}_manifest.json'} not found — "
                "pass --data_root pointing at a dataset folder with "
                "train/val/test manifests.")

    # Sanity-check vocabulary size against config
    vocab_path = data_root / "gloss_vocabulary.json"
    if vocab_path.exists():
        with open(vocab_path, encoding="utf-8") as f:
            n_vocab = len(json.load(f))
        if n_vocab != cfg.data.n_glosses:
            raise ValueError(
                f"gloss_vocabulary.json has {n_vocab} glosses but "
                f"config n_glosses={cfg.data.n_glosses}; update config.py.")

    check_data_provenance(data_root, cfg)

    all_seed_metrics = []

    for seed_idx in range(cfg.train.n_seeds):
        seed = cfg.train.seed + seed_idx
        set_seed(seed)
        logger.info(f"\n{'='*60}\nSeed {seed_idx+1}/{cfg.train.n_seeds} (seed={seed})\n{'='*60}")

        # Model
        model = PhoneGCN(
            n_glosses=cfg.data.n_glosses,
            n_handshapes=cfg.data.n_handshapes,
            n_locations=cfg.data.n_locations,
            hand_embed_dim=cfg.model.hand_embed_dim,
            body_embed_dim=cfg.model.stgcn_hidden_dim,
            face_embed_dim=cfg.model.face_embed_dim,
            fusion_d_model=cfg.model.fusion_embed_dim,
            fusion_n_heads=cfg.model.fusion_n_heads,
            fusion_n_layers=cfg.model.fusion_n_layers,
            fusion_d_ffn=cfg.model.fusion_ffn_dim,
            tau_init=cfg.train.tau_init,
            lambda_pam_init=cfg.model.lambda_pam_init,
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Trainable parameters: {n_params:,}")

        # Data
        loaders = {}
        for split in ("train", "val", "test"):
            ds = KSLDataset(data_root / f"{split}_manifest.json",
                            data_root, cfg, split=split)
            loaders[split] = DataLoader(
                ds, batch_size=cfg.train.batch_size,
                shuffle=(split == "train"), collate_fn=collate_fn,
                num_workers=args.num_workers)

        optimizer = build_optimizer(model, cfg)
        scheduler = build_scheduler(optimizer, cfg,
                                    n_steps=cfg.train.epochs * len(loaders["train"]))

        ckpt_dir = Path(args.output_dir) / f"seed_{seed}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        best_wer = float("inf")
        for epoch in range(1, cfg.train.epochs + 1):
            train_metrics = train_one_epoch(model, loaders["train"], optimizer,
                                            scheduler, cfg, device, epoch)
            val_metrics   = evaluate(model, loaders["val"], cfg, device)
            logger.info(f"Epoch {epoch}/{cfg.train.epochs} "
                        f"loss={train_metrics['total']:.4f} "
                        f"val_ctc={val_metrics['ctc_loss']:.4f} "
                        f"val_wer={val_metrics['wer']:.4f}")

            if val_metrics["wer"] <= best_wer:
                best_wer = val_metrics["wer"]
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_wer": best_wer,
                    "seed": seed,
                }, ckpt_dir / "best_model.pt")

            if epoch % cfg.train.save_every_n_epochs == 0:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "seed": seed,
                }, ckpt_dir / f"checkpoint_epoch{epoch:03d}.pt")

        # Final test evaluation with the best checkpoint
        best = torch.load(ckpt_dir / "best_model.pt", map_location=device,
                          weights_only=False)
        model.load_state_dict(best["model_state_dict"])
        test_metrics = evaluate(model, loaders["test"], cfg, device)
        logger.info(f"Seed {seed_idx+1} done. best_val_wer={best_wer:.4f} "
                    f"test_wer={test_metrics['wer']:.4f} → {ckpt_dir}")

        all_seed_metrics.append({"seed": seed,
                                 "best_val_wer": best_wer,
                                 "test_wer": test_metrics["wer"],
                                 "test_ctc_loss": test_metrics["ctc_loss"]})

        with open(Path(args.output_dir) / "results.json", "w", encoding="utf-8") as f:
            json.dump(all_seed_metrics, f, indent=2)

    if all_seed_metrics:
        wers = [m["test_wer"] for m in all_seed_metrics]
        logger.info(f"Training complete. test WER over {len(wers)} seed(s): "
                    f"{np.mean(wers):.4f} ± {np.std(wers):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PhoneGCN")
    parser.add_argument("--data_root",
                        default="../KSL_Daily_500_dataset/KSL_Daily_500_expert607")
    parser.add_argument("--output_dir",      default="checkpoints/")
    parser.add_argument("--device",          default=None,
                        help="'cuda' or 'cpu' (auto-detected if omitted)")
    parser.add_argument("--num_workers",     type=int, default=0)
    parser.add_argument("--config_override", nargs="*", default=[],
                        help="KEY=VALUE overrides, e.g. train.lr=5e-5")
    args = parser.parse_args()
    main(args)
