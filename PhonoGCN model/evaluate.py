"""
evaluate.py — PhoneGCN Evaluation Script
==========================================
Evaluates a trained PhoneGCN checkpoint on the test split.

Reported metrics (§4):
    - Gloss-level WER (Word Error Rate) — primary recognition metric
    - BLEU-1/2/3/4                      — translation quality
    - ROUGE-L                           — translation coverage
    - Phonological feature accuracy     — handshape, location (Appendix A)
    - Minimal-pair disambiguation accuracy (§4.5)

Usage
-----
    python evaluate.py \\
        --checkpoint checkpoints/seed_0/best_model.pt \\
        --data_root  data/ksl_daily_500 \\
        --split      test \\
        [--all_seeds checkpoints/]  # evaluate mean±std across all seed runs
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from config import get_config
from model  import PhoneGCN
from model.translation import TranslationHead

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation metrics
# ─────────────────────────────────────────────────────────────────────────────

def edit_distance(seq_a: List, seq_b: List) -> int:
    """Levenshtein edit distance between two sequences."""
    m, n = len(seq_a), len(seq_b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if seq_a[i - 1] == seq_b[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[n]


def word_error_rate(hypotheses: List[List], references: List[List]) -> float:
    """
    Compute corpus-level WER.

    Args:
        hypotheses: List of decoded gloss sequences.
        references: List of reference gloss sequences.

    Returns:
        WER as a fraction (lower is better).
    """
    total_edits, total_ref_len = 0, 0
    for hyp, ref in zip(hypotheses, references):
        total_edits  += edit_distance(hyp, ref)
        total_ref_len += len(ref)
    if total_ref_len == 0:
        return 0.0
    return total_edits / total_ref_len


def compute_bleu(hypotheses: List[str], references: List[str],
                 max_n: int = 4) -> Dict[str, float]:
    """
    Corpus BLEU-1 through BLEU-4 using sacrebleu.

    Requires: pip install sacrebleu
    """
    try:
        import sacrebleu
    except ImportError:
        logger.warning("sacrebleu not installed. BLEU scores unavailable. "
                       "pip install sacrebleu")
        return {f"bleu_{n}": float("nan") for n in range(1, max_n + 1)}

    scores = {}
    for n in range(1, max_n + 1):
        bleu = sacrebleu.corpus_bleu(
            hypotheses,
            [references],
            max_ngram_order=n,
            smooth_method="exp",
        )
        scores[f"bleu_{n}"] = round(bleu.score, 2)
    return scores


def compute_rouge_l(hypotheses: List[str], references: List[str]) -> float:
    """
    Corpus ROUGE-L using rouge_score.

    Requires: pip install rouge-score
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        logger.warning("rouge-score not installed. ROUGE-L unavailable. "
                       "pip install rouge-score")
        return float("nan")

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = [scorer.score(ref, hyp)["rougeL"].fmeasure
              for hyp, ref in zip(hypotheses, references)]
    return round(float(np.mean(scores)), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Greedy CTC decode
# ─────────────────────────────────────────────────────────────────────────────

def ctc_greedy_decode(log_probs: torch.Tensor,
                      input_lengths: torch.Tensor,
                      blank_id: int) -> List[List[int]]:
    """
    Greedy (best-path) CTC decoding.

    Args:
        log_probs:     (T, B, C) log probabilities
        input_lengths: (B,) actual sequence lengths
        blank_id:      CTC blank token index

    Returns:
        List of decoded token-id sequences (one per batch element).
    """
    T, B, C = log_probs.shape
    best_path = log_probs.argmax(dim=-1)  # (T, B)
    decoded = []

    for b in range(B):
        length = int(input_lengths[b])
        path = best_path[:length, b].tolist()

        # Collapse repeated tokens and remove blanks
        prev = None
        sequence = []
        for token in path:
            if token != prev:
                if token != blank_id:
                    sequence.append(token)
                prev = token
        decoded.append(sequence)

    return decoded


# ─────────────────────────────────────────────────────────────────────────────
# Minimal-pair evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_minimal_pairs(model: PhoneGCN,
                            minimal_pairs_dir: str,
                            device: torch.device) -> float:
    """
    Evaluate disambiguation accuracy on the minimal-pair test set (§4.5).

    Each pair consists of two clips that differ in exactly one phonological
    feature (handshape OR location OR movement).  The model must assign
    higher probability to the correct gloss.

    NOTE: Implement data loading from minimal_pairs_dir here.
    The expected format is a JSON file listing pairs with clip paths and
    correct/distractor gloss labels.

    Returns:
        Accuracy ∈ [0, 1]
    """
    logger.warning(
        "evaluate_minimal_pairs() stub: implement clip loading from "
        f"{minimal_pairs_dir}"
    )
    return float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation function
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_checkpoint(checkpoint_path: str,
                        data_root: str,
                        split: str = "test",
                        device: torch.device = torch.device("cpu")) -> Dict:
    """
    Load a checkpoint and evaluate on the specified split.

    Returns:
        Dictionary of metric name → value.
    """
    cfg = get_config()

    state = torch.load(checkpoint_path, map_location=device)
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
    ).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    logger.info(f"Loaded checkpoint: {checkpoint_path}")

    # NOTE: Replace with actual data loading once KSLDataset is implemented.
    logger.warning(
        f"Data loading stubbed out for split='{split}'. "
        "Implement KSLDataset.__getitem__() in train.py first."
    )

    metrics = {
        "checkpoint": checkpoint_path,
        "split": split,
        "wer": float("nan"),
        **{f"bleu_{n}": float("nan") for n in range(1, 5)},
        "rouge_l": float("nan"),
        "minimal_pair_acc": float("nan"),
    }

    logger.info(f"Metrics: {json.dumps(metrics, indent=2)}")
    return metrics


def evaluate_all_seeds(checkpoints_dir: str,
                       data_root: str,
                       split: str = "test",
                       device: torch.device = torch.device("cpu")) -> Dict:
    """
    Evaluate all seed checkpoints and report mean ± std.

    Args:
        checkpoints_dir: Root directory containing seed_0/, seed_1/, ...
        data_root:       Path to KSL-Daily-500 data root.
        split:           'val' or 'test'.

    Returns:
        Summary dict with mean and std per metric.
    """
    seed_dirs = sorted(Path(checkpoints_dir).glob("seed_*/best_model.pt"))
    if not seed_dirs:
        raise FileNotFoundError(
            f"No seed checkpoints found in {checkpoints_dir}. "
            "Expected: seed_0/best_model.pt, seed_1/best_model.pt, ..."
        )

    all_metrics: Dict[str, List[float]] = {}
    for ckpt in seed_dirs:
        m = evaluate_checkpoint(str(ckpt), data_root, split, device)
        for k, v in m.items():
            if isinstance(v, float):
                all_metrics.setdefault(k, []).append(v)

    summary = {}
    for k, vals in all_metrics.items():
        arr = np.array([v for v in vals if not np.isnan(v)])
        if len(arr) == 0:
            summary[k] = {"mean": float("nan"), "std": float("nan")}
        else:
            summary[k] = {
                "mean": round(float(arr.mean()), 4),
                "std":  round(float(arr.std()),  4),
                "n":    len(arr),
            }

    logger.info(f"\nAcross {len(seed_dirs)} seeds:")
    for k, v in summary.items():
        if isinstance(v, dict):
            logger.info(f"  {k}: {v['mean']:.4f} ± {v['std']:.4f}")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Evaluate PhoneGCN")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint",  help="Path to a single checkpoint .pt file")
    group.add_argument("--all_seeds",   help="Root directory of seed checkpoints")

    parser.add_argument("--data_root",  default="data/ksl_daily_500")
    parser.add_argument("--split",      default="test", choices=["val", "test"])
    parser.add_argument("--output",     default=None, help="Save metrics JSON to file")
    parser.add_argument("--device",     default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )

    if args.checkpoint:
        metrics = evaluate_checkpoint(args.checkpoint, args.data_root,
                                      args.split, device)
    else:
        metrics = evaluate_all_seeds(args.all_seeds, args.data_root,
                                     args.split, device)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved → {args.output}")
